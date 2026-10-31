"""Registered paired RGB TIFF manifests, independent of Bayer training."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .rgb import read_rgb_tiff, validate_rgb_contract


class PairedRGBDataset(Dataset):
    """Read registered noisy/independent-clean TIFF pairs without color transforms.

    Paths are relative to the manifest, crops/augmentation are synchronized, and
    scene and path leakage across splits is rejected before loading any image.
    Provenance strings are required declarations, not proof of data permissions.
    """

    def __init__(self, manifest, *, split="train", crop_size=128, augment=True):
        self.manifest = Path(manifest)
        content = json.loads(self.manifest.read_text(encoding="utf-8"))
        if content.get("version") != 1 or not isinstance(content.get("pairs"), list):
            raise ValueError("Expected RGB paired manifest version=1 with a pairs list")
        self.contract = validate_rgb_contract(content.get("rgb_contract"))
        if split not in {"train", "validation", "test"}:
            raise ValueError("RGB split must be train, validation or test")
        if crop_size is not None and (not isinstance(crop_size, int) or crop_size < 1):
            raise ValueError("RGB crop_size must be positive or None")
        self.crop_size, self.augment = crop_size, augment
        scenes, paths = {}, {}
        self.entries, self.all_entries = [], []
        for entry in content["pairs"]:
            if not isinstance(entry, dict):
                raise ValueError("RGB pair entries must be objects")
            for key in ("noisy", "clean", "scene", "split", "source", "usage_terms", "target_provenance",
                        "camera", "processing"):
                if not isinstance(entry.get(key), str) or not entry[key].strip():
                    raise ValueError(f"RGB pair requires nonempty {key}")
            if entry["split"] not in {"train", "validation", "test"}:
                raise ValueError("Invalid RGB pair split")
            if not isinstance(entry.get("iso"), int) or entry["iso"] < 1:
                raise ValueError("RGB pair requires positive integer iso")
            previous = scenes.setdefault(entry["scene"], entry["split"])
            if previous != entry["split"]:
                raise ValueError("RGB scene leakage across splits")
            pair_paths = [(self.manifest.parent / entry[k]).resolve() for k in ("noisy", "clean")]
            if pair_paths[0] == pair_paths[1]:
                raise ValueError("RGB noisy and independent clean target must be distinct files")
            for path in pair_paths:
                previous = paths.setdefault(path, entry["split"])
                if previous != entry["split"]:
                    raise ValueError("RGB file leakage across splits")
            resolved = dict(entry, noisy_path=pair_paths[0], clean_path=pair_paths[1])
            self.all_entries.append(resolved)
            if entry["split"] == split:
                self.entries.append(resolved)
        if not self.entries:
            raise ValueError(f"RGB manifest has no {split} pairs")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        images, orientations = [], []
        for key in ("noisy_path", "clean_path"):
            image, metadata = read_rgb_tiff(entry[key], color_space=self.contract["color_space"])
            if metadata.contract != self.contract:
                raise ValueError("RGB pair color-space/ICC contract does not match manifest")
            images.append(image)
            orientations.append(metadata.orientation)
        if orientations[0] != orientations[1]:
            raise ValueError("RGB pairs must have matching orientation; no rotation is performed")
        if images[0].shape != images[1].shape:
            raise ValueError("RGB pairs must have matching registered geometry; no alignment is performed")
        pair = np.stack(images)
        if self.crop_size is not None:
            h, w = pair.shape[-2:]
            size = self.crop_size
            if min(h, w) < size:
                raise ValueError("RGB pair is smaller than crop_size")
            if self.augment:
                y, x = (int(torch.randint(n-size+1, ()).item()) for n in (h, w))
            else:
                y, x = (h-size)//2, (w-size)//2
            pair = pair[..., y:y+size, x:x+size]
        if self.augment:
            if torch.rand(()) < .5:
                pair = pair[..., ::-1]
            if torch.rand(()) < .5:
                pair = pair[..., ::-1, :]
            if torch.rand(()) < .5:
                pair = pair.swapaxes(-1, -2)
        pair = torch.from_numpy(np.ascontiguousarray(pair))
        return {"noisy": pair[0], "clean": pair[1]}
