"""Self-supervised neighbour-pair loss (Neighbor2Neighbor-style) on real noisy packed crops.

For each 2x2 cell of the packed planes, two horizontally or vertically adjacent positions are
drawn (the same for all four planes), giving two half-resolution sub-images with independent
noise when the sensor noise is spatially uncorrelated (true at mid/high ISO on the measured
cameras). The model denoises sub-image 1 and is pulled towards sub-image 2. A consistency term
compares against the same sub-sampling of the full-resolution prediction and corrects for the
one-pixel offset between sub-images. Errors are divided by each crop's noise sigma, so gradients
stay O(1) per pixel, comparable to the Charbonnier supervised loss.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Ordered adjacent (non-diagonal) pairs inside a 2x2 cell indexed 0=tl, 1=tr, 2=bl, 3=br.
_PAIRS = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 3], [3, 1], [3, 2]])


class RealNoisyDataset(Dataset):
    def __init__(self, manifest, root="data/selfsup_real", crop_size=192, min_iso=0):
        entries = json.loads(Path(manifest).read_text())
        self.entries = [e for e in entries if e["iso"] >= min_iso]
        if not self.entries:
            raise ValueError("No real noisy crops selected")
        self.root, self.crop_size = Path(root), int(crop_size)
        if self.crop_size % 32:
            raise ValueError("crop_size must be a multiple of 32 (half-resolution sub-images feed the model)")

    def __len__(self):
        return len(self.entries)

    def sampler(self, num_samples=10**7, camera_weights=None):
        """Per-camera sampling shares (default equal); cameras missing from camera_weights get 0."""
        counts = {}
        for e in self.entries:
            counts[e["camera"]] = counts.get(e["camera"], 0)+1
        share = camera_weights or {c: 1. for c in counts}
        w = torch.tensor([share.get(e["camera"], 0.)/counts[e["camera"]] for e in self.entries], dtype=torch.double)
        return WeightedRandomSampler(w, num_samples, replacement=True)

    def __getitem__(self, index):
        e = self.entries[index]
        a = np.load(self.root/e["crop"]).astype(np.float32)
        c = self.crop_size
        y = int(torch.randint(0, a.shape[1]-c+1, ()).item())
        x = int(torch.randint(0, a.shape[2]-c+1, ()).item())
        return {"noisy": torch.from_numpy(np.ascontiguousarray(a[:, y:y+c, x:x+c])),
                "sigma": torch.tensor(e["sigma"], dtype=torch.float32)}


def _cells(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//2, 2, w//2, 2).permute(0, 1, 2, 4, 3, 5).reshape(b, c, h//2, w//2, 4)


def neighbor_subsample(x, choice):
    """choice: (B, H/2, W/2) indices into _PAIRS; returns the two sub-images."""
    pairs = _PAIRS.to(x.device)[choice]  # B,H/2,W/2,2
    cells = _cells(x)
    idx1 = pairs[..., 0][:, None, ..., None].expand(-1, x.shape[1], -1, -1, 1)
    idx2 = pairs[..., 1][:, None, ..., None].expand(-1, x.shape[1], -1, -1, 1)
    return cells.gather(-1, idx1)[..., 0], cells.gather(-1, idx2)[..., 0]


def neighbor_loss(model, noisy, sigma, gamma=1.):
    b, _, h, w = noisy.shape
    choice = torch.randint(0, len(_PAIRS), (b, h//2, w//2), device=noisy.device)
    with torch.no_grad():
        full = model(noisy).float()
    sub1, sub2 = neighbor_subsample(noisy, choice)
    exact1, exact2 = neighbor_subsample(full, choice)
    pred = model(sub1).float()
    s = sigma.view(-1, 1, 1, 1).clamp_min(1e-6)
    diff = pred-sub2
    rec = (diff.square()/(2*s)).mean()
    reg = ((diff-(exact1-exact2)).square()/(2*s)).mean()
    return rec+gamma*reg
