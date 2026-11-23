"""Released artifacts and offline resolution; no implicit downloads at inference."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

MODELS = {
    'denoiser_1.1': {
        'release': 'model-denoiser_1.1-2026-11-22',
        'snapshot_id': 'mix_r10_r9p20',
        'assets': {'onnx': 'denoiser_1.1_fp32.onnx', 'pt': 'denoiser_1.1_weights.pt',
                   'coreml': 'denoiser_1.1_fp16.mlpackage.zip'},
        'sha256': {
            'onnx': 'bf9421b7f0c9bde177a0a11c6fd15b4c88066c4e29eef9daf534b20d68f9b820',
            'pt': '8f470b206294dd118d380a4bf2a25839b17cd9e6fdf0617ac3e95b838f7a5cab'},
        'cameras': ['X1D II 50C', 'X2D II 100C', 'ILCE-7M3', 'ILCE-5100', 'DSC-RX100M5'],
        'iso_range': [100, 204800],
    },
    'r4_plain': {
        'release': 'model-r4_plain-2026-10-30',
        'snapshot_id': 'r4_w40s4_plain_120k',
        'assets': {'onnx': 'r4_plain_fp32.onnx', 'pt': 'r4_plain_weights.pt',
                   'coreml': 'r4_plain_fp16.mlpackage.zip'},
        'sha256': {
            'onnx': '2f65a38a88144042821e7801c4f1adb620fcf8f50decaf2edfbe7c213edb2fae',
            'pt': 'cb0ab5ada18adb26a2893ee4e72ccf5fe334b98ccbe5691dc800f7804b3ab52a'},
        'cameras': ['X1D II 50C'],
        'iso_range': [100, 25600],
    },
}
DEFAULT_MODEL = 'denoiser_1.1'
BASE_URL = 'https://github.com/sndstone/rawdenoise-lab/releases/download'


def model_hash(path):
    """SHA-256 of a file, or sorted relative names and bytes for a package tree."""
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for file in files:
        if path.is_dir():
            digest.update(file.relative_to(path).as_posix().encode() + b'\0')
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_model(value, backend='auto', directory=Path('models')):
    """Resolve registry names from models/; prefer ONNX, then PT, then Core ML."""
    name = str(value)
    if name not in MODELS:
        return Path(value)
    formats = {'torch': ['pt'], 'onnx': ['onnx'], 'coreml': ['coreml'],
               'auto': ['onnx', 'pt', 'coreml']}[backend]
    for fmt in formats:
        path = Path(directory)/MODELS[name]['assets'][fmt].removesuffix('.zip')
        if path.exists():
            return path
    raise FileNotFoundError(f'Model {name} is not installed; run rawdenoise fetch-model --name {name}')


def _download(url, destination):
    with urllib.request.urlopen(url, timeout=60) as response:
        if not response.geturl().startswith('https://'):
            raise ValueError('Model downloads require HTTPS')
        with destination.open('xb') as stream:
            shutil.copyfileobj(response, stream)


def fetch_model(name=DEFAULT_MODEL, fmt='onnx', directory=Path('models')):
    """Verify in temporary storage, then publish exclusively; never replace a file."""
    entry = MODELS[name]
    asset = entry['assets'][fmt]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/asset.removesuffix('.zip')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {destination}')
    url = f"{BASE_URL}/{entry['release']}"
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        staging = Path(temporary)
        sums = staging/'SHA256SUMS'
        _download(f'{url}/SHA256SUMS', sums)
        checksums = {}
        for line in sums.read_text().splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2:
                checksums[fields[1].lstrip('*').removeprefix('./')] = fields[0].lower()
        expected = checksums.get(asset)
        if expected is None or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError(f'Missing or invalid SHA256SUMS entry for {asset}')
        pinned = entry['sha256'].get(fmt)
        if pinned and pinned != expected:
            raise ValueError('Release checksum differs from pinned SHA-256')
        archive = staging/asset
        _download(f'{url}/{asset}', archive)
        if model_hash(archive) != expected:
            raise ValueError(f'SHA-256 mismatch for {asset}')
        if fmt != 'coreml':
            # Exclusive creation also protects against a concurrent fetch.
            import os
            os.link(archive, destination)
        else:
            root = destination.name
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    parts = Path(member.filename).parts
                    if (not parts or parts[0] != root or '..' in parts or
                            '\\' in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000):
                        raise ValueError('Unsafe or unexpected Core ML archive path')
                package.extractall(staging/'unpacked')
            unpacked = staging/'unpacked'/root
            if not unpacked.is_dir():
                raise ValueError('Archive contains no model package')
            destination.mkdir()  # Exclusive reservation; clean up only our own directory.
            try:
                for child in unpacked.iterdir():
                    shutil.move(str(child), destination/child.name)
            except Exception:
                shutil.rmtree(destination)
                raise
    return destination
