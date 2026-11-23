"""Synthetic batch CLI, packed colour blending and offline release-download checks."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from denoise_lab import cli
from denoise_lab.inference import Denoiser
from denoise_lab import models_registry as registry
from test_inference import checkpoint, fixture_raw


def test_discovery(tmp_path):
    for name in ['a.DNG', 'b.ARW', 'readme.txt', 'a_denoised.dng', 'sub/c.dng']:
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    assert [p.name for p, _ in cli.discover_inputs([tmp_path])] == ['a.DNG', 'b.ARW']
    files = cli.discover_inputs([tmp_path, tmp_path/'a.DNG'], True)
    assert [p.name for p, _ in files] == ['a.DNG', 'b.ARW', 'c.dng']
    assert files[-1][1] == Path('sub')
    assert [p.name for p, _ in cli.discover_inputs([tmp_path], True, 'c.*')] == ['c.dng']


def test_folder_report_and_skip(tmp_path, checkpoint, monkeypatch):
    model, _ = checkpoint
    source = tmp_path/'input'
    source.mkdir()
    fixture_raw(source/'a.dng')
    (source/'sub').mkdir()
    fixture_raw(source/'sub'/'a.dng')
    (source/'broken.dng').write_bytes(b'invalid raw')
    # Use the real tiny checkpoint with smaller tiles for a fast CPU batch test.
    original = Denoiser.denoise_packed
    def small_tiles(self, packed, **kwargs):
        return original(self, packed, **dict(kwargs, tile_size=32, overlap=8))
    monkeypatch.setattr(Denoiser, 'denoise_packed', small_tiles)
    monkeypatch.setattr(cli, 'raw_defect_metadata', lambda *a: ({}, 51200., 1.))
    monkeypatch.setattr(cli, 'model_hash', lambda p: registry.MODELS['r4_plain']['sha256']['pt'])
    out, report = tmp_path/'output', tmp_path/'report.json'
    args = [str(source), '--recursive', '--model', str(model), '--device', 'cpu',
            '--no-prepass', '--format', 'npy', '--output-dir', str(out)]
    with pytest.raises(SystemExit) as failure:
        cli.main(args + ['--report', str(report)])
    assert failure.value.code == 1
    data = json.loads(report.read_text())
    assert data['summary'] == dict(succeeded=2, skipped=0, failed=1)
    assert (out/'sub'/'a_denoised.npy').exists()
    success = next(r for r in data['files'] if r['status'] == 'success')
    assert success['iso'] == 51200 and success['camera']
    assert success['backend'] == 'torch' and success['model_hash']
    assert success['timing_seconds'] > 0
    assert success['settings']['chroma_strength'] == 1
    assert any('Camera not validated' in w for w in success['warnings'])
    assert any('outside training prior' in w for w in success['warnings'])
    assert 'error' in next(r for r in data['files'] if r['status'] == 'failed')
    before = (out/'a_denoised.npy').read_bytes()
    skip_report = tmp_path/'skip.json'
    cli.main(args + ['--glob', 'a.*', '--skip-existing', '--report', str(skip_report)])
    assert json.loads(skip_report.read_text())['summary'] == dict(succeeded=0, skipped=2, failed=0)
    assert (out/'a_denoised.npy').read_bytes() == before
    with pytest.raises(SystemExit) as refused:
        cli.main(args + ['--report', str(report)])
    assert refused.value.code == 2
    with pytest.raises(SystemExit) as refused:
        cli.main(args + ['--glob', 'a.*'])
    assert refused.value.code == 1
    assert (out/'a_denoised.npy').read_bytes() == before


@pytest.mark.parametrize('strength', [0., .37, 1.])
def test_equal_strength_is_exact_legacy(strength):
    packed = np.random.default_rng(4).random((4, 16, 16), dtype=np.float32)
    denoiser = Denoiser()
    denoiser.device = 'cpu'
    denoiser._call = lambda x: x*.7+.03
    full = denoiser.denoise_packed(packed, prepass=False, tile_size=16, overlap=0, highlights='off')
    expected = (strength*torch.from_numpy(full)+(1-strength)*torch.from_numpy(packed)).numpy()
    actual = denoiser.denoise_packed(packed, prepass=False, tile_size=16, overlap=0,
                                    strength=strength, chroma_strength=strength,
                                    white_balance=[2., 1., 1.5], highlights='off')
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, denoiser.denoise_packed(
        packed, prepass=False, tile_size=16, overlap=0, strength=strength, highlights='off'))


@pytest.mark.parametrize('strength,chroma', [(0., 1.), (1., 0.), (.3, .8)])
def test_opponent_strengths(strength, chroma):
    denoiser = Denoiser()
    denoiser.device = 'cpu'
    delta = torch.tensor([.1, .2, -.3, -.1])[None, :, None, None]
    denoiser._call = lambda x: x+delta
    packed = np.zeros((4, 16, 16), np.float32)
    gains = np.array([2., 1., 1.5, 1.])
    def opponent(planes):
        r, g1, b, g2 = planes*gains[:, None, None]
        y = (r+g1+g2+b)/4
        return np.array([y, b-y, r-y, g1-g2])
    full = denoiser.denoise_packed(packed, prepass=False, tile_size=16, overlap=0)
    actual = denoiser.denoise_packed(packed, prepass=False, tile_size=16, overlap=0,
                                    strength=strength, chroma_strength=chroma,
                                    white_balance=gains[:3])
    expected = opponent(full)*np.array([strength, chroma, chroma, strength])[:, None, None]
    np.testing.assert_allclose(opponent(actual), expected, atol=1e-7)


def test_invalid_chroma_and_wb():
    denoiser = Denoiser()
    packed = np.zeros((4, 16, 16), np.float32)
    for settings in [dict(chroma_strength=-.1), dict(chroma_strength=float('nan')),
                     dict(white_balance=[1, 0, 1]), dict(white_balance=[1, 1])]:
        with pytest.raises(ValueError):
            denoiser.denoise_packed(packed, **settings)


@pytest.fixture
def release(monkeypatch):
    entry = dict(release='test-release', assets=dict(registry.MODELS['r4_plain']['assets']), sha256={})
    monkeypatch.setitem(registry.MODELS, 'test', entry)
    payloads = {}
    urls = []
    def urlopen(url, timeout):
        urls.append(url)
        assert url.startswith('https://github.com/sndstone/rawdenoise-lab/releases/download/test-release/')
        result = io.BytesIO(payloads[url.rsplit('/', 1)[-1]])
        result.geturl = lambda: url
        return result
    monkeypatch.setattr(registry.urllib.request, 'urlopen', urlopen)
    def prepare(fmt, content=b'fixture model'):
        asset = entry['assets'][fmt]
        payloads[asset] = content
        digest = hashlib.sha256(content).hexdigest()
        payloads['SHA256SUMS'] = f'{digest}  {asset}\n'.encode()
        return digest
    return entry, payloads, urls, prepare


@pytest.mark.parametrize('fmt', ['onnx', 'pt', 'coreml'])
def test_fetch_and_resolve(tmp_path, release, fmt):
    entry, _, urls, prepare = release
    content = b'model fixture'
    if fmt == 'coreml':
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as package:
            package.writestr('r4_plain_fp16.mlpackage/Manifest.json', '{}')
        content = buffer.getvalue()
    digest = prepare(fmt, content)
    entry['sha256'][fmt] = digest
    result = registry.fetch_model('test', fmt, tmp_path)
    if fmt == 'coreml':
        assert (result/'Manifest.json').read_text() == '{}'
    else:
        assert result.read_bytes() == content
    backend = {'onnx': 'onnx', 'pt': 'torch', 'coreml': 'coreml'}[fmt]
    assert registry.resolve_model('test', backend, tmp_path) == result
    assert len(urls) == 2
    with pytest.raises(FileExistsError):
        registry.fetch_model('test', fmt, tmp_path)
    assert len(urls) == 2
    with pytest.raises(FileNotFoundError):
        registry.resolve_model('test', backend, tmp_path/'absent')


@pytest.mark.parametrize('problem', ['corrupt', 'missing', 'pin', 'traversal'])
def test_fetch_rejects(tmp_path, release, problem):
    entry, payloads, _, prepare = release
    fmt = 'coreml' if problem == 'traversal' else 'onnx'
    if problem == 'traversal':
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as package:
            package.writestr('../escaped.txt', 'unsafe')
        prepare(fmt, buffer.getvalue())
    else:
        prepare(fmt)
    if problem == 'corrupt':
        payloads[entry['assets'][fmt]] = b'corrupted'
    elif problem == 'missing':
        payloads['SHA256SUMS'] = b''
    elif problem == 'pin':
        entry['sha256'][fmt] = '0'*64
    with pytest.raises(ValueError):
        registry.fetch_model('test', fmt, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_version_and_fetch_dispatch(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as result:
        cli.main(['--version'])
    assert result.value.code == 0
    assert 'rawdenoise 0.2.0' in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(cli, 'fetch_model', lambda *args: calls.append(args) or tmp_path/'model.onnx')
    cli.main(['fetch-model', '--format', 'pt', '--dir', str(tmp_path)])
    assert calls == [('denoiser_1.1', 'pt', tmp_path)]


def test_denoiser_1_1_registry_entry():
    assert registry.DEFAULT_MODEL == 'denoiser_1.1'
    entry = registry.MODELS['denoiser_1.1']
    assert entry['snapshot_id'] == 'mix_r10_r9p20'
    assert entry['sha256']['pt'] == '8f470b206294dd118d380a4bf2a25839b17cd9e6fdf0617ac3e95b838f7a5cab'
    assert len(entry['sha256']['onnx']) == 64
    assert {'X1D II 50C', 'X2D II 100C', 'ILCE-7M3', 'ILCE-5100', 'DSC-RX100M5'} <= set(entry['cameras'])
    assert entry['iso_range'] == [100, 204800]
    assert 'r4_plain' in registry.MODELS


def test_package_hash_is_stable_and_sensitive(tmp_path):
    package = tmp_path/'fixture.mlpackage'
    (package/'weights').mkdir(parents=True)
    (package/'Manifest.json').write_text('{}')
    weights = package/'weights'/'data.bin'
    weights.write_bytes(b'weights')
    first = registry.model_hash(package)
    assert registry.model_hash(package) == first
    weights.write_bytes(b'changed')
    assert registry.model_hash(package) != first
    weights.write_bytes(b'weights')
    weights.rename(weights.with_name('renamed.bin'))
    assert registry.model_hash(package) != first


def test_fetch_invalid_archive_is_clean_cli_error(tmp_path, release, capsys):
    _, _, _, prepare = release
    prepare('coreml', b'not a zip')
    with pytest.raises(SystemExit) as failure:
        cli.main(['fetch-model', '--name', 'test', '--format', 'coreml', '--dir', str(tmp_path)])
    assert failure.value.code == 1
    assert 'Model download failed' in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('problem', ['bytes', 'empty', 'config', 'weights', 'not_mapping', 'incompatible'])
def test_malformed_checkpoint_subprocess(tmp_path, problem):
    import subprocess
    import sys
    model = tmp_path/'bad.pt'
    if problem == 'bytes':
        model.write_bytes(b'not a checkpoint\n')
    else:
        saved = {'empty': {}, 'config': dict(model_config=[]),
                 'weights': dict(model_config={}, model=[]), 'not_mapping': [],
                 'incompatible': dict(model_config=dict(width=2, enc_blk_nums=[1],
                                                       dec_blk_nums=[1], middle_blk_num=1),
                                      model={})}[problem]
        torch.save(saved, model)
    source = tmp_path/'source.dng'
    source.touch()
    output = tmp_path/'output'
    report = tmp_path/'report.json'
    result = subprocess.run([sys.executable, '-m', 'denoise_lab.cli', str(source),
                             '--model', str(model), '--backend', 'torch', '--device', 'cpu',
                             '--output-dir', str(output), '--report', str(report)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stderr.startswith('Model load failed: ')
    assert len(result.stderr.splitlines()) == 1
    assert 'Traceback' not in result.stderr+result.stdout
    assert not output.exists() and not report.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['bad.pt', 'source.dng']


@pytest.mark.parametrize('selection,device,expected', [
    (None, 'auto', 'all'), (None, 'cpu', 'cpu_only'),
    ('all', 'auto', 'all'), ('cpu_and_ne', 'auto', 'cpu_and_ne'),
    ('cpu_and_gpu', 'auto', 'cpu_and_gpu'), ('cpu_only', 'auto', 'cpu_only'),
])
def test_coreml_cli_report(tmp_path, monkeypatch, selection, device, expected):
    from denoise_lab.inference import coreml_compute_units
    source = tmp_path/'source.dng'
    fixture_raw(source)
    package = tmp_path/'fixture.mlpackage'
    package.mkdir()
    loaded = []
    def load(path, **kwargs):
        loaded.append(kwargs)
        model = Denoiser()
        model.backend = 'coreml'
        model.device = 'cpu'
        model.compute_units = coreml_compute_units(kwargs['device'], kwargs['compute_units'])
        model.providers = [f'CoreML ({model.compute_units.upper()})']
        model._call = lambda tile: tile
        return model
    monkeypatch.setattr(Denoiser, 'load', load)
    report = tmp_path/'report.json'
    args = [str(source), '--model', str(package), '--device', device, '--format', 'npy',
            '--no-prepass', '--report', str(report)]
    if selection:
        args += ['--compute-units', selection]
    cli.main(args)
    assert loaded[0]['compute_units'] == selection
    record = json.loads(report.read_text())['files'][0]
    assert record['compute_units'] == expected
    assert record['settings']['compute_units'] == selection


@pytest.mark.parametrize('backend,device,selection,message', [
    ('torch', 'auto', 'all', 'only supported'),
    ('onnx', 'auto', 'cpu_only', 'only supported'),
    ('coreml', 'cpu', 'cpu_and_ne', 'conflicts'),
])
def test_cli_compute_unit_errors(tmp_path, capsys, backend, device, selection, message):
    source = tmp_path/'source.dng'
    fixture_raw(source)
    model = tmp_path/'fixture.pt'
    model.touch()
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), '--model', str(model), '--backend', backend,
                  '--device', device, '--compute-units', selection])
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def test_cli_invalid_compute_units(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['source.dng', '--model', 'fixture.mlpackage', '--compute-units', 'invalid'])
    assert exc.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
