"""Synthetic CPU review fixtures; no private photographs or weights."""
import json

import numpy as np
from PIL import Image
import pytest
import torch

from denoise_lab.blind_review import bradley_terry, decode_votes, prepare, score, select_crops
from denoise_lab.dng import RawMetadata, write_dng


def test_crop_determinism():
    image = np.random.default_rng(4).random((72, 90, 3))
    a = select_crops(image, count=8, size=24, seed=9)
    assert a == select_crops(image, count=8, size=24, seed=9)
    assert a != select_crops(image, count=8, size=24, seed=10)
    assert len(set(a)) == 8
    assert all(0 <= y <= 48 and 0 <= x <= 66 for y, x in a)
    assert select_crops(image[:24, :24], count=1, size=24) == [(0, 0)]
    with pytest.raises(ValueError):
        select_crops(image, size=100)


def test_prepare_vote_roundtrip(tmp_path, monkeypatch):
    # Real .pt loading and tiling, with tiny stateless torch architectures.
    import denoise_lab.inference as inference
    monkeypatch.setattr(inference, 'build_model', lambda config: torch.nn.Identity()
                        if config['kind'] == 'identity' else torch.nn.AvgPool2d(3, 1, 1))
    source = tmp_path/'synthetic.dng'
    mosaic = np.random.default_rng(3).uniform(.05, .6, (64, 80)).astype(np.float32)
    write_dng(source, mosaic, RawMetadata())
    arms = []
    for kind in ('identity', 'blur'):
        path = tmp_path/f'{kind}.pt'
        torch.save(dict(model_config=dict(kind=kind), model={}), path)
        arms.append(dict(label=kind, model=str(path), strength=1.))
    out = tmp_path/'review'
    key = prepare([dict(path=str(source), iso=800, scene='fixture', split='acceptance')], arms,
                  out, count=2, size=24, seed=7, prepass=False)
    key = json.loads((out/'private/key.json').read_text())
    assert key['images'][0]['iso'] == 800
    assert key['images'][0]['split'] == 'acceptance'
    page = (out/'index.html').read_text()
    assert 'key.json' not in page and 'synthetic.dng' not in page and 'identity' not in page
    assert 'fetch(' not in page and 'https://' not in page
    assert 'localStorage' in page and 'createObjectURL' in page
    assert len(key['trials']) == 2
    import re
    data = json.loads(re.search(r'const data=(.*), storageKey=', page).group(1))
    for trial in data['trials']:
        with Image.open(out/trial['left']) as left, Image.open(out/trial['right']) as right:
            assert left.size == right.size == (24, 24)
            assert not np.array_equal(np.asarray(left), np.asarray(right))
    for path in (out/'assets').glob('*.png'):
        with Image.open(path) as image:
            assert image.mode == 'RGB'
    # Decode left/right randomisation to make identity win every trial.
    votes = dict(review_id=key['review_id'], votes={tid: 1 if t['left'] == 0 else 2
                 for tid, t in key['trials'].items()})
    votes = json.loads(json.dumps(votes))
    assert all((a == 0 and v == 1.) or (b == 0 and v == 0.)
               for a, b, v, _ in decode_votes(key, votes))
    results = tmp_path/'review_results.md'
    strengths = score(key, votes, results, bootstrap=10)
    assert strengths[0] > strengths[1]
    assert 'ISO 800' in results.read_text() and 'Votes: 2/2' in results.read_text()
    tie_id = next(iter(votes['votes']))
    votes['votes'][tie_id] = 0
    assert any(row[2] == .5 for row in decode_votes(key, votes))
    with pytest.raises(ValueError, match='different review'):
        decode_votes(key, dict(votes, review_id='other'))
    with pytest.raises(ValueError, match='invalid vote'):
        decode_votes(key, dict(votes, votes={tie_id: 3}))
    with pytest.raises(ValueError, match='Unknown trial'):
        decode_votes(key, dict(votes, votes={'unknown': 1}))
    with pytest.raises(FileExistsError):
        prepare([source], arms, out)


def test_bradley_terry_ordering():
    # Exact pair frequencies from log strengths log(4), log(2), log(1).
    rows = []
    for a, b, wins, losses in [(0, 1, 40, 20), (0, 2, 48, 12), (1, 2, 40, 20)]:
        rows.extend([(a, b, 1., 0)]*wins+[(a, b, 0., 0)]*losses)
    actual = bradley_terry(rows, 3)
    np.testing.assert_allclose(actual, [np.log(2), 0., -np.log(2)], atol=.002)
    assert np.all(bradley_terry([(0, 1, .5, 0)]*20, 2) == 0)
    assert np.isfinite(bradley_terry([(0, 1, 1., 0)]*20, 3)).all()
