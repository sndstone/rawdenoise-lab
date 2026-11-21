import numpy as np
import pytest
from denoise_lab.noise import NoiseModel


def profile(**kwargs):
    return {"units": "normalized", "iso": {1600: {"K": .002, "read_std": .01, **kwargs}}}


@pytest.mark.parametrize("signal", [0., .1, .5])
@pytest.mark.parametrize("distribution", ["gaussian", "tukey"])
def test_noise_variance(signal, distribution):
    n = NoiseModel(profile(read_distribution=distribution, tukey_lambda=.14), seed=42)
    clean = np.full((800, 1000), signal, np.float32)
    residual = n(clean, 1600, sample=False)-clean
    expected = .002*signal+.01**2
    assert abs(residual.var()/expected-1) < .025
    assert abs(residual.mean()) < 5*np.sqrt(expected/clean.size)
    assert float(n.noise_map(1600, signal)) == pytest.approx(np.sqrt(expected), rel=1e-6)


def test_row_correlation_and_log_sampling():
    cfg = profile(row_std=.03, K_range=[.001, .004])
    cfg["read_log_linear"] = {"slope": 1., "intercept": 0., "scatter": 0.}
    n = NoiseModel(cfg, seed=0)
    p = n.parameters(1600, sample=True)
    assert .001 <= p["K"] <= .004
    assert p["read_std"] == pytest.approx(p["K"])
    a = n(np.zeros((2000, 200), np.float32), 1600, sample=False)
    assert a.mean(1).std() == pytest.approx(.03, rel=.05)
