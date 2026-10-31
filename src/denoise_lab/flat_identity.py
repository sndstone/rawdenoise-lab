"""Optional clean Bayer identity supervision, independent of dataset RNG streams."""
import math

import torch

from .model import DenoiseNet, NAFNet


class FlatIdentityRegularizer:
    """Continuous scalar/color fields in packed [R, G1, B, G2] order.

    Half the levels are log-uniform (deep shadows); half are uniform through
    white. Random color ratios and gentle planar ramps can partially saturate.
    Only the generated sensor input is clipped; predictions and loss stay signed
    fp32. No evaluation fixtures or enumerated evaluation levels are used.
    """

    def __init__(self, model, *, weight=.1, batch_size=4, size=128, seed=0):
        if (type(model) is not DenoiseNet or type(model.body) is not NAFNet
                or model.cond != "none" or model.body.intro.in_channels != 4
                or model.body.ending.out_channels != 4
                or any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules())):
            raise ValueError("flat_identity requires unconditioned four-channel Bayer NAFNet with dropout=0")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("flat_identity weight must be finite and positive")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch_size, size)):
            raise ValueError("flat_identity batch_size and size must be positive integers")
        self.weight, self.batch_size, self.size = float(weight), batch_size, size
        # A private CPU generator also avoids changing the ordinary CUDA stream.
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, device):
        shape = (self.batch_size, 1, 1, 1)

        def uniform(shape):
            return torch.rand(shape, generator=self.generator, dtype=torch.float32)

        # Keep the green base strictly below white: clipping an over-white
        # scalar flat would otherwise reproduce the exact all-white gate input.
        # R/B gains and ramps still produce random *partial* saturation.
        levels = torch.where(uniform(shape) < .5,
                             (math.log(1e-5) + uniform(shape)*math.log(.9999/1e-5)).exp(),
                             .9999*uniform(shape))
        ratios = .25 + 1.35*uniform((self.batch_size, 2, 1, 1))
        ratios = torch.where(uniform(shape) < .3, torch.ones_like(ratios), ratios)
        # Green is the shared reference; R/B vary independently in color fields.
        rgb = torch.cat((ratios[:, :1], torch.ones_like(levels), ratios[:, 1:]), dim=1)
        planes = rgb[:, [0, 1, 2, 1]]
        axis = torch.linspace(-1, 1, self.size, dtype=torch.float32)
        sx, sy = [(uniform(shape)*2-1)*.15 for _ in range(2)]
        ramp = 1 + sx*axis.view(1, 1, 1, -1) + sy*axis.view(1, 1, -1, 1)
        ramp = torch.where(uniform(shape) < .5, ramp, torch.ones_like(ramp))
        return (levels*planes*ramp).clamp(0, 1).contiguous().to(device)

    def __call__(self, model, criterion, device):
        target = self.sample(device)
        prediction = model(target)
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise FloatingPointError("Invalid flat_identity prediction")
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            loss = criterion(prediction.float(), target)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite flat_identity loss")
        return loss


def make_flat_identity(model, config=None, *, seed=0):
    """Disabled mode creates no generator, validates no model, and draws no RNG."""
    if not config or not config.get("enabled", False):
        return None
    options = dict(config)
    options.pop("enabled")
    options.setdefault("seed", seed + 1000003)
    return FlatIdentityRegularizer(model, **options)
