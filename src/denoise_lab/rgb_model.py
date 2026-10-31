"""Experimental RGB model, independently initialized; never Bayer weights."""
from torch import nn
from torch.nn import functional as F

from .model import NAFNet


class RGBDenoiseNet(nn.Module):
    """Three-channel residual NAFNet in the checkpoint's encoded RGB space.

    Reuses our independent NAFNet blocks, without the Bayer gamma wrapper or
    packing. No pretrained RGB parameters or denoising quality are implied.
    """

    def __init__(self, cond="none", boundary_mode="zeros", upsample_mode="pixelshuffle", **body):
        super().__init__()
        if cond != "none":
            raise ValueError("RGB NAFNet supports only cond=none")
        if "in_channels" in body or "out_channels" in body:
            raise ValueError("rgb_nafnet has exactly 3 input and 3 output channels")
        if boundary_mode not in {"zeros", "replicate"} or upsample_mode not in {"pixelshuffle", "nearest"}:
            raise ValueError("RGB boundary/upsample mode must be zeros/replicate and pixelshuffle/nearest")
        self.cond = cond
        self.boundary_mode = boundary_mode
        self.body = NAFNet(in_channels=3, out_channels=3, **body)
        if upsample_mode == "nearest":
            # Resize then convolve: constant features stay constant across pixel
            # phases. Keep checkpoint flags explicit; existing weights use the
            # original PixelShuffle architecture and remain unchanged.
            self.body.ups = nn.ModuleList(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(up[0].in_channels, up[0].in_channels//2, 1, bias=False)) for up in self.body.ups)
        if boundary_mode == "replicate":
            for module in self.body.modules():
                if isinstance(module, nn.Conv2d) and any(module.padding):
                    module.padding_mode = "replicate"

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("RGB NAFNet requires Bx3xHxW input")
        h, w = x.shape[-2:]
        features = x
        if self.boundary_mode == "replicate":
            features = F.pad(x, (0, (-w) % self.body.multiple, 0, (-h) % self.body.multiple), mode="replicate")
        return x + self.body(features)[..., :h, :w]
