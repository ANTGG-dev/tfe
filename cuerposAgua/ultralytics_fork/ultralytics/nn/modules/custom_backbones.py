# ultralytics/nn/modules/custom_backbones.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Pad to 'same'."""
    if p is None:
        p = (k - 1) // 2 * d
    return p


class ConvBNAct(nn.Module):
    """Conv + BN + Act (SiLU by default)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# Lightweight Attention (ECA)
# -------------------------
class ECALayer(nn.Module):
    """
    Efficient Channel Attention (very light).
    Args: c (channels), k_size (odd int)
    """
    def __init__(self, c: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)              # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y


class TinySCA(nn.Module):
    """
    Lightweight spatial-channel attention for tiny objects.

    ECA reweights channels and a 7x7 spatial gate highlights small local evidence
    that may be diluted in deeper feature maps.
    """
    def __init__(self, c: int, k_size: int = 3, spatial_k: int = 7):
        super().__init__()
        self.eca = ECALayer(c, k_size=k_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, spatial_k, 1, autopad(spatial_k), bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.eca(x)
        avg = torch.mean(y, dim=1, keepdim=True)
        mx = torch.amax(y, dim=1, keepdim=True)
        return y * self.spatial(torch.cat((avg, mx), dim=1))


class SPDConvDown(nn.Module):
    """
    Space-to-depth downsampling followed by convolution.

    Unlike stride-2 convolution, this rearranges 2x2 neighborhoods into channels
    before mixing, preserving tiny-object pixels during scale transitions.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, attn: bool = False):
        super().__init__()
        self.conv = ConvBNAct(c1 * 4, c2, k=k, s=1)
        self.attn = TinySCA(c2) if attn else nn.Identity()

    def forward(self, x):
        # Pad odd image sizes so slicing keeps all pixels.
        _, _, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2))
        x = torch.cat(
            (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]),
            dim=1,
        )
        return self.attn(self.conv(x))


class TinyHRBlock(nn.Module):
    """
    High-resolution residual block for P2/P3 features.

    Uses depthwise convolution for local detail, a dilated depthwise branch for
    context, optional lightweight attention, and a pointwise mixer.
    """
    def __init__(self, c: int, k: int = 3, dilation: int = 2, attn: bool = False, expand: float = 1.5):
        super().__init__()
        hidden = max(c, int(round(c * expand)))
        self.local = nn.Sequential(
            nn.Conv2d(c, c, k, 1, autopad(k), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(c, c, k, 1, autopad(k, d=dilation), dilation=dilation, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.attn = TinySCA(c) if attn else nn.Identity()
        self.mix = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        y = self.local(x) + self.context(x)
        y = self.attn(y)
        return x + self.mix(y)


class TinyHRStage(nn.Module):
    """
    Repeats TinyHRBlock at the same resolution.
    Args: c, n, k, dilation, attn, expand
    """
    def __init__(
        self,
        c: int,
        n: int = 2,
        k: int = 3,
        dilation: int = 2,
        attn: bool = False,
        expand: float = 1.5,
    ):
        super().__init__()
        self.m = nn.Sequential(*[TinyHRBlock(c, k=k, dilation=dilation, attn=attn, expand=expand) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


class TinyTFEBlock(nn.Module):
    """
    Tiny Feature Enhancement block.

    The block keeps the feature map resolution and fuses four cheap cues:
    local detail, dilated context, horizontal traces and vertical traces. A
    lightweight channel-spatial gate then suppresses background activations.
    """
    def __init__(self, c: int, dilation: int = 2, expand: float = 1.25, attn: bool = True):
        super().__init__()
        hidden = max(c, int(round(c * expand)))
        self.local = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, autopad(3), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, autopad(3, d=dilation), dilation=dilation, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.horizontal = nn.Sequential(
            nn.Conv2d(c, c, (1, 5), 1, (0, 2), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(c, c, (5, 1), 1, (2, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 4, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.gate = TinySCA(c, spatial_k=5) if attn else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = torch.cat((self.local(x), self.context(x), self.horizontal(x), self.vertical(x)), dim=1)
        y = self.gate(self.fuse(y))
        return x + self.gamma * y


class TinyTFEStage(nn.Module):
    """
    Repeats TinyTFEBlock at the same resolution.
    Args: c, n, dilation, expand, attn
    """
    def __init__(self, c: int, n: int = 2, dilation: int = 2, expand: float = 1.25, attn: bool = True):
        super().__init__()
        self.m = nn.Sequential(*[TinyTFEBlock(c, dilation=dilation, expand=expand, attn=attn) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


# ============================================================
# 2) ConvNeXt-Lite blocks (CNX)
# ============================================================
class CNXStem(nn.Module):
    """
    Simple stem: 3x3 conv stride 2 -> 3x3 conv stride 2 (gives stride 4 quickly; good for P2).
    Output: (B, c, H/4, W/4)
    """
    def __init__(self, c1: int, c2: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(c1, c2, k=3, s=2),
            ConvBNAct(c2, c2, k=3, s=2),
        )

    def forward(self, x):
        return self.stem(x)


class CNXDown(nn.Module):
    """Downsample by 2 with a 2x2 conv stride 2."""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.down = ConvBNAct(c1, c2, k=2, s=2)

    def forward(self, x):
        return self.down(x)


class CNXBlock(nn.Module):
    """
    ConvNeXt-like block:
      DWConv kxk -> (optional ECA) -> PWConv expand -> GELU -> PWConv -> residual
    Very stable, good receptive field for small objects without expensive attention.
    """
    def __init__(
        self,
        c: int,
        k: int = 7,
        expand: int = 4,
        drop: float = 0.0,
        eca: bool = False,
    ):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=k, stride=1, padding=autopad(k), groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)

        self.eca = ECALayer(c) if eca else nn.Identity()

        hidden = c * expand
        self.pw1 = nn.Conv2d(c, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, c, kernel_size=1, bias=False)

        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.bn(self.dw(x))
        y = self.eca(y)
        y = self.pw2(self.act(self.pw1(y)))
        y = self.drop(y)
        return x + y


class CNXStage(nn.Module):
    """
    Stage: repeat CNXBlock 'n' times.
    Args:
      c: channels
      n: depth
      k: dwconv kernel size
      drop: dropout
      eca: add ECA attention
    """
    def __init__(self, c: int, n: int = 2, k: int = 7, drop: float = 0.0, eca: bool = False):
        super().__init__()
        self.m = nn.Sequential(*[CNXBlock(c, k=k, drop=drop, eca=eca) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


# ============================================================
# 3) FasterNet / PConv blocks (FNET)
# ============================================================
class PConv(nn.Module):
    """
    Partial Convolution (very simplified):
    Convolve only a fraction of channels and concat with untouched channels.
    This is a key idea in FasterNet-style efficiency backbones.
    """
    def __init__(self, c: int, ratio: float = 0.25, k: int = 3):
        super().__init__()
        assert 0.0 < ratio <= 1.0
        self.c = c
        self.cp = max(1, int(round(c * ratio)))  # channels to convolve
        self.cu = c - self.cp                   # untouched channels
        self.conv = nn.Conv2d(self.cp, self.cp, k, 1, autopad(k), bias=False, groups=1)
        self.bn = nn.BatchNorm2d(self.cp)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        xp, xu = x[:, :self.cp, :, :], x[:, self.cp:, :, :]
        yp = self.act(self.bn(self.conv(xp)))
        return torch.cat((yp, xu), dim=1)


class FNetStem(nn.Module):
    """Stem to stride 4 quickly (good for P2)."""
    def __init__(self, c1: int, c2: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(c1, c2, k=3, s=2),
            ConvBNAct(c2, c2, k=3, s=2),
        )

    def forward(self, x):
        return self.stem(x)


class FNetDown(nn.Module):
    """Downsample by 2."""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.down = ConvBNAct(c1, c2, k=2, s=2)

    def forward(self, x):
        return self.down(x)


class PConvBlock(nn.Module):
    """
    PConv -> (optional ECA) -> 1x1 mix -> residual
    """
    def __init__(self, c: int, ratio: float = 0.25, drop: float = 0.0, eca: bool = False):
        super().__init__()
        self.pconv = PConv(c, ratio=ratio, k=3)
        self.eca = ECALayer(c) if eca else nn.Identity()
        self.mix = ConvBNAct(c, c, k=1, s=1)
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.pconv(x)
        y = self.eca(y)
        y = self.mix(y)
        y = self.drop(y)
        return x + y


class PConvStage(nn.Module):
    """
    Repeat PConvBlock.
    Args:
      c: channels
      n: depth
      ratio: partial conv ratio
      drop: dropout
      eca: add ECA attention
    """
    def __init__(self, c: int, n: int = 2, ratio: float = 0.25, drop: float = 0.0, eca: bool = False):
        super().__init__()
        self.m = nn.Sequential(*[PConvBlock(c, ratio=ratio, drop=drop, eca=eca) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


# ============================================================
# 4) Simple DWStage for ablation (PConv OFF)
# ============================================================
class DWStage(nn.Module):
    """
    Depthwise-separable style stage to compare against PConvStage (PConv OFF ablation).
    """
    def __init__(self, c: int, n: int = 2, k: int = 3, drop: float = 0.0, eca: bool = False):
        super().__init__()
        blocks = []
        for _ in range(n):
            blocks += [
                nn.Sequential(
                    nn.Conv2d(c, c, k, 1, autopad(k), groups=c, bias=False),
                    nn.BatchNorm2d(c),
                    nn.SiLU(inplace=True),
                    ECALayer(c) if eca else nn.Identity(),
                    ConvBNAct(c, c, k=1, s=1),
                )
            ]
        self.m = nn.Sequential(*blocks)
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.m(x)
        y = self.drop(y)
        return x + y
