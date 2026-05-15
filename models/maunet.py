"""
MauNet: ViT-MAE + Attention U-Net with Multi-Scale Feature Injection.

Architecture
============
                    ┌──────────────┐
   micrograph ────► │  MAE Encoder │ (frozen)
   1×1024×1024      │  (ViT)       │
                    └──────┬───────┘
                           │ 192×256×256
            ┌──────────────┼──────────────┐
            │  MultiScale  │  Projectors  │
            │  ┌───────┐ ┌─┴──────┐ ┌────┴──┐ ┌──────┐
            │  │256×256│ │128×128 │ │64×64  │ │32×32 │
            │  │ →256ch│ │ →512ch │ │→1024ch│ │→2048ch│
            │  └───┬───┘ └───┬────┘ └──┬───┘ └──┬───┘
            │      │         │         │        │
            │  ┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼────┐
   same ──► │  │ ⊕ s3  │ │⊕ s4 │ │⊕ s5 │ │⊕ b.n. │
   image    │  └───────┘ └──────┘ └──────┘ └───────┘
            │       U-Net  Encoder  →  Decoder
            └──────────────────────────────────────► 1×1024×1024

Fusion uses *zero-init scaling*: each injection point has a learnable scalar
initialised to 0, so the network starts as a pure Attention U-Net and
gradually learns to incorporate MAE features during training.
"""

import math

import torch
import torch.nn as nn
from models.unet_blocks import ConvBlock, EncoderBlock, DecoderBlock


class _Projection(nn.Module):
    """Project MAE features to a target channel count at a given spatial
    scale.  Uses 1×1 → 3×3 → 1×1 conv path with dropout for stronger
    spatial modelling.  Optional spatial down-sampling via average pooling.

    Alpha scalars in MauNet handle the zero-init gating, so the projection
    itself uses standard initialisation to avoid a gradient dead-zone."""

    def __init__(self, in_c: int, out_c: int, pool_factor: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        if pool_factor > 1:
            layers.append(nn.AvgPool2d(pool_factor))
        layers += [
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
        ]
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class MauNet(nn.Module):
    """
    Parameters
    ----------
    mae_extractor : nn.Module
        A ``MAEFeatureExtractor`` that maps (B,1,H,W) → (B, embed_dim, H', W').
    mae_embed_dim : int
        Channel dimension of the MAE feature map (default 192).
    """

    def __init__(
        self,
        mae_extractor: nn.Module,
        mae_embed_dim: int = 192,
        heatmap_bias_pi: float = 0.01,
    ):
        super().__init__()

        # ── frozen MAE branch ──
        self.mae_extractor = mae_extractor

        # ── multi-scale projections from MAE features ──
        #   MAE output: (B, 192, 256, 256) for 1024×1024 input
        #   s3: 256×256×256   s4: 128×128×512   s5: 64×64×1024   bn: 32×32×2048
        self.proj_s3 = _Projection(mae_embed_dim,  256, pool_factor=1)
        self.proj_s4 = _Projection(mae_embed_dim,  512, pool_factor=2)
        self.proj_s5 = _Projection(mae_embed_dim, 1024, pool_factor=4)
        self.proj_bn = _Projection(mae_embed_dim, 2048, pool_factor=8)

        # learnable fusion scalars (zero-init → pure U-Net at start)
        self.alpha_s3 = nn.Parameter(torch.zeros(1))
        self.alpha_s4 = nn.Parameter(torch.zeros(1))
        self.alpha_s5 = nn.Parameter(torch.zeros(1))
        self.alpha_bn = nn.Parameter(torch.zeros(1))

        # ── Attention U-Net ──
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 1024)

        self.b1 = ConvBlock(1024, 2048)

        self.d1 = DecoderBlock([2048, 1024], 1024)
        self.d2 = DecoderBlock([1024, 512], 512)
        self.d3 = DecoderBlock([512, 256], 256)
        self.d4 = DecoderBlock([256, 128], 128)
        self.d5 = DecoderBlock([128, 64], 64)

        # ── Dual prediction heads ──
        #   heatmap_head : CenterNet-style Gaussian center heatmap (main task, output)
        #   mask_head    : auxiliary segmentation supervision (training only)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.mask_head = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # Focal-loss friendly bias init for the heatmap head:
        #   bias = -log((1 - π) / π)  so initial sigmoid(output) ≈ π
        pi = float(max(min(heatmap_bias_pi, 0.5), 1e-4))
        bias_init = -math.log((1.0 - pi) / pi)
        nn.init.zeros_(self.heatmap_head.weight)
        nn.init.constant_(self.heatmap_head.bias, bias_init)

    # ──────────────────────────────────────────────────────────────────

    def load_unet_pretrained(self, state_dict: dict):
        """Load CryoSegNet pre-trained weights into the U-Net sub-modules.
        Unmatched keys (fusion layers, MAE) are silently skipped."""
        own = self.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        self.load_state_dict(own, strict=False)
        print(f"[MauNet] Loaded {loaded} U-Net parameters from CryoSegNet checkpoint")

    # ──────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning both heads' logits as a dict:
            {"heatmap": (B,1,H,W), "mask": (B,1,H,W)}"""
        # ── 1. MAE feature extraction (frozen, no grad) ──
        mae_feat = self.mae_extractor(x)        # (B, D, 256, 256)

        mae_s3 = self.proj_s3(mae_feat)          # (B, 256, 256, 256)
        mae_s4 = self.proj_s4(mae_feat)          # (B, 512, 128, 128)
        mae_s5 = self.proj_s5(mae_feat)          # (B, 1024, 64,  64)
        mae_bn = self.proj_bn(mae_feat)          # (B, 2048, 32,  32)

        # ── 2. U-Net encoder ──
        s1, p1 = self.e1(x)                     # s1: 1024², 64ch
        s2, p2 = self.e2(p1)                    # s2: 512²,  128ch
        s3, p3 = self.e3(p2)                    # s3: 256²,  256ch
        s4, p4 = self.e4(p3)                    # s4: 128²,  512ch
        s5, p5 = self.e5(p4)                    # s5: 64²,   1024ch

        # ── 3. Multi-scale fusion into skip connections ──
        s3 = s3 + self.alpha_s3 * mae_s3
        s4 = s4 + self.alpha_s4 * mae_s4
        s5 = s5 + self.alpha_s5 * mae_s5

        # ── 4. Bottleneck with MAE fusion ──
        b1 = self.b1(p5)                        # 32², 2048ch
        b1 = b1 + self.alpha_bn * mae_bn

        # ── 5. U-Net decoder ──
        d1 = self.d1(b1, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)

        return {
            "heatmap": self.heatmap_head(d5),
            "mask": self.mask_head(d5),
        }
