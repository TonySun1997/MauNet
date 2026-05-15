"""
MAE (Masked Autoencoder) encoder from Cryo-EMMAE, wrapped as a frozen
feature extractor that converts a full-size micrograph into a dense
spatial feature map.

Input:  (B, 1, H, W)  e.g. (B, 1, 1024, 1024)
Output: (B, embed_dim, H/patch_size, W/patch_size)  e.g. (B, 192, 256, 256)
"""

from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


# ── Position-embedding utilities (from facebook/mae) ────────────────────────

def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ── Masked Autoencoder ViT ──────────────────────────────────────────────────

class MaskedAutoencoderViT(nn.Module):
    """Full MAE model (encoder + decoder). We keep both so that the original
    checkpoint can be loaded directly; only the encoder path is used at
    inference time via ``infer_latent``."""

    def __init__(
        self,
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=nn.LayerNorm,
        norm_pix_loss=False, pos_encode_weight=1.0,
    ):
        super().__init__()

        # ── encoder ──
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.in_chans = in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.pos_encode_weight = pos_encode_weight

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ── decoder (kept for checkpoint compatibility) ──
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    # ── weight init ──

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        dec_pos = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ── masking ──

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ── encoder forward ──

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :] * self.pos_encode_weight
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    # ── inference (no masking) ──

    def infer_latent(self, x):
        x, mask, ids_restore = self.forward_encoder(x, mask_ratio=0.0)
        x_ = x[:, 1:, :]
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        return x

    # ── full forward (for reference; not used in MauNet) ──

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask


# ── Feature Extractor Wrapper ───────────────────────────────────────────────

class MAEFeatureExtractor(nn.Module):
    """Wraps ``MaskedAutoencoderViT`` to produce a spatial feature map from a
    full-size micrograph.

    Processing:
        1. Split image into non-overlapping crops (``img_size × img_size``).
        2. Batch-encode all crops through the frozen MAE encoder.
        3. Reassemble patch-token embeddings into a 2-D feature map.

    For a 1024×1024 input with ``img_size=64, patch_size=4, embed_dim=192``:
        crops_per_axis = 1024 / 64 = 16
        patches_per_crop_axis = 64 / 4 = 16
        output spatial = 16 * 16 = 256  →  (B, 192, 256, 256)
    """

    def __init__(self, mae_model: MaskedAutoencoderViT,
                 img_size: int = 64, patch_size: int = 4,
                 embed_dim: int = 192, crop_batch_size: int = 256):
        super().__init__()
        self.mae = mae_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patches_per_axis = img_size // patch_size
        self.crop_batch_size = crop_batch_size

    def _has_trainable_params(self) -> bool:
        return any(p.requires_grad for p in self.mae.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        cs = self.img_size
        crops_h, crops_w = H // cs, W // cs

        # (B, C, crops_h, cs, crops_w, cs) → (B * crops, C, cs, cs)
        crops = (
            x.unfold(2, cs, cs)
             .unfold(3, cs, cs)
             .contiguous()
             .view(B * crops_h * crops_w, C, cs, cs)
        )

        use_grad = self._has_trainable_params() and torch.is_grad_enabled()

        # Chunked encoding to limit GPU memory
        latents = []
        for i in range(0, crops.shape[0], self.crop_batch_size):
            chunk = crops[i : i + self.crop_batch_size]
            if use_grad:
                lat = self.mae.infer_latent(chunk)[:, 1:, :]
            else:
                with torch.no_grad():
                    lat = self.mae.infer_latent(chunk)[:, 1:, :]
            latents.append(lat)
        latent = torch.cat(latents, dim=0)  # (B*crops, patches², embed_dim)

        pa = self.patches_per_axis
        # (B, crops_h, crops_w, pa, pa, D)
        latent = latent.view(B, crops_h, crops_w, pa, pa, self.embed_dim)
        # interleave crop-grid and patch-grid → (B, H_feat, W_feat, D)
        latent = latent.permute(0, 1, 3, 2, 4, 5).contiguous()
        feat_h = crops_h * pa
        feat_w = crops_w * pa
        latent = latent.view(B, feat_h, feat_w, self.embed_dim)
        # channels-first for conv layers
        return latent.permute(0, 3, 1, 2)   # (B, D, feat_h, feat_w)


# ── Convenience: build & load ───────────────────────────────────────────────

def build_mae_feature_extractor(
    checkpoint_path: str,
    img_size: int = 64,
    patch_size: int = 4,
    embed_dim: int = 192,
    depth: int = 14,
    num_heads: int = 1,
    decoder_embed_dim: int = 128,
    decoder_depth: int = 7,
    decoder_num_heads: int = 8,
    mlp_ratio: float = 2.0,
    pos_encode_weight: float = 0.08,
    crop_batch_size: int = 256,
    device: str = "cpu",
    freeze: bool = True,
) -> MAEFeatureExtractor:
    """Instantiate MAE, load checkpoint, wrap in ``MAEFeatureExtractor``."""

    mae = MaskedAutoencoderViT(
        img_size=img_size, patch_size=patch_size, in_chans=1,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_encode_weight=pos_encode_weight,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    mae.load_state_dict(state, strict=True)
    print(f"[MauNet] Loaded MAE checkpoint from {checkpoint_path}")

    if freeze:
        for p in mae.parameters():
            p.requires_grad = False
        mae.eval()

    extractor = MAEFeatureExtractor(
        mae, img_size=img_size, patch_size=patch_size,
        embed_dim=embed_dim, crop_batch_size=crop_batch_size,
    )
    return extractor.to(device)


def unfreeze_mae_last_n_blocks(extractor: MAEFeatureExtractor, n: int):
    """Unfreeze the last *n* encoder Transformer blocks + the final LayerNorm
    inside the wrapped MAE model, enabling fine-tuning of high-level features."""
    if n <= 0:
        return []
    mae = extractor.mae
    total = len(mae.blocks)
    unfreeze_start = max(0, total - n)
    unfrozen_params = []
    for i in range(unfreeze_start, total):
        for p in mae.blocks[i].parameters():
            p.requires_grad = True
            unfrozen_params.append(p)
    for p in mae.norm.parameters():
        p.requires_grad = True
        unfrozen_params.append(p)
    print(f"[MauNet] Unfroze MAE blocks [{unfreeze_start}..{total-1}] + norm "
          f"({sum(p.numel() for p in unfrozen_params)/1e6:.2f}M params)")
    return unfrozen_params
