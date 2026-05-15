"""
MauNet training script — heatmap-based end-to-end particle picking.

The model outputs two heads:
  * heatmap : Gaussian-rendered particle centers (main task, used at inference)
  * mask    : auxiliary segmentation supervision (training only)

Single GPU:
    python train.py --batch_size 4 --num_epochs 200

Multi-GPU (e.g. GPU 0, 1, 3):
    CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 train.py --batch_size 4

Optional: warm-start the U-Net from a CryoSegNet checkpoint:
    ... --unet_checkpoint pretrained_models/cryosegnet.pth

Resume from a MauNet checkpoint:
    ... --maunet_checkpoint MauNet_checkpoint/MauNet_pretrained.pth
"""

import os
import time
from datetime import date

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import config
from models.mae_encoder import build_mae_feature_extractor, unfreeze_mae_last_n_blocks
from models.maunet import MauNet
from dataset.dataset import (
    CryoEMDataset,
    build_train_val_split,
    cryoem_collate,
)
from utils.loss import FocalLoss, TverskyLoss, CenterNetFocalLoss
from utils.accuracy import (
    dice_score,
    jaccard_score,
    decode_heatmap,
    keypoint_f1,
    image_mean_radius,
)


def _setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _compute_loss(
    outputs: dict,
    batch: dict,
    centernet_loss: nn.Module,
    mask_focal: nn.Module,
    mask_tversky: nn.Module,
) -> tuple[torch.Tensor, dict]:
    heatmap_logits = outputs["heatmap"]
    mask_logits = outputs["mask"]
    heatmap_gt = batch["heatmap"]
    mask_gt = batch["mask"]
    fp_weight = batch["fp_weight"]

    loss_hm = centernet_loss(heatmap_logits, heatmap_gt, neg_weight=fp_weight)
    loss_mask = torch.zeros((), device=heatmap_logits.device)
    if config.lambda_mask > 0:
        lf = mask_focal(mask_logits, mask_gt)
        lt = mask_tversky(torch.sigmoid(mask_logits), mask_gt)
        loss_mask = 0.5 * (lf + lt)

    loss = config.lambda_heatmap * loss_hm + config.lambda_mask * loss_mask
    return loss, {
        "loss": loss.detach(),
        "loss_heatmap": loss_hm.detach(),
        "loss_mask": loss_mask.detach(),
    }


def _ddp_reduce_sum(values: list[float], device) -> list[float]:
    """跨 rank 求和聚合一组标量（只在 DDP 初始化时才 reduce）。"""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.cpu().tolist()


@torch.no_grad()
def _validate(model, loader, device, centernet_loss, mask_focal, mask_tversky, is_main):
    model.eval()
    loss_sum = 0.0
    hm_loss_sum = 0.0
    mask_loss_sum = 0.0
    dice_sum = 0.0
    jac_sum = 0.0
    n_batches_local = 0  # 本 rank 实际迭代的 batch 数

    all_pred_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    all_gt_coords: list[torch.Tensor] = []
    all_gt_radii: list[torch.Tensor] = []

    loader_iter = tqdm(loader, desc="[val]", leave=False) if is_main else loader
    for batch in loader_iter:
        x = batch["image"].to(device, non_blocking=True)
        batch_gpu = {
            "heatmap": batch["heatmap"].to(device, non_blocking=True),
            "mask": batch["mask"].to(device, non_blocking=True),
            "fp_weight": batch["fp_weight"].to(device, non_blocking=True),
        }
        outputs = model(x)
        loss, stats = _compute_loss(
            outputs, batch_gpu, centernet_loss, mask_focal, mask_tversky
        )
        loss_sum += stats["loss"].item()
        hm_loss_sum += stats["loss_heatmap"].item()
        mask_loss_sum += stats["loss_mask"].item()
        n_batches_local += 1

        prob_mask = torch.sigmoid(outputs["mask"])
        dice_sum += float(dice_score(batch_gpu["mask"], prob_mask).item())
        jac_sum += float(jaccard_score(batch_gpu["mask"], prob_mask).item())

        # Per-image keypoint decode + collect for F1 (keeps tp/fp/fn counts that are
        # trivially additive across DDP ranks)
        prob_hm = torch.sigmoid(outputs["heatmap"])
        for b in range(prob_hm.size(0)):
            r_hint = image_mean_radius(batch["gt_radii"][b])
            nms_r = max(int(round(config.peak_nms_scale * r_hint)), 1)
            dec = decode_heatmap(
                prob_hm[b : b + 1],
                score_threshold=config.score_threshold,
                nms_radius=nms_r,
                max_predictions=config.max_predictions,
            )[0]
            all_pred_results.append((dec[0].cpu(), dec[1].cpu()))
            all_gt_coords.append(batch["gt_coords"][b])
            all_gt_radii.append(batch["gt_radii"][b])

    # 本 rank 的 F1 中间量（tp/fp/fn 可线性跨 rank 相加）
    f1_local = keypoint_f1(
        all_pred_results,
        all_gt_coords,
        all_gt_radii,
        match_radius_scale=config.match_radius_scale,
        use_hungarian=False,
    )

    # ── DDP 聚合：把所有标量求和后再算指标 ──
    agg = _ddp_reduce_sum(
        [
            loss_sum, hm_loss_sum, mask_loss_sum,
            dice_sum, jac_sum,
            float(n_batches_local),
            float(f1_local["tp"]), float(f1_local["fp"]), float(f1_local["fn"]),
        ],
        device=device,
    )
    (loss_s, hm_s, mask_s, dice_s, jac_s, n_b, tp, fp, fn) = agg
    n_b = max(n_b, 1.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": loss_s / n_b,
        "loss_heatmap": hm_s / n_b,
        "loss_mask": mask_s / n_b,
        "dice": dice_s / n_b,
        "jaccard": jac_s / n_b,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    local_rank, world_size, is_ddp = _setup_ddp()
    is_main = local_rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        os.makedirs(f"{config.output_path}/models", exist_ok=True)

    # ── 1. Build MAE feature extractor (frozen) ──────────────────────
    if is_main:
        print("[MauNet] Building MAE feature extractor ...")
    mae_extractor = build_mae_feature_extractor(
        checkpoint_path=config.mae_checkpoint,
        img_size=config.mae_img_size,
        patch_size=config.mae_patch_size,
        embed_dim=config.mae_embed_dim,
        depth=config.mae_depth,
        num_heads=config.mae_num_heads,
        decoder_embed_dim=config.mae_decoder_embed_dim,
        decoder_depth=config.mae_decoder_depth,
        decoder_num_heads=config.mae_decoder_num_heads,
        mlp_ratio=config.mae_mlp_ratio,
        pos_encode_weight=config.mae_pos_encode_weight,
        crop_batch_size=config.mae_crop_batch_size,
        device=str(device),
        freeze=config.freeze_mae,
    )

    # ── 2. Build MauNet model ────────────────────────────────────────
    if is_main:
        print("[MauNet] Building MauNet network ...")
    model = MauNet(
        mae_extractor,
        mae_embed_dim=config.mae_embed_dim,
        heatmap_bias_pi=config.heatmap_bias_pi,
    ).to(device)

    if config.unet_checkpoint:
        unet_state = torch.load(config.unet_checkpoint, map_location="cpu")
        if "model" in unet_state:
            unet_state = unet_state["model"]
        model.load_unet_pretrained(unet_state)

    if config.maunet_checkpoint:
        ckpt = torch.load(config.maunet_checkpoint, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            print(f"[MauNet] Resumed from {config.maunet_checkpoint}")
            if missing:
                print(f"  ! Missing keys: {len(missing)} (e.g. {missing[:3]})")
            if unexpected:
                print(f"  ! Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    if is_ddp:
        has_mae_unfrozen = config.mae_unfreeze_last_n > 0 and config.freeze_mae
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=has_mae_unfrozen)

    raw_model = model.module if is_ddp else model

    # ── 3. Dataset & DataLoader ──────────────────────────────────────
    train_paths, val_paths, split_mode = build_train_val_split(
        config.train_dataset_path,
        val_ratio=config.val_ratio,
        seed=config.split_seed,
    )
    if not train_paths:
        raise FileNotFoundError(
            f"No training images found under {config.train_dataset_path}\n"
            f"Expected layout: <root>/<EMPIAR_ID>/{{train,val}}/images/*.png "
            f"(val/ optional)"
        )
    if is_main:
        if split_mode == "pre_split":
            print(f"[MauNet] Using pre-split layout: <EMPIAR>/{{train,val}}/ "
                  f"(val_ratio / split_seed ignored)")
        else:
            print(f"[MauNet] No val/ found; randomly splitting train/ "
                  f"(val_ratio={config.val_ratio}, seed={config.split_seed})")

    train_ds = CryoEMDataset(train_paths, augment=True)
    val_ds = CryoEMDataset(val_paths, augment=False) if val_paths else None

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp and val_ds else None

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=config.pin_memory, num_workers=config.num_workers,
        collate_fn=cryoem_collate,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            sampler=val_sampler,
            pin_memory=config.pin_memory, num_workers=config.num_workers,
            collate_fn=cryoem_collate,
        )
        if val_ds else None
    )

    if is_main:
        print(f"[MauNet] Train: {len(train_ds)} images | Val: {len(val_ds) if val_ds else 0} images")

    # ── 3b. Optionally unfreeze last N MAE blocks ─────────────────
    mae_unfrozen_params = unfreeze_mae_last_n_blocks(
        raw_model.mae_extractor, config.mae_unfreeze_last_n
    )

    if is_main:
        total = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"[MauNet] Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
        if is_ddp:
            print(f"[MauNet] DDP: {world_size} GPUs | effective batch size = {config.batch_size * world_size}")
        for name in ["alpha_s3", "alpha_s4", "alpha_s5", "alpha_bn"]:
            print(f"  {name} = {getattr(raw_model, name).item():.4f}")

    # ── 4. Optimizer & Loss ──────────────────────────────────────────
    centernet_loss = CenterNetFocalLoss(
        alpha=config.centernet_alpha, beta=config.centernet_beta
    )
    mask_focal = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    mask_tversky = TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)

    mae_unfrozen_ids = {id(p) for p in mae_unfrozen_params}
    base_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in mae_unfrozen_ids]
    param_groups = [{"params": base_params}]
    if mae_unfrozen_params:
        param_groups.append({
            "params": mae_unfrozen_params,
            "lr": config.learning_rate * config.mae_finetune_lr_scale,
        })
    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr_min
    )

    best_val_f1 = -1.0
    patience = config.early_stopping_patience
    epochs_no_improve = 0
    start = time.time()

    # ── 5. Training loop ─────────────────────────────────────────────
    for epoch in range(config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        if config.mae_unfreeze_last_n == 0:
            raw_model.mae_extractor.eval()

        train_loss_sum = 0.0
        train_hm_sum = 0.0
        train_mask_sum = 0.0
        train_dice_sum = 0.0
        train_jac_sum = 0.0
        n_train_batches_local = 0

        loader_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [train]")
            if is_main else train_loader
        )
        for batch in loader_iter:
            x = batch["image"].to(device, non_blocking=True)
            batch_gpu = {
                "heatmap": batch["heatmap"].to(device, non_blocking=True),
                "mask": batch["mask"].to(device, non_blocking=True),
                "fp_weight": batch["fp_weight"].to(device, non_blocking=True),
            }

            optimizer.zero_grad()
            outputs = model(x)
            loss, stats = _compute_loss(
                outputs, batch_gpu, centernet_loss, mask_focal, mask_tversky
            )

            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip_norm
                )
            optimizer.step()

            train_loss_sum += stats["loss"].item()
            train_hm_sum += stats["loss_heatmap"].item()
            train_mask_sum += stats["loss_mask"].item()
            n_train_batches_local += 1
            with torch.no_grad():
                prob_mask = torch.sigmoid(outputs["mask"])
                train_dice_sum += float(dice_score(batch_gpu["mask"], prob_mask).item())
                train_jac_sum += float(jaccard_score(batch_gpu["mask"], prob_mask).item())

        # ── 训练指标跨 rank 聚合（单卡下 no-op）──
        agg_tr = _ddp_reduce_sum(
            [
                train_loss_sum, train_hm_sum, train_mask_sum,
                train_dice_sum, train_jac_sum,
                float(n_train_batches_local),
            ],
            device=device,
        )
        tr_loss_s, tr_hm_s, tr_mask_s, tr_dice_s, tr_jac_s, tr_n_b = agg_tr
        tr_n_b = max(tr_n_b, 1.0)
        train_loss = tr_loss_s / tr_n_b
        train_hm = tr_hm_s / tr_n_b
        train_mask = tr_mask_s / tr_n_b
        train_dice = tr_dice_s / tr_n_b
        train_jac = tr_jac_s / tr_n_b

        # ── validate ──
        val_stats = None
        if val_loader:
            val_stats = _validate(
                model, val_loader, device,
                centernet_loss, mask_focal, mask_tversky, is_main,
            )

        scheduler.step()

        # ── logging (main process only) ──
        if is_main:
            elapsed = time.time() - start
            msg = (
                f"[Epoch {epoch+1:>3d}] "
                f"train: loss={train_loss:.4f} hm={train_hm:.4f} mask={train_mask:.4f} "
                f"dice={train_dice:.4f} jac={train_jac:.4f}"
            )
            if val_stats:
                msg += (
                    f" | val: loss={val_stats['loss']:.4f} hm={val_stats['loss_heatmap']:.4f} "
                    f"mask={val_stats['loss_mask']:.4f} "
                    f"P={val_stats['precision']:.3f} R={val_stats['recall']:.3f} "
                    f"F1={val_stats['f1']:.3f} "
                    f"tp/fp/fn={val_stats['tp']}/{val_stats['fp']}/{val_stats['fn']} "
                    f"dice={val_stats['dice']:.3f}"
                )
            msg += f" | time={elapsed/60:.1f}min"
            print(msg)
            print(
                f"  alpha_s3={raw_model.alpha_s3.item():.3f} "
                f"alpha_s4={raw_model.alpha_s4.item():.3f} "
                f"alpha_s5={raw_model.alpha_s5.item():.3f} "
                f"alpha_bn={raw_model.alpha_bn.item():.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # ── periodic checkpoint ──
            if (epoch + 1) % config.save_every == 0:
                path = os.path.join(
                    config.output_path, "models",
                    f"maunet_epoch_{epoch+1}_{date.today()}.pth",
                )
                torch.save({"model": raw_model.state_dict(), "epoch": epoch + 1}, path)

            # ── best ckpt by F1 (fallback to loss if no val) ──
            if val_stats is not None:
                if val_stats["f1"] > best_val_f1:
                    best_val_f1 = val_stats["f1"]
                    epochs_no_improve = 0
                    torch.save(
                        {"model": raw_model.state_dict(),
                         "epoch": epoch + 1,
                         "val_f1": best_val_f1},
                        os.path.join(config.output_path, "models", "maunet_best.pth"),
                    )
                    print(f"  ✓ Best val F1 so far: {best_val_f1:.4f}")
                else:
                    epochs_no_improve += 1
                    if patience > 0 and epochs_no_improve >= patience:
                        print(f"  ✗ Early stopping: val F1 连续 {patience} epoch 未提升")
                        break

    if is_main:
        print(f"[MauNet] Training complete in {(time.time()-start)/60:.1f} min "
              f"| Best val F1 = {best_val_f1:.4f}")
    _cleanup_ddp()


if __name__ == "__main__":
    main()
