"""
使用训练好的 MauNet 权重**直接输出蛋白颗粒坐标**（CSV：x,y,score）。

用法（在项目根目录下执行）::

    python test_predict.py \\
        --checkpoint MauNet_checkpoint/MauNet_pretrained.pth \\
        --input_dir dataset/cryo_em_dataset_test/10028_image_1024 \\
        --radius 19 \\
        --output output/predictions/10028_image_1024

说明：
    - 模型输入与训练一致（灰度图、resize 到 config 尺寸、可选 z-score 归一化）。
    - 推理时会用 heatmap 头做 3×3 局部极大值 + score 阈值抽取颗粒中心。
    - ``--radius`` 用于 NMS 半径 = ``peak_nms_scale * radius`` 以及可选的坐标回放尺度。
      若该数据集的颗粒半径明显不同，请传入对应值；未传入则回退到
      ``config.peak_nms_fallback_radius``。
    - 默认只产出坐标 CSV；使用 ``--save_heatmap`` / ``--save_mask`` 可选保存可视化。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

import config
from models.mae_encoder import build_mae_feature_extractor
from models.maunet import MauNet
from utils.accuracy import decode_heatmap


def _list_images(folder: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            paths.append(os.path.join(folder, name))
    return paths


def _load_micrograph(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """返回 (1,1,H,W) 张量 与 原图尺寸 (H_orig, W_orig)；预处理与训练保持一致。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    H_orig, W_orig = img.shape[:2]
    H_out, W_out = config.input_image_height, config.input_image_width
    if (H_orig, W_orig) != (H_out, W_out):
        img_resized = cv2.resize(img, (W_out, H_out), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    img_f = img_resized.astype(np.float32) / 255.0
    if config.normalize_image:
        mean, std = img_f.mean(), img_f.std()
        if std >= 1e-6:
            img_f = (img_f - mean) / std
            img_f = np.clip(img_f, -5.0, 5.0)
        else:
            img_f = img_f - mean
    t = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0).float()
    return t, (H_orig, W_orig)


def _write_coords_csv(path: str, coords_xy: np.ndarray, scores: np.ndarray, radius: float) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "score", "radius"])
        for (x, y), s in zip(coords_xy, scores):
            w.writerow([f"{x:.2f}", f"{y:.2f}", f"{float(s):.4f}", f"{radius:.2f}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MauNet 测试：直接输出颗粒坐标 CSV",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", type=str,
                        default="MauNet_checkpoint/MauNet_pretrained.pth",
                        help="训练权重路径（默认指向仓库附带的预训练权重）")
    parser.add_argument("--input_dir", "--input", dest="input_dir", type=str,
                        default="dataset/cryo_em_dataset_test/10028_image_1024",
                        help="测试图像目录（扁平放置 .png/.jpg 等）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录；默认 output/predictions/<输入文件夹名>")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖 config.device，例如 cuda:0 或 cpu")

    parser.add_argument("--radius", type=float, default=None,
                        help="该数据集的颗粒半径（像素，训练分辨率下）；影响 NMS 半径与 CSV radius 字段")
    parser.add_argument("--score_threshold", type=float, default=None,
                        help="分数阈值；默认取 config.score_threshold")
    parser.add_argument("--nms_scale", type=float, default=None,
                        help="NMS 半径倍率；默认取 config.peak_nms_scale")
    parser.add_argument("--max_predictions", type=int, default=None,
                        help="每张图最多保留的峰值数；默认取 config.max_predictions")

    parser.add_argument("--save_heatmap", action="store_true",
                        help="额外保存 heatmap 概率图 (*_heatmap.png)")
    parser.add_argument("--save_mask", action="store_true",
                        help="额外保存辅助 mask 概率图 (*_mask.png)")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="mask 二值化阈值（仅在 --save_mask 时使用）")
    parser.add_argument("--rescale_to_original", action="store_true",
                        help="将坐标回放到原图分辨率（默认写在训练分辨率）")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理 batch 大小；1024 大图建议保持 1")
    args, _unknown = parser.parse_known_args()

    device_str = args.device or config.device
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[test_predict] CUDA 不可用，改用 CPU", file=sys.stderr)
        device = torch.device("cpu")

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    out_dir = os.path.abspath(args.output) if args.output else os.path.join(
        os.path.abspath(config.output_path),
        "predictions",
        os.path.basename(input_dir.rstrip(os.sep)),
    )
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    paths = _list_images(input_dir)
    if not paths:
        raise FileNotFoundError(f"目录下未找到图像: {input_dir}")

    radius = float(args.radius) if args.radius is not None else float(config.peak_nms_fallback_radius)
    nms_scale = args.nms_scale if args.nms_scale is not None else config.peak_nms_scale
    score_thr = args.score_threshold if args.score_threshold is not None else config.score_threshold
    max_pred = args.max_predictions if args.max_predictions is not None else config.max_predictions
    nms_radius_px = max(int(round(nms_scale * radius)), 1)

    print(f"[test_predict] device={device}  images={len(paths)}")
    print(f"[test_predict] input={input_dir}")
    print(f"[test_predict] output={out_dir}")
    print(f"[test_predict] ckpt={ckpt_path}")
    print(f"[test_predict] radius={radius}  nms_radius_px={nms_radius_px}  "
          f"score_thr={score_thr}  max_pred={max_pred}")

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

    model = MauNet(
        mae_extractor,
        mae_embed_dim=config.mae_embed_dim,
        heatmap_bias_pi=config.heatmap_bias_pi,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[test_predict] 缺失 keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[test_predict] 多余 keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    model.mae_extractor.eval()

    bs = max(1, args.batch_size)

    with torch.no_grad():
        for start in range(0, len(paths), bs):
            batch_paths = paths[start : start + bs]
            tensors_and_sizes = [_load_micrograph(p) for p in batch_paths]
            x = torch.cat([t for t, _ in tensors_and_sizes], dim=0).to(device)
            orig_sizes = [sz for _, sz in tensors_and_sizes]

            outputs = model(x)
            heatmap_prob = torch.sigmoid(outputs["heatmap"])
            mask_prob = torch.sigmoid(outputs["mask"]) if args.save_mask else None

            decoded = decode_heatmap(
                heatmap_prob,
                score_threshold=score_thr,
                nms_radius=nms_radius_px,
                max_predictions=max_pred,
            )

            for i, p in enumerate(batch_paths):
                base = os.path.splitext(os.path.basename(p))[0]
                coords, scores = decoded[i]
                coords_np = coords.cpu().numpy() if coords.numel() else np.zeros((0, 2), dtype=np.float32)
                scores_np = scores.cpu().numpy() if scores.numel() else np.zeros((0,), dtype=np.float32)

                out_radius = radius
                if args.rescale_to_original and coords_np.size:
                    H_orig, W_orig = orig_sizes[i]
                    sx = W_orig / float(config.input_image_width)
                    sy = H_orig / float(config.input_image_height)
                    coords_np = coords_np.copy()
                    coords_np[:, 0] *= sx
                    coords_np[:, 1] *= sy
                    out_radius = radius * float(np.sqrt(sx * sy))

                csv_path = os.path.join(out_dir, f"{base}.csv")
                _write_coords_csv(csv_path, coords_np, scores_np, out_radius)

                if args.save_heatmap:
                    hm_u8 = np.clip(heatmap_prob[i, 0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(out_dir, f"{base}_heatmap.png"), hm_u8)
                if args.save_mask and mask_prob is not None:
                    mask_bin = (mask_prob[i, 0].cpu().numpy() >= args.mask_threshold).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(out_dir, f"{base}_mask.png"), mask_bin)

            print(f"  已处理 {min(start + bs, len(paths))}/{len(paths)}")

    print(f"[test_predict] 完成，CSV 已保存至: {out_dir}")


if __name__ == "__main__":
    main()
