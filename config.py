import argparse
import torch

parser = argparse.ArgumentParser(
    description="MauNet: ViT-MAE + Attention U-Net for Cryo-EM Particle Picking",
    allow_abbrev=False,
)

# ── Data ──
# Raw per-EMPIAR layout expected:
#   <root>/<EMPIAR_ID>/train/images/<base>.png
#   <root>/<EMPIAR_ID>/train/masks/<base>_mask.png
#   <root>/<EMPIAR_ID>/train/particle_coordinates/<base>.csv   (x,y,radius)
#   <root>/<EMPIAR_ID>/train/false_positives/<base>.csv        (x,y,radius)
parser.add_argument("--train_dataset_path", type=str,
                    default="dataset/train_dataset_1024/")
parser.add_argument("--output_path", type=str, default="output")
parser.add_argument("--val_ratio", type=float, default=0.1,
                    help="每个 EMPIAR 内图像级 train/val 切分的 val 比例")
parser.add_argument("--split_seed", type=int, default=42,
                    help="train/val 切分随机种子，保证可复现")
parser.add_argument("--normalize_image", action="store_true", default=True,
                    help="对每张输入做 per-image z-score 归一化，跨 EMPIAR 对比度更稳")

# ── Device ──
parser.add_argument(
    "--device", type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--pin_memory", action="store_true")
parser.add_argument("--num_workers", type=int, default=8)

# ── Input ──
parser.add_argument("--input_image_width", type=int, default=1024)
parser.add_argument("--input_image_height", type=int, default=1024)

# ── Training ──
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--save_every", type=int, default=50)

# ── MAE Encoder ──
parser.add_argument("--mae_checkpoint", type=str,
                    default="MAE_checkpoint/MAE_epoch_500.pth.tar")
parser.add_argument("--mae_img_size", type=int, default=64)
parser.add_argument("--mae_patch_size", type=int, default=4)
parser.add_argument("--mae_embed_dim", type=int, default=192)
parser.add_argument("--mae_depth", type=int, default=14)
parser.add_argument("--mae_num_heads", type=int, default=1)
parser.add_argument("--mae_decoder_embed_dim", type=int, default=128)
parser.add_argument("--mae_decoder_depth", type=int, default=7)
parser.add_argument("--mae_decoder_num_heads", type=int, default=8)
parser.add_argument("--mae_mlp_ratio", type=float, default=2.0)
parser.add_argument("--mae_pos_encode_weight", type=float, default=0.08)
parser.add_argument("--freeze_mae", action="store_true", default=True)
parser.add_argument("--mae_unfreeze_last_n", type=int, default=0,
                    help="解冻 MAE encoder 最后 N 个 Transformer block（0=全冻结）")
parser.add_argument("--mae_finetune_lr_scale", type=float, default=0.1,
                    help="解冻 MAE 层的学习率 = learning_rate * 此倍率")
parser.add_argument("--mae_crop_batch_size", type=int, default=256,
                    help="Max crops per MAE forward pass (tune for GPU memory)")

# ── Heatmap / Keypoint Head ──
parser.add_argument("--heatmap_sigma_scale", type=float, default=1.0 / 3.0,
                    help="高斯核 σ = sigma_scale * particle_radius")
parser.add_argument("--fp_neg_weight", type=float, default=3.0,
                    help="false_positives 圆盘内像素在 heatmap 负样本 loss 上的加权系数")
parser.add_argument("--heatmap_bias_pi", type=float, default=0.01,
                    help="heatmap head 输出层 bias 初始化的先验概率 π（focal loss 稳启动）")

# ── Loss Weights (multi-task) ──
parser.add_argument("--lambda_heatmap", type=float, default=1.0,
                    help="主任务 heatmap loss 权重")
parser.add_argument("--lambda_mask", type=float, default=0.3,
                    help="辅助任务 mask loss 权重；0=禁用 mask 头")

# ── CenterNet Focal ──
parser.add_argument("--centernet_alpha", type=float, default=2.0,
                    help="CenterNet focal loss 的 p-focusing 指数 α")
parser.add_argument("--centernet_beta", type=float, default=4.0,
                    help="CenterNet focal loss 的 target 衰减指数 β")

# ── Auxiliary Mask Loss (sigmoid-supervised) ──
parser.add_argument("--focal_alpha", type=float, default=0.75,
                    help="Mask Focal Loss 前景权重")
parser.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Mask Focal Loss focusing 参数")
parser.add_argument("--tversky_alpha", type=float, default=0.3,
                    help="Mask Tversky Loss FP 权重")
parser.add_argument("--tversky_beta", type=float, default=0.7,
                    help="Mask Tversky Loss FN 权重")

# ── Heatmap decoding / evaluation ──
parser.add_argument("--score_threshold", type=float, default=0.3,
                    help="预测点保留的最小 heatmap 分数")
parser.add_argument("--peak_nms_scale", type=float, default=0.5,
                    help="局部极大值 NMS 半径 = peak_nms_scale * particle_radius；"
                         "推理时颗粒半径未知，则用 peak_nms_fallback_radius")
parser.add_argument("--peak_nms_fallback_radius", type=int, default=10,
                    help="推理时无 GT radius 时使用的固定 NMS 半径（像素）")
parser.add_argument("--match_radius_scale", type=float, default=0.5,
                    help="F1 评估时，pred 与 gt 匹配的距离阈值 = k * gt_radius；"
                         "k 越小越严格（推荐 0.5）")
parser.add_argument("--max_predictions", type=int, default=2000,
                    help="每张图保留的最多预测点（按 score 降序截断）")

# ── Gradient Clipping ──
parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="梯度裁剪 max_norm（0=禁用）")

# ── LR Scheduler ──
parser.add_argument("--lr_min", type=float, default=1e-6,
                    help="CosineAnnealing 最低学习率")

# ── Early Stopping ──
parser.add_argument("--early_stopping_patience", type=int, default=30,
                    help="连续多少个 epoch val F1 不提升则停止（0=禁用）")

# ── Optional pretrained U-Net weights (CryoSegNet) ──
parser.add_argument("--unet_checkpoint", type=str, default=None)

# ── Resume ──
parser.add_argument("--maunet_checkpoint", type=str, default=None,
                    help="从已有的 MauNet checkpoint 恢复训练 / warm-start")

args, _unknown = parser.parse_known_args()

# Expose as module-level variables for easy access
train_dataset_path = args.train_dataset_path
output_path = args.output_path
val_ratio = args.val_ratio
split_seed = args.split_seed
normalize_image = args.normalize_image
device = args.device
pin_memory = args.pin_memory
num_workers = args.num_workers
input_image_width = args.input_image_width
input_image_height = args.input_image_height
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
weight_decay = args.weight_decay
save_every = args.save_every
mae_checkpoint = args.mae_checkpoint
mae_img_size = args.mae_img_size
mae_patch_size = args.mae_patch_size
mae_embed_dim = args.mae_embed_dim
mae_depth = args.mae_depth
mae_num_heads = args.mae_num_heads
mae_decoder_embed_dim = args.mae_decoder_embed_dim
mae_decoder_depth = args.mae_decoder_depth
mae_decoder_num_heads = args.mae_decoder_num_heads
mae_mlp_ratio = args.mae_mlp_ratio
mae_pos_encode_weight = args.mae_pos_encode_weight
freeze_mae = args.freeze_mae
mae_unfreeze_last_n = args.mae_unfreeze_last_n
mae_finetune_lr_scale = args.mae_finetune_lr_scale
mae_crop_batch_size = args.mae_crop_batch_size
heatmap_sigma_scale = args.heatmap_sigma_scale
fp_neg_weight = args.fp_neg_weight
heatmap_bias_pi = args.heatmap_bias_pi
lambda_heatmap = args.lambda_heatmap
lambda_mask = args.lambda_mask
centernet_alpha = args.centernet_alpha
centernet_beta = args.centernet_beta
focal_alpha = args.focal_alpha
focal_gamma = args.focal_gamma
tversky_alpha = args.tversky_alpha
tversky_beta = args.tversky_beta
score_threshold = args.score_threshold
peak_nms_scale = args.peak_nms_scale
peak_nms_fallback_radius = args.peak_nms_fallback_radius
match_radius_scale = args.match_radius_scale
max_predictions = args.max_predictions
grad_clip_norm = args.grad_clip_norm
lr_min = args.lr_min
early_stopping_patience = args.early_stopping_patience
unet_checkpoint = args.unet_checkpoint
maunet_checkpoint = args.maunet_checkpoint
