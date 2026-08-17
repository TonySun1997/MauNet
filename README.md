# CryoMauNet

**CryoMauNet** — *ViT-MAE + Attention U-Net with Multi-Scale Feature Injection* —
an end-to-end deep network for cryo-EM **particle picking** on full-size
micrographs (1024×1024).

Two prediction heads are trained jointly:

- **heatmap** head — CenterNet-style Gaussian-rendered particle centers
(main task, used at inference).
- **mask** head — auxiliary segmentation supervision (training-only,
improves feature learning).

A frozen ViT-MAE encoder produces a dense feature map that is projected to
four spatial scales and **zero-init fused** into an Attention U-Net's skip
connections, so the network starts as a pure U-Net and gradually
incorporates MAE features as training proceeds.

> **Naming note:** The method and this repository are named **CryoMauNet**.
> Internal module / class / checkpoint filenames may still use the historical
> prefix `MauNet` / `maunet` (e.g. `models/maunet.py`, `MauNet_checkpoint/`);
> the conda environment remains `maunet`.

---

## 1. Installation

```bash
git clone https://github.com/TonySun1997/CryoMauNet.git
cd CryoMauNet

conda env create -f environment.yml
conda activate maunet
```

Large artifacts (MAE / CryoMauNet weights, training set, and a test sample) are **not**
in the Git history. Download them from Zenodo and place them as below.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20155024.svg)](https://doi.org/10.5281/zenodo.20155024)

**Record:** [https://doi.org/10.5281/zenodo.20155024](https://doi.org/10.5281/zenodo.20155024)  
The dataset files there are published under **Creative Commons Attribution 4.0
International (CC BY 4.0)**; credit the record when you reuse the data.

### 1.1 Checkpoints (no unpacking)

| Zenodo file | MD5 (verify after download) | Copy to |
| ----------- | --------------------------- | ------- |
| `MAE_epoch_500.pth.tar` | `ff29489d897dee4250a98630c015ceb9` | `MAE_checkpoint/MAE_epoch_500.pth.tar` |
| `MauNet_pretrained.pth` | `33462fe83c6d417240cb13f6b1ebe2fa` | `MauNet_checkpoint/MauNet_pretrained.pth` |

```bash
mkdir -p MAE_checkpoint MauNet_checkpoint
# move or copy the two files you downloaded from Zenodo into these folders
mv /path/to/MAE_epoch_500.pth.tar     MAE_checkpoint/
mv /path/to/MauNet_pretrained.pth     MauNet_checkpoint/
```

### 1.2 Datasets (`.tar` archives — extract)

| Zenodo file | MD5 | Extract so this path exists |
| ----------- | ----- | ---------------------------- |
| `train_dataset_1024.tar` | `3a25500629f9f6e008b16f52f82a4ced` | `dataset/train_dataset_1024/` (see [§4](#4-data-layout-for-training)) |
| `10017_image_1024_all.tar` | `700126bf6626d1a1daf4681424d71c23` | `dataset/test_dataset_1024/10017_image_1024_all/` (used by [§3](#3-quick-start--inference-recommended-for-most-users) CLI example) |

```bash
mkdir -p dataset dataset/test_dataset_1024
tar -xf /path/to/train_dataset_1024.tar       -C dataset
tar -xf /path/to/10017_image_1024_all.tar     -C dataset/test_dataset_1024
```

If the first archive unpacks to a single top-level folder `train_dataset_1024/`,
the command above yields `dataset/train_dataset_1024/...` as required. If your
tar layout differs, adjust paths so they match **Section 4** (data layout) and the
`--input_dir` path in **Section 3** (quick start).

Integrity check (Linux) — compare with the MD5 values in the tables above:

```bash
md5sum MAE_checkpoint/MAE_epoch_500.pth.tar \
       MauNet_checkpoint/MauNet_pretrained.pth \
       /path/to/train_dataset_1024.tar \
       /path/to/10017_image_1024_all.tar
```

---

## 2. Project layout

```
CryoMauNet/
├── config.py                  # argparse + module-level config
├── train.py                   # training entry point (single-/multi-GPU)
├── test_predict.py            # CLI batch inference → particle coordinate CSV
├── flip_star_y.py             # flip STAR Y for MRC / RELION axis alignment
├── gui_predict.py             # Gradio GUI for interactive picking
├── denoise.py                 # MRC / image denoising helpers
├── models/
│   ├── maunet.py              # CryoMauNet network (class MauNet)
│   ├── mae_encoder.py         # MAE ViT + feature-extractor wrapper
│   └── unet_blocks.py         # Attention U-Net building blocks
├── dataset/
│   ├── dataset.py             # CryoEMDataset (heatmap + mask + fp weight)
│   ├── train_dataset_1024/    # from Zenodo train_dataset_1024.tar (§1.2)
│   └── test_dataset_1024/     # e.g. 10017_image_1024_all/ from Zenodo (§1.2)
├── utils/
│   ├── accuracy.py            # heatmap decode + keypoint F1
│   └── loss.py                # FocalLoss / TverskyLoss / CenterNetFocalLoss
├── MAE_checkpoint/            # pre-trained MAE ViT encoder weights
│   └── MAE_epoch_500.pth.tar
├── MauNet_checkpoint/         # ★ pre-trained CryoMauNet weights (ready to use)
│   └── MauNet_pretrained.pth
└── environment.yml
```

---

## 3. Quick start — Inference (recommended for most users)

### 3.1 Graphical User Interface (recommended)

To promote the practical use of CryoMauNet in cryo-EM research, we provide an
intuitive **Gradio** web GUI (`gui_predict.py`) for end-to-end particle picking.
The interface integrates one-click inference, real-time result visualization,
and standard-format export — no command-line expertise required. The layout
is split into two regions: a **left parameter panel** and a **right result
panel**.

#### Launch

```bash
conda activate maunet
python gui_predict.py
# → open http://127.0.0.1:7860  (LAN: http://<server-ip>:7860)
```

The default **Checkpoint path** is pre-filled with
`MauNet_checkpoint/MauNet_pretrained.pth`. Upload micrographs (or point to a
server path), adjust parameters if needed, then click **Run inference**.

Supported formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.mrc`, `.mrcs`
(single-frame and multi-frame stacks).

#### Left panel — parameter configuration

The left panel is the control area for the full picking pipeline: model loading,
data input, inference tuning, and output settings.

**Model and hardware**

| Control | Description |
| ------- | ----------- |
| **Checkpoint path (.pth)** | Path to a trained CryoMauNet weights file. |
| **Device** | `auto` (GPU if available), `cuda:0`, `cuda:1`, or `cpu`. Weights are cached after the first load. |

**Input data**

Two input modes can be combined (paths are de-duplicated):

| Mode | Description |
| ---- | ----------- |
| **Option A: upload** | Drag-and-drop or multi-select local files in the browser. |
| **Option B: server path** | Type a file or directory path on the machine running the GUI. A collapsible **Browse server filesystem** explorer can fill this field. |

**Core inference parameters**

These map directly to CryoMauNet's heatmap decoding logic (`utils/accuracy.py`):

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| **Particle boxsize** | 200 | Particle **diameter** in **original micrograph pixels**. Used to set the display circle size and to derive the NMS radius after letterboxing. Important for downstream box extraction and 3D reconstruction. |
| **Score threshold** | 0.10 | Minimum heatmap confidence; peaks below this value are discarded to suppress false positives. |
| **NMS scale** | 0.5 | Non-maximum suppression radius = `nms_scale × (boxsize / 2) × letterbox_scale`. Larger values merge overlapping detections more aggressively (fewer duplicates, risk of missing adjacent particles); smaller values keep more candidates. |
| **Max picks per image** | 2000 | Upper bound on detections per micrograph (or per MRCS frame), ranked by score. |

**Output and execution**

| Option | Default | Meaning |
| ------ | ------- | ------- |
| **Compute heatmap** | on | Save a heatmap view (inferno colormap) for debugging and qualitative assessment. |
| **Filter false peaks in letterbox padding** | on | Drop detections that fall outside the original image area after aspect-preserving letterbox resize (reduces spurious edge peaks). |
| **Enable denoising** | off | Optional preprocessing (`denoise.py`) before inference; applies to all formats including MRC/MRCS. |
| **STAR: micrograph name = basename** | on | RELION STAR `_rlnMicrographName` uses the file basename; unchecked uses the absolute path. |

Click **Run inference** to run the full pipeline: load → preprocess (letterbox +
per-image z-score) → CryoMauNet forward → heatmap decode → visualize → export.

Per-file statistics appear at the bottom of the left panel after a run completes.

#### Right panel — visualization and export

**Result visualization (plasma colormap)**

Detected particles are overlaid on the letterboxed micrograph. Circle color uses
the **plasma** colormap by normalized confidence within each image:

- **Purple** — lower scores (near the threshold);
- **Yellow** — higher scores (high-confidence picks).

Use the **View** toggle to switch between **Particle overlay** and **Heatmap**.
**Previous** / **Next** browse multiple inputs or MRCS frames; the counter shows
`index / total · filename`.

**Multi-format export**

| Format | Contents |
| ------ | -------- |
| **CSV** | Columns `filename`, `x`, `y`, `score`, `diameter` in **original image coordinates**. |
| **RELION STAR** | Autopick STAR with `_rlnMicrographName`, `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnAutopickFigureOfMerit` — importable into RELION, CryoSPARC, and similar tools. |

Download links appear above the main image after inference completes.

> **MRC / RELION axis note:** GUI coordinates follow the same convention as PNG
> inference (top-origin Y). If picks look vertically flipped on raw MRC stacks in
> RELION, apply the Y-flip step in [§3.3](#33-flip-star-y-for-mrc--relion).

### 3.2 CLI batch inference

```bash
python test_predict.py \
    --checkpoint MauNet_checkpoint/MauNet_pretrained.pth \
    --input_dir  dataset/test_dataset_1024/10017_image_1024_all \
    --output     output/predictions/10017 \
    --radius     12 \
    --score_threshold 0.1
```

This produces one `<base>.csv` per input image with columns
`x, y, score, radius` at **training resolution** (1024×1024).

Useful flags:


| Flag                    | Effect                                                                            |
| ----------------------- | --------------------------------------------------------------------------------- |
| `--save_heatmap`        | Also dump the heatmap probability map as PNG                                      |
| `--save_mask`           | Also dump the auxiliary mask probability map as PNG                               |
| `--rescale_to_original` | Output coordinates in the original micrograph resolution (simple uniform scaling) |

### 3.3 Flip STAR Y for MRC / RELION

CryoMauNet inference on **PNG** (or a display-oriented view) uses a **top-origin**
vertical axis. **MRC** micrographs in RELION often use the opposite convention,
so particle positions in an exported `.star` can appear vertically mirrored
when you open the same picks on the raw MRC stack.

Use `flip_star_y.py` to remap **Y only** (X unchanged):

```text
y_mrc = image_height - y_in
```

`image_height` is the micrograph height in pixels (**MRC `Ny`**, same units as
the coordinates in the STAR file — e.g. from `header.ny` in Python/mrcfile, or
RELION’s micrograph size).

```bash
python flip_star_y.py \
    --input  output/predictions/10017/picks.star \
    --height 4096 \
    --output output/predictions/10017/picks_mrc.star
```

| Flag | Meaning |
| ---- | ------- |
| `--input` / `-i` | Input RELION autopick `.star` (`_rlnCoordinateX`, `_rlnCoordinateY`) |
| `--height` / `-H` | Micrograph height in pixels (required) |
| `--output` / `-o` | Output path (default: `<input_stem>_yflip.star`) |
| `--zero-indexed` | Use `y' = height - 1 - y` instead of `height - y` if your coords are 0-based |

Example with a project STAR file and height 4096:

```bash
python flip_star_y.py -i 10028.star -H 4096 -o 10028_mrc.star
```

---

## 4. Data layout (for training)

Training data is organized **per EMPIAR dataset**. Two layouts are supported,
and the loader **prefers your pre-split directories** if available.

### Layout A — Pre-split (recommended)

If you already have a fixed train/val split, put each split in its own
sub-folder. The loader uses your split as-is, with **no random
re-splitting**.

```
<train_dataset_path>/
├── <EMPIAR_ID_1>/
│   ├── train/
│   │   ├── images/                     <base>.png
│   │   ├── masks/                      <base>_mask.png
│   │   ├── particle_coordinates/       <base>.csv   (x, y, radius)
│   │   └── false_positives/            <base>.csv   (optional, same schema)
│   └── val/
│       ├── images/                     <base>.png
│       ├── masks/                      <base>_mask.png
│       ├── particle_coordinates/       <base>.csv
│       └── false_positives/            <base>.csv   (optional)
├── <EMPIAR_ID_2>/
│   ├── train/ ...
│   └── val/   ...
└── ...
```

In this mode `--val_ratio` and `--split_seed` are **ignored**.
You will see the following line on training start:

```
[MauNet] Using pre-split layout: <EMPIAR>/{train,val}/ (val_ratio / split_seed ignored)
```

### Layout B — Train-only (legacy, auto-split at runtime)

If only `train/` exists (no `val/`), the loader falls back to splitting each
EMPIAR's `train/` images into train+val at run time, so every EMPIAR is
represented in both splits.

```
<train_dataset_path>/
├── <EMPIAR_ID_1>/
│   └── train/
│       ├── images/  masks/  particle_coordinates/  [false_positives/]
├── <EMPIAR_ID_2>/
│   └── train/
│       └── ...
└── ...
```

Controlled by `--val_ratio 0.1` and `--split_seed 42`.
You will see the following line on training start:

```
[MauNet] No val/ found; randomly splitting train/ (val_ratio=0.1, seed=42)
```

### Common notes (both layouts)

- Images are 1024×1024 grayscale PNG (will be resized internally if not).
- CSV columns: `x, y, radius` (header row optional, both header and headerless are supported).
- `false_positives/` is optional. When present, pixels inside FP disks
receive a stronger negative-loss weight (`--fp_neg_weight`, default 3.0).
- The pre-split / random-split decision is made **globally** — if **any**
EMPIAR contains a `val/` directory, Layout A is used for the whole
dataset. Make sure your dataset is consistent (either every EMPIAR
pre-split, or none of them).

---

## 5. Training

### Single GPU

```bash
python train.py \
    --train_dataset_path dataset/train_dataset_1024/ \
    --batch_size 4 \
    --num_epochs 200 \
    --learning_rate 1e-4
```

### Multi-GPU (DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    --batch_size 4 \
    --num_workers 2 \
    --pin_memory \
    --learning_rate 2e-4 \
    --num_epochs 300
```

### Warm-start from the pre-trained CryoMauNet weights

If you want to fine-tune our pre-trained model on your own data instead of
training from scratch:

```bash
python train.py \
    --train_dataset_path dataset/train_dataset_1024/ \
    --maunet_checkpoint  MauNet_checkpoint/MauNet_pretrained.pth \
    --output_path        output/models_finetuned \
    --learning_rate 5e-5 \
    --num_epochs 150 \
    --mae_unfreeze_last_n 2 \
    --mae_finetune_lr_scale 0.1
```

### Outputs

- `output/models/maunet_best.pth` — best new checkpoint by validation F1
(created during your own training run).
- `output/models/maunet_epoch_<N>_<date>.pth` — periodic snapshots
(every `--save_every` epochs, default 50).

> Note: `MauNet_checkpoint/MauNet_pretrained.pth` (the shipped weights)
> is **never** overwritten by training; new runs write into `output/models/`.

---

## 6. Key hyperparameters


| Group    | Flag                    | Default | Notes                       |
| -------- | ----------------------- | ------- | --------------------------- |
| Training | `--batch_size`          | 4       | per-GPU                     |
| Training | `--learning_rate`       | 1e-4    | scale up for DDP            |
| Training | `--num_epochs`          | 200     |                             |
| Loss     | `--lambda_heatmap`      | 1.0     | main task                   |
| Loss     | `--lambda_mask`         | 0.3     | aux task; 0 to disable      |
| Loss     | `--centernet_alpha`     | 2.0     | focal `p`-focusing exponent |
| Loss     | `--centernet_beta`      | 4.0     | target decay exponent       |
| Loss     | `--fp_neg_weight`       | 3.0     | FP-disk negative weight     |
| Heatmap  | `--heatmap_sigma_scale` | 1/3     | Gaussian σ = scale × radius |
| Decode   | `--score_threshold`     | 0.3     | inference cutoff            |
| Decode   | `--peak_nms_scale`      | 0.5     | NMS radius = scale × radius |
| MAE      | `--freeze_mae`          | True    | freeze MAE encoder          |
| MAE      | `--mae_unfreeze_last_n` | 0       | unfreeze last N ViT blocks  |


Run `python train.py -h` for the full list.

---

## 7. License

This repository is licensed under the [MIT License](LICENSE): you may use,
modify, and redistribute the **code** subject to that file. Bundled **model
weights** and **third-party code** (e.g. MAE / U-Net building blocks) remain
subject to their original terms where applicable; cite or comply with those
sources when redistributing derived work.