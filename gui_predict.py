"""
MauNet Particle Picker — Gradio web UI

Run from the project root with the ``maunet`` conda environment active::

    conda activate maunet
    python gui_predict.py

Open the printed URL in a browser (default http://127.0.0.1:7860).

Upload: drag-and-drop PNG/JPG/TIF/MRC/MRCS, or click to select; multiple files
are supported. After inference, download CSV coordinates and a RELION-style
coordinate STAR file.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

# Avoid matplotlib writing to a read-only cwd
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "mplconfig"),
)

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import gradio as gr

# Import from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.mae_encoder import build_mae_feature_extractor
from models.maunet import MauNet
from utils.accuracy import decode_heatmap
from denoise import denoise as _denoise_mrc, denoise_jpg_image as _denoise_img
import mrcfile as _mrcfile

_MRC_EXTS = {".mrc", ".mrcs"}
_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mrc", ".mrcs"}

# View mode strings (must match Radio choices)
_VIEW_OVERLAY = "Particle overlay"
_VIEW_HEATMAP = "Heatmap"


def _gradio_builtin_english_ui() -> gr.I18n:
    """Force English for Gradio stock File-upload strings on localized browsers.

    Gradio translates built-in UI (e.g. drag-and-drop hints) from the browser
    ``Accept-Language``. This app is English-only; we overlay the same English
    strings onto common Asian locales so the File widget matches our labels.
    """
    upload = {
        "click_to_upload": "Click to upload",
        "drop_audio": "Drop audio here",
        "drop_csv": "Drop CSV here",
        "drop_file": "Drop file here",
        "drop_image": "Drop image here",
        "drop_video": "Drop video here",
        "drop_gallery": "Drop media here",
        "paste_clipboard": "Paste from clipboard",
    }
    common = {"or": "- or -"}
    bundle = {"upload": upload, "common": common}
    return gr.I18n(**{"zh-CN": bundle, "zh-TW": bundle, "ja": bundle})


def _maunet_picker_theme() -> gr.themes.Soft:
    """Typography tuned for clear, neutral English UI (replaces rounded Montserrat)."""
    from gradio.themes import GoogleFont, sizes
    from gradio.themes.utils import fonts as theme_fonts

    return gr.themes.Soft(
        font=(
            GoogleFont("Source Sans 3", weights=(400, 600, 700)),
            "Segoe UI",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "Noto Sans",
            "Liberation Sans",
            "DejaVu Sans",
            "sans-serif",
        ),
        font_mono=(
            theme_fonts.LocalFont("IBM Plex Mono"),
            "ui-monospace",
            "Cascadia Mono",
            "Consolas",
            "monospace",
        ),
        text_size=sizes.text_lg,
    )


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Linearly scale a 2D array of any numeric dtype to uint8."""
    d = data.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        return ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    return np.zeros_like(d, dtype=np.uint8)


def _mrc_orient(frame: np.ndarray) -> np.ndarray:
    """Apply orientation fix (.T + rot90) to a single MRC frame."""
    return np.rot90(frame.T)


def _read_mrc_raw(path: str) -> np.ndarray:
    """Load a single-frame MRC as uint8 grayscale; no denoising."""
    data = _mrcfile.read(path)
    if data.ndim == 3:
        data = data[0]
    return _normalize_to_uint8(_mrc_orient(data))


def _collect_paths_from_str(path_str: str) -> list[str]:
    """Collect supported file paths from a single file or a directory path."""
    path_str = path_str.strip()
    if not path_str:
        return []
    if os.path.isfile(path_str):
        if os.path.splitext(path_str)[1].lower() in _SUPPORTED_EXTS:
            return [path_str]
        return []
    if os.path.isdir(path_str):
        return sorted(
            os.path.join(path_str, name)
            for name in os.listdir(path_str)
            if os.path.splitext(name)[1].lower() in _SUPPORTED_EXTS
        )
    return []


# -----------------------------------------------------------------------------
# Global model cache (avoid reloading weights on every run)
# -----------------------------------------------------------------------------
_cache: dict = {"ckpt_path": None, "device_str": None, "model": None, "device": None}


def _get_model(ckpt_path: str, device_str: str):
    """Load model; return cached instance if ckpt and device match."""
    if _cache["ckpt_path"] == ckpt_path and _cache["device_str"] == device_str:
        return _cache["model"], _cache["device"]

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if device_str == "auto":
        device_str_real = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str_real = device_str
    if device_str_real.startswith("cuda") and not torch.cuda.is_available():
        device_str_real = "cpu"
    device = torch.device(device_str_real)

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
    model.load_state_dict(state, strict=False)
    model.eval()
    model.mae_extractor.eval()

    _cache.update(ckpt_path=ckpt_path, device_str=device_str,
                  model=model, device=device)
    return model, device


# -----------------------------------------------------------------------------
# Image preprocessing: letterbox (aspect-preserving resize + padding)
# -----------------------------------------------------------------------------
def _letterbox_preprocess(img: np.ndarray):
    """Letterbox uint8 grayscale to model input size and normalize.

    Returns
    -------
    t : torch.Tensor
        Normalized input (1, 1, H_in, W_in).
    (H_orig, W_orig) : tuple[int, int]
        Original spatial size.
    canvas : np.ndarray
        Letterboxed uint8 image for visualization.
    scale : float
        Isotropic scale factor.
    pad_x, pad_y : float
        Letterbox padding offsets for inverse mapping.
    """
    H_orig, W_orig = img.shape[:2]
    H_in = config.input_image_height
    W_in = config.input_image_width

    scale = min(W_in / W_orig, H_in / H_orig)
    new_w = int(round(W_orig * scale))
    new_h = int(round(H_orig * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((H_in, W_in), dtype=np.uint8)
    pad_y_i = (H_in - new_h) // 2
    pad_x_i = (W_in - new_w) // 2
    canvas[pad_y_i:pad_y_i + new_h, pad_x_i:pad_x_i + new_w] = img_scaled

    pad_x_f = (W_in - W_orig * scale) / 2.0
    pad_y_f = (H_in - H_orig * scale) / 2.0

    img_f = canvas.astype(np.float32) / 255.0
    if config.normalize_image:
        mean, std = img_f.mean(), img_f.std()
        if std >= 1e-6:
            img_f = np.clip((img_f - mean) / std, -5.0, 5.0)
        else:
            img_f = img_f - mean
    t = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0).float()
    return t, (H_orig, W_orig), canvas, scale, pad_x_f, pad_y_f


def _load_micrograph(path: str, apply_denoise: bool = False):
    """Load one image (PNG/JPG/TIF/MRC) and letterbox-preprocess.

    For MRCS stacks use ``_iter_mrcs_frames`` instead.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _MRC_EXTS:
        if apply_denoise:
            img = _denoise_mrc(path)
        else:
            img = _read_mrc_raw(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        if apply_denoise:
            img = _denoise_img(img)
    return _letterbox_preprocess(img)


def _iter_mrcs_frames(path: str, apply_denoise: bool):
    """Yield (frame_label, letterbox bundle) for each frame in an MRCS stack.

    For 2D data (single frame), behaves like a normal MRC; label is the file name.
    For 3D stacks (N x H x W), labels are ``filename.mrcs:0001``, ...
    Each frame uses the same orientation and uint8 normalization as ``_read_mrc_raw``.
    """
    fname = os.path.basename(path)
    with _mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = mrc.data.copy()  # read into memory before close

    if data.ndim == 2:
        img = _normalize_to_uint8(_mrc_orient(data))
        if apply_denoise:
            img = _denoise_img(img)
        yield fname, _letterbox_preprocess(img)
    elif data.ndim == 3:
        n_frames = data.shape[0]
        for i in range(n_frames):
            img = _normalize_to_uint8(_mrc_orient(data[i]))
            if apply_denoise:
                img = _denoise_img(img)
            label = f"{fname}:{i + 1:04d}"
            yield label, _letterbox_preprocess(img)
    else:
        raise ValueError(f"Unsupported MRC dimensionality: {data.ndim}D (path: {path})")


# -----------------------------------------------------------------------------
# Visualization: circles on grayscale
# -----------------------------------------------------------------------------
def _draw_particles(gray_img: np.ndarray,
                    coords_xy: np.ndarray,
                    scores: np.ndarray,
                    radius: float) -> np.ndarray:
    """Return RGB uint8; particles drawn with small circles (plasma, bright = high score)."""
    rgb = cv2.cvtColor(
        np.clip(gray_img, 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB,
    )
    if coords_xy.size == 0:
        return rgb

    norm_scores = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-6)
    cmap = cm.get_cmap("plasma")
    r_px = max(int(round(radius)), 3)

    for (x, y), ns in zip(coords_xy, norm_scores):
        cx, cy = int(round(x)), int(round(y))
        color_rgba = cmap(float(ns))
        color_bgr = (
            int(color_rgba[2] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[0] * 255),
        )
        cv2.circle(rgb, (cx, cy), r_px, color_bgr, 1, lineType=cv2.LINE_AA)
        cv2.circle(rgb, (cx, cy), 2, color_bgr, -1, lineType=cv2.LINE_AA)

    return rgb


def _write_relion_autopick_star(path: str, rows: list[tuple[str, float, float, float]]) -> None:
    """Write RELION-style autopick STAR (data_ + loop_ + four columns)."""
    lines = [
        "data_",
        "",
        "loop_",
        "_rlnMicrographName #1",
        "_rlnCoordinateX #2",
        "_rlnCoordinateY #3",
        "_rlnAutopickFigureOfMerit #4",
    ]
    for fname, x, y, fom in rows:
        xi, yi = int(round(x)), int(round(y))
        fom_c = float(np.clip(fom, 0.0, 1.0))
        lines.append(f"{fname} {xi} {yi} {fom_c:.6g}")
    lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def _make_heatmap_image(heatmap_np: np.ndarray) -> np.ndarray:
    """Convert [0,1] heatmap to colored RGB (inferno colormap)."""
    hm_u8 = np.clip(heatmap_np * 255, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Image navigation helpers
# -----------------------------------------------------------------------------
_EMPTY_STATE: dict = {"overlays": [], "heatmaps": [], "idx": 0}


def _get_display(state: dict, view_mode: str):
    """Return current frame image and counter text from ``state``."""
    images = state.get("heatmaps" if view_mode == _VIEW_HEATMAP else "overlays", [])
    if not images:
        return None, "— / —"
    idx = state.get("idx", 0)
    img, fname = images[idx]
    return img, f"{idx + 1} / {len(images)}  ·  {fname}"


def go_prev(state: dict, view_mode: str):
    if not state.get("overlays"):
        return None, "— / —", state
    state = dict(state)
    state["idx"] = (state["idx"] - 1) % len(state["overlays"])
    img, counter = _get_display(state, view_mode)
    return img, counter, state


def go_next(state: dict, view_mode: str):
    if not state.get("overlays"):
        return None, "— / —", state
    state = dict(state)
    state["idx"] = (state["idx"] + 1) % len(state["overlays"])
    img, counter = _get_display(state, view_mode)
    return img, counter, state


def switch_view(state: dict, view_mode: str):
    img, counter = _get_display(state, view_mode)
    return img, counter


# -----------------------------------------------------------------------------
# Main inference (Gradio callback)
# -----------------------------------------------------------------------------
def run_predict(
    image_files,
    path_input: str,
    ckpt_path: str,
    device_choice: str,
    boxsize: float,
    score_threshold: float,
    nms_scale: float,
    max_predictions: int,
    show_heatmap: bool,
    filter_padding: bool,
    apply_denoise: bool,
    star_use_basename: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Run picking; returns (state, image, counter, stats markdown, csv path, star path).

    Two input sources (merged, de-duplicated, order preserved):
      1. Upload widget (``image_files``)
      2. Server path textbox (``path_input``) — file or directory
    """
    def _fail(msg):
        return _EMPTY_STATE, None, "— / —", msg, None, None

    upload_paths = [
        (f if isinstance(f, str) else f.name)
        for f in (image_files or [])
    ]
    str_paths = _collect_paths_from_str(path_input or "")
    seen: set[str] = set()
    all_paths: list[str] = []
    for p in upload_paths + str_paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            all_paths.append(p)

    if not all_paths:
        return _fail("Please upload images or enter a file / directory path.")
    if not ckpt_path or not ckpt_path.strip():
        return _fail("Please enter the checkpoint path.")

    progress(0, desc="Loading checkpoint...")
    try:
        model, device = _get_model(ckpt_path.strip(), device_choice)
    except Exception as e:
        return _fail(f"Failed to load model: {e}")

    overlay_images: list = []
    heatmap_images: list = []
    csv_rows: list[list] = [["filename", "x", "y", "score", "diameter"]]
    star_rows: list[tuple[str, float, float, float]] = []
    stats_lines: list[str] = []

    for i, path in enumerate(all_paths):
        fname = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        progress((i + 1) / len(all_paths), desc=f"Inference {fname} ({i+1}/{len(all_paths)})")

        if ext == ".mrcs":
            try:
                frames = list(_iter_mrcs_frames(path, apply_denoise))
            except Exception as e:
                stats_lines.append(f"{fname}: read failed — {e}")
                continue
        else:
            try:
                frames = [(fname, _load_micrograph(path, apply_denoise=apply_denoise))]
            except Exception as e:
                stats_lines.append(f"{fname}: read failed — {e}")
                continue

        file_particle_count = 0
        for frame_label, (t, (H_orig, W_orig), gray_r, lb_scale, lb_pad_x, lb_pad_y) in frames:
            radius_1024 = (boxsize / 2.0) * lb_scale
            nms_radius_px = max(int(round(nms_scale * radius_1024)), 1)

            x_in = t.to(device)
            with torch.no_grad():
                outputs = model(x_in)
            heatmap_prob = torch.sigmoid(outputs["heatmap"])

            decoded = decode_heatmap(
                heatmap_prob,
                score_threshold=score_threshold,
                nms_radius=nms_radius_px,
                max_predictions=max_predictions,
            )
            coords, scores = decoded[0]
            coords_np = coords.cpu().numpy() if coords.numel() else np.zeros((0, 2), np.float32)
            scores_np = scores.cpu().numpy() if scores.numel() else np.zeros((0,), np.float32)

            if coords_np.size:
                x_orig = (coords_np[:, 0] - lb_pad_x) / lb_scale
                y_orig = (coords_np[:, 1] - lb_pad_y) / lb_scale
                coords_orig = np.stack([x_orig, y_orig], axis=1)
                if filter_padding:
                    mask = (
                        (x_orig >= 0) & (x_orig <= W_orig) &
                        (y_orig >= 0) & (y_orig <= H_orig)
                    )
                    coords_orig = coords_orig[mask]
                    scores_orig = scores_np[mask]
                else:
                    scores_orig = scores_np
            else:
                coords_orig = coords_np
                scores_orig = scores_np

            overlay = _draw_particles(gray_r, coords_np, scores_np, radius_1024)
            overlay_images.append((overlay, frame_label))

            if show_heatmap:
                hm_np = heatmap_prob[0, 0].cpu().numpy()
                heatmap_images.append((_make_heatmap_image(hm_np), f"{frame_label} heatmap"))

            if star_use_basename:
                mg_name = frame_label
            else:
                mg_name = os.path.abspath(path)
            for (x, y), s in zip(coords_orig, scores_orig):
                csv_rows.append([frame_label, f"{x:.2f}", f"{y:.2f}",
                                  f"{float(s):.4f}", f"{boxsize:.2f}"])
                star_rows.append((mg_name, float(x), float(y), float(s)))

            file_particle_count += len(coords_orig)

        n_frames = len(frames)
        denoise_tag = "denoised" if apply_denoise else "raw"
        frame_info = f", {n_frames} frames" if n_frames > 1 else ""
        stats_lines.append(
            f"**{fname}**{frame_info}: particles {file_particle_count}  |  "
            f"boxsize {boxsize:.0f} px  |  {denoise_tag}"
        )

    csv_path = None
    if len(csv_rows) > 1:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        csv.writer(tmp).writerows(csv_rows)
        tmp.close()
        csv_path = tmp.name

    star_path = None
    if star_rows:
        st = tempfile.NamedTemporaryFile(mode="w", suffix=".star", delete=False, newline="")
        st.close()
        _write_relion_autopick_star(st.name, star_rows)
        star_path = st.name

    total = len(csv_rows) - 1
    stats_text = (
        f"Processed {len(all_paths)} image(s); total particles: {total}\n\n"
        + "\n\n".join(stats_lines)
    )

    state = {"overlays": overlay_images, "heatmaps": heatmap_images, "idx": 0}
    first_img, counter = _get_display(state, _VIEW_OVERLAY)
    return state, first_img, counter, stats_text, csv_path, star_path


# -----------------------------------------------------------------------------
# Build Gradio UI
# -----------------------------------------------------------------------------
def build_ui():
    default_ckpt = "MauNet_checkpoint/MauNet_pretrained.pth"

    _css = """
    html, body { height: 100%; overflow: hidden; }
    .gradio-container {
        height: 100vh !important; max-width: 100% !important;
        padding: 8px !important; box-sizing: border-box;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    #main-row   { height: calc(100vh - 58px); overflow: hidden; }
    #left-panel { overflow-y: auto; height: 100%; padding-right: 4px; }
    /* Right: flex column — fixed toolbar, image fills remainder */
    #right-panel { height: 100%; display: flex; flex-direction: column;
                   gap: 4px; overflow: hidden; }
    #toolbar-row { flex: 0 0 auto; }
    #main-image  { flex: 1 1 0; min-height: 0; overflow: hidden; }
    #main-image > div { height: 100% !important; }
    #main-image img {
        height: 100% !important;
        max-height: 100% !important;
        width: 100% !important;
        object-fit: contain !important;
    }
    /* File widget: avoid clipping (do not use a tiny fixed height on gr.File) */
    #micrograph-upload {
        min-height: 168px !important;
    }
    """

    with gr.Blocks(title="MauNet Particle Picker", css=_css) as demo:
        img_state = gr.State(dict(_EMPTY_STATE))

        gr.Markdown("# MauNet Particle Picker &nbsp;|&nbsp; Cryo-EM particle autopicking")

        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, min_width=300, elem_id="left-panel"):
                gr.Markdown("#### Model")
                ckpt_input = gr.Textbox(
                    label="Checkpoint path (.pth)",
                    value=default_ckpt,
                    placeholder="MauNet_checkpoint/MauNet_pretrained.pth",
                )
                device_input = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda:0", "cuda:1", "cpu"],
                    value="auto",
                )

                gr.Markdown("#### Input")
                image_upload = gr.File(
                    label="Option A: upload (PNG/JPG/TIF/MRC/MRCS, multiple)",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff",
                                ".mrc", ".mrcs"],
                    elem_id="micrograph-upload",
                )
                path_input = gr.Textbox(
                    label="Option B: server file or directory path",
                    placeholder="/data/images/063.mrc  or  /data/images/",
                    lines=1,
                )
                with gr.Accordion("Browse server filesystem", open=False):
                    gr.Markdown(
                        "<small>Select a file or folder below, then click "
                        "<strong>Confirm path</strong> to fill the text box above.</small>",
                    )
                    file_explorer = gr.FileExplorer(
                        glob="**/*",
                        root_dir="/",
                        file_count="single",
                        label="",
                        show_label=False,
                        height=260,
                    )
                    confirm_path_btn = gr.Button("Confirm path", size="sm")

                gr.Markdown("#### Inference")
                boxsize_input = gr.Number(
                    label="Particle boxsize (diameter in original pixels)",
                    value=200,
                    minimum=2,
                    precision=0,
                )
                threshold_input = gr.Slider(
                    label="Score threshold (score_threshold)",
                    minimum=0.01, maximum=0.99, step=0.01, value=0.10,
                )
                nms_scale_input = gr.Slider(
                    label="NMS scale (nms_scale)",
                    minimum=0.1, maximum=2.0, step=0.05, value=0.5,
                )
                max_pred_input = gr.Number(
                    label="Max picks per image (max_predictions)",
                    value=2000, minimum=1, maximum=10000, precision=0,
                )

                gr.Markdown("#### Output")
                show_heatmap_cb = gr.Checkbox(label="Compute heatmap", value=True)
                filter_padding_cb = gr.Checkbox(
                    label="Filter false peaks in letterbox padding (recommended)",
                    value=True,
                )
                denoise_cb = gr.Checkbox(
                    label="Enable denoising (all formats, including MRC)",
                    value=False,
                )
                star_basename_cb = gr.Checkbox(
                    label="STAR: micrograph name = basename; unchecked = absolute path",
                    value=True,
                )

                run_btn = gr.Button("Run inference", variant="primary", size="lg")

                gr.Markdown("---")
                stats_out = gr.Markdown(value="*Statistics appear here after inference.*")

            with gr.Column(scale=3, elem_id="right-panel"):
                with gr.Row(elem_id="toolbar-row"):
                    view_mode = gr.Radio(
                        choices=[_VIEW_OVERLAY, _VIEW_HEATMAP],
                        value=_VIEW_OVERLAY,
                        label="View",
                        scale=2,
                    )
                    prev_btn = gr.Button("◀ Previous", size="sm", scale=1)
                    next_btn = gr.Button("Next ▶", size="sm", scale=1)
                    with gr.Column(scale=3, min_width=0):
                        counter_md = gr.Markdown(
                            value="Upload images, then click **Run inference**",
                        )

                with gr.Row(elem_id="downloads-row"):
                    csv_out = gr.File(
                        label="Download coordinates (CSV)",
                        interactive=False,
                        scale=1,
                    )
                    star_out = gr.File(
                        label="Download RELION coordinates (STAR)",
                        interactive=False,
                        scale=1,
                    )

                img_display = gr.Image(
                    label="",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_id="main-image",
                )

        def _confirm_explorer_path(selected):
            if not selected:
                return ""
            p = selected if isinstance(selected, str) else selected[0]
            return p

        confirm_path_btn.click(
            fn=_confirm_explorer_path,
            inputs=[file_explorer],
            outputs=[path_input],
        )

        run_btn.click(
            fn=run_predict,
            inputs=[
                image_upload, path_input, ckpt_input, device_input,
                boxsize_input, threshold_input, nms_scale_input,
                max_pred_input, show_heatmap_cb, filter_padding_cb, denoise_cb,
                star_basename_cb,
            ],
            outputs=[img_state, img_display, counter_md, stats_out, csv_out, star_out],
        )
        prev_btn.click(
            fn=go_prev,
            inputs=[img_state, view_mode],
            outputs=[img_display, counter_md, img_state],
        )
        next_btn.click(
            fn=go_next,
            inputs=[img_state, view_mode],
            outputs=[img_display, counter_md, img_state],
        )
        view_mode.change(
            fn=switch_view,
            inputs=[img_state, view_mode],
            outputs=[img_display, counter_md],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",  # allow LAN / remote access
        server_port=7860,
        share=False,            # set True for a temporary public URL
        inbrowser=False,        # False when running on a headless server
        theme=_maunet_picker_theme(),
        i18n=_gradio_builtin_english_ui(),
        head='<script>document.documentElement.lang="en";</script>',
    )
