#!/usr/bin/env python3
"""
spatial_normalization.py

This script consolidates the sequential spatial-normalization steps that were
originally implemented across two notebooks:

  00_denoising and add margin.ipynb
    (1) Intensity-threshold denoising
    (2) Horizontal aspect alignment via symmetric padding/cropping ("add margin")

  01_x_translation.ipynb
    (3) Horizontal x-translation using the thoracic centroid estimated from the
        lower half ROI

Running this file performs the *same ordered processing* as the two notebooks:
denoise -> pad/crop-to-ratio -> x-translation, and writes results to disk.

Notes
-----
- All comments are in English, as requested.
- Paths are handled recursively and directory structure is preserved.
- The "target ratio" follows the notebook logic:
    target_ratio = (max_width over dataset) / (max_height over dataset)
  computed from a user-specified ratio-source directory (or from input_root).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# OpenCV is used for warpAffine to match the notebook implementation style.
import cv2


# -----------------------------------------------------------------------------
# Utility: file discovery
# -----------------------------------------------------------------------------
def iter_image_files(root: Path, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")) -> Iterable[Path]:
    """
    Role:
      Recursively enumerate image files under 'root' with allowed extensions.

    Why it exists:
      The notebooks used os.walk / listdir to traverse folders. This helper
      standardizes traversal while preserving relative paths.
    """
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# -----------------------------------------------------------------------------
# Step 1: Intensity-threshold denoising
# -----------------------------------------------------------------------------
def denoise_intensity_threshold(img_u8: np.ndarray, thr: int = 15) -> np.ndarray:
    """
    Role:
      Implements the notebook's denoising rule:
        - Set pixels with intensity < thr to 0.
      This corresponds to "pixel intensity-based denoising".

    Inputs:
      img_u8: uint8 grayscale image (H, W)
      thr:   intensity threshold (default=15, matching the notebook comment)

    Output:
      uint8 grayscale image (H, W) after thresholding
    """
    if img_u8.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img_u8.dtype}")
    out = img_u8.copy()
    out[out < thr] = 0
    return out


# -----------------------------------------------------------------------------
# Step 2: Horizontal padding/cropping to match a dataset-wide target ratio
# -----------------------------------------------------------------------------
def compute_target_ratio(ratio_source_dir: Path) -> float:
    """
    Role:
      Computes the target aspect ratio used in the "add margin" notebook cell:
        target_ratio = max_width / max_height
      where max_width and max_height are computed over the ratio-source dataset.

    Why it exists:
      The notebook first measured (width, height) across a directory to compute
      a dataset-wide reference ratio, then applied padding/cropping to each image.

    Output:
      target_ratio (float)
    """
    max_w = 0
    max_h = 0

    for img_path in iter_image_files(ratio_source_dir):
        with Image.open(img_path) as im:
            im = im.convert("L")
            w, h = im.size
            max_w = max(max_w, w)
            max_h = max(max_h, h)

    if max_w == 0 or max_h == 0:
        raise RuntimeError(f"No images found in ratio_source_dir: {ratio_source_dir}")

    return max_w / max_h


def pad_or_crop_to_target_ratio(img_pil: Image.Image, target_ratio: float, eps: float = 1e-4) -> Image.Image:
    """
    Role:
      Matches the notebook logic for "add margin" (horizontal adjustment only):
        - Let current_ratio = w / h.
        - Compute target_width = round(h * target_ratio).
        - If current_ratio < target_ratio: symmetric horizontal padding to target_width.
        - If current_ratio > target_ratio: symmetric horizontal crop to target_width.
        - If close enough: keep as-is.

    Inputs:
      img_pil:       PIL Image in grayscale ("L")
      target_ratio:  dataset-wide target ratio (maxW/maxH)
      eps:           tolerance for ratio equality

    Output:
      Processed PIL Image ("L") with adjusted width while preserving height.
    """
    if img_pil.mode != "L":
        img_pil = img_pil.convert("L")

    w, h = img_pil.size
    current_ratio = w / h
    target_width = int(round(h * target_ratio))

    # If already matching ratio within tolerance, do nothing.
    if abs(current_ratio - target_ratio) < eps:
        return img_pil

    # If image is narrower than target ratio, pad left/right.
    if current_ratio < target_ratio:
        pad_total = max(target_width - w, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return ImageOps.expand(img_pil, border=(pad_left, 0, pad_right, 0), fill=0)

    # If image is wider than target ratio, crop left/right.
    crop_total = max(w - target_width, 0)
    crop_left = crop_total // 2
    crop_right = crop_total - crop_left
    # PIL crop box is (left, upper, right, lower)
    return img_pil.crop((crop_left, 0, w - crop_right, h))


# -----------------------------------------------------------------------------
# Step 3: X-translation via centroid estimation from lower half ROI
# -----------------------------------------------------------------------------
def compute_centroid_x_lower_half(img_norm: np.ndarray, bin_thr: float = 50 / 255.0) -> Optional[float]:
    """
    Role:
      Computes the x-centroid from the *lower half* of the image using a binary mask,
      matching the notebook's description:
        - Use only pixels with brightness >= 50 (on uint8 scale).
        - Compute centroid along x using image moments.

    Inputs:
      img_norm: float32 image in [0, 1], shape (H, W)
      bin_thr:  threshold in [0, 1] corresponding to 50/255

    Output:
      centroid_x (float) in pixel coordinates of the full image frame, or None if
      foreground mass is empty.
    """
    h, w = img_norm.shape
    lower = img_norm[h // 2 :, :]  # lower half
    mask = (lower >= bin_thr)

    # If nothing passes threshold, translation cannot be estimated.
    m00 = float(mask.sum())
    if m00 <= 0:
        return None

    # Compute m10 over x coordinates within the lower-half mask.
    # x coordinate is 0..W-1 (same width as full image).
    xs = np.arange(w, dtype=np.float32)
    m10 = float((mask.astype(np.float32) * xs[None, :]).sum())

    cx = m10 / m00
    return cx


def translate_x(img_norm: np.ndarray, dx: float) -> np.ndarray:
    """
    Role:
      Applies x-axis translation using OpenCV warpAffine, matching notebook style.
      Border is filled with 0 (black), consistent with padding behavior.

    Inputs:
      img_norm: float32 image in [0, 1], shape (H, W)
      dx:       translation along x (pixels). Positive dx shifts content right.

    Output:
      translated float32 image in [0, 1], shape (H, W)
    """
    h, w = img_norm.shape
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = cv2.warpAffine(
        img_norm,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return out


def apply_x_alignment(img_u8: np.ndarray, bin_thr_u8: int = 50) -> np.ndarray:
    """
    Role:
      End-to-end x-alignment stage:
        - Normalize to [0, 1]
        - Estimate centroid_x on lower half ROI
        - Compute dx = (W/2 - centroid_x)
        - Translate image by dx
        - Return uint8 output

    This matches the notebook's sequence: normalize -> compute transform -> apply -> save.

    Inputs:
      img_u8:      uint8 grayscale image
      bin_thr_u8:  threshold on uint8 scale (default=50)

    Output:
      uint8 grayscale image after x-alignment
    """
    img_norm = img_u8.astype(np.float32) / 255.0
    cx = compute_centroid_x_lower_half(img_norm, bin_thr=float(bin_thr_u8) / 255.0)

    # If centroid cannot be estimated, keep the image unchanged (safe fallback).
    if cx is None:
        return img_u8

    h, w = img_norm.shape
    target_cx = w / 2.0
    dx = target_cx - cx

    aligned = translate_x(img_norm, dx)
    aligned_u8 = np.clip(aligned * 255.0, 0, 255).astype(np.uint8)
    return aligned_u8


# -----------------------------------------------------------------------------
# Full pipeline: denoise -> pad/crop-to-ratio -> x-translation
# -----------------------------------------------------------------------------
def spatial_normalize_image(
    img_u8: np.ndarray,
    denoise_thr_u8: int,
    target_ratio: float,
    centroid_bin_thr_u8: int,
) -> np.ndarray:
    """
    Role:
      Runs the exact notebook-ordered spatial normalization steps in-memory:

        (1) denoise by intensity threshold
        (2) horizontal padding/cropping to match target ratio
        (3) x-translation alignment using lower-half centroid

    Output:
      uint8 grayscale image after full spatial normalization
    """
    # Step 1: denoise
    img_u8 = denoise_intensity_threshold(img_u8, thr=denoise_thr_u8)

    # Step 2: pad/crop to target ratio (PIL-based for exact padding/crop semantics)
    pil = Image.fromarray(img_u8, mode="L")
    pil = pad_or_crop_to_target_ratio(pil, target_ratio=target_ratio)
    img_u8 = np.array(pil, dtype=np.uint8)

    # Step 3: x-alignment (OpenCV warpAffine)
    img_u8 = apply_x_alignment(img_u8, bin_thr_u8=centroid_bin_thr_u8)

    return img_u8


def process_folder(
    input_root: Path,
    output_root: Path,
    target_ratio: float,
    denoise_thr_u8: int = 15,
    centroid_bin_thr_u8: int = 50,
) -> None:
    """
    Role:
      Batch-processes all images under input_root, preserving directory structure
      under output_root, and writing spatial-normalized outputs.

    This corresponds to the notebooks' os.walk style but without intermediate folders.
    The effective transformation is identical because operations are purely functional
    and applied in the same order.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for in_path in iter_image_files(input_root):
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load as grayscale uint8
        with Image.open(in_path) as im:
            im = im.convert("L")
            img_u8 = np.array(im, dtype=np.uint8)

        out_u8 = spatial_normalize_image(
            img_u8=img_u8,
            denoise_thr_u8=denoise_thr_u8,
            target_ratio=target_ratio,
            centroid_bin_thr_u8=centroid_bin_thr_u8,
        )

        # Save
        Image.fromarray(out_u8, mode="L").save(out_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """
    Role:
      Command-line interface to reproduce the notebook pipeline reliably.
    """
    p = argparse.ArgumentParser(description="Spatial normalization (denoise -> margin -> x-translation)")
    p.add_argument("--input_root", type=str, required=True, help="Input root directory containing images.")
    p.add_argument("--output_root", type=str, required=True, help="Output root directory to write processed images.")
    p.add_argument(
        "--ratio_source",
        type=str,
        default=None,
        help=(
            "Directory used to compute target_ratio = maxW/maxH. "
            "If omitted, input_root is used."
        ),
    )
    p.add_argument("--denoise_thr", type=int, default=15, help="Denoising threshold on uint8 scale (default=15).")
    p.add_argument(
        "--centroid_bin_thr",
        type=int,
        default=50,
        help="Binary threshold for centroid estimation on uint8 scale (default=50).",
    )
    return p


def main() -> None:
    """
    Role:
      Entry point. Computes target_ratio, then runs full pipeline over the folder.
    """
    args = build_argparser().parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ratio_source = Path(args.ratio_source) if args.ratio_source else input_root

    target_ratio = compute_target_ratio(ratio_source)
    print(f"[INFO] target_ratio = {target_ratio:.12f}  (computed from: {ratio_source})")

    process_folder(
        input_root=input_root,
        output_root=output_root,
        target_ratio=target_ratio,
        denoise_thr_u8=args.denoise_thr,
        centroid_bin_thr_u8=args.centroid_bin_thr,
    )

    print("[INFO] Done. Spatial-normalized images have been written.")


if __name__ == "__main__":
    main()
