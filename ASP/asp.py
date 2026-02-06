#!/usr/bin/env python3
"""
asp.py

Role:
  Build the Anatomical Saliency Prior (ASP) from a directory of sample images,
  and save ONLY one ASP data pair:
    - outputs_asp_demo/asp_map.npy
    - outputs_asp_demo/asp_map_3d.png   (3D surface visualization)

ASP math:
  mean_scaled = (mean - min) / (max - min + 1e-6)
  std_norm    = std / (std.max + 1e-6)
  asp_map     = (1 - std_norm**alpha) * mean_scaled
  asp_map     = clip(asp_map, 0, 1)

3D PNG export style:
  Matches the user's reference code:
    - Convert tensor-like map to 2D (here directly numpy)
    - Create X/Y meshgrid with flipped Y axis
    - Smooth Z with gaussian_filter(sigma=3)
    - plot_surface with cmap='jet', alpha=0.7, vmin=0.0, vmax=2
    - zlim fixed to [0,2]
    - Insert a grayscale base image on Z=0 plane
    - Add colorbar
    
    
Usage example:
    python asp.py \
        --root_dir "samples_asp" \ \
        --out_dir "outputs_asp_demo" \
        --base_image "./samples_asp/dog_01/dog_01_sample.png" \
        --w 256 --h 256 \
        --alpha 1.0
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# I/O utilities: collect sample images
# -----------------------------------------------------------------------------
def collect_image_paths(root_dir: Path, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    """
    Role:
      Collect images from subfolders under root_dir in deterministic order.

    Expected structure:
      root_dir/
        subject_01/*.png
        subject_02/*.png
        ...
    """
    root_dir = Path(root_dir)
    paths: List[Path] = []
    for sub in sorted(os.listdir(root_dir)):
        sub_path = root_dir / sub
        if not sub_path.is_dir():
            continue
        for ext in exts:
            paths += [Path(p) for p in sorted(glob(str(sub_path / f"*{ext}")))]
    return paths


def read_gray_resize(path: Path, size_wh: Tuple[int, int]) -> np.ndarray:
    """
    Role:
      Read an image as grayscale and resize to (W, H).
      Returns float32 array in [0,255].
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.resize(img, size_wh)  # (W, H)
    return img.astype(np.float32)


# -----------------------------------------------------------------------------
# Core ASP computation (reference-equivalent)
# -----------------------------------------------------------------------------
def compute_anatomical_map(
    root_dir: str | Path,
    image_size: Tuple[int, int] = (256, 256),
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Role:
      Compute ASP (anatomical saliency prior) from cohort images.

    Returns:
      asp_map: float32 in [0,1], shape (H, W)
    """
    root_dir = Path(root_dir)
    img_paths = collect_image_paths(root_dir)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found under: {root_dir}")

    imgs = [read_gray_resize(p, size_wh=image_size) for p in img_paths]
    stack = np.stack(imgs, axis=0)  # (N, H, W)

    mean_map = np.mean(stack, axis=0).astype(np.float32)
    std_map = np.std(stack, axis=0).astype(np.float32)

    mean_scaled = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)
    std_norm = std_map / (std_map.max() + 1e-6)

    asp_map = (1.0 - np.power(std_norm, alpha)) * mean_scaled
    asp_map = np.clip(asp_map, 0.0, 1.0).astype(np.float32)
    return asp_map


# -----------------------------------------------------------------------------
# Export: save ONLY one ASP pair (npy + 3D png)
# -----------------------------------------------------------------------------
def save_asp_npy(asp_map: np.ndarray, out_dir: Path) -> Path:
    """
    Role:
      Save the ASP numeric map as a single .npy file (exact float32).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "asp_map.npy"
    np.save(str(npy_path), asp_map.astype(np.float32))
    return npy_path


def save_asp_3d_png(
    asp_map: np.ndarray,
    out_dir: Path,
    base_image_path: Path,
    sigma: float = 3.0,
    vmin: float = 0.0,
    vmax: float = 2.0,
) -> Path:
    """
    Role:
      Save a 3D-encoded visualization PNG following the user's reference style:
        - Smooth Z with Gaussian filter
        - Surface: jet colormap, alpha=0.7, vmin/vmax fixed
        - Insert a grayscale base image on Z=0 plane
        - Add colorbar

    Important:
      The ASP values are in [0,1], but vmax is set to 2.0 to match the reference.
    """
    # Imports are local to keep minimal runtime dependencies for non-visual use.
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "asp_map_3d.png"

    anatomical_map_2d = asp_map  # already 2D numpy
    h, w = anatomical_map_2d.shape

    # Coordinate system (flip Y to match image coordinates)
    X_full, Y_full = np.meshgrid(np.arange(w), np.arange(h))
    Y_full = h - 1 - Y_full

    # Smooth Z over the full region
    Z_smoothed_full = gaussian_filter(anatomical_map_2d, sigma=sigma)

    # Load base image and resize to match (w, h)
    img = Image.open(base_image_path).convert("L").resize((w, h))
    img_array = np.array(img, dtype=np.float32) / 255.0

    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X_full, Y_full, Z_smoothed_full,
        cmap="jet",
        edgecolor="none",
        alpha=0.7,
        vmin=vmin, vmax=vmax,
    )

    ax.set_zlim(vmin, vmax)

    # Insert base image at Z=0 plane (keep same flipped Y coordinate system)
    ax.plot_surface(
        X_full, Y_full, np.zeros_like(anatomical_map_2d),
        rstride=1, cstride=1,
        facecolors=plt.cm.gray(img_array),
        shade=False,
        zorder=0,
        alpha=1.0,
    )

    ax.set_title("Smoothed 3D Surface with Image in Image Coordinate")
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    ax.set_zlabel("Value")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Save PNG (no interactive show; this is a repository artifact)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return png_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ASP and save only asp_map.npy + asp_map_3d.png.")
    p.add_argument("--root_dir", type=str, required=True, help="Directory containing subfolders of cohort images.")
    p.add_argument("--out_dir", type=str, default="outputs_asp_demo", help="Output dir for asp_map.(npy|png).")
    p.add_argument(
        "--base_image",
        type=str,
        required=True,
        help="A single grayscale image path used as the ground texture at Z=0.",
    )
    p.add_argument("--w", type=int, default=256, help="Resize width.")
    p.add_argument("--h", type=int, default=256, help="Resize height.")
    p.add_argument("--alpha", type=float, default=1.0, help="Std softening exponent.")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing sigma for Z.")
    p.add_argument("--vmin", type=float, default=0.0, help="Surface colormap vmin and z-lim min.")
    p.add_argument("--vmax", type=float, default=2.0, help="Surface colormap vmax and z-lim max.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    base_image = Path(args.base_image)

    asp_map = compute_anatomical_map(
        root_dir=args.root_dir,
        image_size=(args.w, args.h),
        alpha=args.alpha,
    )

    npy_path = save_asp_npy(asp_map, out_dir=out_dir)
    png_path = save_asp_3d_png(
        asp_map,
        out_dir=out_dir,
        base_image_path=base_image,
        sigma=args.sigma,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    print(f"[INFO] Saved: {npy_path}")
    print(f"[INFO] Saved: {png_path}")

    # Optional: show how it is typically consumed in training code (your style)
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        anatomical_map_tensor = (
            torch.tensor(asp_map, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        print(f"[INFO] anatomical_map_tensor shape: {tuple(anatomical_map_tensor.shape)} on {device}")
    except Exception as e:
        print(f"[WARN] Torch tensor conversion skipped: {e}")


if __name__ == "__main__":
    main()
