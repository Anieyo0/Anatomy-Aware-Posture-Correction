"""
Progressive Anatomical Emphasis (PAE) schedule.

This module provides a smooth schedule that increases the anatomy-weight term
from 0 to 1 as training progresses. The schedule is implemented using a
shifted and scaled hyperbolic tangent (tanh) curve.

Usage
-----
1) Import in training code:
    from pae import smooth_pae_weights
    w_mse, w_anatomy = smooth_pae_weights(iteration, total_iterations, transition_point, sharpness)

2) Run as a script to generate demo outputs:
    python pae.py

Running this file will generate:
    outputs_pae_demo/pae_weights.csv
    outputs_pae_demo/progressive_anatomical_emphasis.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PAEConfig:
    """Configuration for the PAE schedule."""
    total_iterations: int = 1000
    transition_point: float = 0.3
    sharpness: float = 10.0
    output_dir: str = "outputs_pae_demo"


def smooth_pae_weights(
    iteration: int,
    total_iterations: int,
    transition_point: float = 0.3,
    sharpness: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute the smooth PAE weights at a given iteration.

    Parameters
    ----------
    iteration : int
        Current training iteration (0 <= iteration <= total_iterations).
    total_iterations : int
        Total number of iterations used to normalize the progress.
    transition_point : float
        Normalized point in [0, 1] where the schedule starts to increase more sharply.
        Example: 0.3 means the transition begins around 30% of the training.
    sharpness : float
        Controls the steepness of the transition. Larger values yield a sharper rise.

    Returns
    -------
    w_mse : float
        Weight for the baseline MSE term (kept at 1.0 by design in this schedule).
    w_anatomy : float
        Weight for the anatomy-weighted term, smoothly increasing from 0 to 1.
    """
    if total_iterations <= 0:
        raise ValueError("total_iterations must be a positive integer.")

    # Normalize progress to t in [0, 1]
    t = float(iteration) / float(total_iterations)

    # Smooth transition using tanh: maps (-inf, +inf) -> (-1, +1), then shifted to (0, 1)
    w_anatomy = 0.5 * (np.tanh((t - transition_point) * sharpness) + 1.0)

    # The baseline similarity weight can remain constant
    w_mse = 1.0
    return w_mse, float(w_anatomy)


def _ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _save_csv(iterations: np.ndarray, w_mse: np.ndarray, w_anatomy: np.ndarray, out_path: str) -> None:
    """Save schedule values to a CSV file."""
    header = "iteration,w_mse,w_anatomy"
    data = np.stack([iterations, w_mse, w_anatomy], axis=1)
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt=["%d", "%.8f", "%.8f"])


def _plot_and_save_curve(
    iterations: np.ndarray,
    w_anatomy: np.ndarray,
    transition_iter: int,
    out_path: str,
) -> None:
    """Plot the anatomy weight curve and save it as an image."""
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, w_anatomy, linewidth=2, color='blue', label=r"$\lambda_{anatomy}$ (Progressive Anatomical Emphasis)")
    plt.axvline(x=transition_iter, color='red',linestyle="--", linewidth=2, label="Transition Start")

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(r"Anatomical Weight ($\lambda_{anatomy}$)", fontsize=12)
    plt.title("Progressive Anatomical Emphasis Schedule", fontsize=14)
    plt.ylim(0.0, 1.05)
    plt.xlim(int(iterations.min()), int(iterations.max()))

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(config: PAEConfig) -> None:
    """
    Generate a demo schedule and save outputs under config.output_dir.

    Outputs
    -------
    - outputs_pae_demo/pae_weights.csv
    - outputs_pae_demo/progressive_anatomical_emphasis.png
    """
    _ensure_dir(config.output_dir)

    iterations = np.arange(0, config.total_iterations + 1, dtype=int)

    # Compute weights over the entire iteration range
    w_mse_values = np.zeros_like(iterations, dtype=np.float64)
    w_anatomy_values = np.zeros_like(iterations, dtype=np.float64)

    for idx, it in enumerate(iterations):
        w_mse, w_anatomy = smooth_pae_weights(
            iteration=int(it),
            total_iterations=config.total_iterations,
            transition_point=config.transition_point,
            sharpness=config.sharpness,
        )
        w_mse_values[idx] = w_mse
        w_anatomy_values[idx] = w_anatomy

    transition_iter = int(round(config.total_iterations * config.transition_point))

    csv_path = os.path.join(config.output_dir, "pae_weights.csv")
    fig_path = os.path.join(config.output_dir, "progressive_anatomical_emphasis.png")

    _save_csv(iterations, w_mse_values, w_anatomy_values, csv_path)
    _plot_and_save_curve(iterations, w_anatomy_values, transition_iter, fig_path)

    print(f"Saved schedule table: {csv_path}")
    print(f"Saved schedule figure: {fig_path}")


if __name__ == "__main__":
    cfg = PAEConfig(
        total_iterations=1000,
        transition_point=0.3,
        sharpness=10.0,
        output_dir="outputs_pae_demo",
    )
    main(cfg)
