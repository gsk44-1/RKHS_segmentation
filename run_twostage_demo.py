"""Two-stage pipeline demo: Stage 1 (model fit) -> Stage 2 (segmentation).

Synthesizes a noisy disk image, runs fit_rkhs_decomposition to get
M = Kd + Psi*beta and g(Psi*beta), then feeds those into
rkhs_segment.segment to obtain the segmentation u.

Produces a diagnostic figure:
    input | Kd | Psi*beta | g(Psi*beta) | M | u > 0.5
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rkhs_modelfit import fit_rkhs_decomposition
from rkhs_segment import segment, default_config as seg_default_config


def synth(H=64, W=64, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    img = 0.15 + 0.7 * (r < min(H, W) / 4).astype(float)
    return np.clip(img + noise * rng.standard_normal(img.shape), 0., 1.)


def make_mask(H, W):
    """Centred rectangle initial mask."""
    mask = np.zeros((H, W))
    mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    return mask


def main():
    image = synth()
    H, W = image.shape
    mask_init = make_mask(H, W)

    iota = 1e3  # must match Stage 1's iota_edgegate

    # --- Stage 1: decomposition ---
    print("=== Stage 1: RKHS decomposition ===")
    s1 = fit_rkhs_decomposition(image, verbose=True, maxiter=30)
    M        = s1["M"]
    Kd       = s1["Kd"]
    Psi_beta = s1["Psi_beta"]
    g        = 1.0 / (1.0 + iota * (Psi_beta ** 2))

    # --- Stage 2: segmentation ---
    print("\n=== Stage 2: segmentation ===")
    s2 = segment(M, g, mask_init=mask_init, verbose=True,
                 lambda_regionfit=1e-4, mu_boundwt=1e-2, maxiter=40)
    u = s2["u"]

    # --- plot ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input z")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(Kd, cmap="gray")
    axes[0, 1].set_title(f"Kd (smooth)\n[{Kd.min():.2f}, {Kd.max():.2f}]")
    axes[0, 1].axis("off")

    pb_lim = max(abs(Psi_beta.min()), abs(Psi_beta.max()), 1e-6)
    axes[0, 2].imshow(Psi_beta, cmap="seismic", vmin=-pb_lim, vmax=pb_lim)
    axes[0, 2].set_title(f"Psi*beta (edges)\n[{Psi_beta.min():.2f}, {Psi_beta.max():.2f}]")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(g, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("g(Psi*beta)\n(edge-stopping)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(M, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("M = Kd + Psi*beta")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].contour(mask_init, [0.5], colors="red", linewidths=1.0, linestyles="dashed")
    axes[1, 2].contour(u, [0.5], colors="lime", linewidths=2.0)
    axes[1, 2].set_title(f"Segmentation (u > 0.5)\nc1={s2['c1']:.3f}, c2={s2['c2']:.3f}")
    axes[1, 2].axis("off")

    plt.suptitle("Two-stage RKHS segmentation", fontsize=14)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "twostage_demo.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure to {out_path}")


if __name__ == "__main__":
    main()
