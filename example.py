"""
Example: load (or synthesize) a 2D image, run global RKHS segmentation,
and show a before / mask-init / after plot.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rkhs_seg import run_global_segmentation


def synthesize_disk(H=64, W=64, noise=0.05, seed=0):
    """A simple bright-disk-on-dark-background test image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W))
    cy, cx = H // 2, W // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    img = 0.15 + 0.7 * (r < min(H, W) / 4).astype(float)
    img = img + noise * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


def main():
    # Default demo: synthetic bright disk on dark background. To use your own
    # 2D image instead, replace these two lines with e.g.
    #     image = np.load("path/to/your_image.npy").astype(float)
    # and pick an initial mask (a thresholded version of the image, an ROI you
    # know contains the object, etc.). Default model parameters are very
    # gentle (small lambda); for real data you'll likely want to bump
    # mdl_lambda_regionfit up by 2-3 orders of magnitude.
    image = synthesize_disk()
    source = "synthetic disk"
    H, W = image.shape

    # Initial mask: a centred rectangle. Replace with your own initialization
    # (clicked ROI, thresholded image, etc.) for real images.
    mask_init = np.zeros((H, W))
    sy, sx = H // 4, W // 4
    ey, ex = 3 * H // 4, 3 * W // 4
    mask_init[sy:ey, sx:ex] = 1.0

    print(f"image: {source}, shape {image.shape}, range [{image.min():.3g}, {image.max():.3g}]")

    result = run_global_segmentation(
        image, mask_init,
        opt_maxiter_loopcap=20,
        verbose=True,
    )
    u = result["u"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Input ({source})")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].contour(mask_init, [0.5], colors="red", linewidths=1.5)
    axes[1].set_title("Initial mask")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(u, [0.5], colors="lime", linewidths=1.5)
    axes[2].set_title("Final segmentation (u > 0.5)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "example_result.png", dpi=120)
    print(f"saved figure to {Path(__file__).resolve().parent / 'example_result.png'}")
    plt.show()


if __name__ == "__main__":
    main()
