"""Stage 1 decomposition demo / sanity check for the user.

Runs fit_rkhs_decomposition on a noisy disk image (matching example.py's
synth) and saves a 5-panel figure: input | Kd | Psi*beta | M=Kd+Psi*beta |
residual curve. Also runs a second test with the example.py overlap=2 to
confirm convergence is robust to the patch layout.
"""
import sys
sys.path.insert(0, '/sessions/tender-modest-cannon/mnt/project_RKHS/project_RKHS_python')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rkhs_modelfit import fit_rkhs_decomposition


def synth(H=64, W=64, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    img = 0.15 + 0.7 * (r < min(H, W) / 4).astype(float)
    return np.clip(img + noise * rng.standard_normal(img.shape), 0., 1.)


def demo(out_path, image, label, **overrides):
    res = fit_rkhs_decomposition(image, verbose=False, **overrides)
    Kd, Pb, M = res["Kd"], res["Psi_beta"], res["M"]
    history   = res["history"]
    diag      = res["diagnostics"]
    residuals = [h["residual"] for h in history]

    print(f"[{label}] zeta2_eff={diag['zeta2_eff']:.1f} "
          f"(L_beta={diag['L_beta']:.1f}). "
          f"residual: {residuals[0]:.3f} -> {residuals[-1]:.3f} "
          f"over {len(residuals)} iters.")

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"input z\n{label}")
    axes[0].axis("off")

    axes[1].imshow(Kd, cmap="gray")
    axes[1].set_title(f"Kd  [{Kd.min():.2f},{Kd.max():.2f}]")
    axes[1].axis("off")

    pb_lim = max(abs(Pb.min()), abs(Pb.max()), 1e-6)
    axes[2].imshow(Pb, cmap="seismic", vmin=-pb_lim, vmax=pb_lim)
    axes[2].set_title(f"Psi*beta  [{Pb.min():.2f},{Pb.max():.2f}]")
    axes[2].axis("off")

    axes[3].imshow(M, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"M = Kd + Psi*beta")
    axes[3].axis("off")

    axes[4].plot(residuals, marker="o", markersize=2)
    axes[4].set_xlabel("iteration")
    axes[4].set_ylabel("||z - M||")
    axes[4].set_title("residual")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return res


if __name__ == "__main__":
    img = synth()
    demo("/sessions/tender-modest-cannon/mnt/outputs/modelfit_disk_defaults.png",
         img, label="defaults (sigma=12, ovl=3)", maxiter=40)
    demo("/sessions/tender-modest-cannon/mnt/outputs/modelfit_disk_ovl2.png",
         img, label="overlap=2 like rkhs_seg", maxiter=40, overlap=2)
    # Stress test with no edge stopping (iota=0) — should still converge.
    demo("/sessions/tender-modest-cannon/mnt/outputs/modelfit_disk_noiota.png",
         img, label="iota=0 (no edge gate)", maxiter=40, iota_edgegate=0.0)
