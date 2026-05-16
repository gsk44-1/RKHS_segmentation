"""Numerical parity test: rkhs_modelfit (numpy) vs rkhs_modelfit_torch.

Runs both implementations on the synth disk image from run_modelfit_demo.py
and asserts the Kd / Psi_beta / M outputs match. Default tolerances:
  * float64 on CPU  : 1e-10 max-abs on image-level outputs (Kd, Psi_beta, M).
  * float32 on CPU  : 5e-4  max-abs on image-level outputs (T4 looks the same).

Per-patch d is verified separately at a much looser tolerance because the
d-update applies ``A_d_inv`` (cond ~ 1/zeta1, default ~1e9), so
``A_inv @ rhs`` (looping reference) vs ``rhs @ A_inv`` (batched torch /
matmul) can disagree by ~1e-6 in float64 purely from BLAS summation order
along ``A_d_inv``'s ill-conditioned directions. Those differences live in
``K``'s near-null space, so ``Kd = K @ d`` still agrees at 1e-10.

Run from the project_RKHS_python directory:

    python test_modelfit_parity.py            # float64 strict parity
    python test_modelfit_parity.py --dtype 32 # float32 sanity check
    python test_modelfit_parity.py --iters 5  # quick smoke test
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Make sure we import the modules sitting next to this file regardless of
# where the script was launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rkhs_modelfit as ref_np
import rkhs_modelfit_torch as ref_t


def synth(H=64, W=64, noise=0.05, seed=0):
    """Same synth as run_modelfit_demo.py."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    img = 0.15 + 0.7 * (r < min(H, W) / 4).astype(float)
    return np.clip(img + noise * rng.standard_normal(img.shape), 0., 1.)


def compare(a, b, label, tol):
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    rms     = float(np.sqrt((diff ** 2).mean()))
    status  = "OK " if max_abs < tol else "BAD"
    print(f"  [{status}] {label:14s}  "
          f"max|Δ|={max_abs:.3e}  rms={rms:.3e}  tol={tol:.0e}")
    return max_abs < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["32", "64"], default="64",
                    help="torch dtype: 32=float32, 64=float64 (parity)")
    ap.add_argument("--iters", type=int, default=15,
                    help="number of BCD iterations (numpy + torch must match)")
    ap.add_argument("--device", default="cpu",
                    help="torch device ('cpu' or 'cuda')")
    ap.add_argument("--overlap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "32" else torch.float64
    # Image-level outputs (Kd, Psi_beta, M, beta) match tightly. The per-patch
    # d only matches at ~1e-6 in fp64 because A_d_inv is ill-conditioned; see
    # the module docstring.
    tol     = 5e-4 if dtype == torch.float32 else 1e-10
    tol_d   = 5e-4 if dtype == torch.float32 else 1e-4

    img = synth(seed=args.seed)

    print(f"running numpy reference ({args.iters} iters)...")
    t0 = time.time()
    res_np = ref_np.fit_rkhs_decomposition(
        img, maxiter=args.iters, overlap=args.overlap, verbose=False)
    t_np = time.time() - t0

    print(f"running torch port    (device={args.device}, dtype={dtype}, "
          f"{args.iters} iters)...")
    t0 = time.time()
    res_t = ref_t.fit_rkhs_decomposition(
        img, maxiter=args.iters, overlap=args.overlap,
        device=args.device, dtype=dtype, verbose=False)
    t_t = time.time() - t0

    print()
    print(f"timing: numpy={t_np:.3f}s   torch[{args.device}]={t_t:.3f}s   "
          f"speedup={t_np / max(t_t, 1e-9):.2f}x")

    print()
    print(f"parity check (tol={tol:.0e}, tol_d={tol_d:.0e}):")
    all_ok = True
    for key in ("Kd", "Psi_beta", "M"):
        all_ok &= compare(res_np[key], res_t[key], key, tol)
    # Per-patch beta matches at the image-level tolerance.
    all_ok &= compare(res_np["beta"], res_t["beta"], "beta (patch)", tol)
    # Per-patch d uses a looser tolerance: see docstring at top of file.
    all_ok &= compare(res_np["d"],    res_t["d"],    "d (patch)",    tol_d)

    print()
    print(f"residuals (final): numpy={res_np['history'][-1]['residual']:.6f} "
          f"vs torch={res_t['history'][-1]['residual']:.6f}")

    print()
    print(f"diagnostics agreement:")
    print(f"  PtP_op:    np={res_np['diagnostics']['PtP_op']:.6f}  "
          f"t={res_t['diagnostics']['PtP_op']:.6f}")
    print(f"  L_beta:    np={res_np['diagnostics']['L_beta']:.6f}  "
          f"t={res_t['diagnostics']['L_beta']:.6f}")
    print(f"  zeta2_eff: np={res_np['diagnostics']['zeta2_eff']:.6f}  "
          f"t={res_t['diagnostics']['zeta2_eff']:.6f}")

    print()
    if all_ok:
        print("PARITY OK")
        sys.exit(0)
    else:
        print("PARITY FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
