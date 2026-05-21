"""
PyTorch / GPU port of Stage 2 (segmentation) from rkhs_segment.py.

Public API mirrors ``rkhs_segment.segment``.  The numerical recipe is the
same Chan-Vese + edge-weighted TV splitting; the only structural change is
that all image-level operations (FFT solve, soft threshold, gradient) use
``torch.fft`` and element-wise torch ops so the entire Stage 2 stays on
device when paired with ``rkhs_modelfit_torch.fit_rkhs_decomposition``.

Entry point: ``segment``.
"""

from copy import deepcopy

import numpy as np
import torch


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Default config dict for Stage 2 segmentation.

    Identical to ``rkhs_segment.default_config``; duplicated so this module
    is self-contained.
    """
    return {
        "mdl": {
            "lambda_regionfit": 1e-6,
            "mu_boundwt":       1e-3,
        },
        "opt": {
            "rho_gradsplit": 1e-9,
            "zeta_uprox":    4.0e-6,
            "zeta_inprox":   1e-9,
            "zeta_outprox":  1e-9,
            "maxiter":       20,
        },
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _soft_threshold(x, t):
    """Element-wise soft threshold; ``t`` may be scalar or tensor."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0.0)


def _psf2otf(psf_np, shape, *, device, complex_dtype):
    """Torch equivalent of MATLAB psf2otf, returning a complex tensor."""
    psf = torch.as_tensor(psf_np, dtype=torch.float64, device=device)
    pad = torch.zeros(shape, dtype=torch.float64, device=device)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    for axis, axis_size in enumerate(psf.shape):
        pad = torch.roll(pad, -(axis_size // 2), dims=axis)
    otf = torch.fft.fft2(pad)
    return otf.to(complex_dtype)


def _matlab_gradient(F):
    """Return (Fx, Fy) matching MATLAB's [Fx, Fy] = gradient(F).

    Central differences, matching ``np.gradient`` / the MATLAB ``gradient``
    function used in ``update_membership_split.m``.

    Fx = column-direction (axis=1), Fy = row-direction (axis=0).
    """
    # Interior: central differences
    Fx = torch.empty_like(F)
    Fx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / 2.0
    # Boundaries: one-sided differences
    Fx[:, 0]  = F[:, 1] - F[:, 0]
    Fx[:, -1] = F[:, -1] - F[:, -2]

    Fy = torch.empty_like(F)
    Fy[1:-1, :] = (F[2:, :] - F[:-2, :]) / 2.0
    Fy[0, :]  = F[1, :] - F[0, :]
    Fy[-1, :] = F[-1, :] - F[-2, :]

    return Fx, Fy


# ---------------------------------------------------------------------------
# FFT cache
# ---------------------------------------------------------------------------

def _build_fft_cache(im_shape, device, dtype):
    """Precompute the FFT difference operators for the u-update."""
    H, W = im_shape
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    Dx_otf = _psf2otf(np.array([[1.0, -1.0]]),  (H, W),
                       device=device, complex_dtype=complex_dtype)
    Dy_otf = _psf2otf(np.array([[1.0], [-1.0]]), (H, W),
                       device=device, complex_dtype=complex_dtype)
    Dx_conj = torch.conj(Dx_otf)
    Dy_conj = torch.conj(Dy_otf)
    Lap_otf = (Dx_otf * Dx_conj + Dy_otf * Dy_conj).real

    return {
        "Dx_conj": Dx_conj,
        "Dy_conj": Dy_conj,
        "Lap_otf": Lap_otf,
        "device":  device,
        "dtype":   dtype,
    }


# ---------------------------------------------------------------------------
# iteration updates
# ---------------------------------------------------------------------------

def _update_region_means(J, u, c1, c2, cfg):
    """Update the Chan-Vese inside/outside means c1, c2."""
    eps = torch.finfo(u.dtype).eps
    z_in  = cfg["opt"]["zeta_inprox"]
    z_out = cfg["opt"]["zeta_outprox"]

    one_minus_u = 1.0 - u
    c1_new = float(((J * u).sum()           + z_in  * c1) / (u.sum()           + z_in  + eps))
    c2_new = float(((J * one_minus_u).sum() + z_out * c2) / (one_minus_u.sum() + z_out + eps))
    return c1_new, c2_new


def _update_membership(J, g, u, wx, wy, b2x, b2y, c1, c2, fft_cache, cfg):
    """Update u (membership) and the gradient split variables.

    Mirrors ``rkhs_segment._update_membership`` using torch FFT ops.
    """
    eps   = torch.finfo(u.dtype).eps
    lam   = cfg["mdl"]["lambda_regionfit"]
    mu    = cfg["mdl"]["mu_boundwt"]
    rho   = cfg["opt"]["rho_gradsplit"]
    zeta5 = cfg["opt"]["zeta_uprox"]

    Dx_conj = fft_cache["Dx_conj"]
    Dy_conj = fft_cache["Dy_conj"]
    Lap_otf = fft_cache["Lap_otf"]

    fft2  = torch.fft.fft2
    ifft2 = torch.fft.ifft2

    region_force = (J - c1) ** 2 - (J - c2) ** 2

    # u FFT solve
    u_ra = rho * (Dx_conj * fft2(wx + b2x) + Dy_conj * fft2(wy + b2y))
    u_rb = lam * fft2(region_force)
    u_rc = zeta5 * fft2(u)
    u_l  = rho * Lap_otf + zeta5 + eps

    u_new = torch.real(ifft2((u_ra - u_rb + u_rc) / u_l))
    u_new = u_new.clamp(0.0, 1.0)

    # gradient split: soft threshold with edge-weighted threshold
    ux, uy = _matlab_gradient(u_new)
    wx_new = _soft_threshold(ux - b2x, g * (mu / rho))
    wy_new = _soft_threshold(uy - b2y, g * (mu / rho))

    # Bregman dual update
    b2x_new = b2x + wx_new - ux
    b2y_new = b2y + wy_new - uy

    return u_new, wx_new, wy_new, b2x_new, b2y_new


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def segment(M, g, mask_init=None, cfg=None, *,
            device=None, dtype=torch.float32,
            verbose=False, return_numpy=True, **overrides):
    """Run Stage 2 segmentation given a fixed reconstruction and edge map.

    Parameters
    ----------
    M : (H, W) array or torch.Tensor
        Reconstructed image from Stage 1 (= Kd + Psi*beta).
    g : (H, W) array or torch.Tensor
        Edge-stopping function from Stage 1.
    mask_init : (H, W) array, torch.Tensor, or None
        Initial binary mask.
    cfg : dict, optional
        Config dict shaped like ``default_config()``.
    device : str | torch.device | None
        Defaults to ``'cuda'`` if available, else ``'cpu'``.
    dtype : torch.dtype
        Computation dtype.
    verbose : bool
        Print per-iteration diagnostics.
    return_numpy : bool
        If True, return numpy arrays on CPU. If False, return torch tensors
        on ``device``.
    **overrides
        Flat keyword overrides (e.g. ``lambda_regionfit=1e-4``).

    Returns
    -------
    result : dict with keys
        u, c1, c2, cfg, history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    def to_t(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(x, dtype=np.float64),
                               dtype=dtype, device=device)

    M_t = to_t(M)
    g_t = to_t(g)

    if M_t.ndim != 2:
        raise ValueError("M must be 2D (H, W)")
    if g_t.shape != M_t.shape:
        raise ValueError("g shape must match M shape")

    cfg = deepcopy(default_config()) if cfg is None else deepcopy(cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    H, W = M_t.shape
    fft_cache = _build_fft_cache((H, W), device, dtype)
    eps = torch.finfo(dtype).eps

    # --- initialise state ---
    z = lambda: torch.zeros((H, W), dtype=dtype, device=device)

    if mask_init is not None:
        mask_t = to_t(mask_init)
        u = (mask_t > 0).to(dtype)
    else:
        u = z()

    wx, wy = _matlab_gradient(u)
    b2x, b2y = z(), z()

    one_minus_u = 1.0 - u
    c1 = float((M_t * u).sum() / (u.sum() + eps))
    c2 = float((M_t * one_minus_u).sum() / (one_minus_u.sum() + eps))

    # --- iterate ---
    n_iter  = int(cfg["opt"]["maxiter"])
    history = []

    for it in range(n_iter):
        c1, c2 = _update_region_means(M_t, u, c1, c2, cfg)
        u, wx, wy, b2x, b2y = _update_membership(
            M_t, g_t, u, wx, wy, b2x, b2y, c1, c2, fft_cache, cfg)

        area = int((u > 0.5).sum().item())

        if return_numpy:
            history.append({
                "iter": it, "u": u.detach().cpu().numpy().copy(),
                "c1": c1, "c2": c2, "area": area,
            })
        else:
            history.append({
                "iter": it, "u": u.clone(),
                "c1": c1, "c2": c2, "area": area,
            })

        if verbose:
            print("iter %3d/%d  c1=%+.4f  c2=%+.4f  area=%6d"
                  % (it + 1, n_iter, c1, c2, area))

    if return_numpy:
        u_out = u.detach().cpu().numpy()
    else:
        u_out = u

    return {
        "u":       u_out,
        "c1":      c1,
        "c2":      c2,
        "cfg":     cfg,
        "history": history,
    }


# ---------------------------------------------------------------------------
# convenience: both stages in one call
# ---------------------------------------------------------------------------

def twostage_segment(image, mask_init=None, stage1_cfg=None, stage2_cfg=None,
                     *, iota_edgegate=1e3, device=None, dtype=torch.float32,
                     verbose=False, return_numpy=True, **stage1_overrides):
    """Convenience wrapper: torch Stage 1 decomposition then torch Stage 2
    segmentation, everything on-device.

    Parameters
    ----------
    image : (H, W) array or torch.Tensor
    mask_init : (H, W) array, torch.Tensor, or None
        Initial mask for Stage 2.
    stage1_cfg, stage2_cfg : dict or None
    iota_edgegate : float
        Iota for computing g = 1 / (1 + iota * |Psi*beta|^2).
    device, dtype : torch device/dtype for both stages.
    verbose : bool
    **stage1_overrides
        Flat overrides forwarded to Stage 1.

    Returns
    -------
    result : dict with keys from both stages.
    """
    from rkhs_modelfit_torch import fit_rkhs_decomposition

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    stage1 = fit_rkhs_decomposition(
        image, cfg=stage1_cfg, device=device, dtype=dtype,
        verbose=verbose, return_numpy=False, **stage1_overrides)

    M_t        = stage1["M"]
    Psi_beta_t = stage1["Psi_beta"]
    g_t        = 1.0 / (1.0 + iota_edgegate * (Psi_beta_t ** 2))

    stage2 = segment(
        M_t, g_t, mask_init=mask_init, cfg=stage2_cfg,
        device=device, dtype=dtype,
        verbose=verbose, return_numpy=return_numpy)

    if return_numpy:
        result = {
            "u":        stage2["u"],
            "c1":       stage2["c1"],
            "c2":       stage2["c2"],
            "M":        M_t.detach().cpu().numpy(),
            "Kd":       stage1["Kd"].detach().cpu().numpy(),
            "Psi_beta": Psi_beta_t.detach().cpu().numpy(),
            "g":        g_t.detach().cpu().numpy(),
        }
    else:
        result = {
            "u":        stage2["u"],
            "c1":       stage2["c1"],
            "c2":       stage2["c2"],
            "M":        M_t,
            "Kd":       stage1["Kd"],
            "Psi_beta": Psi_beta_t,
            "g":        g_t,
        }

    result["stage1_result"] = stage1
    result["stage2_result"] = stage2
    return result


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

def _apply_overrides(cfg, overrides):
    """Route flat keyword overrides into the nested cfg dict."""
    sections = ("mdl", "opt")
    for k, v in overrides.items():
        placed = False
        for sec in sections:
            if k in cfg.get(sec, {}):
                cfg[sec][k] = v
                placed = True
                break
        if not placed:
            raise KeyError("unknown override: %r" % k)
    return cfg
