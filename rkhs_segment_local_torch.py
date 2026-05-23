"""
Local Chan-Vese extension of the RKHS Stage 2 segmentation.

Adds a local fitting term to the split-Bregman Chan-Vese segmentation in
``rkhs_segment_torch``.  The energy functional becomes:

    E(u) = alpha * integral [(J - c1)^2 u + (J - c2)^2 (1-u)] dx      (global)
         + beta  * integral [(D - d1)^2 u + (D - d2)^2 (1-u)] dx      (local)
         + mu    * integral  g |grad u| dx                             (boundary)

where D = boxfilter(J, k) - J is the local difference image, and d1, d2 are
the u-weighted means of D inside/outside.

The u-update is the same FFT / split-Bregman solve as rkhs_segment_torch,
with the region force extended to alpha*(global) + beta*(local).

Entry point: ``segment``.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Default config dict.

    mdl
    ---
    alpha_global    : weight on the global Chan-Vese fitting term  (1.0)
    beta_local      : weight on the local fitting term             (1.0)
    lambda_regionfit: overall coupling of region force to u-update (1e-6)
    mu_boundwt      : edge-weighted TV penalty                     (1e-3)
    k_local         : box-filter window size for the local term    (15)

    opt
    ---
    rho_gradsplit : ADMM penalty for the gradient split   (1e-9)
    zeta_uprox    : proximal weight on u                  (4e-6)
    zeta_inprox   : proximal weight on c1 update          (1e-9)
    zeta_outprox  : proximal weight on c2 update          (1e-9)
    maxiter       : number of outer iterations            (20)
    """
    return {
        "mdl": {
            "alpha_global":     1.0,
            "beta_local":       1.0,
            "lambda_regionfit": 1e-6,
            "mu_boundwt":       1e-3,
            "k_local":          15,
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
# helpers (unchanged from rkhs_segment_torch)
# ---------------------------------------------------------------------------

def _soft_threshold(x, t):
    return torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0.0)


def _psf2otf(psf_np, shape, *, device, complex_dtype):
    psf = torch.as_tensor(psf_np, dtype=torch.float64, device=device)
    pad = torch.zeros(shape, dtype=torch.float64, device=device)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    for axis, axis_size in enumerate(psf.shape):
        pad = torch.roll(pad, -(axis_size // 2), dims=axis)
    otf = torch.fft.fft2(pad)
    return otf.to(complex_dtype)


def _matlab_gradient(F_tensor):
    Fx = torch.empty_like(F_tensor)
    Fx[:, 1:-1] = (F_tensor[:, 2:] - F_tensor[:, :-2]) / 2.0
    Fx[:, 0]  = F_tensor[:, 1] - F_tensor[:, 0]
    Fx[:, -1] = F_tensor[:, -1] - F_tensor[:, -2]

    Fy = torch.empty_like(F_tensor)
    Fy[1:-1, :] = (F_tensor[2:, :] - F_tensor[:-2, :]) / 2.0
    Fy[0, :]  = F_tensor[1, :] - F_tensor[0, :]
    Fy[-1, :] = F_tensor[-1, :] - F_tensor[-2, :]

    return Fx, Fy


# ---------------------------------------------------------------------------
# box filter for local term
# ---------------------------------------------------------------------------

def _box_filter(image, k, device, dtype):
    """k x k box (averaging) filter via conv2d with replicate padding."""
    pad_before = k // 2
    pad_after  = k - 1 - pad_before
    img_4d = image.unsqueeze(0).unsqueeze(0)
    img_padded = F.pad(img_4d,
                       (pad_before, pad_after, pad_before, pad_after),
                       mode='replicate')
    kernel = torch.ones(1, 1, k, k, device=device, dtype=dtype) / (k * k)
    out = F.conv2d(img_padded, kernel)
    return out.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# FFT cache
# ---------------------------------------------------------------------------

def _build_fft_cache(im_shape, device, dtype):
    H, W = im_shape
    complex_dtype = (torch.complex64 if dtype == torch.float32
                     else torch.complex128)

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
    }


# ---------------------------------------------------------------------------
# iteration updates
# ---------------------------------------------------------------------------

def _update_region_means(J, diff, u, c1, c2, d1, d2, cfg):
    """Update global means (c1, c2) and local means (d1, d2)."""
    eps   = torch.finfo(u.dtype).eps
    z_in  = cfg["opt"]["zeta_inprox"]
    z_out = cfg["opt"]["zeta_outprox"]

    one_minus_u = 1.0 - u
    sum_u   = u.sum() + eps
    sum_1mu = one_minus_u.sum() + eps

    c1_new = float(((J * u).sum()           + z_in  * c1) / (sum_u   + z_in))
    c2_new = float(((J * one_minus_u).sum() + z_out * c2) / (sum_1mu + z_out))

    d1_new = float(((diff * u).sum()           + z_in  * d1) / (sum_u   + z_in))
    d2_new = float(((diff * one_minus_u).sum() + z_out * d2) / (sum_1mu + z_out))

    return c1_new, c2_new, d1_new, d2_new


def _update_membership(J, diff, g, u, wx, wy, b2x, b2y,
                       c1, c2, d1, d2, fft_cache, cfg):
    """Update u (membership) and the gradient-split variables."""
    eps   = torch.finfo(u.dtype).eps
    alpha = cfg["mdl"]["alpha_global"]
    beta  = cfg["mdl"]["beta_local"]
    lam   = cfg["mdl"]["lambda_regionfit"]
    mu    = cfg["mdl"]["mu_boundwt"]
    rho   = cfg["opt"]["rho_gradsplit"]
    zeta5 = cfg["opt"]["zeta_uprox"]

    Dx_conj = fft_cache["Dx_conj"]
    Dy_conj = fft_cache["Dy_conj"]
    Lap_otf = fft_cache["Lap_otf"]

    fft2  = torch.fft.fft2
    ifft2 = torch.fft.ifft2

    # combined region force: alpha * global + beta * local
    region_force = (alpha * ((J - c1) ** 2    - (J - c2) ** 2)
                  + beta  * ((diff - d1) ** 2 - (diff - d2) ** 2))

    # u FFT solve  (same structure as rkhs_segment_torch)
    u_ra = rho * (Dx_conj * fft2(wx + b2x) + Dy_conj * fft2(wy + b2y))
    u_rb = lam * fft2(region_force)
    u_rc = zeta5 * fft2(u)
    u_l  = rho * Lap_otf + zeta5 + eps

    u_new = torch.real(ifft2((u_ra - u_rb + u_rc) / u_l))
    u_new = u_new.clamp(0.0, 1.0)

    # gradient split: soft threshold with edge weight
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
    """Run local Chan-Vese segmentation with edge-weighted TV.

    Parameters
    ----------
    M : (H, W) array or torch.Tensor
        Reconstructed image (e.g. Kd + Psi*beta from Stage 1).
    g : (H, W) array or torch.Tensor
        Edge-stopping function (e.g. 1 / (1 + iota * |Psi*beta|^2)).
    mask_init : (H, W) array, torch.Tensor, or None
        Initial binary mask.  If None, defaults to M > mean(M).
    cfg : dict, optional
        Config dict shaped like ``default_config()``.
    device, dtype, verbose, return_numpy : see ``rkhs_segment_torch``.
    **overrides
        Flat keyword overrides applied into the nested cfg dict.

    Returns
    -------
    result : dict
        u      : soft membership in [0, 1]  (H, W)
        c1, c2 : global inside/outside means
        d1, d2 : local inside/outside means
        cfg    : config dict used
        history: list of per-iteration records
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
    k = int(cfg["mdl"]["k_local"])
    fft_cache = _build_fft_cache((H, W), device, dtype)
    eps = torch.finfo(dtype).eps

    # --- difference image for local term (computed once) ---
    diff = _box_filter(M_t, k, device, dtype) - M_t

    # --- initialise state ---
    z = lambda: torch.zeros((H, W), dtype=dtype, device=device)

    if mask_init is not None:
        mask_t = to_t(mask_init)
        u = (mask_t > 0).to(dtype)
    else:
        # Default: threshold at the mean so c1/c2 start meaningfully
        u = (M_t > M_t.mean()).to(dtype)

    wx, wy = _matlab_gradient(u)
    b2x, b2y = z(), z()

    one_minus_u = 1.0 - u
    sum_u   = u.sum() + eps
    sum_1mu = one_minus_u.sum() + eps
    c1 = float((M_t * u).sum() / sum_u)
    c2 = float((M_t * one_minus_u).sum() / sum_1mu)
    d1 = float((diff * u).sum() / sum_u)
    d2 = float((diff * one_minus_u).sum() / sum_1mu)

    # --- iterate ---
    n_iter  = int(cfg["opt"]["maxiter"])
    history = []

    for it in range(n_iter):
        c1, c2, d1, d2 = _update_region_means(
            M_t, diff, u, c1, c2, d1, d2, cfg)
        u, wx, wy, b2x, b2y = _update_membership(
            M_t, diff, g_t, u, wx, wy, b2x, b2y,
            c1, c2, d1, d2, fft_cache, cfg)

        area = int((u > 0.5).sum().item())

        rec = {"iter": it, "c1": c1, "c2": c2,
               "d1": d1, "d2": d2, "area": area}
        if return_numpy:
            rec["u"] = u.detach().cpu().numpy().copy()
        else:
            rec["u"] = u.clone()
        history.append(rec)

        if verbose:
            print("iter %3d/%d  c1=%+.4f  c2=%+.4f  "
                  "d1=%+.4f  d2=%+.4f  area=%6d"
                  % (it + 1, n_iter, c1, c2, d1, d2, area))

    if return_numpy:
        u_out = u.detach().cpu().numpy()
    else:
        u_out = u

    return {
        "u":       u_out,
        "c1":      c1,
        "c2":      c2,
        "d1":      d1,
        "d2":      d2,
        "cfg":     cfg,
        "history": history,
    }


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

def _apply_overrides(cfg, overrides):
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
