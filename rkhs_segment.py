"""
RKHS segmentation — Stage 2 of the two-stage global segmentation method
from Burrows, Guo, Chen & Torella, "Reproducing Kernel Hilbert Space Based
Global and Local Image Segmentation", Inverse Problems and Imaging, 2020
(IPI2020_RKHSSeg.pdf).

Stage 1 (``rkhs_modelfit.fit_rkhs_decomposition``) produces a decomposition

    M = K d + Psi beta

and the edge-stopping function

    g(Psi beta) = 1 / (1 + iota * |Psi beta|^2).

Stage 2 (this module) takes M and g as *fixed* inputs and finds the
segmentation map u in [0, 1] by minimising a Chan-Vese energy with
edge-weighted total variation:

    min_{u, c1, c2}   lambda * int [ (M - c1)^2 u + (M - c2)^2 (1 - u) ] dx
                     + mu * int g |grad u| dx

where c1, c2 are the region means (updated analytically at each iteration)
and u is solved via an ADMM / Bregman splitting on grad u.

This corresponds to the MATLAB code path ``use_selective_two_stage`` in
``solve_global_combined_model.m``, which first freezes u and runs
reconstruction iterations (Stage 1), then freezes the reconstruction and
runs segmentation iterations (Stage 2).

Entry point: ``segment``.
"""

from copy import deepcopy

import numpy as np

EPS = float(np.finfo(float).eps)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Default config dict for Stage 2 segmentation.

    Sections:
      mdl : model term weights
        lambda_regionfit  : weight on the Chan-Vese data-fidelity term.
        mu_boundwt        : weight on the edge-weighted TV boundary term.
      opt : solver / proximal / split parameters
        rho_gradsplit     : ADMM penalty for the grad u = (wx, wy) split.
        zeta_uprox        : proximal weight on u (stabilises the FFT solve).
        zeta_inprox       : proximal weight on c1 update.
        zeta_outprox      : proximal weight on c2 update.
        maxiter           : number of outer iterations.
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
    """Element-wise soft threshold; ``t`` may be scalar or array."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _psf2otf(psf, shape):
    """Equivalent of MATLAB psf2otf: zero-pad, circshift, FFT2."""
    psf = np.asarray(psf, dtype=float)
    pad = np.zeros(shape, dtype=float)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    for axis, axis_size in enumerate(psf.shape):
        pad = np.roll(pad, -(axis_size // 2), axis=axis)
    return np.fft.fft2(pad)


def _matlab_gradient(F):
    """Return (Fx, Fy) matching MATLAB's [Fx, Fy] = gradient(F).

    Fx is the column-direction (axis=1) gradient, Fy is the row-direction
    (axis=0). This uses central differences, matching the MATLAB ``gradient``
    function used in ``update_membership_split.m``.
    """
    Fy, Fx = np.gradient(F)
    return Fx, Fy


# ---------------------------------------------------------------------------
# FFT cache (constant across iterations)
# ---------------------------------------------------------------------------

def _build_fft_cache(im_shape):
    """Precompute the FFT difference operators for the u-update."""
    H, W = im_shape
    Dx_otf  = _psf2otf(np.array([[1.0, -1.0]]),  (H, W))
    Dy_otf  = _psf2otf(np.array([[1.0], [-1.0]]), (H, W))
    Dx_conj = np.conj(Dx_otf)
    Dy_conj = np.conj(Dy_otf)
    Lap_otf = (Dx_otf * Dx_conj + Dy_otf * Dy_conj).real
    return {
        "Dx_conj": Dx_conj,
        "Dy_conj": Dy_conj,
        "Lap_otf": Lap_otf,
    }


# ---------------------------------------------------------------------------
# iteration updates
# ---------------------------------------------------------------------------

def _update_region_means(J, u, c1, c2, cfg):
    """Update the Chan-Vese inside/outside means c1, c2.

    With proximal regularisation to stabilise early iterations:
        c1_new = (sum(J * u) + zeta_in * c1_old) / (sum(u) + zeta_in)
    and similarly for c2.
    """
    z_in  = cfg["opt"]["zeta_inprox"]
    z_out = cfg["opt"]["zeta_outprox"]

    one_minus_u = 1.0 - u
    c1_new = float(((J * u).sum()           + z_in  * c1) / (u.sum()           + z_in  + EPS))
    c2_new = float(((J * one_minus_u).sum() + z_out * c2) / (one_minus_u.sum() + z_out + EPS))
    return c1_new, c2_new


def _update_membership(J, g, u, wx, wy, b2x, b2y, c1, c2, fft_cache, cfg):
    """Update u (membership) and the gradient split (wx, wy, b2x, b2y).

    Mirrors ``update_membership_split.m`` for the global branch (no selective
    geodesic term). The u-subproblem is:

        (rho * Lap + zeta5) u = rho * div(w + b2) - lambda * region_force + zeta5 * u_old

    solved via FFT, then projected to [0, 1]. The (wx, wy) subproblem is
    edge-weighted soft thresholding on grad(u).
    """
    lam   = cfg["mdl"]["lambda_regionfit"]
    mu    = cfg["mdl"]["mu_boundwt"]
    rho   = cfg["opt"]["rho_gradsplit"]
    zeta5 = cfg["opt"]["zeta_uprox"]

    Dx_conj = fft_cache["Dx_conj"]
    Dy_conj = fft_cache["Dy_conj"]
    Lap_otf = fft_cache["Lap_otf"]

    fft2  = np.fft.fft2
    ifft2 = np.fft.ifft2

    region_force = (J - c1) ** 2 - (J - c2) ** 2

    # u FFT solve
    u_ra = rho * (Dx_conj * fft2(wx + b2x) + Dy_conj * fft2(wy + b2y))
    u_rb = lam * fft2(region_force)
    u_rc = zeta5 * fft2(u)
    u_l  = rho * Lap_otf + zeta5 + EPS

    u_new = np.real(ifft2((u_ra - u_rb + u_rc) / u_l))
    u_new = np.clip(u_new, 0.0, 1.0)

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

def segment(M, g, mask_init=None, cfg=None, *, verbose=False, **overrides):
    """Run Stage 2 segmentation given a fixed reconstruction and edge map.

    Parameters
    ----------
    M : (H, W) array
        Reconstructed image from Stage 1 (= Kd + Psi*beta). This is used
        as the data image J in the Chan-Vese region-fitting term.
    g : (H, W) array
        Edge-stopping function from Stage 1: g = 1 / (1 + iota * |Psi*beta|^2).
        Weights the TV penalty on grad(u) so that the boundary prefers to
        align with detected edges.
    mask_init : (H, W) array or None
        Initial binary mask (will be cast to {0, 1}). If None, starts from
        all zeros — note this can be very slow to evolve; providing any
        reasonable init (e.g. a thresholded version of M, or a centred box)
        is strongly recommended.
    cfg : dict, optional
        Config dict shaped like ``default_config()``.
    verbose : bool
        Print per-iteration diagnostics.
    **overrides
        Flat keyword overrides routed to the matching cfg sub-dict.
        Examples: ``lambda_regionfit=1e-4``, ``mu_boundwt=5e-3``,
        ``maxiter=50``.

    Returns
    -------
    result : dict with keys
        u       : (H, W) float — final soft membership in [0, 1].
                  Threshold at 0.5 to obtain a binary segmentation.
        c1      : float — final inside-region mean.
        c2      : float — final outside-region mean.
        cfg     : the resolved config dict that was used.
        history : list of per-iteration dicts (u, c1, c2, area).
    """
    M = np.asarray(M, dtype=float)
    g = np.asarray(g, dtype=float)
    if M.ndim != 2:
        raise ValueError("M must be 2D (H, W)")
    if g.shape != M.shape:
        raise ValueError(f"g shape {g.shape} must match M shape {M.shape}")

    cfg = deepcopy(default_config()) if cfg is None else deepcopy(cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    H, W = M.shape
    fft_cache = _build_fft_cache((H, W))

    # --- initialise state ---
    if mask_init is None:
        u = np.zeros((H, W))
    else:
        u = (np.asarray(mask_init) > 0).astype(float)

    wx, wy = _matlab_gradient(u)
    b2x = np.zeros((H, W))
    b2y = np.zeros((H, W))

    one_minus_u = 1.0 - u
    c1 = float((M * u).sum() / (u.sum() + EPS))
    c2 = float((M * one_minus_u).sum() / (one_minus_u.sum() + EPS))

    # --- iterate ---
    n_iter  = int(cfg["opt"]["maxiter"])
    history = []

    for it in range(n_iter):
        c1, c2 = _update_region_means(M, u, c1, c2, cfg)
        u, wx, wy, b2x, b2y = _update_membership(
            M, g, u, wx, wy, b2x, b2y, c1, c2, fft_cache, cfg)

        area = int((u > 0.5).sum())
        history.append({
            "iter": it,
            "u":    u.copy(),
            "c1":   c1,
            "c2":   c2,
            "area": area,
        })
        if verbose:
            print(f"iter {it+1:3d}/{n_iter}  c1={c1:+.4f}  c2={c2:+.4f}  "
                  f"area={area:>6d}")

    return {
        "u":       u,
        "c1":      c1,
        "c2":      c2,
        "cfg":     cfg,
        "history": history,
    }


# ---------------------------------------------------------------------------
# convenience: run both stages from a single call
# ---------------------------------------------------------------------------

def twostage_segment(image, mask_init=None, stage1_cfg=None, stage2_cfg=None,
                     *, iota_edgegate=1e3, verbose=False,
                     **stage1_overrides):
    """Convenience wrapper: Stage 1 decomposition then Stage 2 segmentation.

    Parameters
    ----------
    image : (H, W) array
        Input image.
    mask_init : (H, W) array or None
        Initial mask for Stage 2 (not used by Stage 1).
    stage1_cfg, stage2_cfg : dict or None
        Separate configs for each stage.
    iota_edgegate : float
        The iota parameter for computing g = 1 / (1 + iota * |Psi*beta|^2).
        Should match the iota used in Stage 1 if edge gating was active there.
    verbose : bool
    **stage1_overrides
        Flat overrides forwarded to ``rkhs_modelfit.fit_rkhs_decomposition``.

    Returns
    -------
    result : dict with keys
        u, c1, c2 : from Stage 2
        M, Kd, Psi_beta, g : from Stage 1
        stage1_result, stage2_result : full result dicts from each stage
    """
    try:
        from rkhs_modelfit_torch import fit_rkhs_decomposition
        stage1 = fit_rkhs_decomposition(image, cfg=stage1_cfg, verbose=verbose,
                                        return_numpy=True, **stage1_overrides)
    except ImportError:
        from rkhs_modelfit import fit_rkhs_decomposition
        stage1 = fit_rkhs_decomposition(image, cfg=stage1_cfg, verbose=verbose,
                                        **stage1_overrides)
    M        = stage1["M"]
    Psi_beta = stage1["Psi_beta"]
    g        = 1.0 / (1.0 + iota_edgegate * (Psi_beta ** 2))

    stage2 = segment(M, g, mask_init=mask_init, cfg=stage2_cfg, verbose=verbose)

    return {
        "u":       stage2["u"],
        "c1":      stage2["c1"],
        "c2":      stage2["c2"],
        "M":       M,
        "Kd":      stage1["Kd"],
        "Psi_beta": Psi_beta,
        "g":       g,
        "stage1_result": stage1,
        "stage2_result": stage2,
    }


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
