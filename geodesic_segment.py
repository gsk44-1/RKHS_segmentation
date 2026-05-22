"""
Geodesic Active Contour segmentation — PyTorch implementation.

Based on:
  Caselles, Kimmel & Sapiro, "Geodesic Active Contours",
  International Journal of Computer Vision 22(1), 61-79 (1997).

The level-set PDE (Eq. 19 in the paper) is:

    du/dt = g(c + kappa)|grad u| + grad u . grad g

where
    g       = stopping function (edge indicator), e.g. 1/(1 + |grad I_hat|^p)
    kappa   = curvature of the level sets = div(grad u / |grad u|)
    c       = constant balloon velocity (default 0 = pure geodesic)
    grad g  = spatial gradient of g (attracts contour toward edges)

The first term is curvature-driven smoothing modulated by g (slows near
edges).  The second term (nabla g . nabla u) is the geodesic attraction
force that pulls the contour toward boundaries even when g does not reach
zero.  Setting c != 0 adds a balloon force equivalent to an area penalty
in the energy.

The level set function phi is evolved with explicit forward Euler on a
regular grid using central finite differences.

Entry point: ``segment``
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Default parameter dict for geodesic active contour segmentation.

    Sections
    --------
    mdl : model / energy weights
        c           : constant balloon velocity.  0 gives the pure geodesic
                      flow (Eq. 13); nonzero adds an area penalty (Eq. 18).
                      With our convention (phi > 0 inside), positive c
                      expands the contour outward and negative c shrinks it.
                      Note: this is sign-flipped relative to the paper, which
                      uses phi < 0 inside.  (0.0)
        sigma       : std-dev for Gaussian smoothing of the image when
                      computing g internally.  Only used if g is not
                      provided.  (1.5)
        p           : exponent in the stopping function g = 1/(1+|grad I|^p).
                      Only used if g is not provided.  (2)
    opt : solver parameters
        dt          : time step for the explicit Euler update.  (0.1)
        reinit_every: reinitialise phi toward a signed distance function
                      every this many iterations.  0 disables.  (10)
        maxiter     : maximum number of iterations.  (500)
        tol_length  : threshold on |L(t) - L(t-1)| for the termination
                      criterion.  (5.0)
        tol_iters   : number of consecutive iterations the length criterion
                      must hold before stopping.  (10)
    init : initialisation of the level set function phi
        mode        : 'circle' | 'checkerboard' | 'rectangle'
        radius_frac : fraction of min(H,W)/2 used for circle/rectangle.
                      (0.4)
        block_size  : checkerboard block size in pixels.  (30)
    """
    return {
        "mdl": {
            "c":     0.0,
            "sigma": 1.5,
            "p":     2,
        },
        "opt": {
            "dt":           0.1,
            "reinit_every": 10,
            "maxiter":      500,
            "tol_length":   0.5,
            "tol_iters":    10,
        },
        "init": {
            "mode":        "checkerboard",
            "radius_frac": 0.4,
            "block_size":  30,
        },
    }


# ---------------------------------------------------------------------------
# smoothed Heaviside / Dirac  (used only for curve-length measurement)
# ---------------------------------------------------------------------------

def _dirac(z, eps):
    """Smoothed Dirac delta: delta_eps(z) = (1/pi) * eps / (eps^2 + z^2)."""
    return (1.0 / torch.pi) * eps / (eps**2 + z**2)


# ---------------------------------------------------------------------------
# spatial derivative helpers  (central finite differences, h = 1)
# ---------------------------------------------------------------------------

def _grad_x(phi):
    """Central difference in x (column direction), one-sided at boundaries."""
    gx = torch.empty_like(phi)
    gx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / 2.0
    gx[:, 0]    = phi[:, 1] - phi[:, 0]
    gx[:, -1]   = phi[:, -1] - phi[:, -2]
    return gx


def _grad_y(phi):
    """Central difference in y (row direction), one-sided at boundaries."""
    gy = torch.empty_like(phi)
    gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / 2.0
    gy[0, :]    = phi[1, :] - phi[0, :]
    gy[-1, :]   = phi[-1, :] - phi[-2, :]
    return gy


def _grad_magnitude(phi):
    """Magnitude of the spatial gradient |grad phi|."""
    gx = _grad_x(phi)
    gy = _grad_y(phi)
    return torch.sqrt(gx**2 + gy**2 + 1e-10)


def _curvature(phi):
    """Curvature kappa = div(grad phi / |grad phi|).

    Uses second-order central differences for phi_xx, phi_yy and the
    cross-derivative phi_xy.  Matches lcv_segment._curvature.
    """
    eta = 1e-8

    phi_x = _grad_x(phi)
    phi_y = _grad_y(phi)

    # second derivatives
    phi_xx = torch.empty_like(phi)
    phi_xx[:, 1:-1] = phi[:, 2:] + phi[:, :-2] - 2.0 * phi[:, 1:-1]
    phi_xx[:, 0]    = phi_xx[:, 1]
    phi_xx[:, -1]   = phi_xx[:, -2]

    phi_yy = torch.empty_like(phi)
    phi_yy[1:-1, :] = phi[2:, :] + phi[:-2, :] - 2.0 * phi[1:-1, :]
    phi_yy[0, :]    = phi_yy[1, :]
    phi_yy[-1, :]   = phi_yy[-2, :]

    phi_xy = torch.empty_like(phi)
    phi_xy[1:-1, 1:-1] = (phi[2:, 2:]   - phi[:-2, 2:] -
                           phi[2:, :-2]  + phi[:-2, :-2]) / 4.0
    phi_xy[0, :]   = phi_xy[1, :]
    phi_xy[-1, :]  = phi_xy[-2, :]
    phi_xy[:, 0]   = phi_xy[:, 1]
    phi_xy[:, -1]  = phi_xy[:, -2]

    numer = (phi_xx * phi_y**2
             - 2.0 * phi_xy * phi_x * phi_y
             + phi_yy * phi_x**2)
    denom = (phi_x**2 + phi_y**2).pow(1.5) + eta

    return numer / denom


# ---------------------------------------------------------------------------
# stopping function (fallback when g is not provided)
# ---------------------------------------------------------------------------

def _compute_stopping_function(image, sigma, p, device, dtype):
    """Compute g = 1 / (1 + |grad I_hat|^p)  (Eq. 17).

    Parameters
    ----------
    image : (H, W) tensor
        Input image.
    sigma : float
        Gaussian smoothing std-dev.
    p : int
        Exponent (1 or 2).

    Returns
    -------
    g : (H, W) tensor
        Stopping function in (0, 1].
    """
    # Gaussian smoothing via conv2d
    if sigma > 0:
        ks = int(4 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        ax = torch.arange(-ks // 2 + 1, ks // 2 + 1,
                           device=device, dtype=dtype)
        kernel_1d = torch.exp(-ax**2 / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # separable: smooth rows then columns
        img = image.unsqueeze(0).unsqueeze(0)
        pad_size = ks // 2
        # row convolution
        k_row = kernel_1d.reshape(1, 1, 1, -1)
        img = F.pad(img, (pad_size, pad_size, 0, 0), mode='replicate')
        img = F.conv2d(img, k_row)
        # column convolution
        k_col = kernel_1d.reshape(1, 1, -1, 1)
        img = F.pad(img, (0, 0, pad_size, pad_size), mode='replicate')
        img = F.conv2d(img, k_col)
        I_hat = img.squeeze(0).squeeze(0)
    else:
        I_hat = image

    grad_mag = _grad_magnitude(I_hat)
    g = 1.0 / (1.0 + grad_mag**p)
    return g


# ---------------------------------------------------------------------------
# initialisation of the level set function phi
# ---------------------------------------------------------------------------

def _init_phi(H, W, cfg_init, device, dtype):
    """Create the initial level set function phi_0 as a signed distance."""
    mode = cfg_init.get("mode", "checkerboard")

    if mode == "circle":
        frac = cfg_init.get("radius_frac", 0.4)
        cy, cx = H / 2.0, W / 2.0
        r = frac * min(H, W) / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij')
        phi = r - torch.sqrt((xx - cx)**2 + (yy - cy)**2)
        return phi

    elif mode == "rectangle":
        frac = cfg_init.get("radius_frac", 0.4)
        margin_y = int((1.0 - frac) * H / 2.0)
        margin_x = int((1.0 - frac) * W / 2.0)
        phi = -2.0 * torch.ones(H, W, device=device, dtype=dtype)
        phi[margin_y:H - margin_y, margin_x:W - margin_x] = 2.0
        return phi

    elif mode == "checkerboard":
        bs = cfg_init.get("block_size", 30)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij')
        phi = torch.sin(torch.pi * xx / bs) * torch.sin(torch.pi * yy / bs)
        return phi

    else:
        raise ValueError(f"Unknown init mode: {mode!r}")


# ---------------------------------------------------------------------------
# curve length
# ---------------------------------------------------------------------------

def _curve_length(phi, eps=1.0):
    """Approximate curve length: integral of delta_eps(phi) |grad phi|."""
    delta = _dirac(phi, eps)
    grad_mag = _grad_magnitude(phi)
    return (delta * grad_mag).sum().item()


# ---------------------------------------------------------------------------
# reinitialisation
# ---------------------------------------------------------------------------

def _reinitialise_phi(phi, n_steps=5, dt_reinit=0.5):
    """Reinitialise phi toward a signed distance function.

    Solves  dphi/dtau = sign(phi_0) * (1 - |grad phi|)  for *n_steps*
    pseudo-time steps using first-order upwind differences
    (Sussman et al. 1994).
    """
    phi0 = phi.clone()
    sign_phi = phi0 / torch.sqrt(phi0**2 + 1.0)

    for _ in range(n_steps):
        # forward / backward differences along each axis
        Dxm = phi - torch.roll(phi, 1, dims=1)
        Dxp = torch.roll(phi, -1, dims=1) - phi
        Dym = phi - torch.roll(phi, 1, dims=0)
        Dyp = torch.roll(phi, -1, dims=0) - phi

        # Neumann BC
        Dxm[:, 0]  = 0.0;  Dxp[:, -1] = 0.0
        Dym[0, :]  = 0.0;  Dyp[-1, :] = 0.0

        # Godunov upwind
        Gp = torch.sqrt(
            torch.clamp(Dxm, min=0.0)**2 + torch.clamp(-Dxp, min=0.0)**2 +
            torch.clamp(Dym, min=0.0)**2 + torch.clamp(-Dyp, min=0.0)**2
        )
        Gm = torch.sqrt(
            torch.clamp(-Dxm, min=0.0)**2 + torch.clamp(Dxp, min=0.0)**2 +
            torch.clamp(-Dym, min=0.0)**2 + torch.clamp(Dyp, min=0.0)**2
        )

        S_pos = sign_phi.clamp(min=0.0)
        S_neg = (-sign_phi).clamp(min=0.0)

        grad_mag = S_pos * Gp + S_neg * Gm
        phi = phi - dt_reinit * sign_phi * (grad_mag - 1.0)

    return phi


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def segment(g=None, image=None, cfg=None, *, phi_init=None,
            device=None, dtype=torch.float64,
            verbose=False, return_numpy=True, **overrides):
    """Run geodesic active contour level-set segmentation.

    Parameters
    ----------
    g : (H, W) numpy array or torch.Tensor, optional
        Precomputed stopping function (edge indicator in (0, 1]).
        If provided, this is used directly and ``image`` is ignored
        for the PDE.  The spatial gradient nabla g is computed
        numerically from this array.
    image : (H, W) numpy array, optional
        Input grayscale image.  Used to compute g internally
        (via Gaussian smoothing + gradient + Eq. 17) only when
        ``g`` is not provided.
    cfg : dict, optional
        Config dict shaped like ``default_config()``.
    phi_init : (H, W) numpy array or torch.Tensor, optional
        Custom initial level set function.  If None, one is created
        from ``cfg["init"]``.
    device : str | torch.device | None
        Defaults to CUDA if available.
    dtype : torch.dtype
        Computation dtype (float64 recommended for PDE stability).
    verbose : bool
        Print per-iteration diagnostics.
    return_numpy : bool
        If True return numpy arrays; else torch tensors.
    **overrides
        Flat keyword overrides applied into the nested cfg dict.

    Returns
    -------
    result : dict
        phi       : final level set function  (H, W)
        mask      : binary segmentation  (H, W), True where phi >= 0
        cfg       : config dict used
        history   : list of per-iteration records
        converged : bool, whether the termination criterion was met
        n_iters   : number of iterations actually run
    """
    if g is None and image is None:
        raise ValueError("At least one of g or image must be provided.")

    # ---- device / dtype setup ----
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

    # ---- config ----
    cfg = deepcopy(default_config()) if cfg is None else deepcopy(cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    c_balloon    = cfg["mdl"]["c"]
    sigma        = cfg["mdl"]["sigma"]
    p_exp        = cfg["mdl"]["p"]
    dt           = cfg["opt"]["dt"]
    maxiter      = int(cfg["opt"]["maxiter"])
    tol_L        = cfg["opt"]["tol_length"]
    tol_it       = int(cfg["opt"]["tol_iters"])
    reinit_every = int(cfg["opt"].get("reinit_every", 0))

    # ---- prepare g ----
    if g is not None:
        g_t = to_t(g)
    else:
        u0 = to_t(image)
        if u0.ndim == 3:
            u0 = u0.mean(dim=-1)
        g_t = _compute_stopping_function(u0, sigma, p_exp, device, dtype)

    assert g_t.ndim == 2, "g must be 2-D (H, W)"
    H, W = g_t.shape

    # precompute grad g (constant throughout the evolution)
    grad_g_x = _grad_x(g_t)
    grad_g_y = _grad_y(g_t)

    # ---- initialise phi ----
    if phi_init is not None:
        phi = to_t(phi_init)
    else:
        phi = _init_phi(H, W, cfg["init"], device, dtype)

    # ---- iteration loop ----
    history = []
    converged = False
    consec = 0
    prev_length = None

    for it in range(maxiter):
        # -- curvature and gradient of phi --
        kappa = _curvature(phi)
        phi_x = _grad_x(phi)
        phi_y = _grad_y(phi)
        grad_mag = torch.sqrt(phi_x**2 + phi_y**2 + 1e-10)

        # -- geodesic PDE velocity --
        # Term 1: g * (c + kappa) * |grad phi|
        velocity = g_t * (c_balloon + kappa) * grad_mag

        # Term 2: grad phi . grad g
        velocity = velocity + (phi_x * grad_g_x + phi_y * grad_g_y)

        # -- explicit Euler step --
        phi = phi + dt * velocity

        # -- periodic reinitialisation --
        if reinit_every > 0 and (it + 1) % reinit_every == 0:
            phi = _reinitialise_phi(phi)

        # -- termination criterion (curve length stability) --
        cur_length = _curve_length(phi)

        if prev_length is not None:
            if abs(cur_length - prev_length) <= tol_L:
                consec += 1
            else:
                consec = 0
            if consec >= tol_it:
                converged = True

        prev_length = cur_length

        # -- record --
        if verbose and (it % 50 == 0 or it == maxiter - 1 or converged):
            mask_count = int((phi >= 0).sum().item())
            print(f"iter {it+1:4d}/{maxiter}  length={cur_length:.1f}  "
                  f"area={mask_count}")

        history.append({
            "iter":   it,
            "length": cur_length,
        })

        if converged:
            if verbose:
                print(f"Converged at iteration {it+1} "
                      f"(length stable for {tol_it} iters)")
            break

    # ---- build output ----
    mask = phi >= 0

    def maybe_np(t):
        return t.detach().cpu().numpy() if return_numpy else t

    return {
        "phi":       maybe_np(phi),
        "mask":      maybe_np(mask),
        "g":         maybe_np(g_t),
        "cfg":       cfg,
        "history":   history,
        "converged": converged,
        "n_iters":   it + 1,
    }


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

def _apply_overrides(cfg, overrides):
    """Route flat keyword overrides into the nested cfg dict."""
    sections = ("mdl", "opt", "init")
    for k, v in overrides.items():
        placed = False
        for sec in sections:
            if k in cfg.get(sec, {}):
                cfg[sec][k] = v
                placed = True
                break
        if not placed:
            raise KeyError(f"unknown override: {k!r}")
    return cfg
