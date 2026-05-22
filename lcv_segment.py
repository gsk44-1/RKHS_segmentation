"""
Local Chan-Vese (LCV) segmentation — PyTorch implementation.

Based on:
  Wang, Huang & Xu, "An efficient local Chan-Vese model for image
  segmentation", Pattern Recognition 43 (2010) 603-618.

This implements the LCV model *without* the extended structure tensor (EST)
texture component.  The energy functional is:

    E_LCV = alpha * E_G  +  beta * E_L  +  E_R

where
    E_G  = global Chan-Vese fitting term  (Eq. 10)
    E_L  = local fitting term on the difference image  g_k * u0 - u0  (Eq. 14)
    E_R  = mu * length_penalty  +  lambda_p * distance_penalty         (Eq. 22, lambda_p=1 in paper)

The level set PDE (Eq. 28a) is evolved with explicit forward Euler on a
regular grid using finite differences (Eq. 34).

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
    """Default parameter dict for LCV segmentation.

    Sections
    --------
    mdl : model / energy weights
        alpha       : weight on the global Chan-Vese term  (1.0)
        beta        : weight on the local term             (1.0 for homogeneous,
                      0.1 for inhomogeneous images)
        mu          : length penalty coefficient, formatted as mu_0 * 255^2
                      with mu_0 in [0, 1].  Default 0.01 * 255^2.
                      Smaller mu detects smaller objects; larger mu detects
                      larger objects.
        lambda_p    : weight on the distance-function penalty P(phi).
                      Fixed at 1 in the paper; exposed here because the
                      effective balance with the data terms shifts with
                      image intensity range.  (1.0)
        k           : window size of the averaging (box-filter) convolution
                      operator in the local term  (15)
    opt : solver parameters
        dt          : time step for the explicit Euler update  (0.1)
        epsilon     : width parameter for the smoothed Heaviside / Dirac
                      approximations  (1.0)
        maxiter     : maximum number of iterations  (500)
        tol_length  : threshold on |L(t) - L(t-1)| for the termination
                      criterion  (5.0)
        tol_iters   : number of consecutive iterations the length criterion
                      must hold before stopping  (10)
    init : initialisation
        mode        : 'circle' | 'checkerboard' | 'rectangle'
        radius_frac : fraction of min(H,W)/2 used for circle radius  (0.4)
        block_size  : checkerboard block size in pixels  (30)
    """
    return {
        "mdl": {
            "alpha":  1.0,
            "beta":   1.0,
            "mu":       0.01 * 255**2,
            "lambda_p": 1.0,
            "k":        15,
        },
        "opt": {
            "dt":         0.1,
            "epsilon":    1.0,
            "maxiter":    500,
            "tol_length": 5.0,
            "tol_iters":  10,
        },
        "init": {
            "mode":        "checkerboard",
            "radius_frac": 0.4,
            "block_size":  30,
        },
    }


# ---------------------------------------------------------------------------
# smoothed Heaviside / Dirac
# ---------------------------------------------------------------------------

def _heaviside(z, eps):
    """Smoothed Heaviside (Eq. 24): H_eps(z) = 0.5*(1 + (2/pi)*arctan(z/eps))"""
    return 0.5 * (1.0 + (2.0 / torch.pi) * torch.atan(z / eps))


def _dirac(z, eps):
    """Smoothed Dirac delta (Eq. 25): delta_eps(z) = (1/pi) * eps / (eps^2 + z^2)"""
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


def _laplacian(phi):
    """5-point discrete Laplacian with Neumann (replicate) boundary."""
    # pad with replicate boundary then apply the [0,1,0; 1,-4,1; 0,1,0] stencil
    p = F.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    lap = (p[:, :, 1:-1, 2:]  + p[:, :, 1:-1, :-2] +
           p[:, :, 2:, 1:-1]  + p[:, :, :-2, 1:-1] -
           4.0 * p[:, :, 1:-1, 1:-1])
    return lap.squeeze(0).squeeze(0)


def _curvature(phi):
    """Curvature kappa = div(grad phi / |grad phi|)  (Eq. 32).

    Uses second-order central differences for phi_xx, phi_yy and the
    cross-derivative phi_xy.
    """
    eta = 1e-8  # small constant to avoid division by zero

    # first derivatives (central)
    phi_x = _grad_x(phi)
    phi_y = _grad_y(phi)

    # second derivatives (central, h=1)
    phi_xx = torch.empty_like(phi)
    phi_xx[:, 1:-1] = phi[:, 2:] + phi[:, :-2] - 2.0 * phi[:, 1:-1]
    phi_xx[:, 0]    = phi_xx[:, 1]
    phi_xx[:, -1]   = phi_xx[:, -2]

    phi_yy = torch.empty_like(phi)
    phi_yy[1:-1, :] = phi[2:, :] + phi[:-2, :] - 2.0 * phi[1:-1, :]
    phi_yy[0, :]    = phi_yy[1, :]
    phi_yy[-1, :]   = phi_yy[-2, :]

    # cross derivative  (Eq. 33 style)
    phi_xy = torch.empty_like(phi)
    phi_xy[1:-1, 1:-1] = (phi[2:, 2:]   - phi[:-2, 2:] -
                           phi[2:, :-2]  + phi[:-2, :-2]) / 4.0
    # replicate at borders
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
# local averaging operator
# ---------------------------------------------------------------------------

def _box_filter(image, k, device, dtype):
    """Apply a k x k box (averaging) filter via conv2d with replicate padding."""
    pad_size = k // 2
    # reshape for conv2d: (1, 1, H, W)
    img_4d = image.unsqueeze(0).unsqueeze(0)
    img_padded = F.pad(img_4d, (pad_size, pad_size, pad_size, pad_size),
                       mode='replicate')
    kernel = torch.ones(1, 1, k, k, device=device, dtype=dtype) / (k * k)
    out = F.conv2d(img_padded, kernel)
    return out.squeeze(0).squeeze(0)


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

def _curve_length(phi, eps):
    """Approximate curve length: integral of delta_eps(phi) |grad phi|."""
    delta = _dirac(phi, eps)
    gx = _grad_x(phi)
    gy = _grad_y(phi)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-10)
    return (delta * grad_mag).sum().item()


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def segment(image, cfg=None, *, phi_init=None,
            device=None, dtype=torch.float64,
            verbose=False, return_numpy=True, **overrides):
    """Run LCV level-set segmentation on a grayscale image.

    Parameters
    ----------
    image : (H, W) numpy array
        Input grayscale image (any numeric dtype; will be cast to float).
    cfg : dict, optional
        Config dict shaped like ``default_config()``.  If None, defaults are
        used.
    phi_init : (H, W) numpy array or torch.Tensor, optional
        Custom initial level set function.  If None, one is created from
        ``cfg["init"]``.
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
        phi      : final level set function  (H, W)
        mask     : binary segmentation  (H, W), True where phi >= 0
        c1, c2   : global inside/outside means at convergence
        d1, d2   : local inside/outside means at convergence
        cfg      : config dict used
        history  : list of per-iteration records
        converged: bool, whether the termination criterion was met
        n_iters  : number of iterations actually run
    """
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

    alpha    = cfg["mdl"]["alpha"]
    beta     = cfg["mdl"]["beta"]
    mu       = cfg["mdl"]["mu"]
    lambda_p = cfg["mdl"]["lambda_p"]
    k        = cfg["mdl"]["k"]
    dt      = cfg["opt"]["dt"]
    eps     = cfg["opt"]["epsilon"]
    maxiter = int(cfg["opt"]["maxiter"])
    tol_L   = cfg["opt"]["tol_length"]
    tol_it  = int(cfg["opt"]["tol_iters"])

    # ---- prepare image tensors ----
    u0 = to_t(image)
    if u0.ndim == 3:
        # convert colour to grayscale
        u0 = u0.mean(dim=-1)
    assert u0.ndim == 2, "image must be 2-D (H, W)"
    H, W = u0.shape

    # difference image for local term
    u0_avg = _box_filter(u0, k, device, dtype)
    diff = u0_avg - u0  # g_k * u0 - u0

    # ---- initialise phi ----
    if phi_init is not None:
        phi = to_t(phi_init)
    else:
        phi = _init_phi(H, W, cfg["init"], device, dtype)

    # ---- precompute smoothed Heaviside ----
    eps_t = torch.tensor(eps, device=device, dtype=dtype)

    # ---- iteration loop ----
    history = []
    converged = False
    consec = 0          # consecutive iterations satisfying length criterion
    prev_length = None

    for it in range(maxiter):
        # -- smoothed Heaviside and Dirac of current phi --
        H_phi = _heaviside(phi, eps_t)
        delta_phi = _dirac(phi, eps_t)

        # -- update region constants (Eq. 27) --
        sum_H     = H_phi.sum() + 1e-10
        sum_1mH   = (1.0 - H_phi).sum() + 1e-10

        c1 = (u0 * H_phi).sum() / sum_H
        c2 = (u0 * (1.0 - H_phi)).sum() / sum_1mH
        d1 = (diff * H_phi).sum() / sum_H
        d2 = (diff * (1.0 - H_phi)).sum() / sum_1mH

        # -- data driving force (inside the delta_eps bracket in Eq. 28a) --
        force_global = alpha * ((u0 - c2)**2 - (u0 - c1)**2)
        force_local  = beta  * ((diff - d2)**2 - (diff - d1)**2)
        data_force   = delta_phi * (force_global + force_local)

        # -- curvature and Laplacian --
        kappa = _curvature(phi)
        lap   = _laplacian(phi)

        # -- length + distance regularisation (Eq. 28a last two terms) --
        # mu * delta(phi) * kappa       = length penalty
        # lambda_p * (lap - kappa)      = distance-function penalty
        reg = mu * delta_phi * kappa + lambda_p * (lap - kappa)

        # -- explicit Euler step --
        phi = phi + dt * (data_force + reg)

        # -- termination criterion (Section 3.5) --
        cur_length = _curve_length(phi, eps_t)

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
            print(f"iter {it+1:4d}/{maxiter}  c1={c1.item():+.4f}  "
                  f"c2={c2.item():+.4f}  length={cur_length:.1f}  "
                  f"area={mask_count}")

        history.append({
            "iter":   it,
            "c1":     c1.item(),
            "c2":     c2.item(),
            "d1":     d1.item(),
            "d2":     d2.item(),
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
        "c1":        c1.item(),
        "c2":        c2.item(),
        "d1":        d1.item(),
        "d2":        d2.item(),
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
