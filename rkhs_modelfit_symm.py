"""
Symmetric-edge variant of rkhs_modelfit_torch.py.

Key changes from the original:
  1. The edge basis Psi uses a *symmetric* arctan:
         psi(t) = (1/pi) * arctan(t / delta) * exp(-edge_decay * t^2)
     instead of the offset Heaviside approximation 0.5 + (1/pi)*arctan(...).
     This basis is odd-symmetric about the edge line (positive on one side,
     negative on the other), which makes it a true *edge detector* rather
     than a step-function approximation.

  2. Post-fit basis swap:  after fitting with the symmetric arctan basis you
     can replace Psi with a localised *bump* basis (Gaussian or ridge) while
     keeping the fitted beta coefficients.  This lets you
       - fit   with arctan edges  (good gradient signal for edge detection)
       - render with Gaussian bumps (localised ridges highlighting edges)

  3. A ``reconstruct_with_new_basis`` helper rebuilds the Psi*beta image
     using an alternative Psi matrix (bump, ridge, or custom), so you don't
     have to re-run the optimisation.

Public API
----------
fit_rkhs_decomposition      -- same signature as rkhs_modelfit_torch
build_bump_basis             -- build a Gaussian-bump or ridge Psi matrix
reconstruct_with_new_basis   -- rebuild Pb image from fitted beta + new Psi
"""

from copy import deepcopy

import numpy as np
import torch


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Same structure as rkhs_modelfit_torch.default_config with one addition:

    bss.edge_type : str
        "symmetric_arctan"  (default) -- (1/pi)*arctan(Z) * Gaussian envelope
        "arctan"            -- original 0.5 + (1/pi)*arctan(Z) * Gaussian envelope
    """
    return {
        "mdl": {
            "gamma_smoothpen":  1e-6,
            "alpha_edgesparse": 1.0,
            "nu_tvweight":      1e-3,
            "iota_edgegate":    1e3,
        },
        "bss": {
            "sigma_kerwidth":  12.0,
            "delta_rampwidth": 1e-4,
            "ell_numdirs":     12,
            "num_offsets":     None,
            "edge_decay":      10.0,
            # NEW: controls the shape of the edge basis
            "edge_type":       "symmetric_arctan",
        },
        "ptc": {
            "patchsize": 4,
            "overlap":   3,
        },
        "opt": {
            "rho1_betasplit":        2.0,
            "rho2_Wsplit":           1.0,
            "rho3_gradsplit":        1.0,
            "zeta1_dprox":           1e-9,
            "zeta2_betaprox":        10.0,
            "zeta2_betaprox_safety": 1.05,
            "beta_extrap_omega":     0.0,
            "maxiter":               30,
        },
    }


# ---------------------------------------------------------------------------
# basis construction
# ---------------------------------------------------------------------------

def _build_gaussian_kernel_matrix_np(n_gridx, n_gridy, sigma):
    nx_denom = max(n_gridx - 1, 1)
    ny_denom = max(n_gridy - 1, 1)
    tx = np.arange(n_gridx) / nx_denom
    ty = np.arange(n_gridy) / ny_denom

    X = np.tile(tx, n_gridy)
    Y = np.repeat(ty, n_gridx)

    a = X[:, None] - X[None, :]
    b = Y[:, None] - Y[None, :]
    tao2 = a * a + b * b
    K = np.exp(-tao2 / (2.0 * sigma * sigma))
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return (coeff * coeff) * K


def _build_edge_basis_np(n_gridx, n_gridy, delta, num_dirs,
                         num_offsets=None, edge_decay=None,
                         edge_type="symmetric_arctan"):
    """Build the edge basis matrix Psi.

    Parameters
    ----------
    edge_type : str
        "symmetric_arctan" -- (1/pi)*arctan(Z)  (odd-symmetric about edge)
        "arctan"           -- 0.5 + (1/pi)*arctan(Z)  (original Heaviside approx)

    The Gaussian envelope  exp(-edge_decay * t^2)  is applied in both cases
    when edge_decay > 0.

    Returns
    -------
    Psi : ndarray, shape (n_gridx*n_gridy, num_dirs*num_offsets)
    T_unscaled : ndarray, same shape -- raw signed distances (kept for
        building alternative bases on the same edge geometry)
    """
    nx_denom = max(n_gridx - 1, 1)
    ny_denom = max(n_gridy - 1, 1)
    tx = np.arange(n_gridx) / nx_denom
    ty = np.arange(n_gridy) / ny_denom

    X = np.tile(tx, n_gridy)
    Y = np.repeat(ty, n_gridx)

    if num_offsets is None:
        num_offsets = n_gridx * n_gridy

    theta = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
    c = np.arange(num_offsets) / max(num_offsets - 1, 1)

    C_all = np.tile(c, num_dirs)
    Theta_all = np.repeat(theta, num_offsets)

    cos_t = np.cos(Theta_all)
    sin_t = np.sin(Theta_all)
    # Unscaled signed distance from the edge line
    T_unscaled = (cos_t[None, :] * X[:, None]
                  + sin_t[None, :] * Y[:, None]
                  + C_all[None, :])
    Z = T_unscaled / delta

    if edge_type == "symmetric_arctan":
        H = (1.0 / np.pi) * np.arctan(Z)
    elif edge_type == "arctan":
        H = 0.5 + (1.0 / np.pi) * np.arctan(Z)
    else:
        raise ValueError(f"Unknown edge_type: {edge_type!r}. "
                         f"Use 'symmetric_arctan' or 'arctan'.")

    if edge_decay and edge_decay > 0:
        H = H * np.exp(-edge_decay * T_unscaled ** 2)

    return H, T_unscaled


# ---------------------------------------------------------------------------
# bump / ridge basis builders  (post-fit swap)
# ---------------------------------------------------------------------------

def build_bump_basis(n_gridx, n_gridy, num_dirs, num_offsets=None,
                     bump_type="gaussian", bump_width=10.0):
    """Build an alternative Psi matrix using localised bump functions.

    The bump basis lives on the *same* grid of edge lines (angles x offsets)
    as the fitting basis, so the fitted beta coefficients can be reused
    directly.

    Parameters
    ----------
    n_gridx, n_gridy : int
        Patch dimensions (must match the fitting basis).
    num_dirs : int
        Number of edge directions (must match fitting basis).
    num_offsets : int or None
        Number of offset positions per direction (None -> n_gridx*n_gridy).
    bump_type : str
        "gaussian"  -- exp(-bump_width * t^2)
                       Symmetric Gaussian bump centred on the edge line.
        "ridge"     -- exp(-bump_width * |t|)
                       Laplacian / ridge profile (sharper peak, heavier tails).
    bump_width : float
        Controls the width of the bump.  Larger values = narrower bump.
        For "gaussian" this is the inverse-variance parameter in the exponent.
        For "ridge" this is the inverse scale.

    Returns
    -------
    Psi_bump : ndarray, shape (n_gridx*n_gridy, num_dirs*num_offsets)
    """
    nx_denom = max(n_gridx - 1, 1)
    ny_denom = max(n_gridy - 1, 1)
    tx = np.arange(n_gridx) / nx_denom
    ty = np.arange(n_gridy) / ny_denom

    X = np.tile(tx, n_gridy)
    Y = np.repeat(ty, n_gridx)

    if num_offsets is None:
        num_offsets = n_gridx * n_gridy

    theta = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
    c = np.arange(num_offsets) / max(num_offsets - 1, 1)

    C_all = np.tile(c, num_dirs)
    Theta_all = np.repeat(theta, num_offsets)

    cos_t = np.cos(Theta_all)
    sin_t = np.sin(Theta_all)
    T_unscaled = (cos_t[None, :] * X[:, None]
                  + sin_t[None, :] * Y[:, None]
                  + C_all[None, :])

    if bump_type == "gaussian":
        Psi_bump = np.exp(-bump_width * T_unscaled ** 2)
    elif bump_type == "ridge":
        Psi_bump = np.exp(-bump_width * np.abs(T_unscaled))
    else:
        raise ValueError(f"Unknown bump_type: {bump_type!r}. "
                         f"Use 'gaussian' or 'ridge'.")

    return Psi_bump


# ---------------------------------------------------------------------------
# post-fit reconstruction with a swapped basis
# ---------------------------------------------------------------------------

def reconstruct_with_new_basis(result, Psi_new, *, abs_beta=True,
                               return_numpy=True):
    """Rebuild the Psi*beta component using a new basis matrix.

    Parameters
    ----------
    result : dict
        Output of ``fit_rkhs_decomposition``.
    Psi_new : ndarray, shape (ps^2, n_basis)
        Alternative basis matrix (e.g. from ``build_bump_basis``).
        Must have the same number of columns as the original Psi.
    abs_beta : bool
        If True (default), use ``|beta|`` instead of ``beta`` when
        multiplying by the new basis.  This makes sense when Psi_new is
        a positive bump/ridge basis and you want every detected edge to
        produce a positive response regardless of the sign the symmetric
        arctan basis assigned during fitting.
    return_numpy : bool
        If True, return numpy arrays; else torch tensors.

    Returns
    -------
    dict with keys:
        "Pb_new"  -- the new Psi_new @ |beta| image  (H, W)
        "M_new"   -- Kd + Pb_new                      (H, W)
        "Kd"      -- unchanged smooth component       (H, W)
    """
    cfg = result["cfg"]
    ps = cfg["ptc"]["patchsize"]
    overlap = cfg["ptc"]["overlap"]

    # Get beta in (n2, P) layout and Kd image
    beta = result["beta"]    # (n2, P) if return_numpy was True at fit time
    Kd   = result["Kd"]      # (H, W)

    if isinstance(beta, torch.Tensor):
        beta_np = beta.detach().cpu().numpy()
        Kd_np   = Kd.detach().cpu().numpy()
    else:
        beta_np = np.asarray(beta)
        Kd_np   = np.asarray(Kd)

    if abs_beta:
        beta_np = np.abs(beta_np)

    Psi_new = np.asarray(Psi_new)
    n2_new = Psi_new.shape[1]
    n2_old = beta_np.shape[0]
    if n2_new != n2_old:
        raise ValueError(
            f"Psi_new has {n2_new} columns but beta has {n2_old} rows. "
            f"They must match (same num_dirs * num_offsets).")

    im_h, im_w = Kd_np.shape

    # Rebuild patch layout to scatter patches back to the image
    layout = _build_patch_layout(im_h, im_w, ps, overlap)
    grid_x = layout["grid_x"]
    grid_y = layout["grid_y"]

    # Reconstruct Pb image by scattering patches
    Pb_img = np.zeros((im_h, im_w), dtype=np.float64)
    cnt    = np.zeros((im_h, im_w), dtype=np.float64)

    pc = 0
    for xx in grid_x:
        for yy in grid_y:
            # beta_np[:, pc] is the coefficient vector for patch pc
            pb_patch = Psi_new @ beta_np[:, pc]   # (ps^2,)
            # Reshape in F-order to match the original patch extraction
            pb_2d = pb_patch.reshape(ps, ps, order='F')
            Pb_img[xx:xx+ps, yy:yy+ps] += pb_2d
            cnt[xx:xx+ps, yy:yy+ps]    += 1.0
            pc += 1

    cnt = np.maximum(cnt, 1.0)
    Pb_img /= cnt

    M_new = Kd_np + Pb_img

    if return_numpy:
        return {"Pb_new": Pb_img, "M_new": M_new, "Kd": Kd_np}
    else:
        device = result.get("diagnostics", {}).get("device", "cpu")
        return {
            "Pb_new": torch.as_tensor(Pb_img, device=device),
            "M_new":  torch.as_tensor(M_new,  device=device),
            "Kd":     torch.as_tensor(Kd_np,  device=device),
        }


# ---------------------------------------------------------------------------
# patch layout
# ---------------------------------------------------------------------------

def _build_patch_layout(im_height, im_width, patch_size, overlap):
    if patch_size > im_height or patch_size > im_width:
        raise ValueError("patch_size must not exceed image dimensions")
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("overlap must be strictly less than patch_size")

    def make_grid(N, ps):
        upper = N - (N % step + 1 + step)
        if upper < 1:
            grid = []
        else:
            grid = list(range(1, upper + 1, step))
        forbidden = set(range(N - ps + 1, N + 1))
        grid = [g for g in grid if g not in forbidden]
        grid.append(N - ps + 1)
        return grid

    gx = np.array([g - 1 for g in make_grid(im_height, patch_size)], dtype=np.int64)
    gy = np.array([g - 1 for g in make_grid(im_width,  patch_size)], dtype=np.int64)
    return {"step": step, "grid_x": gx, "grid_y": gy,
            "num_patches": len(gx) * len(gy)}


# ---------------------------------------------------------------------------
# torch helpers
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


def _forward_grad(F):
    Fx = torch.roll(F, -1, dims=-1) - F
    Fy = torch.roll(F, -1, dims=-2) - F
    return Fx, Fy


# ---------------------------------------------------------------------------
# cache / state
# ---------------------------------------------------------------------------

def _build_cache(im_shape, cfg, device, dtype):
    im_h, im_w = im_shape
    ps      = cfg["ptc"]["patchsize"]
    overlap = cfg["ptc"]["overlap"]
    sigma_k = cfg["bss"]["sigma_kerwidth"]
    delta   = cfg["bss"]["delta_rampwidth"]
    L       = cfg["bss"]["ell_numdirs"]
    n_off   = cfg["bss"].get("num_offsets", None)
    e_decay = cfg["bss"].get("edge_decay", None)
    e_type  = cfg["bss"].get("edge_type", "symmetric_arctan")

    gamma = cfg["mdl"]["gamma_smoothpen"]
    rho2  = cfg["opt"]["rho2_Wsplit"]
    zeta1 = cfg["opt"]["zeta1_dprox"]
    zeta2_user   = cfg["opt"]["zeta2_betaprox"]
    zeta2_safety = cfg["opt"].get("zeta2_betaprox_safety", 0.0)

    # --- float64 numpy build ---
    K_np = _build_gaussian_kernel_matrix_np(ps, ps, sigma_k)
    Psi_np, T_unscaled_np = _build_edge_basis_np(
        ps, ps, delta, L, num_offsets=n_off,
        edge_decay=e_decay, edge_type=e_type)

    Kt_np    = K_np.T
    Psit_np  = Psi_np.T
    KtPsi_np = Kt_np @ Psi_np
    n1 = K_np.shape[1]
    n2 = Psi_np.shape[1]
    A_d_np = (1.0 + rho2) * (Kt_np @ K_np) + 2.0 * gamma * K_np + zeta1 * np.eye(n1)
    A_d_inv_np = np.linalg.inv(A_d_np)

    K_op   = float(np.linalg.norm(K_np, ord=2))
    PtP_op = float(np.linalg.norm(Psit_np @ Psi_np, ord=2))
    L_beta = (1.0 + rho2) * PtP_op
    if zeta2_safety > 0.0:
        zeta2_eff = max(zeta2_user, zeta2_safety * L_beta)
    else:
        zeta2_eff = zeta2_user

    # --- cast onto device ---
    def t(x):
        return torch.as_tensor(x, dtype=dtype, device=device)
    K     = t(K_np)
    Psi   = t(Psi_np)
    Kt    = t(Kt_np)
    Psit  = t(Psit_np)
    KtPsi = t(KtPsi_np)
    A_d_inv = t(A_d_inv_np)

    # --- patch layout + flat indices ---
    layout = _build_patch_layout(im_h, im_w, ps, overlap)
    grid_x = torch.as_tensor(layout["grid_x"], dtype=torch.long, device=device)
    grid_y = torch.as_tensor(layout["grid_y"], dtype=torch.long, device=device)

    top_left = (grid_x[:, None] * im_w + grid_y[None, :]).reshape(-1)

    k_range = torch.arange(ps * ps, dtype=torch.long, device=device)
    offsets = (k_range % ps) * im_w + (k_range // ps)

    patch_flat_idx = top_left[:, None] + offsets[None, :]
    patch_flat_idx_flat = patch_flat_idx.reshape(-1)

    cnt = torch.zeros(im_h * im_w, dtype=dtype, device=device)
    cnt.index_add_(0, patch_flat_idx_flat,
                   torch.ones_like(patch_flat_idx_flat, dtype=dtype))
    cnt = cnt.reshape(im_h, im_w).clamp(min=1.0)

    # --- FFT operators ---
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    Dx_otf  = _psf2otf(np.array([[1.0, -1.0]]),  (im_h, im_w),
                       device=device, complex_dtype=complex_dtype)
    Dy_otf  = _psf2otf(np.array([[1.0], [-1.0]]), (im_h, im_w),
                       device=device, complex_dtype=complex_dtype)
    Dx_conj = torch.conj(Dx_otf)
    Dy_conj = torch.conj(Dy_otf)
    Lap_otf = (Dx_otf * Dx_conj + Dy_otf * Dy_conj).real

    return {
        "im_h": im_h, "im_w": im_w, "ps": ps,
        "K": K, "Kt": Kt, "Psi": Psi, "Psit": Psit, "KtPsi": KtPsi,
        "n1": n1, "n2": n2,
        "A_d_inv": A_d_inv,
        "K_op": K_op, "PtP_op": PtP_op, "L_beta": L_beta, "zeta2_eff": zeta2_eff,
        "layout": layout,
        "patch_flat_idx":      patch_flat_idx,
        "patch_flat_idx_flat": patch_flat_idx_flat,
        "patch_count":         cnt,
        "Dx_otf": Dx_otf, "Dy_otf": Dy_otf,
        "Dx_conj": Dx_conj, "Dy_conj": Dy_conj,
        "Lap_otf": Lap_otf,
        "device": device, "dtype": dtype, "complex_dtype": complex_dtype,
    }


def _initialize_state(image_t, cache):
    im_h, im_w = cache["im_h"], cache["im_w"]
    n_patches  = cache["layout"]["num_patches"]
    n1, n2     = cache["n1"], cache["n2"]
    device, dtype = cache["device"], cache["dtype"]

    z = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)
    return {
        "patch": {
            "d":         z(n_patches, n1),
            "beta":      z(n_patches, n2),
            "beta_prev": z(n_patches, n2),
            "theta":     z(n_patches, n2),
            "b1":        z(n_patches, n2),
        },
        "img": {
            "Wrec": image_t.clone(),
            "b2":   z(im_h, im_w),
            "vx":   z(im_h, im_w),
            "vy":   z(im_h, im_w),
            "b3x":  z(im_h, im_w),
            "b3y":  z(im_h, im_w),
            "Kd":   z(im_h, im_w),
            "Pb":   z(im_h, im_w),
        },
    }


# ---------------------------------------------------------------------------
# iteration updates
# ---------------------------------------------------------------------------

def _patch_update(image_t, state, cache, cfg):
    im_h, im_w = cache["im_h"], cache["im_w"]
    ps         = cache["ps"]

    K       = cache["K"]
    Psi     = cache["Psi"]
    KtPsi   = cache["KtPsi"]
    A_inv   = cache["A_d_inv"]
    idx     = cache["patch_flat_idx"]
    cnt     = cache["patch_count"]

    alpha = cfg["mdl"]["alpha_edgesparse"]
    nu    = cfg["mdl"]["nu_tvweight"]
    iota  = cfg["mdl"]["iota_edgegate"]
    rho1  = cfg["opt"]["rho1_betasplit"]
    rho2  = cfg["opt"]["rho2_Wsplit"]
    zeta1 = cfg["opt"]["zeta1_dprox"]
    zeta2 = cache["zeta2_eff"]
    omega = float(cfg["opt"].get("beta_extrap_omega", 0.0))

    Wrec = state["img"]["Wrec"]
    b2   = state["img"]["b2"]
    vx   = state["img"]["vx"]
    vy   = state["img"]["vy"]

    d_batch     = state["patch"]["d"]
    beta_batch  = state["patch"]["beta"]
    beta_prev   = state["patch"]["beta_prev"]
    theta_batch = state["patch"]["theta"]
    b1_batch    = state["patch"]["b1"]

    flat_im = image_t.reshape(-1)
    flat_W  = Wrec    .reshape(-1)
    flat_b2 = b2      .reshape(-1)
    flat_vx = vx      .reshape(-1)
    flat_vy = vy      .reshape(-1)

    im_b = flat_im[idx]
    W_b  = flat_W [idx]
    b2_b = flat_b2[idx]
    vx_b = flat_vx[idx]
    vy_b = flat_vy[idx]
    v_norm_b = vx_b.abs() + vy_b.abs()

    # d-update
    d_rhs = (im_b @ K
             - (1.0 + rho2) * (beta_batch @ KtPsi.T)
             + zeta1 * d_batch
             + rho2 * ((W_b + b2_b) @ K))
    d_new = d_rhs @ A_inv
    kd_b  = d_new @ K

    # beta prox-linear
    beta_hat     = beta_batch + omega * (beta_batch - beta_prev)
    psi_beta_hat = beta_hat @ Psi.T
    g_hat        = 1.0 / (1.0 + iota * psi_beta_hat.pow(2))

    residual_b = im_b - (kd_b + psi_beta_hat)
    p_data = -(residual_b @ Psi)
    p_Wsp  = -rho2 * ((W_b - (kd_b + psi_beta_hat) + b2_b) @ Psi)
    p_tv   = -2.0 * nu * iota * ((v_norm_b * g_hat.pow(2) * psi_beta_hat) @ Psi)
    p_hat  = p_data + p_Wsp + p_tv

    beta_new  = (rho1 * (theta_batch + b1_batch)
                 + zeta2 * beta_hat
                 - p_hat) / (rho1 + zeta2)
    theta_new = _soft_threshold(beta_new - b1_batch, alpha / rho1)
    b1_new    = b1_batch + theta_new - beta_new

    state["patch"]["beta_prev"] = beta_batch
    state["patch"]["d"]         = d_new
    state["patch"]["beta"]      = beta_new
    state["patch"]["theta"]     = theta_new
    state["patch"]["b1"]        = b1_new

    # scatter-add
    pb_b = beta_new @ Psi.T
    idx_flat = cache["patch_flat_idx_flat"]
    Kd_img = torch.zeros(im_h * im_w, dtype=cache["dtype"], device=cache["device"])
    Pb_img = torch.zeros_like(Kd_img)
    Kd_img.index_add_(0, idx_flat, kd_b.reshape(-1))
    Pb_img.index_add_(0, idx_flat, pb_b.reshape(-1))
    Kd_img = Kd_img.reshape(im_h, im_w) / cnt
    Pb_img = Pb_img.reshape(im_h, im_w) / cnt

    state["img"]["Kd"] = Kd_img
    state["img"]["Pb"] = Pb_img
    return Kd_img, Pb_img


def _W_update(state, cache, cfg):
    rho2 = cfg["opt"]["rho2_Wsplit"]
    rho3 = cfg["opt"]["rho3_gradsplit"]
    Dx_conj = cache["Dx_conj"]
    Dy_conj = cache["Dy_conj"]
    Lap_otf = cache["Lap_otf"]

    Kd  = state["img"]["Kd"]
    Pb  = state["img"]["Pb"]
    b2  = state["img"]["b2"]
    vx  = state["img"]["vx"]
    vy  = state["img"]["vy"]
    b3x = state["img"]["b3x"]
    b3y = state["img"]["b3y"]

    fft2  = torch.fft.fft2
    ifft2 = torch.fft.ifft2

    eps = torch.finfo(cache["dtype"]).eps
    rhs   = (rho2 * fft2(Kd + Pb - b2)
             + rho3 * (Dx_conj * fft2(vx + b3x)
                       + Dy_conj * fft2(vy + b3y)))
    denom = rho2 + rho3 * Lap_otf + eps
    W_new = torch.real(ifft2(rhs / denom))

    state["img"]["Wrec"] = W_new
    return W_new


def _v_and_bregman_update(state, cache, cfg):
    rho3 = cfg["opt"]["rho3_gradsplit"]
    nu   = cfg["mdl"]["nu_tvweight"]
    iota = cfg["mdl"]["iota_edgegate"]

    Wrec = state["img"]["Wrec"]
    Kd   = state["img"]["Kd"]
    Pb   = state["img"]["Pb"]
    b2   = state["img"]["b2"]
    b3x  = state["img"]["b3x"]
    b3y  = state["img"]["b3y"]

    g = 1.0 / (1.0 + iota * Pb.pow(2))

    Wx, Wy = _forward_grad(Wrec)
    thresh = (nu / rho3) * g
    vx_new = _soft_threshold(Wx - b3x, thresh)
    vy_new = _soft_threshold(Wy - b3y, thresh)

    state["img"]["b3x"] = b3x + vx_new - Wx
    state["img"]["b3y"] = b3y + vy_new - Wy
    state["img"]["b2"]  = b2  + Wrec   - (Kd + Pb)
    state["img"]["vx"]  = vx_new
    state["img"]["vy"]  = vy_new


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fit_rkhs_decomposition(
    image, cfg=None, *,
    device=None, dtype=torch.float32, verbose=False,
    return_numpy=True, **overrides,
):
    """Fit the RKHS + symmetric-edge decomposition.

    Same signature and return dict as rkhs_modelfit_torch.fit_rkhs_decomposition,
    with the addition that ``cfg["bss"]["edge_type"]`` selects the basis shape.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if isinstance(image, torch.Tensor):
        image_t = image.to(device=device, dtype=dtype)
    else:
        image_t = torch.as_tensor(np.asarray(image, dtype=np.float64),
                                  dtype=dtype, device=device)
    if image_t.ndim != 2:
        raise ValueError("image must be 2D (H, W)")

    cfg = deepcopy(default_config()) if cfg is None else deepcopy(cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    cache = _build_cache(image_t.shape, cfg, device, dtype)
    state = _initialize_state(image_t, cache)

    if verbose:
        print(f"[stage1-symm] device={device} dtype={dtype} "
              f"edge_type={cfg['bss'].get('edge_type', 'symmetric_arctan')!r} "
              f"||Psi^T Psi||_op={cache['PtP_op']:.3f} "
              f"L_beta={cache['L_beta']:.3f} "
              f"zeta2_eff={cache['zeta2_eff']:.3f} "
              f"(user={cfg['opt']['zeta2_betaprox']:.3f}, "
              f"safety={cfg['opt'].get('zeta2_betaprox_safety', 0.0)})")

    n_iter = int(cfg["opt"]["maxiter"])
    history = []

    for it in range(n_iter):
        _patch_update(image_t, state, cache, cfg)
        _W_update(state, cache, cfg)
        _v_and_bregman_update(state, cache, cfg)

        Kd = state["img"]["Kd"]
        Pb = state["img"]["Pb"]
        M  = Kd + Pb
        residual = float(torch.linalg.norm(image_t - M).item())

        if return_numpy:
            history.append({
                "iter":     it,
                "Kd":       Kd.detach().cpu().numpy().copy(),
                "Pb":       Pb.detach().cpu().numpy().copy(),
                "M":        M.detach().cpu().numpy().copy(),
                "residual": residual,
            })
        else:
            history.append({
                "iter":     it,
                "Kd":       Kd.clone(),
                "Pb":       Pb.clone(),
                "M":        M.clone(),
                "residual": residual,
            })

        if verbose:
            beta_l1 = float(torch.sum(torch.abs(state["patch"]["beta"])).item())
            print(f"iter {it+1:3d}/{n_iter}  "
                  f"residual={residual:.6f}  |beta|_1={beta_l1:.6f}")

    Kd_out = state["img"]["Kd"]
    Pb_out = state["img"]["Pb"]
    d_out    = state["patch"]["d"].t()
    beta_out = state["patch"]["beta"].t()

    matrices_t = {
        "K":       cache["K"],
        "Psi":     cache["Psi"],
        "A_d_inv": cache["A_d_inv"],
        "KtPsi":   cache["KtPsi"],
    }

    if return_numpy:
        result = {
            "Kd":       Kd_out.detach().cpu().numpy(),
            "Psi_beta": Pb_out.detach().cpu().numpy(),
            "M":        (Kd_out + Pb_out).detach().cpu().numpy(),
            "d":        d_out.detach().cpu().numpy(),
            "beta":     beta_out.detach().cpu().numpy(),
            "matrices": {k: v.detach().cpu().numpy() for k, v in matrices_t.items()},
        }
    else:
        result = {
            "Kd":       Kd_out,
            "Psi_beta": Pb_out,
            "M":        Kd_out + Pb_out,
            "d":        d_out,
            "beta":     beta_out,
            "matrices": matrices_t,
        }
    result.update({
        "cfg":     cfg,
        "history": history,
        "diagnostics": {
            "K_op":      cache["K_op"],
            "PtP_op":    cache["PtP_op"],
            "L_beta":    cache["L_beta"],
            "zeta2_eff": cache["zeta2_eff"],
            "device":    str(device),
            "dtype":     str(dtype),
        },
    })
    return result


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

def _apply_overrides(cfg, overrides):
    sections = ("mdl", "bss", "ptc", "opt")
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
