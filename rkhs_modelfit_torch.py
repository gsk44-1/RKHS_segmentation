"""
PyTorch / GPU port of Stage 1 (model fitting) from rkhs_modelfit.py.

Public API mirrors rkhs_modelfit.fit_rkhs_decomposition. The numerical
recipe is the same paper model (8) with the splitting (9). The only
structural change is that the per-iteration patch sweep is vectorised
across patches into batched matmuls, which lets a T4 (or any CUDA GPU)
do all patches in a single dispatch.

Within a single BCD iteration the patches are independent:
  * they READ from shared image-level state (image, Wrec, b2, vx, vy);
  * they WRITE to disjoint columns of d / beta / theta / b1 / beta_prev;
  * they ACCUMULATE into Kd_img / Pb_img via overlap averaging.

So the inner double loop is expressed
here as

    flat_image[idx]                   -> (P, ps^2)   batch extract
    batch @ M.T                       -> (P, n_out)  batched matmul
    flat_image.index_add_(0, idx, .)  -> overlap-aware scatter add

Numpy/MATLAB ``flatten(order='F')`` is mapped onto torch row-major
indexing via a precomputed offset table  ``offsets[k] = (k%ps)*W + (k//ps)``
so the patch element ordering matches how K and Psi were constructed in
the reference implementation.

Tested for numerical parity with rkhs_modelfit.py in float64 on CPU; see
test_modelfit_parity.py.
"""

from copy import deepcopy

import numpy as np
import torch


def test_func(strg):
    print(strg)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Same defaults as rkhs_modelfit.default_config (see that file for the
    meaning of each knob). Duplicated rather than imported so this module
    can be dropped onto a Colab without copying both files."""
    return {
        "mdl": {
            "gamma_smoothpen":  1e-6,
            "alpha_edgesparse": 1.0,
            "nu_tvweight":      1e-3,
            "iota_edgegate":    1e3,
            "nonneg_beta":      False,
        },
        "bss": {
            "sigma_kerwidth":  12.0,
            "delta_rampwidth": 1e-4,
            "ell_numdirs":     12,
            # See rkhs_modelfit.default_config: None -> ps^2 (paper default).
            "num_offsets":     None,
            # Gaussian decay applied to the edge basis functions:
            #   psi(t) = (0.5 + (1/pi)*arctan(t/delta)) * exp(-edge_decay * t^2)
            # where t is the unscaled signed distance from the edge line in
            # normalised [0,1] patch coordinates.  Set to 0 or None to recover
            # the original (non-decaying) Heaviside basis from the paper.
            "edge_decay":      10.0,
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
# basis construction (computed in float64 on CPU, cast to device dtype later)
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


def _build_heaviside_basis_np(n_gridx, n_gridy, delta, num_dirs,
                              num_offsets=None, edge_decay=None):
    """Same as rkhs_modelfit._build_heaviside_basis; see that file for the
    full docstring. ``num_offsets=None`` -> ``n_gridx * n_gridy`` (the paper
    default). Smaller values reduce Psi's column redundancy and thus PtP_op.

    If ``edge_decay`` is a positive number, a Gaussian envelope is applied::

        psi(t) = (0.5 + (1/pi)*arctan(t/delta)) * exp(-edge_decay * t^2)

    where ``t`` is the *unscaled* signed distance from the edge line
    (``cos(theta)*X + sin(theta)*Y + c``).  This localises each basis
    function around its edge so that the smooth RKHS component (Kd) is
    forced to carry the background intensity.  Set to 0 or None to
    recover the original (non-decaying) Heaviside from the paper.
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
    H = 0.5 + (1.0 / np.pi) * np.arctan(Z)

    if edge_decay and edge_decay > 0:
        H = H * np.exp(-edge_decay * T_unscaled ** 2)

    return H



def _build_gaussian_edge_basis_np(n_gridx, n_gridy, num_dirs, num_offsets=None,
                                   gauss_width=10.0):
    """Build a Gaussian bump edge basis: w_j(x) = exp(-gauss_width * t_j^2).

    Uses the same direction/offset grid as _build_heaviside_basis_np so that
    the columns are index-aligned: column j here corresponds to the same
    (theta_j, c_j) pair as column j in Psi.

    Parameters
    ----------
    gauss_width : float
        Width parameter alpha in exp(-alpha * t^2).  Larger = narrower bumps.
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
    W = np.exp(-gauss_width * T_unscaled ** 2)
    return W


def compute_gaussian_edge_map(result, gauss_width=10.0, return_numpy=True,
                              use_theta=False, binarize_coeffs=False):
    """Compute an alternative edge map W*coeffs using a Gaussian bump basis.

    The original fit produces Psi*beta where Psi has columns of the form
        psi_j(x) = (0.5 + (1/pi)*arctan(t_j/delta)) * exp(-decay * t_j^2).

    This function builds a *new* basis W with columns
        w_j(x) = exp(-gauss_width * t_j^2)
    using the same (direction, offset) grid, then evaluates W*coeffs with the
    *already-fitted* coefficients.

    Parameters
    ----------
    result : dict
        Output of ``fit_rkhs_decomposition``.
    gauss_width : float
        The alpha parameter in exp(-alpha * t^2).  Larger = narrower bumps
        (more localised edges).
    return_numpy : bool
        If True, return a numpy array; otherwise return a torch tensor on
        the same device as the fitted coefficients.
    use_theta : bool
        If True, use the theta (projected) coefficients instead of beta.
        When nonneg_beta=True, theta is guaranteed nonneg at every iteration,
        so this gives a nonneg edge map.
    binarize_coeffs : bool
        If True, replace each coefficient with 1 where it is positive and
        0 otherwise, before multiplying by the basis.  This produces an
        equalized edge indicator: every detected edge component contributes
        equally regardless of its fitted magnitude.  Best combined with
        use_theta=True and nonneg_beta=True so that the support of the
        coefficients reflects genuine edge detections.

    Returns
    -------
    Wb_img : ndarray or torch.Tensor, shape (H, W)
        The edge map W*coeffs, assembled from patches by overlap averaging
        (same procedure as Psi*beta).
    """
    cfg = result["cfg"]
    ps      = cfg["ptc"]["patchsize"]
    overlap = cfg["ptc"]["overlap"]
    L       = cfg["bss"]["ell_numdirs"]
    n_off   = cfg["bss"].get("num_offsets", None)

    # Build new basis (float64 numpy, then cast)
    W_np = _build_gaussian_edge_basis_np(ps, ps, L, num_offsets=n_off,
                                          gauss_width=gauss_width)

    # Select coefficients: theta (projected, nonneg) or beta (raw)
    coeff_key = "theta" if use_theta else "beta"
    beta_arr = result[coeff_key]       # (n2, P) in the returned layout
    is_numpy = isinstance(beta_arr, np.ndarray)
    if is_numpy:
        device = torch.device("cpu")
        dtype  = torch.float64
        beta_t = torch.as_tensor(beta_arr, dtype=dtype, device=device)
    else:
        device = beta_arr.device
        dtype  = beta_arr.dtype
        beta_t = beta_arr

    W_basis = torch.as_tensor(W_np, dtype=dtype, device=device)  # (ps^2, n2)

    # beta_t is (n2, P) — transpose to (P, n2) for batched matmul
    beta_batch = beta_t.t()                                        # (P, n2)

    if binarize_coeffs:
        beta_batch = (beta_batch > 0).to(dtype=beta_batch.dtype)

    # Recover image shape from Kd
    Kd = result["Kd"]
    if isinstance(Kd, np.ndarray):
        im_h, im_w = Kd.shape
    else:
        im_h, im_w = Kd.shape

    # Rebuild patch layout and flat indices (must match the fit exactly)
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

    # Compute W*beta per patch and scatter-add
    Wb_patches = beta_batch @ W_basis.T                           # (P, ps^2)
    Wb_img = torch.zeros(im_h * im_w, dtype=dtype, device=device)
    Wb_img.index_add_(0, patch_flat_idx_flat, Wb_patches.reshape(-1))
    Wb_img = Wb_img.reshape(im_h, im_w) / cnt

    if return_numpy and not is_numpy:
        return Wb_img.detach().cpu().numpy()
    elif return_numpy:
        return Wb_img.detach().cpu().numpy()
    else:
        return Wb_img


# ---------------------------------------------------------------------------
# patch layout (matches rkhs_modelfit._build_patch_layout exactly)
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


def _forward_grad(F):
    """Periodic forward difference along last (Fx) and second-last (Fy) axes.

    Matches ``psf2otf([1, -1], ..)`` / ``psf2otf([[1],[-1]], ..)`` used in
    the W FFT solve, so grad / grad* / Laplacian are operator-consistent
    across the W, v and b3 updates (same fix as in rkhs_modelfit.py).
    """
    Fx = torch.roll(F, -1, dims=-1) - F
    Fy = torch.roll(F, -1, dims=-2) - F
    return Fx, Fy


# ---------------------------------------------------------------------------
# cache / state
# ---------------------------------------------------------------------------

def _build_cache(im_shape, cfg, device, dtype):
    """Precompute K, Psi, A_d_inv, FFT operators, and patch indices.

    The kernel / basis / inverse are computed in float64 on CPU then cast
    to ``dtype``; this keeps the one-shot ``inv(A_d)`` (where ``zeta1`` is
    typically 1e-9) accurate even when the iteration itself runs float32.
    """
    im_h, im_w = im_shape
    ps      = cfg["ptc"]["patchsize"]
    overlap = cfg["ptc"]["overlap"]
    sigma_k = cfg["bss"]["sigma_kerwidth"]
    delta   = cfg["bss"]["delta_rampwidth"]
    L       = cfg["bss"]["ell_numdirs"]
    n_off   = cfg["bss"].get("num_offsets", None)
    e_decay = cfg["bss"].get("edge_decay", None)

    gamma = cfg["mdl"]["gamma_smoothpen"]
    rho2  = cfg["opt"]["rho2_Wsplit"]
    zeta1 = cfg["opt"]["zeta1_dprox"]
    zeta2_user   = cfg["opt"]["zeta2_betaprox"]
    zeta2_safety = cfg["opt"].get("zeta2_betaprox_safety", 0.0)

    # --- float64 numpy build (matches rkhs_modelfit exactly) ---------------
    K_np   = _build_gaussian_kernel_matrix_np(ps, ps, sigma_k)
    Psi_np = _build_heaviside_basis_np(ps, ps, delta, L, num_offsets=n_off,
                                       edge_decay=e_decay)
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

    # --- cast onto device --------------------------------------------------
    def t(x):
        return torch.as_tensor(x, dtype=dtype, device=device)
    K     = t(K_np)
    Psi   = t(Psi_np)
    Kt    = t(Kt_np)
    Psit  = t(Psit_np)
    KtPsi = t(KtPsi_np)
    A_d_inv = t(A_d_inv_np)

    # --- patch layout + flat indices (F-order to match numpy reference) ----
    layout = _build_patch_layout(im_h, im_w, ps, overlap)
    grid_x = torch.as_tensor(layout["grid_x"], dtype=torch.long, device=device)
    grid_y = torch.as_tensor(layout["grid_y"], dtype=torch.long, device=device)

    # Top-left flat indices in row-major (C-order) flatten of the image,
    # ordered (x, y) iterating y inside x to match the numpy loop's `pc`
    # increment order (for xx in grid_x: for yy in grid_y: ...).
    top_left = (grid_x[:, None] * im_w + grid_y[None, :]).reshape(-1)  # (P,)

    # Offset within a patch in F-order (k % ps is the row-offset, fast index).
    k_range = torch.arange(ps * ps, dtype=torch.long, device=device)
    offsets = (k_range % ps) * im_w + (k_range // ps)                  # (ps^2,)

    patch_flat_idx = top_left[:, None] + offsets[None, :]              # (P, ps^2)
    patch_flat_idx_flat = patch_flat_idx.reshape(-1)                   # (P*ps^2,)

    # Overlap count: increment one per patch element, scatter into image.
    cnt = torch.zeros(im_h * im_w, dtype=dtype, device=device)
    cnt.index_add_(0, patch_flat_idx_flat,
                   torch.ones_like(patch_flat_idx_flat, dtype=dtype))
    cnt = cnt.reshape(im_h, im_w).clamp(min=1.0)

    # --- FFT operators (complex precision matched to dtype) ---------------
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
        "patch_flat_idx":      patch_flat_idx,      # (P, ps^2)
        "patch_flat_idx_flat": patch_flat_idx_flat, # (P*ps^2,)
        "patch_count":         cnt,                 # (H, W)
        "Dx_otf": Dx_otf, "Dy_otf": Dy_otf,
        "Dx_conj": Dx_conj, "Dy_conj": Dy_conj,
        "Lap_otf": Lap_otf,
        "device": device, "dtype": dtype, "complex_dtype": complex_dtype,
    }


def _initialize_state(image_t, cache):
    """Mirror rkhs_modelfit._initialize_state. All zeros except W = z."""
    im_h, im_w = cache["im_h"], cache["im_w"]
    n_patches  = cache["layout"]["num_patches"]
    n1, n2     = cache["n1"], cache["n2"]
    device, dtype = cache["device"], cache["dtype"]

    z = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)
    return {
        # Patches stacked as (P, dim) so they batch along dim=0 in matmuls.
        # The numpy reference uses (dim, P) — we transpose at the boundary
        # in the public-API return value to keep call sites compatible.
        "patch": {
            "d":         z(n_patches, n1),
            "beta":      z(n_patches, n2),
            "beta_prev": z(n_patches, n2),
            "theta":     z(n_patches, n2),
            "b1":        z(n_patches, n2),
        },
        "img": {
            "Wrec": image_t.clone(),   # W^(0) = z, matches numpy version
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
# iteration updates (batched across patches)
# ---------------------------------------------------------------------------

def _patch_update(image_t, state, cache, cfg):
    """Batched BCD sweep over patches: update (d, beta, theta, b1).

    Replaces the double for-loop in rkhs_modelfit with a single dispatch
    of batched matmuls. For each patch p:
        d-update:        d_p     = A_inv (K^T z_p - (1+rho2) K^T Psi beta_p
                                          + zeta1 d_p
                                          + rho2 K^T (W_p + b2_p))
        beta prox-lin:   beta_p  = (rho1 (theta_p + b1_p) + zeta2 beta_hat_p
                                    - p_hat_p) / (rho1 + zeta2)
        theta:           theta_p = S_{alpha/rho1}(beta_p - b1_p)
        Bregman b1:      b1_p    = b1_p + theta_p - beta_p

    All of these are linear in beta_p / d_p / theta_p, so once the patch
    pixels and per-patch state are stacked into (P, dim) tensors the whole
    sweep is a sequence of (P, dim_in) @ (dim_in, dim_out) products.

    Numpy → torch matmul translations (for M of shape (n_out, n_in)):
        M @ v             →   v_batch @ M.T      where v_batch is (P, n_in)
        M^T @ v           →   v_batch @ M
    K and A_d_inv are symmetric so M.T == M for them.
    """
    im_h, im_w = cache["im_h"], cache["im_w"]
    ps         = cache["ps"]

    K       = cache["K"]
    Psi     = cache["Psi"]
    KtPsi   = cache["KtPsi"]
    A_inv   = cache["A_d_inv"]
    idx     = cache["patch_flat_idx"]       # (P, ps^2)
    cnt     = cache["patch_count"]          # (H, W)

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

    d_batch     = state["patch"]["d"]          # (P, n1)
    beta_batch  = state["patch"]["beta"]       # (P, n2)
    beta_prev   = state["patch"]["beta_prev"]  # (P, n2)
    theta_batch = state["patch"]["theta"]      # (P, n2)
    b1_batch    = state["patch"]["b1"]         # (P, n2)

    # ---- batch-extract patches in F-order ---------------------------------
    flat_im = image_t.reshape(-1)
    flat_W  = Wrec    .reshape(-1)
    flat_b2 = b2      .reshape(-1)
    flat_vx = vx      .reshape(-1)
    flat_vy = vy      .reshape(-1)

    im_b = flat_im[idx]          # (P, ps^2)
    W_b  = flat_W [idx]
    b2_b = flat_b2[idx]
    vx_b = flat_vx[idx]
    vy_b = flat_vy[idx]
    v_norm_b = vx_b.abs() + vy_b.abs()

    # ---- d-update (paper eq (11)) -----------------------------------------
    # K is symmetric so Kt = K. Kept the names below to mirror the numpy code.
    d_rhs = (im_b @ K                                  # Kt @ im
             - (1.0 + rho2) * (beta_batch @ KtPsi.T)   # KtPsi @ beta
             + zeta1 * d_batch
             + rho2 * ((W_b + b2_b) @ K))              # Kt @ (W + b2)
    d_new = d_rhs @ A_inv                              # A_inv @ d_rhs (A_inv sym.)
    kd_b  = d_new @ K                                  # K @ d   (K symmetric)

    # ---- beta prox-linear (paper eqs (12)-(14)) ---------------------------
    beta_hat     = beta_batch + omega * (beta_batch - beta_prev)
    psi_beta_hat = beta_hat @ Psi.T                    # Psi @ beta_hat
    g_hat        = 1.0 / (1.0 + iota * psi_beta_hat.pow(2))

    residual_b = im_b - (kd_b + psi_beta_hat)
    p_data = -(residual_b @ Psi)                       # Psit @ (...)
    p_Wsp  = -rho2 * ((W_b - (kd_b + psi_beta_hat) + b2_b) @ Psi)
    p_tv   = -2.0 * nu * iota * ((v_norm_b * g_hat.pow(2) * psi_beta_hat) @ Psi)
    p_hat  = p_data + p_Wsp + p_tv

    beta_new  = (rho1 * (theta_batch + b1_batch)
                 + zeta2 * beta_hat
                 - p_hat) / (rho1 + zeta2)
    theta_new = _soft_threshold(beta_new - b1_batch, alpha / rho1)
    if cfg["mdl"].get("nonneg_beta", False):
        theta_new = torch.clamp(theta_new, min=0.0)
    b1_new    = b1_batch + theta_new - beta_new

    # Roll beta_prev BEFORE overwriting beta_batch (same ordering as numpy).
    state["patch"]["beta_prev"] = beta_batch
    state["patch"]["d"]         = d_new
    state["patch"]["beta"]      = beta_new
    state["patch"]["theta"]     = theta_new
    state["patch"]["b1"]        = b1_new

    # ---- scatter-add aggregated reconstructions back to the image ---------
    pb_b = beta_new @ Psi.T                            # (P, ps^2)
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
    """Image-level W update via FFT (paper eq (19))."""
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
    """v update (paper eq (20)) and Bregman duals (eqs (22), (23))."""
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
    """Fit the RKHS + approximated-Heaviside decomposition (Stage 1).

    Parameters
    ----------
    image : (H, W) ndarray or torch.Tensor
        2D image to decompose.
    cfg : dict, optional
        Same shape as ``rkhs_modelfit.default_config()``.
    device : str | torch.device | None
        Defaults to ``'cuda'`` if available, else ``'cpu'``. Pass
        ``'cuda'`` explicitly on Colab to ensure the T4 is used.
    dtype : torch.dtype
        Computation dtype. ``torch.float32`` is recommended for T4
        (fp32 throughput >> fp64). The cache (K, Psi, A_d_inv) is
        always built in float64 internally then cast, so the one-shot
        matrix inverse is precise regardless.
    verbose : bool
        Print Lipschitz / zeta2 diagnostics and per-iter residuals.
    return_numpy : bool
        If True, decode all returned arrays back to numpy on CPU so the
        signature matches rkhs_modelfit.fit_rkhs_decomposition. Set
        False to get raw torch tensors (still on `device`).
    **overrides
        Flat keyword overrides, identical routing to the numpy version.
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
        print(f"[stage1-torch] device={device} dtype={dtype} "
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
            split_res = float(torch.linalg.norm(
                state["patch"]["theta"] - state["patch"]["beta"]).item())
            print(f"iter {it+1:3d}/{n_iter}  "
                  f"residual={residual:.6f}  |beta|_1={beta_l1:.6f}  "
                  f"||theta-beta||={split_res:.6f}")

    Kd_out = state["img"]["Kd"]
    Pb_out = state["img"]["Pb"]
    # Transpose patches back to (dim, P) so the return shape matches the
    # numpy reference at call sites that index by [:, pc].
    d_out     = state["patch"]["d"].t()
    beta_out  = state["patch"]["beta"].t()
    theta_out = state["patch"]["theta"].t()

    # Primal residual: ||theta - beta|| measures ADMM convergence of the
    # beta <-> theta splitting.  When nonneg_beta is True, this tells you
    # how close beta is to satisfying the nonnegativity constraint.
    primal_res_beta_theta = float(
        torch.linalg.norm(state["patch"]["theta"] - state["patch"]["beta"]).item()
    )

    # Build Psi*theta image (scatter-add, same procedure as Psi*beta).
    # theta is the projected variable, so Psi*theta is guaranteed nonneg
    # when nonneg_beta=True.
    Psi = cache["Psi"]
    idx_flat = cache["patch_flat_idx_flat"]
    im_h, im_w = cache["im_h"], cache["im_w"]
    cnt = cache["patch_count"]
    pt_b = state["patch"]["theta"] @ Psi.T                # (P, ps^2)
    Pt_img = torch.zeros(im_h * im_w, dtype=cache["dtype"], device=cache["device"])
    Pt_img.index_add_(0, idx_flat, pt_b.reshape(-1))
    Pt_img = Pt_img.reshape(im_h, im_w) / cnt

    # Raw basis matrices, useful for inspecting conditioning ||K||, ||Psi^T Psi||,
    # rank, column overlap, etc. when sweeping hyperparameters.
    matrices_t = {
        "K":       cache["K"],        # (ps^2, ps^2)
        "Psi":     cache["Psi"],      # (ps^2, num_offsets * L)
        "A_d_inv": cache["A_d_inv"],  # (ps^2, ps^2)
        "KtPsi":   cache["KtPsi"],    # (ps^2, num_offsets * L)
    }

    if return_numpy:
        result = {
            "Kd":        Kd_out.detach().cpu().numpy(),
            "Psi_beta":  Pb_out.detach().cpu().numpy(),
            "Psi_theta": Pt_img.detach().cpu().numpy(),
            "M":         (Kd_out + Pb_out).detach().cpu().numpy(),
            "d":         d_out.detach().cpu().numpy(),
            "beta":      beta_out.detach().cpu().numpy(),
            "theta":     theta_out.detach().cpu().numpy(),
            "matrices":  {k: v.detach().cpu().numpy() for k, v in matrices_t.items()},
        }
    else:
        result = {
            "Kd":        Kd_out,
            "Psi_beta":  Pb_out,
            "Psi_theta": Pt_img,
            "M":         Kd_out + Pb_out,
            "d":         d_out,
            "beta":      beta_out,
            "theta":     theta_out,
            "matrices":  matrices_t,
        }
    result.update({
        "cfg":     cfg,
        "history": history,
        "diagnostics": {
            "K_op":      cache["K_op"],
            "PtP_op":    cache["PtP_op"],
            "L_beta":    cache["L_beta"],
            "zeta2_eff": cache["zeta2_eff"],
            "primal_res_beta_theta": primal_res_beta_theta,
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
