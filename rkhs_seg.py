"""
RKHS segmentation — Python port (global-mode only).

Translation of the global-mode segmentation pipeline from the MATLAB project at
../project_RKHS/. Only the `cfg.mode.segmentation_variant = 'global'` path is
ported — the localized / selective-geodesic / two-stage / experimental-local-stats
branches are deliberately left out.

Entry point: ``run_global_segmentation``.
"""

from copy import deepcopy

import numpy as np

EPS = float(np.finfo(float).eps)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Default config dict; mirrors `default_global_combined_cfg.m` (global path).

    Sections:
      mdl  — model term weights (lambda, mu, gamma, alpha, iota, eta)
      bss  — basis (Gaussian kernel + Heaviside ramp) parameters
      ptc  — patch size and overlap
      opt  — solver / proximal / split parameters
      term — boolean toggles for individual energy terms
    """
    return {
        "mdl": {
            "mdl_lambda_regionfit": 1e-6,
            "mdl_mu_boundwt":       1e-3,
            "mdl_gamma_smoothpen":  1e-6,
            "mdl_alpha_edgesparse": 1.0,
            "mdl_iota_edgegate":    1e3,
            "mdl_eta_reconfit":     1e3,
        },
        "bss": {
            "bss_sigma_kerwidth":  4.0,
            "bss_delta_rampwidth": 1e-4,
            "bss_ell_numdirs":     12,
        },
        "ptc": {
            "ptc_patchsize_patchsz": 4,
            "ptc_overlap_overlap":   2,
        },
        "opt": {
            "opt_rho1_betasplit":   2.0,
            "opt_rho2_gradsplit":   1e-9,
            "opt_zeta1_dprox":      1e-9,
            "opt_zeta2_betaprox":   10.0,
            "opt_zeta3_inprox":     1e-9,
            "opt_zeta4_outprox":    1e-9,
            "opt_zeta5_uprox":      4.0e-6,
            "opt_maxiter_loopcap":  20,
        },
        "term": {
            "enable_region_fit":     True,
            "enable_boundary_reg":   True,
            "enable_edge_gate":      True,
            "enable_edge_sparsity":  True,
            "enable_recon_datafit":  True,
            "enable_smooth_recon":   True,
        },
    }


def _resolve_term_weights(cfg):
    """Apply boolean toggles to scalar coefficients.

    Mirrors `resolve_term_weights.m` for the global branch (no selective term).
    """
    en_region   = bool(cfg["term"]["enable_region_fit"])
    en_bound    = bool(cfg["term"]["enable_boundary_reg"])
    en_gate_req = bool(cfg["term"]["enable_edge_gate"])
    en_gate     = en_bound and en_gate_req
    en_sparse   = bool(cfg["term"]["enable_edge_sparsity"])
    en_recon    = bool(cfg["term"]["enable_recon_datafit"])
    en_smooth   = bool(cfg["term"]["enable_smooth_recon"])

    t = {
        "mdl_lambda_regionfit": cfg["mdl"]["mdl_lambda_regionfit"] * float(en_region),
        "mdl_mu_boundwt":       cfg["mdl"]["mdl_mu_boundwt"]       * float(en_bound),
        "mdl_gamma_smoothpen":  cfg["mdl"]["mdl_gamma_smoothpen"]  * float(en_smooth),
        "mdl_alpha_edgesparse": cfg["mdl"]["mdl_alpha_edgesparse"] * float(en_sparse),
        "mdl_iota_edgegate":    cfg["mdl"]["mdl_iota_edgegate"]    * float(en_gate),
        "mdl_eta_reconfit":     cfg["mdl"]["mdl_eta_reconfit"]     * float(en_recon),
        "recon_datafit_scale":  float(en_recon),
    }
    t["edge_gate_active"] = en_gate and t["mdl_iota_edgegate"] > 0
    return t


# ---------------------------------------------------------------------------
# basis construction
# ---------------------------------------------------------------------------

def _build_gaussian_kernel_matrix(n_gridx, n_gridy, sigma, coeff1=0):
    """Same construction as build_gaussian_kernel_matrix.m (legacy type 9)."""
    nx_denom = max(n_gridx - 1, 1)
    ny_denom = max(n_gridy - 1, 1)
    tx = np.arange(n_gridx) / nx_denom
    ty = np.arange(n_gridy) / ny_denom

    # MATLAB column-major flattening of meshgrid(tx, ty) of size (n_gridx, n_gridy).
    X = np.tile(tx, n_gridy)        # length n_gridx * n_gridy
    Y = np.repeat(ty, n_gridx)      # length n_gridx * n_gridy

    a = X[:, None] - X[None, :]
    b = Y[:, None] - Y[None, :]
    tao = np.sqrt(a * a + b * b)
    tao[tao == 0] = EPS

    K = np.exp(-(tao * tao) / (2.0 * sigma * sigma))

    if coeff1 == 1:
        coeff = 1.0
    else:
        coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return (coeff * coeff) * K


def _build_heaviside_basis(n_gridx, n_gridy, delta, num_dirs):
    """Approximated-Heaviside basis matrix; mirrors build_heaviside_basis_matrix.m."""
    nx_denom = max(n_gridx - 1, 1)
    ny_denom = max(n_gridy - 1, 1)
    tx = np.arange(n_gridx) / nx_denom
    ty = np.arange(n_gridy) / ny_denom

    X = np.tile(tx, n_gridy)        # length n
    Y = np.repeat(ty, n_gridx)
    n = n_gridx * n_gridy

    # angles 0, 2pi/L, ..., 2pi*(L-1)/L
    theta = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)

    c = np.arange(n) / max(n - 1, 1)

    # MATLAB column-major flattening of [n × num_dirs] grids.
    C_all = np.tile(c, num_dirs)            # length n * num_dirs
    Theta_all = np.repeat(theta, n)

    # Z[i, j] = cos(theta_j) * x_i + sin(theta_j) * y_i + c_j   (then / delta)
    cos_t = np.cos(Theta_all)
    sin_t = np.sin(Theta_all)
    Z = (cos_t[None, :] * X[:, None]
         + sin_t[None, :] * Y[:, None]
         + C_all[None, :]) / delta
    return 0.5 + (1.0 / np.pi) * np.arctan(Z)


# ---------------------------------------------------------------------------
# patch layout
# ---------------------------------------------------------------------------

def _build_patch_layout(im_height, im_width, patch_size, overlap):
    """0-indexed equivalent of build_patch_layout.m. Returns dict with grid_x, grid_y."""
    if patch_size > im_height or patch_size > im_width:
        raise ValueError("patch_size must not exceed image dimensions")
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("overlap must be strictly less than patch_size")

    def make_grid_1based(N, ps):
        upper = N - (N % step + 1 + step)
        if upper < 1:
            grid = []
        else:
            grid = list(range(1, upper + 1, step))
        forbidden = set(range(N - ps + 1, N + 1))
        grid = [g for g in grid if g not in forbidden]
        grid.append(N - ps + 1)
        return grid

    gx = np.array([g - 1 for g in make_grid_1based(im_height, patch_size)], dtype=int)
    gy = np.array([g - 1 for g in make_grid_1based(im_width,  patch_size)], dtype=int)
    return {"step": step, "grid_x": gx, "grid_y": gy, "num_patches": len(gx) * len(gy)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _psf2otf(psf, shape):
    """Equivalent of MATLAB psf2otf: zero-pad to ``shape``, circshift so the PSF
    ``(1,1)`` element ends up at ``(0,0)`` (after the shift), and FFT2."""
    psf = np.asarray(psf, dtype=float)
    pad = np.zeros(shape, dtype=float)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    for axis, axis_size in enumerate(psf.shape):
        pad = np.roll(pad, -(axis_size // 2), axis=axis)
    return np.fft.fft2(pad)


def _matlab_gradient(F):
    """Return (Fx, Fy) matching MATLAB's [Fx, Fy] = gradient(F).

    Fx is the column-direction (axis=1) gradient, Fy is the row-direction (axis=0).
    """
    Fy, Fx = np.gradient(F)
    return Fx, Fy


# ---------------------------------------------------------------------------
# solver cache + state
# ---------------------------------------------------------------------------

def _build_solver_cache(im_shape, cfg):
    """Build the constant operators reused across iterations."""
    H, W = im_shape
    ps      = cfg["ptc"]["ptc_patchsize_patchsz"]
    overlap = cfg["ptc"]["ptc_overlap_overlap"]
    sigma_k = cfg["bss"]["bss_sigma_kerwidth"]
    delta   = cfg["bss"]["bss_delta_rampwidth"]
    L       = cfg["bss"]["bss_ell_numdirs"]

    term = _resolve_term_weights(cfg)
    gamma = term["mdl_gamma_smoothpen"]
    eta   = term["mdl_eta_reconfit"]
    zeta1 = cfg["opt"]["opt_zeta1_dprox"]

    K   = _build_gaussian_kernel_matrix(ps, ps, sigma_k)        # (n1, n1)
    Psi = _build_heaviside_basis(ps, ps, delta, L)              # (n1, n2)

    Kt    = K.T
    Psit  = Psi.T
    KtPsi = Kt @ Psi

    n1 = K.shape[1]
    n2 = Psi.shape[1]

    A_d = eta * (Kt @ K) + 2.0 * gamma * K + zeta1 * np.eye(n1)
    A_d_inv = np.linalg.inv(A_d)

    layout = _build_patch_layout(H, W, ps, overlap)

    Dx_otf  = _psf2otf(np.array([[1.0, -1.0]]), (H, W))
    Dy_otf  = _psf2otf(np.array([[1.0], [-1.0]]), (H, W))
    Dx_conj = np.conj(Dx_otf)
    Dy_conj = np.conj(Dy_otf)
    Lap_otf = (Dx_otf * Dx_conj + Dy_otf * Dy_conj).real

    return {
        "im_height": H, "im_width": W,
        "K": K, "Kt": Kt, "Psi": Psi, "Psit": Psit, "KtPsi": KtPsi,
        "n1": n1, "n2": n2,
        "A_d_inv": A_d_inv,
        "layout": layout,
        "Dx_otf": Dx_otf, "Dy_otf": Dy_otf,
        "Dx_conj": Dx_conj, "Dy_conj": Dy_conj,
        "Lap_otf": Lap_otf,
    }


def _initialize_state(image, mask_init, cache):
    """Initialize evolving solver variables (mirrors initialize_solver_state.m)."""
    H, W = cache["im_height"], cache["im_width"]
    if mask_init is None:
        u = np.zeros((H, W))
    else:
        u = (np.asarray(mask_init) > 0).astype(float)

    wx, wy = _matlab_gradient(u)

    one_minus_u = 1.0 - u
    c1 = float((image * u).sum() / (u.sum() + EPS))
    c2 = float((image * one_minus_u).sum() / (one_minus_u.sum() + EPS))

    n_patches = cache["layout"]["num_patches"]
    n2 = cache["n2"]
    n1 = cache["n1"]

    return {
        "patch": {
            "b1":        np.zeros((n2, n_patches)),
            "theta":     np.zeros((n2, n_patches)),
            "d":         np.zeros((n1, n_patches)),
            "beta":      np.zeros((n2, n_patches)),
            "beta_prev": np.zeros((n2, n_patches)),
        },
        "global": {
            "u":   u,
            "wx":  wx, "wy": wy,
            "b2x": np.zeros((H, W)), "b2y": np.zeros((H, W)),
            "c1":  c1, "c2": c2,
        },
    }


# ---------------------------------------------------------------------------
# iteration updates
# ---------------------------------------------------------------------------

def _update_patch_reconstruction(image, state, cache, cfg):
    """Patchwise update of (d, beta, theta, b1) and aggregation into (J, Kd, Pb).

    Mirrors update_patch_reconstruction.m.
    """
    H, W = cache["im_height"], cache["im_width"]
    ps   = cfg["ptc"]["ptc_patchsize_patchsz"]

    term       = _resolve_term_weights(cfg)
    lam        = term["mdl_lambda_regionfit"]
    mu         = term["mdl_mu_boundwt"]
    alpha      = term["mdl_alpha_edgesparse"]
    iota       = term["mdl_iota_edgegate"]
    eta        = term["mdl_eta_reconfit"]
    recon_scl  = term["recon_datafit_scale"]
    gate_act   = term["edge_gate_active"]

    rho1  = cfg["opt"]["opt_rho1_betasplit"]
    zeta1 = cfg["opt"]["opt_zeta1_dprox"]
    zeta2 = cfg["opt"]["opt_zeta2_betaprox"]

    K      = cache["K"]
    Kt     = cache["Kt"]
    Psi    = cache["Psi"]
    Psit   = cache["Psit"]
    KtPsi  = cache["KtPsi"]
    A_inv  = cache["A_d_inv"]

    u  = state["global"]["u"]
    wx = state["global"]["wx"]
    wy = state["global"]["wy"]
    c1 = state["global"]["c1"]
    c2 = state["global"]["c2"]

    d_all         = state["patch"]["d"]
    beta_all      = state["patch"]["beta"]
    beta_prev_all = state["patch"]["beta_prev"]
    theta_all     = state["patch"]["theta"]
    b1_all        = state["patch"]["b1"]

    A      = np.zeros((H, W))
    B      = np.zeros((H, W))
    Kd_img = np.zeros((H, W))
    Pb_img = np.zeros((H, W))

    grid_x = cache["layout"]["grid_x"]
    grid_y = cache["layout"]["grid_y"]

    pc = 0  # patch counter — outer over x (rows), inner over y (cols), matching MATLAB
    for xx in grid_x:
        for yy in grid_y:
            sl_x = slice(xx, xx + ps)
            sl_y = slice(yy, yy + ps)

            im_patch = image[sl_x, sl_y]
            u_patch  = u[sl_x, sl_y]
            wx_patch = wx[sl_x, sl_y]
            wy_patch = wy[sl_x, sl_y]

            # MATLAB column-major flatten, hence order='F'.
            im_vec = im_patch.flatten(order="F")
            u_vec  = u_patch.flatten(order="F")

            b1         = b1_all[:, pc]
            theta_v    = theta_all[:, pc]
            beta       = beta_all[:, pc]
            d          = d_all[:, pc]
            older_beta = beta_prev_all[:, pc]

            # --- d-update (proximal linear solve) ---
            d_rhs = (eta * (Kt @ im_vec)
                     - (eta + 2.0 * lam) * (KtPsi @ beta)
                     + zeta1 * d
                     + 2.0 * lam * (Kt @ (c1 * u_vec + c2 * (1.0 - u_vec))))
            d = A_inv @ d_rhs
            kd_vec = K @ d

            # --- beta-update (forward/backward + extrapolation, then ADMM split) ---
            beta_hat = beta + (beta - older_beta)
            psi_beta_hat = Psi @ beta_hat
            if gate_act:
                g_beta_hat = 1.0 / (1.0 + iota * (psi_beta_hat ** 2))
            else:
                g_beta_hat = np.ones_like(psi_beta_hat)

            p_hat1 = -recon_scl * (Psit @ (im_vec - (kd_vec + psi_beta_hat)))

            w_for_p = (np.abs(wx_patch) + np.abs(wy_patch)).flatten(order="F")
            p_hat3 = -2.0 * mu * iota * (Psit @ ((g_beta_hat ** 2) * psi_beta_hat * w_for_p))
            p_hat4 = 2.0 * lam * (Psit @ (kd_vec + psi_beta_hat - c1 * u_vec - c2 * (1.0 - u_vec)))
            p_hat  = p_hat1 + p_hat3 + p_hat4

            beta_new  = (zeta2 * beta_hat - p_hat + rho1 * (theta_v + b1)) / (rho1 + zeta2)
            theta_new = _soft_threshold(beta_new - b1, alpha / rho1)
            b1_new    = b1 + (theta_new - beta_new)

            d_all[:, pc]      = d
            beta_all[:, pc]   = beta_new
            theta_all[:, pc]  = theta_new
            b1_all[:, pc]     = b1_new

            # --- aggregate into image-sized accumulators ---
            kd_patch  = (K @ d).reshape((ps, ps), order="F")
            psi_patch = (Psi @ beta_new).reshape((ps, ps), order="F")
            recon_patch = kd_patch + psi_patch

            A[sl_x, sl_y]      += recon_patch
            B[sl_x, sl_y]      += 1.0
            Kd_img[sl_x, sl_y] += kd_patch
            Pb_img[sl_x, sl_y] += psi_patch

            pc += 1

    # Mirror the MATLAB behaviour: prev := current after the sweep.
    state["patch"]["beta_prev"] = beta_all.copy()

    B[B == 0] = 1.0
    J         = A / B
    Kd_smooth = Kd_img / B
    Pb_edge   = Pb_img / B
    if gate_act:
        g_edgegate = 1.0 / (1.0 + iota * (Pb_edge ** 2))
    else:
        g_edgegate = np.ones_like(Pb_edge)

    return {
        "J":          J,
        "Kd_smooth":  Kd_smooth,
        "Pb_edge":    Pb_edge,
        "g_edgegate": g_edgegate,
    }


def _update_region_means(J, state, cfg):
    """Global-branch Chan-Vese means with proximal regularization."""
    u  = state["global"]["u"]
    c1 = state["global"]["c1"]
    c2 = state["global"]["c2"]
    z3 = cfg["opt"]["opt_zeta3_inprox"]
    z4 = cfg["opt"]["opt_zeta4_outprox"]

    one_minus_u = 1.0 - u
    state["global"]["c1"] = float(((J * u).sum()           + z3 * c1) / (u.sum()           + z3 + EPS))
    state["global"]["c2"] = float(((J * one_minus_u).sum() + z4 * c2) / (one_minus_u.sum() + z4 + EPS))


def _update_membership_split(iter_frame, state, cache, cfg):
    """Update u (membership) and (wx, wy, b2x, b2y) via Bregman split.

    Global branch — selective_force is identically zero.
    Mirrors update_membership_split.m.
    """
    term     = _resolve_term_weights(cfg)
    lam      = term["mdl_lambda_regionfit"]
    mu       = term["mdl_mu_boundwt"]
    iota     = term["mdl_iota_edgegate"]
    gate_act = term["edge_gate_active"]

    rho2  = cfg["opt"]["opt_rho2_gradsplit"]
    zeta5 = cfg["opt"]["opt_zeta5_uprox"]

    J       = iter_frame["J"]
    Pb_edge = iter_frame["Pb_edge"]

    u   = state["global"]["u"]
    wx  = state["global"]["wx"]
    wy  = state["global"]["wy"]
    b2x = state["global"]["b2x"]
    b2y = state["global"]["b2y"]
    c1  = state["global"]["c1"]
    c2  = state["global"]["c2"]

    if gate_act:
        g = 1.0 / (1.0 + iota * (Pb_edge ** 2))
    else:
        g = np.ones_like(Pb_edge)

    region_force = (J - c1) ** 2 - (J - c2) ** 2

    Dx_conj = cache["Dx_conj"]
    Dy_conj = cache["Dy_conj"]
    Lap_otf = cache["Lap_otf"]

    fft2  = np.fft.fft2
    ifft2 = np.fft.ifft2

    u_ra = rho2 * (Dx_conj * fft2(wx + b2x) + Dy_conj * fft2(wy + b2y))
    u_rb = lam  * fft2(region_force)
    u_rc = zeta5 * fft2(u)
    u_l  = rho2 * Lap_otf + zeta5 + EPS

    u_new = np.real(ifft2((u_ra - u_rb + u_rc) / u_l))
    u_new = np.clip(u_new, 0.0, 1.0)

    ux, uy = _matlab_gradient(u_new)
    wx_new = _soft_threshold(ux - b2x, g * (mu / rho2))
    wy_new = _soft_threshold(uy - b2y, g * (mu / rho2))

    b2x_new = b2x + wx_new - ux
    b2y_new = b2y + wy_new - uy

    state["global"]["u"]   = u_new
    state["global"]["wx"]  = wx_new
    state["global"]["wy"]  = wy_new
    state["global"]["b2x"] = b2x_new
    state["global"]["b2y"] = b2y_new

    iter_frame["g_edgegate"] = g


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def run_global_segmentation(image, mask_init=None, cfg=None, *, verbose=False, **overrides):
    """Run global RKHS segmentation on a 2D image.

    Parameters
    ----------
    image : (H, W) array
        Real-valued image. Will be cast to float. Best results if normalized to
        roughly [0, 1].
    mask_init : (H, W) array or None
        Initial 0/1 mask. If None, starts from an all-zero mask (note that a
        zero init can be sluggish — providing any reasonable init helps).
    cfg : dict, optional
        Full config dict shaped like ``default_config()``. Overrides the
        default if provided.
    verbose : bool
        Print iteration progress.
    **overrides
        Flat keyword overrides such as ``mdl_lambda_regionfit=1e-5`` or
        ``opt_maxiter_loopcap=50``. Names are routed to the matching cfg
        sub-dict by prefix.

    Returns
    -------
    result : dict with keys
        u : (H, W) float — final soft membership; threshold at 0.5 to get a binary mask.
        J : (H, W) float — final reconstructed image (Kd + Psi*beta aggregate).
        cfg : the resolved config dict that was used.
        history : list of per-iteration dicts (u, J, Kd, Pb, c1, c2).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D (H, W)")

    cfg = deepcopy(default_config()) if cfg is None else deepcopy(cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    cache = _build_solver_cache(image.shape, cfg)
    state = _initialize_state(image, mask_init, cache)

    n_iter = int(cfg["opt"]["opt_maxiter_loopcap"])
    history = []

    for it in range(n_iter):
        frame = _update_patch_reconstruction(image, state, cache, cfg)
        _update_region_means(frame["J"], state, cfg)
        _update_membership_split(frame, state, cache, cfg)

        history.append({
            "iter": it,
            "u":  state["global"]["u"].copy(),
            "J":  frame["J"].copy(),
            "Kd": frame["Kd_smooth"].copy(),
            "Pb": frame["Pb_edge"].copy(),
            "c1": state["global"]["c1"],
            "c2": state["global"]["c2"],
        })
        if verbose:
            u_now = state["global"]["u"]
            print(f"iter {it+1:3d}/{n_iter}  c1={state['global']['c1']:+.4f}  "
                  f"c2={state['global']['c2']:+.4f}  area={(u_now > 0.5).sum():>6d}")

    return {
        "u": state["global"]["u"],
        "J": history[-1]["J"] if history else image.copy(),
        "cfg": cfg,
        "history": history,
    }


def _apply_overrides(cfg, overrides):
    """Route flat keyword overrides into the nested cfg dict by prefix."""
    sections = ("mdl", "bss", "ptc", "opt", "term")
    for k, v in overrides.items():
        placed = False
        for sec in sections:
            if k in cfg.get(sec, {}):
                cfg[sec][k] = v
                placed = True
                break
        if not placed:
            for sec in sections:
                if k.startswith(sec + "_"):
                    cfg.setdefault(sec, {})[k] = v
                    placed = True
                    break
        if not placed:
            raise KeyError(f"unknown override: {k!r}")
    return cfg
