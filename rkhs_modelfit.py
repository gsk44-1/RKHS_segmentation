"""
RKHS image decomposition — Stage 1 of the two-stage global segmentation method
from Burrows, Guo, Chen & Torella, "Reproducing Kernel Hilbert Space Based
Global and Local Image Segmentation", Inverse Problems and Imaging, 2020
(IPI2020_RKHSSeg.pdf).

Implements paper model (8):

    min_{d, beta}   (1/2) ||z - (K d + Psi beta)||^2
                  + gamma * d^T K d
                  + alpha * ||beta||_1
                  + nu * g^T |grad(K d + Psi beta)|

with the edge-stopping function g(Psi beta) = 1 / (1 + iota * |Psi beta|^2).

Algorithm (9) of the paper: introduce three auxiliary splits

    theta = beta            (handles the ||beta||_1 term)
    W     = K d + Psi beta  (handles the grad-on-reconstruction term)
    v     = grad W          (handles the weighted-TV shrinkage)

and run a block coordinate descent (BCD) on small overlapping patches with
Bregman updates b1, b2, b3 for the three splits. The smooth (K d) and edge
(Psi beta) components are returned separately so they can be fed into a
downstream Stage 2 segmentation, OR inspected on their own.

This is intentionally NOT a copy of `rkhs_seg.py`'s combined-model solver:
that file implements paper model (26) where the TV term acts on the
membership u (grad u), not on the reconstruction (grad W). The two models
share helpers (kernel/basis builders, patch layout, forward-difference
operator, psf2otf) but solve different optimisation problems.

Entry point: ``fit_rkhs_decomposition``.

Notes on the original divergence
--------------------------------
The previous version of this file diverged with the nominal paper defaults
because of three coupled issues:

  1. ``zeta2_betaprox = 10`` is much smaller than the Lipschitz constant of
     the beta-subproblem's gradient. For the proximal-linear step (eqs
     (12)-(14)) to be non-expansive we need
        zeta2 >= ||grad f||_Lip = (1 + rho2) * ||Psi^T Psi||_op + O(nu, iota)
     and for ps=4, L=12 we have ||Psi^T Psi||_op ~ 2014, so with rho2=1 the
     required floor is ~ 4028. ``_build_cache`` now computes ||Psi^T Psi||_op
     and auto-bumps zeta2 above that floor (controlled by
     ``zeta2_betaprox_safety``).
  2. ``np.gradient`` (central differences) was being used to compute grad W
     for the v / b3 updates, but the W FFT solve uses ``psf2otf([1, -1], ..)``
     which is a *forward* difference. That operator mismatch broke the
     identity grad* grad = -Laplacian inside the splitting. Replaced with
     ``_forward_grad`` so all three subproblems (W, v, b3) share the same
     discrete grad / grad* pair.
  3. The Nesterov-style extrapolation
        beta_hat = beta + omega (beta - beta_prev)
     looked like it was running with omega = 1, but ``beta_prev`` was
     overwritten to the *new* ``beta`` at the end of each sweep, so on the
     next iteration ``older == beta`` and ``beta_hat == beta``. Per-patch
     bookkeeping is now done BEFORE ``beta`` is overwritten, and the
     extrapolation weight is exposed as ``beta_extrap_omega`` (default 0).
"""

from copy import deepcopy

import numpy as np

EPS = float(np.finfo(float).eps)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def default_config():
    """Default config dict for Stage 1 model fitting.

    Sections:
      mdl  : model term weights
        gamma_smoothpen   : gamma in   gamma * d^T K d
        alpha_edgesparse  : alpha in   alpha * ||beta||_1
        nu_tvweight       : nu in      nu * g^T |grad W|
        iota_edgegate     : iota in    g = 1 / (1 + iota * |Psi beta|^2)
      bss  : basis (Gaussian kernel + Heaviside ramp) parameters.
                          Paper "Parameter selection": sigma=12, delta=1e-4,
                          iota=1e3, L=12 are the recommended values.
      ptc  : patch size and overlap (paper: 4 with overlap 3).
      opt  : splitting / proximal parameters and iteration cap.
        zeta2_betaprox_safety : safety factor applied to the auto-computed
                                Lipschitz lower bound for zeta2. Set <= 0 to
                                disable auto-tuning and use the literal
                                ``zeta2_betaprox`` instead. The proximal-
                                linear beta step diverges when zeta2 < L, so
                                this is the knob you want set to >= 1.
        beta_extrap_omega     : Nesterov-style extrapolation weight omega in
                                beta_hat = beta + omega (beta - beta_prev).
                                Paper uses omega in [0, 1). Default 0 (no
                                extrapolation).
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
            # Number of arctan-ramp offsets sampled along [0, 1] per direction.
            # None -> ps^2 (original paper recipe; highly redundant at large ps).
            # Lower values (e.g. ps) substantially reduce ||Psi^T Psi||_op.
            "num_offsets":     None,
        },
        "ptc": {
            "patchsize": 4,
            "overlap":   3,
        },
        "opt": {
            "rho1_betasplit":        2.0,   # for theta = beta
            "rho2_Wsplit":           1.0,   # for W     = K d + Psi beta
            "rho3_gradsplit":        1.0,   # for v     = grad W
            "zeta1_dprox":           1e-9,
            "zeta2_betaprox":        10.0,
            "zeta2_betaprox_safety": 1.05,  # auto-bump zeta2 >= L * safety
            "beta_extrap_omega":     0.0,
            "maxiter":               30,
        },
    }


# ---------------------------------------------------------------------------
# basis construction
# ---------------------------------------------------------------------------

def _build_gaussian_kernel_matrix(n_gridx, n_gridy, sigma):
    nx_denom = max(n_gridx - 1, 1)#
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


def _build_heaviside_basis(n_gridx, n_gridy, delta, num_dirs, num_offsets=None):
    """Build the Heaviside / arctan-ramp basis.

    Each column is ``0.5 + (1/pi) arctan((cos(theta) X + sin(theta) Y + c) / delta)``
    sampled on the patch grid. The basis has ``num_dirs * num_offsets`` columns;
    along each direction theta we sample ``num_offsets`` offset positions c in
    [0, 1].

    ``num_offsets`` controls Psi's column redundancy:
      * ``None`` (default) -> ``num_offsets = n_gridx * n_gridy = ps^2``,
        matching the original paper construction.
      * Smaller values reduce ||Psi^T Psi||_op (PtP_op) roughly proportionally
        and free the beta prox-linear step to take meaningful sizes. A
        sensible choice is ``num_offsets ~ n_gridx`` (one offset per pixel
        of patch extent) paired with ``delta_rampwidth ~ 1/n_gridx`` so the
        adjacent ramps are spaced at ~1 pixel and distinguishable.
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
    Z = (cos_t[None, :] * X[:, None]
         + sin_t[None, :] * Y[:, None]
         + C_all[None, :]) / delta
    return 0.5 + (1.0 / np.pi) * np.arctan(Z)


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

    gx = np.array([g - 1 for g in make_grid(im_height, patch_size)], dtype=int)
    gy = np.array([g - 1 for g in make_grid(im_width,  patch_size)], dtype=int)
    return {"step": step, "grid_x": gx, "grid_y": gy,
            "num_patches": len(gx) * len(gy)}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _soft_threshold(x, t):
    """Element-wise soft threshold; `t` may be scalar or array of x's shape."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _psf2otf(psf, shape):
    """Equivalent of MATLAB psf2otf."""
    psf = np.asarray(psf, dtype=float)
    pad = np.zeros(shape, dtype=float)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    for axis, axis_size in enumerate(psf.shape):
        pad = np.roll(pad, -(axis_size // 2), axis=axis)
    return np.fft.fft2(pad)


def _forward_grad(F):
    """Periodic forward difference in column (Fx) and row (Fy) directions.

    Matches the operator implied by ``psf2otf([1, -1], ...)`` /
    ``psf2otf([[1], [-1]], ...)`` used in the W FFT solve, so that the
    grad / grad* / Laplacian operators are consistent across the W, v and
    b3 updates. Previously this used ``np.gradient`` (central differences),
    which is a different operator and made the W / v subproblems
    inconsistent.
    """
    Fx = np.roll(F, -1, axis=1) - F
    Fy = np.roll(F, -1, axis=0) - F
    return Fx, Fy


# ---------------------------------------------------------------------------
# solver cache + state
# ---------------------------------------------------------------------------

def _build_cache(im_shape, cfg):
    im_h, im_w = im_shape
    ps      = cfg["ptc"]["patchsize"]
    overlap = cfg["ptc"]["overlap"]
    sigma_k = cfg["bss"]["sigma_kerwidth"]
    delta   = cfg["bss"]["delta_rampwidth"]
    L       = cfg["bss"]["ell_numdirs"]
    n_off   = cfg["bss"].get("num_offsets", None)

    gamma = cfg["mdl"]["gamma_smoothpen"]
    rho2  = cfg["opt"]["rho2_Wsplit"]
    zeta1 = cfg["opt"]["zeta1_dprox"]
    zeta2_user   = cfg["opt"]["zeta2_betaprox"]
    zeta2_safety = cfg["opt"].get("zeta2_betaprox_safety", 0.0)

    K   = _build_gaussian_kernel_matrix(ps, ps, sigma_k)     # (ps^2, ps^2)
    Psi = _build_heaviside_basis(ps, ps, delta, L, num_offsets=n_off)
    # Psi shape: (ps^2, L * num_offsets);
    # default num_offsets = ps^2 reproduces the paper's (ps^2, L*ps^2) basis.

    Kt    = K.T
    Psit  = Psi.T
    KtPsi = Kt @ Psi

    n1 = K.shape[1]
    n2 = Psi.shape[1]

    # d-update system matrix (paper eq (11)):
    #   A = (1 + rho2) K^T K + 2 gamma K + zeta1 I
    A_d = (1.0 + rho2) * (Kt @ K) + 2.0 * gamma * K + zeta1 * np.eye(n1)
    A_d_inv = np.linalg.inv(A_d)

    # Lipschitz bound for the beta prox-linear step. The quadratic part of
    # grad f(beta) is (1 + rho2) Psi^T Psi beta + ...; the prox-linear
    # iteration's contraction factor along eigendirection lambda_i of
    # (1+rho2) Psi^T Psi is |zeta2 - lambda_i| / (rho1 + zeta2), so zeta2
    # must satisfy zeta2 >= ||(1+rho2) Psi^T Psi||_op for the step to be
    # non-expansive. For ps=4 L=12, ||Psi^T Psi||_op ~ 2014; with the
    # nominal zeta2=10 the iteration diverges by factor ~ 334 per step.
    # We auto-bump zeta2 here whenever zeta2_betaprox_safety > 0.
    K_op   = float(np.linalg.norm(K, ord=2))
    PtP_op = float(np.linalg.norm(Psit @ Psi, ord=2))
    L_beta = (1.0 + rho2) * PtP_op
    if zeta2_safety > 0.0:
        zeta2_eff = max(zeta2_user, zeta2_safety * L_beta)
    else:
        zeta2_eff = zeta2_user

    layout = _build_patch_layout(im_h, im_w, ps, overlap)

    # FFT operators for the image-level W subproblem (paper eq (19)). Dx /
    # Dy here are PERIODIC FORWARD differences (matching ``_forward_grad``
    # in the v-update); psf2otf([1, -1], ...) yields exactly that operator.
    Dx_otf  = _psf2otf(np.array([[1.0, -1.0]]), (im_h, im_w))
    Dy_otf  = _psf2otf(np.array([[1.0], [-1.0]]), (im_h, im_w))
    Dx_conj = np.conj(Dx_otf)
    Dy_conj = np.conj(Dy_otf)
    Lap_otf = (Dx_otf * Dx_conj + Dy_otf * Dy_conj).real

    return {
        "im_h": im_h, "im_w": im_w,
        "ps": ps,
        "K": K, "Kt": Kt, "Psi": Psi, "Psit": Psit, "KtPsi": KtPsi,
        "n1": n1, "n2": n2,
        "A_d_inv": A_d_inv,
        "K_op":       K_op,
        "PtP_op":     PtP_op,
        "L_beta":     L_beta,
        "zeta2_eff":  zeta2_eff,
        "layout": layout,
        "Dx_otf": Dx_otf, "Dy_otf": Dy_otf,
        "Dx_conj": Dx_conj, "Dy_conj": Dy_conj,
        "Lap_otf": Lap_otf,
    }


def _initialize_state(image, cache):
    im_h, im_w = cache["im_h"], cache["im_w"]
    n_patches  = cache["layout"]["num_patches"]
    n1, n2     = cache["n1"], cache["n2"]

    return {
        "patch": {
            "d":         np.zeros((n1, n_patches)),
            "beta":      np.zeros((n2, n_patches)),
            "beta_prev": np.zeros((n2, n_patches)),
            "theta":     np.zeros((n2, n_patches)),
            "b1":        np.zeros((n2, n_patches)),
        },
        "img": {
            # W is the (image-level) split that should equal Kd + Psi*beta.
            # Warm-start at the input image: W^(0) = z.
            "Wrec": image.copy(),
            "b2":   np.zeros((im_h, im_w)),
            # v = grad W split, plus its Bregman dual b3.
            "vx":   np.zeros((im_h, im_w)),
            "vy":   np.zeros((im_h, im_w)),
            "b3x":  np.zeros((im_h, im_w)),
            "b3y":  np.zeros((im_h, im_w)),
            # Aggregated patch reconstructions (populated each iteration).
            "Kd":   np.zeros((im_h, im_w)),
            "Pb":   np.zeros((im_h, im_w)),
        },
    }


# ---------------------------------------------------------------------------
# iteration updates
# ---------------------------------------------------------------------------

def _patch_update(image, state, cache, cfg):
    """Per-iteration BCD sweep over patches: update (d, beta, theta, b1)."""
    im_h, im_w = cache["im_h"], cache["im_w"]
    ps         = cache["ps"]

    K     = cache["K"]
    Kt    = cache["Kt"]
    Psi   = cache["Psi"]
    Psit  = cache["Psit"]
    KtPsi = cache["KtPsi"]
    A_inv = cache["A_d_inv"]

    alpha = cfg["mdl"]["alpha_edgesparse"]
    nu    = cfg["mdl"]["nu_tvweight"]
    iota  = cfg["mdl"]["iota_edgegate"]
    rho1  = cfg["opt"]["rho1_betasplit"]
    rho2  = cfg["opt"]["rho2_Wsplit"]
    zeta1 = cfg["opt"]["zeta1_dprox"]
    # zeta2_eff is the auto-bumped value from _build_cache (>= Lipschitz
    # bound of the quadratic part of grad f).
    zeta2 = cache["zeta2_eff"]
    omega = float(cfg["opt"].get("beta_extrap_omega", 0.0))

    Wrec = state["img"]["Wrec"]
    b2   = state["img"]["b2"]
    vx   = state["img"]["vx"]
    vy   = state["img"]["vy"]

    d_all     = state["patch"]["d"]
    beta_all  = state["patch"]["beta"]
    beta_prev = state["patch"]["beta_prev"]
    theta_all = state["patch"]["theta"]
    b1_all    = state["patch"]["b1"]

    grid_x = cache["layout"]["grid_x"]
    grid_y = cache["layout"]["grid_y"]

    Kd_img = np.zeros((im_h, im_w))
    Pb_img = np.zeros((im_h, im_w))
    cnt    = np.zeros((im_h, im_w))

    pc = 0
    for xx in grid_x:
        for yy in grid_y:
            sl_x = slice(xx, xx + ps)
            sl_y = slice(yy, yy + ps)

            # MATLAB column-major flatten to match the construction of K and
            # Psi above.
            im_vec = image[sl_x, sl_y].flatten(order="F")
            W_vec  = Wrec [sl_x, sl_y].flatten(order="F")
            b2_vec = b2   [sl_x, sl_y].flatten(order="F")
            v_norm = (np.abs(vx[sl_x, sl_y])
                      + np.abs(vy[sl_x, sl_y])).flatten(order="F")

            b1     = b1_all[:, pc]
            theta  = theta_all[:, pc]
            beta   = beta_all[:, pc]    # beta^(k-1)
            d      = d_all[:, pc]
            older  = beta_prev[:, pc]   # beta^(k-2) (only used if omega>0)

            # ---- d-update (paper eq (11)) ------------------------------
            d_rhs = (Kt @ im_vec
                     - (1.0 + rho2) * (KtPsi @ beta)
                     + zeta1 * d
                     + rho2 * (Kt @ (W_vec + b2_vec)))
            d = A_inv @ d_rhs
            kd_vec = K @ d

            # ---- beta prox-linear update (paper eqs (12)-(14)) ----------
            # beta_hat^(k-1) = beta^(k-1) + omega * (beta^(k-1) - beta^(k-2))
            beta_hat     = beta + omega * (beta - older)
            psi_beta_hat = Psi @ beta_hat
            g_hat        = 1.0 / (1.0 + iota * (psi_beta_hat ** 2))

            # p_hat = grad f(beta_hat); contributions from the data fit, the
            # W-split penalty, and the weighted-TV term.
            p_data = -(Psit @ (im_vec - (kd_vec + psi_beta_hat)))
            p_Wsp  = -rho2 * (Psit @ (W_vec - (kd_vec + psi_beta_hat) + b2_vec))
            p_tv   = -2.0 * nu * iota * (
                Psit @ (v_norm * (g_hat ** 2) * psi_beta_hat))
            p_hat  = p_data + p_Wsp + p_tv

            beta_new  = (rho1 * (theta + b1)
                         + zeta2 * beta_hat
                         - p_hat) / (rho1 + zeta2)

            # ---- theta shrinkage (paper eq (18)) -----------------------
            theta_new = _soft_threshold(beta_new - b1, alpha / rho1)

            # ---- Bregman b1 update -------------------------------------
            b1_new    = b1 + theta_new - beta_new

            # Roll beta_prev BEFORE overwriting beta_all so the next
            # iteration's `older` = beta^(k-1) and not beta^(k).
            beta_prev[:, pc] = beta
            d_all[:, pc]     = d
            beta_all[:, pc]  = beta_new
            theta_all[:, pc] = theta_new
            b1_all[:, pc]    = b1_new

            kd_patch = kd_vec.reshape((ps, ps), order="F")
            pb_patch = (Psi @ beta_new).reshape((ps, ps), order="F")
            Kd_img[sl_x, sl_y] += kd_patch
            Pb_img[sl_x, sl_y] += pb_patch
            cnt   [sl_x, sl_y] += 1.0
            pc += 1

    cnt[cnt == 0] = 1.0
    Kd_img /= cnt
    Pb_img /= cnt
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

    fft2  = np.fft.fft2
    ifft2 = np.fft.ifft2

    rhs   = (rho2 * fft2(Kd + Pb - b2)
             + rho3 * (Dx_conj * fft2(vx + b3x)
                       + Dy_conj * fft2(vy + b3y)))
    denom = rho2 + rho3 * Lap_otf + EPS
    W_new = np.real(ifft2(rhs / denom))

    state["img"]["Wrec"] = W_new
    return W_new


def _v_and_bregman_update(state, cache, cfg):
    """v update (paper eq (20)) and final Bregman updates (eqs (22), (23))."""
    rho3 = cfg["opt"]["rho3_gradsplit"]
    nu   = cfg["mdl"]["nu_tvweight"]
    iota = cfg["mdl"]["iota_edgegate"]

    Wrec = state["img"]["Wrec"]
    Kd   = state["img"]["Kd"]
    Pb   = state["img"]["Pb"]
    b2   = state["img"]["b2"]
    b3x  = state["img"]["b3x"]
    b3y  = state["img"]["b3y"]

    g = 1.0 / (1.0 + iota * (Pb ** 2))

    # Forward differences to match the FFT operator used in the W solve.
    Wx, Wy = _forward_grad(Wrec)
    thresh = (nu / rho3) * g
    vx_new = _soft_threshold(Wx - b3x, thresh)
    vy_new = _soft_threshold(Wy - b3y, thresh)

    b3x_new = b3x + vx_new - Wx
    b3y_new = b3y + vy_new - Wy
    b2_new  = b2  + Wrec   - (Kd + Pb)

    state["img"]["vx"]  = vx_new
    state["img"]["vy"]  = vy_new
    state["img"]["b3x"] = b3x_new
    state["img"]["b3y"] = b3y_new
    state["img"]["b2"]  = b2_new


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fit_rkhs_decomposition(image, cfg=None, *, verbose=False, **overrides):
    """Fit the RKHS + approximated-Heaviside decomposition (Stage 1 of the
    paper's two-stage global segmentation method).

    Parameters
    ----------
    image : (H, W) array
        2D image to decompose. Will be cast to float.
    cfg : dict, optional
        Config dict shaped like ``default_config()``.
    verbose : bool
        Print resolved Lipschitz / zeta2 diagnostics and per-iteration
        residual / ||beta||_1.
    **overrides
        Flat keyword overrides. Names route to the matching cfg sub-section.
        Examples: ``gamma_smoothpen=1e-5``, ``nu_tvweight=5e-3``,
        ``maxiter=50``, ``zeta2_betaprox_safety=2.0``.

    Returns
    -------
    result : dict with keys
        Kd, Psi_beta, M, d, beta, cfg, history, diagnostics
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D (H, W)")

    cfg = deepcopy(default_config()) if cfg is None else deepcopy(cfg) #why deepcopy?
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    cache = _build_cache(image.shape, cfg)
    state = _initialize_state(image, cache)

    if verbose:
        print(f"[stage1] ||Psi^T Psi||_op = {cache['PtP_op']:.3f}, "
              f"L_beta = (1+rho2)||Psi^T Psi|| = {cache['L_beta']:.3f}, "
              f"zeta2_eff = {cache['zeta2_eff']:.3f} "
              f"(user zeta2 = {cfg['opt']['zeta2_betaprox']:.3f}, "
              f"safety = {cfg['opt'].get('zeta2_betaprox_safety', 0.0)})")

    n_iter  = int(cfg["opt"]["maxiter"])
    history = []

    for it in range(n_iter):
        _patch_update(image, state, cache, cfg)
        _W_update(state, cache, cfg)
        _v_and_bregman_update(state, cache, cfg)

        Kd = state["img"]["Kd"]
        Pb = state["img"]["Pb"]
        M  = Kd + Pb
        residual = float(np.linalg.norm(image - M))

        history.append({
            "iter":     it,
            "Kd":       Kd.copy(),
            "Pb":       Pb.copy(),
            "M":        M.copy(),
            "residual": residual,
        })
        if verbose:
            beta_l1 = float(np.sum(np.abs(state["patch"]["beta"])))
            print(f"iter {it+1:3d}/{n_iter}  "
                  f"residual={residual:.6f}  |beta|_1={beta_l1:.6f}")

    return {
        "Kd":       state["img"]["Kd"],
        "Psi_beta": state["img"]["Pb"],
        "M":        state["img"]["Kd"] + state["img"]["Pb"],
        "d":        state["patch"]["d"],
        "beta":     state["patch"]["beta"],
        "cfg":      cfg,
        "history":  history,
        # Raw basis matrices and the d-update inverse, for users who want to
        # inspect conditioning / column-overlap diagnostics directly.
        "matrices": {
            "K":       cache["K"],        # (ps^2, ps^2)
            "Psi":     cache["Psi"],      # (ps^2, num_offsets * L)
            "A_d_inv": cache["A_d_inv"],  # (ps^2, ps^2)
            "KtPsi":   cache["KtPsi"],    # (ps^2, num_offsets * L)
        },
        "diagnostics": {
            "K_op":      cache["K_op"],
            "PtP_op":    cache["PtP_op"],
            "L_beta":    cache["L_beta"],
            "zeta2_eff": cache["zeta2_eff"],
        },
    }


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

def _apply_overrides(cfg, overrides):
    """Route flat keyword overrides into the nested cfg dict."""
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
