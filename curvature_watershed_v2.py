"""
Curvature-based seeded watershed segmentation for 2D cell images.

Implements the method from:
  Atta-Fosu et al., J. Imaging 2016.
  "3D Clumped Cell Segmentation Using Curvature Based Seeded Watershed"

2D case: grayscale image f(x,y) as surface (u, v, f(u,v)) in R^3.

Shape matrix (from paper eq. 5):

    A = -II_p * Ip^{-1} = -(1/l) * H_f * Ip^{-1}

  H_f = [[fuu, fuv], [fuv, fvv]]
  Ip  = [[1+fu^2, fu*fv], [fu*fv, 1+fv^2]]
  l   = sqrt(1 + fu^2 + fv^2)

Note: the negative sign comes from eq. (5) in the paper: A = -II_p * Ip^{-1}.
The explicit matrix formula on p.7 of the paper omits this sign, but without
it the eigenvalues are negative at bright peaks (where they should be positive
per the paper convention that k_i >= 0 at convex dome regions).

Returns:
  Casorati curvature (magnitude): C = sqrt((k1^2 + k2^2) / 2)
  Shape index (quality):          S = arctan((k1 + k2) / (k2 - k1))

Dependencies: numpy only (+ matplotlib for demo).
"""

from __future__ import annotations
import numpy as np


def _gaussian_kernel_1d(sigma, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        return np.array([1.0])
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolve_separable(image, kernel):
    r = len(kernel) // 2
    padded = np.pad(image, ((r, r), (0, 0)), mode="reflect")
    tmp = np.zeros_like(image)
    for i, w in enumerate(kernel):
        tmp += w * padded[i : i + image.shape[0], :]
    padded = np.pad(tmp, ((0, 0), (r, r)), mode="reflect")
    out = np.zeros_like(image)
    for j, w in enumerate(kernel):
        out += w * padded[:, j : j + image.shape[1]]
    return out


def gaussian_filter_m(image, sigma):
    if sigma <= 0:
        return image.copy()
    k = _gaussian_kernel_1d(sigma)
    return _convolve_separable(image, k)


def compute_ct(image, sigma=2.0):
    """Compute Casorati curvature and Shape index from a 2D grayscale image.

    Parameters
    ----------
    image : 2D array
        Grayscale image treated as a surface (u, v, f(u,v)).
    sigma : float
        Gaussian smoothing scale applied before differentiation.

    Returns
    -------
    casorati : 2D array
        Casorati curvature C = sqrt((k1^2 + k2^2) / 2).
    shape_index : 2D array
        Shape index S = arctan((k1 + k2) / (k2 - k1)).
        Undefined (set to 0) at umbilical points where k1 == k2.
    """
    f = image.astype(np.float64)
    rows, cols = f.shape

    if sigma > 0:
        f = gaussian_filter_m(f, sigma=sigma)

    # Grid spacing: normalize so the larger dimension spans [0, 1].
    # Makes curvatures resolution-invariant.
    h = 1.0 / max(rows, cols)

    # Derivatives via central finite differences.
    fu = np.gradient(f, h, axis=0)
    fv = np.gradient(f, h, axis=1)
    fuu = np.gradient(fu, h, axis=0)
    fuv = np.gradient(fu, h, axis=1)
    fvv = np.gradient(fv, h, axis=1)

    l = np.sqrt(1.0 + fu**2 + fv**2)

    E = 1.0 + fu**2
    F = fu * fv
    G = 1.0 + fv**2
    det_Ip = E * G - F * F

    inv_E = G / det_Ip
    inv_F = -F / det_Ip
    inv_G = E / det_Ip

    # A = -(1/l) * H_f * Ip^{-1}   [paper eq. 5]
    inv_l = -1.0 / l
    a11 = inv_l * (fuu * inv_E + fuv * inv_F)
    a12 = inv_l * (fuu * inv_F + fuv * inv_G)
    a21 = inv_l * (fuv * inv_E + fvv * inv_F)
    a22 = inv_l * (fuv * inv_F + fvv * inv_G)

    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = np.sqrt(np.maximum(trace**2 - 4.0 * det, 0.0))

    k1 = 0.5 * (trace + disc)
    k2 = 0.5 * (trace - disc)

    # Casorati curvature (magnitude measure)
    casorati = np.sqrt((k1**2 + k2**2) / 2.0)

    # Shape index (quality measure)
    denom = k2 - k1
    # Guard against umbilical points where k1 == k2
    umbilical = np.abs(denom) < 1e-12
    safe_denom = np.where(umbilical, 1.0, denom)
    shape_index = np.arctan((k1 + k2) / safe_denom)
    shape_index[umbilical] = 0.0

    return casorati, shape_index
