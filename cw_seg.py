"""
Curvature-based seeded watershed segmentation for 2D cell images.

Implements the method from:
  "3D Clumped Cell Segmentation Using Curvature Based Seeded Watershed"
  Atta-Fosu et al., J. Imaging 2016.

Adapted here for the 2D case: a grayscale image f(x,y) is treated as a
surface in R^3 with parametrization X = (u, v, f(u,v)).

The shape matrix (Weingarten operator) is:

    A = (1/l) * H_f * Ip^{-1}

where
    H_f = [[f_uu, f_uv],        (matrix of second-order partials)
            [f_uv, f_vv]]

    Ip  = [[1 + f_u^2,  f_u f_v],   (first fundamental form)
            [f_u f_v,  1 + f_v^2]]

    l   = sqrt(1 + f_u^2 + f_v^2)

The eigenvalues k1, k2 of A are the principal curvatures.
Seeds are identified via:

    Ct = k1^+ * k2^+,    where ki^+ = max(ki, 0)

Connected components of Ct > 0 serve as markers for a geodesic watershed
on |nabla f| (the gradient magnitude of the original image).

Dependencies: numpy only (+ matplotlib for the demo script).
"""

from __future__ import annotations
import numpy as np
from collections import deque


# ======================================================================
# Internal helpers (pure numpy replacements for scipy/skimage)
# ======================================================================

def _gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """1-D Gaussian kernel, normalised to sum to 1."""
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        return np.array([1.0])
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolve_separable(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable 2-D convolution (reflect-pad) using a symmetric 1-D kernel."""
    r = len(kernel) // 2
    # Pad and convolve along axis 0
    padded = np.pad(image, ((r, r), (0, 0)), mode="reflect")
    tmp = np.zeros_like(image)
    for i, w in enumerate(kernel):
        tmp += w * padded[i : i + image.shape[0], :]
    # Convolve along axis 1
    padded = np.pad(tmp, ((0, 0), (r, r)), mode="reflect")
    out = np.zeros_like(image)
    for j, w in enumerate(kernel):
        out += w * padded[:, j : j + image.shape[1]]
    return out


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """2-D Gaussian smoothing (separable, reflect boundary)."""
    if sigma <= 0:
        return image.copy()
    k = _gaussian_kernel_1d(sigma)
    return _convolve_separable(image, k)


def _label_connected_components(mask: np.ndarray) -> np.ndarray:
    """Label connected components of a boolean mask (4-connectivity)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current_label = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and labels[r, c] == 0:
                current_label += 1
                queue = deque()
                queue.append((r, c))
                labels[r, c] = current_label
                while queue:
                    y, x = queue.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))
    return labels


def _watershed_from_markers(elevation: np.ndarray,
                            markers: np.ndarray) -> np.ndarray:
    """
    Priority-flood (Meyer) watershed on *elevation* seeded by *markers*.

    Uses 8-connectivity.  Background (marker == 0) pixels are assigned
    to the nearest seed in terms of the elevation (topographic distance
    approximation via priority queue).
    """
    rows, cols = elevation.shape
    labels = markers.copy()
    visited = labels > 0

    # Build initial queue from boundary pixels of each seed
    # We use a simple list-based bucket sort (quantised elevation).
    # For floating-point elevation, quantise to 2^16 buckets.
    e_min, e_max = elevation.min(), elevation.max()
    e_range = e_max - e_min if e_max > e_min else 1.0
    n_buckets = 1 << 16
    buckets: list[list[tuple[int, int]]] = [[] for _ in range(n_buckets)]

    def _bucket(val):
        return min(int((val - e_min) / e_range * (n_buckets - 1)), n_buckets - 1)

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    # Seed the queue with all labeled pixels adjacent to unlabeled pixels
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] > 0:
                for dy, dx in neighbors:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if labels[nr, nc] == 0:
                            b = _bucket(elevation[r, c])
                            buckets[b].append((r, c))
                            break

    # Flood
    for b in range(n_buckets):
        queue = buckets[b]
        qi = 0
        while qi < len(queue):
            r, c = queue[qi]
            qi += 1
            for dy, dx in neighbors:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc]:
                        visited[nr, nc] = True
                        labels[nr, nc] = labels[r, c]
                        nb = _bucket(elevation[nr, nc])
                        if nb == b:
                            queue.append((nr, nc))
                        else:
                            buckets[nb].append((nr, nc))

    return labels


# ======================================================================
# Curvature detection
# ======================================================================

def compute_ct(image: np.ndarray,
               sigma: float = 2.0,
               ct_threshold: float = 0.0) -> np.ndarray:
    """
    Compute the curvature seed map Ct for a 2D grayscale image.

    The image f(x,y) is interpreted as a surface (u, v, f(u,v)) in R^3.
    The principal curvatures k1, k2 are the eigenvalues of the shape
    matrix  A = (1/l) * H_f * Ip^{-1},  and

        Ct = max(k1, 0) * max(k2, 0).

    Parameters
    ----------
    image : 2D ndarray (float or uint8)
        Grayscale input image.  Will be converted to float64 internally.
    sigma : float, optional
        Standard deviation for Gaussian smoothing applied before computing
        derivatives (mollification step from the paper).  Default 2.0.
    ct_threshold : float, optional
        Values of Ct below this are zeroed out.  Useful for suppressing
        noise.  Default 0.0 (keep all positive Ct).

    Returns
    -------
    ct : 2D ndarray of float64, same shape as *image*.
        The Ct curvature map.  Positive where both principal curvatures
        are positive (locally convex dome regions).
    """
    f = image.astype(np.float64)

    # Mollify (Gaussian smoothing) -----------------------------------------
    if sigma > 0:
        f = gaussian_filter(f, sigma=sigma)

    # First-order partial derivatives (central differences) -----------------
    fu = np.gradient(f, axis=0)   # d/d(row)
    fv = np.gradient(f, axis=1)   # d/d(col)

    # Second-order partial derivatives
    fuu = np.gradient(fu, axis=0)
    fuv = np.gradient(fu, axis=1)
    fvv = np.gradient(fv, axis=1)

    # l = sqrt(1 + fu^2 + fv^2)  -------------------------------------------
    l = np.sqrt(1.0 + fu**2 + fv**2)

    # First fundamental form  Ip = [[E, F], [F, G]]  -----------------------
    E = 1.0 + fu**2
    F = fu * fv
    G = 1.0 + fv**2

    det_Ip = E * G - F * F          # always >= 1, so safe to invert

    # Inverse of Ip (2x2 per pixel)
    inv_E =  G / det_Ip
    inv_F = -F / det_Ip
    inv_G =  E / det_Ip

    # Shape matrix  A = (1/l) * [[fuu, fuv],[fuv, fvv]] * Ip^{-1}  ---------
    inv_l = 1.0 / l
    a11 = inv_l * (fuu * inv_E + fuv * inv_F)
    a12 = inv_l * (fuu * inv_F + fuv * inv_G)
    a21 = inv_l * (fuv * inv_E + fvv * inv_F)
    a22 = inv_l * (fuv * inv_F + fvv * inv_G)

    # Eigenvalues of 2x2 matrix via closed-form ----------------------------
    trace = a11 + a22
    det   = a11 * a22 - a12 * a21
    disc  = np.sqrt(np.maximum(trace**2 - 4.0 * det, 0.0))

    k1 = 0.5 * (trace + disc)
    k2 = 0.5 * (trace - disc)

    # Ct = k1^+ * k2^+  ----------------------------------------------------
    k1_plus = np.maximum(k1, 0.0)
    k2_plus = np.maximum(k2, 0.0)
    ct = k1_plus * k2_plus

    if ct_threshold > 0:
        ct[ct < ct_threshold] = 0.0

    return ct


# ======================================================================
# Geodesic watershed segmentation
# ======================================================================

def geodesic_watershed(ct: np.ndarray,
                       grad_mag: np.ndarray | None = None,
                       image: np.ndarray | None = None,
                       sigma_grad: float = 1.0,
                       ct_label_threshold: float = 0.0,
                       min_seed_size: int = 5) -> np.ndarray:
    """
    Run a seeded watershed using connected components of Ct as markers.

    The watershed is computed on a topographic surface, which is ideally
    |nabla f| (gradient magnitude of the original image).  You can either
    pass *grad_mag* directly, or pass the original *image* and let this
    function compute it.

    Parameters
    ----------
    ct : 2D ndarray
        Curvature seed map (output of ``compute_ct``).
    grad_mag : 2D ndarray, optional
        Pre-computed gradient magnitude |nabla f|.  If supplied, *image*
        is ignored.
    image : 2D ndarray, optional
        Original grayscale image.  Used to compute |nabla f| when
        *grad_mag* is not given.  Exactly one of *grad_mag* or *image*
        must be provided.
    sigma_grad : float, optional
        Gaussian sigma used when computing the gradient magnitude from
        *image*.  Ignored when *grad_mag* is supplied.  Default 1.0.
    ct_label_threshold : float, optional
        Threshold applied to *ct* before labelling connected components.
        Default 0.0 (any positive Ct is a seed pixel).
    min_seed_size : int, optional
        Connected components with fewer than this many pixels are
        discarded (noise suppression).  Default 5.

    Returns
    -------
    labels : 2D ndarray of int32, same shape as *ct*.
        Watershed label array.  Each cell gets a unique positive integer
        label.
    """
    if grad_mag is None and image is None:
        raise ValueError("Provide either grad_mag or image.")

    # Compute gradient magnitude if not supplied ----------------------------
    if grad_mag is None:
        img = image.astype(np.float64)
        if sigma_grad > 0:
            img = gaussian_filter(img, sigma=sigma_grad)
        gy = np.gradient(img, axis=0)
        gx = np.gradient(img, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)

    # Label connected components of Ct as seeds ----------------------------
    seed_mask = ct > ct_label_threshold
    markers = _label_connected_components(seed_mask)

    # Remove tiny components (noise) ----------------------------------------
    if min_seed_size > 1:
        for region_id in range(1, markers.max() + 1):
            if np.sum(markers == region_id) < min_seed_size:
                markers[markers == region_id] = 0
        # Relabel contiguously
        unique_ids = np.unique(markers)
        unique_ids = unique_ids[unique_ids > 0]
        new_markers = np.zeros_like(markers)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_markers[markers == old_id] = new_id
        markers = new_markers

    if markers.max() == 0:
        return np.zeros_like(ct, dtype=np.int32)

    # Watershed on gradient magnitude ---------------------------------------
    labels = _watershed_from_markers(grad_mag, markers)

    return labels.astype(np.int32)


# ======================================================================
# Synthetic test-image generation
# ======================================================================

def _draw_ellipse(image: np.ndarray,
                  cy: float, cx: float,
                  ry: float, rx: float,
                  angle_deg: float = 0.0,
                  intensity: float = 200.0):
    """Draw a filled, Gaussian-profile ellipse onto *image* (in-place)."""
    rows, cols = image.shape
    yy, xx = np.ogrid[:rows, :cols]

    theta = np.deg2ra