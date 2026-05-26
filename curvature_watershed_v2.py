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

Ct = max(k1,0) * max(k2,0)  where k1,k2 = eigenvalues of A.

Dependencies: numpy only (+ matplotlib for demo).
"""

from __future__ import annotations
import numpy as np
from collections import deque


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


def _label_connected_components(mask):
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


def _watershed_from_markers(elevation, markers):
    rows, cols = elevation.shape
    labels = markers.copy()
    visited = labels > 0
    e_min, e_max = elevation.min(), elevation.max()
    e_range = e_max - e_min if e_max > e_min else 1.0
    n_buckets = 1 << 16
    buckets = [[] for _ in range(n_buckets)]

    def bkt(val):
        return min(int((val - e_min) / e_range * (n_buckets - 1)),
                   n_buckets - 1)

    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if labels[r, c] > 0:
                for dy, dx in nbrs:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if labels[nr, nc] == 0:
                            buckets[bkt(elevation[r, c])].append((r, c))
                            break

    for b in range(n_buckets):
        queue = buckets[b]
        qi = 0
        while qi < len(queue):
            r, c = queue[qi]
            qi += 1
            for dy, dx in nbrs:
                nr, nc = r + dy, c + dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc]:
                        visited[nr, nc] = True
                        labels[nr, nc] = labels[r, c]
                        nb = bkt(elevation[nr, nc])
                        if nb == b:
                            queue.append((nr, nc))
                        else:
                            buckets[nb].append((nr, nc))
    return labels


def compute_ct(image, sigma=2.0, eps = 1, ct_threshold=0.0):
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

    k1_plus = np.maximum(k1, 0.0)
    k2_plus = np.maximum(k2, 0.0)

    ct = k1_plus * k2_plus
    ellip_ct = (k1 > -eps) * (k2 > -eps) #experiment
    if ct_threshold > 0:
        ct[ct < ct_threshold] = 0.0
    return ct, ellip_ct, k1, k2


def geodesic_watershed(ct, grad_mag=None, image=None,
                       sigma_grad=1.0, ct_label_threshold=0.0,
                       min_seed_size=5):
    if grad_mag is None and image is None:
        raise ValueError("Provide either grad_mag or image.")

    if grad_mag is None:
        img = image.astype(np.float64)
        if sigma_grad > 0:
            img = gaussian_filter_m(img, sigma=sigma_grad)
        gy = np.gradient(img, axis=0)
        gx = np.gradient(img, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)

    seed_mask = ct > ct_label_threshold
    markers = _label_connected_components(seed_mask)

    if min_seed_size > 1:
        for rid in range(1, markers.max() + 1):
            if np.sum(markers == rid) < min_seed_size:
                markers[markers == rid] = 0
        unique_ids = np.unique(markers)
        unique_ids = unique_ids[unique_ids > 0]
        new_markers = np.zeros_like(markers)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_markers[markers == old_id] = new_id
        markers = new_markers

    if markers.max() == 0:
        return np.zeros_like(ct, dtype=np.int32)

    labels = _watershed_from_markers(grad_mag, markers)
    return labels.astype(np.int32)


def _draw_ellipse(image, cy, cx, ry, rx, angle_deg=0.0, intensity=200.0):
    rows, cols = image.shape
    yy, xx = np.ogrid[:rows, :cols]
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dy = yy - cy
    dx = xx - cx
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy
    r2 = (u / rx)**2 + (v / ry)**2
    image += intensity * np.exp(-3.0 * r2)


def generate_cell_image(shape=(256, 256), background=10.0,
                        noise_std=5.0, seed=42):
    rng = np.random.default_rng(seed)
    img = np.full(shape, background, dtype=np.float64)
    H, W = shape
    oblongs = [
        (0.20*H, 0.18*W, 0.10*H, 0.04*W, 0, 210),
        (0.20*H, 0.12*W, 0.09*H, 0.03*W, 0, 190),
        (0.20*H, 0.38*W, 0.31*H, 0.05*W, 0, 200),
        (0.25*H, 0.45*W, 0.31*H, 0.05*W, 0, 200),
        (0.15*H, 0.5*W, 0.31*H, 0.05*W, 0, 200),
    ]
    cluster_cy, cluster_cx = 0.65*H, 0.65*W
    radii = [0.07*H, 0.065*H, 0.06*H, 0.055*H, 0.07*H]
    angles = np.linspace(0, 2*np.pi, len(radii), endpoint=False)
    spacing = 0.05 * H
    circles = []
    for a, r in zip(angles, radii):
        cy = cluster_cy + spacing * np.sin(a)
        cx = cluster_cx + spacing * np.cos(a)
        circles.append((cy, cx, r, r*0.95, 0, 180 + rng.uniform(-10, 10)))
    for cy, cx, ry, rx, angle, intensity in oblongs + circles:
        _draw_ellipse(img, cy, cx, ry, rx, angle, intensity)
    if noise_std > 0:
        img = img + rng.normal(0, noise_std, shape)
    return np.clip(img, 0, 255)


def segment_cells(image, sigma=2.0, ct_threshold=0.0,
                  sigma_grad=1.0, min_seed_size=5):
    ct, ellip_ct = compute_ct(image, sigma=sigma, ct_threshold=ct_threshold)
    labels = geodesic_watershed(
        ct, image=image, sigma_grad=sigma_grad,
        min_seed_size=min_seed_size)
    return ct, ellip_ct, labels
