"""
Curvature-based seeded watershed for 2D cell segmentation.
Core library: compute_ct, geodesic_watershed, generate_cell_image, segment_cells.
"""
from __future__ import annotations
import numpy as np
from collections import deque


def _gauss_k1d(sigma, trunc=4.0):
    r = int(trunc * sigma + 0.5)
    if r == 0:
        return np.array([1.0])
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _conv_sep(img, kern):
    r = len(kern) // 2
    p = np.pad(img, ((r, r), (0, 0)), mode="reflect")
    t = np.zeros_like(img)
    for i, w in enumerate(kern):
        t += w * p[i:i + img.shape[0], :]
    p = np.pad(t, ((0, 0), (r, r)), mode="reflect")
    o = np.zeros_like(img)
    for j, w in enumerate(kern):
        o += w * p[:, j:j + img.shape[1]]
    return o


def gaussian_filter(img, sigma):
    if sigma <= 0:
        return img.copy()
    return _conv_sep(img, _gauss_k1d(sigma))


def _label_cc(mask):
    labs = np.zeros(mask.shape, dtype=np.int32)
    cur = 0
    R, C = mask.shape
    for r in range(R):
        for c in range(C):
            if mask[r, c] and labs[r, c] == 0:
                cur += 1
                q = deque([(r, c)])
                labs[r, c] = cur
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < R and 0 <= nx < C:
                            if mask[ny, nx] and labs[ny, nx] == 0:
                                labs[ny, nx] = cur
                                q.append((ny, nx))
    return labs


def _watershed(elev, markers):
    R, C = elev.shape
    labs = markers.copy()
    vis = labs > 0
    emin, emax = elev.min(), elev.max()
    er = emax - emin if emax > emin else 1.0
    NB = 1 << 16
    bkts = [[] for _ in range(NB)]
    def bk(v):
        return min(int((v - emin) / er * (NB - 1)), NB - 1)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(R):
        for c in range(C):
            if labs[r, c] > 0:
                for dy, dx in nbrs:
                    nr, nc = r+dy, c+dx
                    if 0 <= nr < R and 0 <= nc < C and labs[nr, nc] == 0:
                        bkts[bk(elev[r, c])].append((r, c))
                        break
    for b in range(NB):
        q = bkts[b]
        qi = 0
        while qi < len(q):
            r, c = q[qi]; qi += 1
            for dy, dx in nbrs:
                nr, nc = r+dy, c+dx
                if 0 <= nr < R and 0 <= nc < C and not vis[nr, nc]:
                    vis[nr, nc] = True
                    labs[nr, nc] = labs[r, c]
                    nb = bk(elev[nr, nc])
                    if nb == b:
                        q.append((nr, nc))
                    else:
                        bkts[nb].append((nr, nc))
    return labs


def compute_ct(image, sigma=2.0, ct_threshold=0.0):
    """Curvature seed map Ct = max(k1,0)*max(k2,0)."""
    f = image.astype(np.float64)
    if sigma > 0:
        f = gaussian_filter(f, sigma)
    fu = np.gradient(f, axis=0)
    fv = np.gradient(f, axis=1)
    fuu = np.gradient(fu, axis=0)
    fuv = np.gradient(fu, axis=1)
    fvv = np.gradient(fv, axis=1)
    l = np.sqrt(1.0 + fu**2 + fv**2)
    E = 1.0 + fu**2; F = fu*fv; G = 1.0 + fv**2
    dI = E*G - F*F
    iE = G/dI; iF = -F/dI; iG = E/dI
    il = 1.0/l
    a11 = il*(fuu*iE + fuv*iF)
    a12 = il*(fuu*iF + fuv*iG)
    a21 = il*(fuv*iE + fvv*iF)
    a22 = il*(fuv*iF + fvv*iG)
    tr = a11 + a22
    dt = a11*a22 - a12*a21
    disc = np.sqrt(np.maximum(tr**2 - 4.0*dt, 0.0))
    k1 = 0.5*(tr + disc)
    k2 = 0.5*(tr - disc)
    ct = np.maximum(k1, 0.0) * np.maximum(k2, 0.0)
    if ct_threshold > 0:
        ct[ct < ct_threshold] = 0.0
    return ct
