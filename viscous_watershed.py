"""
Oil-model viscous watershed transform.

Implements the *oil* viscous closing  T(f)  of

    C. Vachier and F. Meyer,
    "The Viscous Watershed Transform",
    Journal of Mathematical Imaging and Vision 22:251-267, 2005.

The idea (paper, Sec. 3.2-3.3): instead of flooding a relief with a viscous
fluid, modify the relief itself so that flooding the modified relief with an
ordinary (non-viscous) fluid yields the same lakes.  The standard watershed
algorithm can then be run on the returned relief.

For the oil model the modification is a *viscous closing*: each level set
X_h(f) = {x : f(x) >= h} is closed by a disk of radius r(h), where the
viscosity radius r(h) decreases with the altitude h.  The paper's explicit
formula (Sec 3.3) is

    T(f) = \\/_{h>=0}  h * chi_h( phi_{r(h)}(f) )
         = \\/_{h>=0}  h * chi_{ phi_{r(h)}(X_h(f)) }

so, pixelwise,

    T(f)(p) = max{ h : p in phi_{r(h)}(X_h(f)) }.

Under the idealization that the closed level sets are nested this is equivalent
to the level-set statement  X_h(T(f)) = phi_{r(h)}(X_h(f)).

Low contours (small h) get a large disk -> strong regularization; high
contours (large h) get a small disk -> they are preserved.  A *constant*
r(h) = r0 reduces T to the ordinary morphological closing phi_{r0}.

Properties: T is idempotent, increasing, and EXTENSIVE, sitting between the
identity and a plain closing:

    f <= T(f) <= phi_{r0}(f)

NOTE on a contradiction in the paper.  Section 3.3 calls T "anti-extensive
(T(f) <= f)" yet one line later states "f <= T(f) <= phi_{r0}(f)".  These two
statements are mutually exclusive.  A morphological closing is extensive
(phi(f) >= f), and the construction here closes the land level sets X_h(f)
(equivalently opens/shrinks the lakes), which raises the relief.  So the
correct property is f <= T(f): T is EXTENSIVE, and the "anti-extensive
(T(f) <= f)" phrasing in the paper appears to be an error.  This
implementation produces T >= f, consistent with the f <= T <= phi_{r0}
inequality and with the closing interpretation.  Flagged, not resolved.

Dependencies: numpy, scipy.ndimage only.
Performance: the per-level binary closing uses the Euclidean distance
transform, so cost is O(N) per gray level and independent of the disk radius.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


# --------------------------------------------------------------------------- #
# structuring element
# --------------------------------------------------------------------------- #
def disk(radius: float) -> np.ndarray:
    """Flat boolean disk structuring element: {(x, y) : x^2 + y^2 <= r^2}.

    Provided as a utility (e.g. for your own watershed/markers); the internal
    closing uses an equivalent distance-transform implementation for speed.

    A radius < 1 returns a single pixel (the identity structuring element),
    so closing by it is the identity operation.
    """
    r = int(round(radius))
    if r < 1:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= (r * r)


# --------------------------------------------------------------------------- #
# binary closing with safe borders
# --------------------------------------------------------------------------- #
def _binary_closing_disk(mask: np.ndarray, radius: float) -> np.ndarray:
    """Flat binary closing of `mask` by a Euclidean disk of the given radius.

    Implemented with the Euclidean distance transform rather than an explicit
    structuring-element pass:

        dilation by disk r : pixels within Euclidean distance r of foreground
                             -> edt(~mask) <= r
        erosion  by disk r : foreground pixels with no background within r
                             -> edt(dilation) > r

    This is bit-for-bit identical to a closing by the boolean disk
    `disk(r)` (both test dx^2 + dy^2 <= r^2), but runs in O(N) per level
    independently of r, instead of O(N * disk_area).  That difference is what
    keeps the full transform from taking minutes on real images.

    The array is padded by (radius + 1) with edge replication before the
    transform and cropped afterwards, so foreground touching the image border
    is not spuriously eroded (a standard closing artefact at edges).
    """
    r = int(round(radius))
    if r < 1:
        return mask.astype(bool, copy=True)
    pad = r + 1
    m = np.pad(mask, pad, mode="edge")
    dil = distance_transform_edt(~m) <= r          # dilation by disk r
    clo = distance_transform_edt(dil) > r          # erosion  by disk r
    return clo[pad:-pad, pad:-pad]


# --------------------------------------------------------------------------- #
# default viscosity profile r(h)
# --------------------------------------------------------------------------- #
def linear_radius(r0: float, n_levels: int):
    """Return r(h) = r0 * (M - h) / M, M = n_levels - 1, clamped at 0.

    r0 at the bottom (h = 0, low gradient / fuzzy contour -> strong smoothing)
    down to 0 at the top (h = M, high gradient / sharp contour -> preserved).
    This is the default profile.

    The paper's synthetic examples (Sec 4.1) instead decrease r by 1 per gray
    level above the source:  radius_fn = lambda h: max(r0 - h, 0).  Pass such a
    callable as `radius_fn` to reproduce that behaviour.
    """
    M = max(n_levels - 1, 1)

    def r(h: int) -> float:
        return max(r0 * (M - h) / M, 0.0)

    return r


# --------------------------------------------------------------------------- #
# main transform
# --------------------------------------------------------------------------- #
def viscous_closing_oil(
    f: np.ndarray,
    r0: float = 20.0,
    n_levels: int = 256,
    radius_fn=None,
    return_float: bool = True,
):
    """Oil-model viscous closing  T(f)  of a topographic relief.

    Parameters
    ----------
    f : 2-D array
        The relief to be flooded, e.g. a gradient-magnitude image.  Higher
        values = contours / crest lines.  May be float or integer.
    r0 : float
        Reference viscosity radius (disk radius, in pixels, at the lowest /
        most-viscous level).  Ignored if `radius_fn` is given.
    n_levels : int
        Number of quantization levels.  f is linearly quantized to integer
        levels [0, M] with M = n_levels - 1, and the closing is applied level
        by level.  Fewer levels = faster (the paper notes the transform is
        normally run on gradient images with few gray levels).
    radius_fn : callable, optional
        h -> r(h), the disk radius at integer level h in [0, M].  If None,
        uses `linear_radius(r0, n_levels)`.  Use `lambda h: r0` to recover the
        standard morphological closing by a disk of radius r0.
    return_float : bool
        If True (default) the result is rescaled back to f's original value
        range as a float array.  If False the integer level image [0, M] is
        returned.

    Returns
    -------
    T : 2-D array
        The modified relief.  Run any standard watershed (e.g.
        skimage.segmentation.watershed) on this.  T is EXTENSIVE:
        f <= T <= phi_{r0}(f)  (up to quantization).  See the module
        docstring for the sign/extensivity note re. the paper.
    """
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError("f must be a 2-D array")

    M = n_levels - 1
    if M < 1:
        raise ValueError("n_levels must be >= 2")

    # quantize the relief to integer levels [0, M]
    fmin = float(f.min())
    fmax = float(f.max())
    if fmax > fmin:
        fq = np.round((f - fmin) / (fmax - fmin) * M).astype(np.int64)
    else:
        fq = np.zeros_like(f, dtype=np.int64)

    if radius_fn is None:
        radius_fn = linear_radius(r0, n_levels)

    # This is the paper's explicit formula (Sec 3.3):
    #   T(f) = \/_{h>=0} h * chi_h( phi_{r(h)}(f) )
    # i.e.  T(p) = max{ h : p in phi_{r(h)}(X_h(f)) }.
    # Iterating h increasing and overwriting yields that maximum.
    #
    # Subtlety: the idealized identity X_h(T(f)) = phi_{r(h)}(X_h(f)) holds
    # only when the closed level sets C_h = phi_{r(h)}(X_h(f)) are nested
    # (C_{h+1} subset of C_h).  The paper assumes this, but closings by disks
    # of *different* radii are not guaranteed to be ordered, so C_h need not
    # be perfectly nested.  The sup/max formula above is well defined either
    # way (it just takes the deepest level reached), so we use it directly.
    Tq = np.zeros_like(fq, dtype=np.int64)
    for h in range(1, M + 1):
        level_set = fq >= h                       # X_h(f), binary
        if not level_set.any():
            break                                 # all higher level sets empty
        closed = _binary_closing_disk(level_set, radius_fn(h))
        Tq[closed] = h

    if not return_float:
        return Tq

    # map integer levels back to the original value range
    return fmin + (Tq.astype(np.float64) / M) * (fmax - fmin)


if __name__ == "__main__":
    # tiny smoke test
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    n_levels = 64

    # Extensivity is exact in level space (Tq >= fq).  The float round-trip
    # rescales levels back and can shift a value by up to half a level
    # (~ range / (2 * (n_levels - 1))), so test the clean invariant on levels.
    Tq = viscous_closing_oil(img, r0=8, n_levels=n_levels, return_float=False)
    M = n_levels - 1
    fq = np.round((img - img.min()) / (img.max() - img.min()) * M).astype(np.int64)
    assert Tq.shape == img.shape
    assert np.all(Tq >= fq)                      # EXTENSIVE in level space

    T = viscous_closing_oil(img, r0=8, n_levels=n_levels)  # float output
    tol = (img.max() - img.min()) / (2 * M) + 1e-6
    assert np.all(T >= img - tol)                # extensive up to quantization
    print("ok:", T.shape, T.dtype, float(T.min()), float(T.max()))
