"""
viscous_watershed.py
====================

Reference implementation of the *viscous watershed transform* of

    C. Vachier and F. Meyer,
    "The Viscous Watershed Transform",
    Journal of Mathematical Imaging and Vision 22: 251-267, 2005.

The idea: instead of simulating a viscous fluid during flooding, we modify the
topographic relief `f` (typically a gradient / contour image) into a new relief
`T(f)` such that running the *standard* (non-viscous) watershed on `T(f)`
reproduces the *viscous* watershed of `f`. This module returns the modified
relief; feed it to any standard watershed (e.g.
`skimage.segmentation.watershed`).

Two models (Sec. 3 of the paper):

* Oil model      T(f):  level-set indexed closing
      X_h(T(f)) = phi_{r(h)}( X_h(f) )
  where X_h(f) = { p : f(p) >= h } is the level set at altitude h, phi_r is the
  flat morphological closing by a disk of radius r, and r(h) is a *decreasing*
  viscosity radius (strong smoothing at low altitude, weak at high altitude).

* Mercury model  T~(f):  infimum of translated closings
      T~(f) = inf_{t >= 0}  phi_{r(t)}( f + t )
  with (f + t) clamped to the top level M.

Both are extensive (T(f) >= f) and finer than the plain closing phi_{r0}:
      f <= T~(f) <= T(f) <= phi_{r0}(f).

NB: the paper's p.11 parenthetical "anti-extensive (T(f) <= f)" contradicts its
own "f <= T(f) <= phi_{r0}(f)" two lines later and the fact that T is a closing.
A closing is *extensive*. We implement T(f) >= f and assert it.

Backend: uses scikit-image if available, else falls back to scipy.ndimage.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

# ---- Morphology backend ---------------------------------------------------
# Prefer scikit-image; fall back to scipy.ndimage. Only one is required.
_BACKEND = None
try:
    from skimage.morphology import (
        disk as _disk,
        binary_closing as _binary_closing,
        closing as _grey_closing,
        dilation as _dilation,
        erosion as _erosion,
    )
    _BACKEND = "skimage"

    def disk(r):
        return _disk(int(r))

    def binary_closing(img, fp):
        return _binary_closing(img, fp)

    def closing(img, fp):
        return _grey_closing(img, fp)

    def dilation(img, fp):
        return _dilation(img, fp)

    def erosion(img, fp):
        return _erosion(img, fp)

except ImportError:
    try:
        from scipy import ndimage as _ndi
        _BACKEND = "scipy"

        def disk(r):
            """Boolean disk of radius r (matches skimage.morphology.disk)."""
            r = int(r)
            y, x = np.ogrid[-r:r + 1, -r:r + 1]
            return (x * x + y * y <= r * r)

        def binary_closing(img, fp):
            return _ndi.binary_closing(img, structure=fp)

        def closing(img, fp):
            return _ndi.grey_closing(img, footprint=fp)

        def dilation(img, fp):
            return _ndi.grey_dilation(img, footprint=fp)

        def erosion(img, fp):
            return _ndi.grey_erosion(img, footprint=fp)

    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Need scikit-image or scipy: pip install scikit-image  (or scipy)"
        ) from e


# --------------------------------------------------------------------------- #
# Viscosity radius schedule
# --------------------------------------------------------------------------- #
def linear_radius(level: int, r0: float, n_levels: int) -> float:
    """Default viscosity schedule: r decreases linearly from r0 (at level 0)
    to 0 (at the top level n_levels-1):

        r(h) = r0 * (1 - h / (n_levels - 1)).

    Low altitude -> large disk -> strong regularization; high altitude (strong
    contour) -> small disk -> little regularization. Override `radius_fn` to use
    another decreasing schedule, e.g. a fixed decrement r(h) = max(r0 - h, 0)
    as in the paper's thin-contour example (Sec. 4.1).
    """
    if n_levels <= 1:
        return r0
    return r0 * (1.0 - level / (n_levels - 1))


# --------------------------------------------------------------------------- #
# Quantization helpers
# --------------------------------------------------------------------------- #
def _quantize(relief: np.ndarray, n_levels: int):
    """Map an arbitrary 2D relief to integer levels 0 .. n_levels-1.

    Returns (levels_int, lo, hi) so the result can be mapped back later.
    """
    relief = np.asarray(relief, dtype=float)
    if relief.ndim != 2:
        raise ValueError(f"relief must be 2D, got shape {relief.shape}")
    lo = float(relief.min())
    hi = float(relief.max())
    if hi <= lo:
        return np.zeros_like(relief, dtype=int), lo, hi
    scaled = (relief - lo) / (hi - lo) * (n_levels - 1)
    return np.round(scaled).astype(int), lo, hi


def _dequantize(levels: np.ndarray, lo: float, hi: float, n_levels: int):
    """Map integer levels back to the original [lo, hi] intensity range."""
    if n_levels <= 1 or hi <= lo:
        return levels.astype(float)
    return lo + (levels.astype(float) / (n_levels - 1)) * (hi - lo)


def _disk_footprint(r: float):
    """Disk structuring element; None if radius rounds to < 1 (identity op)."""
    ri = int(round(r))
    if ri < 1:
        return None
    return disk(ri)


# --------------------------------------------------------------------------- #
# Oil model:  X_h(T(f)) = phi_{r(h)}( X_h(f) )
# --------------------------------------------------------------------------- #
def viscous_transform_oil(
    levels: np.ndarray,
    n_levels: int,
    r0: float,
    radius_fn: Callable[[int, float, int], float],
) -> np.ndarray:
    """Oil viscous closing on an already-quantized integer relief.

    Reconstructs T(f) by stacking closed level sets:
        T(f)(p) = max{ h : p in phi_{r(h)}( X_h(f) ) }.
    """
    out = np.zeros(levels.shape, dtype=float)
    for h in range(1, n_levels):
        Xh = levels >= h                      # level set at altitude h
        fp = _disk_footprint(radius_fn(h, r0, n_levels))
        # OR with Xh: a closing is extensive by definition (guards against
        # backend border artifacts that could otherwise drop edge pixels)
        Ch = Xh if fp is None else (binary_closing(Xh, fp) | Xh)
        # ascending h overwrites -> final value is the max h with p in Ch
        out[Ch] = h
    return out


# --------------------------------------------------------------------------- #
# Mercury model:  T~(f) = inf_t phi_{r(t)}( f + t )
# --------------------------------------------------------------------------- #
def viscous_transform_mercury(
    levels: np.ndarray,
    n_levels: int,
    r0: float,
    radius_fn: Callable[[int, float, int], float],
) -> np.ndarray:
    """Mercury viscous transform on an already-quantized integer relief."""
    M = n_levels - 1
    out = np.full(levels.shape, float(M), dtype=float)   # neutral for the inf
    for t in range(0, n_levels):
        shifted = np.minimum(levels + t, M)              # f + t, clamped to M
        fp = _disk_footprint(radius_fn(t, r0, n_levels))
        # max with input: grayscale closing is extensive by definition
        g = shifted if fp is None else np.maximum(closing(shifted, fp), shifted)
        out = np.minimum(out, g.astype(float))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def viscous_transform(
    relief: np.ndarray,
    model: str = "oil",
    r0: float = 20.0,
    n_levels: int = 256,
    radius_fn: Optional[Callable[[int, float, int], float]] = None,
    return_scale: str = "original",
    check: bool = True,
) -> np.ndarray:
    """Compute the viscous transform of a topographic relief.

    Parameters
    ----------
    relief : 2D np.ndarray
        Relief to flood, usually a gradient / contour image. Any dtype/range;
        internally quantized to `n_levels` integer levels.
    model : {"oil", "mercury"}
        Oil: viscosity indexed on flooding level (low gradient = blur = smooth
        more; high gradient = sharp = smooth less). Mercury: viscosity indexed
        on valley depth (high contours leak and need more regularization). They
        coincide when all minima sit at level 0 (e.g. gradients of
        piecewise-constant images).
    r0 : float
        Reference viscosity radius r(0) = max disk radius (the paper's "size").
    n_levels : int
        Number of quantization levels. Gradient images usually have few levels;
        fewer levels => faster (one closing per level).
    radius_fn : callable(level, r0, n_levels) -> float, optional
        Decreasing viscosity schedule. Defaults to `linear_radius`.
    return_scale : {"original", "levels"}
        "original": rescale output back to relief's [min, max]. "levels": return
        in 0..n_levels-1 level units. Watershed is invariant to monotone
        rescaling, so either floods identically.
    check : bool
        If True, assert the result is extensive (T(f) >= f) in level units.

    Returns
    -------
    2D np.ndarray (float)
        The viscous-transformed relief; feed it to a standard watershed.
    """
    if radius_fn is None:
        radius_fn = linear_radius
    levels, lo, hi = _quantize(relief, n_levels)

    model = model.lower()
    if model == "oil":
        out_levels = viscous_transform_oil(levels, n_levels, r0, radius_fn)
    elif model == "mercury":
        out_levels = viscous_transform_mercury(levels, n_levels, r0, radius_fn)
    else:
        raise ValueError(f"model must be 'oil' or 'mercury', got {model!r}")

    if check:
        assert out_levels.min() >= -1e-9, "negative levels produced"
        assert (out_levels + 1e-6 >= levels).all(), (
            "viscous transform should be extensive (T(f) >= f)"
        )

    if return_scale == "levels":
        return out_levels
    elif return_scale == "original":
        return _dequantize(out_levels, lo, hi, n_levels)
    else:
        raise ValueError("return_scale must be 'original' or 'levels'")


def morphological_gradient(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Flat morphological gradient dilation(f) - erosion(f) by a disk.

    The relief to flood is typically the gradient norm; the paper uses the
    morphological gradient delta(f) - eps(f) (dilation minus erosion).
    """
    fp = disk(radius)
    img = np.asarray(image, dtype=float)
    return dilation(img, fp) - erosion(img, fp)


__all__ = [
    "viscous_transform",
    "viscous_transform_oil",
    "viscous_transform_mercury",
    "morphological_gradient",
    "linear_radius",
]


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # synthetic thin, broken, noisy contour line (cf. paper Fig. 22)
    f = np.zeros((120, 120), dtype=float)
    ys = np.arange(120)
    xs = (60 + 25 * np.sin(ys / 12.0)).astype(int)
    for y, x in zip(ys, xs):
        if y % 7:                       # break the line
            f[y, x] = 200
    f += rng.normal(0, 8, f.shape)      # noise
    f = np.clip(f, 0, 255)

    print("backend:", _BACKEND)
    for model in ("oil", "mercury"):
        g = viscous_transform(f, model=model, r0=15, n_levels=64)
        print(f"{model:8s} in[min,max]=({f.min():.1f},{f.max():.1f}) "
              f"out[min,max]=({g.min():.1f},{g.max():.1f}) shape={g.shape}")
    print("OK")
