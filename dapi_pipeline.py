"""
Post-detector pipeline: raw per-FOV sensor stacks g -> stitched morphology image.

This replays SOFTWARE, not physics. Everything here is a transformation the
real XOA pipeline applied, so your synthetic data must undergo it too.

    per-FOV stack (pe)
      -> lens distortion correction      [resampling: correlates noise]
      -> focus map on 16x16 patches      [per-patch best-Z selection]
      -> deconvolution + background sub  [ringing, intensity compression]
      -> middle slice                    [2D projection]
      -> stitch across FOVs              [seams, blend zones, ghosting]
      -> pyramid / 16-bit / JPEG-2000    [quantization, compression]

Documented constants are marked DOC. Everything else is a knob to fit.
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from scipy.signal import fftconvolve

FOV_ROWS, FOV_COLS = 3520, 2960   # DOC
FOV_OVERLAP = 128                 # DOC, pixels on each edge
FOCUS_PATCH = 16                  # DOC, focus map patch size
DETECTOR_DX = 0.2125              # DOC, um/px


# ----------------------------------------------------------------------
# 1. Lens distortion correction
# ----------------------------------------------------------------------

def correct_distortion(stack, k1=-2e-8, k1_residual=2e-10, order=1):
    """Undo radial lens distortion by resampling.

    Radial model: r_src = r_dst * (1 + k1 * r_dst^2), r in pixels from center.

    THE POINT OF THIS STAGE IS NOT GEOMETRY. It is that interpolation makes
    neighbouring pixels share information, so noise stops being independent
    per pixel. Everything downstream (deconvolution especially) sees
    correlated noise, not white noise. Skipping this and adding correlation
    later will not reproduce the same structure.

    k1_residual is calibration error left over after correction -- real
    systems never null it perfectly, and the residual grows toward corners.
    Use order=1 (bilinear); higher orders correlate noise more aggressively.
    """
    nz, ny, nx = stack.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    r2 = dy**2 + dx**2

    scale = 1.0 + k1_residual * r2          # only the residual is applied:
    src_y = cy + dy * scale                 # the bulk correction is assumed
    src_x = cx + dx * scale                 # to have succeeded

    out = np.empty_like(stack)
    for i in range(nz):
        out[i] = map_coordinates(stack[i], [src_y, src_x],
                                 order=order, mode='nearest')
    return out


# ----------------------------------------------------------------------
# 2. Focus map
# ----------------------------------------------------------------------

def focus_map(stack, patch=FOCUS_PATCH, smooth=2.0):
    """Per-patch best-focus Z index, from a gradient-energy sharpness metric.

    Returns a (ny//patch, nx//patch) integer map.

    Real tissue sections are not flat -- they tilt and wrinkle -- so the
    chosen Z varies across the field. That is why this exists. It is also a
    source of artifact: adjacent patches can land on different Z-slices,
    producing 16-px-scale discontinuities in apparent sharpness. Smoothing
    the map suppresses this; XOA calls it a *global* focus map, implying
    fairly heavy regularization.
    """
    nz, ny, nx = stack.shape
    py, px = ny // patch, nx // patch
    sharp = np.empty((nz, py, px))
    for i in range(nz):
        gy, gx = np.gradient(stack[i].astype(np.float64))
        energy = gy**2 + gx**2
        sharp[i] = (energy[:py*patch, :px*patch]
                    .reshape(py, patch, px, patch).mean(axis=(1, 3)))
    smoothed = gaussian_filter(sharp, sigma=(0, smooth, smooth))
    return np.argmax(smoothed, axis=0)


def extract_focal_substack(stack, fmap, half_depth=2, patch=FOCUS_PATCH):
    """Resample a thin sub-volume that follows the focus surface.

    Each patch contributes slices [z*-half_depth .. z*+half_depth], so the
    result is a flattened stack: the tissue's tilt has been straightened out.
    """
    nz, ny, nx = stack.shape
    n_out = 2 * half_depth + 1
    out = np.zeros((n_out, ny, nx), dtype=stack.dtype)
    for iy in range(fmap.shape[0]):
        for ix in range(fmap.shape[1]):
            z0 = int(np.clip(fmap[iy, ix], half_depth, nz - half_depth - 1))
            ys, xs = slice(iy*patch, (iy+1)*patch), slice(ix*patch, (ix+1)*patch)
            out[:, ys, xs] = stack[z0-half_depth:z0+half_depth+1, ys, xs]
    return out


# ----------------------------------------------------------------------
# 3. Deconvolution + background removal
# ----------------------------------------------------------------------

def richardson_lucy(volume, psf, iters=12, eps=1e-7):
    """Poisson-ML deconvolution.

    Multiplicative updates keep the estimate non-negative for free.

    ITERS IS THE MOST IMPORTANT KNOB IN THIS FILE. Too few: soft, barely
    changed. Too many: noise amplification and hard ringing halos. XOA does
    not publish its count, so fit this by matching edge profiles across
    nuclear boundaries in real data. The ringing you get is not a bug --
    real morphology_focus images have it.
    """
    psf_flip = psf[::-1, ::-1, ::-1]
    est = np.full_like(volume, volume.mean(), dtype=np.float64)
    for _ in range(iters):
        blurred = fftconvolve(est, psf, mode='same')
        ratio = volume / np.maximum(blurred, eps)
        est *= fftconvolve(ratio, psf_flip, mode='same')
        est = np.maximum(est, 0.0)
    return est


def remove_background(volume, sigma=40.0):
    """Subtract a smooth low-frequency estimate (autofluorescence + stray).

    XOA offsets after subtraction to avoid clipping negatives; mirror that
    rather than clamping at zero, or you will bias the background statistics.
    """
    bg = gaussian_filter(volume, sigma=(0, sigma, sigma))
    out = volume - bg
    return out - out.min()          # the documented offset trick


# ----------------------------------------------------------------------
# 4. Stitching
# ----------------------------------------------------------------------

def stitch(fovs, positions, overlap=FOV_OVERLAP, jitter_px=0.3, rng=None):
    """Blend FOVs into a mosaic with linear feathering in overlap regions.

    fovs      : list of 2D arrays
    positions : list of (row, col) top-left placements, in pixels

    TWO SIGNATURES WORTH REPRODUCING:

    1. Overlap zones average two independent noise realizations, so their
       variance drops by ~2x (std by sqrt(2)). A variance map of a real
       mosaic shows a visible grid because of this. It is one of the
       easiest ways to detect synthetic data that skipped stitching.

    2. Feature matching fails in low-texture regions, leaving sub-pixel
       misregistration -- modelled here as `jitter_px`. Where it is large,
       structures ghost/double across the seam.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    h = max(p[0] for p in positions) + fovs[0].shape[0]
    w = max(p[1] for p in positions) + fovs[0].shape[1]
    acc = np.zeros((h, w)); wsum = np.zeros((h, w))

    for img, (r0, c0) in zip(fovs, positions):
        ny, nx = img.shape
        if jitter_px:                                # residual misregistration
            dy, dx = rng.normal(0, jitter_px, 2)
            yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
            img = map_coordinates(img, [yy + dy, xx + dx],
                                  order=1, mode='nearest')
        ramp_y = np.clip(np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
                         / max(overlap, 1), 0, 1)
        ramp_x = np.clip(np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
                         / max(overlap, 1), 0, 1)
        wgt = np.outer(ramp_y, ramp_x) + 1e-6
        acc[r0:r0+ny, c0:c0+nx] += img * wgt
        wsum[r0:r0+ny, c0:c0+nx] += wgt

    return acc / np.maximum(wsum, 1e-6)


def add_vignetting(fov, strength=0.06):
    """Residual per-FOV illumination falloff after imperfect flat-fielding.

    Because it repeats identically in every tile, it appears in the mosaic
    as periodic intensity modulation at the FOV pitch -- a strong, easily
    measured signature, and a common giveaway when it is absent.
    """
    ny, nx = fov.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = ((yy - ny/2)/(ny/2))**2 + ((xx - nx/2)/(nx/2))**2
    return fov * (1.0 - strength * r2)


# ----------------------------------------------------------------------
# 5. Output encoding
# ----------------------------------------------------------------------

def to_uint16(img):
    """Xenium morphology images are 16-bit, values in photoelectrons.

    No rescaling to [0, 65535] -- pe is a physical unit and the pipeline
    preserves it. Just round and clip. If your synthetic values do not land
    in a plausible pe range, the problem is upstream radiometry, not here.
    """
    return np.clip(np.rint(img), 0, 65535).astype(np.uint16)


def build_pyramid(img, levels=5):
    """Each level halves resolution (doubles um/px). Level 0 = 0.2125 um."""
    out = [img]
    for _ in range(levels - 1):
        out.append(zoom(out[-1].astype(np.float64), 0.5, order=1))
    return [to_uint16(a) for a in out]


# ----------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------

def process_fov(stack, psf, rl_iters=12, return_focus_map=False,
                focus='surface', focus_order=1):
    """Single FOV: raw sensor stack -> 2D focus image.

    Set return_focus_map=True to also get the focus map. The label map must
    be sliced at the SAME z this function used, or image and target describe
    different planes -- so the caller needs it.
    """
    stack = correct_distortion(stack)
    # 'surface' fits a smooth plane (RMSE 1.17 slices on a tilted phantom vs
    # 2.12 for raw per-patch argmax, and it stops single nuclei being carved
    # along patch boundaries). 'argmax' keeps the old behaviour.
    fmap = (focus_surface(stack, order=focus_order) if focus == 'surface'
            else focus_map(stack))
    sub = extract_focal_substack(stack, fmap)
    dec = richardson_lucy(sub.astype(np.float64), psf, iters=rl_iters)
    dec = remove_background(dec)
    mid = dec[dec.shape[0] // 2]              # DOC: middle slice, not a MIP
    out = add_vignetting(mid)
    return (out, fmap) if return_focus_map else out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    import psfmodels as psfm

    psf = psfm.vectorial_psf_centered(nz=5, dz=0.75, nx=15, dxy=DETECTOR_DX,
                                      pz=3.0, wvl=0.461,
                                      params=dict(NA=0.70, ni=1.0, ni0=1.0,
                                                  ns=1.40, tg=0, tg0=0))
    psf /= psf.sum()

    # stand-in for forward-model output: small FOVs so this runs fast
    ny = nx = 128
    fovs, positions = [], []
    for i in range(2):
        for j in range(2):
            truth = rng.poisson(30.0, (7, ny, nx)).astype(np.float64)
            truth[3, 40:70, 40:70] += 400          # a bright blob
            fovs.append(process_fov(truth, psf))
            positions.append((i * (ny - 32), j * (nx - 32)))

    mosaic = stitch(fovs, positions, overlap=32, rng=rng)
    pyr = build_pyramid(to_uint16(mosaic), levels=3)
    print('mosaic  ', mosaic.shape, 'pe', round(mosaic.min(), 1),
          '-', round(mosaic.max(), 1))
    print('pyramid ', [p.shape for p in pyr])

    # the seam signature: overlap columns should be quieter
    seam = mosaic[:, 96-16:96+16]
    interior = mosaic[:, 10:42]
    print('var interior/seam', round(interior.var(), 1), '/', round(seam.var(), 1))


# ----------------------------------------------------------------------
# 2b. Focus surface (replacement for per-patch argmax)
# ----------------------------------------------------------------------

def _sharpness_volume(stack, patch=FOCUS_PATCH):
    """Normalized variance per patch per slice.

        F = (1/(N*mu)) * sum (I - mu)^2

    The 1/mu divisor is what makes it scale-invariant: without it a bright
    patch outscores a sharp one, and the metric tracks brightness rather
    than focus. Sun et al. (2004) rank this at or near the top for
    fluorescence; it also beats gradient energy on noisy data because
    squaring a gradient amplifies shot noise.
    """
    nz, ny, nx = stack.shape
    py, px = ny // patch, nx // patch
    blocks = (stack[:, :py*patch, :px*patch]
              .reshape(nz, py, patch, px, patch).astype(np.float64))
    mu = blocks.mean(axis=(2, 4))
    var = blocks.var(axis=(2, 4))
    return var / np.maximum(mu, 1e-6), mu


def focus_surface(stack, patch=FOCUS_PATCH, order=2, robust_iters=3):
    """Fit a smooth focus surface z*(y, x) instead of trusting per-patch argmax.

    WHY. A tissue section is a gently tilted, slightly warped sheet, so the
    true focus surface has only a few degrees of freedom. Per-patch argmax
    has one degree of freedom PER PATCH and estimates each from 256 noisy
    pixels -- so it is wildly overparameterized and its errors carve single
    nuclei into pieces along patch boundaries. Fitting a low-order polynomial
    imposes the prior that the surface is smooth, which is physically true.

    Steps:
      1. normalized-variance sharpness per patch per slice
      2. sub-slice peak by parabolic interpolation (depth of field is
         comparable to the z-step, so the peak is barely sampled and the
         integer argmax is coarse)
      3. per-patch confidence = peak prominence x signal level; empty or
         dim patches carry almost no weight, which is the whole game --
         they are where argmax returns pure noise
      4. weighted least-squares polynomial fit, iteratively reweighted to
         suppress outliers

    order: 1 = tilted plane, 2 = plane + quadratic warp (default).
    Returns a float array of shape (ny//patch, nx//patch).
    """
    S, mu = _sharpness_volume(stack, patch)
    nz, py, px = S.shape

    k = np.argmax(S, axis=0)
    kc = np.clip(k, 1, nz - 2)
    iy, ix = np.mgrid[0:py, 0:px]
    Fm, F0, Fp = S[kc-1, iy, ix], S[kc, iy, ix], S[kc+1, iy, ix]

    denom = Fm - 2*F0 + Fp
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (Fm - Fp) / denom, 0.0)
    z_est = kc + np.clip(delta, -1, 1)             # sub-slice refinement

    prominence = F0 - 0.5 * (Fm + Fp)              # how peaked is the curve
    w = np.maximum(prominence, 0) * np.maximum(mu[kc, iy, ix], 0)
    w = w / (w.max() + 1e-12)

    # design matrix: polynomial in normalized patch coordinates
    yy = (iy / max(py - 1, 1) - 0.5).ravel()
    xx = (ix / max(px - 1, 1) - 0.5).ravel()
    cols = [np.ones_like(yy)]
    for o in range(1, order + 1):
        for j in range(o + 1):
            cols.append(yy**(o - j) * xx**j)
    A = np.stack(cols, axis=1)
    z = z_est.ravel()
    wt = w.ravel()

    for _ in range(robust_iters):
        W = np.sqrt(wt)[:, None]
        coef, *_ = np.linalg.lstsq(A * W, z * W[:, 0], rcond=None)
        resid = np.abs(z - A @ coef)
        scale = 1.4826 * np.median(resid[wt > 0.01]) + 1e-6   # robust sigma
        wt = wt / (1.0 + (resid / (3 * scale))**2)            # Cauchy weights

    return np.clip((A @ coef).reshape(py, px), 0, nz - 1)
