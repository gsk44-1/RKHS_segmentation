"""
Forward model: fluorophore concentration f -> observed image g (photoelectrons).

Pipeline:
    f (fine grid, object space)
      + autofluorescence        <- blurred, so added HERE
      -> depth-dependent PSF blur
      -> radiometric scaling (photons)
      -> pixel-area integration (block average to detector grid)
      + scatter/stray floor     <- NOT blurred, added HERE
      -> QE + PRNU              (photons -> expected photoelectrons)
      -> Poisson (shot noise)
      + read noise (per-pixel sigma)
      = g, in photoelectrons

All lengths in microns.
"""

import numpy as np
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------
# Geometry / instrument constants
# ----------------------------------------------------------------------

DETECTOR_DX = 0.2125      # um per pixel, Xenium full-res (documented)
SUPERSAMPLE = 4           # fine grid factor; fine dx = 0.0531 um
FINE_DX = DETECTOR_DX / SUPERSAMPLE
DZ = 0.75                 # um, acquisition z-step (documented)

# Optics -- RECONSTRUCTED, not published by 10x. Tune against real data.
WAVELENGTH = 0.461        # um, DAPI emission peak
NA = 0.70                 # numerical aperture (inferred)
N_IMM = 1.0               # immersion refractive index (air)
N_SAMPLE = 1.40           # tissue refractive index (mismatch drives GL asymmetry)

# Camera -- also tune. Xenium output is already in pe, so no gain/offset.
QE = 0.80                 # quantum efficiency at ~460 nm
READ_NOISE_MEAN = 1.4     # electrons RMS, typical sCMOS
READ_NOISE_SPREAD = 0.4   # per-pixel variation in that RMS
PRNU_SIGMA = 0.01         # fractional per-pixel gain variation


# ----------------------------------------------------------------------
# 1. Ground truth object
# ----------------------------------------------------------------------

def make_ball(shape_zyx, center_zyx, radius_um, dx=FINE_DX, dz=DZ, edge_um=0.3):
    """Soft-edged sphere of unit concentration on the fine grid.

    edge_um smooths the boundary; a hard edge would alias badly at this
    sampling and produce ringing that is a discretization artifact rather
    than an optical one.
    """
    nz, ny, nx = shape_zyx
    z = (np.arange(nz) - center_zyx[0]) * dz
    y = (np.arange(ny) - center_zyx[1]) * dx
    x = (np.arange(nx) - center_zyx[2]) * dx
    r = np.sqrt(z[:, None, None]**2 + y[None, :, None]**2 + x[None, None, :]**2)
    return 0.5 * (1 - np.tanh((r - radius_um) / edge_um))


# ----------------------------------------------------------------------
# 2. PSF
# ----------------------------------------------------------------------

def optical_psf(nz, nxy, depth_um, dx=FINE_DX, dz=DZ,
                scatter_frac=None, scatter_hwhm=3.0):
    """Effective PSF = Gibson-Lanni core + incoherent scatter tail.

        h_eff = (1 - a) * h_GL + a * h_scatter

    h_GL comes from psfmodels (vectorial model, more accurate than scalar).
    Use the *_centered variant: with refractive index mismatch the brightest
    plane shifts off geometric focus, so a symmetric zv gives an off-center
    kernel.

    h_scatter is a wide Lorentzian standing in for sub-micron refractive
    index fluctuation. Heavy-tailed on purpose -- it is what produces the
    diffuse veiling glare around dense nuclei and the overall contrast loss.
    Scatter fraction grows with path length through tissue.
    """
    import psfmodels as psfm

    core = psfm.vectorial_psf_centered(
        nz=nz, dz=dz, nx=nxy, dxy=dx, pz=depth_um, wvl=WAVELENGTH,
        params=dict(NA=NA, ni=N_IMM, ni0=N_IMM, ns=N_SAMPLE, tg=0, tg0=0),
    )
    core /= core.sum()

    if scatter_frac is None:                     # ~1% per um of tissue traversed
        scatter_frac = min(0.15, 0.01 * depth_um)

    c = nxy // 2
    yy, xx = np.mgrid[0:nxy, 0:nxy]
    r = np.sqrt((yy - c)**2 + (xx - c)**2) * dx
    lorentz = 1.0 / (1.0 + (r / scatter_hwhm)**2)
    tail = np.repeat(lorentz[None], nz, axis=0)  # near-flat in z: scattered
    tail /= tail.sum()                           # light is not confined axially

    return (1 - scatter_frac) * core + scatter_frac * tail


# ----------------------------------------------------------------------
# 3. Depth-dependent blur
# ----------------------------------------------------------------------

def blur_depth_dependent(f, psf_bank, slab_size=8):
    """Blur f in slabs, each with its own PSF.

    A single fftconvolve assumes shift invariance in z, which Gibson-Lanni
    explicitly violates. Slabbing is the cheap compromise: constant PSF
    within a slab, varying between slabs. slab_size=1 is exact and slow.
    """
    out = np.zeros_like(f)
    nz = f.shape[0]
    for start in range(0, nz, slab_size):
        stop = min(start + slab_size, nz)
        slab = np.zeros_like(f)
        slab[start:stop] = f[start:stop]
        psf = psf_bank[start // slab_size]
        out += fftconvolve(slab, psf, mode='same')
    return out


# ----------------------------------------------------------------------
# 4. Detector
# ----------------------------------------------------------------------

def integrate_pixels(g_fine, factor=None):
    """Sum the fine grid over each detector pixel footprint.

    Sum, not mean: a pixel collects all photons landing in its area.
    """
    if factor is None:
        factor = SUPERSAMPLE   # read at call time, not def time
    nz, ny, nx = g_fine.shape
    ny_c, nx_c = ny // factor, nx // factor
    return (g_fine[:, :ny_c*factor, :nx_c*factor]
            .reshape(nz, ny_c, factor, nx_c, factor)
            .sum(axis=(2, 4)))


def make_sensor_maps(shape_yx, rng):
    """Fixed-pattern maps. Generated ONCE and reused for every FOV --
    that is what 'fixed pattern' means, and it is why the same speckle
    tiles across a stitched Xenium mosaic."""
    prnu = rng.normal(1.0, PRNU_SIGMA, shape_yx)
    sigma_read = np.abs(rng.normal(READ_NOISE_MEAN, READ_NOISE_SPREAD, shape_yx))
    return prnu, sigma_read


def detect(photons, prnu, sigma_read, rng, scatter_floor=2.0):
    """photons (per pixel) -> photoelectrons, with noise."""
    photons = photons + scatter_floor          # unblurred stray/scatter
    rate = photons * QE * prnu                 # expected pe; PRNU scales the
                                               # RATE, so it enters before Poisson
    electrons = rng.poisson(rate).astype(np.float64)
    electrons += rng.normal(0.0, sigma_read)   # read noise, signal-independent
    return electrons                           # already in pe -- no gain division


# ----------------------------------------------------------------------
# 5. Assemble
# ----------------------------------------------------------------------

def forward(f_fine, depth_um=5.0, photons_per_unit=3000.0,
            autofluor=0.02, rng=None, slab_size=8):
    if rng is None:
        rng = np.random.default_rng(0)

    # autofluorescence lives in the object -> gets blurred with the signal
    f_fine = f_fine + autofluor

    n_slabs = int(np.ceil(f_fine.shape[0] / slab_size))
    psf_bank = [
        optical_psf(nz=11, nxy=63,
                    depth_um=depth_um + (i * slab_size) * DZ)
        for i in range(n_slabs)
    ]

    g_fine = blur_depth_dependent(f_fine, psf_bank, slab_size)
    g_fine *= photons_per_unit / SUPERSAMPLE**2   # radiometric scale
    photons = integrate_pixels(g_fine)

    prnu, sigma_read = make_sensor_maps(photons.shape[1:], rng)
    return np.stack([detect(p, prnu, sigma_read, rng) for p in photons])


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    # nucleus ~7 um diameter, on the fine grid
    f = make_ball((16, 160, 160), (8, 80, 80), radius_um=3.5)
    g = forward(f, rng=rng)
    print('fine grid  ', f.shape)
    print('detector   ', g.shape, '-> um/px', DETECTOR_DX)
    print('pe range   ', g.min().round(1), g.max().round(1))
    print('bg mean/var', g[0].mean().round(2), g[0].var().round(2),
          '(Poisson-dominated: these should be comparable)')
