"""
Driver: ground truth nuclei -> forward model -> XOA-like pipeline -> PNG.

Renders a 4-panel comparison so you can see what each stage contributes.
Scaled down from real Xenium (which is 3520x2960 per FOV) so it runs in
under a minute; the geometry is proportionally the same.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psfmodels as psfm

import dapi_forward as fwd
import dapi_pipeline as pipe

# --- scaled-down geometry -------------------------------------------------
SS = 2                      # supersample (4 in the real model; 2 to stay fast)
FOV_DET = 256               # detector px per FOV (3520x2960 in reality)
OVERLAP = 32                # px (128 in reality)
NZ = 12                     # z-slices at 0.75 um -> 9 um section
FINE_DX = fwd.DETECTOR_DX / SS


# --- 1. ground truth ------------------------------------------------------

def nuclei_field(shape_zyx, n_nuclei, rng, dx=FINE_DX, dz=fwd.DZ):
    """Ellipsoidal nuclei with chromatin texture.

    Ellipsoids not spheres, and textured not uniform, because a smooth ball
    hides exactly the high-frequency behaviour you are trying to inspect --
    ringing, focus selection, and deconvolution all act on fine detail.
    """
    nz, ny, nx = shape_zyx
    f = np.zeros(shape_zyx)
    z = np.arange(nz)[:, None, None] * dz
    y = np.arange(ny)[None, :, None] * dx
    x = np.arange(nx)[None, None, :] * dx

    for _ in range(n_nuclei):
        cz = rng.uniform(2, nz - 2) * dz
        cy, cx = rng.uniform(0, ny) * dx, rng.uniform(0, nx) * dx
        rz = rng.uniform(2.0, 3.0)                 # nuclei are flattened in z
        ry, rx = rng.uniform(2.5, 4.0), rng.uniform(2.5, 4.0)
        th = rng.uniform(0, np.pi)
        dy_, dx_ = y - cy, x - cx
        u = dy_ * np.cos(th) + dx_ * np.sin(th)
        v = -dy_ * np.sin(th) + dx_ * np.cos(th)
        d = np.sqrt((z - cz)**2 / rz**2 + u**2 / ry**2 + v**2 / rx**2)
        f += 0.5 * (1 - np.tanh((d - 1.0) / 0.15)) * rng.uniform(0.7, 1.3)

    # chromatin: coarse mottling inside nuclei, not white noise
    from scipy.ndimage import gaussian_filter
    tex = gaussian_filter(rng.normal(0, 1, shape_zyx), sigma=(0.5, 3, 3))
    tex /= tex.std()
    return np.clip(f * (1 + 0.35 * tex), 0, None)


# --- 2. run -----------------------------------------------------------------

def main():
    rng = np.random.default_rng(7)
    fine = (NZ, FOV_DET * SS, FOV_DET * SS)

    print('ground truth ...')
    truth = nuclei_field(fine, n_nuclei=60, rng=rng)

    print('forward model (optics + detector) ...')
    # patch the module constants so forward() uses our scaled grid
    fwd.SUPERSAMPLE, fwd.FINE_DX = SS, FINE_DX
    raw = fwd.forward(truth, depth_um=3.0, photons_per_unit=2500.0,
                      autofluor=0.03, rng=rng, slab_size=6)
    print('   raw stack', raw.shape, 'pe %.0f-%.0f' % (raw.min(), raw.max()))

    print('pipeline (distortion, focus, deconv, slice) ...')
    psf = psfm.vectorial_psf_centered(nz=5, dz=fwd.DZ, nx=15,
                                      dxy=fwd.DETECTOR_DX, pz=3.0, wvl=0.461,
                                      params=dict(NA=fwd.NA, ni=1.0, ni0=1.0,
                                                  ns=fwd.N_SAMPLE, tg=0, tg0=0))
    psf /= psf.sum()
    processed = pipe.process_fov(raw, psf, rl_iters=12)

    print('stitching 2x2 mosaic ...')
    # reuse the same FOV four times with independent detector noise, so the
    # seam structure is visible without simulating four separate volumes
    fovs = []
    for _ in range(4):
        prnu, sread = pipe.__dict__.get('_maps', (None, None))
        noisy = raw + rng.normal(0, np.sqrt(np.maximum(raw, 1)) * 0.15)
        fovs.append(pipe.process_fov(noisy, psf, rl_iters=12))
    step = FOV_DET - OVERLAP
    positions = [(0, 0), (0, step), (step, 0), (step, step)]
    mosaic = pipe.stitch(fovs, positions, overlap=OVERLAP, jitter_px=0.4,
                         rng=rng)

    # --- 3. render ---------------------------------------------------------
    truth_det = truth.max(axis=0).reshape(FOV_DET, SS, FOV_DET, SS).mean((1, 3))
    fmap = pipe.focus_map(raw)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    panels = [
        (truth_det, 'ground truth $f$ (MIP)', 'gray', None),
        (raw[NZ // 2], 'raw sensor, middle slice (pe)', 'gray', None),
        (processed, 'after pipeline: focus + deconv + BG', 'gray', None),
        (mosaic, '2x2 stitched mosaic', 'gray', None),
    ]
    for a, (img, title, cm, _) in zip(ax, panels):
        lo, hi = np.percentile(img, [1, 99.5])
        a.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
        a.set_title(title, fontsize=11)
        a.axis('off')
    # mark the seams so they are findable by eye
    for xy in (step, ):
        ax[3].axvline(xy + OVERLAP / 2, color='tab:orange', lw=0.6, alpha=0.7)
        ax[3].axhline(xy + OVERLAP / 2, color='tab:orange', lw=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig('dapi_demo.png', dpi=110, bbox_inches='tight')
    print('wrote dapi_demo.png')

    # --- 4. diagnostics ----------------------------------------------------
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.2))
    im = ax2[0].imshow(fmap, cmap='viridis', interpolation='nearest')
    ax2[0].set_title('focus map (best-Z per 16x16 patch)', fontsize=10)
    plt.colorbar(im, ax=ax2[0], fraction=0.046)

    row = FOV_DET // 2
    ax2[1].plot(raw[NZ // 2, row], lw=0.7, label='raw')
    ax2[1].plot(processed[row], lw=0.7, label='processed')
    ax2[1].set_title('intensity profile across one row', fontsize=10)
    ax2[1].set_xlabel('px'); ax2[1].set_ylabel('pe'); ax2[1].legend(fontsize=8)

    # seam signature: local variance should dip in the blend zone
    from scipy.ndimage import uniform_filter
    lv = uniform_filter(mosaic**2, 9) - uniform_filter(mosaic, 9)**2
    ax2[2].plot(lv[10:step-10].mean(axis=0), lw=0.7)
    ax2[2].axvspan(step, step + OVERLAP, color='tab:orange', alpha=0.25,
                   label='overlap')
    ax2[2].set_title('local variance vs column (seam dip)', fontsize=10)
    ax2[2].set_yscale('log'); ax2[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('dapi_diagnostics.png', dpi=110, bbox_inches='tight')
    print('wrote dapi_diagnostics.png')


if __name__ == '__main__':
    main()
