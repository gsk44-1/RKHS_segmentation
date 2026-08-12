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

def nuclei_field(shape_zyx, n_nuclei, rng, dx=FINE_DX, dz=fwd.DZ,
                 texture=0.35, return_labels=True, section_um=5.0,
                 section_center=None):
    """Ellipsoidal nuclei with chromatin texture, plus an instance label map.

    Returns (f, labels) where labels is int32, 0 = background and k = the
    k-th nucleus.

    OVERLAP RESOLUTION. Each nucleus defines a normalized radial distance

        d_k = sqrt( (z-cz)^2/rz^2 + u^2/ry^2 + v^2/rx^2 )

    which equals 1 exactly on its surface. A voxel is labelled by whichever
    nucleus has the smallest d, and left as background if every d > 1.

    Using d rather than Euclidean distance to the centre means the partition
    respects each nucleus's size and orientation -- a large nucleus wins more
    of a contested region, which is what you want. Formally this is a
    Laguerre/power-style tessellation, not a plain Voronoi diagram; plain
    Voronoi on centres would cut a big nucleus in half against a small
    neighbour sitting close to it.

    Rendering is done per-nucleus into a bounding box (~50x faster than
    broadcasting each ellipsoid over the whole volume).
    """
    from scipy.ndimage import gaussian_filter, zoom

    nz, ny, nx = shape_zyx

    # SECTION SLAB. Nuclei are confined to a section of thickness section_um
    # (5 um FFPE by default) sitting inside a taller imaged volume. Letting
    # them fill all of z -- the previous behaviour -- makes the focus problem
    # ILL-POSED: every slice contains tissue, the sharpness curve is flat, and
    # focus_surface fits noise. It also means no single slice can show most
    # nuclei in focus, since depth of field (~0.9 um at NA 0.7) is far less
    # than the volume depth.
    half = 0.5 * section_um / dz
    zc = nz / 2.0 if section_center is None else section_center
    z_lo, z_hi = zc - half, zc + half     # centres may sit at the surfaces,
                                          # so nuclei get microtome-truncated
    f = np.zeros(shape_zyx, dtype=np.float32)
    best_d = np.full(shape_zyx, np.inf, dtype=np.float32)
    labels = np.zeros(shape_zyx, dtype=np.int32)

    for k in range(1, n_nuclei + 1):
        # MICROTOME TRUNCATION: centres are allowed outside the section, so
        # nuclei get cut by the top and bottom surfaces exactly as a real
        # 5 um FFPE section cuts 5-8 um nuclei. Keeping every nucleus whole
        # (the old cz in [1.5, nz-1.5]) removes a genuinely hard case:
        # telling a truncated nucleus from a defocused one.
        cz = rng.uniform(z_lo, z_hi)
        cy, cx = rng.uniform(0, ny), rng.uniform(0, nx)
        rz = rng.uniform(2.0, 3.0)                # nuclei are flattened in z
        ry, rx = rng.uniform(2.5, 4.0), rng.uniform(2.5, 4.0)
        th = rng.uniform(0, np.pi)
        amp = rng.uniform(0.7, 1.3)

        # bounding box: 1.6x the semi-axes covers the soft edge
        pz = int(1.6 * rz / dz) + 2
        py, px = int(1.6 * max(ry, rx) / dx) + 2, int(1.6 * max(ry, rx) / dx) + 2
        z0, z1 = max(0, int(cz) - pz), min(nz, int(cz) + pz + 1)
        y0, y1 = max(0, int(cy) - py), min(ny, int(cy) + py + 1)
        x0, x1 = max(0, int(cx) - px), min(nx, int(cx) + px + 1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue

        z = (np.arange(z0, z1)[:, None, None] - cz) * dz
        y = (np.arange(y0, y1)[None, :, None] - cy) * dx
        x = (np.arange(x0, x1)[None, None, :] - cx) * dx
        u = y * np.cos(th) + x * np.sin(th)
        v = -y * np.sin(th) + x * np.cos(th)
        d = np.sqrt(z**2 / rz**2 + u**2 / ry**2 + v**2 / rx**2).astype(np.float32)

        box = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        profile = (0.5 * (1 - np.tanh((d - 1.0) / 0.15)) * amp).astype(np.float32)

        # EXCLUSIVE OCCUPANCY. Previously this was `f += profile`, which let
        # nuclei interpenetrate and made contested voxels 2-3x too bright --
        # physically impossible, since nuclei have membranes. The winner of
        # the tessellation now owns both the label and the intensity.
        win = d < best_d[box]
        best_d[box] = np.where(win, d, best_d[box])
        labels[box] = np.where(win, k, labels[box])
        f[box] = np.where(win, profile, f[box])

    labels[best_d > 1.0] = 0                      # outside every ellipsoid

    if texture:
        # generate coarse noise at 1/4 scale and upsample: same look, ~8x cheaper
        small = rng.normal(0, 1, (nz, max(2, ny // 4), max(2, nx // 4)))
        small = gaussian_filter(small, sigma=(0.5, 0.75, 0.75))
        tex = zoom(small, (1, ny / small.shape[1], nx / small.shape[2]), order=1)
        tex = tex[:, :ny, :nx] / (tex.std() + 1e-9)
        f = np.clip(f * (1 + texture * tex), 0, None).astype(np.float32)

    return (f, labels) if return_labels else f


def focal_slice_labels(labels3d, fmap, patch=pipe.FOCUS_PATCH,
                       ignore_halfdepth=0):
    """2D target taken at the SAME slice the image was taken at.

    labels3d : (nz, ny, nx) instance map, already on the DETECTOR grid
    fmap     : the focus map passed to pipe.extract_focal_substack()

    WHY THIS AND NOT A PROJECTION. morphology_focus is a single deconvolved
    slice. Slicing the label volume at exactly the z the image was sliced at
    makes image and target two views of one volume -- no majority vote, no
    z-buffer, no focus weighting. Consistency by construction.

    PER-PIXEL, NOT PER-PATCH. Sampling one integer slice per 16x16 patch
    quantizes the smooth focus surface onto the patch grid and chops nuclei
    into rectangles along patch boundaries. This upsamples the surface to
    pixel resolution and rounds (int() would truncate, biasing half a slice).

    THE IGNORE CLASS. A nucleus missing the focal slice is not invisible: it
    contributes out-of-focus haze, and 3D deconvolution drags some of it back.
    Calling those pixels background punishes the model for detecting real
    signal, so they are marked -1 and should be masked out of the loss.
    Set ignore_halfdepth=0 (the default) to disable it: badly defocused
    nuclei then simply count as background, which is fine if you would
    rather the model skip very fuzzy nuclei than lose supervision on the
    boundary pixels the ignore band was swallowing. Nonzero values should
    track depth of field, ~lambda*n/NA^2 (about 0.9 um at NA 0.7, i.e.
    roughly one 0.75 um slice either side).

    Returns int32: 0 background, k instance, -1 ignore.
    """
    nz, ny, nx = labels3d.shape
    surf = pipe.upsample_surface(fmap, (ny, nx), patch)
    z0 = np.clip(np.rint(surf), 0, nz - 1).astype(np.intp)

    out = np.take_along_axis(labels3d, z0[None], axis=0)[0].astype(np.int32)

    if ignore_halfdepth > 0:
        nearby = np.zeros((ny, nx), dtype=bool)
        for dz in range(-ignore_halfdepth, ignore_halfdepth + 1):
            zi = np.clip(z0 + dz, 0, nz - 1)
            nearby |= np.take_along_axis(labels3d, zi[None], axis=0)[0] > 0
        out[(out == 0) & nearby] = -1
    return out


def downsample_labels(labels, factor):
    """Fine grid -> detector grid. NEAREST NEIGHBOUR ONLY.

    Never average or interpolate a label map: mean of labels 3 and 7 is 5,
    a nucleus that does not exist. This subsamples, which is standard and
    slightly biases boundaries; a proper block-mode is more accurate if you
    care about exact edge pixels.
    """
    return labels[..., ::factor, ::factor]   # works for 2D or 3D


# --- 2. run -----------------------------------------------------------------

def main():
    rng = np.random.default_rng(7)
    fine = (NZ, FOV_DET * SS, FOV_DET * SS)

    print('ground truth ...')
    truth, lab3d = nuclei_field(fine, n_nuclei=60, rng=rng)

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
    processed, fmap = pipe.process_fov(raw, psf, rl_iters=12,
                                       return_focus_map=True,
                                       focus='surface', focus_order=1)

    # label map: fine grid -> detector grid, then sliced at the SAME focal
    # plane the image was sliced at
    lab3d_det = downsample_labels(lab3d, SS)
    lab2d = focal_slice_labels(lab3d_det, fmap)

    print('stitching 2x2 mosaic ...')
    # reuse the same FOV four times with independent detector noise, so the
    # seam structure is visible without simulating four separate volumes
    fovs = []
    for _ in range(4):
        prnu, sread = pipe.__dict__.get('_maps', (None, None))
        noisy = raw + rng.normal(0, np.sqrt(np.maximum(raw, 1)) * 0.15)
        fovs.append(pipe.process_fov(noisy, psf, rl_iters=12,
                                     focus='surface', focus_order=1))
    step = FOV_DET - OVERLAP
    positions = [(0, 0), (0, step), (step, 0), (step, step)]
    mosaic = pipe.stitch(fovs, positions, overlap=OVERLAP, jitter_px=0.4,
                         rng=rng)

    # --- 3. render ---------------------------------------------------------
    truth_det = truth.max(axis=0).reshape(FOV_DET, SS, FOV_DET, SS).mean((1, 3))
    fmap = pipe.focus_map(raw)

    fig, ax = plt.subplots(1, 5, figsize=(25, 5.4))
    panels = [
        (truth_det, 'ground truth $f$ (MIP)', 'gray', None),
        (lab2d, 'instance labels at focal slice', 'label', None),
        (raw[NZ // 2], 'raw sensor, middle slice (pe)', 'gray', None),
        (processed, 'after pipeline: focus + deconv + BG', 'gray', None),
        (mosaic, '2x2 stitched mosaic', 'gray', None),
    ]
    for a, (img, title, cm, _) in zip(ax, panels):
        if cm == 'label':
            shown = np.where(img > 0, (img * 37 % 200) + 40, 0).astype(float)
            a.imshow(shown, cmap='nipy_spectral', vmin=0, vmax=255,
                     interpolation='nearest')
            a.imshow(np.where(img < 0, 1.0, np.nan), cmap='gray_r',
                     vmin=0, vmax=1.6, interpolation='nearest')
        else:
            lo, hi = np.percentile(img, [1, 99.5])
            a.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
        a.set_title(title, fontsize=11)
        a.axis('off')
    # mark the seams so they are findable by eye
    for xy in (step, ):
        ax[4].axvline(xy + OVERLAP / 2, color='tab:orange', lw=0.6, alpha=0.7)
        ax[4].axhline(xy + OVERLAP / 2, color='tab:orange', lw=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig('dapi_demo.png', dpi=110, bbox_inches='tight')
    print('wrote dapi_demo.png')
    ids = np.unique(lab2d); ids = ids[ids > 0]
    print('   %d instances visible, %.1f%% foreground, %.1f%% ignore'
          % (len(ids), 100 * (lab2d > 0).mean(), 100 * (lab2d < 0).mean()))
    print('   f.max() = %.2f  (was 3.3 when nuclei could interpenetrate)'
          % truth.max())

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
