# RKHS segmentation — Python port (global-only)

Bare-bones Python translation of the global-segmentation path from the MATLAB
project in `../project_RKHS/`. Only the `cfg.mode.segmentation_variant = 'global'`
code path is ported — the localized / selective-geodesic / experimental-local-stats
branches are deliberately left out for clarity.

## Files

- `rkhs_seg.py` — single-module implementation: config dict, basis construction,
  patch layout, ADMM-style solver, and a `run_global_segmentation(image, mask_init, **overrides)`
  entry point.
- `example.py` — loads `patch_dapi_test.npy` from the MATLAB project's `artifacts/`
  folder if present (falls back to a synthetic disk), runs the segmentation,
  and shows a before / mask-init / after plot.
- `requirements.txt` — minimal deps.
