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

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

## Mapping to the MATLAB project

| Python                          | MATLAB                                      |
| ------------------------------- | ------------------------------------------- |
| `default_config()`              | `default_global_combined_cfg.m`             |
| `_build_gaussian_kernel_matrix` | `build_gaussian_kernel_matrix.m`            |
| `_build_heaviside_basis`        | `build_heaviside_basis_matrix.m`            |
| `_build_patch_layout`           | `build_patch_layout.m`                      |
| `_build_solver_cache`           | `build_solver_cache.m`                      |
| `_initialize_state`             | `initialize_solver_state.m`                 |
| `_update_patch_reconstruction`  | `update_patch_reconstruction.m`             |
| `_update_region_means`          | `update_region_means.m` (global branch)     |
| `_update_membership_split`      | `update_membership_split.m` (global branch) |
| `run_global_segmentation`       | `run_global_combined_case.m` (global path)  |

## Parameters

The config dict mirrors the MATLAB `cfg.mdl / cfg.bss / cfg.ptc / cfg.opt / cfg.term`
structure. You can override individual scalars by keyword:

```python
result = run_global_segmentation(
    image, mask_init,
    mdl_lambda_regionfit=1e-5,
    opt_maxiter_loopcap=50,
)
```

or pass a fully-formed `cfg=...` dict (start from `default_config()` and edit).

## What's intentionally *not* ported

- Interactive UI (App Designer / `uifigure`).
- Localized / selective-geodesic / two-stage / experimental-local-stats variants.
- Synthetic noise + brightness/contrast preview controls — preprocess your image
  yourself if you need them.
- Energy traces and auxiliary outputs beyond the final `u` and reconstruction —
  the per-iteration history is returned in `result["history"]` if you want to
  inspect it.

## Running on Colab

1. Push this folder to GitHub (or copy it onto Drive).
2. In a Colab notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import sys
   sys.path.append('/content/drive/MyDrive/path/to/project_RKHS_python')
   from rkhs_seg import run_global_segmentation, default_config
   ```
3. Load your image (e.g. `np.load(...)` or `imageio.imread(...)`), call
   `run_global_segmentation`, plot the result.
