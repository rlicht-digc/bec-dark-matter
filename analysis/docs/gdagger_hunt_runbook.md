# g† Hunt Runbook

## What is this?

The **g† Hunt** module systematically searches for the BEC acceleration scale `g† = 1.2e-10 m/s²` (log g† = -9.92) in contexts beyond the radial acceleration relation (RAR). The central question is whether g† is:

- **(A)** a RAR-specific scale that only appears in rotation-curve kinematics, or
- **(B)** a universal scale that reappears in independent physical contexts.

A "hit" requires **three things simultaneously**:

1. **A naturally motivated dimensionless variable** (not ad hoc parameter hunting)
2. **A specific kernel/functional form** (preferably BE-like) that explains the data
3. **Strong null controls** showing the preference is sharply centered on g† and not generic dimensional fitting

---

## Quick Start

### Run locally (smoke test, ~2 min)

```bash
cd /path/to/bec-dark-matter
python3 analysis/tests/test_gdagger_hunt_pilot.py --smoke
```

### Run locally (standard, ~10 min)

```bash
python3 analysis/tests/test_gdagger_hunt_pilot.py --n-shuffles 200
```

### Run referee-grade (parallel, ~30 min)

```bash
python3 analysis/tests/test_gdagger_hunt_pilot.py \
    --n-shuffles 1000 \
    --parallel --n-workers 8
```

### Run via CLI module interface

```bash
python3 -m analysis.gdagger_hunt --dataset sparc_rar --n-shuffles 500
python3 -m analysis.gdagger_hunt --dataset tian2020_cluster --tag cluster_v1
python3 -m analysis.gdagger_hunt --dataset synthetic --tag sanity
python3 -m analysis.gdagger_hunt --config configs/gdagger_hunt.yaml
```

---

## Output Structure

Every run produces a timestamped directory:

```
outputs/gdagger_hunt/
  YYYYMMDD_HHMMSS_<tag>/
    params.json         # Full run configuration
    summary.json        # All metrics, verdicts, scan results
    metrics.parquet     # Kernel comparison table (or .csv fallback)
    logs.txt            # Detailed run log
    figures/
      data_overview.png          # Data + best-fit curve
      scale_scan_aic.png         # AIC vs acceleration scale
      scale_scan_rms.png         # RMS vs acceleration scale
      kernel_comparison.png      # All kernels ranked by AIC
      nearby_scale_aic.png       # Named scales (g†, aΛ, cH₀, etc.)
      shuffle_null_scales.png    # Null distribution of best scale
      shuffle_null_rms.png       # Null distribution of RMS
  pilot_summary.json    # Combined pilot results
```

---

## What Qualifies as a "Hit"

### Strong hit (publication-worthy)

All of the following must hold:

1. **Scale**: Best-fit scale within **±0.05 dex** of log g† = -9.92
2. **Kernel**: BE-family kernel (BE_RAR, BE_occupation, or BE_cousin) preferred over all control kernels (logistic, power-law) by **ΔAIC > 10**
3. **Sharpness**: Scale scan shows a **sharp optimum** (peak sharpness > 50 in second derivative of AIC)
4. **Shuffle null**: p(null best scale within ±0.1 dex of g†) **< 0.01** with ≥1000 shuffles
5. **Nearby-scale**: g† preferred over cH₀, cH₀/6, and aΛ by **ΔAIC > 5**

### Marginal hit (worth investigating)

- Scale within ±0.1 dex
- BE kernel preferred by ΔAIC > 2
- Shuffle p < 0.05

### Miss

- Scale > 0.3 dex from g†, OR
- Non-BE kernel preferred, OR
- Shuffle p > 0.10

---

## How to Interpret Outputs

### summary.json

The key fields to check:

```json
{
  "best_kernel_name": "BE_RAR",         // Which kernel won?
  "best_log_scale": -9.93,              // How close to -9.92?
  "best_within_0p1_dex": true,          // Quick pass/fail

  "scale_scan": {
    "best_log_scale": -9.93,            // Scale scan optimum
    "delta_aic_at_gdagger": 0.3,        // AIC penalty at g† vs best
    "peak_sharpness": 127.5,            // 2nd deriv. (>50 = sharp)
    "within_0p1_dex": true
  },

  "shuffle_null": {
    "p_value_rms": 0.000,               // Is fit significant?
    "p_value_near_gdagger": 0.015,      // Could chance hit g†?
    "observed_best_log_scale": -9.93,
    "null_rms_mean": 0.482              // How much worse is null?
  },

  "nearby_scales": {
    "best_name": "g†",                  // Which named scale wins?
    "aic_at_scale": {"g†": 123.4, ...}  // Compare AIC values
  },

  "verdicts": {
    "scale_recovery": "HIT",            // STRONG_HIT/HIT/MARGINAL/MISS
    "kernel_family": "BE_FAMILY",       // BE_FAMILY or NON_BE
    "shuffle_control": "SIGNIFICANT",   // SIGNIFICANT/MARGINAL/NOT_SIGNIFICANT
    "peak_sharpness": "SHARP"           // SHARP/MODERATE/BROAD
  }
}
```

### Key plots

- **scale_scan_aic.png**: Look for a sharp V-shaped minimum near the red dashed line (g†). A broad flat minimum means the scale is not well-constrained.
- **shuffle_null_scales.png**: The blue observed line should be far from the gray null distribution. The null should be spread across many scales.
- **kernel_comparison.png**: BE kernels should cluster at low AIC. Large gaps between BE and non-BE kernels indicate strong preference.

---

## Increasing to Referee-Grade

For a publication-quality analysis, increase these parameters:

| Parameter | Smoke | Standard | Referee |
|-----------|-------|----------|---------|
| n_shuffles | 20 | 200 | 2000 |
| n_grid | 80 | 200 | 500 |
| n_scan | 100 | 300 | 1000 |
| n_cv_folds | 5 | 5 | 10 |

### Via YAML config

Create `configs/gdagger_hunt.yaml`:

```yaml
gdagger_hunt:
  tag: referee_v1
  seed: 42
  n_shuffles: 2000
  n_grid: 500
  n_scan: 1000
  n_cv_folds: 10
  parallel: true
  n_workers: 8
  scale_range: [1.0e-13, 1.0e-8]
  output_base: outputs/gdagger_hunt
```

Then:

```bash
python3 -m analysis.gdagger_hunt --config configs/gdagger_hunt.yaml --dataset sparc_rar
```

---

## Running on TNG Lab

TNG Lab provides compute proximity to TNG simulation data. The module is designed to run locally but supports TNG Lab via:

1. **Upload the analysis/ directory** to TNG Lab Jupyter
2. **No internet needed at runtime** (all kernels are local math)
3. **If accessing TNG API** (for future TNG-based experiments):
   - Set `TNG_API_KEY` as an environment variable
   - Or create `.env.tng` in project root with `TNG_API_KEY=your_key`
   - The module reads from env only; never hardcodes keys

### TNG Lab example

```python
# In a TNG Lab Jupyter notebook
import sys
sys.path.insert(0, '/path/to/bec-dark-matter/analysis')

from gdagger_hunt import ExperimentConfig, run_experiment

# Use your TNG-extracted data
config = ExperimentConfig(
    tag="tng_rar_v1",
    n_shuffles=2000,
    parallel=True,
    n_workers=16,
    output_base="/path/to/outputs/gdagger_hunt",
)

# x = your acceleration data, y = your response
summary = run_experiment(x, y, config, dataset_name="TNG_RAR")
```

---

## Adding New Datasets ("Drop-in Adapters")

To test g† in a new context, you need:

1. **Physical data** as two arrays: `x` (the independent variable, typically an acceleration or acceleration proxy in m/s²) and `y` (the response, typically a ratio like g_obs/g_bar or a dimensionless mapping).

2. **A loader function** that returns `(x, y)` plus metadata.

### Template

```python
def load_my_dataset(project_root=None):
    """Load my new dataset for g† hunt.

    Returns
    -------
    x : array
        Acceleration or acceleration proxy in m/s².
    y : array
        Dimensionless response (mapping ratio, residual, etc.).
    meta : dict
        Dataset metadata.
    """
    # Load your data
    # ...

    # Transform to (x, y) where:
    #   x has units of m/s² (acceleration)
    #   y is dimensionless or has a natural BE-like shape
    x = ...  # physical acceleration
    y = ...  # mapping ratio or response

    meta = {
        "source": "My2026 paper",
        "n_points": len(x),
        "transform": "y = ...",
    }
    return x, y, meta
```

### Promising contexts to try next

1. **Brouwer+2021 lensing RAR** (data in `data/brouwer2021/`): Weak lensing ESD → g_obs mapping. Different physics (no rotation curves).

2. **Velocity dispersion profiles**: σ(r) → dynamical mass → acceleration. Elliptical galaxies.

3. **Galaxy cluster temperature profiles**: T(r) → hydrostatic mass → cluster-scale acceleration.

4. **Satellite galaxy kinematics**: v_sat(r) around host → acceleration.

5. **Tully-Fisher residuals**: δV_flat vs galaxy properties — check if scatter structure uses g†.

---

## Module API Reference

### Core functions

| Function | Description |
|----------|-------------|
| `generate_pi_groups()` | Find dimensionless groups matching target dimensions |
| `match_kernels()` | Fit all kernels to data, return AIC-ranked list |
| `fit_kernel()` | Fit a single kernel with scale optimization |
| `scale_injection_scan()` | Scan fit quality across acceleration scales |
| `shuffle_null_test()` | Permutation null hypothesis test |
| `nearby_scale_comparison()` | Compare g† against cH₀, aΛ, etc. |
| `run_experiment()` | Full pipeline: matching + all controls + plots |

### Data loaders

| Function | Description |
|----------|-------------|
| `load_sparc_rar()` | SPARC rotation curve RAR data |
| `load_tian2020_cluster_rar()` | Tian+2020 CLASH cluster RAR |
| `generate_synthetic()` | Synthetic data from known kernel + scale |

### Kernels

| Name | Formula | Physics |
|------|---------|---------|
| BE_RAR | 1/(1-exp(-√(x/s))) | RAR mapping (g_obs/g_bar) |
| BE_occupation | 1/(exp(√(x/s))-1) | Bose-Einstein occupation number |
| BE_cousin | ε/(1-exp(-ε)) | BE partition function cousin |
| coth | coth(√(x/s)) | Hyperbolic cotangent |
| tanh | tanh(√(x/s)) | Hyperbolic tangent |
| logistic | 1/(1+exp(-(√(x/s)-1))) | Control: no special scale |
| power_law | √(x/s)+1 | Control: no special shape |

---

## Troubleshooting

**"SPARC table2 not found"**: Ensure data symlink exists: `ls -la data/sparc/SPARC_table2_rotmods.dat`

**Slow shuffles**: Use `--parallel --n-workers N` where N ≤ your CPU count.

**joblib not available**: Install with `pip install joblib` or shuffles will run serially (slower but correct).

**PyYAML not available**: Only needed for `--config` YAML files. Install with `pip install pyyaml`.

**Parquet write fails**: Falls back to CSV automatically. Install pyarrow for Parquet: `pip install pyarrow`.
