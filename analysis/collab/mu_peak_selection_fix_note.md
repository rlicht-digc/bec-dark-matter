# mu_peak Selection Fix — All Phase-Peak Scripts

## Date
2026-03-02

## Files Patched

| File | Lines Changed | Function |
|------|--------------|----------|
| `analysis/pipeline/test_probes_control_suite.py` | fit_edge (L435–440), fit_pd_fb (L454–458) | Already patched in prior session |
| `analysis/pipeline/test_probes_massmatched_controls.py` | fit_edge_model (L569–578), fit_peak_dip_fallback (L621–630) | Patched in this session |
| `analysis/pipeline/run_referee_required_tests.py` | fit_edge_model (L395–406), fit_peak_dip_fallback (L474–485) | Patched in this session |

Summary output provenance blocks added to:
- `test_probes_massmatched_controls.py` — phase_results dict (L981)
- `test_probes_massmatched_controls.py` — triad phase results (L1090)
- `run_referee_required_tests.py` — Test 1 summary (L714)
- `run_referee_required_tests.py` — Test 2 summary (L985)

## Old Behavior
`mu_peak = float(best.x[3])` — unconditionally returns `mup`, the center of the
positive-amplitude Gaussian in the M2b_edge model (params: s0, s1, Ap, **mup**, wp,
Ad, **mud**, wd, E, xe, de). When the sample is small or noisy (e.g., 63 weighted
SPARC galaxies), the optimizer can latch `mup` onto a noise feature far from g†
(observed: mup = −11.86, nearly 2 dex from log10(g†) = −9.921).

## New Behavior
```python
mup = float(best.x[3]); mud = float(best.x[6])
mu_peak = mup if abs(mup - LOG_G_DAGGER) <= abs(mud - LOG_G_DAGGER) else mud
```
Selects whichever Gaussian center (peak or dip) is nearest to log10(g†), ensuring
that `mu_peak` tracks the g†-adjacent feature rather than an arbitrary noise bump.
Both raw values (`mup_raw`, `mud_raw`) are preserved in the return dict and in the
summary JSON for full provenance.

## Why It Matters
The M2b_edge model has two Gaussian components whose physical roles (peak vs. dip)
are not enforced by the optimizer — only their amplitude signs are constrained
(Ap > 0, Ad < 0). In well-sampled data (>100 galaxies), both components typically
land near g† and the distinction is moot. In small samples, the positive Gaussian
can wander to a noise bump, producing a `mu_peak` that is not comparable to Test 1/2
values. The "nearest to g†" rule ensures cross-test comparability.

## Provenance Metadata
All summary JSONs now include a `_provenance` block:
```json
{
  "x_col": "log_gbar",
  "log_base": "log10",
  "units": "m/s^2",
  "peak_definition": "M2b_edge Gaussian center nearest log10(g_dagger)",
  "log_g_dagger": -9.921
}
```

## Confirmation: C and p Values Unchanged

### PROBES massmatched controls (seed=42, n_perm=1000)
| Dataset | C (before) | C (after) | p (before) | p (after) |
|---------|-----------|-----------|-----------|-----------|
| PROBES | 0.2681 | 0.2681 | 1.000 | 1.000 |
| SPARC_matched | 0.5183 | 0.5183 | 0.958 | 0.958 |
| TNG_matched | 0.6111 | 0.6111 | 0.013 | 0.013 |

### mu_peak Values (PROBES massmatched controls)
| Dataset | mu_peak (before) | mu_peak (after) | mup_raw | mud_raw | Changed? |
|---------|-----------------|----------------|---------|---------|----------|
| PROBES | −9.998 | −9.998 | −9.998 | −11.581 | No |
| SPARC_matched | −9.987 | −9.987 | −9.987 | −12.000 | No |
| TNG_matched | −10.255 | −10.255 | −10.255 | −10.807 | No |

In this test the fix had no numerical effect — `mup` was already the g†-nearest
component for all three datasets. The fix is a safety net that prevents silent
regression in future runs with smaller or noisier samples.
