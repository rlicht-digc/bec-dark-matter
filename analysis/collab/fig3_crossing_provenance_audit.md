# Figure 3 Crossing Provenance Audit

**Date:** 2026-03-01
**Auditor:** Claude Code CLI
**Scope:** Trace the provenance of two crossing values in Figure 3 (scatter inversion)
and verify downstream consistency.

---

## 1. What is "-9.80" (actually -9.795)?

| Attribute | Value |
|-----------|-------|
| **Definition** | Zero-crossing of d(sigma_MAD)/d(log g_bar) nearest to g-dagger |
| **Dataset/Tier** | SPARC core sample (2,800 points after Q<=2, 30<=Inc<=85) |
| **Scatter estimator** | `1.4826 * MAD` (robust, median-based) |
| **Binning** | Fixed edges `np.arange(-12.5, -8.5, 0.30)`, min_pts=10 |
| **Source code** | `paper/make_figures.py` lines 366-382 (`_scatter_profile`) and lines 440-457 (crossing annotation) |
| **Computed at** | Runtime -- dynamically from the plotted derivative curve |

The variable `best` (line 450) is the crossing nearest to `LOG_GDAG` among all
sign-change crossings of `dsigma`. The `dsigma` array comes from `_scatter_profile()`
which computes `sigma = 1.4826 * MAD` per bin (line 376).

## 2. What is "-9.97 +/- 0.006"?

| Attribute | Value |
|-----------|-------|
| **Definition** | Zero-crossing of d(sigma_std)/d(log g_bar) nearest to g-dagger, averaged across 4 non-parametric methods |
| **Dataset/Tier** | SPARC core sample (131 galaxies, 2,777 points after Q<=2, 30<=Inc<=85, Vobs>5, >=5 pts/galaxy) |
| **Scatter estimator** | `np.std()` (standard deviation) |
| **Binning** | Adaptive edges `np.arange(percentile(2), percentile(98), 0.30)`, min_count=20 |
| **Source data** | `analysis/results/summary_nonparametric_inversion.json` (test B4) |
| **Source code** | `analysis/pipeline/test_nonparametric_inversion.py` lines 333-350 (`compute_scatter_profile`) |
| **4 methods** | RAR parametric (-9.9712), LOESS (-9.9713), cubic spline (-9.9727), isotonic (-9.9663) |
| **Range** | max - min = 0.0064 dex, reported as "+/- 0.006" |

The value is **hardcoded** in the figure annotation at `make_figures.py` line 467.

## 3. Root Cause of the Discrepancy

The discrepancy is **not a tier mismatch** -- both values use the SPARC core sample.
It is a **scatter estimator mismatch**:

| Factor | Figure curve | Yellow box / text |
|--------|-------------|-------------------|
| Scatter metric | 1.4826 * MAD (robust) | np.std() (standard dev) |
| Bin edges | Fixed -12.5 to -8.5 | Data-adaptive (2nd-98th percentile) |
| Min count | 10 | 20 |
| N points | 2,800 | 2,777 |

### Controlled experiment (same data, same bins, different estimator)

Using identical pipeline-style bins on the same 2,800 SPARC points:

- **std-based crossing**: -9.889
- **MAD-based crossing**: -9.745
- **Difference**: 0.14 dex

The MAD estimator is more resistant to outliers in the heavy tails, producing a
different derivative profile shape. This shifts the zero-crossing by ~0.13-0.17 dex
depending on exact binning.

## 4. Is the Figure Mislabeled, the Text Wrong, or Both Acceptable?

**The figure is internally inconsistent.** Specifically:

- The plotted curve in panel (b) is the derivative of **MAD-based** scatter
  (correctly labeled "Robust scatter" in the caption).
- The blue arrow annotation says "plotted curve / crossing = -9.80", which is
  correct for what is plotted.
- The yellow box says "4 methods agree: crossing at -9.97 +/- 0.006", which is
  correct for the std-based pipeline result.
- **But the two numbers refer to different scatter estimators applied to the
  same data.** A reader would reasonably expect the yellow box to describe the
  plotted curve, not a separate computation.

The caption (main.tex lines 350-352) says: "The zero-crossing (blue dashed line)
coincides with g-dagger. Four independent non-parametric methods agree on the
crossing location to within 0.006 dex." This implies the blue line and the 0.006 dex
agreement are from the same computation -- they are not.

## 5. Downstream Consistency Check

| Location | Value quoted | Source | Correct? |
|----------|-------------|--------|----------|
| main.tex line 337 | -9.970 +/- 0.006 | summary_nonparametric_inversion.json (std) | Consistent with source |
| main.tex lines 420-421 | -9.890 +/- 0.180 | summary_binning_robustness.json (std, 25 configs) | Consistent with source |
| main.tex line 511 | -9.865 (Tier K) | summary_extended_rar_inversion.json (std) | Consistent with source |
| main.tex line 513 | -9.947 (Tier C) | summary_extended_rar_inversion.json (std) | Consistent with source |
| main.tex line 527 | -11.12 (PROBES) | summary_probes_inversion_replication.json (std) | Consistent with source |
| Abstract line 64 | 1.2 dex from g-dagger | PROBES result | Consistent |
| Yellow box (fig) | -9.97 +/- 0.006 | Hardcoded from nonparametric test (std) | Internally consistent but mismatched with plotted curve |

**No other downstream numbers are mixed.** All text values trace back to
std-based pipeline results. The mismatch is isolated to the figure: the plotted
curve uses MAD, but the annotation and text describe the std-based result.

### Additional note: Figure numbering vs filename

The paper's Figure 3 is generated from `fig4_inversion.pdf` and the paper's
Figure 4 is from `fig3_kurtosis.pdf`. The filenames are swapped relative to the
paper's figure ordering. This is cosmetic but could cause confusion during editing.

## 6. Recommended Fix

**Option A (preferred): Change the figure to plot std-based scatter.**
- In `make_figures.py` `_scatter_profile()`, replace `1.4826 * MAD` with `np.std()`
- This makes the plotted curve consistent with the yellow box, caption, and text
- The crossing will shift from -9.80 to approximately -9.90 (closer to the -9.97
  value, though exact match depends on bin edge alignment)
- Adjust min_pts from 10 to 20 to match pipeline

**Option B: Keep MAD but fix the annotation.**
- Remove the yellow box or change it to quote the MAD-based result
- Update main.tex line 337 and related text to report the MAD crossing
- This requires re-running the nonparametric test with MAD, which may change
  the "4 methods agree" result

**Option C (minimal): Add a footnote clarifying the two estimators.**
- Keep the figure as-is but add a caption note: "The plotted curve uses robust
  (MAD) scatter; the yellow box reports the standard-deviation-based result from
  the nonparametric inversion test (Section X)."
- This is transparent but may invite criticism

**Recommendation**: Option A is cleanest. The pipeline tests all use std.
The figure should match.

---

## Appendix: Diagnostic Commands

```bash
# Reproduce the MAD-based crossing (what the figure plots)
cd /Users/russelllicht/bec-dark-matter
python3 -c "
import sys; sys.path.insert(0, 'paper')
from make_figures import load_sparc, rar_pred, _scatter_profile, LOG_GDAG
import numpy as np
g, o = load_sparc(); r = o - rar_pred(g)
c, s, dc, ds = _scatter_profile(g, r, bin_width=0.30, offset=0.0)
for i in range(len(ds)-1):
    if np.isfinite(ds[i]) and np.isfinite(ds[i+1]) and ds[i]*ds[i+1]<0:
        xc = dc[i] + (dc[i+1]-dc[i])*(-ds[i]/(ds[i+1]-ds[i]))
        print(f'MAD crossing: {xc:.4f} (dist from g†: {xc-LOG_GDAG:+.4f})')
"

# Reproduce the std-based crossing (what the text reports)
python3 analysis/pipeline/test_nonparametric_inversion.py 2>&1 | grep "crossing"
```
