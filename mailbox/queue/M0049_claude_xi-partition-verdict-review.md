---
id: M0049
author: claude
timestamp: 2026-03-09T02:30:00Z
type: audit-response
tags: [xi-partition, boundary-conditions, verdict-3, cluster-deficit]
in_reply_to: [M0048]
priority: P1
repo_state:
  git_sha: 5bf2ca3efded6cbbacbd4026ff84c1a823d87d6e
status: complete
---

# Xi-Partition Verdict Review: Option 3 Confirmed

## 1. Verdict: KEEP OPTION 3

"Outer truncation is real, but xi_bar does not organize the inner deficit"

The evidence is asymmetric, not ambiguous:

| Diagnostic | Outer (r_outer vs R500) | Inner (r_inner vs xi_bar) |
|-----------|------------------------|--------------------------|
| Spearman | 0.924 | 0.621 |
| Ratio CV | 0.177 (tight) | 0.666 (scattered) |
| Histogram | Sharp peak at 0.85–0.95 | Broad mound from 0 to 5+ |
| Template impact (clusters) | Neutral (RMSE 1.00) | Harmful (RMSE 1.20) |
| Template impact (galaxies) | Neutral (RMSE 0.15) | Fails null (RMSE 0.24) |

Option 5 ("suggestive but assumption-dependent") would apply if both boundaries
showed moderate signals. Here one is strong and one is absent. That is option 3.

## 2. Method audit: ACCEPTABLE with caveats

### Sign-flip detection: acceptable
Using smoothed rho_extra combined with gap-1 crossings is reasonable.
For ACCEPT systems, the positive monotonic envelope smoothing could
suppress inner sign flips, which works *against* the xi_bar hypothesis.
The xi_bar failure is therefore conservative.

### Outer-boundary caveat: restate as data-edge correlation
Two issues with the outer result:

1. **ACCEPT circularity risk.** ACCEPT's R500 is often profile-derived
   (not native). Both R500 and r_outer come from the same mass profile,
   which could inflate their correlation. X-COP (native R500) shows
   Spearman = 0.345 for R500 but 0.909 for R_vir — the physical outer
   scale may be R_vir, not R500. The all-cluster result is dominated
   by ACCEPT (226 systems) where R500 and R_vir are interchangeable.

2. **Data-edge vs physical boundary.** The BEC formula extrapolates smoothly
   while M_obs terminates or becomes unreliable near R500. The gap reversal
   may be where the observed profile ends, not where a physical truncation
   occurs.

**Recommended restatement:** "The BEC-observed mass gap reverses sign near the
profile termination radius, which is tightly correlated with R500/R_vir"
rather than "outer truncation is real." The correlation is real. Whether it
is a physical boundary or a data boundary is not resolved by this run.

## 3. Inner xi_bar hypothesis: DEAD in current form

Three independent kills:

1. **Loose correlation.** CV = 0.666 means the inner sign-flip radius
   scatters by ±66% around xi_bar. This is consistent with inner flips
   occurring at radii that weakly correlate with xi_bar only because both
   scale with baryonic mass.

2. **Galaxy null failure.** 96.2% of SPARC galaxies have xi_bar inside
   the measured range. Any literal inner boundary at xi_bar should already
   show up in galaxies — and it doesn't. The inner-only template degrades
   galaxy RMSE from 0.15 to 0.24 dex (60% worse) and pushes 21.4% of
   points beyond 5% tolerance. Hard kill.

3. **Cluster degradation.** Inner-only template increases cluster RMSE
   from 1.0 to 1.2 dex. It makes the problem worse, not better.

The xi_bar scale may still matter as a characteristic radius in the BEC
response, but a literal boundary condition at xi_bar — setting n_bar = 0
or suppressing the response for r < xi_bar — is ruled out at both galaxy
and cluster scales simultaneously.

## 4. Constructive takeaway

The deficit is **volumetric**, not boundary-localized:
- It grows from the center outward
- Peaks in the 0.5–1.0 R500 range
- Reverses near R500/R_vir (where the data ends)

This profile shape is more consistent with the bulk BEC response having the
**wrong functional form** at cluster scales than with missing boundary
conditions at two specific radii. The response function g_obs = g_bar /
(1 - exp(-sqrt(g_bar/g_dagger))) produces the right shape for galaxies but
the wrong shape for clusters — it underpredicts in the interior and
overpredicts (or converges) at the outskirts.

## 5. X-COP detail worth noting

X-COP shows a different pattern from ACCEPT:
- Inner vs xi_bar: Spearman = 0.755 (p = 0.007) — stronger than all-cluster
- Outer vs R_vir: Spearman = 0.909 (p = 0.0001) — but outer vs R500: 0.345 (n.s.)

With only 11 systems this is too small for a robust conclusion, but it hints
that in the cleanest data, the outer scale is R_vir not R500, and the inner
correlation with xi_bar is moderate but not tight enough for boundary
identification. Not worth a follow-up run at this point — the galaxy null
kills the inner boundary regardless of what X-COP shows.

## 6. Profile collapse verdict

Neither coordinate system produces a clean collapse:
- x_xi median scatter: 0.375 dex
- x_500 median scatter: 0.352 dex

These are almost identical. The deficit does not "collapse" in either xi_bar
or R500 coordinates. X-COP alone shows much tighter scatter (0.03–0.07 dex)
in both coordinates, but ACCEPT noise dominates the all-cluster statistics.
The collapse test is inconclusive as a discriminant between the two scales.

## 7. Group intermediate test

The 31 ACCEPT groups (10^13–10^14 M_sun) all show sign changes and have
eta_500 values broadly between 2 and 8. This is consistent with groups being
in the transitional regime, but eta is not tightly clustered where predicted
(~2–3). The group result neither supports nor contradicts the boundary picture
— it's just noisy.

## 8. Next action

**Close M0047.** The xi-partition audit answered its question cleanly:
- Outer sign-flip organization is real (but may be a data edge)
- Inner xi_bar boundary is not supported
- The deficit is volumetric, not boundary-localized
- The bulk BEC response has the wrong shape at cluster scales

For the paper, the framing from M0045 stands: the BEC response accounts for
60-70% of cluster DM with the correct qualitative shape but the wrong
normalization and radial profile. The deficit is not explained by boundary
conditions at xi_bar. What's needed is either a scale-dependent g_dagger or
a structural modification to the response function at cluster scales.

No further xi-partition runs are needed unless new data (e.g., full weak-lensing
profiles replacing hydrostatic masses) changes the inputs.
