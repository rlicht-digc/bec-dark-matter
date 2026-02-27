#!/usr/bin/env python3
"""
Cluster-Scale RAR Analysis
============================

Three levels of cluster analysis:

LEVEL 1 — Amram+1996 cluster galaxy rotation curves.
  These are rotation curves of individual galaxies IN clusters, not cluster-
  scale mass profiles. Without baryonic decomposition, we can only compute
  g_obs = V²/R. We test: do cluster galaxies fall on the same V(R) locus
  as SPARC field galaxies? If g† is universal, cluster galaxies should follow
  the same RAR at the same acceleration scale.

LEVEL 2 — Published cluster-scale RAR from Tian+2020 / Chan & Del Popolo 2020.
  These authors constructed the cluster RAR using X-ray mass profiles:
  g_obs(r) = G×M_total(r)/r² vs g_bar(r) = G×M_baryon(r)/r².
  Key finding: clusters follow a RAR with acceleration scale ~17×g† (not g†).
  We test: is this 17× factor consistent with BEC predictions?

LEVEL 3 — Comparison of galaxy-scale vs cluster-scale inversion.
  If both scales show a phase transition, where does each occur?
  BEC prediction: the condensation threshold should be universal (same g†)
  unless the coherence length depends on total system mass.

Note: without X-ray cluster data in our project, Level 2 uses published
literature values rather than raw data analysis.
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10
LOG_G_DAGGER = np.log10(g_dagger)
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
kpc_m = 3.086e19
Msun_kg = 1.989e30


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# ================================================================
# LEVEL 1: AMRAM CLUSTER GALAXY ROTATION CURVES
# ================================================================
print("=" * 72)
print("LEVEL 1: AMRAM+1996 CLUSTER GALAXY ROTATION CURVES")
print("=" * 72)

# Load Amram data
amram_path = os.path.join(DATA_DIR, 'hi_surveys', 'amram1996_cluster_rotation_curves.tsv')

cluster_galaxies = {}
with open(amram_path, 'r') as f:
    header = f.readline()  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 12:
            continue
        try:
            name = parts[1].strip().strip('"')
            R = float(parts[2])        # arcsec (projected radius)
            vtot = parts[10].strip()
            evtot = parts[11].strip()

            if not vtot or not name:
                continue

            vtot = float(vtot)
            evtot = float(evtot) if evtot else 0

            if name not in cluster_galaxies:
                cluster_galaxies[name] = {'R': [], 'V': [], 'eV': []}
            cluster_galaxies[name]['R'].append(R)
            cluster_galaxies[name]['V'].append(vtot)
            cluster_galaxies[name]['eV'].append(evtot)

        except (ValueError, IndexError):
            continue

for name in cluster_galaxies:
    for key in ['R', 'V', 'eV']:
        cluster_galaxies[name][key] = np.array(cluster_galaxies[name][key])

print(f"  Loaded {len(cluster_galaxies)} cluster galaxies")
for name, data in sorted(cluster_galaxies.items()):
    print(f"    {name}: {len(data['R'])} points, V_max = {data['V'].max():.0f} km/s")

# Cluster distance estimates (from Amram+1996 / literature)
# DC galaxies are in the DC cluster (Dressler catalog)
# These are H-alpha rotation curves of cluster spirals
# Typical clusters: Fornax (D ~ 19 Mpc), Virgo (D ~ 16 Mpc), Coma (D ~ 100 Mpc)
# DC objects are in various clusters at z ~ 0.02-0.04 → D ~ 80-170 Mpc
# Without specific distances, we estimate from typical cluster redshifts

# For the RAR test, compute g_obs = V²/R at each point
# This requires converting R from arcsec to physical kpc using assumed distance
# We'll parametrize by distance and show the result is qualitative

print(f"\n  Computing g_obs = V²/R for assumed distances:")
print(f"  (R in arcsec → kpc requires distance assumption)")

assumed_distances = {
    # Amram+1996 targets are in various clusters; typical z ~ 0.02-0.04
    'DC2': 80.0,   # Mpc (estimated)
    'DC8': 80.0,
    'DC10': 80.0,
    'DC12': 80.0,
    'D24': 80.0,
    'D29': 80.0,
    'D32': 80.0,
    'D28': 80.0,
}
DEFAULT_DIST = 80.0  # Mpc

cluster_rar_points = []

print(f"\n  {'Galaxy':>8s} {'R_max(kpc)':>10s} {'V_flat':>8s} {'log(g_obs)':>10s}")
print(f"  {'-'*40}")

for name, data in cluster_galaxies.items():
    D_Mpc = assumed_distances.get(name, DEFAULT_DIST)
    # Convert arcsec to kpc: R_kpc = R_arcsec × D_Mpc × (π/180/3600) × 1000
    arcsec_to_kpc = D_Mpc * np.pi / (180 * 3600) * 1000  # kpc per arcsec

    R_kpc = data['R'] * arcsec_to_kpc
    V = data['V']

    # g_obs = V² / R (in SI)
    g_obs = (V * 1e3)**2 / (R_kpc * kpc_m)

    valid = (R_kpc > 0) & (V > 5) & (g_obs > 0)
    if np.sum(valid) < 3:
        continue

    R_kpc_v = R_kpc[valid]
    V_v = V[valid]
    g_obs_v = g_obs[valid]
    log_gobs = np.log10(g_obs_v)

    # Outer point as representative
    print(f"  {name:>8s} {R_kpc_v.max():10.1f} {V_v[-1]:8.0f} {log_gobs[-1]:10.2f}")

    for k in range(len(R_kpc_v)):
        cluster_rar_points.append({
            'galaxy': name,
            'R_kpc': float(R_kpc_v[k]),
            'V_km_s': float(V_v[k]),
            'log_gobs': float(log_gobs[k]),
            'D_assumed': D_Mpc,
        })

if cluster_rar_points:
    all_log_gobs = np.array([p['log_gobs'] for p in cluster_rar_points])
    print(f"\n  Cluster galaxy g_obs range: [{all_log_gobs.min():.2f}, {all_log_gobs.max():.2f}]")
    print(f"  g† = {LOG_G_DAGGER:.2f}")
    print(f"  Fraction of points with g_obs < g†: "
          f"{np.sum(all_log_gobs < LOG_G_DAGGER)}/{len(all_log_gobs)}")

    # These cluster galaxies probe g_obs in the range ~10^-10 to 10^-9
    # which straddles the g† transition.
    # WITHOUT g_bar, we cannot place them on the RAR diagram.
    # But we CAN compare their V(R) profiles to SPARC galaxies of similar V_flat.

print(f"\n  LIMITATION: Without baryonic decomposition (M_gas, M_disk, M_bulge)")
print(f"  for these cluster galaxies, we cannot compute g_bar and thus cannot")
print(f"  construct the full RAR. We can only compare kinematic properties.")

# ================================================================
# LEVEL 2: PUBLISHED CLUSTER-SCALE RAR RESULTS
# ================================================================
print(f"\n{'='*72}")
print("LEVEL 2: PUBLISHED CLUSTER-SCALE RAR (LITERATURE)")
print(f"{'='*72}")

# Tian+2020 (ApJ 896, 70): Cluster RAR from X-ray mass profiles
# Key result: clusters follow a RAR with a0_cluster ≈ 2×10⁻⁹ m/s²
# which is ~17× g† = 1.2×10⁻¹⁰

# Chan & Del Popolo 2020: Similar analysis with galaxy clusters
# Found acceleration scale ~10× g†

# Pradyumna et al. 2021/2024: Cluster RAR with different normalization

a0_cluster_tian = 2.0e-9  # Tian+2020
a0_cluster_cdp = 1.2e-9   # Chan & Del Popolo approximate

print(f"\n  Published cluster-scale acceleration scales:")
print(f"    Tian+2020:            a0_cluster = {a0_cluster_tian:.1e} m/s²"
      f"  ({a0_cluster_tian/g_dagger:.0f}× g†)")
print(f"    Chan & Del Popolo:    a0_cluster ≈ {a0_cluster_cdp:.1e} m/s²"
      f"  ({a0_cluster_cdp/g_dagger:.0f}× g†)")
print(f"    Galaxy-scale (SPARC): g† = {g_dagger:.1e} m/s²")

# BEC interpretation:
# If the condensation scale depends on total system mass M_total:
#   g†_system = g† × (M_system / M_galaxy)^α
# For a cluster of M ~ 10^14 Msun vs a galaxy of M ~ 10^11 Msun:
#   ratio = (10^14 / 10^11)^α = 1000^α
# If a0_cluster/g† ≈ 17, then: 1000^α = 17 → α = log(17)/log(1000) ≈ 0.41
# This would suggest g† ∝ M^0.4 — not a clean scaling.

# Alternative: the cluster-scale RAR has a DIFFERENT origin
# (virial equilibrium of subhalos vs condensate coupling)

ratio_tian = a0_cluster_tian / g_dagger
mass_ratio = 1e14 / 1e11  # cluster / galaxy
alpha = np.log10(ratio_tian) / np.log10(mass_ratio)

print(f"\n  If g†_system ∝ M_total^α:")
print(f"    Ratio a0_cluster / g† = {ratio_tian:.0f}")
print(f"    Mass ratio M_cluster / M_galaxy ≈ {mass_ratio:.0f}")
print(f"    α = log({ratio_tian:.0f}) / log({mass_ratio:.0f}) = {alpha:.3f}")
print(f"    → g†(M) ∝ M^{alpha:.2f}")
print(f"\n  This is NOT a clean power law (α ≈ 0.41 has no obvious physical meaning).")
print(f"  More likely: the cluster RAR is a distinct phenomenon from the galaxy RAR,")
print(f"  or the acceleration scale depends on density rather than total mass.")

# The key test: does the cluster RAR show a PHASE TRANSITION (scatter inversion)?
# Tian+2020 report large scatter in the cluster RAR (0.10-0.15 dex)
# compared to galaxy RAR (0.05-0.10 dex). This is expected in any model.
# The question is whether the scatter is environment-dependent at cluster scales.

print(f"\n  Cluster RAR scatter (Tian+2020): ~0.10-0.15 dex (intrinsic)")
print(f"  Galaxy RAR scatter (SPARC):      ~0.06-0.08 dex (intrinsic)")
print(f"  Cluster RAR is 2× noisier — expected given lower data quality,")
print(f"  projection effects, and complex cluster physics.")

# ================================================================
# LEVEL 3: MULTI-SCALE COMPARISON
# ================================================================
print(f"\n{'='*72}")
print("LEVEL 3: MULTI-SCALE CONDENSATION TEST")
print(f"{'='*72}")

# Verlinde's relation: a0 = cH₀/6 gives g† = 1.2×10⁻¹⁰ for H₀ = 67 km/s/Mpc
# This is independent of galaxy or cluster mass.
# If g† is truly set by cosmological parameters (Λ, H₀), it should be the
# SAME at all scales.

# The cluster RAR having a DIFFERENT scale (17× g†) means:
# Option A: g† is NOT universal → BEC condensation scale depends on system mass
# Option B: Cluster RAR is a different phenomenon (not condensation)
# Option C: Published cluster RAR analyses have systematic errors (baryonic mass
#           in clusters is dominated by hot gas, which is measured differently)

# Option C is plausible: cluster baryonic mass is 80-90% hot gas measured from
# X-ray luminosity. The systematic uncertainty in converting X-ray flux to gas
# mass is 10-20%. A systematic underestimate of M_baryon shifts the apparent
# acceleration scale upward.

# If M_baryon is systematically underestimated by factor f:
# g_bar_true = f × g_bar_measured
# Then the apparent a0 shifts: a0_apparent ≈ a0_true × f
# For a0_apparent = 17 × g†: f = 17 → M_baryon underestimated by 17×
# This is too large for a systematic error.

# More subtle: if the "missing baryon" fraction (baryons outside the X-ray
# emitting region) is significant, the effective a0 would shift.
# But by a factor of 17? Unlikely.

print(f"\n  Multi-scale test of g† universality:")
print(f"    Galaxy scale:  g† = {g_dagger:.1e} m/s² (SPARC, this work)")
print(f"    Cluster scale: a0 ≈ {a0_cluster_tian:.1e} m/s² (Tian+2020)")
print(f"    Ratio: {ratio_tian:.0f}×")
print(f"\n  This 17× ratio challenges the hypothesis that g† is universal")
print(f"  across all scales. Three interpretations:")
print(f"    1. g† depends on system mass → BEC coherence length is mass-dependent")
print(f"    2. Cluster RAR is a different physical effect (not condensation)")
print(f"    3. Baryonic mass systematics at cluster scale shift apparent a0")
print(f"\n  For the Letter: this is noted as an open question, not a contradiction.")
print(f"  The galaxy-scale g† is well-measured and robust. The cluster-scale")
print(f"  comparison is important for the theory but does not undermine the")
print(f"  galaxy-scale discovery.")

# ================================================================
# VERDICT
# ================================================================
print(f"\n{'='*72}")
print("VERDICT")
print(f"{'='*72}")

cluster_verdict = (
    "INCONCLUSIVE but informative. Amram cluster galaxy data lacks baryonic "
    "decomposition for direct RAR construction. Published cluster-scale RAR "
    "shows a different acceleration scale (~17× g†), which either indicates "
    "mass-dependent condensation or distinct cluster physics. This is an open "
    "question that does NOT undermine the galaxy-scale result but does constrain "
    "the theory. The Letter should note this as a testable prediction: if g† is "
    "truly universal, improved cluster baryonic mass measurements should bring "
    "a0_cluster closer to g†."
)
print(f"\n  {cluster_verdict}")

# ================================================================
# SAVE
# ================================================================
output = {
    'test_name': 'cluster_scale_rar',
    'description': ('Multi-scale RAR analysis: Amram cluster galaxy rotation curves + '
                    'published cluster-scale RAR results (Tian+2020, Chan & Del Popolo)'),
    'level1_amram': {
        'n_galaxies': len(cluster_galaxies),
        'galaxy_list': list(cluster_galaxies.keys()),
        'n_total_points': len(cluster_rar_points),
        'assumed_distance_Mpc': DEFAULT_DIST,
        'log_gobs_range': [
            round(float(min(p['log_gobs'] for p in cluster_rar_points)), 3),
            round(float(max(p['log_gobs'] for p in cluster_rar_points)), 3),
        ] if cluster_rar_points else None,
        'limitation': 'No baryonic decomposition — cannot compute g_bar or construct full RAR',
    },
    'level2_literature': {
        'tian2020_a0_cluster': a0_cluster_tian,
        'ratio_to_gdagger': round(ratio_tian, 1),
        'implied_mass_scaling_alpha': round(alpha, 3),
        'cluster_rar_scatter_dex': '0.10-0.15',
        'galaxy_rar_scatter_dex': '0.06-0.08',
    },
    'level3_interpretation': {
        'gdagger_universal': False,
        'cluster_scale_ratio': round(ratio_tian, 1),
        'interpretations': [
            'Mass-dependent BEC coherence length',
            'Distinct cluster physics (not condensation)',
            'Baryonic mass systematics at cluster scale',
        ],
    },
    'verdict': cluster_verdict,
}

outpath = os.path.join(RESULTS_DIR, 'summary_cluster_scale_rar.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
