#!/usr/bin/env python3
"""
TNG100-1 Radial Mass Profile Extraction — For TNG JupyterLab
=============================================================

Extracts enclosed mass profiles (M_star, M_gas, M_dm) at 15 logarithmically
spaced radii from 0.5×R_half to 5×R_half for ~20K TNG galaxies.

Purpose: enable ACF / Lomb-Scargle coherence analysis on simulation data
with sufficient radial resolution (15 points per galaxy vs the 2-4 available
in the group catalog apertures).

=== INSTRUCTIONS ===

1. Go to https://www.tng-project.org/data/lab/
2. Log in with your TNG account
3. Open a new terminal (File > New > Terminal)
4. Upload this script:
     - Click the upload button in the file browser, OR
     - Use: wget <url-of-this-file>
5. Run:  python3 tng_extract_profiles.py
6. Wait ~20-40 minutes (progress will be printed)
7. Download the output file: tng_mass_profiles.npz  (~5-20 MB)
8. Place it in ~/Desktop/tng_cross_validation/

=== OUTPUT FORMAT ===

NPZ file with arrays:
  galaxy_ids      (N,)     Subhalo IDs
  r_half_kpc      (N,)     Stellar half-mass radius [pkpc]
  vmax            (N,)     Maximum circular velocity [km/s]
  m_star_total    (N,)     Total stellar mass [M_sun]
  radii_kpc       (N, 15)  Physical radii of each shell [pkpc]
  m_star_enc      (N, 15)  Enclosed stellar mass [M_sun]
  m_gas_enc       (N, 15)  Enclosed gas mass [M_sun]
  m_dm_enc        (N, 15)  Enclosed DM mass [M_sun]
  r_multipliers   (15,)    Radii as multiples of R_half

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import time
import numpy as np

# ── Try to import illustris_python (standard on TNG JupyterLab) ──
try:
    import illustris_python as il
    print("illustris_python loaded successfully")
except ImportError:
    print("ERROR: illustris_python not found.")
    print("This script must be run on the TNG JupyterLab server.")
    print("Go to: https://www.tng-project.org/data/lab/")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# TNG data paths (try common JupyterLab locations)
POSSIBLE_PATHS = [
    '/home/tnguser/sims.TNG/TNG100-1/output',
    '/virgo/simulations/IllustrisTNG/TNG100-1/output',
    '/home/tnguser/sims.TNG/L75n1820TNG/output',
    os.path.expanduser('~/sims.TNG/TNG100-1/output'),
]

basePath = None
for p in POSSIBLE_PATHS:
    if os.path.isdir(p):
        basePath = p
        break

if basePath is None:
    print("ERROR: Could not find TNG100-1 simulation data.")
    print("Tried paths:")
    for p in POSSIBLE_PATHS:
        print(f"  {p}")
    print("\nPlease set the correct basePath in this script.")
    sys.exit(1)

SNAP_NUM = 99          # z = 0
N_RADII = 15           # Number of radial bins per galaxy
R_MIN_FACTOR = 0.5     # Inner radius = 0.5 × R_half
R_MAX_FACTOR = 5.0     # Outer radius = 5.0 × R_half
M_STAR_MIN = 1e9       # Minimum stellar mass [M_sun]
R_HALF_MIN = 0.5       # Minimum R_half [pkpc] — avoid resolution artifacts
N_STAR_MIN = 100       # Minimum star particles per galaxy
CHECKPOINT_EVERY = 1000

# TNG unit conversions (at z=0, comoving = physical)
H = 0.6774
CONV_MASS = 1e10 / H   # code mass → M_sun
CONV_DIST = 1.0 / H    # code length → pkpc (at z=0)

# Output
OUT_FILE = 'tng_mass_profiles.npz'
CHECKPOINT_FILE = 'tng_mass_profiles_checkpoint.npz'

# Log-spaced radius multipliers
R_MULTIPLIERS = np.logspace(np.log10(R_MIN_FACTOR), np.log10(R_MAX_FACTOR), N_RADII)

print("=" * 70)
print("TNG100-1 RADIAL MASS PROFILE EXTRACTION")
print("=" * 70)
print(f"  basePath: {basePath}")
print(f"  Snapshot: {SNAP_NUM} (z=0)")
print(f"  Radii: {N_RADII} points, {R_MIN_FACTOR}–{R_MAX_FACTOR} × R_half")
print(f"  R_multipliers: {np.array2string(R_MULTIPLIERS, precision=3)}")
print(f"  Selection: M* > {M_STAR_MIN:.0e} M_sun, R_half > {R_HALF_MIN} pkpc, "
      f"N_star >= {N_STAR_MIN}")
print(f"  Output: {OUT_FILE}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD GROUP CATALOG & SELECT GALAXIES
# ══════════════════════════════════════════════════════════════════════
t0 = time.time()

print(f"\n[1] Loading group catalog...")
fields = [
    'SubhaloMassType',          # (N, 6) — mass by particle type
    'SubhaloHalfmassRadType',   # (N, 6) — half-mass radius by type
    'SubhaloPos',               # (N, 3) — position in box
    'SubhaloFlag',              # (N,)   — 1 = cosmological origin
    'SubhaloVmax',              # (N,)   — max circular velocity
    'SubhaloLenType',           # (N, 6) — particle count by type
]
subhalos = il.groupcat.loadSubhalos(basePath, SNAP_NUM, fields=fields)
header = il.groupcat.loadHeader(basePath, SNAP_NUM)
box_size = header['BoxSize'] * CONV_DIST  # pkpc

n_total = subhalos['count']
print(f"  Total subhalos: {n_total}")
print(f"  Box size: {box_size:.1f} pkpc")

# Convert units
m_star_all = subhalos['SubhaloMassType'][:, 4] * CONV_MASS   # M_sun
r_half_all = subhalos['SubhaloHalfmassRadType'][:, 4] * CONV_DIST  # pkpc
flags = subhalos['SubhaloFlag']
n_star_all = subhalos['SubhaloLenType'][:, 4]
vmax_all = subhalos['SubhaloVmax']  # km/s (already physical)
pos_all = subhalos['SubhaloPos'] * CONV_DIST  # pkpc

# Selection
sel = ((flags == 1) &
       (m_star_all > M_STAR_MIN) &
       (r_half_all > R_HALF_MIN) &
       (n_star_all >= N_STAR_MIN))
indices = np.where(sel)[0]
n_sel = len(indices)

print(f"  Selected: {n_sel} galaxies")
print(f"  M_star range: [{m_star_all[indices].min():.2e}, {m_star_all[indices].max():.2e}] M_sun")
print(f"  R_half range: [{r_half_all[indices].min():.2f}, {r_half_all[indices].max():.2f}] pkpc")
print(f"  Median N_star: {np.median(n_star_all[indices]):.0f}")

elapsed = time.time() - t0
print(f"  [{elapsed:.0f}s]")


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: CHECK FOR EXISTING CHECKPOINT
# ══════════════════════════════════════════════════════════════════════
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    try:
        ckpt = np.load(CHECKPOINT_FILE, allow_pickle=True)
        start_idx = int(ckpt['n_processed'])
        print(f"\n[!] Resuming from checkpoint: {start_idx}/{n_sel} already processed")
    except Exception as e:
        print(f"\n[!] Checkpoint exists but unreadable ({e}), starting from scratch")
        start_idx = 0


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: EXTRACT MASS PROFILES
# ══════════════════════════════════════════════════════════════════════
print(f"\n[2] Extracting mass profiles for {n_sel} galaxies...")
print(f"    Particle types: 0=gas, 1=DM, 4=stars")
print(f"    Method: il.snapshot.loadSubhalo() → distance sort → cumulative mass")

# Pre-allocate output arrays
out_galaxy_ids = np.zeros(n_sel, dtype=np.int64)
out_r_half = np.zeros(n_sel, dtype=np.float64)
out_vmax = np.zeros(n_sel, dtype=np.float64)
out_m_star_total = np.zeros(n_sel, dtype=np.float64)
out_radii = np.zeros((n_sel, N_RADII), dtype=np.float64)
out_m_star_enc = np.zeros((n_sel, N_RADII), dtype=np.float64)
out_m_gas_enc = np.zeros((n_sel, N_RADII), dtype=np.float64)
out_m_dm_enc = np.zeros((n_sel, N_RADII), dtype=np.float64)

# Load checkpoint data if resuming
if start_idx > 0:
    out_galaxy_ids[:start_idx] = ckpt['galaxy_ids'][:start_idx]
    out_r_half[:start_idx] = ckpt['r_half_kpc'][:start_idx]
    out_vmax[:start_idx] = ckpt['vmax'][:start_idx]
    out_m_star_total[:start_idx] = ckpt['m_star_total'][:start_idx]
    out_radii[:start_idx] = ckpt['radii_kpc'][:start_idx]
    out_m_star_enc[:start_idx] = ckpt['m_star_enc'][:start_idx]
    out_m_gas_enc[:start_idx] = ckpt['m_gas_enc'][:start_idx]
    out_m_dm_enc[:start_idx] = ckpt['m_dm_enc'][:start_idx]

n_errors = 0
t_extract = time.time()


def compute_enclosed_mass(coords, masses, center, radii_kpc, box_size):
    """Compute enclosed mass at each radius, handling periodic boundaries."""
    dx = coords - center
    # Periodic wrapping
    dx[dx > box_size / 2] -= box_size
    dx[dx < -box_size / 2] += box_size
    r = np.sqrt(np.sum(dx**2, axis=1))

    # Sort by radius and compute cumulative mass
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    m_cumul = np.cumsum(masses[sort_idx])

    # Interpolate to get enclosed mass at target radii
    # np.searchsorted is faster than np.interp for this case
    enclosed = np.zeros(len(radii_kpc))
    for j, rj in enumerate(radii_kpc):
        idx = np.searchsorted(r_sorted, rj, side='right')
        if idx > 0:
            enclosed[j] = m_cumul[idx - 1]
    return enclosed


for i in range(start_idx, n_sel):
    sub_id = int(indices[i])
    rh = r_half_all[sub_id]
    center = pos_all[sub_id]
    radii_kpc = R_MULTIPLIERS * rh

    out_galaxy_ids[i] = sub_id
    out_r_half[i] = rh
    out_vmax[i] = vmax_all[sub_id]
    out_m_star_total[i] = m_star_all[sub_id]
    out_radii[i] = radii_kpc

    # Load and process each particle type
    for ptype, out_arr in [(4, out_m_star_enc),
                            (0, out_m_gas_enc),
                            (1, out_m_dm_enc)]:
        try:
            data = il.snapshot.loadSubhalo(basePath, SNAP_NUM, sub_id, ptype,
                                            fields=['Coordinates', 'Masses'])
            if data['count'] == 0:
                continue

            coords = data['Coordinates'].astype(np.float64) * CONV_DIST  # pkpc
            masses = data['Masses'].astype(np.float64) * CONV_MASS       # M_sun

            enclosed = compute_enclosed_mass(coords, masses, center, radii_kpc, box_size)
            out_arr[i] = enclosed

        except Exception as e:
            if i < 5:  # Only print errors for first few galaxies
                print(f"    WARNING: sub_id={sub_id}, ptype={ptype}: {e}")
            n_errors += 1
            continue

    # Progress reporting
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_extract
        rate = (i + 1 - start_idx) / max(elapsed, 1)
        remaining = (n_sel - i - 1) / max(rate, 0.01)
        print(f"  {i+1}/{n_sel}  ({rate:.1f} gal/s, ~{remaining/60:.0f} min remaining)  "
              f"[{elapsed:.0f}s elapsed]")
        sys.stdout.flush()

    # Checkpoint save
    if (i + 1) % CHECKPOINT_EVERY == 0:
        np.savez_compressed(CHECKPOINT_FILE,
                            galaxy_ids=out_galaxy_ids,
                            r_half_kpc=out_r_half,
                            vmax=out_vmax,
                            m_star_total=out_m_star_total,
                            radii_kpc=out_radii,
                            m_star_enc=out_m_star_enc,
                            m_gas_enc=out_m_gas_enc,
                            m_dm_enc=out_m_dm_enc,
                            r_multipliers=R_MULTIPLIERS,
                            n_processed=i + 1)
        print(f"    [Checkpoint saved at {i+1}]")

elapsed_extract = time.time() - t_extract
print(f"\n  Extraction complete: {n_sel} galaxies in {elapsed_extract:.0f}s")
print(f"  Errors (individual particle loads): {n_errors}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 4: VALIDATE & SAVE
# ══════════════════════════════════════════════════════════════════════
print(f"\n[3] Validating and saving...")

# Sanity checks
n_valid = np.sum(out_m_star_enc[:, -1] > 0)  # Has any stellar mass at outermost radius
n_with_gas = np.sum(out_m_gas_enc[:, -1] > 0)
n_with_dm = np.sum(out_m_dm_enc[:, -1] > 0)

print(f"  Galaxies with stellar mass profile:  {n_valid}/{n_sel}")
print(f"  Galaxies with gas profile:           {n_with_gas}/{n_sel}")
print(f"  Galaxies with DM profile:            {n_with_dm}/{n_sel}")

# Spot-check a few galaxies
print(f"\n  Spot check (first 3 galaxies):")
for j in range(min(3, n_sel)):
    gid = out_galaxy_ids[j]
    rh = out_r_half[j]
    ms = out_m_star_enc[j]
    mg = out_m_gas_enc[j]
    md = out_m_dm_enc[j]
    print(f"    sub_id={gid}, R_half={rh:.2f} pkpc")
    print(f"      Radii (pkpc): {np.array2string(out_radii[j], precision=1)}")
    print(f"      M_star at R_half: {ms[np.argmin(np.abs(R_MULTIPLIERS-1.0))]:.2e} M_sun")
    print(f"      M_dm at 5R_half:  {md[-1]:.2e} M_sun")
    print(f"      M_gas at 5R_half: {mg[-1]:.2e} M_sun")

# Save final output
np.savez_compressed(OUT_FILE,
                    galaxy_ids=out_galaxy_ids,
                    r_half_kpc=out_r_half,
                    vmax=out_vmax,
                    m_star_total=out_m_star_total,
                    radii_kpc=out_radii,
                    m_star_enc=out_m_star_enc,
                    m_gas_enc=out_m_gas_enc,
                    m_dm_enc=out_m_dm_enc,
                    r_multipliers=R_MULTIPLIERS,
                    n_radii=N_RADII,
                    r_min_factor=R_MIN_FACTOR,
                    r_max_factor=R_MAX_FACTOR,
                    m_star_min=M_STAR_MIN,
                    h=H,
                    description=('TNG100-1 enclosed mass profiles at 15 log-spaced radii '
                                 'from 0.5*R_half to 5*R_half. Units: mass in M_sun, '
                                 'radii in physical kpc.'))

file_size = os.path.getsize(OUT_FILE) / 1e6
total_time = time.time() - t0

print(f"\n  Saved: {OUT_FILE} ({file_size:.1f} MB)")

# Clean up checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"  Removed checkpoint file")

print(f"\n  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print("=" * 70)
print("Done! Download tng_mass_profiles.npz and place it in your local")
print("bec-dark-matter project for coherence analysis.")
print("=" * 70)
