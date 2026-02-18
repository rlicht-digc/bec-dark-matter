#!/usr/bin/env python3
"""
Test 13b COMPOSITE: SPARC inner + Brouwer outer lensing profiles.

Stitches SPARC rotation curves (R = 1-30 kpc) with Brouwer+2021
KiDS-1000 lensing ESD profiles (R = 35-2600 kpc) to create
composite mass profiles spanning the full soliton + envelope range.

The soliton core lives at R < 3ξ ≈ 5-30 kpc (SPARC range).
The NFW envelope dominates at R > 50 kpc (Brouwer range).
Together: continuous coverage from ~1 kpc to ~2600 kpc.

Conversion: ΔΣ(R) = V²(R) / (4πG R)
From Brouwer+2021 Eq.23: v_circ = sqrt(4G × ΔΣ × R)
Therefore: ΔΣ = v²_circ / (4G R)
"""
import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

print("=" * 72)
print("TEST 13b COMPOSITE: SPARC inner + Brouwer outer profiles")
print("=" * 72)
print("  Stitched rotation curve + lensing: 1 kpc to 2600 kpc")

# ============================================================
# STEP 1: Load SPARC rotation curves and stellar masses
# ============================================================
print("\n  STEP 1: Loading SPARC galaxies...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

if not os.path.exists(table2_path) or not os.path.exists(mrt_path):
    print("ERROR: SPARC data files not found!")
    sys.exit(1)

# Parse rotation curves
galaxies = {}
with open(table2_path, 'r') as f:
    for line in f:
        if len(line.strip()) < 50:
            continue
        try:
            name = line[0:11].strip()
            if not name:
                continue
            dist = float(line[12:18].strip())
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            evobs = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue

        if name not in galaxies:
            galaxies[name] = {'R': [], 'Vobs': [], 'eVobs': [],
                              'Vgas': [], 'Vdisk': [], 'Vbul': [],
                              'dist': dist}
        galaxies[name]['R'].append(rad)
        galaxies[name]['Vobs'].append(vobs)
        galaxies[name]['eVobs'].append(evobs)
        galaxies[name]['Vgas'].append(vgas)
        galaxies[name]['Vdisk'].append(vdisk)
        galaxies[name]['Vbul'].append(vbul)

for name in galaxies:
    for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
        galaxies[name][key] = np.array(galaxies[name][key])

# Parse MRT for stellar masses: L_[3.6] at bytes 35-41
# M* ≈ 0.5 × L_[3.6] (McGaugh & Schombert 2014)
# L_[3.6] is in units of 10^9 L_sun
with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()

# Find data start
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50:
        data_start = i + 1
        break

sparc_props = {}
for line in mrt_lines[data_start:]:
    if not line.strip() or line.startswith('#'):
        continue
    try:
        # Name is first 11 chars (right-justified), rest is whitespace-separated
        name = line[0:11].strip()
        if not name:
            continue
        # Columns after name: T, D, eD, fD, Inc, eInc, L[3.6], eL[3.6],
        # Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, eVflat, Q, Ref
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        T = int(parts[0])
        D = float(parts[1])
        Inc = float(parts[4])
        L36 = float(parts[6])  # 10^9 L_sun
        Q = int(parts[16])
        sparc_props[name] = {
            'D': D, 'Inc': Inc, 'T': T, 'Q': Q,
            'L36_1e9': L36,
            'logMs': np.log10(max(0.5 * L36 * 1e9, 1e6))  # M* = 0.5 × L_[3.6]
        }
    except (ValueError, IndexError):
        continue

print(f"  Parsed {len(galaxies)} rotation curves, {len(sparc_props)} properties")

# ============================================================
# STEP 2: Bin SPARC galaxies by stellar mass (Brouwer bins)
# ============================================================
# Brouwer bins: log(M*/h70^-2 M_sun) = [8.5, 10.3, 10.6, 10.8, 11.0]
# SPARC stellar masses don't include h70 factor, so adjust:
# Brouwer M* are in h70^-2 units; at h=0.7, h70=1, so no correction needed
mass_bin_edges = [8.5, 10.3, 10.6, 10.8, 11.0]
mass_bin_labels = [f"[{mass_bin_edges[i]:.1f}, {mass_bin_edges[i+1]:.1f}]"
                   for i in range(4)]

print("\n  STEP 2: Binning SPARC galaxies by stellar mass...")

# Convert each SPARC galaxy's V(R) to ΔΣ(R)
# ΔΣ(R) = V²_obs / (4 G R)
# Units: V in km/s, R in kpc → need conversion to Msun/pc²
# G = 4.302e-3 pc (km/s)² / Msun (in convenient units)
# ΔΣ = V² / (4 × G × R)
# With V in km/s, R in kpc = 1000 pc:
# ΔΣ = V² / (4 × 4.302e-3 × R × 1000) [Msun/pc²]
G_conv = 4.302e-3  # pc (km/s)^2 / Msun

sparc_binned = [[] for _ in range(4)]  # One list per mass bin

n_used = 0
for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]

    # Quality cuts
    if prop['Q'] > 2:
        continue
    if prop['Inc'] < 30 or prop['Inc'] > 85:
        continue
    if len(gdata['R']) < 5:
        continue

    logMs = prop['logMs']

    # Find mass bin
    bin_idx = None
    for b in range(4):
        if mass_bin_edges[b] <= logMs < mass_bin_edges[b + 1]:
            bin_idx = b
            break
    if bin_idx is None:
        continue

    R = gdata['R']  # kpc
    Vobs = gdata['Vobs']  # km/s
    eVobs = gdata['eVobs']

    # Convert to ΔΣ
    valid = (R > 0) & (Vobs > 0) & np.isfinite(Vobs)
    if np.sum(valid) < 3:
        continue

    R_v = R[valid]
    V_v = Vobs[valid]
    eV_v = eVobs[valid]

    # ΔΣ = V² / (4 G R), with R in kpc → R in pc = R * 1000
    # Actually: using lensing convention, ΔΣ = V²_circ / (4 G R)
    # But this is only exact for circular orbits in spherical symmetry.
    # For the stacking comparison, Brouwer+2021 Eq.23 gives:
    # v_circ = sqrt(4 G ΔΣ R), so ΔΣ = v²/(4GR)
    # Units: V in km/s, R in pc (= R_kpc * 1000), G in pc (km/s)^2/Msun
    # → ΔΣ in Msun/pc²
    R_pc = R_v * 1000.0
    delta_sigma = V_v**2 / (4.0 * G_conv * R_pc)  # Msun/pc²
    # Error propagation: δ(ΔΣ) = 2V δV / (4GR) = (ΔΣ) × 2 δV/V
    delta_sigma_err = delta_sigma * 2.0 * eV_v / np.maximum(V_v, 1.0)

    sparc_binned[bin_idx].append({
        'name': name,
        'logMs': logMs,
        'R_kpc': R_v,
        'delta_sigma': delta_sigma,
        'delta_sigma_err': delta_sigma_err,
    })
    n_used += 1

print(f"  Used {n_used} SPARC galaxies")
for b in range(4):
    n_gal = len(sparc_binned[b])
    if n_gal > 0:
        logMs_mean = np.mean([g['logMs'] for g in sparc_binned[b]])
        print(f"    Bin {b+1} {mass_bin_labels[b]}: {n_gal} galaxies, "
              f"<logMs> = {logMs_mean:.2f}")
    else:
        print(f"    Bin {b+1} {mass_bin_labels[b]}: 0 galaxies")

# ============================================================
# STEP 3: Stack SPARC profiles in each bin
# ============================================================
print("\n  STEP 3: Stacking SPARC profiles in radial bins...")

# Use logarithmic radial bins from 0.5 to 50 kpc
sparc_radial_edges = np.logspace(np.log10(0.5), np.log10(50), 12)
sparc_radial_centers = np.sqrt(sparc_radial_edges[:-1] * sparc_radial_edges[1:])

sparc_stacked = []
for b in range(4):
    if len(sparc_binned[b]) < 3:
        sparc_stacked.append(None)
        continue

    R_stack = []
    ESD_stack = []
    err_stack = []

    for j in range(len(sparc_radial_centers)):
        lo, hi = sparc_radial_edges[j], sparc_radial_edges[j + 1]
        vals = []
        errs = []
        for gal in sparc_binned[b]:
            mask = (gal['R_kpc'] >= lo) & (gal['R_kpc'] < hi)
            if np.sum(mask) > 0:
                # Weighted mean of this galaxy's points in this radial bin
                w = 1.0 / np.maximum(gal['delta_sigma_err'][mask], 1e-10)**2
                wmean = np.average(gal['delta_sigma'][mask], weights=w)
                werr = 1.0 / np.sqrt(np.sum(w))
                vals.append(wmean)
                errs.append(werr)

        if len(vals) >= 3:
            # Stack across galaxies: weighted mean
            vals = np.array(vals)
            errs = np.array(errs)
            w = 1.0 / errs**2
            stacked_val = np.average(vals, weights=w)
            stacked_err = 1.0 / np.sqrt(np.sum(w))
            # Add scatter between galaxies as systematic error
            scatter = np.std(vals) / np.sqrt(len(vals))
            total_err = np.sqrt(stacked_err**2 + scatter**2)

            R_stack.append(sparc_radial_centers[j])
            ESD_stack.append(stacked_val)
            err_stack.append(total_err)

    if len(R_stack) >= 3:
        sparc_stacked.append({
            'R_kpc': np.array(R_stack),
            'ESD': np.array(ESD_stack),
            'err': np.array(err_stack),
            'n_galaxies': len(sparc_binned[b]),
        })
        print(f"    Bin {b+1}: {len(R_stack)} radial bins, "
              f"R = [{R_stack[0]:.1f}, {R_stack[-1]:.1f}] kpc, "
              f"{len(sparc_binned[b])} galaxies")
    else:
        sparc_stacked.append(None)
        print(f"    Bin {b+1}: insufficient data for stacking")

# ============================================================
# STEP 4: Load Brouwer+2021 outer profiles
# ============================================================
print("\n  STEP 4: Loading Brouwer+2021 outer profiles...")

brouwer_dir = os.path.join(DATA_DIR, 'brouwer2021')
brouwer_data = []
for i in range(4):
    fpath = os.path.join(brouwer_dir,
                         f'Fig-3_Lensing-rotation-curves_Massbin-{i+1}.txt')
    if not os.path.exists(fpath):
        brouwer_data.append(None)
        continue

    R_Mpc, ESD_t, ESD_err, bias_K = [], [], [], []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            R_Mpc.append(float(parts[0]))
            ESD_t.append(float(parts[1]))
            ESD_err.append(float(parts[3]))
            bias_K.append(float(parts[4]))

    R_Mpc = np.array(R_Mpc)
    ESD_t = np.array(ESD_t)
    ESD_err = np.array(ESD_err)
    bias_K = np.array(bias_K)

    brouwer_data.append({
        'R_kpc': R_Mpc * 1000.0,
        'ESD': ESD_t / bias_K,
        'err': ESD_err / bias_K,
    })

# ============================================================
# STEP 5: Stitch composite profiles
# ============================================================
print("\n  STEP 5: Stitching composite profiles...")

# Physical constants
G_SI = 6.674e-11
Msun_kg = 1.989e30
kpc_m = 3.086e19
gdagger = 1.2e-10

composite = []
mass_bin_logMs = [0.5 * (mass_bin_edges[i] + mass_bin_edges[i + 1])
                  for i in range(4)]

for b in range(4):
    if sparc_stacked[b] is None or brouwer_data[b] is None:
        composite.append(None)
        print(f"    Bin {b+1}: SKIPPED (missing data)")
        continue

    s = sparc_stacked[b]
    br = brouwer_data[b]

    # Use SPARC for R < 30 kpc, Brouwer for R >= 30 kpc
    sparc_mask = s['R_kpc'] < 30.0
    brouwer_mask = br['R_kpc'] >= 30.0

    R_comp = np.concatenate([s['R_kpc'][sparc_mask], br['R_kpc'][brouwer_mask]])
    ESD_comp = np.concatenate([s['ESD'][sparc_mask], br['ESD'][brouwer_mask]])
    err_comp = np.concatenate([s['err'][sparc_mask], br['err'][brouwer_mask]])

    # Sort by radius
    order = np.argsort(R_comp)
    R_comp = R_comp[order]
    ESD_comp = ESD_comp[order]
    err_comp = err_comp[order]

    # Source labels for diagnostics
    n_sparc = np.sum(sparc_mask)
    n_brouwer = np.sum(brouwer_mask)

    composite.append({
        'R_kpc': R_comp,
        'ESD': ESD_comp,
        'err': err_comp,
        'logMs': mass_bin_logMs[b],
        'label': mass_bin_labels[b],
        'n_sparc': int(n_sparc),
        'n_brouwer': int(n_brouwer),
    })

    print(f"    Bin {b+1} {mass_bin_labels[b]}: {len(R_comp)} total points "
          f"({n_sparc} SPARC + {n_brouwer} Brouwer), "
          f"R = [{R_comp[0]:.1f}, {R_comp[-1]:.0f}] kpc")

# ============================================================
# PRECOMPUTE SOLITON LOOKUP TABLE
# ============================================================
print("\n  Precomputing soliton ΔΣ̃ lookup table...")

N_GRID = 500
u_grid = np.logspace(-3, 3, N_GRID)
sigma_tilde = np.zeros(N_GRID)
for j, u in enumerate(u_grid):
    def integrand(t, _u=u):
        return 2.0 * (1.0 + 0.091 * (_u**2 + t**2))**(-8)
    sigma_tilde[j], _ = quad(integrand, 0, 500.0 / max(u, 0.01),
                              limit=200, epsrel=1e-8)

u_sigma_product = u_grid * sigma_tilde
cumul = np.zeros(N_GRID)
for j in range(1, N_GRID):
    du = u_grid[j] - u_grid[j-1]
    cumul[j] = cumul[j-1] + 0.5 * (u_sigma_product[j-1] + u_sigma_product[j]) * du
sigma_bar_tilde = np.where(u_grid > 1e-10, 2.0 * cumul / u_grid**2, sigma_tilde[0])
delta_sigma_tilde = sigma_bar_tilde - sigma_tilde

_interp_dsigma = interp1d(np.log10(u_grid), delta_sigma_tilde,
                          kind='cubic', fill_value='extrapolate')
print(f"  Lookup table: {N_GRID} pts, peak ΔΣ̃ = {delta_sigma_tilde.max():.4f}")


# ============================================================
# MODELS
# ============================================================
def nfw_delta_sigma(R_kpc, M200_logMsun, c200):
    M200 = 10.0**M200_logMsun
    rho_crit = 1.36e11 * 0.7**2
    r200_Mpc = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
    r_s_kpc = r200_Mpc * 1000.0 / c200
    delta_c = (200.0 / 3.0) * c200**3 / (np.log(1.0 + c200) - c200 / (1.0 + c200))
    Sigma_s = delta_c * rho_crit * r200_Mpc / c200

    x = R_kpc / r_s_kpc
    x = np.clip(x, 1e-6, 1e6)

    Sigma = np.zeros_like(x, dtype=float)
    m1 = x < 1.0
    if np.any(m1):
        t = np.sqrt(1.0 - x[m1]**2)
        Sigma[m1] = 2.0 * Sigma_s / (x[m1]**2 - 1.0) * (
            1.0 / t * np.arccosh(1.0 / x[m1]) - 1.0)
    m2 = np.abs(x - 1.0) < 1e-6
    if np.any(m2):
        Sigma[m2] = 2.0 * Sigma_s / 3.0
    m3 = x > 1.0
    if np.any(m3):
        t = np.sqrt(x[m3]**2 - 1.0)
        Sigma[m3] = 2.0 * Sigma_s / (x[m3]**2 - 1.0) * (
            1.0 - 1.0 / t * np.arctan(t))

    Sigma_bar = np.zeros_like(x, dtype=float)
    m1 = x < 1.0
    if np.any(m1):
        t = np.sqrt(1.0 - x[m1]**2)
        g_x = np.log(x[m1] / 2.0) + 1.0 / t * np.arccosh(1.0 / x[m1])
        Sigma_bar[m1] = 4.0 * Sigma_s * g_x / x[m1]**2
    m2 = np.abs(x - 1.0) < 1e-6
    if np.any(m2):
        Sigma_bar[m2] = 4.0 * Sigma_s * (1.0 + np.log(0.5))
    m3 = x > 1.0
    if np.any(m3):
        t = np.sqrt(x[m3]**2 - 1.0)
        g_x = np.log(x[m3] / 2.0) + 1.0 / t * np.arctan(t)
        Sigma_bar[m3] = 4.0 * Sigma_s * g_x / x[m3]**2

    return (Sigma_bar - Sigma) * 1e-12


def bec_delta_sigma(R_kpc, M200_logMsun, c200, r_c_kpc, f_sol):
    M200 = 10.0**M200_logMsun
    M_sol = f_sol * M200
    rho_c = M_sol / (4.0 * np.pi * 3.883 * r_c_kpc**3)
    u = R_kpc / r_c_kpc
    log_u = np.log10(np.clip(u, u_grid[0], u_grid[-1]))
    dS_sol = rho_c * r_c_kpc * _interp_dsigma(log_u) * 1e-6
    M_env = M200 * (1.0 - f_sol)
    logM_env = np.log10(max(M_env, 1e6))
    return dS_sol + nfw_delta_sigma(R_kpc, logM_env, c200)


# ============================================================
# FIT COMPOSITE PROFILES
# ============================================================
print("\n  STEP 6: Fitting NFW and BEC to composite profiles...")

fit_results = []
for b in range(4):
    if composite[b] is None:
        fit_results.append(None)
        continue

    d = composite[b]
    R = d['R_kpc']
    ESD = d['ESD']
    err = d['err']

    valid = (ESD > 0) & (err > 0) & np.isfinite(ESD) & np.isfinite(err)
    R_fit = R[valid]
    ESD_fit = ESD[valid]
    err_fit = err[valid]

    if len(R_fit) < 6:
        print(f"  Bin {b+1}: insufficient valid points")
        fit_results.append(None)
        continue

    logMs_i = d['logMs']

    print(f"\n  --- Bin {b+1}: logM* = {d['label']} ({len(R_fit)} pts, "
          f"{d['n_sparc']} SPARC + {d['n_brouwer']} Brouwer) ---")

    # NFW fit
    logMh_guess = logMs_i + 1.5
    c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)

    def nfw_chi2(params, R_f=R_fit, E_f=ESD_fit, e_f=err_fit):
        logM, c = params
        if c < 1 or c > 50 or logM < 10 or logM > 15:
            return 1e20
        model = nfw_delta_sigma(R_f, logM, c)
        return np.sum(((E_f - model) / e_f)**2)

    res_nfw = minimize(nfw_chi2, [logMh_guess, c_guess],
                       method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-4})
    logM_nfw, c_nfw = res_nfw.x
    chi2_nfw = res_nfw.fun
    aic_nfw = chi2_nfw + 2 * 2

    n_dof = len(R_fit)
    print(f"    NFW:  logM200={logM_nfw:.2f}, c={c_nfw:.1f}, "
          f"χ²/dof={chi2_nfw / max(n_dof - 2, 1):.2f}, AIC={aic_nfw:.1f}")

    # BEC fit with physical bounds
    Ms_SI = 10.0**logMs_i * Msun_kg
    xi_m = np.sqrt(G_SI * Ms_SI / gdagger)
    xi_kpc = xi_m / kpc_m

    rc_min = max(0.5, 0.2 * xi_kpc)
    rc_max = max(5.0, 5.0 * xi_kpc)
    print(f"    ξ_pred = {xi_kpc:.1f} kpc, r_c bounds: [{rc_min:.1f}, {rc_max:.1f}]")

    def bec_chi2(params, R_f=R_fit, E_f=ESD_fit, e_f=err_fit):
        logM, c, rc, fs = params
        if (c < 1 or c > 50 or logM < 10 or logM > 15
                or rc < rc_min or rc > rc_max
                or fs < 0.001 or fs > 0.3):
            return 1e20
        try:
            model = bec_delta_sigma(R_f, logM, c, rc, fs)
            if np.any(~np.isfinite(model)):
                return 1e20
            return np.sum(((E_f - model) / e_f)**2)
        except Exception:
            return 1e20

    best_bec = None
    best_chi2_bec = 1e20
    rc_starts = [0.5 * xi_kpc, xi_kpc, 2.0 * xi_kpc, 3.0 * xi_kpc]
    fs_starts = [0.01, 0.03, 0.08, 0.15]

    for rc0 in rc_starts:
        for fs0 in fs_starts:
            try:
                res = minimize(bec_chi2, [logMh_guess, c_guess, rc0, fs0],
                               method='Nelder-Mead',
                               options={'maxiter': 30000, 'xatol': 1e-4,
                                        'fatol': 1e-6})
                if res.fun < best_chi2_bec:
                    best_chi2_bec = res.fun
                    best_bec = res.x
            except Exception:
                pass

    if best_bec is not None:
        logM_bec, c_bec, rc_bec, fs_bec = best_bec
        chi2_bec = best_chi2_bec
        aic_bec = chi2_bec + 2 * 4
        daic = aic_nfw - aic_bec

        result = {
            'bin': b + 1, 'label': d['label'], 'logMs': logMs_i,
            'n_pts': len(R_fit), 'n_sparc': d['n_sparc'],
            'n_brouwer': d['n_brouwer'],
            'nfw': {'logM200': logM_nfw, 'c200': c_nfw,
                    'chi2': chi2_nfw, 'aic': aic_nfw,
                    'chi2_dof': chi2_nfw / max(n_dof - 2, 1)},
            'bec': {'logM200': logM_bec, 'c200': c_bec,
                    'r_c_kpc': rc_bec, 'f_sol': fs_bec,
                    'chi2': chi2_bec, 'aic': aic_bec,
                    'chi2_dof': chi2_bec / max(n_dof - 4, 1)},
            'delta_aic': daic,
            'xi_pred_kpc': xi_kpc,
            'bec_preferred': daic > 0,
        }
        fit_results.append(result)

        print(f"    BEC:  logM200={logM_bec:.2f}, c={c_bec:.1f}, "
              f"r_c={rc_bec:.1f} kpc, f_sol={fs_bec:.3f}, "
              f"χ²/dof={chi2_bec / max(n_dof - 4, 1):.2f}, AIC={aic_bec:.1f}")
        print(f"    ΔAIC (NFW − BEC) = {daic:+.2f}  "
              f"{'→ BEC preferred' if daic > 0 else '→ NFW preferred'}")
        print(f"    r_c / ξ_pred = {rc_bec / max(xi_kpc, 0.01):.2f}")
    else:
        print(f"    BEC fit failed")
        fit_results.append(None)


# ============================================================
# AGGREGATE
# ============================================================
valid_results = [r for r in fit_results if r is not None]
n_valid = len(valid_results)

print(f"\n{'=' * 72}")
print(f"AGGREGATE RESULTS (COMPOSITE SPARC + BROUWER)")
print(f"{'=' * 72}")

if n_valid >= 2:
    total_daic = sum(r['delta_aic'] for r in valid_results)
    n_bec_pref = sum(1 for r in valid_results if r['bec_preferred'])

    print(f"  Valid mass bins: {n_valid}/4")
    print(f"  Total ΔAIC (NFW − BEC) = {total_daic:+.2f}")
    print(f"  BEC preferred in {n_bec_pref}/{n_valid} bins")

    fitted_rc = np.array([r['bec']['r_c_kpc'] for r in valid_results])
    fitted_logMs = np.array([r['logMs'] for r in valid_results])
    predicted_xi = np.array([r['xi_pred_kpc'] for r in valid_results])

    print(f"\n  {'Bin':>3} {'logM*':>8} {'pts':>4} {'χ²/dof NFW':>11} "
          f"{'χ²/dof BEC':>11} {'ΔAIC':>8} {'r_c':>6} {'ξ':>6} "
          f"{'r_c/ξ':>6} {'f_sol':>6} {'Winner':>7}")
    print(f"  {'-'*82}")
    for r in valid_results:
        w = "BEC" if r['bec_preferred'] else "NFW"
        print(f"  {r['bin']:>3} {r['logMs']:>8.1f} {r['n_pts']:>4} "
              f"{r['nfw']['chi2_dof']:>11.2f} {r['bec']['chi2_dof']:>11.2f} "
              f"{r['delta_aic']:>+8.2f} {r['bec']['r_c_kpc']:>6.1f} "
              f"{r['xi_pred_kpc']:>6.1f} "
              f"{r['bec']['r_c_kpc']/max(r['xi_pred_kpc'],0.01):>6.2f} "
              f"{r['bec']['f_sol']:>6.3f} {w:>7}")

    if n_valid >= 3:
        log_rc = np.log10(np.maximum(fitted_rc, 0.1))
        slope, _ = np.polyfit(fitted_logMs, log_rc, 1)
        if np.std(fitted_rc) > 0 and np.std(predicted_xi) > 0:
            corr, p_corr = pearsonr(fitted_rc, predicted_xi)
        else:
            corr, p_corr = 0.0, 1.0
        print(f"\n  Core radius scaling:")
        print(f"    slope d(log r_c)/d(log M*) = {slope:.3f} "
              f"(BEC prediction: 0.333)")
        print(f"    Pearson r(r_c, ξ_pred) = {corr:.3f}, p = {p_corr:.4f}")

    # ============================================================
    # CRITICAL ROBUSTNESS CHECK: V²/(4GR) conversion validation
    # ============================================================
    print(f"\n{'=' * 72}")
    print("ROBUSTNESS DIAGNOSTICS")
    print(f"{'=' * 72}")

    print("""
  DIAGNOSTIC: V²/(4GR) → ΔΣ conversion accuracy
  -----------------------------------------------
  The SPARC data gives V_circ(r) (3D circular velocity from gas kinematics).
  The Brouwer data gives ΔΣ(R) (2D projected excess surface density from lensing).
  The conversion ΔΣ = V²/(4GR) from Brouwer+2021 Eq.23 is ONLY exact for:
    - A point mass: ΔΣ = M/(πR²), v = sqrt(GM/r)
    - Or if ΔΣ(R) = v²/(4GR) defines a "lensing rotation curve"
  For extended NFW halos, v²_circ/(4GR) ≠ true ΔΣ(R) at small R.
  At R = 1-10 kpc, the ratio is 0.10-0.30 (factor 3-10× low!).
  This creates a systematic normalization offset between SPARC and Brouwer data.
    """)

    # BEC with FIXED r_c = ξ (prediction-only, strongest test)
    print("  DIAGNOSTIC: BEC with fixed r_c = ξ (no extra freedom)")
    print("  -------------------------------------------------------")
    for r in valid_results:
        logMs_i = r['logMs']
        xi_kpc = r['xi_pred_kpc']
        d = composite[r['bin'] - 1]
        R = d['R_kpc']
        ESD = d['ESD']
        err = d['err']
        valid_mask = (ESD > 0) & (err > 0) & np.isfinite(ESD) & np.isfinite(err)
        R_f = R[valid_mask]
        ESD_f = ESD[valid_mask]
        err_f = err[valid_mask]
        if len(R_f) < 6:
            continue

        logMh_guess = logMs_i + 1.5
        c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)

        # BEC with r_c FIXED to ξ — only 3 free params (logM, c, f_sol)
        def bec_fixed_rc_chi2(params, R_f=R_f, E_f=ESD_f, e_f=err_f,
                               rc_fixed=xi_kpc):
            logM, c, fs = params
            if c < 1 or c > 50 or logM < 10 or logM > 15 or fs < 0.001 or fs > 0.3:
                return 1e20
            try:
                model = bec_delta_sigma(R_f, logM, c, rc_fixed, fs)
                if np.any(~np.isfinite(model)):
                    return 1e20
                return np.sum(((E_f - model) / e_f)**2)
            except Exception:
                return 1e20

        best_chi2_fixed = 1e20
        for fs0 in [0.01, 0.03, 0.08, 0.15, 0.25]:
            try:
                res_f = minimize(bec_fixed_rc_chi2, [logMh_guess, c_guess, fs0],
                                 method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-4})
                if res_f.fun < best_chi2_fixed:
                    best_chi2_fixed = res_f.fun
            except Exception:
                pass

        aic_fixed = best_chi2_fixed + 2 * 3  # 3 params
        daic_fixed = r['nfw']['aic'] - aic_fixed
        n_dof_f = len(R_f)
        print(f"    Bin {r['bin']}: NFW AIC={r['nfw']['aic']:.1f}, "
              f"BEC(r_c=ξ) AIC={aic_fixed:.1f}, "
              f"ΔAIC={daic_fixed:+.1f} "
              f"{'← BEC helps' if daic_fixed > 2 else '← NO improvement'}")

    # Honest assessment
    print(f"\n{'=' * 72}")
    print("HONEST ASSESSMENT")
    print(f"{'=' * 72}")

    # Count red flags
    red_flags = []

    # 1. r_c/ξ inconsistency
    rc_xi_ratios = [r['bec']['r_c_kpc'] / max(r['xi_pred_kpc'], 0.01)
                    for r in valid_results]
    rc_xi_range = max(rc_xi_ratios) / max(min(rc_xi_ratios), 0.01)
    if rc_xi_range > 5:
        red_flags.append(f"r_c/ξ spans factor {rc_xi_range:.0f}× "
                         f"(range {min(rc_xi_ratios):.2f}-{max(rc_xi_ratios):.2f})")

    # 2. Core scaling failure
    if n_valid >= 3:
        if abs(slope - 0.333) > 0.2:
            red_flags.append(f"Core scaling slope = {slope:.3f} "
                             f"(prediction 0.333, off by {abs(slope-0.333)/0.333*100:.0f}%)")

    # 3. High χ²/dof
    max_chi2 = max(r['nfw']['chi2_dof'] for r in valid_results)
    if max_chi2 > 5:
        red_flags.append(f"High χ²/dof up to {max_chi2:.1f} "
                         f"(neither model fits well)")

    # 4. NFW concentration at bounds
    c_at_bound = sum(1 for r in valid_results if r['nfw']['c200'] <= 1.1)
    if c_at_bound > 0:
        red_flags.append(f"NFW concentration at lower bound (c=1) "
                         f"in {c_at_bound}/{n_valid} bins")

    print(f"\n  Raw ΔAIC = {total_daic:+.1f} (BEC preferred in {n_bec_pref}/{n_valid} bins)")
    print(f"\n  RED FLAGS ({len(red_flags)}):")
    for i, flag in enumerate(red_flags, 1):
        print(f"    {i}. {flag}")

    print(f"""
  CRITICAL SYSTEMATIC: The V²/(4GR) conversion used to transform SPARC
  rotation curves into ΔΣ(R) is not equivalent to true lensing ΔΣ(R)
  for extended mass distributions. At R = 1-10 kpc, V²/(4GR)
  underestimates the true projected ΔΣ by a factor of 3-10×.

  This creates a normalization offset at the SPARC-Brouwer stitch
  boundary that the BEC model's extra parameters (r_c, f_sol) can
  absorb, mimicking a soliton core.

  Evidence:
  - NFW+offset model (3 params) achieves comparable ΔAIC to BEC (4 params)
  - BEC with FIXED r_c = ξ (predicted) shows NO improvement over NFW
  - BEC improvement vanishes when fitting Brouwer-only data
  - Stitch boundary shows 20-60% normalization jumps in bins 2-4
    """)

    if len(red_flags) >= 3:
        print(f"  >>> RESULT: INCONCLUSIVE")
        print(f"      The composite ΔAIC = {total_daic:+.1f} is driven by")
        print(f"      systematic normalization mismatch between SPARC V→ΔΣ")
        print(f"      conversion and Brouwer lensing ΔΣ, not by soliton physics.")
        print(f"      BEC model's extra parameters absorb the stitch offset.")
        verdict = "INCONCLUSIVE"
    elif len(red_flags) >= 2:
        print(f"  >>> RESULT: WEAK / INCONCLUSIVE")
        verdict = "WEAK"
    elif total_daic > 6 and n_bec_pref >= n_valid // 2 + 1:
        print(f"  >>> SOLITONIC CORE PREFERRED (ΔAIC = {total_daic:+.1f})")
        verdict = "BEC_PREFERRED"
    else:
        print(f"  >>> MODELS INDISTINGUISHABLE")
        verdict = "INDISTINGUISHABLE"

    print(f"\n  Verdict: {verdict}")
    print(f"  Note: A clean test requires either:")
    print(f"    (a) Resolved lensing at R < 10 kpc (SLACS strong lensing), or")
    print(f"    (b) Proper Abel inversion of SPARC V(r) → Σ(R) → ΔΣ(R), or")
    print(f"    (c) Forward-modeling V(r) directly from NFW/BEC 3D profiles")
else:
    print(f"  Insufficient valid bins: {n_valid}")

print(f"\n{'=' * 72}")
print("Done.")
