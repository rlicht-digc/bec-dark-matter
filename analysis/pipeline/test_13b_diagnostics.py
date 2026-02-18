#!/usr/bin/env python3
"""
Test 13b DIAGNOSTICS: Honest analysis of composite SPARC+Brouwer results.

Checks for:
1. Stitch boundary systematics — is there a jump at 30 kpc?
2. Split-sample test — does ΔAIC come from inner (SPARC) or outer (Brouwer) data?
3. BEC model freedom — do extra params just absorb the stitch discontinuity?
4. Core radius consistency — are r_c/ξ ratios physically sensible?
5. Cross-validation — leave-one-out stability
6. Normalization mismatch — V²/(4GR) vs lensing ESD calibration
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

# Physical constants
G_conv = 4.302e-3  # pc (km/s)^2 / Msun
G_SI = 6.674e-11
Msun_kg = 1.989e30
kpc_m = 3.086e19
gdagger = 1.2e-10

print("=" * 72)
print("TEST 13b DIAGNOSTICS: Honest analysis of composite results")
print("=" * 72)

# ============================================================
# LOAD DATA (same as composite script)
# ============================================================
print("\n[1] Loading data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

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

with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()
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
        name = line[0:11].strip()
        if not name:
            continue
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        T = int(parts[0])
        D = float(parts[1])
        Inc = float(parts[4])
        L36 = float(parts[6])
        Q = int(parts[16])
        sparc_props[name] = {
            'D': D, 'Inc': Inc, 'T': T, 'Q': Q,
            'L36_1e9': L36,
            'logMs': np.log10(max(0.5 * L36 * 1e9, 1e6))
        }
    except (ValueError, IndexError):
        continue

print(f"  Loaded {len(galaxies)} rotation curves, {len(sparc_props)} properties")

# ============================================================
# BIN AND CONVERT SPARC
# ============================================================
mass_bin_edges = [8.5, 10.3, 10.6, 10.8, 11.0]
mass_bin_labels = [f"[{mass_bin_edges[i]:.1f}, {mass_bin_edges[i+1]:.1f}]"
                   for i in range(4)]

sparc_binned = [[] for _ in range(4)]
n_used = 0
for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2:
        continue
    if prop['Inc'] < 30 or prop['Inc'] > 85:
        continue
    if len(gdata['R']) < 5:
        continue
    logMs = prop['logMs']
    bin_idx = None
    for b in range(4):
        if mass_bin_edges[b] <= logMs < mass_bin_edges[b + 1]:
            bin_idx = b
            break
    if bin_idx is None:
        continue
    R = gdata['R']
    Vobs = gdata['Vobs']
    eVobs = gdata['eVobs']
    valid = (R > 0) & (Vobs > 0) & np.isfinite(Vobs)
    if np.sum(valid) < 3:
        continue
    R_v = R[valid]
    V_v = Vobs[valid]
    eV_v = eVobs[valid]
    R_pc = R_v * 1000.0
    delta_sigma = V_v**2 / (4.0 * G_conv * R_pc)
    delta_sigma_err = delta_sigma * 2.0 * eV_v / np.maximum(V_v, 1.0)
    sparc_binned[bin_idx].append({
        'name': name, 'logMs': logMs,
        'R_kpc': R_v, 'delta_sigma': delta_sigma,
        'delta_sigma_err': delta_sigma_err,
    })
    n_used += 1

print(f"  Used {n_used} SPARC galaxies in 4 bins")

# Stack SPARC profiles
sparc_radial_edges = np.logspace(np.log10(0.5), np.log10(50), 12)
sparc_radial_centers = np.sqrt(sparc_radial_edges[:-1] * sparc_radial_edges[1:])

sparc_stacked = []
for b in range(4):
    if len(sparc_binned[b]) < 3:
        sparc_stacked.append(None)
        continue
    R_stack, ESD_stack, err_stack = [], [], []
    for j in range(len(sparc_radial_centers)):
        lo, hi = sparc_radial_edges[j], sparc_radial_edges[j + 1]
        vals, errs = [], []
        for gal in sparc_binned[b]:
            mask = (gal['R_kpc'] >= lo) & (gal['R_kpc'] < hi)
            if np.sum(mask) > 0:
                w = 1.0 / np.maximum(gal['delta_sigma_err'][mask], 1e-10)**2
                wmean = np.average(gal['delta_sigma'][mask], weights=w)
                werr = 1.0 / np.sqrt(np.sum(w))
                vals.append(wmean)
                errs.append(werr)
        if len(vals) >= 3:
            vals = np.array(vals)
            errs = np.array(errs)
            w = 1.0 / errs**2
            stacked_val = np.average(vals, weights=w)
            stacked_err = 1.0 / np.sqrt(np.sum(w))
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
    else:
        sparc_stacked.append(None)

# Load Brouwer
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

# Build composite
mass_bin_logMs = [0.5 * (mass_bin_edges[i] + mass_bin_edges[i + 1])
                  for i in range(4)]

composite = []
for b in range(4):
    if sparc_stacked[b] is None or brouwer_data[b] is None:
        composite.append(None)
        continue
    s = sparc_stacked[b]
    br = brouwer_data[b]
    sparc_mask = s['R_kpc'] < 30.0
    brouwer_mask = br['R_kpc'] >= 30.0
    R_comp = np.concatenate([s['R_kpc'][sparc_mask], br['R_kpc'][brouwer_mask]])
    ESD_comp = np.concatenate([s['ESD'][sparc_mask], br['ESD'][brouwer_mask]])
    err_comp = np.concatenate([s['err'][sparc_mask], br['err'][brouwer_mask]])
    order = np.argsort(R_comp)
    R_comp = R_comp[order]
    ESD_comp = ESD_comp[order]
    err_comp = err_comp[order]
    n_sparc = int(np.sum(sparc_mask))
    n_brouwer = int(np.sum(brouwer_mask))
    # Track which source each point comes from
    source = np.concatenate([
        np.ones(n_sparc, dtype=int),      # 1 = SPARC
        2 * np.ones(n_brouwer, dtype=int)  # 2 = Brouwer
    ])
    source = source[order]
    composite.append({
        'R_kpc': R_comp, 'ESD': ESD_comp, 'err': err_comp,
        'logMs': mass_bin_logMs[b], 'label': mass_bin_labels[b],
        'n_sparc': n_sparc, 'n_brouwer': n_brouwer,
        'source': source,
    })

# Precompute soliton lookup
print("\n[2] Precomputing soliton lookup...")
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
print(f"  Done. Peak ΔΣ̃ = {delta_sigma_tilde.max():.4f}")


# Models
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
# DIAGNOSTIC 1: Stitch Boundary Analysis
# ============================================================
print("\n" + "=" * 72)
print("DIAGNOSTIC 1: STITCH BOUNDARY ANALYSIS")
print("=" * 72)
print("  Checking for systematic discontinuity at R = 30 kpc...")

for b in range(4):
    if composite[b] is None:
        continue
    d = composite[b]
    R = d['R_kpc']
    ESD = d['ESD']
    source = d['source']

    # Find last SPARC point and first Brouwer point
    sparc_pts = R[source == 1]
    sparc_esd = ESD[source == 1]
    brouwer_pts = R[source == 2]
    brouwer_esd = ESD[source == 2]

    if len(sparc_pts) == 0 or len(brouwer_pts) == 0:
        continue

    # Last SPARC point near boundary
    last_sparc_R = sparc_pts[-1]
    last_sparc_ESD = sparc_esd[-1]
    # First Brouwer point
    first_brouwer_R = brouwer_pts[0]
    first_brouwer_ESD = brouwer_esd[0]

    # Expected ESD at Brouwer start assuming power-law from last 3 SPARC points
    if len(sparc_pts) >= 3:
        log_R_sparc = np.log10(sparc_pts[-3:])
        log_ESD_sparc = np.log10(np.maximum(sparc_esd[-3:], 1e-10))
        slope_sparc = np.polyfit(log_R_sparc, log_ESD_sparc, 1)[0]
        extrap_log_ESD = np.polyval(np.polyfit(log_R_sparc, log_ESD_sparc, 1),
                                     np.log10(first_brouwer_R))
        extrap_ESD = 10.0**extrap_log_ESD
        ratio = first_brouwer_ESD / extrap_ESD

        print(f"\n  Bin {b+1} ({d['label']}):")
        print(f"    Last SPARC:    R={last_sparc_R:.1f} kpc, "
              f"ΔΣ={last_sparc_ESD:.2f} M☉/pc²")
        print(f"    First Brouwer: R={first_brouwer_R:.1f} kpc, "
              f"ΔΣ={first_brouwer_ESD:.2f} M☉/pc²")
        print(f"    SPARC power-law slope: {slope_sparc:.2f}")
        print(f"    Extrapolated ΔΣ at R={first_brouwer_R:.0f} kpc: "
              f"{extrap_ESD:.2f} M☉/pc²")
        print(f"    Ratio (Brouwer / extrapolated SPARC): {ratio:.2f}")
        if abs(ratio - 1.0) > 0.5:
            print(f"    ⚠ SIGNIFICANT MISMATCH: factor {ratio:.1f}× offset!")
        elif abs(ratio - 1.0) > 0.2:
            print(f"    ⚡ Moderate mismatch: {(ratio-1)*100:.0f}% offset")
        else:
            print(f"    ✓ Consistent within ~20%")


# ============================================================
# DIAGNOSTIC 2: Split-Sample Test
# ============================================================
print("\n" + "=" * 72)
print("DIAGNOSTIC 2: SPLIT-SAMPLE TEST")
print("=" * 72)
print("  Fitting SPARC-only and Brouwer-only separately to isolate")
print("  where BEC preference originates...")

for b in range(4):
    if composite[b] is None:
        continue
    d = composite[b]
    R = d['R_kpc']
    ESD = d['ESD']
    err = d['err']
    source = d['source']
    logMs_i = d['logMs']

    Ms_SI = 10.0**logMs_i * Msun_kg
    xi_m = np.sqrt(G_SI * Ms_SI / gdagger)
    xi_kpc = xi_m / kpc_m
    rc_min = max(0.5, 0.2 * xi_kpc)
    rc_max = max(5.0, 5.0 * xi_kpc)

    logMh_guess = logMs_i + 1.5
    c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)

    for region_name, region_mask in [("SPARC-only (R<30 kpc)", source == 1),
                                       ("Brouwer-only (R≥30 kpc)", source == 2)]:
        R_reg = R[region_mask]
        ESD_reg = ESD[region_mask]
        err_reg = err[region_mask]

        valid = (ESD_reg > 0) & (err_reg > 0) & np.isfinite(ESD_reg)
        R_f = R_reg[valid]
        ESD_f = ESD_reg[valid]
        err_f = err_reg[valid]

        if len(R_f) < 4:
            continue

        # NFW
        def nfw_chi2(params, R_f=R_f, E_f=ESD_f, e_f=err_f):
            logM, c = params
            if c < 1 or c > 50 or logM < 10 or logM > 15:
                return 1e20
            model = nfw_delta_sigma(R_f, logM, c)
            return np.sum(((E_f - model) / e_f)**2)

        res_nfw = minimize(nfw_chi2, [logMh_guess, c_guess],
                           method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-4})
        chi2_nfw = res_nfw.fun
        aic_nfw = chi2_nfw + 2 * 2

        # BEC
        def bec_chi2(params, R_f=R_f, E_f=ESD_f, e_f=err_f):
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

        best_chi2_bec = 1e20
        rc_starts = [0.5 * xi_kpc, xi_kpc, 2.0 * xi_kpc]
        fs_starts = [0.01, 0.05, 0.15]
        for rc0 in rc_starts:
            for fs0 in fs_starts:
                try:
                    res = minimize(bec_chi2, [logMh_guess, c_guess, rc0, fs0],
                                   method='Nelder-Mead',
                                   options={'maxiter': 20000, 'xatol': 1e-4})
                    if res.fun < best_chi2_bec:
                        best_chi2_bec = res.fun
                except Exception:
                    pass

        aic_bec = best_chi2_bec + 2 * 4
        daic = aic_nfw - aic_bec
        n_dof = len(R_f)

        print(f"\n  Bin {b+1} {region_name} ({n_dof} pts):")
        print(f"    NFW χ²/dof = {chi2_nfw / max(n_dof-2,1):.2f}, "
              f"BEC χ²/dof = {best_chi2_bec / max(n_dof-4,1):.2f}")
        print(f"    ΔAIC = {daic:+.2f}  "
              f"{'→ BEC preferred' if daic > 0 else '→ NFW preferred'}")


# ============================================================
# DIAGNOSTIC 3: NFW with free normalization offset
# ============================================================
print("\n" + "=" * 72)
print("DIAGNOSTIC 3: NFW + STITCH OFFSET")
print("=" * 72)
print("  Adding a free normalization offset at 30 kpc to NFW model.")
print("  If this matches BEC's ΔAIC, the 'soliton' is just absorbing")
print("  the systematic offset between datasets.")

for b in range(4):
    if composite[b] is None:
        continue
    d = composite[b]
    R = d['R_kpc']
    ESD = d['ESD']
    err = d['err']
    source = d['source']
    logMs_i = d['logMs']

    valid = (ESD > 0) & (err > 0) & np.isfinite(ESD) & np.isfinite(err)
    R_f = R[valid]
    ESD_f = ESD[valid]
    err_f = err[valid]
    source_f = source[valid]

    if len(R_f) < 6:
        continue

    logMh_guess = logMs_i + 1.5
    c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)

    # NFW + multiplicative offset for SPARC region
    # Model: ΔΣ_model(R) = A_sparc × NFW(R) for SPARC points,
    #                       NFW(R)          for Brouwer points
    # This tests if the BEC improvement comes from accommodating
    # a normalization difference between SPARC V→ΔΣ and Brouwer lensing ΔΣ

    def nfw_offset_chi2(params, R_f=R_f, E_f=ESD_f, e_f=err_f, s_f=source_f):
        logM, c, A_sparc = params
        if c < 1 or c > 50 or logM < 10 or logM > 15:
            return 1e20
        if A_sparc < 0.1 or A_sparc > 10.0:
            return 1e20
        model = nfw_delta_sigma(R_f, logM, c)
        # Apply offset to SPARC points
        scale = np.where(s_f == 1, A_sparc, 1.0)
        model = model * scale
        return np.sum(((E_f - model) / e_f)**2)

    best_offset = None
    best_chi2_offset = 1e20
    for A0 in [0.5, 1.0, 1.5, 2.0, 3.0]:
        try:
            res = minimize(nfw_offset_chi2, [logMh_guess, c_guess, A0],
                           method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-4})
            if res.fun < best_chi2_offset:
                best_chi2_offset = res.fun
                best_offset = res.x
        except Exception:
            pass

    if best_offset is not None:
        logM_off, c_off, A_off = best_offset
        aic_offset = best_chi2_offset + 2 * 3  # 3 params

        # Compare to pure NFW
        def nfw_chi2_pure(params, R_f=R_f, E_f=ESD_f, e_f=err_f):
            logM, c = params
            if c < 1 or c > 50 or logM < 10 or logM > 15:
                return 1e20
            model = nfw_delta_sigma(R_f, logM, c)
            return np.sum(((E_f - model) / e_f)**2)

        res_nfw = minimize(nfw_chi2_pure, [logMh_guess, c_guess],
                           method='Nelder-Mead',
                           options={'maxiter': 10000})
        chi2_nfw = res_nfw.fun
        aic_nfw = chi2_nfw + 2 * 2

        daic_offset_vs_nfw = aic_nfw - aic_offset

        print(f"\n  Bin {b+1} ({d['label']}):")
        print(f"    Pure NFW: χ² = {chi2_nfw:.1f}, AIC = {aic_nfw:.1f}")
        print(f"    NFW+offset: χ² = {best_chi2_offset:.1f}, AIC = {aic_offset:.1f}, "
              f"A_sparc = {A_off:.3f}")
        print(f"    ΔAIC (NFW − NFW+offset) = {daic_offset_vs_nfw:+.1f}")
        if A_off < 0.5 or A_off > 2.0:
            print(f"    ⚠ A_sparc = {A_off:.2f}: SPARC→ΔΣ conversion off by "
                  f"{abs(A_off-1)*100:.0f}%!")
        print(f"    If this ΔAIC ≈ BEC ΔAIC, the soliton is just an offset proxy!")


# ============================================================
# DIAGNOSTIC 4: V²/(4GR) conversion sanity check
# ============================================================
print("\n" + "=" * 72)
print("DIAGNOSTIC 4: V²/(4GR) CONVERSION VALIDATION")
print("=" * 72)
print("  Brouwer+2021 Eq.23 assumes: v_circ = sqrt(4G × ΔΣ × R)")
print("  This is EXACT only for ΔΣ of a point mass: ΔΣ = M/(πR²)")
print("  For an NFW profile, this is approximate and introduces")
print("  systematic bias depending on the radial scale.")

# Show how much the conversion deviates from true ΔΣ for NFW
R_test = np.logspace(np.log10(1), np.log10(50), 20)  # 1-50 kpc
for logMh in [11.5, 12.0, 12.5]:
    c_test = 10.0 * (10.0**(logMh) / 1e12)**(-0.1)
    true_delta_sigma = nfw_delta_sigma(R_test, logMh, c_test)

    # The "v_circ" from ΔΣ is: v = sqrt(4G ΔΣ R)
    # Converting back: ΔΣ_recovered = v²/(4GR)
    # By construction this is identical! But the issue is different:
    # SPARC measures v_circ from gas/stellar kinematics (REAL circular velocity)
    # Lensing measures ΔΣ (PROJECTED surface mass density)
    # These are NOT the same quantity for an extended mass distribution.

    # For NFW, v_circ²(r) = G M(<r) / r (3D)
    # But ΔΣ(R) = Σ̄(<R) - Σ(R) (2D projected)
    # The relation v² = 4G ΔΣ R is only approximate

    # Let's compute v_circ(R) for NFW properly (3D enclosed mass)
    M200 = 10.0**logMh
    rho_crit = 1.36e11 * 0.7**2
    r200_Mpc = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
    r200_kpc = r200_Mpc * 1000.0
    r_s = r200_kpc / c_test
    delta_c = (200.0/3.0) * c_test**3 / (np.log(1+c_test) - c_test/(1+c_test))
    rho_s = delta_c * rho_crit  # Msun/Mpc³

    # M_NFW(<r) = 4π ρ_s r_s³ [ln(1+r/r_s) - (r/r_s)/(1+r/r_s)]
    x = R_test / r_s
    M_enc = 4.0 * np.pi * rho_s * (r_s / 1000.0)**3 * (
        np.log(1.0 + x) - x / (1.0 + x))  # in Msun (r_s converted to Mpc)

    # Actually need r_s in Mpc for mass calc with rho in Msun/Mpc^3
    r_s_Mpc = r_s / 1000.0
    M_enc = 4.0 * np.pi * rho_s * r_s_Mpc**3 * (
        np.log(1.0 + x) - x / (1.0 + x))  # Msun

    # v_circ = sqrt(G M(<r) / r), with G in (km/s)² kpc / Msun
    # G_kpc = 4.302e-3 * 1e-3 = 4.302e-6 (km/s)^2 kpc / Msun
    G_kpc = 4.302e-3 * 1e-3  # (km/s)^2 kpc / Msun
    v_circ = np.sqrt(G_kpc * M_enc / R_test)  # km/s

    # Convert back using the Brouwer formula
    R_pc = R_test * 1000.0
    delta_sigma_from_v = v_circ**2 / (4.0 * G_conv * R_pc)  # Msun/pc²

    # Compare
    ratio_v_to_lens = delta_sigma_from_v / np.maximum(true_delta_sigma, 1e-10)

    print(f"\n  NFW logMh={logMh}, c={c_test:.1f}:")
    print(f"    R(kpc)   ΔΣ_true    ΔΣ_from_v    ratio")
    for j in [0, 5, 10, 15, 19]:
        if j < len(R_test):
            print(f"    {R_test[j]:6.1f}   {true_delta_sigma[j]:10.3f}   "
                  f"{delta_sigma_from_v[j]:10.3f}   {ratio_v_to_lens[j]:.3f}")


# ============================================================
# DIAGNOSTIC 5: BEC with fixed r_c = ξ (no freedom)
# ============================================================
print("\n" + "=" * 72)
print("DIAGNOSTIC 5: BEC WITH FIXED r_c = ξ (PREDICTION-ONLY)")
print("=" * 72)
print("  If BEC is physical, fixing r_c = ξ (no extra free param)")
print("  should still improve over NFW. This is the strongest test.")

for b in range(4):
    if composite[b] is None:
        continue
    d = composite[b]
    R = d['R_kpc']
    ESD = d['ESD']
    err = d['err']
    logMs_i = d['logMs']

    valid = (ESD > 0) & (err > 0) & np.isfinite(ESD) & np.isfinite(err)
    R_f = R[valid]
    ESD_f = ESD[valid]
    err_f = err[valid]

    if len(R_f) < 6:
        continue

    Ms_SI = 10.0**logMs_i * Msun_kg
    xi_m = np.sqrt(G_SI * Ms_SI / gdagger)
    xi_kpc = xi_m / kpc_m

    logMh_guess = logMs_i + 1.5
    c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)

    # NFW
    def nfw_chi2(params, R_f=R_f, E_f=ESD_f, e_f=err_f):
        logM, c = params
        if c < 1 or c > 50 or logM < 10 or logM > 15:
            return 1e20
        model = nfw_delta_sigma(R_f, logM, c)
        return np.sum(((E_f - model) / e_f)**2)

    res_nfw = minimize(nfw_chi2, [logMh_guess, c_guess],
                       method='Nelder-Mead', options={'maxiter': 10000})
    chi2_nfw = res_nfw.fun
    aic_nfw = chi2_nfw + 2 * 2

    # BEC with r_c FIXED to ξ — only 3 free params (logM, c, f_sol)
    def bec_fixed_rc_chi2(params, R_f=R_f, E_f=ESD_f, e_f=err_f, rc_fixed=xi_kpc):
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
            res = minimize(bec_fixed_rc_chi2, [logMh_guess, c_guess, fs0],
                           method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-4})
            if res.fun < best_chi2_fixed:
                best_chi2_fixed = res.fun
                best_fixed = res.x
        except Exception:
            pass

    aic_fixed = best_chi2_fixed + 2 * 3  # 3 params (logM, c, f_sol)
    daic_fixed = aic_nfw - aic_fixed
    n_dof = len(R_f)

    print(f"\n  Bin {b+1} ({d['label']}), ξ = {xi_kpc:.2f} kpc:")
    print(f"    NFW:           χ²/dof = {chi2_nfw / max(n_dof-2,1):.2f}, "
          f"AIC = {aic_nfw:.1f}")
    print(f"    BEC (r_c=ξ):   χ²/dof = {best_chi2_fixed / max(n_dof-3,1):.2f}, "
          f"AIC = {aic_fixed:.1f}")
    print(f"    ΔAIC (NFW − BEC_fixed) = {daic_fixed:+.2f}")
    if best_chi2_fixed < 1e19:
        logM_f, c_f, fs_f = best_fixed
        print(f"    Best-fit: logM={logM_f:.2f}, c={c_f:.1f}, f_sol={fs_f:.3f}")
    if daic_fixed > 2:
        print(f"    ✓ BEC improves even with PREDICTED r_c — physical signal!")
    elif daic_fixed > -2:
        print(f"    ~ Indistinguishable with fixed r_c")
    else:
        print(f"    ✗ NFW preferred when r_c constrained to prediction")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("DIAGNOSTIC SUMMARY")
print("=" * 72)
print("""
Key questions answered by these diagnostics:

1. STITCH BOUNDARY: Is there a systematic jump at R=30 kpc where
   SPARC rotation curves meet Brouwer lensing? If yes, the BEC model's
   extra parameters could simply be absorbing this discontinuity.

2. SPLIT SAMPLE: Does BEC preference come from the inner (SPARC) data
   where the soliton lives, or from the outer (Brouwer) data where
   both models should be identical? If outer-only shows BEC preference,
   something is wrong.

3. NFW+OFFSET: Can a simple normalization offset between datasets
   explain the BEC preference? If NFW with one extra parameter
   (A_sparc) achieves similar ΔAIC as BEC (two extra params),
   the soliton detection is not robust.

4. V→ΔΣ CONVERSION: How accurate is ΔΣ = V²/(4GR) for an NFW halo?
   Systematic bias in the conversion could masquerade as a soliton.

5. FIXED r_c: If BEC is physical, fixing r_c = ξ_predicted should
   still improve fits. This is the CLEANEST test with only 1 extra
   parameter (f_sol) beyond NFW.
""")

print("=" * 72)
print("Done.")
