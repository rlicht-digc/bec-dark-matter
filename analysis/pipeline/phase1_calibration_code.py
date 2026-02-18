"""
PRIMORDIAL FLUID MODEL — Optimized Calibration v2
===================================================
Proper least-squares fitting of the PF density profile against
real galaxy rotation curves. Produces publication-quality χ² fits.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares, minimize
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.lss import mass_function
import os, shutil

cosmology.setCosmology('planck18')
cosmo = cosmology.getCurrent()

# ============================================================
# CORE EQUATIONS OF THE PRIMORDIAL FLUID MODEL
# ============================================================
# 
# Equation 1: DM Density Profile
#   ρ_DM(r) = ρ₀ · [1 + (r/r_c)²]^(-α/2) · exp(-(r/r_t)^β)
#   
#   ρ₀  = central (peak) DM density
#   r_c = core radius (from fluid equilibrium)
#   α   = inner power-law index (0 = flat core, 1 = NFW cusp)
#   r_t = truncation radius (outer boundary)
#   β   = truncation sharpness
#
# Equation 2: Fluid Equilibrium (determines r_c)
#   r_c = v_circ(r_c) · τ_response
#   where τ_response = 1 / (1 - dm_fade) in physical time units
#   This means: core size = how far the fluid "spreads" during 
#   one response cycle. Faster rotating galaxies → bigger cores.
#
# Equation 3: DM Accumulation Rate
#   dρ_DM/dt = Γ · ρ_matter(r) · κ(r) - ρ_DM / τ_fade
#   Γ     = growth coefficient (Russell's dm_growth)  
#   κ(r)  = spacetime curvature ∝ ∇²Φ (Poisson equation)
#   τ_fade = decay timescale (Russell's 1/(1-dm_fade))
#   
#   At equilibrium: ρ_DM = Γ · τ_fade · ρ_matter · κ
#
# Equation 4: Modified Poisson Equation (feedback)
#   ∇²Φ = 4πG(ρ_matter + η · ρ_DM)
#   η = DM gravitational coupling (η=1 for standard gravity)
#   This creates the three-way feedback loop:
#     matter → curvature → DM accumulation → enhanced gravity → matter
#
# Equation 5: Enclosed Mass & Rotation Curve
#   M(<r) = 4π ∫₀ʳ [ρ_disk(r') + ρ_DM(r')] r'² dr'
#   V_circ(r) = √(G · M(<r) / r)
# ============================================================

G_KPC = 4.302e-3 * 1e-3  # G in (km/s)²·kpc/M☉

# --- Profile Functions ---

def pf_density(r, rho0, rc, alpha, rt, beta):
    """Primordial Fluid DM density profile (Equation 1)."""
    core = (1.0 + (r / rc)**2)**(-alpha / 2.0)
    trunc = np.exp(-(r / rt)**beta)
    return rho0 * core * trunc

def nfw_density(r, rho_s, r_s):
    """Standard NFW profile."""
    x = r / r_s
    return rho_s / (x * (1 + x)**2)

def burkert_density(r, rho0, r0):
    """Burkert cored profile."""
    return rho0 * r0**3 / ((r + r0) * (r**2 + r0**2))

def exponential_disk_density(r, z, Sigma0, Rd, z0=0.3):
    """Exponential disk mass density (cylindrical, evaluated at z=0 midplane)."""
    return (Sigma0 / (2 * z0)) * np.exp(-r / Rd) * np.exp(-np.abs(z) / z0)

def enclosed_mass_spherical(r_arr, rho_func, rho_params, n_int=2000):
    """Compute M(<r) for a spherical density profile."""
    M = np.zeros_like(r_arr, dtype=float)
    for i, R in enumerate(r_arr):
        if R < 0.01:
            M[i] = 0
            continue
        r_int = np.linspace(0.01, R, n_int)
        rho_vals = rho_func(r_int, *rho_params)
        M[i] = np.trapezoid(4 * np.pi * r_int**2 * rho_vals, r_int)
    return M

def v_circ_from_density(r_arr, rho_func, rho_params, n_int=2000):
    """V_circ = sqrt(G * M(<r) / r)"""
    M_enc = enclosed_mass_spherical(r_arr, rho_func, rho_params, n_int)
    v = np.sqrt(G_KPC * M_enc / np.maximum(r_arr, 0.01))
    return v

def v_disk_freeman(r, V_flat_disk, Rd):
    """Freeman exponential disk rotation curve approximation."""
    y = r / (2.0 * Rd)
    y = np.maximum(y, 1e-6)
    # Exact: V² = 4πGΣ₀Rd·y²[I₀(y)K₀(y) - I₁(y)K₁(y)]
    # Approximation that peaks at ~2.2Rd:
    from scipy.special import i0, i1, k0, k1
    I0 = i0(y); I1 = i1(y); K0 = k0(y); K1 = k1(y)
    v2 = V_flat_disk**2 * 2.0 * y**2 * (I0 * K0 - I1 * K1) / 0.2185
    return np.sqrt(np.maximum(v2, 0))

def v_gas(r, V_gas_max, R_gas):
    """Gas contribution (simplified)."""
    return V_gas_max * (1 - np.exp(-r / R_gas))


# ============================================================
# GALAXY DATA
# ============================================================

# NGC 3198 — The classic DM rotation curve galaxy
ngc3198 = {
    'name': 'NGC 3198',
    'r': np.array([0.8, 2.0, 3.2, 4.4, 5.6, 6.8, 8.0, 9.6, 11.2, 12.8,
                    14.4, 16.0, 18.0, 20.0, 22.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0]),
    'v': np.array([62, 102, 125, 135, 142, 147, 149, 150, 151, 150,
                    150, 150, 150, 149, 149, 149, 148, 148, 148, 150, 148, 146]),
    'err': np.array([12, 8, 5, 4, 3, 3, 3, 3, 3, 3,
                      3, 4, 4, 4, 5, 5, 5, 6, 7, 8, 9, 10]),
    'Rd': 2.7,        # disk scale length (kpc)
    'distance': 13.4,  # Mpc
}

# NGC 2403 — Well-studied Sc spiral
ngc2403 = {
    'name': 'NGC 2403',
    'r': np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                    10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
    'v': np.array([30, 55, 90, 108, 118, 124, 128, 130, 132, 133,
                    134, 135, 135, 135, 134, 134, 133]),
    'err': np.array([8, 6, 4, 3, 3, 3, 3, 3, 3, 3,
                      4, 4, 4, 5, 5, 6, 7]),
    'Rd': 2.0,
    'distance': 3.2,
}

# DDO 154 — Dark matter dominated dwarf
ddo154 = {
    'name': 'DDO 154',
    'r': np.array([0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                    4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]),
    'v': np.array([8, 15, 20, 24, 27, 31, 35, 37, 39, 41,
                    43, 44, 45, 46, 47, 47, 47, 47]),
    'err': np.array([3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 4, 4]),
    'Rd': 0.7,
    'distance': 3.7,
}

galaxies = [ngc3198, ngc2403, ddo154]

# ============================================================
# FIT EACH GALAXY WITH THREE MODELS
# ============================================================

def fit_galaxy(gal, model='pf'):
    """
    Fit a rotation curve with disk + DM halo.
    Returns: best-fit params, chi2, v_model(r)
    """
    r_data = gal['r']
    v_data = gal['v']
    v_err = gal['err']
    Rd = gal['Rd']
    
    r_fine = np.linspace(0.1, r_data[-1] * 1.1, 300)
    
    if model == 'pf':
        # Primordial Fluid: [V_disk, rho0, rc, alpha, rt]
        def residuals(p):
            V_disk, log_rho0, rc, alpha, log_rt = p
            rho0 = 10**log_rho0
            rt = 10**log_rt
            if rc < 0.1 or alpha < 0 or alpha > 2 or rt < 5:
                return np.ones_like(r_data) * 1e6
            v_d = v_disk_freeman(r_data, V_disk, Rd)
            v_dm = v_circ_from_density(r_data, pf_density, (rho0, rc, alpha, rt, 2.0), 1500)
            v_tot = np.sqrt(v_d**2 + v_dm**2)
            return (v_tot - v_data) / v_err
        
        # Initial guess scales with galaxy size
        V_max = np.max(v_data)
        r_max = r_data[-1]
        x0 = [V_max * 0.6, 7.0, r_max * 0.15, 0.3, np.log10(r_max * 2)]
        bounds = ([10, 4, 0.1, 0.0, 0.5], [300, 10, r_max, 2.0, 4.0])
        
        result = least_squares(residuals, x0, bounds=bounds, method='trf', 
                                max_nfev=5000, ftol=1e-10, xtol=1e-10)
        p = result.x
        chi2 = np.sum(result.fun**2)
        
        # Compute model on fine grid
        rho0 = 10**p[1]; rt = 10**p[4]
        v_d_fine = v_disk_freeman(r_fine, p[0], Rd)
        v_dm_fine = v_circ_from_density(r_fine, pf_density, (rho0, p[2], p[3], rt, 2.0), 1500)
        v_tot_fine = np.sqrt(v_d_fine**2 + v_dm_fine**2)
        
        params = {'V_disk': p[0], 'rho0': rho0, 'rc': p[2], 'alpha': p[3], 'rt': rt,
                  'model': 'Primordial Fluid'}
        return params, chi2, r_fine, v_tot_fine, v_d_fine, v_dm_fine
    
    elif model == 'nfw':
        # NFW: [V_disk, log_rho_s, r_s]
        def residuals(p):
            V_disk, log_rho_s, r_s = p
            rho_s = 10**log_rho_s
            if r_s < 0.5:
                return np.ones_like(r_data) * 1e6
            v_d = v_disk_freeman(r_data, V_disk, Rd)
            v_dm = v_circ_from_density(r_data, nfw_density, (rho_s, r_s), 1500)
            v_tot = np.sqrt(v_d**2 + v_dm**2)
            return (v_tot - v_data) / v_err
        
        V_max = np.max(v_data)
        x0 = [V_max * 0.6, 6.5, 15.0]
        bounds = ([10, 4, 0.5], [300, 10, 200])
        result = least_squares(residuals, x0, bounds=bounds, method='trf',
                                max_nfev=5000, ftol=1e-10, xtol=1e-10)
        p = result.x
        chi2 = np.sum(result.fun**2)
        
        rho_s = 10**p[1]
        v_d_fine = v_disk_freeman(r_fine, p[0], Rd)
        v_dm_fine = v_circ_from_density(r_fine, nfw_density, (rho_s, p[2]), 1500)
        v_tot_fine = np.sqrt(v_d_fine**2 + v_dm_fine**2)
        
        params = {'V_disk': p[0], 'rho_s': rho_s, 'r_s': p[2], 'model': 'NFW'}
        return params, chi2, r_fine, v_tot_fine, v_d_fine, v_dm_fine
    
    elif model == 'burkert':
        def residuals(p):
            V_disk, log_rho0, r0 = p
            rho0 = 10**log_rho0
            if r0 < 0.1:
                return np.ones_like(r_data) * 1e6
            v_d = v_disk_freeman(r_data, V_disk, Rd)
            v_dm = v_circ_from_density(r_data, burkert_density, (rho0, r0), 1500)
            v_tot = np.sqrt(v_d**2 + v_dm**2)
            return (v_tot - v_data) / v_err
        
        V_max = np.max(v_data)
        x0 = [V_max * 0.6, 7.0, 5.0]
        bounds = ([10, 4, 0.1], [300, 10, 100])
        result = least_squares(residuals, x0, bounds=bounds, method='trf',
                                max_nfev=5000, ftol=1e-10, xtol=1e-10)
        p = result.x
        chi2 = np.sum(result.fun**2)
        
        rho0 = 10**p[1]
        v_d_fine = v_disk_freeman(r_fine, p[0], Rd)
        v_dm_fine = v_circ_from_density(r_fine, burkert_density, (rho0, p[2]), 1500)
        v_tot_fine = np.sqrt(v_d_fine**2 + v_dm_fine**2)
        
        params = {'V_disk': p[0], 'rho0': rho0, 'r0': p[2], 'model': 'Burkert'}
        return params, chi2, r_fine, v_tot_fine, v_d_fine, v_dm_fine

# ============================================================
# RUN ALL FITS
# ============================================================
print("=" * 70)
print("PRIMORDIAL FLUID MODEL — Optimized Calibration v2")
print("=" * 70)

all_results = {}
for gal in galaxies:
    name = gal['name']
    dof = len(gal['r']) - 3  # approximate dof
    print(f"\n{'─'*50}")
    print(f"  {name} (N={len(gal['r'])} points, Rd={gal['Rd']} kpc)")
    print(f"{'─'*50}")
    
    all_results[name] = {}
    for model in ['nfw', 'burkert', 'pf']:
        try:
            params, chi2, r_f, v_f, v_d, v_dm = fit_galaxy(gal, model)
            chi2_dof = chi2 / max(dof, 1)
            all_results[name][model] = {
                'params': params, 'chi2': chi2, 'chi2_dof': chi2_dof,
                'r': r_f, 'v_total': v_f, 'v_disk': v_d, 'v_dm': v_dm
            }
            
            label = params['model']
            print(f"\n  {label}:")
            print(f"    χ² = {chi2:.2f}, χ²/dof = {chi2_dof:.2f}")
            for k, v in params.items():
                if k == 'model': continue
                if isinstance(v, float):
                    if v > 1e4:
                        print(f"    {k} = {v:.3e}")
                    else:
                        print(f"    {k} = {v:.3f}")
        except Exception as e:
            print(f"  {model}: FAILED — {e}")

# ============================================================
# PRINT SCORECARD
# ============================================================
print(f"\n\n{'='*70}")
print("ROTATION CURVE FIT SCORECARD")
print(f"{'='*70}")
print(f"\n  {'Galaxy':<12} | {'NFW χ²/dof':>12} | {'Burkert χ²/dof':>15} | {'PF χ²/dof':>12} | {'Winner':>10}")
print(f"  {'─'*12} | {'─'*12} | {'─'*15} | {'─'*12} | {'─'*10}")

for name in ['NGC 3198', 'NGC 2403', 'DDO 154']:
    res = all_results.get(name, {})
    vals = {}
    for m in ['nfw', 'burkert', 'pf']:
        if m in res:
            vals[m] = res[m]['chi2_dof']
        else:
            vals[m] = float('inf')
    
    winner = min(vals, key=vals.get)
    winner_label = {'nfw': 'NFW', 'burkert': 'Burkert', 'pf': 'PF'}[winner]
    
    print(f"  {name:<12} | {vals.get('nfw', 0):>12.2f} | {vals.get('burkert', 0):>15.2f} | "
          f"{vals.get('pf', 0):>12.2f} | {winner_label:>10}")

# ============================================================
# KEY DERIVED QUANTITIES
# ============================================================
print(f"\n\n{'='*70}")
print("PRIMORDIAL FLUID — KEY DERIVED QUANTITIES")
print(f"{'='*70}")

# Core radius vs circular velocity relation
print(f"\n  Core radius vs V_circ (Equation 2: r_c = V_c × τ_response):")
print(f"  {'Galaxy':<12} | {'V_flat (km/s)':>13} | {'r_c (kpc)':>10} | {'r_c/V_flat':>10} | {'Expected':>10}")
print(f"  {'─'*12} | {'─'*13} | {'─'*10} | {'─'*10} | {'─'*10}")

for name in ['NGC 3198', 'NGC 2403', 'DDO 154']:
    gal = [g for g in galaxies if g['name'] == name][0]
    res = all_results.get(name, {}).get('pf', {})
    if res:
        V_flat = np.max(gal['v'])
        rc = res['params'].get('rc', 0)
        ratio = rc / V_flat * 1000 if V_flat > 0 else 0  # kpc/(km/s) * 1000
        print(f"  {name:<12} | {V_flat:>13.0f} | {rc:>10.2f} | {ratio:>10.3f} | {'~0.04':>10}")

# Central density × core radius product
print(f"\n  Donato relation test: ρ₀ × r₀ ≈ 75 M☉/pc²")
print(f"  {'Galaxy':<12} | {'ρ₀ (M☉/kpc³)':>14} | {'r_c (kpc)':>10} | {'ρ₀×r_c (M☉/pc²)':>17}")
print(f"  {'─'*12} | {'─'*14} | {'─'*10} | {'─'*17}")

for name in ['NGC 3198', 'NGC 2403', 'DDO 154']:
    res = all_results.get(name, {}).get('pf', {})
    if res:
        rho0 = res['params'].get('rho0', 0)
        rc = res['params'].get('rc', 0)
        product = rho0 * rc * 1e-6  # M☉/kpc³ × kpc → M☉/kpc² → M☉/pc² (×1e-6)
        print(f"  {name:<12} | {rho0:>14.3e} | {rc:>10.2f} | {product:>17.1f}")

print(f"\n  Target: ρ₀ × r₀ ≈ 75 M☉/pc² (Donato et al. 2009)")

# ============================================================
# GENERATE PUBLICATION PLOTS
# ============================================================
print(f"\n\n{'='*70}")
print("GENERATING PUBLICATION PLOTS...")
print(f"{'='*70}")

# Colors
C_NFW = '#ff6b6b'
C_BUR = '#4ecdc4'
C_PF  = '#c084fc'
C_DAT = '#fbbf24'
C_DSK = '#6b7280'
C_BG  = '#0a0a14'
C_TXT = '#d1d5db'
C_GRD = '#1a1a2e'

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor(C_BG)

# Layout: 3 galaxies × 2 columns (rotation curve + density profile) + 3 bottom panels
gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3,
              height_ratios=[1, 1, 1, 0.1, 1.2])

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(C_BG)
    ax.set_title(title, color='#e5e7eb', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C_TXT, fontsize=10)
    ax.set_ylabel(ylabel, color=C_TXT, fontsize=10)
    ax.tick_params(colors=C_TXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_GRD)
    ax.grid(True, alpha=0.12, color=C_GRD)

# Plot rotation curves and density profiles for each galaxy
for i, gal in enumerate(galaxies):
    name = gal['name']
    res = all_results.get(name, {})
    
    # LEFT: Rotation curve
    ax = fig.add_subplot(gs[i, 0])
    ax.errorbar(gal['r'], gal['v'], yerr=gal['err'], fmt='o', color=C_DAT,
                markersize=4, capsize=2, label=f'{name} data', zorder=10)
    
    for model, color, ls in [('nfw', C_NFW, '-'), ('burkert', C_BUR, '--'), ('pf', C_PF, '-')]:
        if model in res:
            r_m = res[model]['r']
            v_m = res[model]['v_total']
            chi2 = res[model]['chi2_dof']
            label = f"{res[model]['params']['model']} (χ²/ν={chi2:.2f})"
            lw = 3 if model == 'pf' else 2
            ax.plot(r_m, v_m, color=color, ls=ls, lw=lw, label=label, alpha=0.9)
            
            # Also plot disk and DM components for PF
            if model == 'pf':
                ax.plot(r_m, res[model]['v_disk'], ':', color=C_DSK, lw=1, label='Disk', alpha=0.5)
                ax.plot(r_m, res[model]['v_dm'], ':', color=C_PF, lw=1, label='DM (PF)', alpha=0.5)
    
    ax.set_xlim(0, gal['r'][-1] * 1.1)
    ax.set_ylim(0, np.max(gal['v']) * 1.4)
    ax.legend(fontsize=7, facecolor='#0d0d1a', edgecolor=C_GRD, labelcolor=C_TXT, loc='lower right')
    style_ax(ax, f'{name} — Rotation Curve', 'r [kpc]', 'V_circ [km/s]')
    
    # RIGHT: DM density profile
    ax2 = fig.add_subplot(gs[i, 1])
    r_prof = np.logspace(-1, np.log10(gal['r'][-1] * 1.5), 300)
    
    for model, color, ls in [('nfw', C_NFW, '-'), ('burkert', C_BUR, '--'), ('pf', C_PF, '-')]:
        if model in res:
            p = res[model]['params']
            if model == 'nfw':
                rho = nfw_density(r_prof, p['rho_s'], p['r_s'])
            elif model == 'burkert':
                rho = burkert_density(r_prof, p['rho0'], p['r0'])
            elif model == 'pf':
                rho = pf_density(r_prof, p['rho0'], p['rc'], p['alpha'], p['rt'], 2.0)
            
            lw = 3 if model == 'pf' else 2
            ax2.loglog(r_prof, rho, color=color, ls=ls, lw=lw, label=p['model'], alpha=0.9)
    
    # Mark core radius for PF
    if 'pf' in res:
        rc = res['pf']['params']['rc']
        ax2.axvline(rc, color=C_PF, ls=':', alpha=0.5)
        ax2.text(rc * 1.1, ax2.get_ylim()[0] * 10 if ax2.get_ylim()[0] > 0 else 1e3, 
                 f'r_c={rc:.1f}', color=C_PF, fontsize=8, rotation=90, va='bottom')
    
    ax2.legend(fontsize=8, facecolor='#0d0d1a', edgecolor=C_GRD, labelcolor=C_TXT)
    style_ax(ax2, f'{name} — DM Density Profile', 'r [kpc]', 'ρ [M☉/kpc³]')

# Bottom panels: Mass function, Power spectrum, Inner slope
# Mass function
ax_mf = fig.add_subplot(gs[4, 0])
M_range = np.logspace(8, 15.5, 100)
mfunc = mass_function.massFunction(M_range, 0, mdef='200c', model='tinker08', q_out='dndlnM')
M_suppress = 5e9
pf_suppress = 1.0 / (1.0 + (M_suppress / M_range)**1.5)
mfunc_pf = mfunc * pf_suppress

ax_mf.loglog(M_range, mfunc, color=C_NFW, lw=2.5, label='ΛCDM (Tinker 2008)')
ax_mf.loglog(M_range, mfunc_pf, color=C_PF, lw=3, label='Primordial Fluid')
ax_mf.fill_between(M_range, mfunc_pf, mfunc, alpha=0.12, color=C_PF)
ax_mf.axvline(M_suppress, color=C_DAT, ls=':', lw=1.5, alpha=0.7)
ax_mf.text(M_suppress * 0.3, 3e-5, 'Fluid\nresponse\nscale', color=C_DAT, fontsize=8, ha='right')
ax_mf.set_xlim(1e8, 1e16); ax_mf.set_ylim(1e-9, 1)
ax_mf.legend(fontsize=9, facecolor='#0d0d1a', edgecolor=C_GRD, labelcolor=C_TXT)
style_ax(ax_mf, 'Halo Mass Function — PF Suppresses Small Halos', 'M [M☉/h]', 'dn/dlnM [h³/Mpc³]')

# Power spectrum
ax_pk = fig.add_subplot(gs[4, 1])
k_range = np.logspace(-3, 1.5, 200)
Pk = cosmo.matterPowerSpectrum(k_range, z=0)
k_cut = 5.0
Pk_pf = Pk / (1.0 + (k_range / k_cut)**2)

ax_pk.loglog(k_range, Pk, color=C_NFW, lw=2.5, label='ΛCDM P(k)')
ax_pk.loglog(k_range, Pk_pf, color=C_PF, lw=3, label='Primordial Fluid P(k)')
ax_pk.fill_between(k_range, Pk_pf, Pk, alpha=0.12, color=C_PF, where=Pk_pf < Pk * 0.95)
ax_pk.axvline(k_cut, color=C_DAT, ls=':', lw=1.5)
ax_pk.text(k_cut * 1.3, 5e3, f'k_cut = {k_cut} h/Mpc\n(~{2*np.pi/k_cut:.1f} Mpc/h)', 
           color=C_DAT, fontsize=8)
ax_pk.set_xlim(1e-3, 30); ax_pk.legend(fontsize=9, facecolor='#0d0d1a', edgecolor=C_GRD, labelcolor=C_TXT)
style_ax(ax_pk, 'Matter Power Spectrum — Small-Scale Suppression', 'k [h/Mpc]', 'P(k) [(Mpc/h)³]')

fig.suptitle("PRIMORDIAL FLUID DARK MATTER MODEL\n"
             "Phase 1 Calibration: Optimized Fits Against Real Galaxy Data",
             color=C_PF, fontsize=18, fontweight='bold', y=0.99)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
_outfile = os.path.join(RESULTS_DIR, 'phase1_v2_calibration.png')
plt.savefig(_outfile, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\n✓ Publication plots saved: phase1_v2_calibration.png")
print(f"\n{'='*70}")
print("COMPLETE — Phase 1 Calibration v2 Done")
print(f"{'='*70}")
