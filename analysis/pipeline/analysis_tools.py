#!/usr/bin/env python3
"""
analysis_tools.py — Shared analysis functions for BEC dark matter pipeline.

Consolidates duplicated code from ~30 pipeline scripts into reusable functions.
All test scripts should import from here rather than redefining these.

Modules:
  - Physics constants
  - RAR function and derivatives
  - Environment classification
  - Binned statistics (scatter, kurtosis, skewness)
  - Derivative analysis and zero-crossing finder
  - Inversion point finder
  - ΛCDM mock galaxy generator
  - NFW and disk profile functions
"""

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew


# ============================================================
# PHYSICS CONSTANTS
# ============================================================

g_dagger = 1.20e-10          # m/s² — BEC condensation scale
LOG_G_DAGGER = np.log10(g_dagger)  # -9.9208...

G_SI = 6.674e-11             # m³ kg⁻¹ s⁻² — gravitational constant
M_SUN = 1.989e30             # kg — solar mass
KPC_M = 3.086e19             # m per kpc
MPC_M = 3.086e22             # m per Mpc

H0 = 67.74                   # km/s/Mpc (Planck 2015)
h_HUBBLE = H0 / 100.0
RHO_CRIT = 1.27e11 * h_HUBBLE**2   # M_sun / Mpc³
OMEGA_M = 0.3089


# ============================================================
# RAR FUNCTION AND DERIVATIVES
# ============================================================

def rar_function(log_gbar, a0=g_dagger):
    """Standard RAR: log g_obs from log g_bar (McGaugh+2016 / BEC form).

    g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))

    Parameters:
        log_gbar: array-like, log10(g_bar) in m/s²
        a0: acceleration scale (default: g† = 1.2e-10)

    Returns:
        log10(g_obs) array
    """
    gbar = 10.0**np.asarray(log_gbar, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def rar_residuals(log_gbar, log_gobs, a0=g_dagger):
    """RAR residuals: log g_obs(observed) - log g_obs(predicted)."""
    return np.asarray(log_gobs) - rar_function(log_gbar, a0)


def rar_slope(log_gbar, a0=g_dagger):
    """Analytic RAR slope: d(log g_obs)/d(log g_bar).

    = 1 - [eps * exp(-eps)] / [2 * (1 - exp(-eps))]
    where eps = sqrt(g_bar / a0).

    This is the sensitivity of g_obs to perturbations in g_bar.
    Deep-MOND limit: slope -> 0.5. Newtonian limit: slope -> 1.
    """
    gbar = 10.0**np.asarray(log_gbar, dtype=float)
    eps = np.sqrt(gbar / a0)
    exp_neg = np.exp(-eps)
    denom = np.maximum(1.0 - exp_neg, 1e-30)
    return 1.0 - (eps * exp_neg) / (2.0 * denom)


def condensate_fraction(log_gbar, a0=g_dagger):
    """Local condensate fraction f_c = 1 - g_bar/g_obs = n̄/(1+n̄).

    f_c = exp(-sqrt(g_bar/a0))

    The order parameter of the BEC phase transition (§8.26).
    """
    gbar = 10.0**np.asarray(log_gbar, dtype=float)
    eps = np.sqrt(gbar / a0)
    return np.exp(-eps)


def susceptibility(log_gbar, a0=g_dagger):
    """Generalized susceptibility: |df_c / d(log g_bar)|.

    = (ln(10)/2) * eps * exp(-eps)

    Peaks at g†. Measures response of condensate fraction to
    perturbations in the baryonic field.
    """
    gbar = 10.0**np.asarray(log_gbar, dtype=float)
    eps = np.sqrt(gbar / a0)
    return (np.log(10.0) / 2.0) * eps * np.exp(-eps)


# ============================================================
# ENVIRONMENT CLASSIFICATION
# ============================================================

# Ursa Major cluster membership (Verheijen+2001, Tully+2013)
UMA_GALAXIES = {
    'NGC3726', 'NGC3769', 'NGC3877', 'NGC3893', 'NGC3917',
    'NGC3949', 'NGC3953', 'NGC3972', 'NGC3992', 'NGC4010',
    'NGC4013', 'NGC4051', 'NGC4085', 'NGC4088', 'NGC4100',
    'NGC4138', 'NGC4157', 'NGC4183', 'NGC4217',
    'UGC06399', 'UGC06446', 'UGC06667', 'UGC06786', 'UGC06787',
    'UGC06818', 'UGC06917', 'UGC06923', 'UGC06930', 'UGC06973',
    'UGC06983', 'UGC07089',
}

# Known group memberships (NED, Tully+2013 2MASS groups)
GROUP_MEMBERS = {
    'NGC2403': 'M81', 'NGC2976': 'M81', 'IC2574': 'M81',
    'DDO154': 'M81', 'DDO168': 'M81', 'UGC04483': 'M81',
    'NGC0300': 'Sculptor', 'NGC0055': 'Sculptor',
    'NGC0247': 'Sculptor', 'NGC7793': 'Sculptor',
    'NGC2915': 'CenA', 'UGCA442': 'CenA', 'ESO444-G084': 'CenA',
    'UGC07577': 'CVnI', 'UGC07232': 'CVnI', 'NGC3741': 'CVnI',
    'NGC4068': 'CVnI', 'UGC07866': 'CVnI', 'UGC07524': 'CVnI',
    'UGC08490': 'CVnI', 'UGC07559': 'CVnI',
    'NGC3109': 'Antlia', 'NGC5055': 'M101',
}


def classify_env(name):
    """Classify galaxy as 'dense' or 'field' based on SPARC membership lists.

    Returns: ('dense', group_name) or ('field', 'field')
    """
    if name in UMA_GALAXIES:
        return 'dense', 'UMa'
    if name in GROUP_MEMBERS:
        return 'dense', GROUP_MEMBERS[name]
    return 'field', 'field'


def classify_env_simple(name):
    """Simplified: returns just 'dense' or 'field'."""
    return classify_env(name)[0]


def classify_env_extended(name, source=None, extra_dense=None):
    """Extended environment classification for multi-survey datasets.

    Handles SPARC, Verheijen (all UMa), deBlok (all field), Korsaga,
    and any extra dense galaxy sets passed in.

    Parameters:
        name: galaxy name
        source: data source string (e.g. 'SPARC', 'Verheijen2001', etc.)
        extra_dense: optional set of galaxy names known to be dense

    Returns: 'dense', 'field', or 'unclassified'
    """
    import re
    def _norm(n):
        n = n.strip().strip('"')
        n = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+',
                   r'\1', n, flags=re.IGNORECASE)
        return n.upper()

    norm = _norm(name)
    uma_norms = {_norm(u) for u in UMA_GALAXIES}
    group_norms = {_norm(g) for g in GROUP_MEMBERS}

    if norm in uma_norms or norm in group_norms:
        return 'dense'
    if extra_dense and name in extra_dense:
        return 'dense'
    if source == 'Verheijen2001':
        return 'dense'
    if source == 'deBlok2002':
        return 'field'
    if source in ('SPARC', 'THINGS'):
        return 'field'
    return 'unclassified'


# ============================================================
# BINNED STATISTICS
# ============================================================

def binned_stats(log_gbar, log_gobs, n_bins=15, gbar_range=(-12.5, -8.5),
                 min_n=20, a0=g_dagger):
    """Compute binned RAR residual statistics (σ, skewness, kurtosis).

    Parameters:
        log_gbar, log_gobs: arrays of log10(acceleration) in m/s²
        n_bins: number of equal-width bins
        gbar_range: (lo, hi) range for binning
        min_n: minimum points per bin (else NaN)
        a0: acceleration scale for RAR prediction

    Returns:
        list of dicts with keys: center, n, sigma, skewness, kurtosis
    """
    resid = rar_residuals(log_gbar, log_gobs, a0)
    edges = np.linspace(gbar_range[0], gbar_range[1], n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    results = []
    for j in range(n_bins):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j + 1])
        res = resid[mask]

        if len(res) < min_n:
            results.append({
                'center': float(centers[j]), 'n': int(len(res)),
                'sigma': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
            })
            continue

        results.append({
            'center': float(centers[j]),
            'n': int(len(res)),
            'sigma': float(np.std(res)),
            'skewness': float(scipy_skew(res)),
            'kurtosis': float(scipy_kurtosis(res, fisher=True)),
        })

    return results


def bootstrap_kurtosis(log_gbar, log_gobs, n_bins=15, gbar_range=(-12.5, -8.5),
                       min_n=20, n_boot=10000, a0=g_dagger):
    """Bootstrap 95% CI for kurtosis in each acceleration bin.

    Returns:
        list of dicts with keys: center, n, kurtosis_mean, kurtosis_median,
        kurtosis_ci_lo, kurtosis_ci_hi, kurtosis_std
    """
    resid = rar_residuals(log_gbar, log_gobs, a0)
    edges = np.linspace(gbar_range[0], gbar_range[1], n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    results = []
    for j in range(n_bins):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j + 1])
        res = resid[mask]

        if len(res) < min_n:
            results.append({
                'center': float(centers[j]), 'n': int(len(res)),
                'kurtosis_mean': np.nan, 'kurtosis_median': np.nan,
                'kurtosis_ci_lo': np.nan, 'kurtosis_ci_hi': np.nan,
                'kurtosis_std': np.nan,
            })
            continue

        n_pts = len(res)
        kurt_samples = np.zeros(n_boot)
        for b in range(n_boot):
            idx = np.random.randint(0, n_pts, n_pts)
            kurt_samples[b] = scipy_kurtosis(res[idx], fisher=True)

        results.append({
            'center': float(centers[j]),
            'n': int(n_pts),
            'kurtosis_mean': float(np.mean(kurt_samples)),
            'kurtosis_median': float(np.median(kurt_samples)),
            'kurtosis_ci_lo': float(np.percentile(kurt_samples, 2.5)),
            'kurtosis_ci_hi': float(np.percentile(kurt_samples, 97.5)),
            'kurtosis_std': float(np.std(kurt_samples)),
        })

    return results


# ============================================================
# DERIVATIVE ANALYSIS
# ============================================================

def numerical_derivative(x, y):
    """Central difference derivative. Forward/backward at edges."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dy = np.zeros_like(y)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(y) - 1):
        dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return dy


def find_zero_crossings(x, y, direction=None):
    """Find x-values where y crosses zero (linear interpolation).

    Parameters:
        x, y: arrays
        direction: None (all crossings), 'pos_to_neg' (peaks), 'neg_to_pos' (troughs)

    Returns: list of x-values at crossings
    """
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            if direction == 'pos_to_neg' and y[i] < 0:
                continue
            if direction == 'neg_to_pos' and y[i] > 0:
                continue
            x_cross = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            crossings.append(float(x_cross))
    return crossings


def find_inversion_point(log_gbar, log_gobs, bin_width=0.30, offset=0.0,
                         min_pts=10, a0=g_dagger):
    """Find the scatter derivative zero-crossing nearest to g†.

    The 'inversion point' is where dσ/d(log g_bar) = 0 (local extremum in scatter).
    BEC predicts this at g†.

    Parameters:
        log_gbar, log_gobs: RAR data arrays
        bin_width: acceleration bin width in dex
        offset: phase shift for bin edges
        min_pts: minimum points per bin
        a0: acceleration scale

    Returns:
        (nearest_crossing, all_crossings, bin_centers, bin_sigmas)
        nearest_crossing is None if no crossing found.
    """
    resid = rar_residuals(log_gbar, log_gobs, a0)
    lo = log_gbar.min() + offset
    hi = log_gbar.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    if len(edges) < 3:
        return None, [], np.array([]), np.array([])

    centers = []
    sigmas = []
    for j in range(len(edges) - 1):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j + 1])
        if np.sum(mask) >= min_pts:
            centers.append(0.5 * (edges[j] + edges[j + 1]))
            sigmas.append(np.std(resid[mask]))

    if len(centers) < 4:
        return None, [], np.array(centers), np.array(sigmas)

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    dsigma = numerical_derivative(centers, sigmas)
    crossings = find_zero_crossings(centers, dsigma, direction='pos_to_neg')

    if not crossings:
        return None, crossings, centers, sigmas

    nearest_idx = np.argmin(np.abs(np.array(crossings) - LOG_G_DAGGER))
    return crossings[nearest_idx], crossings, centers, sigmas


# ============================================================
# ΛCDM MOCK GALAXY GENERATOR
# ============================================================

def generate_lcdm_mock(n_gal=500, n_radii=20, seed=42, add_noise=True):
    """Generate semi-analytic ΛCDM mock RAR data.

    Uses NFW halos (Dutton & Macciò 2014 c-M), Moster+2013 SMHM,
    exponential disks (Kravtsov 2013 scaling), gas fraction scaling.

    Parameters:
        n_gal: number of mock galaxies
        n_radii: radial points per galaxy
        seed: random seed
        add_noise: if True, add realistic observational noise

    Returns:
        (log_gbar, log_gobs) arrays
    """
    rng = np.random.RandomState(seed)

    # Halo masses (log-uniform, matching SPARC range)
    log_Mhalo = rng.uniform(10.0, 13.0, n_gal)
    M_halo = 10**log_Mhalo

    # Concentration-mass: Dutton & Macciò 2014
    log_c200 = (0.905 - 0.101 * (log_Mhalo - 12.0 + np.log10(h_HUBBLE))
                + rng.normal(0, 0.11, n_gal))
    c200 = 10**log_c200

    # Virial radius
    rho_200 = 200.0 * RHO_CRIT
    R200_kpc = (3.0 * M_halo / (4.0 * np.pi * rho_200))**(1.0 / 3.0) * 1000.0
    r_s = R200_kpc / c200

    # Stellar masses: Moster+2013 SMHM
    M1 = 10**11.590
    ratio = M_halo / M1
    f_star = 2.0 * 0.0351 / (ratio**(-1.376) + ratio**0.608)
    log_Mstar = np.log10(f_star * M_halo) + rng.normal(0, 0.15, n_gal)
    M_star = 10**log_Mstar

    # Gas masses
    log_fgas = np.clip(-0.5 * (log_Mstar - 9.0) + 0.3
                       + rng.normal(0, 0.3, n_gal), -1.5, 1.5)
    M_gas = 10**log_fgas * M_star

    # Disk scale lengths
    R_d = np.clip(10**(np.log10(0.015 * R200_kpc) + rng.normal(0, 0.2, n_gal)),
                  0.3, 30.0)
    R_gas = 2.0 * R_d

    all_log_gbar = []
    all_log_gobs = []

    for i in range(n_gal):
        r_min = max(0.5 * R_d[i], 0.3)
        r_max = min(10.0 * R_d[i], R200_kpc[i] * 0.15)
        if r_max <= r_min:
            r_max = r_min * 10.0
        radii = np.linspace(r_min, r_max, n_radii)

        # Baryonic enclosed mass (exponential disk)
        y_s = radii / R_d[i]
        M_star_enc = M_star[i] * (1.0 - (1.0 + y_s) * np.exp(-y_s))
        y_g = radii / R_gas[i]
        M_gas_enc = M_gas[i] * (1.0 - (1.0 + y_g) * np.exp(-y_g))
        M_bar_enc = M_star_enc + M_gas_enc

        # NFW DM enclosed mass
        f_b = min((M_star[i] + M_gas[i]) / M_halo[i], 0.90)
        x = radii / r_s[i]
        x200 = c200[i]
        nfw_norm = np.log(1.0 + x200) - x200 / (1.0 + x200)
        M_DM_enc = (M_halo[i] * (1.0 - f_b) *
                    (np.log(1.0 + x) - x / (1.0 + x)) / nfw_norm)
        M_total_enc = M_DM_enc + M_bar_enc

        # Accelerations
        r_m = radii * KPC_M
        g_bar = G_SI * M_bar_enc * M_SUN / r_m**2
        g_obs = G_SI * M_total_enc * M_SUN / r_m**2

        if add_noise:
            log_noise = 2 * 0.10 / np.log(10)  # 10% velocity → ~0.087 dex
            log_gbar_n = (np.log10(np.maximum(g_bar, 1e-15))
                          + rng.normal(0, log_noise * 0.5, n_radii))
            log_gobs_n = (np.log10(np.maximum(g_obs, 1e-15))
                          + rng.normal(0, log_noise, n_radii))
        else:
            log_gbar_n = np.log10(np.maximum(g_bar, 1e-15))
            log_gobs_n = np.log10(np.maximum(g_obs, 1e-15))

        valid = ((log_gbar_n > -13.0) & (log_gbar_n < -8.0) &
                 (log_gobs_n > -13.0) & (log_gobs_n < -8.0))
        if np.sum(valid) >= 5:
            all_log_gbar.extend(log_gbar_n[valid])
            all_log_gobs.extend(log_gobs_n[valid])

    return np.array(all_log_gbar), np.array(all_log_gobs)


# ============================================================
# NFW AND DISK PROFILE FUNCTIONS
# ============================================================

def nfw_enclosed_mass(r_kpc, M200, c, R200):
    """NFW enclosed mass within radius r.

    Parameters:
        r_kpc: radius in kpc
        M200: virial mass in M_sun
        c: concentration
        R200: virial radius in kpc

    Returns: enclosed mass in M_sun
    """
    rs = R200 / c
    x = r_kpc / rs
    x200 = c
    nfw_norm = np.log(1.0 + x200) - x200 / (1.0 + x200)
    return M200 * (np.log(1.0 + x) - x / (1.0 + x)) / nfw_norm


def exponential_disk_enclosed_mass(r_kpc, M_total, Rd):
    """Enclosed mass of an exponential disk within radius r.

    M(<r) = M_total * [1 - (1 + r/Rd) * exp(-r/Rd)]
    """
    y = r_kpc / Rd
    return M_total * (1.0 - (1.0 + y) * np.exp(-y))


# ============================================================
# CONVENIENCE: get stat at g† from a binned profile
# ============================================================

def get_at_gdagger(profile, key='kurtosis', tol=0.15):
    """Extract a statistic from the bin containing g†.

    Parameters:
        profile: list of dicts from binned_stats() or bootstrap_kurtosis()
        key: which statistic to extract
        tol: maximum distance from g† in dex

    Returns: dict for the matching bin, or None
    """
    for b in profile:
        if abs(b['center'] - LOG_G_DAGGER) < tol:
            val = b.get(key, np.nan)
            if not np.isnan(val):
                return b
    return None
