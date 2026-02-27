#!/usr/bin/env python3
"""
GPP (Gross-Pitaevskii-Poisson) Inversion of the RAR
====================================================
Direct extraction of effective condensate parameters from observed RAR data.

Pipeline:
  1. g_DM(r) = g_obs - g_bar
  2. Spherical Poisson inversion -> rho_DM(r)
  3. Hydrostatic TF closure -> c_s^2(r)
  4. Universality tests across galaxies
  5. Thomas-Fermi regime diagnostics
  6. Boson mass estimate from (c_s^2, xi_eff)
  7. Figures + summary JSON
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu, norm, pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
G_SI = 6.674e-11       # m^3 kg^-1 s^-2
g_dag = 1.2e-10        # m/s^2
log_gdag = np.log10(g_dag)
hbar = 1.0546e-34      # J s
eV = 1.602e-19         # J
c_light = 3e8          # m/s
kpc_to_m = 3.086e19    # m

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "analysis" / "results"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

RAR_FILE = DATA / "rar_points_unified.csv"
GAL_FILE = DATA / "galaxy_results_unified.csv"
SPARC_META_FILE = DATA / "galaxy_results_sparc_orig_haubner.csv"

OUT_JSON = DATA / "summary_gpp_inversion.json"
OUT_GAL_CSV = DATA / "gpp_galaxy_cs2_summary.csv"
OUT_POINT_CSV = DATA / "gpp_pointwise_inversion.csv"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _odd_window(n: int, preferred: int = 5) -> int:
    if n < 3:
        return 1
    w = min(preferred, n)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def _savgol_safe(y: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = _odd_window(n, preferred=window)
    if w < 3 or n < 3:
        return y.copy()
    p = min(poly, w - 1)
    try:
        return savgol_filter(y, w, p)
    except Exception:
        return y.copy()


def numerical_derivative(x: np.ndarray, y: np.ndarray, sigma_y: np.ndarray | None = None):
    """
    Returns dy/dx and an approximate sigma(dy/dx) if sigma_y is provided.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        dy = np.gradient(y, x) if len(x) >= 2 else np.full_like(y, np.nan)
        return dy, np.full_like(dy, np.nan)

    y_s = _savgol_safe(y, window=5, poly=2)
    dy = np.gradient(y_s, x)

    if sigma_y is None:
        return dy, np.full_like(dy, np.nan)

    sy = np.asarray(sigma_y, dtype=float)
    fin = np.isfinite(sy)
    if not np.any(fin):
        return dy, np.full_like(dy, np.nan)
    sy = sy.copy()
    sy[~fin] = np.nanmedian(sy[fin])
    sy_s = _savgol_safe(np.abs(sy), window=5, poly=2)
    sigma_dy = np.abs(np.gradient(sy_s, x))
    return dy, sigma_dy


def bootstrap_median_error(values: np.ndarray, sigmas: np.ndarray, n_boot: int = 300, seed: int = 42) -> float:
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    good = np.isfinite(values)
    values = values[good]
    sigmas = sigmas[good]
    if len(values) < 3:
        return np.nan

    sigmas = np.where(np.isfinite(sigmas) & (sigmas > 0), sigmas, np.nan)
    if np.all(~np.isfinite(sigmas)):
        s_fallback = np.std(values) * 0.1
        sigmas = np.full_like(values, max(s_fallback, 1e-12))
    else:
        med_sig = np.nanmedian(sigmas[np.isfinite(sigmas)])
        sigmas = np.where(np.isfinite(sigmas), sigmas, med_sig)

    rng = np.random.default_rng(seed)
    draws = values[None, :] + rng.normal(0.0, sigmas[None, :], size=(n_boot, len(values)))
    meds = np.nanmedian(draws, axis=1)
    return float(np.nanstd(meds))


def _safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_data():
    pts = pd.read_csv(RAR_FILE)
    gal = pd.read_csv(GAL_FILE)

    req = [
        "galaxy",
        "source",
        "log_gbar",
        "log_gobs",
        "log_res",
        "sigma_log_gobs",
        "R_kpc",
        "env_dense",
        "logMh",
    ]
    missing = [c for c in req if c not in pts.columns]
    if missing:
        raise ValueError(f"Missing columns in {RAR_FILE}: {missing}")

    # Optional morphology for SPARC subset
    morph = None
    if SPARC_META_FILE.exists():
        sm = pd.read_csv(SPARC_META_FILE)
        if "galaxy" in sm.columns and "T" in sm.columns:
            morph = sm[["galaxy", "T"]].drop_duplicates("galaxy")

    if morph is not None:
        pts = pts.merge(morph, on="galaxy", how="left")
    else:
        pts["T"] = np.nan

    # Normalize source/env columns
    pts["source"] = pts["source"].astype(str)
    pts["env_dense"] = pts["env_dense"].astype(str).str.lower()

    print(f"Loaded {len(pts)} RAR points from {pts['galaxy'].nunique()} galaxies")
    print(f"Sources (top): {pts['source'].value_counts().head(8).to_dict()}")
    return pts, gal


# -----------------------------------------------------------------------------
# Step 1
# -----------------------------------------------------------------------------
def step1_extract_gdm(pts: pd.DataFrame):
    g_obs = 10.0 ** pts["log_gobs"].to_numpy(dtype=float)
    g_bar = 10.0 ** pts["log_gbar"].to_numpy(dtype=float)
    g_dm = g_obs - g_bar

    sigma_log = pd.to_numeric(pts["sigma_log_gobs"], errors="coerce").to_numpy(dtype=float)
    med_sigma_log = np.nanmedian(sigma_log[np.isfinite(sigma_log)]) if np.any(np.isfinite(sigma_log)) else 0.05
    sigma_log = np.where(np.isfinite(sigma_log), sigma_log, med_sigma_log)

    # First-order propagation: sigma_gobs = ln(10) * g_obs * sigma_log_gobs
    sigma_gobs = np.log(10.0) * g_obs * sigma_log
    sigma_gdm = sigma_gobs  # g_bar uncertainty not provided in this dataset

    out = pts.copy()
    out["g_obs"] = g_obs
    out["g_bar"] = g_bar
    out["g_dm"] = g_dm
    out["sigma_gdm"] = sigma_gdm
    out["R_m"] = pd.to_numeric(out["R_kpc"], errors="coerce") * kpc_to_m

    finite = np.isfinite(g_dm)
    n_neg = int(np.sum(g_dm[finite] <= 0))
    n_fin = int(np.sum(finite))
    frac_neg = n_neg / n_fin if n_fin > 0 else np.nan

    print("\nStep 1: g_DM extracted")
    print(f"  Finite g_DM: {n_fin}/{len(out)}")
    print(f"  Non-positive g_DM: {n_neg}/{n_fin} = {frac_neg:.1%}" if n_fin > 0 else "  No finite g_DM")

    return out, frac_neg


# -----------------------------------------------------------------------------
# Step 2
# -----------------------------------------------------------------------------
def step2_extract_density(pts: pd.DataFrame):
    results = []
    skipped = 0
    negative_rho_count = 0

    # Keep only finite, positive radius
    work = pts[np.isfinite(pts["R_m"]) & (pts["R_m"] > 0)].copy()

    for gal_name, group in work.groupby("galaxy"):
        g = group[group["g_dm"] > 0].copy()
        if len(g) < 5:
            skipped += 1
            continue

        g = g.sort_values("R_m")
        # Aggregate duplicate radii (rare but avoids gradient singularities)
        g = (
            g.groupby("R_m", as_index=False)
            .agg(
                {
                    "R_kpc": "median",
                    "g_dm": "median",
                    "sigma_gdm": "median",
                    "g_bar": "median",
                    "g_obs": "median",
                    "log_gbar": "median",
                    "env_dense": "first",
                    "logMh": "median",
                    "sigma_log_gobs": "median",
                    "source": "first",
                    "T": "median",
                }
            )
        )

        if len(g) < 5:
            skipped += 1
            continue

        r = g["R_m"].to_numpy(dtype=float)
        g_dm = g["g_dm"].to_numpy(dtype=float)
        sigma_gdm = np.abs(g["sigma_gdm"].to_numpy(dtype=float))

        r2g = r**2 * g_dm
        sigma_r2g = r**2 * sigma_gdm

        dr2g_dr, sigma_dr2g_dr = numerical_derivative(r, r2g, sigma_r2g)

        denom = 4.0 * np.pi * G_SI * r**2
        rho_dm = dr2g_dr / denom
        sigma_rho_dm = np.abs(sigma_dr2g_dr) / denom

        M_dm_enc = g_dm * r**2 / G_SI
        sigma_M_dm_enc = sigma_gdm * r**2 / G_SI

        neg_here = int(np.sum(np.isfinite(rho_dm) & (rho_dm <= 0)))
        negative_rho_count += neg_here

        for i in range(len(g)):
            results.append(
                {
                    "galaxy": gal_name,
                    "source": g["source"].iloc[i],
                    "R_m": r[i],
                    "R_kpc": g["R_kpc"].iloc[i],
                    "g_dm": g_dm[i],
                    "sigma_gdm": sigma_gdm[i],
                    "g_bar": g["g_bar"].iloc[i],
                    "g_obs": g["g_obs"].iloc[i],
                    "rho_dm": rho_dm[i],
                    "sigma_rho_dm": sigma_rho_dm[i],
                    "M_dm_enc": M_dm_enc[i],
                    "sigma_M_dm_enc": sigma_M_dm_enc[i],
                    "log_gbar": g["log_gbar"].iloc[i],
                    "env_dense": g["env_dense"].iloc[i],
                    "logMh": g["logMh"].iloc[i],
                    "sigma_log_gobs": g["sigma_log_gobs"].iloc[i],
                    "T": g["T"].iloc[i],
                }
            )

    df = pd.DataFrame(results)

    npts = len(df)
    frac_neg_rho = negative_rho_count / npts if npts > 0 else np.nan
    ngal = df["galaxy"].nunique() if npts > 0 else 0

    print(f"\nStep 2: rho_DM extracted for {ngal} galaxies ({npts} points)")
    print(f"  Skipped galaxies (insufficient positive g_DM points): {skipped}")
    if npts > 0:
        print(f"  Non-positive rho_DM: {negative_rho_count}/{npts} = {frac_neg_rho:.1%}")
    else:
        print("  No valid points after inversion")

    return df


# -----------------------------------------------------------------------------
# Step 3
# -----------------------------------------------------------------------------
def step3_extract_cs2(df: pd.DataFrame):
    results = []

    for gal_name, group in df.groupby("galaxy"):
        g = group.sort_values("R_m")
        # Need positive density for ln(rho)
        g = g[np.isfinite(g["rho_dm"]) & (g["rho_dm"] > 0) & np.isfinite(g["g_dm"]) & (g["g_dm"] > 0)]
        if len(g) < 5:
            continue

        r = g["R_m"].to_numpy(dtype=float)
        rho = g["rho_dm"].to_numpy(dtype=float)
        sigma_rho = np.abs(g["sigma_rho_dm"].to_numpy(dtype=float))
        g_dm = g["g_dm"].to_numpy(dtype=float)
        sigma_gdm = np.abs(g["sigma_gdm"].to_numpy(dtype=float))
        M_enc = g["M_dm_enc"].to_numpy(dtype=float)

        ln_rho = np.log(rho)
        sigma_ln_rho = np.where(rho > 0, sigma_rho / np.maximum(rho, 1e-300), np.nan)
        dln_rho_dr, sigma_dln_rho_dr = numerical_derivative(r, ln_rho, sigma_ln_rho)

        for i in range(len(g)):
            dln = dln_rho_dr[i]
            sigma_dln = sigma_dln_rho_dr[i]

            # Physical sign condition: density should decrease outward.
            if np.isfinite(dln) and dln < 0 and np.abs(dln) > 1e-30:
                cs2 = g_dm[i] / (-dln)
                frac_g = sigma_gdm[i] / max(g_dm[i], 1e-30)
                frac_d = sigma_dln / max(abs(dln), 1e-30) if np.isfinite(sigma_dln) else np.nan
                if np.isfinite(frac_d):
                    sigma_cs2 = cs2 * np.sqrt(frac_g**2 + frac_d**2)
                else:
                    sigma_cs2 = cs2 * frac_g
            else:
                cs2 = np.nan
                sigma_cs2 = np.nan

            xi_eff = np.sqrt(G_SI * M_enc[i] / g_dag) if M_enc[i] > 0 else np.nan
            X_TF = r[i] / xi_eff if np.isfinite(xi_eff) and xi_eff > 0 else np.nan
            g_star = cs2 / xi_eff if np.isfinite(cs2) and np.isfinite(xi_eff) and xi_eff > 0 else np.nan

            results.append(
                {
                    "galaxy": gal_name,
                    "source": g["source"].iloc[i],
                    "R_m": r[i],
                    "R_kpc": g["R_kpc"].iloc[i],
                    "g_dm": g_dm[i],
                    "sigma_gdm": sigma_gdm[i],
                    "rho_dm": rho[i],
                    "sigma_rho_dm": sigma_rho[i],
                    "M_dm_enc": M_enc[i],
                    "cs2": cs2,
                    "sigma_cs2": sigma_cs2,
                    "xi_eff": xi_eff,
                    "X_TF": X_TF,
                    "g_star": g_star,
                    "log_gbar": g["log_gbar"].iloc[i],
                    "env_dense": g["env_dense"].iloc[i],
                    "logMh": g["logMh"].iloc[i],
                    "sigma_log_gobs": g["sigma_log_gobs"].iloc[i],
                    "T": g["T"].iloc[i],
                }
            )

    df_cs = pd.DataFrame(results)
    valid = np.isfinite(df_cs.get("cs2", np.array([]))) & (df_cs.get("cs2", np.array([])) > 0)
    df_valid = df_cs[valid].copy() if len(df_cs) > 0 else df_cs.copy()

    print("\nStep 3: c_s^2 extracted")
    if len(df_cs) == 0:
        print("  No point survived Step 3")
        return df_cs, df_valid

    frac = len(df_valid) / len(df_cs)
    print(f"  Valid c_s^2 points: {len(df_valid)}/{len(df_cs)} = {frac:.1%}")
    print(f"  Galaxies with valid c_s^2: {df_valid['galaxy'].nunique()}")

    if len(df_valid) > 0:
        log_cs2 = np.log10(df_valid["cs2"].to_numpy(dtype=float))
        print(f"  log10(c_s^2) range: [{np.nanmin(log_cs2):.2f}, {np.nanmax(log_cs2):.2f}]")
        print(f"  Median c_s^2 = {np.nanmedian(df_valid['cs2']):.3e} m^2/s^2")

    return df_cs, df_valid


# -----------------------------------------------------------------------------
# Step 4
# -----------------------------------------------------------------------------
def step4_test_universality(df_valid: pd.DataFrame):
    gal_rows = []
    rng = np.random.default_rng(42)

    for gal_name, g in df_valid.groupby("galaxy"):
        if len(g) < 5:
            continue

        vals = g["cs2"].to_numpy(dtype=float)
        errs = g["sigma_cs2"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        errs = errs[np.isfinite(vals)] if len(errs) == len(g) else np.full_like(vals, np.nan)

        if len(vals) < 5:
            continue

        cs2_median = float(np.median(vals))
        cs2_mean = float(np.mean(vals))
        cs2_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        cs2_median_err = bootstrap_median_error(vals, errs if len(errs) == len(vals) else np.full_like(vals, np.nan), n_boot=300, seed=int(rng.integers(1e9)))

        gal_rows.append(
            {
                "galaxy": gal_name,
                "cs2_median": cs2_median,
                "cs2_mean": cs2_mean,
                "cs2_std": cs2_std,
                "cs2_median_err": cs2_median_err,
                "n_points": int(len(vals)),
                "logMh": _safe_float(g["logMh"].median()),
                "env_dense": str(g["env_dense"].iloc[0]).lower(),
                "source": str(g["source"].iloc[0]),
                "T": _safe_float(g["T"].median()),
            }
        )

    gal_cs2 = pd.DataFrame(gal_rows)
    if len(gal_cs2) == 0:
        raise RuntimeError("No galaxies with >=5 valid c_s^2 points")

    gal_cs2["log_cs2"] = np.log10(gal_cs2["cs2_median"])
    gal_cs2["log_cs2_err"] = gal_cs2["cs2_median_err"] / np.maximum(gal_cs2["cs2_median"] * np.log(10.0), 1e-30)

    n_gal = len(gal_cs2)
    log_vals = gal_cs2["log_cs2"].to_numpy(dtype=float)
    cs2_med = float(np.median(gal_cs2["cs2_median"]))
    cs2_std_dex = float(np.std(log_vals, ddof=1)) if len(log_vals) > 1 else np.nan
    cs2_mean_dex = float(np.mean(log_vals))

    print("\nStep 4: Universality test")
    print(f"  Galaxies with N>=5 valid c_s^2 points: {n_gal}")
    print(f"  Median c_s^2 = {cs2_med:.3e} m^2/s^2")
    print(f"  log10(c_s^2): mean={cs2_mean_dex:.3f}, std={cs2_std_dex:.3f} dex")

    # Mass trend
    m = np.isfinite(gal_cs2["logMh"]) & np.isfinite(gal_cs2["log_cs2"])
    if int(np.sum(m)) > 10:
        x = gal_cs2.loc[m, "logMh"].to_numpy(dtype=float)
        y = gal_cs2.loc[m, "log_cs2"].to_numpy(dtype=float)
        yerr = gal_cs2.loc[m, "log_cs2_err"].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nanmedian(yerr[np.isfinite(yerr) & (yerr > 0)]) if np.any(np.isfinite(yerr) & (yerr > 0)) else 0.1)
        w = 1.0 / np.maximum(yerr, 1e-4)
        slope, intercept = np.polyfit(x, y, 1, w=w)
        r_mass, p_mass = pearsonr(x, y)

        # Bootstrap slope error
        rng_bs = np.random.default_rng(123)
        bs = []
        for _ in range(1200):
            idx = rng_bs.integers(0, len(x), len(x))
            try:
                b = np.polyfit(x[idx], y[idx], 1, w=w[idx])[0]
                if np.isfinite(b):
                    bs.append(b)
            except Exception:
                continue
        slope_err = float(np.std(bs)) if len(bs) > 10 else np.nan
        print(f"  Mass correlation: r={r_mass:.3f}, p={p_mass:.3e}")
        print(f"  Slope: {slope:.4f} +/- {slope_err:.4f} dex/dex")
    else:
        slope = np.nan
        intercept = np.nan
        slope_err = np.nan
        r_mass = np.nan
        p_mass = np.nan

    # Environment test
    env = gal_cs2["env_dense"].astype(str).str.lower()
    field = gal_cs2[env == "field"]["log_cs2"].dropna()
    dense = gal_cs2[env == "dense"]["log_cs2"].dropna()
    if len(field) > 3 and len(dense) > 3:
        _, p_env = mannwhitneyu(field, dense, alternative="two-sided")
        print(f"  Field median log c_s^2 = {np.median(field):.3f} ({len(field)} gal)")
        print(f"  Dense median log c_s^2 = {np.median(dense):.3f} ({len(dense)} gal)")
        print(f"  Environment Mann-Whitney p = {p_env:.3e}")
    else:
        p_env = np.nan

    # Morphology test (if available)
    mt = np.isfinite(gal_cs2["T"]) & np.isfinite(gal_cs2["log_cs2"])
    if int(np.sum(mt)) > 10:
        rho_T, p_T = spearmanr(gal_cs2.loc[mt, "T"], gal_cs2.loc[mt, "log_cs2"])
    else:
        rho_T, p_T = np.nan, np.nan

    # Universality verdict
    if np.isfinite(cs2_std_dex) and cs2_std_dex <= 0.20 and (np.isnan(slope) or abs(slope) <= 0.05):
        verdict = "CONFIRMED"
    elif np.isfinite(cs2_std_dex) and cs2_std_dex <= 0.35:
        verdict = "MARGINAL"
    else:
        verdict = "REJECTED"

    print(f"  -> Universality: {verdict}")

    return gal_cs2, {
        "n_galaxies": int(n_gal),
        "cs2_median": float(cs2_med),
        "cs2_mean_dex": float(cs2_mean_dex),
        "cs2_std_dex": float(cs2_std_dex),
        "mass_slope": _safe_float(slope),
        "mass_slope_err": _safe_float(slope_err),
        "mass_r": _safe_float(r_mass),
        "mass_p": _safe_float(p_mass),
        "env_p": _safe_float(p_env),
        "morph_rho": _safe_float(rho_T),
        "morph_p": _safe_float(p_T),
        "universality": verdict,
        "fit_intercept": _safe_float(intercept),
    }


# -----------------------------------------------------------------------------
# Step 4b (TF restricted)
# -----------------------------------------------------------------------------
def step4b_cs2_TF_restricted(df_valid: pd.DataFrame, X_threshold: float = 3.0):
    df_tf = df_valid[np.isfinite(df_valid["X_TF"]) & (df_valid["X_TF"] > X_threshold)].copy()
    if len(df_tf) == 0:
        print(f"\n  TF-restricted (X>{X_threshold}): no points")
        return None

    rows = []
    for gal_name, g in df_tf.groupby("galaxy"):
        vals = g["cs2"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) < 3:
            continue
        rows.append({"galaxy": gal_name, "cs2_median": np.median(vals), "n_points": len(vals), "logMh": np.median(g["logMh"])})

    if len(rows) < 3:
        print(f"\n  TF-restricted (X>{X_threshold}): too few galaxies")
        return None

    gt = pd.DataFrame(rows)
    gt["log_cs2"] = np.log10(gt["cs2_median"])
    med = float(np.median(gt["cs2_median"]))
    std = float(np.std(gt["log_cs2"], ddof=1)) if len(gt) > 1 else np.nan

    print(f"\n  TF-restricted (X>{X_threshold}): {len(gt)} galaxies, median c_s^2={med:.3e}, scatter={std:.3f} dex")
    return {"n_gal": int(len(gt)), "cs2_median": med, "cs2_std_dex": std}


# -----------------------------------------------------------------------------
# Step 5
# -----------------------------------------------------------------------------
def step5_test_thomas_fermi(df_cs: pd.DataFrame):
    valid = np.isfinite(df_cs["X_TF"]) & (df_cs["X_TF"] > 0)
    X_vals = df_cs.loc[valid, "X_TF"].to_numpy(dtype=float)

    n_total = int(len(X_vals))
    n_TF = int(np.sum(X_vals > 1.0)) if n_total > 0 else 0
    n_quantum = int(np.sum(X_vals <= 1.0)) if n_total > 0 else 0
    n_TF_strict = int(np.sum(X_vals > 3.0)) if n_total > 0 else 0

    frac_TF = n_TF / n_total if n_total > 0 else np.nan
    frac_quantum = n_quantum / n_total if n_total > 0 else np.nan
    med_X = float(np.median(X_vals)) if n_total > 0 else np.nan

    print("\nStep 5: Thomas-Fermi regime test")
    print(f"  Total valid points: {n_total}")
    if n_total > 0:
        print(f"  TF regime (X>1): {n_TF} ({frac_TF:.1%})")
        print(f"  Quantum regime (X<=1): {n_quantum} ({frac_quantum:.1%})")
        print(f"  Strict TF (X>3): {n_TF_strict} ({n_TF_strict/n_total:.1%})")
        print(f"  Median X = {med_X:.2f}")

    return {
        "n_total": n_total,
        "n_TF": n_TF,
        "n_quantum": n_quantum,
        "n_TF_strict": n_TF_strict,
        "frac_TF": frac_TF,
        "median_X": med_X,
    }


# -----------------------------------------------------------------------------
# Step 6
# -----------------------------------------------------------------------------
def step6_boson_mass(gal_cs2: pd.DataFrame, df_valid: pd.DataFrame):
    cs2_samples = gal_cs2["cs2_median"].to_numpy(dtype=float)
    cs2_samples = cs2_samples[np.isfinite(cs2_samples) & (cs2_samples > 0)]

    xi_samples = df_valid["xi_eff"].to_numpy(dtype=float)
    xi_samples = xi_samples[np.isfinite(xi_samples) & (xi_samples > 0)]

    if len(cs2_samples) == 0 or len(xi_samples) == 0:
        raise RuntimeError("Insufficient data for boson mass estimate")

    C_med = float(np.median(cs2_samples))
    xi_med = float(np.median(xi_samples))

    m_kg = hbar / (xi_med * np.sqrt(2.0 * C_med))
    m_eV = float(m_kg * c_light**2 / eV)

    # Bootstrap 95% CI
    rng = np.random.default_rng(777)
    n_boot = 5000
    C_draw = rng.choice(cs2_samples, size=n_boot, replace=True)
    xi_draw = rng.choice(xi_samples, size=n_boot, replace=True)
    m_draw = hbar / (xi_draw * np.sqrt(2.0 * C_draw)) * c_light**2 / eV
    m_draw = m_draw[np.isfinite(m_draw) & (m_draw > 0)]

    if len(m_draw) > 20:
        m_lo, m_hi = np.percentile(m_draw, [2.5, 97.5])
    else:
        m_lo, m_hi = np.nan, np.nan

    lyman_ok = bool(m_eV > 1e-21)

    print("\nStep 6: Boson mass constraint")
    print(f"  Median xi_eff = {xi_med:.3e} m = {xi_med/kpc_to_m:.2f} kpc")
    print(f"  Median c_s^2 = {C_med:.3e} m^2/s^2")
    print(f"  Boson mass = {m_eV:.3e} eV/c^2")
    if np.isfinite(m_lo):
        print(f"  95% CI = [{m_lo:.3e}, {m_hi:.3e}] eV/c^2")
    print(f"  Lyman-alpha compatible (m>1e-21 eV): {lyman_ok}")

    return {
        "xi_median_kpc": float(xi_med / kpc_to_m),
        "xi_16_kpc": float(np.percentile(xi_samples, 16) / kpc_to_m),
        "xi_84_kpc": float(np.percentile(xi_samples, 84) / kpc_to_m),
        "boson_mass_eV": m_eV,
        "boson_mass_CI95": [float(m_lo), float(m_hi)] if np.isfinite(m_lo) else [None, None],
        "lyman_alpha_compatible": lyman_ok,
    }


# -----------------------------------------------------------------------------
# Step 7
# -----------------------------------------------------------------------------
def step7_plots(df_valid: pd.DataFrame, gal_cs2: pd.DataFrame, df_cs: pd.DataFrame, tf_info: dict, univ_results: dict):
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # A) c_s^2(r) for representative galaxies
    ax_a = fig.add_subplot(gs[0, 0])
    gmh = gal_cs2[np.isfinite(gal_cs2["logMh"])].sort_values("logMh")

    rep_galaxies = []
    if len(gmh) >= 12:
        qs = np.linspace(0, 1, 5)
        for i in range(4):
            sub = gmh.iloc[int(qs[i] * len(gmh)): int(qs[i + 1] * len(gmh))]
            sub = sub.nlargest(3, "n_points")
            rep_galaxies.extend(sub["galaxy"].tolist())
        rep_galaxies = rep_galaxies[:12]
    else:
        rep_galaxies = gmh["galaxy"].tolist()[:12]

    if len(rep_galaxies) > 0:
        cmap = plt.cm.viridis
        mh_min = np.nanmin(gal_cs2["logMh"]) if np.any(np.isfinite(gal_cs2["logMh"])) else 10
        mh_max = np.nanmax(gal_cs2["logMh"]) if np.any(np.isfinite(gal_cs2["logMh"])) else 13
        denom = max(mh_max - mh_min, 1e-6)

        for gal_name in rep_galaxies:
            gd = df_valid[df_valid["galaxy"] == gal_name].sort_values("R_kpc")
            if len(gd) < 3:
                continue
            mh = gd["logMh"].iloc[0]
            if not np.isfinite(mh):
                mh = (mh_min + mh_max) / 2
            col = cmap((mh - mh_min) / denom)
            ax_a.semilogy(gd["R_kpc"], gd["cs2"], "-o", color=col, markersize=2, linewidth=0.8, alpha=0.75)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(mh_min, mh_max))
        plt.colorbar(sm, ax=ax_a, label="log M_h")

    ax_a.axhline(univ_results["cs2_median"], color="red", ls="--", lw=1.3, label="Global median")
    ax_a.set_xlabel("R [kpc]")
    ax_a.set_ylabel("c_s^2 [m^2/s^2]")
    ax_a.set_title("A: c_s^2(r) representative galaxies")
    ax_a.legend(fontsize=8)

    # B) median c_s^2 vs logMh
    ax_b = fig.add_subplot(gs[0, 1])
    m = np.isfinite(gal_cs2["logMh"]) & np.isfinite(gal_cs2["log_cs2"])
    if np.sum(m) > 0:
        x = gal_cs2.loc[m, "logMh"].to_numpy(dtype=float)
        y = gal_cs2.loc[m, "log_cs2"].to_numpy(dtype=float)
        yerr = gal_cs2.loc[m, "log_cs2_err"].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.08)
        sc = ax_b.scatter(x, y, c=gal_cs2.loc[m, "n_points"], cmap="plasma", s=18, alpha=0.7, edgecolors="none")
        ax_b.errorbar(x, y, yerr=yerr, fmt="none", ecolor="gray", alpha=0.35, lw=0.7)
        plt.colorbar(sc, ax=ax_b, label="N points")

        if np.isfinite(univ_results["mass_slope"]):
            xfit = np.linspace(np.min(x), np.max(x), 100)
            yfit = univ_results["fit_intercept"] + univ_results["mass_slope"] * xfit
            ax_b.plot(xfit, yfit, "r-", lw=1.8, label=f"slope={univ_results['mass_slope']:.3f}")

    ax_b.set_xlabel("log M_h [M_sun]")
    ax_b.set_ylabel("log c_s^2 [m^2/s^2]")
    ax_b.set_title("B: Galaxy-median c_s^2 vs halo mass")
    ax_b.legend(fontsize=8)

    # C) histogram of galaxy-median c_s^2
    ax_c = fig.add_subplot(gs[1, 0])
    lv = gal_cs2["log_cs2"].to_numpy(dtype=float)
    lv = lv[np.isfinite(lv)]
    if len(lv) > 0:
        ax_c.hist(lv, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="navy")
        if len(lv) > 5:
            mu, sig = norm.fit(lv)
            xg = np.linspace(np.min(lv), np.max(lv), 250)
            ax_c.plot(xg, norm.pdf(xg, mu, sig), "r-", lw=2, label=f"mu={mu:.2f}, sigma={sig:.2f}")
            ax_c.axvline(mu, color="red", ls=":", alpha=0.6)
            ax_c.legend(fontsize=8)
    ax_c.set_xlabel("log c_s^2 [m^2/s^2]")
    ax_c.set_ylabel("Density")
    ax_c.set_title("C: Distribution of galaxy-median c_s^2")

    # D) X = r/xi_eff distribution
    ax_d = fig.add_subplot(gs[1, 1])
    X = df_cs["X_TF"].to_numpy(dtype=float)
    X = X[np.isfinite(X) & (X > 0)]
    if len(X) > 0:
        lx = np.log10(X)
        ax_d.hist(lx, bins=50, density=True, alpha=0.75, color="forestgreen", edgecolor="darkgreen")
        ax_d.axvline(0.0, color="red", ls="--", lw=1.7, label="X=1")
        ax_d.axvline(np.log10(3.0), color="orange", ls=":", lw=1.4, label="X=3")
        if np.isfinite(tf_info["frac_TF"]):
            ax_d.text(
                0.98,
                0.95,
                f"TF (X>1): {tf_info['frac_TF']:.0%}\nQuantum: {1 - tf_info['frac_TF']:.0%}",
                transform=ax_d.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    ax_d.set_xlabel("log10(X) = log10(r/xi_eff)")
    ax_d.set_ylabel("Density")
    ax_d.set_title("D: Thomas-Fermi parameter distribution")
    ax_d.legend(fontsize=8)

    plt.suptitle("GPP Inversion: Sound-speed extraction from unified RAR", fontsize=14, y=1.01)
    out_main = FIG / "gpp_inversion_cs2.png"
    fig.savefig(out_main, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots saved to {out_main}")

    # Additional systematics plot: SPARC vs WALLABY only
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    src = gal_cs2["source"].astype(str)
    sparc = gal_cs2[src.str.upper() == "SPARC"]["log_cs2"].dropna()
    wallaby = gal_cs2[src.str.upper().str.startswith("WALLABY")]["log_cs2"].dropna()

    if len(sparc) > 0:
        ax.hist(sparc, bins=20, alpha=0.55, density=True, label=f"SPARC (N={len(sparc)})", color="royalblue")
    if len(wallaby) > 0:
        ax.hist(wallaby, bins=20, alpha=0.55, density=True, label=f"WALLABY (N={len(wallaby)})", color="coral")

    ax.set_xlabel("log c_s^2 [m^2/s^2]")
    ax.set_ylabel("Density")
    ax.set_title("Systematics: SPARC vs WALLABY")
    ax.legend()
    out_sys = FIG / "gpp_inversion_systematics.png"
    fig2.tight_layout()
    fig2.savefig(out_sys, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Systematics plot saved to {out_sys}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_all():
    print("=" * 72)
    print("GPP INVERSION OF THE RADIAL ACCELERATION RELATION")
    print("=" * 72)

    pts, _ = load_data()

    pts, frac_neg_gdm = step1_extract_gdm(pts)
    df_rho = step2_extract_density(pts)
    df_cs, df_valid = step3_extract_cs2(df_rho)

    if len(df_valid) == 0:
        raise RuntimeError("No valid c_s^2 points after Step 3")

    gal_cs2, univ = step4_test_universality(df_valid)
    tf_restricted = step4b_cs2_TF_restricted(df_valid, X_threshold=3.0)
    tf_info = step5_test_thomas_fermi(df_cs)
    boson = step6_boson_mass(gal_cs2, df_valid)

    step7_plots(df_valid, gal_cs2, df_cs, tf_info, univ)

    # SPARC vs WALLABY systematic table
    src_upper = gal_cs2["source"].astype(str).str.upper()
    sparc_sub = gal_cs2[src_upper == "SPARC"]
    wallaby_sub = gal_cs2[src_upper.str.startswith("WALLABY")]

    def sub_stats(sub):
        if len(sub) == 0:
            return None
        return {
            "n_galaxies": int(len(sub)),
            "median_log_cs2": float(np.median(sub["log_cs2"])),
            "std_log_cs2": float(np.std(sub["log_cs2"], ddof=1)) if len(sub) > 1 else np.nan,
            "median_cs2": float(np.median(sub["cs2_median"])),
        }

    sparc_stats = sub_stats(sparc_sub)
    wallaby_stats = sub_stats(wallaby_sub)

    # Verdict
    if univ["universality"] == "CONFIRMED" and boson["lyman_alpha_compatible"]:
        verdict = (
            f"c_s^2 universality is supported (scatter={univ['cs2_std_dex']:.2f} dex) with weak mass trend; "
            f"implied boson mass m~{boson['boson_mass_eV']:.2e} eV is Lyman-alpha compatible."
        )
    elif univ["universality"] == "MARGINAL":
        verdict = (
            "c_s^2 universality is marginal; differentiation noise and TF-breakdown points still contribute appreciable scatter."
        )
    else:
        verdict = (
            "c_s^2 varies strongly across galaxies; either TF assumptions fail broadly or coupling is not universal in this form."
        )

    summary = {
        "test": "GPP_inversion_cs2_universality",
        "n_galaxies": int(univ["n_galaxies"]),
        "n_points_total": int(len(pts)),
        "n_points_valid_cs2": int(len(df_valid)),
        "n_points_TF_regime": int(tf_info["n_TF"]),
        "n_points_quantum_regime": int(tf_info["n_quantum"]),
        "frac_negative_gDM": _safe_float(frac_neg_gdm),
        "cs2_median": _safe_float(univ["cs2_median"]),
        "cs2_std_dex": _safe_float(univ["cs2_std_dex"]),
        "cs2_mass_slope": _safe_float(univ["mass_slope"]),
        "cs2_mass_slope_err": _safe_float(univ["mass_slope_err"]),
        "cs2_mass_slope_pvalue": _safe_float(univ["mass_p"]),
        "cs2_universality": univ["universality"],
        "boson_mass_eV": _safe_float(boson["boson_mass_eV"]),
        "boson_mass_CI95": boson["boson_mass_CI95"],
        "lyman_alpha_compatible": bool(boson["lyman_alpha_compatible"]),
        "TF_regime_fraction": _safe_float(tf_info["frac_TF"]),
        "TF_median_X": _safe_float(tf_info["median_X"]),
        "TF_restricted_X3": tf_restricted,
        "morphology_spearman_rho": _safe_float(univ["morph_rho"]),
        "morphology_spearman_p": _safe_float(univ["morph_p"]),
        "sparc_systematics": sparc_stats,
        "wallaby_systematics": wallaby_stats,
        "verdict": verdict,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    gal_cs2.to_csv(OUT_GAL_CSV, index=False)
    df_valid.to_csv(OUT_POINT_CSV, index=False)

    print(f"\nSummary saved to {OUT_JSON}")
    print(f"Galaxy summary saved to {OUT_GAL_CSV}")
    print(f"Pointwise inversion saved to {OUT_POINT_CSV}")

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for k in [
        "n_galaxies",
        "n_points_total",
        "n_points_valid_cs2",
        "n_points_TF_regime",
        "n_points_quantum_regime",
        "cs2_median",
        "cs2_std_dex",
        "cs2_mass_slope",
        "cs2_mass_slope_pvalue",
        "cs2_universality",
        "boson_mass_eV",
        "boson_mass_CI95",
        "lyman_alpha_compatible",
        "TF_regime_fraction",
    ]:
        print(f"  {k}: {summary.get(k)}")
    print(f"\n  VERDICT: {summary['verdict']}")

    return summary


if __name__ == "__main__":
    run_all()
