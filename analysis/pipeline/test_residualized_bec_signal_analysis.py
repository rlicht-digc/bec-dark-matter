#!/usr/bin/env python3
"""
Residualized BEC Signal Analysis
================================

Tests whether BEC/RAR-related signals strengthen after removing confounds.
Outputs:
  - summary_residualization.json
  - galaxy_residuals.csv
  - Plot A/B/C/D in output directory
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ANALYSIS_RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results")

G_DAGGER = 1.2e-10
ACCEL_SPLIT = -10.5
MIN_GALAXIES = 20
MIN_POINTS_ACF = 6
R2_DELTA_TOL = 0.01


@dataclass
class Paths:
    rar_points_unified: str
    galaxy_results_unified: str
    galaxy_results_sparc: str
    rar_points_wallaby: str
    output_dir: str
    output_dir_fallback_used: bool


def resolve_input_path(preferred_path, fallback_path):
    if os.path.exists(preferred_path):
        return preferred_path
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Could not find file at {preferred_path} or {fallback_path}")


def resolve_output_dir():
    preferred = os.path.join(ANALYSIS_RESULTS_DIR, "residualization")
    fallback = "/mnt/user-data/outputs/residualization"

    try:
        os.makedirs(preferred, exist_ok=True)
        testfile = os.path.join(preferred, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return preferred, False
    except Exception:
        os.makedirs(fallback, exist_ok=True)
        return fallback, True


def build_paths():
    paths = Paths(
        rar_points_unified=resolve_input_path(
            os.path.join(ANALYSIS_RESULTS_DIR, "rar_points_unified.csv"),
            "/mnt/project/rar_points_unified.csv",
        ),
        galaxy_results_unified=resolve_input_path(
            os.path.join(ANALYSIS_RESULTS_DIR, "galaxy_results_unified.csv"),
            "/mnt/project/galaxy_results_unified.csv",
        ),
        galaxy_results_sparc=resolve_input_path(
            os.path.join(ANALYSIS_RESULTS_DIR, "galaxy_results_sparc_orig_haubner.csv"),
            "/mnt/project/galaxy_results_sparc_orig_haubner.csv",
        ),
        rar_points_wallaby=resolve_input_path(
            os.path.join(ANALYSIS_RESULTS_DIR, "rar_points_wallaby_hubble_nodesi.csv"),
            "/mnt/project/rar_points_wallaby_hubble_nodesi.csv",
        ),
        output_dir="",
        output_dir_fallback_used=False,
    )
    outdir, fallback_used = resolve_output_dir()
    paths.output_dir = outdir
    paths.output_dir_fallback_used = fallback_used
    return paths


def setup_plot_style():
    plt.style.use("dark_background")
    bg = "#0d1117"
    fg = "#c9d1d9"
    grid = "#30363d"
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "savefig.edgecolor": bg,
        "axes.edgecolor": fg,
        "axes.labelcolor": fg,
        "axes.titlecolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "grid.color": grid,
        "text.color": fg,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": grid,
    })


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def safe_mannwhitney(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def linear_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, np.nan
    x_aug = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    yhat = x_aug @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), float(beta[0]), float(beta[1])


def standardize_matrix(df, cols):
    x = df[cols].astype(float).copy()
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0, 1.0)
    xz = (x - means) / stds
    return xz.to_numpy(dtype=float), means.to_dict(), stds.to_dict()


def fit_ols(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_aug = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    yhat = x_aug @ beta
    resid = y - yhat
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, yhat, resid, float(r2)


def compute_base_tables(rar, gal_unified, gal_sparc):
    # Per-galaxy extent from point table
    point_agg = (
        rar.groupby("galaxy", as_index=False)
        .agg(
            Rext_kpc=("R_kpc", "max"),
            n_points_points=("R_kpc", "size"),
            mean_log_gbar_points=("log_gbar", "mean"),
            logMh_points=("logMh", "median"),
            env_dense_points=("env_dense", "first"),
        )
    )

    gal = gal_unified.copy()
    gal = gal.drop_duplicates(subset=["galaxy"])
    base = point_agg.merge(
        gal[["galaxy", "n_points", "mean_log_gbar", "logMh", "env_dense"]],
        on="galaxy",
        how="left",
    )

    base["n_points"] = base["n_points"].fillna(base["n_points_points"])
    base["mean_log_gbar"] = base["mean_log_gbar"].fillna(base["mean_log_gbar_points"])
    base["logMh"] = base["logMh"].fillna(base["logMh_points"])
    base["env_dense"] = base["env_dense"].fillna(base["env_dense_points"])

    sparc = gal_sparc[["galaxy", "sigma_D_dex", "Inc"]].drop_duplicates("galaxy")
    base = base.merge(sparc, on="galaxy", how="left")

    base["sigma_D_dex"] = base["sigma_D_dex"].fillna(0.10)
    base["Inc"] = base["Inc"].fillna(60.0)
    base["env_binary"] = (base["env_dense"].astype(str).str.lower() == "dense").astype(int)
    base["log_Rext"] = np.log10(base["Rext_kpc"].where(base["Rext_kpc"] > 0))
    base["log_npoints"] = np.log10(base["n_points"].where(base["n_points"] > 0))

    return base


def compute_t2_scatter_transition(rar):
    rows = []
    for galaxy, gdf in rar.groupby("galaxy"):
        low = gdf.loc[gdf["log_gbar"] < ACCEL_SPLIT, "log_res"].dropna()
        high = gdf.loc[gdf["log_gbar"] > ACCEL_SPLIT, "log_res"].dropna()
        if len(low) >= 3 and len(high) >= 3:
            sigma_low = float(np.std(low, ddof=1))
            sigma_high = float(np.std(high, ddof=1))
            rows.append({
                "galaxy": galaxy,
                "T2_scatter_transition": abs(sigma_low - sigma_high),
                "T2_sigma_low": sigma_low,
                "T2_sigma_high": sigma_high,
            })
    return pd.DataFrame(rows)


def compute_t3_acf_lag1(rar):
    rows = []
    for galaxy, gdf in rar.groupby("galaxy"):
        g = gdf[["R_kpc", "log_res"]].dropna().sort_values("R_kpc")
        if len(g) < MIN_POINTS_ACF:
            continue
        x0 = g["log_res"].to_numpy()[:-1]
        x1 = g["log_res"].to_numpy()[1:]
        if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
            continue
        r, _ = stats.pearsonr(x0, x1)
        rows.append({"galaxy": galaxy, "T3_acf_lag1": float(r)})
    return pd.DataFrame(rows)


def compute_t5_xi_coherence_ratio(rar):
    rows = []
    for galaxy, gdf in rar.groupby("galaxy"):
        g = gdf[["R_kpc", "log_gobs", "log_gbar"]].dropna()
        if len(g) < 3:
            continue
        med_r = float(np.median(g["R_kpc"]))
        gdm = (10.0 ** g["log_gobs"].to_numpy()) - (10.0 ** g["log_gbar"].to_numpy())
        gdm = gdm[np.isfinite(gdm) & (gdm > 0)]
        if len(gdm) < 3:
            continue
        med_gdm = float(np.median(gdm))
        if med_gdm <= 0:
            continue
        xi_eff = med_r * np.sqrt(med_gdm / G_DAGGER)
        if xi_eff <= 0:
            continue
        rows.append({
            "galaxy": galaxy,
            "T5_xi_coherence_ratio": float(med_r / xi_eff),
            "T5_xi_eff": float(xi_eff),
            "T5_median_gdm": med_gdm,
        })
    return pd.DataFrame(rows)


def analyze_galaxy_target(name, full_df, target_col, confound_cols):
    out = {
        "target": name,
        "n_valid": 0,
        "skipped": False,
        "skip_reason": None,
    }

    cols = list(dict.fromkeys(["galaxy", "logMh", "env_binary", "mean_log_gbar", target_col] + confound_cols))
    df = full_df[cols].dropna().copy()
    out["n_valid"] = int(len(df))
    if len(df) < MIN_GALAXIES:
        out["skipped"] = True
        out["skip_reason"] = f"fewer than {MIN_GALAXIES} galaxies after confound matching"
        out["data"] = None
        return out

    xz, means, stds = standardize_matrix(df, confound_cols)
    beta, yhat, resid, r2_conf = fit_ols(df[target_col].to_numpy(dtype=float), xz)
    df["target_resid"] = resid

    # Before/after tests
    rho_m_raw, p_m_raw = safe_spearman(df["logMh"], df[target_col])
    rho_m_res, p_m_res = safe_spearman(df["logMh"], df["target_resid"])
    rho_a_raw, p_a_raw = safe_spearman(df["mean_log_gbar"], df[target_col])
    rho_a_res, p_a_res = safe_spearman(df["mean_log_gbar"], df["target_resid"])

    field_raw = df.loc[df["env_binary"] == 0, target_col]
    dense_raw = df.loc[df["env_binary"] == 1, target_col]
    field_res = df.loc[df["env_binary"] == 0, "target_resid"]
    dense_res = df.loc[df["env_binary"] == 1, "target_resid"]
    env_raw_p = safe_mannwhitney(field_raw, dense_raw)
    env_res_p = safe_mannwhitney(field_res, dense_res)

    r2_raw, _, _ = linear_r2(df["logMh"], df[target_col])
    r2_resid, _, _ = linear_r2(df["logMh"], df["target_resid"])
    delta_r2 = (r2_resid - r2_raw) if np.isfinite(r2_raw) and np.isfinite(r2_resid) else np.nan

    if np.isfinite(delta_r2) and delta_r2 > R2_DELTA_TOL:
        verdict = "SIGNAL_STRENGTHENED"
    elif np.isfinite(delta_r2) and delta_r2 < -R2_DELTA_TOL:
        verdict = "WEAKENED"
    else:
        verdict = "UNCHANGED"

    # Individual confound R^2
    indiv_r2 = {}
    y = df[target_col].to_numpy(dtype=float)
    for c in confound_cols:
        xc = df[[c]].to_numpy(dtype=float)
        xc = (xc - np.mean(xc, axis=0)) / (np.std(xc, axis=0, ddof=0) + 1e-12)
        _, _, _, c_r2 = fit_ols(y, xc)
        indiv_r2[c] = float(c_r2)

    out.update({
        "confound_model_r2": float(r2_conf),
        "confound_model_coefficients": {
            "intercept": float(beta[0]),
            **{confound_cols[i]: float(beta[i + 1]) for i in range(len(confound_cols))}
        },
        "confound_standardization": {"means": means, "stds": stds},
        "indiv_confound_r2": indiv_r2,
        "signal_recovery": {
            "R2_raw": float(r2_raw) if np.isfinite(r2_raw) else None,
            "R2_resid": float(r2_resid) if np.isfinite(r2_resid) else None,
            "delta_R2": float(delta_r2) if np.isfinite(delta_r2) else None,
            "rho_mass_raw": rho_m_raw,
            "rho_mass_resid": rho_m_res,
            "rho_mass_p_raw": p_m_raw,
            "rho_mass_p_resid": p_m_res,
            "rho_accel_raw": rho_a_raw,
            "rho_accel_resid": rho_a_res,
            "rho_accel_p_raw": p_a_raw,
            "rho_accel_p_resid": p_a_res,
            "env_field_median_raw": float(np.nanmedian(field_raw)) if len(field_raw) else None,
            "env_dense_median_raw": float(np.nanmedian(dense_raw)) if len(dense_raw) else None,
            "env_field_median_resid": float(np.nanmedian(field_res)) if len(field_res) else None,
            "env_dense_median_resid": float(np.nanmedian(dense_res)) if len(dense_res) else None,
            "env_mannwhitney_p_raw": env_raw_p,
            "env_mannwhitney_p_resid": env_res_p,
        },
        "verdict": verdict,
        "flag_confound_dominated": bool(np.isfinite(r2_conf) and r2_conf > 0.5),
        "data": df[["galaxy", "logMh", "env_binary", "mean_log_gbar", target_col, "target_resid"]].copy(),
    })
    return out


def add_t4_gas_dominated_flag(rar, wallaby):
    w_galaxies = set(wallaby["galaxy"].astype(str).unique())
    is_wallaby_source = rar["source"].astype(str).str.upper().str.contains("WALLABY", na=False)
    is_wallaby_gal = rar["galaxy"].astype(str).isin(w_galaxies)
    rar = rar.copy()
    rar["gas_dominated"] = (is_wallaby_source | is_wallaby_gal).astype(int)
    return rar


def compute_env_delta_per_bin(df, value_col, bin_col):
    rows = []
    for b in sorted(df[bin_col].dropna().unique()):
        bdf = df[df[bin_col] == b]
        f = bdf.loc[bdf["env_binary"] == 0, value_col]
        d = bdf.loc[bdf["env_binary"] == 1, value_col]
        if len(f) < 5 or len(d) < 5:
            rows.append({
                "bin": int(b),
                "n_field": int(len(f)),
                "n_dense": int(len(d)),
                "sigma_field": None,
                "sigma_dense": None,
                "delta_sigma_field_minus_dense": None,
            })
            continue
        sf = float(np.std(f, ddof=1))
        sd = float(np.std(d, ddof=1))
        rows.append({
            "bin": int(b),
            "n_field": int(len(f)),
            "n_dense": int(len(d)),
            "sigma_field": sf,
            "sigma_dense": sd,
            "delta_sigma_field_minus_dense": sf - sd,
        })
    return rows


def analyze_t4_point_level(rar):
    out = {
        "target": "T4_env_delta_scatter",
        "n_valid": 0,
        "skipped": False,
        "skip_reason": None,
    }

    df = rar[["galaxy", "log_gbar", "log_res", "logMh", "env_dense", "gas_dominated"]].dropna().copy()
    df["env_binary"] = (df["env_dense"].astype(str).str.lower() == "dense").astype(int)
    bins = np.linspace(-13.0, -9.0, 6)
    df["log_gbar_bin"] = pd.cut(df["log_gbar"], bins=bins, include_lowest=True, labels=False)
    df = df.dropna(subset=["log_gbar_bin"]).copy()
    df["log_gbar_bin"] = df["log_gbar_bin"].astype(int)
    out["n_valid"] = int(len(df))
    if len(df) < 20:
        out["skipped"] = True
        out["skip_reason"] = f"fewer than {MIN_GALAXIES} valid points for T4 residualization"
        out["data"] = None
        return out

    # Residualize point-level log_res using bin dummies + gas_dominated
    bin_dummies = pd.get_dummies(df["log_gbar_bin"], prefix="bin", drop_first=True, dtype=float)
    x = pd.concat([bin_dummies, df[["gas_dominated"]].astype(float)], axis=1)
    x_mat = x.to_numpy(dtype=float)
    y = df["log_res"].to_numpy(dtype=float)
    beta, yhat, resid, r2_conf = fit_ols(y, x_mat)
    df["log_res_resid"] = resid
    df["abs_scatter_raw"] = np.abs(df["log_res"])
    df["abs_scatter_resid"] = np.abs(df["log_res_resid"])

    # Before/after tests on scatter proxy |residual|
    rho_m_raw, p_m_raw = safe_spearman(df["logMh"], df["abs_scatter_raw"])
    rho_m_res, p_m_res = safe_spearman(df["logMh"], df["abs_scatter_resid"])
    rho_a_raw, p_a_raw = safe_spearman(df["log_gbar"], df["abs_scatter_raw"])
    rho_a_res, p_a_res = safe_spearman(df["log_gbar"], df["abs_scatter_resid"])

    field_raw = df.loc[df["env_binary"] == 0, "abs_scatter_raw"]
    dense_raw = df.loc[df["env_binary"] == 1, "abs_scatter_raw"]
    field_res = df.loc[df["env_binary"] == 0, "abs_scatter_resid"]
    dense_res = df.loc[df["env_binary"] == 1, "abs_scatter_resid"]
    env_raw_p = safe_mannwhitney(field_raw, dense_raw)
    env_res_p = safe_mannwhitney(field_res, dense_res)

    r2_raw, _, _ = linear_r2(df["logMh"], df["abs_scatter_raw"])
    r2_resid, _, _ = linear_r2(df["logMh"], df["abs_scatter_resid"])
    delta_r2 = (r2_resid - r2_raw) if np.isfinite(r2_raw) and np.isfinite(r2_resid) else np.nan

    if np.isfinite(delta_r2) and delta_r2 > R2_DELTA_TOL:
        verdict = "SIGNAL_STRENGTHENED"
    elif np.isfinite(delta_r2) and delta_r2 < -R2_DELTA_TOL:
        verdict = "WEAKENED"
    else:
        verdict = "UNCHANGED"

    # Per-bin env deltas (raw and residualized log_res)
    raw_bin = compute_env_delta_per_bin(df, "log_res", "log_gbar_bin")
    resid_bin = compute_env_delta_per_bin(df, "log_res_resid", "log_gbar_bin")
    raw_deltas = [r["delta_sigma_field_minus_dense"] for r in raw_bin if r["delta_sigma_field_minus_dense"] is not None]
    resid_deltas = [r["delta_sigma_field_minus_dense"] for r in resid_bin if r["delta_sigma_field_minus_dense"] is not None]

    agg_raw = float(np.mean(raw_deltas)) if raw_deltas else None
    agg_resid = float(np.mean(resid_deltas)) if resid_deltas else None

    # Individual confound R^2 for plot B
    indiv_r2 = {}
    _, _, _, r2_gd = fit_ols(y, df[["gas_dominated"]].to_numpy(dtype=float))
    _, _, _, r2_bin = fit_ols(y, bin_dummies.to_numpy(dtype=float))
    indiv_r2["gas_dominated"] = float(r2_gd)
    indiv_r2["log_gbar_bin"] = float(r2_bin)

    out.update({
        "confound_model_r2": float(r2_conf),
        "indiv_confound_r2": indiv_r2,
        "signal_recovery": {
            "R2_raw": float(r2_raw) if np.isfinite(r2_raw) else None,
            "R2_resid": float(r2_resid) if np.isfinite(r2_resid) else None,
            "delta_R2": float(delta_r2) if np.isfinite(delta_r2) else None,
            "rho_mass_raw": rho_m_raw,
            "rho_mass_resid": rho_m_res,
            "rho_mass_p_raw": p_m_raw,
            "rho_mass_p_resid": p_m_res,
            "rho_accel_raw": rho_a_raw,
            "rho_accel_resid": rho_a_res,
            "rho_accel_p_raw": p_a_raw,
            "rho_accel_p_resid": p_a_res,
            "env_field_median_raw": float(np.nanmedian(field_raw)) if len(field_raw) else None,
            "env_dense_median_raw": float(np.nanmedian(dense_raw)) if len(dense_raw) else None,
            "env_field_median_resid": float(np.nanmedian(field_res)) if len(field_res) else None,
            "env_dense_median_resid": float(np.nanmedian(dense_res)) if len(dense_res) else None,
            "env_mannwhitney_p_raw": env_raw_p,
            "env_mannwhitney_p_resid": env_res_p,
        },
        "env_delta_scatter_bins_raw": raw_bin,
        "env_delta_scatter_bins_resid": resid_bin,
        "env_delta_scatter_aggregate_raw": agg_raw,
        "env_delta_scatter_aggregate_resid": agg_resid,
        "verdict": verdict,
        "flag_confound_dominated": bool(np.isfinite(r2_conf) and r2_conf > 0.5),
        "data": df[[
            "galaxy", "logMh", "env_binary", "log_gbar", "log_gbar_bin",
            "abs_scatter_raw", "abs_scatter_resid"
        ]].copy(),
    })
    return out


def fit_line_for_plot(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 3 or np.std(x) < 1e-12:
        return None, None
    r2, b0, b1 = linear_r2(x, y)
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = b0 + b1 * xx
    return (xx, yy), r2


def make_plot_a(target_plot_data, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Raw vs Residualized Targets vs logMh", fontsize=14)
    order = ["T2", "T3", "T4", "T5"]
    labels = {
        "T2": "T2 scatter_transition",
        "T3": "T3 acf_lag1",
        "T4": "T4 env_delta scatter proxy",
        "T5": "T5 xi_coherence_ratio",
    }

    for i, key in enumerate(order):
        d = target_plot_data.get(key)
        ax_raw = axes[0, i]
        ax_res = axes[1, i]
        if d is None or d.get("x") is None:
            ax_raw.set_title(f"{labels[key]} raw (skipped)")
            ax_res.set_title(f"{labels[key]} resid (skipped)")
            continue

        x = d["x"]
        y_raw = d["y_raw"]
        y_res = d["y_resid"]
        n = len(x)
        alpha = 0.25 if n > 2000 else 0.5
        s = 8 if n > 2000 else 14

        ax_raw.scatter(x, y_raw, s=s, alpha=alpha, color="#58a6ff", edgecolors="none")
        ax_res.scatter(x, y_res, s=s, alpha=alpha, color="#f78166", edgecolors="none")

        line_raw, r2_raw = fit_line_for_plot(x, y_raw)
        line_res, r2_res = fit_line_for_plot(x, y_res)
        if line_raw is not None:
            ax_raw.plot(line_raw[0], line_raw[1], color="#2ea043", lw=2)
        if line_res is not None:
            ax_res.plot(line_res[0], line_res[1], color="#2ea043", lw=2)

        ax_raw.set_title(f"{labels[key]} raw\nR2={r2_raw:.3f}" if r2_raw is not None else f"{labels[key]} raw")
        ax_res.set_title(f"{labels[key]} resid\nR2={r2_res:.3f}" if r2_res is not None else f"{labels[key]} resid")
        ax_raw.set_xlabel("logMh")
        ax_res.set_xlabel("logMh")
        ax_raw.set_ylabel("value")
        ax_res.set_ylabel("residualized value")
        ax_raw.grid(alpha=0.25)
        ax_res.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(output_dir, "plot_A_raw_vs_residualized_targets.png")
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_plot_b(indiv_confound_r2, output_dir):
    confounds = [
        "log_Rext", "log_npoints", "sigma_D_dex", "Inc", "mean_log_gbar", "log_gbar_bin", "gas_dominated"
    ]
    targets = ["T2", "T3", "T4", "T5"]
    vals = np.zeros((len(targets), len(confounds)))
    vals[:] = np.nan
    for i, t in enumerate(targets):
        d = indiv_confound_r2.get(t, {})
        for j, c in enumerate(confounds):
            vals[i, j] = d.get(c, np.nan)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(targets))
    width = 0.11
    colors = ["#58a6ff", "#2ea043", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff", "#8b949e"]
    for j, c in enumerate(confounds):
        ax.bar(x + (j - 3) * width, vals[:, j], width=width, label=c, color=colors[j % len(colors)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Individual confound R2")
    ax.set_title("Confound R2 per target (single-confound models)")
    ax.set_ylim(0, max(0.8, np.nanmax(vals) + 0.05 if np.isfinite(np.nanmax(vals)) else 0.8))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    out = os.path.join(output_dir, "plot_B_confound_r2_bar_chart.png")
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_plot_c(delta_r2_map, output_dir):
    labels = ["T2", "T3", "T4", "T5"]
    vals = [delta_r2_map.get(k, np.nan) for k in labels]
    colors = ["#2ea043" if (np.isfinite(v) and v > 0) else "#f85149" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vals, color=colors, alpha=0.9)
    ax.axhline(0, color="#8b949e", lw=1)
    ax.set_ylabel("delta R2 (residualized - raw)")
    ax.set_title("Signal Recovery Summary")
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top")
    plt.tight_layout()
    out = os.path.join(output_dir, "plot_C_signal_recovery_delta_R2.png")
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_plot_d(t3_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = np.linspace(-1.0, 1.0, 26)
    for ax, col, title in [
        (axes[0], "T3_acf_lag1", "ACF lag-1 raw"),
        (axes[1], "T3_resid", "ACF lag-1 residualized"),
    ]:
        field = t3_df.loc[t3_df["env_binary"] == 0, col].dropna()
        dense = t3_df.loc[t3_df["env_binary"] == 1, col].dropna()
        ax.hist(field, bins=bins, alpha=0.5, label=f"field (n={len(field)})", color="#58a6ff", density=True)
        ax.hist(dense, bins=bins, alpha=0.5, label=f"dense (n={len(dense)})", color="#f78166", density=True)
        ax.set_title(title)
        ax.set_xlabel("acf_lag1")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("density")
    fig.suptitle("T3 ACF lag-1 distributions by environment")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(output_dir, "plot_D_acf_lag1_env_histograms.png")
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    setup_plot_style()
    paths = build_paths()

    print("=" * 72)
    print("RESIDUALIZED BEC SIGNAL ANALYSIS")
    print("=" * 72)
    print(f"RAR points unified: {paths.rar_points_unified}")
    print(f"Galaxy results unified: {paths.galaxy_results_unified}")
    print(f"SPARC galaxy results: {paths.galaxy_results_sparc}")
    print(f"WALLABY points: {paths.rar_points_wallaby}")
    print(f"Output directory: {paths.output_dir}")
    if paths.output_dir_fallback_used:
        print("NOTE: /mnt/user-data was not writable; using fallback output directory.")

    rar = pd.read_csv(paths.rar_points_unified)
    gal_unified = pd.read_csv(paths.galaxy_results_unified)
    gal_sparc = pd.read_csv(paths.galaxy_results_sparc)
    wallaby = pd.read_csv(paths.rar_points_wallaby)

    # Normalize string fields
    for df in [rar, gal_unified]:
        df["env_dense"] = df["env_dense"].astype(str).str.lower()

    base = compute_base_tables(rar, gal_unified, gal_sparc)

    # Targets
    t2 = compute_t2_scatter_transition(rar)
    t3 = compute_t3_acf_lag1(rar)
    t5 = compute_t5_xi_coherence_ratio(rar)
    cs2_available = "cs2_median" in base.columns

    confounds = ["log_Rext", "log_npoints", "sigma_D_dex", "Inc", "mean_log_gbar"]
    t2_df = base.merge(t2, on="galaxy", how="left")
    t3_df = base.merge(t3, on="galaxy", how="left")
    t5_df = base.merge(t5, on="galaxy", how="left")

    res_t2 = analyze_galaxy_target("T2_scatter_transition", t2_df, "T2_scatter_transition", confounds)
    res_t3 = analyze_galaxy_target("T3_acf_lag1", t3_df, "T3_acf_lag1", confounds)
    res_t5 = analyze_galaxy_target("T5_xi_coherence_ratio", t5_df, "T5_xi_coherence_ratio", confounds)

    rar_t4 = add_t4_gas_dominated_flag(rar, wallaby)
    res_t4 = analyze_t4_point_level(rar_t4)

    target_results = {
        "T2": res_t2,
        "T3": res_t3,
        "T4": res_t4,
        "T5": res_t5,
    }

    # Plot data containers
    plot_data = {}
    if not res_t2["skipped"]:
        d = res_t2["data"]
        plot_data["T2"] = {
            "x": d["logMh"].to_numpy(),
            "y_raw": d["T2_scatter_transition"].to_numpy(),
            "y_resid": d["target_resid"].to_numpy(),
        }
    else:
        plot_data["T2"] = None
    if not res_t3["skipped"]:
        d = res_t3["data"]
        plot_data["T3"] = {
            "x": d["logMh"].to_numpy(),
            "y_raw": d["T3_acf_lag1"].to_numpy(),
            "y_resid": d["target_resid"].to_numpy(),
        }
    else:
        plot_data["T3"] = None
    if not res_t4["skipped"]:
        d = res_t4["data"]
        # Subsample for scatter visibility
        if len(d) > 4000:
            d = d.sample(4000, random_state=42)
        plot_data["T4"] = {
            "x": d["logMh"].to_numpy(),
            "y_raw": d["abs_scatter_raw"].to_numpy(),
            "y_resid": d["abs_scatter_resid"].to_numpy(),
        }
    else:
        plot_data["T4"] = None
    if not res_t5["skipped"]:
        d = res_t5["data"]
        plot_data["T5"] = {
            "x": d["logMh"].to_numpy(),
            "y_raw": d["T5_xi_coherence_ratio"].to_numpy(),
            "y_resid": d["target_resid"].to_numpy(),
        }
    else:
        plot_data["T5"] = None

    indiv_r2_plot = {
        "T2": res_t2.get("indiv_confound_r2", {}),
        "T3": res_t3.get("indiv_confound_r2", {}),
        "T4": res_t4.get("indiv_confound_r2", {}),
        "T5": res_t5.get("indiv_confound_r2", {}),
    }
    delta_r2_plot = {
        "T2": res_t2.get("signal_recovery", {}).get("delta_R2"),
        "T3": res_t3.get("signal_recovery", {}).get("delta_R2"),
        "T4": res_t4.get("signal_recovery", {}).get("delta_R2"),
        "T5": res_t5.get("signal_recovery", {}).get("delta_R2"),
    }

    plot_a = make_plot_a(plot_data, paths.output_dir)
    plot_b = make_plot_b(indiv_r2_plot, paths.output_dir)
    plot_c = make_plot_c(delta_r2_plot, paths.output_dir)

    if not res_t3["skipped"]:
        t3_plot_df = res_t3["data"].copy()
        t3_plot_df["T3_resid"] = t3_plot_df["target_resid"]
        plot_d = make_plot_d(t3_plot_df, paths.output_dir)
    else:
        plot_d = None

    # Build per-galaxy residual output table
    gal_out = base[["galaxy", "logMh", "env_dense", "log_Rext", "log_npoints", "sigma_D_dex", "Inc", "mean_log_gbar"]].copy()
    if not res_t2["skipped"]:
        d = res_t2["data"][["galaxy", "T2_scatter_transition", "target_resid"]].rename(columns={"target_resid": "T2_scatter_transition_resid"})
        gal_out = gal_out.merge(d, on="galaxy", how="left")
    if not res_t3["skipped"]:
        d = res_t3["data"][["galaxy", "T3_acf_lag1", "target_resid"]].rename(columns={"target_resid": "T3_acf_lag1_resid"})
        gal_out = gal_out.merge(d, on="galaxy", how="left")
    if not res_t5["skipped"]:
        d = res_t5["data"][["galaxy", "T5_xi_coherence_ratio", "target_resid"]].rename(columns={"target_resid": "T5_xi_coherence_ratio_resid"})
        gal_out = gal_out.merge(d, on="galaxy", how="left")
    if not res_t4["skipped"]:
        t4_pg = (
            res_t4["data"]
            .groupby("galaxy", as_index=False)
            .agg(
                T4_abs_scatter_raw=("abs_scatter_raw", "mean"),
                T4_abs_scatter_resid=("abs_scatter_resid", "mean"),
            )
        )
        gal_out = gal_out.merge(t4_pg, on="galaxy", how="left")

    gal_csv_path = os.path.join(paths.output_dir, "galaxy_residuals.csv")
    gal_out.to_csv(gal_csv_path, index=False)

    # Assemble required JSON structure
    confound_r2_per_target = {
        "T1_cs2_median": None,
        "T2_scatter_transition": None if res_t2["skipped"] else res_t2["confound_model_r2"],
        "T3_acf_lag1": None if res_t3["skipped"] else res_t3["confound_model_r2"],
        "T4_env_delta": None if res_t4["skipped"] else res_t4["confound_model_r2"],
        "T5_xi_coherence": None if res_t5["skipped"] else res_t5["confound_model_r2"],
    }

    signal_recovery = {
        "T1": None,
        "T2": None if res_t2["skipped"] else res_t2["signal_recovery"],
        "T3": None if res_t3["skipped"] else res_t3["signal_recovery"],
        "T4": None if res_t4["skipped"] else res_t4["signal_recovery"],
        "T5": None if res_t5["skipped"] else res_t5["signal_recovery"],
    }

    verdict_per_target = {
        "T1": "SKIPPED_NOT_AVAILABLE",
        "T2": "SKIPPED_INSUFFICIENT_SAMPLE" if res_t2["skipped"] else res_t2["verdict"],
        "T3": "SKIPPED_INSUFFICIENT_SAMPLE" if res_t3["skipped"] else res_t3["verdict"],
        "T4": "SKIPPED_INSUFFICIENT_SAMPLE" if res_t4["skipped"] else res_t4["verdict"],
        "T5": "SKIPPED_INSUFFICIENT_SAMPLE" if res_t5["skipped"] else res_t5["verdict"],
    }

    valid_verdicts = [v for k, v in verdict_per_target.items() if k != "T1" and not v.startswith("SKIPPED")]
    n_strengthened = sum(v == "SIGNAL_STRENGTHENED" for v in valid_verdicts)
    n_weakened = sum(v == "WEAKENED" for v in valid_verdicts)
    n_unchanged = sum(v == "UNCHANGED" for v in valid_verdicts)
    if len(valid_verdicts) == 0:
        overall_verdict = "INSUFFICIENT_DATA"
    elif n_strengthened > n_weakened:
        overall_verdict = "SIGNAL_RECOVERY_AFTER_RESIDUALIZATION"
    elif n_weakened > n_strengthened:
        overall_verdict = "SIGNAL_WEAKENED_AFTER_RESIDUALIZATION"
    else:
        overall_verdict = "MIXED_OR_UNCHANGED"

    flags = {
        "T2_confound_R2_gt_0p5": False if res_t2["skipped"] else bool(res_t2["flag_confound_dominated"]),
        "T3_confound_R2_gt_0p5": False if res_t3["skipped"] else bool(res_t3["flag_confound_dominated"]),
        "T4_confound_R2_gt_0p5": False if res_t4["skipped"] else bool(res_t4["flag_confound_dominated"]),
        "T5_confound_R2_gt_0p5": False if res_t5["skipped"] else bool(res_t5["flag_confound_dominated"]),
    }

    summary = {
        "n_galaxies": int(base["galaxy"].nunique()),
        "n_points_unified": int(len(rar)),
        "paths": {
            "rar_points_unified": paths.rar_points_unified,
            "galaxy_results_unified": paths.galaxy_results_unified,
            "galaxy_results_sparc": paths.galaxy_results_sparc,
            "rar_points_wallaby": paths.rar_points_wallaby,
            "output_dir": paths.output_dir,
            "output_dir_fallback_used": paths.output_dir_fallback_used,
        },
        "targets_available": {
            "T1_cs2_median_available": bool(cs2_available),
            "T2_valid_n": res_t2["n_valid"],
            "T3_valid_n": res_t3["n_valid"],
            "T4_valid_n_points": res_t4["n_valid"],
            "T5_valid_n": res_t5["n_valid"],
        },
        "confound_R2_per_target": confound_r2_per_target,
        "signal_recovery": signal_recovery,
        "verdict_per_target": verdict_per_target,
        "overall_verdict": overall_verdict,
        "flags_confound_dominance": flags,
        "t4_env_delta_bins": {
            "raw": None if res_t4["skipped"] else res_t4["env_delta_scatter_bins_raw"],
            "residualized": None if res_t4["skipped"] else res_t4["env_delta_scatter_bins_resid"],
            "aggregate_raw": None if res_t4["skipped"] else res_t4["env_delta_scatter_aggregate_raw"],
            "aggregate_residualized": None if res_t4["skipped"] else res_t4["env_delta_scatter_aggregate_resid"],
        },
        "plot_paths": {
            "plot_A": plot_a,
            "plot_B": plot_b,
            "plot_C": plot_c,
            "plot_D": plot_d,
        },
        "notes": [
            "T1 (cs2_median) skipped because no cs2_median column was found in available input tables.",
            "For T4, point-level residualization uses confounds [log_gbar_bin, gas_dominated] as requested.",
            "Targets with fewer than 20 matched samples are skipped and marked explicitly.",
        ],
    }

    summary_path = os.path.join(paths.output_dir, "summary_residualization.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {summary_path}")
    print(f"Saved: {gal_csv_path}")
    print(f"Saved plots: {plot_a}, {plot_b}, {plot_c}, {plot_d}")
    print(f"Overall verdict: {overall_verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
