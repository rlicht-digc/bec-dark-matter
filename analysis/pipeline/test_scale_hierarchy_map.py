#!/usr/bin/env python3
"""
Scale Hierarchy Map for RAR Residual Structure
==============================================

Builds a canonical galaxy-scale table from existing summary JSONs and pipeline
tables, then evaluates:
  1) Within-galaxy coherence peak in R/xi space + xi-permutation null
  2) Between-galaxy matched high-xi vs low-xi effect on Lc/xi

Outputs (under analysis/results/scale_hierarchy by default):
  - galaxy_scale_table.parquet
  - point_scale_table.parquet
  - results_scale_hierarchy.parquet
  - fig_scale_map.png
  - scale_map_report.md
  - run_log.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


G_SI = 6.674e-11
M_SUN = 1.989e30
KPC_M = 3.086e19
G_DAGGER = 1.2e-10


def rar_pred_log(log_gbar: np.ndarray, gdagger: float = G_DAGGER) -> np.ndarray:
    gbar = 10.0 ** np.asarray(log_gbar, dtype=float)
    term = 1.0 - np.exp(-np.sqrt(np.maximum(gbar / gdagger, 1e-30)))
    gobs = gbar / np.maximum(term, 1e-30)
    return np.log10(np.maximum(gobs, 1e-30))


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sig = 1.4826 * mad
    if sig <= 0:
        sig = np.std(x, ddof=1)
    return float(sig) if np.isfinite(sig) else np.nan


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    # Probability(X>Y) - Probability(X<Y)
    diff = a[:, None] - b[None, :]
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    return float((gt - lt) / (a.size * b.size))


def bootstrap_cliffs_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> Tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    ia = np.arange(a.size)
    ib = np.arange(b.size)
    for i in range(n_boot):
        sa = a[rng.choice(ia, size=a.size, replace=True)]
        sb = b[rng.choice(ib, size=b.size, replace=True)]
        vals[i] = cliffs_delta(sa, sb)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.std(vals, ddof=1))


def median_diff_perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 1000, seed: int = 42) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = np.median(a) - np.median(b)
    pooled = np.concatenate([a, b])
    n_a = a.size
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = np.median(perm[:n_a]) - np.median(perm[n_a:])
        if abs(stat) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def get_git_hash(project_root: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_sparc_mstar_from_mrt(mrt_path: str) -> pd.DataFrame:
    """
    Parses SPARC_Lelli2016c.mrt and computes Mstar from L[3.6] using fixed
    M/L=0.5 (same convention used elsewhere in this project).
    """
    lines = open(mrt_path, "r").read().splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---") and i > 50:
            data_start = i + 1
            break

    rows = []
    for line in lines[data_start:]:
        if not line.strip() or line.startswith("#"):
            continue
        name = line[0:11].strip()
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        try:
            inc = float(parts[4])
            q = int(parts[16])
            l36 = float(parts[6])  # units 1e9 Lsun
            mstar = 0.5 * l36 * 1.0e9
            rows.append(
                {
                    "gal_id": name,
                    "mstar_solar_sparc": mstar,
                    "incl_mrt": inc,
                    "q_mrt": q,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates(subset=["gal_id"])


def enrich_with_summary_metrics(gal: pd.DataFrame, summary_dir: str) -> pd.DataFrame:
    # Healing-length scaling summary (67-gal core set)
    hls = load_json(os.path.join(summary_dir, "summary_healing_length_scaling.json"))
    hls_rows = pd.DataFrame(hls.get("per_galaxy", []))
    if not hls_rows.empty:
        hls_rows = hls_rows.rename(
            columns={
                "name": "gal_id",
                "n_pts": "Npts_hls",
                "dR_kpc": "dR_hls",
                "R_extent_kpc": "Rext_hls",
                "xi_kpc": "xi_hls_kpc",
                "Lc_acf_kpc": "Lc",
                "lambda_short_kpc": "lambda_peak_hls",
                "acf1": "acf_peak_hls",
                "lambda_all_kpc": "acf_scale_hls",
                "M_b_Msun": "Mb_hls",
                "env": "env_hls",
            }
        )
        keep = [
            "gal_id",
            "Npts_hls",
            "dR_hls",
            "Rext_hls",
            "xi_hls_kpc",
            "Lc",
            "lambda_peak_hls",
            "acf_peak_hls",
            "acf_scale_hls",
            "Mb_hls",
            "env_hls",
        ]
        gal = gal.merge(hls_rows[keep], on="gal_id", how="left")

    # ACF summary
    hacf = load_json(os.path.join(summary_dir, "summary_healing_length_acf.json"))
    hacf_rows = pd.DataFrame(hacf.get("per_galaxy", []))
    if not hacf_rows.empty:
        hacf_rows = hacf_rows.rename(
            columns={
                "name": "gal_id",
                "n": "Npts_hacf",
                "r1": "r1",
                "ar1g": "AR1g",
                "ar1": "AR1",
                "R_ext": "Rext_hacf",
                "dR": "dR_hacf",
                "xi_kpc": "xi_hacf_kpc",
                "xi_Re": "xi_over_Re_hacf",
                "Inc": "incl_hacf",
                "eD": "dist_err_frac_hacf",
                "env": "env_hacf",
            }
        )
        keep = [
            "gal_id",
            "Npts_hacf",
            "r1",
            "AR1g",
            "AR1",
            "Rext_hacf",
            "dR_hacf",
            "xi_hacf_kpc",
            "xi_over_Re_hacf",
            "incl_hacf",
            "dist_err_frac_hacf",
            "env_hacf",
        ]
        gal = gal.merge(hacf_rows[keep], on="gal_id", how="left")

    # kstar summary
    hk = load_json(os.path.join(summary_dir, "summary_healing_length_kstar.json"))
    hk_rows = pd.DataFrame(hk.get("per_galaxy", []))
    if not hk_rows.empty:
        hk_rows = hk_rows.rename(
            columns={
                "name": "gal_id",
                "kstar": "kstar",
                "acf1": "acf_peak_kstar",
                "R_ext": "Rext_kstar",
                "n": "Npts_kstar",
                "xi_kpc": "xi_kstar_kpc",
            }
        )
        keep = ["gal_id", "kstar", "acf_peak_kstar", "Rext_kstar", "Npts_kstar", "xi_kstar_kpc"]
        gal = gal.merge(hk_rows[keep], on="gal_id", how="left")

    # periodic lambda-xi summary (periodic subset)
    ps = load_json(os.path.join(summary_dir, "summary_periodic_properties_and_scaling.json"))
    p_rows = pd.DataFrame(ps.get("part2_lambda_xi_scaling", {}).get("per_galaxy", []))
    if not p_rows.empty:
        p_rows = p_rows.rename(
            columns={
                "name": "gal_id",
                "lambda_peak_kpc": "lambda_peak_periodic",
                "lambda_over_xi": "lambda_over_xi_periodic",
                "xi_kpc": "xi_periodic_kpc",
            }
        )
        keep = ["gal_id", "lambda_peak_periodic", "lambda_over_xi_periodic", "xi_periodic_kpc"]
        gal = gal.merge(p_rows[keep], on="gal_id", how="left")

    # window matching summary
    ws = load_json(os.path.join(summary_dir, "summary_window_matching.json"))
    w_rows = pd.DataFrame(ws.get("per_galaxy", []))
    if not w_rows.empty:
        w_rows = w_rows.rename(
            columns={
                "name": "gal_id",
                "window_wl": "acf_scale_window",
                "window_power": "acf_power_window",
                "window_detected": "window_detected",
            }
        )
        keep = ["gal_id", "acf_scale_window", "acf_power_window", "window_detected"]
        gal = gal.merge(w_rows[keep], on="gal_id", how="left")

    return gal


def build_canonical_galaxy_table(project_root: str) -> pd.DataFrame:
    results_dir = os.path.join(project_root, "analysis", "results")
    summary_dir = results_dir

    # Unified galaxy-level table
    g = pd.read_csv(os.path.join(results_dir, "galaxy_results_unified.csv"))
    g = g.rename(columns={"galaxy": "gal_id", "n_points": "Npts", "env_dense": "env"})
    g["source"] = g["source"].astype(str)
    g["gal_id"] = g["gal_id"].astype(str)

    # Unified point table for geometry
    p = pd.read_csv(os.path.join(results_dir, "rar_points_unified.csv"))
    p = p.rename(columns={"galaxy": "gal_id", "R_kpc": "R"})
    p["source"] = p["source"].astype(str)
    p["gal_id"] = p["gal_id"].astype(str)
    p = p.sort_values(["source", "gal_id", "R"])

    def _agg_geom(df: pd.DataFrame) -> pd.Series:
        r = np.asarray(df["R"], dtype=float)
        r = r[np.isfinite(r)]
        r.sort()
        d = np.diff(r)
        d = d[d > 0]
        return pd.Series(
            {
                "Rext": np.nanmax(r) if r.size else np.nan,
                "Npts_geom": int(r.size),
                "dR": float(np.median(d)) if d.size else np.nan,
                "mean_log_gbar": float(np.nanmean(df["log_gbar"])) if len(df) else np.nan,
            }
        )

    geom = p.groupby(["source", "gal_id"], as_index=False).apply(_agg_geom, include_groups=False).reset_index()
    if "level_2" in geom.columns:
        geom = geom.drop(columns=["level_2"])

    gal = g.merge(geom, on=["source", "gal_id"], how="left")

    # Add SPARC-specific inclination + distance uncertainty + halo mass references
    sparc_haub = pd.read_csv(os.path.join(results_dir, "galaxy_results_sparc_orig_haubner.csv"))
    sparc_haub = sparc_haub.rename(
        columns={
            "galaxy": "gal_id",
            "Inc": "incl_haubner",
            "sigma_D_frac": "dist_err_frac_haubner",
            "logMh": "logMh_haubner",
            "n_points": "Npts_haubner",
            "env_binary": "env_binary_haubner",
        }
    )
    keep = ["gal_id", "incl_haubner", "dist_err_frac_haubner", "logMh_haubner", "Npts_haubner", "env_binary_haubner"]
    gal = gal.merge(sparc_haub[keep], on="gal_id", how="left")

    # SPARC stellar mass from MRT
    sparc_mrt = parse_sparc_mstar_from_mrt(os.path.join(project_root, "data", "sparc", "SPARC_Lelli2016c.mrt"))
    gal = gal.merge(sparc_mrt, on="gal_id", how="left")

    # Merge SPARC-derived structure summaries
    gal = enrich_with_summary_metrics(gal, summary_dir)

    # Collapse / harmonize core fields
    gal["Npts"] = gal["Npts"].fillna(gal["Npts_geom"]).fillna(gal.get("Npts_hacf"))
    gal["Rext"] = gal.get("Rext").fillna(gal.get("Rext_hls")).fillna(gal.get("Rext_hacf")).fillna(gal.get("Rext_kstar"))
    gal["dR"] = gal.get("dR").fillna(gal.get("dR_hls")).fillna(gal.get("dR_hacf"))
    gal["incl"] = gal.get("incl_haubner").fillna(gal.get("incl_hacf"))
    gal["dist_err_frac"] = gal.get("dist_err_frac_haubner").fillna(gal.get("dist_err_frac_hacf"))
    gal["env"] = gal.get("env").fillna(gal.get("env_hls")).fillna(gal.get("env_hacf")).fillna(gal.get("env_binary_haubner"))

    # Mass fields
    gal["Mh"] = np.where(np.isfinite(gal["logMh"]), 10.0 ** gal["logMh"], np.nan)
    # keep Haubner override where unified logMh missing
    mh_h = np.where(np.isfinite(gal.get("logMh_haubner")), 10.0 ** gal.get("logMh_haubner"), np.nan)
    gal["Mh"] = np.where(np.isfinite(gal["Mh"]), gal["Mh"], mh_h)
    gal["Mstar"] = gal.get("mstar_solar_sparc")

    # Xi from available mass proxy
    xi_star = np.sqrt(np.maximum(G_SI * np.maximum(gal["Mstar"], 0) * M_SUN / G_DAGGER, 0)) / KPC_M
    xi_halo = np.sqrt(np.maximum(G_SI * np.maximum(gal["Mh"], 0) * M_SUN / G_DAGGER, 0)) / KPC_M
    gal["xi"] = np.where(np.isfinite(xi_star), xi_star, xi_halo)
    xi_basis = np.full(len(gal), "unknown", dtype=object)
    xi_basis[np.isfinite(xi_halo)] = "Mh"
    xi_basis[np.isfinite(xi_star)] = "Mstar"
    gal["xi_mass_basis"] = xi_basis
    gal["xi_summary_kpc"] = gal.get("xi_hls_kpc").fillna(gal.get("xi_hacf_kpc")).fillna(gal.get("xi_kstar_kpc")).fillna(gal.get("xi_periodic_kpc"))
    gal["xi"] = np.where(np.isfinite(gal["xi_summary_kpc"]), gal["xi_summary_kpc"], gal["xi"])
    gal["xi_source"] = np.where(np.isfinite(gal["xi_summary_kpc"]), "summary", gal["xi_mass_basis"])

    # Structure metrics
    gal["lambda_peak"] = gal.get("lambda_peak_periodic").fillna(gal.get("lambda_peak_hls"))
    gal["acf_peak"] = gal.get("acf_peak_hls").fillna(gal.get("acf_peak_kstar"))
    gal["acf_scale"] = gal.get("acf_scale_window").fillna(gal.get("acf_scale_hls"))
    gal["Lc_over_xi"] = gal["Lc"] / gal["xi"]
    gal["lambda_over_xi"] = gal["lambda_peak"] / gal["xi"]
    gal["xi_over_Rext"] = gal["xi"] / gal["Rext"]
    gal["lambda_over_Rext"] = gal["lambda_peak"] / gal["Rext"]
    gal["Lc_over_Rext"] = gal["Lc"] / gal["Rext"]

    # Tag structured SPARC sample
    gal["has_structure_metrics"] = (
        np.isfinite(gal["Lc"]) | np.isfinite(gal["lambda_peak"]) | np.isfinite(gal["r1"]) | np.isfinite(gal["AR1g"])
    )
    gal["scale_analysis_sample"] = (gal["source"] == "SPARC") & gal["has_structure_metrics"] & np.isfinite(gal["xi"])

    # Add TNG galaxy rows
    tng_points_path = os.path.join(project_root, "rar_points.parquet")
    tng_npz_path = os.path.join(project_root, "tng_mass_profiles.npz")
    if os.path.exists(tng_points_path) and os.path.exists(tng_npz_path):
        tp = pd.read_parquet(tng_points_path)
        tp["source"] = "TNG"
        tp["gal_id"] = tp["SubhaloID"].astype(str)
        tp = tp.sort_values(["gal_id", "r_kpc"])

        def _agg_tng(df: pd.DataFrame) -> pd.Series:
            r = np.asarray(df["r_kpc"], dtype=float)
            r = r[np.isfinite(r)]
            r.sort()
            d = np.diff(r)
            d = d[d > 0]
            return pd.Series(
                {
                    "source": "TNG",
                    "Rext": np.nanmax(r) if r.size else np.nan,
                    "Npts": int(r.size),
                    "dR": float(np.median(d)) if d.size else np.nan,
                    "mean_log_gbar": float(np.nanmean(df["log_gbar"])) if len(df) else np.nan,
                }
            )

        tng_geom = tp.groupby("gal_id", as_index=False).apply(_agg_tng, include_groups=False).reset_index()
        if "level_1" in tng_geom.columns:
            tng_geom = tng_geom.drop(columns=["level_1"])

        npz = np.load(tng_npz_path, allow_pickle=True)
        mass_map = dict(
            zip(np.asarray(npz["galaxy_ids"], dtype=int).astype(str), np.asarray(npz["m_star_total"], dtype=float))
        )
        tng_geom["Mstar"] = tng_geom["gal_id"].map(mass_map)
        tng_geom["Mh"] = np.nan
        tng_geom["env"] = np.nan
        tng_geom["incl"] = np.nan
        tng_geom["dist_err_frac"] = np.nan
        tng_geom["Lc"] = np.nan
        tng_geom["lambda_peak"] = np.nan
        tng_geom["r1"] = np.nan
        tng_geom["AR1g"] = np.nan
        tng_geom["acf_peak"] = np.nan
        tng_geom["acf_scale"] = np.nan
        tng_geom["xi"] = np.sqrt(np.maximum(G_SI * np.maximum(tng_geom["Mstar"], 0) * M_SUN / G_DAGGER, 0)) / KPC_M
        tng_geom["xi_source"] = "Mstar"
        tng_geom["xi_mass_basis"] = "Mstar"
        tng_geom["Lc_over_xi"] = np.nan
        tng_geom["lambda_over_xi"] = np.nan
        tng_geom["xi_over_Rext"] = tng_geom["xi"] / tng_geom["Rext"]
        tng_geom["lambda_over_Rext"] = np.nan
        tng_geom["Lc_over_Rext"] = np.nan
        tng_geom["has_structure_metrics"] = False
        tng_geom["scale_analysis_sample"] = False

        # Keep only rows not already present
        key_existing = set(zip(gal["source"].astype(str), gal["gal_id"].astype(str)))
        tng_geom = tng_geom[~tng_geom.apply(lambda r: ("TNG", str(r["gal_id"])) in key_existing, axis=1)].copy()
        gal = pd.concat([gal, tng_geom], ignore_index=True, sort=False)

    # Final schema ordering
    out_cols = [
        "gal_id",
        "source",
        "Rext",
        "Npts",
        "dR",
        "incl",
        "dist_err_frac",
        "xi",
        "Mstar",
        "Mh",
        "env",
        "Lc",
        "lambda_peak",
        "r1",
        "AR1g",
        "acf_peak",
        "acf_scale",
        "kstar",
        "mean_log_gbar",
        "Lc_over_xi",
        "lambda_over_xi",
        "xi_over_Rext",
        "lambda_over_Rext",
        "Lc_over_Rext",
        "xi_source",
        "xi_mass_basis",
        "has_structure_metrics",
        "scale_analysis_sample",
    ]
    for c in out_cols:
        if c not in gal.columns:
            gal[c] = np.nan

    gal = gal[out_cols].copy()
    gal["gal_id"] = gal["gal_id"].astype(str)
    gal["source"] = gal["source"].astype(str)
    return gal


def build_point_table(project_root: str, gal: pd.DataFrame) -> pd.DataFrame:
    results_dir = os.path.join(project_root, "analysis", "results")

    # Unified observed points
    pu = pd.read_csv(os.path.join(results_dir, "rar_points_unified.csv"))
    pu = pu.rename(columns={"galaxy": "gal_id", "R_kpc": "R"})
    pu["source"] = pu["source"].astype(str)
    pu["gal_id"] = pu["gal_id"].astype(str)
    pu = pu[["source", "gal_id", "R", "log_res", "log_gbar", "log_gobs"]].copy()
    pu["residual"] = pu["log_res"]

    # TNG point cloud
    tng_points_path = os.path.join(project_root, "rar_points.parquet")
    if os.path.exists(tng_points_path):
        tp = pd.read_parquet(tng_points_path)
        tp = tp.rename(columns={"r_kpc": "R"})
        tp["source"] = "TNG"
        tp["gal_id"] = tp["SubhaloID"].astype(str)
        tp["residual"] = tp["log_gobs"] - rar_pred_log(tp["log_gbar"].values)
        tp = tp[["source", "gal_id", "R", "residual", "log_gbar", "log_gobs"]].copy()
        pts = pd.concat([pu[["source", "gal_id", "R", "residual", "log_gbar", "log_gobs"]], tp], ignore_index=True)
    else:
        pts = pu[["source", "gal_id", "R", "residual", "log_gbar", "log_gobs"]].copy()

    pts = pts.replace([np.inf, -np.inf], np.nan).dropna(subset=["source", "gal_id", "R", "residual"])
    pts["source"] = pts["source"].astype(str)
    pts["gal_id"] = pts["gal_id"].astype(str)
    pts["R"] = pd.to_numeric(pts["R"], errors="coerce")
    pts["residual"] = pd.to_numeric(pts["residual"], errors="coerce")
    pts = pts.dropna(subset=["R", "residual"])

    # Join xi + sample flag
    key_cols = ["source", "gal_id", "xi", "scale_analysis_sample", "has_structure_metrics"]
    pts = pts.merge(gal[key_cols], on=["source", "gal_id"], how="left")
    pts["R_over_xi"] = pts["R"] / pts["xi"]

    # Local per-radius coherence metric
    pts = pts.sort_values(["source", "gal_id", "R"]).copy()
    pts["local_coherence"] = np.nan

    for (src, gid), idx in pts.groupby(["source", "gal_id"]).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        rr = pts.loc[ii, "residual"].to_numpy(dtype=float)
        lc = np.full(rr.size, np.nan, dtype=float)
        if rr.size >= 2:
            lc[:-1] = rr[:-1] * rr[1:]
        pts.loc[ii, "local_coherence"] = lc

    return pts


def coherence_curve_peak(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray
) -> Tuple[float, float, int]:
    """
    Returns (x_peak, peak_strength, n_valid_bins)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.sum(m) < 5:
        return np.nan, np.nan, 0
    x = x[m]
    y = y[m]
    bi = np.digitize(x, bins) - 1
    n_bins = len(bins) - 1
    means = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        yy = y[bi == b]
        if yy.size >= 2:
            means[b] = np.nanmean(yy)
    good = np.isfinite(means)
    n_good = int(np.sum(good))
    if n_good < 3:
        return np.nan, np.nan, n_good
    centers = 0.5 * (bins[:-1] + bins[1:])
    j = int(np.nanargmax(means))
    peak = float(centers[j])
    peak_strength = float(means[j] - np.nanmedian(means[good]))
    return peak, peak_strength, n_good


def run_within_galaxy_test(
    pts: pd.DataFrame,
    bins: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Within-galaxy coherence curves vs R/xi for scale-analysis sample.
    Null: permute xi across galaxies and recompute x_peak concentration (std).
    """
    # Restrict to the structured SPARC analysis sample
    sub = pts[pts["scale_analysis_sample"] == True].copy()
    if sub.empty:
        return pd.DataFrame(), {"n_gal": 0}

    # Pre-build per-gal arrays for faster null loops
    gal_arrays = {}
    for (src, gid), gdf in sub.groupby(["source", "gal_id"]):
        gdf = gdf.sort_values("R")
        x = gdf["R_over_xi"].to_numpy(dtype=float)
        y = gdf["local_coherence"].to_numpy(dtype=float)
        xi = safe_float(gdf["xi"].iloc[0])
        R = gdf["R"].to_numpy(dtype=float)
        if np.sum(np.isfinite(y)) < 5 or not np.isfinite(xi) or xi <= 0:
            continue
        gal_arrays[(src, gid)] = {"R": R, "y": y, "xi": xi}

    if len(gal_arrays) < 10:
        return pd.DataFrame(), {"n_gal": len(gal_arrays)}

    # Observed x_peak per galaxy
    rows = []
    for (src, gid), d in gal_arrays.items():
        x = d["R"] / d["xi"]
        peak, strength, n_good = coherence_curve_peak(x, d["y"], bins)
        rows.append(
            {
                "source": src,
                "gal_id": gid,
                "x_peak": peak,
                "peak_strength": strength,
                "n_valid_bins": n_good,
                "xi_used": d["xi"],
            }
        )
    w = pd.DataFrame(rows)
    w = w[np.isfinite(w["x_peak"])].copy()
    if len(w) < 10:
        return w, {"n_gal": len(w)}

    obs_std = float(np.std(w["x_peak"], ddof=1))
    obs_mean = float(np.mean(w["x_peak"]))

    # Null by permuting xi across galaxies
    rng = np.random.default_rng(seed)
    keys = list(gal_arrays.keys())
    xi_vals = np.array([gal_arrays[k]["xi"] for k in keys], dtype=float)
    null_std = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        perm_xi = rng.permutation(xi_vals)
        peaks = []
        for k, xi_perm in zip(keys, perm_xi):
            d = gal_arrays[k]
            x = d["R"] / xi_perm
            peak, _, _ = coherence_curve_peak(x, d["y"], bins)
            if np.isfinite(peak):
                peaks.append(peak)
        if len(peaks) >= 10:
            null_std[i] = np.std(peaks, ddof=1)
        else:
            null_std[i] = np.nan

    null_std = null_std[np.isfinite(null_std)]
    perm_p = float((np.sum(null_std <= obs_std) + 1) / (len(null_std) + 1))

    summary = {
        "n_gal": int(len(w)),
        "x_peak_mean": obs_mean,
        "x_peak_std": obs_std,
        "null_std_mean": float(np.mean(null_std)),
        "null_std_std": float(np.std(null_std, ddof=1)),
        "perm_p_concentration": perm_p,
        "n_perm_effective": int(len(null_std)),
    }
    return w, summary


def zscore_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce")
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd <= 0:
            out[c + "_z"] = np.nan
        else:
            out[c + "_z"] = (v - mu) / sd
    return out


def match_high_low_by_confounds(
    df: pd.DataFrame,
    outcome_col: str,
    xi_col: str,
    confounds: Sequence[str],
    caliper: float,
) -> Dict[str, float]:
    """
    Nearest-neighbor high-vs-low xi matching with per-confound caliper in z-space.
    """
    d = df.copy()
    d = d[np.isfinite(d[outcome_col]) & np.isfinite(d[xi_col])].copy()
    if len(d) < 12:
        return {"pairs": 0}

    # median split
    med_xi = float(np.median(d[xi_col]))
    d["high_xi"] = d[xi_col] >= med_xi

    # confound preparation
    for c in confounds:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(np.nanmedian(d[c]))

    d = zscore_columns(d, confounds)
    zcols = [c + "_z" for c in confounds]
    d = d.dropna(subset=zcols + [outcome_col, "high_xi"])
    if len(d) < 12:
        return {"pairs": 0}

    hi = d[d["high_xi"]].copy().reset_index(drop=True)
    lo = d[~d["high_xi"]].copy().reset_index(drop=True)
    if len(hi) < 5 or len(lo) < 5:
        return {"pairs": 0}

    # Match smaller group into larger group
    if len(hi) <= len(lo):
        A, B = hi, lo
        a_is_high = True
    else:
        A, B = lo, hi
        a_is_high = False

    used = np.zeros(len(B), dtype=bool)
    pairs = []
    for i in range(len(A)):
        a = A.loc[i, zcols].to_numpy(dtype=float)
        bmat = B[zcols].to_numpy(dtype=float)
        dz = np.abs(bmat - a[None, :])
        ok = np.all(dz <= caliper, axis=1) & (~used)
        if not np.any(ok):
            continue
        dist = np.sqrt(np.sum((dz[ok]) ** 2, axis=1))
        cand = np.where(ok)[0]
        j = cand[int(np.argmin(dist))]
        used[j] = True
        pairs.append((i, j))

    if len(pairs) < 6:
        return {"pairs": len(pairs)}

    A_vals = np.array([A.loc[i, outcome_col] for i, _ in pairs], dtype=float)
    B_vals = np.array([B.loc[j, outcome_col] for _, j in pairs], dtype=float)
    if a_is_high:
        high_vals = A_vals
        low_vals = B_vals
    else:
        high_vals = B_vals
        low_vals = A_vals

    cd = cliffs_delta(high_vals, low_vals)
    ci_lo, ci_hi, cd_sd = bootstrap_cliffs_ci(high_vals, low_vals, n_boot=1000, seed=42)
    p_perm = median_diff_perm_p(high_vals, low_vals, n_perm=1000, seed=42)

    return {
        "pairs": int(len(pairs)),
        "median_high": float(np.median(high_vals)),
        "median_low": float(np.median(low_vals)),
        "median_ratio_high_low": float(np.median(high_vals) / np.median(low_vals))
        if abs(np.median(low_vals)) > 1e-30
        else np.nan,
        "median_diff_high_low": float(np.median(high_vals) - np.median(low_vals)),
        "cliffs_delta": float(cd),
        "cliffs_ci95_lo": float(ci_lo),
        "cliffs_ci95_hi": float(ci_hi),
        "cliffs_boot_sd": float(cd_sd),
        "perm_p_median_diff": float(p_perm),
    }


def run_between_galaxy_tests(gal: pd.DataFrame) -> Dict[str, object]:
    # Structured SPARC sample with key fields
    d = gal[(gal["source"] == "SPARC") & (gal["has_structure_metrics"] == True)].copy()
    d = d[np.isfinite(d["xi"]) & np.isfinite(d["Lc_over_xi"])].copy()

    confounds = ["Rext", "Npts", "dR", "incl", "dist_err_frac"]
    outcome = "Lc_over_xi"

    if len(d) < 12:
        return {"n_sample": int(len(d)), "unmatched": {}, "matched_by_caliper": []}

    # Unmatched high/low split
    med_xi = float(np.median(d["xi"]))
    hi = d[d["xi"] >= med_xi][outcome].to_numpy(dtype=float)
    lo = d[d["xi"] < med_xi][outcome].to_numpy(dtype=float)
    cd = cliffs_delta(hi, lo)
    ci_lo, ci_hi, cd_sd = bootstrap_cliffs_ci(hi, lo, n_boot=1000, seed=42)
    p_perm = median_diff_perm_p(hi, lo, n_perm=1000, seed=42)

    unmatched = {
        "n_hi": int(len(hi)),
        "n_lo": int(len(lo)),
        "median_hi": float(np.median(hi)),
        "median_lo": float(np.median(lo)),
        "median_ratio_hi_lo": float(np.median(hi) / np.median(lo)) if abs(np.median(lo)) > 1e-30 else np.nan,
        "cliffs_delta": float(cd),
        "cliffs_ci95_lo": float(ci_lo),
        "cliffs_ci95_hi": float(ci_hi),
        "cliffs_boot_sd": float(cd_sd),
        "perm_p_median_diff": float(p_perm),
    }

    calipers = [0.50, 0.75, 1.00, 1.25, 1.50]
    matched = []
    for c in calipers:
        res = match_high_low_by_confounds(
            d,
            outcome_col=outcome,
            xi_col="xi",
            confounds=confounds,
            caliper=c,
        )
        res["caliper_z"] = c
        matched.append(res)

    # choose primary as largest-pairs tie-break by smallest perm p then |delta|
    mdf = pd.DataFrame(matched)
    if "pairs" in mdf.columns and (mdf["pairs"] > 0).any():
        m2 = mdf[mdf["pairs"] > 0].copy()
        m2["perm_p_median_diff"] = pd.to_numeric(m2["perm_p_median_diff"], errors="coerce").fillna(1.0)
        m2["abs_cd"] = np.abs(pd.to_numeric(m2["cliffs_delta"], errors="coerce"))
        m2 = m2.sort_values(["pairs", "perm_p_median_diff", "abs_cd"], ascending=[False, True, False])
        primary = m2.iloc[0].to_dict()
    else:
        primary = {}

    return {
        "n_sample": int(len(d)),
        "outcome": outcome,
        "confounds": confounds,
        "unmatched": unmatched,
        "matched_by_caliper": matched,
        "primary_matched": primary,
    }


def make_figure(
    out_png: str,
    within_df: pd.DataFrame,
    within_summary: Dict[str, float],
    between: Dict[str, object],
    gal: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: x_peak distribution
    ax = axes[0]
    if len(within_df) > 0 and np.isfinite(within_df["x_peak"]).any():
        vals = within_df["x_peak"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=12, alpha=0.8, color="#1f77b4", edgecolor="white")
        ax.set_xlabel("x_peak (R/xi)")
        ax.set_ylabel("N galaxies")
        ax.set_title("Within-Galaxy x_peak Distribution")
        txt = (
            f"N={within_summary.get('n_gal', 0)}\n"
            f"std={within_summary.get('x_peak_std', np.nan):.3f}\n"
            f"perm p={within_summary.get('perm_p_concentration', np.nan):.3g}"
        )
        ax.text(0.98, 0.98, txt, ha="right", va="top", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No valid x_peak sample", ha="center", va="center")
        ax.set_title("Within-Galaxy x_peak Distribution")

    # Panel 2: effect size vs controls (caliper sweep + unmatched)
    ax = axes[1]
    mdf = pd.DataFrame(between.get("matched_by_caliper", []))
    if not mdf.empty and "caliper_z" in mdf.columns:
        mdf = mdf.sort_values("caliper_z")
        y = pd.to_numeric(mdf["cliffs_delta"], errors="coerce")
        ylo = pd.to_numeric(mdf["cliffs_ci95_lo"], errors="coerce")
        yhi = pd.to_numeric(mdf["cliffs_ci95_hi"], errors="coerce")
        x = pd.to_numeric(mdf["caliper_z"], errors="coerce")
        ax.plot(x, y, marker="o", label="Matched")
        for xi, yi, lo, hi, p in zip(x, y, ylo, yhi, mdf["pairs"]):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(xi, lo, hi, color="C0", alpha=0.7)
            ax.text(xi, yi, f" n={int(p)}", fontsize=8, va="bottom")
    um = between.get("unmatched", {})
    if um and np.isfinite(um.get("cliffs_delta", np.nan)):
        ax.axhline(um["cliffs_delta"], color="C3", linestyle="--", label="Unmatched")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Confound Caliper (z-units)")
    ax.set_ylabel("Cliff's delta (high-xi vs low-xi)\nOutcome: Lc/xi")
    ax.set_title("Effect Size vs Controls")
    ax.legend(loc="best", fontsize=8)

    # Panel 3: ratio histograms
    ax = axes[2]
    d = gal[(gal["source"] == "SPARC") & (gal["has_structure_metrics"] == True)].copy()
    rcols = [
        ("Lc_over_xi", "Lc/xi"),
        ("lambda_over_xi", "lambda/xi"),
        ("xi_over_Rext", "xi/Rext"),
    ]
    colors = ["C0", "C1", "C2"]
    any_hist = False
    for (c, label), col in zip(rcols, colors):
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) >= 5:
                ax.hist(np.log10(v), bins=14, alpha=0.45, color=col, label=label, density=True)
                any_hist = True
    if any_hist:
        ax.set_xlabel("log10(ratio)")
        ax.set_ylabel("Density")
        ax.set_title("Dimensionless Ratio Histograms")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient ratio data", ha="center", va="center")
        ax.set_title("Dimensionless Ratio Histograms")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_scale_map_report(
    out_md: str,
    gal: pd.DataFrame,
    within_df: pd.DataFrame,
    within_summary: Dict[str, float],
    between: Dict[str, object],
    outputs: Dict[str, str],
) -> None:
    d = gal[(gal["source"] == "SPARC") & (gal["has_structure_metrics"] == True)].copy()
    ratio_stats = {}
    for c in ["Lc_over_xi", "lambda_over_xi", "xi_over_Rext"]:
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce")
            v = v[np.isfinite(v)]
            if len(v) > 0:
                ratio_stats[c] = {
                    "n": int(len(v)),
                    "median": float(np.median(v)),
                    "p16": float(np.percentile(v, 16)),
                    "p84": float(np.percentile(v, 84)),
                }

    lines = []
    lines.append("# Scale Map Report")
    lines.append("")
    lines.append("## Data Integration")
    lines.append(f"- Canonical galaxies: **{len(gal)}** ({gal['source'].nunique()} sources)")
    lines.append(f"- Structured SPARC sample: **{int(d['gal_id'].nunique())}** galaxies")
    lines.append(f"- Output table: `{outputs['galaxy_table']}`")
    lines.append("")
    lines.append("## Within-Galaxy Test (Coherence vs R/xi)")
    if within_summary.get("n_gal", 0) > 0:
        lines.append(f"- Galaxies used: **{within_summary['n_gal']}**")
        lines.append(
            f"- x_peak concentration: observed std={within_summary['x_peak_std']:.3f}, "
            f"null mean std={within_summary['null_std_mean']:.3f}"
        )
        lines.append(
            f"- Xi-permutation p (smaller std than null): **{within_summary['perm_p_concentration']:.4f}**"
        )
    else:
        lines.append("- Insufficient valid sample for within-galaxy test.")
    lines.append("")
    lines.append("## Between-Galaxy Matched Test (High-xi vs Low-xi)")
    um = between.get("unmatched", {})
    if um:
        lines.append(
            f"- Unmatched Cliff's delta: **{um.get('cliffs_delta', np.nan):.3f}** "
            f"[{um.get('cliffs_ci95_lo', np.nan):.3f}, {um.get('cliffs_ci95_hi', np.nan):.3f}], "
            f"perm p={um.get('perm_p_median_diff', np.nan):.4f}"
        )
    pm = between.get("primary_matched", {})
    if pm:
        lines.append(
            f"- Primary matched (caliper={pm.get('caliper_z', np.nan):.2f}, pairs={int(pm.get('pairs', 0))}): "
            f"Cliff's delta={pm.get('cliffs_delta', np.nan):.3f} "
            f"[{pm.get('cliffs_ci95_lo', np.nan):.3f}, {pm.get('cliffs_ci95_hi', np.nan):.3f}], "
            f"perm p={pm.get('perm_p_median_diff', np.nan):.4f}"
        )
    else:
        lines.append("- No matched configuration met minimum pair count.")
    lines.append("")
    lines.append("## Dimensionless Scales")
    if ratio_stats:
        for k, v in ratio_stats.items():
            lines.append(
                f"- `{k}`: median={v['median']:.3f}, p16-p84=[{v['p16']:.3f}, {v['p84']:.3f}], n={v['n']}"
            )
    else:
        lines.append("- Ratio statistics unavailable.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- If x_peak concentration remains significant under xi-permutation null, xi is acting as an organizing radial scale.")
    lines.append("- If matched high-vs-low xi effect survives geometry/sampling controls, xi carries independent information beyond size/sampling artifacts.")
    lines.append("- Compare with universal-scale diagnostics (g† phase-peak tests) to maintain a two-scale interpretation: global acceleration scale + local coherence scale.")
    lines.append("")
    lines.append("## Output Files")
    for k, p in outputs.items():
        lines.append(f"- `{k}`: `{p}`")
    lines.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scale hierarchy map and run scale tests.")
    _repo_root = str(Path(__file__).resolve().parent.parent.parent)
    parser.add_argument(
        "--project-root",
        default=_repo_root,
        help="Project root path",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(_repo_root, "analysis", "results", "scale_hierarchy"),
        help="Output directory",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Permutation count for null tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 80)
    print("SCALE HIERARCHY MAP")
    print("=" * 80)
    print(f"Project root: {args.project_root}")
    print(f"Output dir: {args.outdir}")
    print(f"Permutations: {args.n_perm}")

    gal = build_canonical_galaxy_table(args.project_root)
    pts = build_point_table(args.project_root, gal)

    bins = np.array([0.0, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0], dtype=float)
    within_df, within_summary = run_within_galaxy_test(
        pts=pts,
        bins=bins,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    between = run_between_galaxy_tests(gal)

    # Merge per-gal within outputs
    if not within_df.empty:
        within_merge = within_df.rename(columns={"gal_id": "gal_id_w", "source": "source_w"})
        results = gal.merge(
            within_merge[["source_w", "gal_id_w", "x_peak", "peak_strength", "n_valid_bins"]],
            left_on=["source", "gal_id"],
            right_on=["source_w", "gal_id_w"],
            how="left",
        ).drop(columns=["source_w", "gal_id_w"])
    else:
        results = gal.copy()
        results["x_peak"] = np.nan
        results["peak_strength"] = np.nan
        results["n_valid_bins"] = np.nan

    out_gal = os.path.join(args.outdir, "galaxy_scale_table.parquet")
    out_pts = os.path.join(args.outdir, "point_scale_table.parquet")
    out_res = os.path.join(args.outdir, "results_scale_hierarchy.parquet")
    out_fig = os.path.join(args.outdir, "fig_scale_map.png")
    out_md = os.path.join(args.outdir, "scale_map_report.md")
    out_log = os.path.join(args.outdir, "run_log.json")

    gal.to_parquet(out_gal, index=False)
    pts.to_parquet(out_pts, index=False)
    results.to_parquet(out_res, index=False)

    make_figure(out_fig, within_df, within_summary, between, gal)

    outputs = {
        "galaxy_table": out_gal,
        "point_table": out_pts,
        "results_table": out_res,
        "figure": out_fig,
        "report": out_md,
        "run_log": out_log,
    }
    write_scale_map_report(out_md, gal, within_df, within_summary, between, outputs)

    run_log = {
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": args.project_root,
        "outdir": args.outdir,
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
        "git_hash": get_git_hash(args.project_root),
        "inputs": {
            "summary_healing_length_scaling": os.path.join(args.project_root, "analysis/results/summary_healing_length_scaling.json"),
            "summary_healing_length_acf": os.path.join(args.project_root, "analysis/results/summary_healing_length_acf.json"),
            "summary_healing_length_kstar": os.path.join(args.project_root, "analysis/results/summary_healing_length_kstar.json"),
            "summary_periodic_properties_and_scaling": os.path.join(args.project_root, "analysis/results/summary_periodic_properties_and_scaling.json"),
            "summary_window_matching": os.path.join(args.project_root, "analysis/results/summary_window_matching.json"),
            "rar_points_unified": os.path.join(args.project_root, "analysis/results/rar_points_unified.csv"),
            "galaxy_results_unified": os.path.join(args.project_root, "analysis/results/galaxy_results_unified.csv"),
            "galaxy_results_sparc_orig_haubner": os.path.join(args.project_root, "analysis/results/galaxy_results_sparc_orig_haubner.csv"),
            "tng_rar_points": os.path.join(args.project_root, "rar_points.parquet"),
            "tng_mass_profiles": os.path.join(args.project_root, "tng_mass_profiles.npz"),
        },
        "counts": {
            "n_galaxy_rows": int(len(gal)),
            "n_point_rows": int(len(pts)),
            "n_result_rows": int(len(results)),
            "n_scale_sample": int(np.sum(gal["scale_analysis_sample"] == True)),
            "n_within_valid": int(within_summary.get("n_gal", 0)),
            "n_between_sample": int(between.get("n_sample", 0)),
        },
        "within_summary": within_summary,
        "between_summary": between,
        "outputs": outputs,
    }
    with open(out_log, "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    print("\nKey Results")
    print(f"- Galaxy rows: {len(gal)}")
    print(f"- Point rows: {len(pts)}")
    print(f"- Structured SPARC sample: {int(np.sum(gal['scale_analysis_sample'] == True))}")
    print(f"- Within-gal valid: {within_summary.get('n_gal', 0)}")
    if within_summary.get("n_gal", 0) > 0:
        print(
            f"  x_peak std={within_summary['x_peak_std']:.3f}, "
            f"null_std_mean={within_summary['null_std_mean']:.3f}, "
            f"perm p={within_summary['perm_p_concentration']:.4f}"
        )
    pm = between.get("primary_matched", {})
    if pm:
        print(
            f"- Matched effect (primary): caliper={pm.get('caliper_z')}, pairs={pm.get('pairs')}, "
            f"Cliff's delta={pm.get('cliffs_delta'):.3f}, p={pm.get('perm_p_median_diff'):.4f}"
        )
    print("\nSaved:")
    for k, p in outputs.items():
        print(f"- {k}: {p}")


if __name__ == "__main__":
    main()
