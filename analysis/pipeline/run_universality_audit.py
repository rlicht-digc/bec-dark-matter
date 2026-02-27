#!/usr/bin/env python3
"""
Universality + GPP invariant audit (SPARC + TNG datasets).

This script implements:
  - Phase 0: project discovery map
  - Phase 1: canonical galaxy/residual dataframes
  - Phase 2: alpha/beta invariant sweeps + ratio residual test
  - Phase 3: residual-shape PCA with missing-bin handling
  - Phase 4: minimal boundary transparency model
  - Phase 5: falsification controls
  - Phase 6: output tables, plots, and markdown report

Writes ONLY under:
  /Users/russelllicht/bec-dark-matter/outputs/universality_audit/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "universality_audit"

G_SI = 6.67430e-11
KPC_M = 3.085677581e19
MSUN_KG = 1.98847e30
G_DAG = 1.2e-10
LOG_G_DAG = float(np.log10(G_DAG))

ALPHAS = np.round(np.arange(-2.0, 2.0 + 1e-12, 0.01), 2)
BETAS = np.round(np.arange(-4.0, 4.0 + 1e-12, 0.02), 2)

RNG = np.random.default_rng(42)
MAX_CONTROL_N = 8000


@dataclass
class DatasetPack:
    name: str
    df_gal: pd.DataFrame
    df_points: pd.DataFrame
    id_col: str


def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def spearman_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    out = spearmanr(x[m], y[m])
    return float(out.statistic), float(out.pvalue)


def rar_pred_log_gobs(log_gbar: np.ndarray, gdag: float = G_DAG) -> np.ndarray:
    gbar = 10.0 ** np.asarray(log_gbar, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pred = np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / gdag))))
    return pred


def compute_mbar_proxy_msun(log_gbar: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    gbar = 10.0 ** np.asarray(log_gbar, dtype=float)
    r_m = np.asarray(r_kpc, dtype=float) * KPC_M
    mbar_kg = gbar * r_m**2 / G_SI
    return mbar_kg / MSUN_KG


def phase0_project_map() -> Dict[str, object]:
    top_dirs = sorted([p.name for p in REPO_ROOT.iterdir() if p.is_dir()])
    notebooks = sorted([str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*.ipynb")])
    scripts = sorted([str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*.py")])

    keyword_pat = "RAR|gbar|gobs|g_dagger|g†|cs2|c_s|healing length|xi|SPARC|TNG|SubhaloID|df_binned"
    rg_hits = []
    try:
        cmd = [
            "rg",
            "-n",
            "--glob",
            "*.py",
            "--glob",
            "*.ipynb",
            keyword_pat,
            str(REPO_ROOT),
        ]
        out = subprocess.run(cmd, check=False, capture_output=True, text=True)
        lines = out.stdout.strip().splitlines()
        rg_hits = lines[:300]
    except Exception:
        rg_hits = ["rg unavailable; skipped keyword search"]

    map_obj = {
        "repo_root": str(REPO_ROOT),
        "top_level_directories": top_dirs,
        "notebooks_count": len(notebooks),
        "scripts_count": len(scripts),
        "notebooks_sample": notebooks[:120],
        "scripts_sample": scripts[:120],
        "keyword_hits_sample": rg_hits,
    }

    lines = []
    lines.append("# Project Map")
    lines.append("")
    lines.append(f"- repo root: `{REPO_ROOT}`")
    lines.append(f"- top-level dirs: {', '.join(top_dirs)}")
    lines.append(f"- notebooks found: {len(notebooks)}")
    lines.append(f"- scripts found: {len(scripts)}")
    lines.append("")
    lines.append("## Keyword hit sample")
    lines.extend([f"- `{x}`" for x in rg_hits[:80]])
    (OUT_DIR / "project_map.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "project_map.json").write_text(json.dumps(map_obj, indent=2), encoding="utf-8")

    print("\n[PHASE 0] Project map")
    print(f"  repo: {REPO_ROOT}")
    print(f"  top dirs: {len(top_dirs)}")
    print(f"  notebooks: {len(notebooks)}")
    print(f"  scripts: {len(scripts)}")
    print(f"  keyword hits(sample): {len(rg_hits)}")
    return map_obj


def build_sparc_pack() -> DatasetPack:
    p_cs2 = REPO_ROOT / "analysis" / "results" / "gpp_galaxy_cs2_summary.csv"
    p_pts = REPO_ROOT / "analysis" / "results" / "rar_points_unified.csv"
    p_gal = REPO_ROOT / "analysis" / "results" / "galaxy_results_unified.csv"

    cs2 = pd.read_csv(p_cs2)
    pts = pd.read_csv(p_pts)
    gal = pd.read_csv(p_gal)

    cs2 = cs2[(cs2["source"] == "SPARC") & np.isfinite(cs2["cs2_median"]) & (cs2["cs2_median"] > 0)].copy()
    pts = pts[(pts["source"] == "SPARC") & np.isfinite(pts["log_gbar"]) & np.isfinite(pts["log_gobs"])].copy()
    gal_s = gal[gal["source"] == "SPARC"].copy()

    recs = []
    for gid, g in pts.groupby("galaxy"):
        g = g.sort_values("R_kpc")
        top = g.tail(3)
        mbar = compute_mbar_proxy_msun(top["log_gbar"].to_numpy(), top["R_kpc"].to_numpy())
        mbar = mbar[np.isfinite(mbar) & (mbar > 0)]
        mbar_proxy = float(np.median(mbar)) if mbar.size else np.nan
        recs.append(
            {
                "galaxy": gid,
                "Mb_proxy_msun": mbar_proxy,
                "log_Mb": np.log10(mbar_proxy) if np.isfinite(mbar_proxy) and mbar_proxy > 0 else np.nan,
                "mean_log_gbar_pts": float(g["log_gbar"].mean()),
                "n_points_pts": int(len(g)),
                "env_dense_pts": g["env_dense"].iloc[0] if len(g) else np.nan,
            }
        )
    mass_df = pd.DataFrame(recs)

    df = cs2.merge(mass_df, on="galaxy", how="left")
    df = df.merge(
        gal_s[["galaxy", "mean_log_gbar", "n_points", "env_dense", "logMh"]], on="galaxy", how="left", suffixes=("", "_gal")
    )
    df["mean_log_gbar"] = df["mean_log_gbar"].fillna(df["mean_log_gbar_pts"])
    df["env_label"] = df["env_dense"].fillna(df["env_dense_pts"])
    df["env_bin"] = df["env_label"].map({"field": 0, "dense": 1})
    df["env_code"] = df["env_bin"]
    df["xi_m"] = np.sqrt(np.maximum(G_SI * (df["Mb_proxy_msun"] * MSUN_KG), 0) / G_DAG)
    df["log_xi"] = np.log10(df["xi_m"])
    df["log_cs2"] = np.log10(df["cs2_median"])
    df["dataset"] = "SPARC"
    df["id"] = df["galaxy"].astype(str)

    use_cols = [
        "dataset",
        "id",
        "galaxy",
        "source",
        "log_cs2",
        "cs2_median",
        "log_Mb",
        "Mb_proxy_msun",
        "xi_m",
        "log_xi",
        "env_label",
        "env_bin",
        "env_code",
        "mean_log_gbar",
        "n_points",
        "n_points_pts",
        "logMh",
    ]
    df = df[use_cols].copy()
    df = df[np.isfinite(df["log_cs2"]) & np.isfinite(df["log_Mb"]) & np.isfinite(df["mean_log_gbar"])].copy()

    pts_use = pts[pts["galaxy"].isin(df["galaxy"])].copy()
    pts_use["id"] = pts_use["galaxy"].astype(str)
    pts_use["dataset"] = "SPARC"

    print("\n[PHASE 1] SPARC dataframe")
    print(f"  galaxies: {df['id'].nunique()} | points: {len(pts_use)}")
    return DatasetPack(name="SPARC", df_gal=df, df_points=pts_use, id_col="id")


def compute_cs2_proxy_from_points(g: pd.DataFrame) -> Tuple[float, int]:
    g = g.sort_values("r_kpc")
    r_kpc = g["r_kpc"].to_numpy(dtype=float)
    lgbar = g["log_gbar"].to_numpy(dtype=float)
    lgobs = g["log_gobs"].to_numpy(dtype=float)

    gbar = 10.0 ** lgbar
    gobs = 10.0 ** lgobs
    gdm = gobs - gbar

    m = np.isfinite(r_kpc) & np.isfinite(gdm) & (r_kpc > 0) & (gdm > 0)
    if m.sum() < 5:
        return np.nan, 0

    r_m = r_kpc[m] * KPC_M
    gdm = gdm[m]
    menc = gdm * r_m**2 / G_SI
    menc = np.maximum(menc, 1e-30)

    lnr = np.log(r_m)
    lnm = np.log(menc)
    if len(lnr) < 3:
        return np.nan, 0
    dlnm_dlnr = np.gradient(lnm, lnr)
    v = np.isfinite(dlnm_dlnr) & (np.abs(dlnm_dlnr) > 0.01)
    if v.sum() < 3:
        return np.nan, 0

    cs2 = np.abs(gdm[v]) * r_m[v] / np.abs(dlnm_dlnr[v])
    cs2 = cs2[np.isfinite(cs2) & (cs2 > 0)]
    if cs2.size < 3:
        return np.nan, 0
    return float(np.median(cs2)), int(cs2.size)


def build_tng_dev_pack() -> DatasetPack:
    p_pts = REPO_ROOT / "datasets" / "TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED" / "rar_points_CLEAN.parquet"
    p_master = REPO_ROOT / "datasets" / "TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED" / "meta" / "master_catalog.csv"
    p_env = REPO_ROOT / "datasets" / "TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED" / "galaxy_scatter_dm_with_env.csv"

    pts = pd.read_parquet(p_pts).copy()
    master = pd.read_csv(p_master).copy()
    env = pd.read_csv(p_env).copy() if p_env.exists() else pd.DataFrame({"SubhaloID": []})

    pts = pts.rename(columns={"SubhaloID": "id", "r_kpc": "r_kpc", "log_gbar": "log_gbar", "log_gobs": "log_gobs"})
    master = master.rename(columns={"SubhaloID": "id"})
    env = env.rename(columns={"SubhaloID": "id"})

    rows = []
    for gid, g in pts.groupby("id"):
        cs2_med, n_valid = compute_cs2_proxy_from_points(g)
        if not np.isfinite(cs2_med) or cs2_med <= 0:
            continue
        gg = g.sort_values("r_kpc")
        top = gg.tail(3)
        mbar = compute_mbar_proxy_msun(top["log_gbar"].to_numpy(), top["r_kpc"].to_numpy())
        mbar = mbar[np.isfinite(mbar) & (mbar > 0)]
        mbar_proxy = float(np.median(mbar)) if mbar.size else np.nan
        rows.append(
            {
                "id": int(gid),
                "cs2_median": cs2_med,
                "log_cs2": np.log10(cs2_med),
                "n_points_cs2": n_valid,
                "Mb_proxy_msun": mbar_proxy,
                "log_Mb": np.log10(mbar_proxy) if np.isfinite(mbar_proxy) and mbar_proxy > 0 else np.nan,
                "mean_log_gbar": float(gg["log_gbar"].mean()),
            }
        )
    base = pd.DataFrame(rows)
    env_cols = [c for c in ["id", "env", "logM200c", "M200c_Msun", "n_dm_pts", "sigma_dm_robust"] if c in env.columns]
    if env_cols:
        df = base.merge(master, on="id", how="left").merge(env[env_cols], on="id", how="left").copy()
    else:
        df = base.merge(master, on="id", how="left").copy()

    env_code_map = {"field(<1e12.5)": 0, "group(1e12.5-1e14)": 1, "cluster(>1e14)": 2}
    df["env_label"] = df["env"] if "env" in df.columns else np.nan
    df["env_code"] = df["env_label"].map(env_code_map)
    if "logM200c" in df.columns:
        # Fallback keeps env correlation tests defined even if categorical labels are missing.
        df["env_code"] = df["env_code"].fillna(df["logM200c"])
    df["env_bin"] = np.where(df["env_code"].fillna(0) > 0, 1, 0)
    df["xi_m"] = np.sqrt(np.maximum(G_SI * (df["Mb_proxy_msun"] * MSUN_KG), 0) / G_DAG)
    df["log_xi"] = np.log10(df["xi_m"])
    df["dataset"] = "TNG_DEV3000"
    df["source"] = "TNG100-1"

    use_cols = [
        "dataset",
        "id",
        "source",
        "log_cs2",
        "cs2_median",
        "log_Mb",
        "Mb_proxy_msun",
        "xi_m",
        "log_xi",
        "env_label",
        "env_bin",
        "env_code",
        "mean_log_gbar",
        "n_points_cs2",
        "Mstar_Msun",
        "SFR",
        "Rhalf_star_kpc",
        "logM200c",
        "M200c_Msun",
        "n_dm_pts",
        "sigma_dm_robust",
    ]
    use_cols = [c for c in use_cols if c in df.columns]
    df = df[use_cols].copy()
    df["id"] = df["id"].astype(str)
    df = df[np.isfinite(df["log_cs2"]) & np.isfinite(df["log_Mb"]) & np.isfinite(df["mean_log_gbar"])].copy()

    pts["id"] = pts["id"].astype(str)
    pts_use = pts[pts["id"].isin(df["id"])].copy()
    pts_use["id"] = pts_use["id"].astype(str)
    pts_use["dataset"] = "TNG_DEV3000"

    print("\n[PHASE 1] TNG dataframe")
    print(f"  galaxies: {df['id'].nunique()} | points: {len(pts_use)}")
    return DatasetPack(name="TNG_DEV3000", df_gal=df, df_points=pts_use, id_col="id")


def build_tng_big_pack() -> DatasetPack | None:
    p_pts = REPO_ROOT / "datasets" / "TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE" / "rar_points.parquet"
    p_master = REPO_ROOT / "datasets" / "TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE" / "meta" / "master_catalog.csv"

    if not p_pts.exists() or not p_master.exists():
        print("\n[PHASE 1] TNG big-base dataframe")
        print("  dataset missing; skipping TNG_BIG48133")
        return None

    pts = pd.read_parquet(p_pts).copy()
    master = pd.read_csv(p_master).copy()
    pts = pts.rename(columns={"SubhaloID": "id", "r_kpc": "r_kpc", "log_gbar": "log_gbar", "log_gobs": "log_gobs"})
    master = master.rename(columns={"SubhaloID": "id"})

    rows = []
    for gid, g in pts.groupby("id"):
        cs2_med, n_valid = compute_cs2_proxy_from_points(g)
        if not np.isfinite(cs2_med) or cs2_med <= 0:
            continue
        gg = g.sort_values("r_kpc")
        top = gg.tail(3)
        mbar = compute_mbar_proxy_msun(top["log_gbar"].to_numpy(), top["r_kpc"].to_numpy())
        mbar = mbar[np.isfinite(mbar) & (mbar > 0)]
        mbar_proxy = float(np.median(mbar)) if mbar.size else np.nan
        rows.append(
            {
                "id": int(gid),
                "cs2_median": cs2_med,
                "log_cs2": np.log10(cs2_med),
                "n_points_cs2": n_valid,
                "Mb_proxy_msun": mbar_proxy,
                "log_Mb": np.log10(mbar_proxy) if np.isfinite(mbar_proxy) and mbar_proxy > 0 else np.nan,
                "mean_log_gbar": float(gg["log_gbar"].mean()),
            }
        )
    base = pd.DataFrame(rows)
    df = base.merge(master, on="id", how="left").copy()

    # Big base has no explicit env labels; use host-group occupancy as a local-density proxy.
    if "SubhaloGrNr" in df.columns:
        grp_n = df.groupby("SubhaloGrNr")["id"].transform("count")
        df["group_member_count"] = grp_n.astype(float)
        df["env_code"] = np.log10(np.clip(df["group_member_count"], 1.0, None))
        med_env = float(np.nanmedian(df["env_code"]))
        df["env_bin"] = (df["env_code"] >= med_env).astype(int)
        df["env_label"] = np.where(df["env_bin"] == 1, "high_group_density", "low_group_density")
    else:
        df["group_member_count"] = np.nan
        df["env_code"] = np.nan
        df["env_bin"] = np.nan
        df["env_label"] = "unknown"

    df["xi_m"] = np.sqrt(np.maximum(G_SI * (df["Mb_proxy_msun"] * MSUN_KG), 0) / G_DAG)
    df["log_xi"] = np.log10(df["xi_m"])
    df["dataset"] = "TNG_BIG48133"
    df["source"] = "TNG100-1"

    use_cols = [
        "dataset",
        "id",
        "source",
        "log_cs2",
        "cs2_median",
        "log_Mb",
        "Mb_proxy_msun",
        "xi_m",
        "log_xi",
        "env_label",
        "env_bin",
        "env_code",
        "group_member_count",
        "mean_log_gbar",
        "n_points_cs2",
        "SubhaloGrNr",
        "Mstar_Msun",
        "SFR",
        "Rhalf_star_kpc",
    ]
    use_cols = [c for c in use_cols if c in df.columns]
    df = df[use_cols].copy()
    df["id"] = df["id"].astype(str)
    df = df[np.isfinite(df["log_cs2"]) & np.isfinite(df["log_Mb"]) & np.isfinite(df["mean_log_gbar"])].copy()

    pts["id"] = pts["id"].astype(str)
    pts_use = pts[pts["id"].isin(df["id"])].copy()
    pts_use["id"] = pts_use["id"].astype(str)
    pts_use["dataset"] = "TNG_BIG48133"

    print("\n[PHASE 1] TNG big-base dataframe")
    print(f"  galaxies: {df['id'].nunique()} | points: {len(pts_use)}")
    return DatasetPack(name="TNG_BIG48133", df_gal=df, df_points=pts_use, id_col="id")


def build_resid_df(points: pd.DataFrame, dataset: str, id_col: str, bin_edges: np.ndarray) -> pd.DataFrame:
    p = points.copy()
    p["log_rar_pred"] = rar_pred_log_gobs(p["log_gbar"].to_numpy())
    p["delta"] = p["log_gobs"] - p["log_rar_pred"]
    p["bin_idx"] = pd.cut(p["log_gbar"], bins=bin_edges, labels=False, include_lowest=True)
    p = p[np.isfinite(p["delta"]) & p["bin_idx"].notna()].copy()
    p["bin_idx"] = p["bin_idx"].astype(int)
    p = p.rename(columns={id_col: "id"})
    p["id"] = p["id"].astype(str)
    agg = (
        p.groupby(["id", "bin_idx"], as_index=False)
        .agg(delta_mean=("delta", "mean"), delta_median=("delta", "median"), n_in_bin=("delta", "size"))
    )
    ids = pd.Index(sorted(p["id"].unique()), name="id")
    bins = pd.Index(np.arange(len(bin_edges) - 1), name="bin_idx")
    full = pd.MultiIndex.from_product([ids, bins], names=["id", "bin_idx"]).to_frame(index=False)
    out = full.merge(agg, on=["id", "bin_idx"], how="left")
    out["dataset"] = dataset
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    out["bin_center"] = out["bin_idx"].map({i: float(c) for i, c in enumerate(centers)})
    return out


def run_sweep(y: np.ndarray, x: np.ndarray, env: np.ndarray, grid: np.ndarray, grid_name: str, dataset: str) -> pd.DataFrame:
    rows = []
    for val in grid:
        z = y + val * x
        sig = robust_sigma(z)
        rho_m, p_m = spearman_safe(z, x)
        rho_e, p_e = spearman_safe(z, env)
        rows.append(
            {
                "dataset": dataset,
                grid_name: float(val),
                "robust_sigma": float(sig) if np.isfinite(sig) else np.nan,
                "rho_mass": rho_m,
                "p_mass": p_m,
                "rho_env": rho_e,
                "p_env": p_e,
            }
        )
    return pd.DataFrame(rows)


def pick_best_from_sweep(df: pd.DataFrame, xcol: str) -> Dict[str, object]:
    d = df.copy()
    d = d[np.isfinite(d["robust_sigma"])]
    raw_row = d.iloc[(d[xcol] - 0.0).abs().argmin()]
    constrained = d[np.abs(d["rho_mass"]) <= 0.05]
    if len(constrained) > 0:
        best = constrained.loc[constrained["robust_sigma"].idxmin()]
        used_constraint = True
    else:
        best = d.loc[d["robust_sigma"].idxmin()]
        used_constraint = False
    pct_drop = 100.0 * (raw_row["robust_sigma"] - best["robust_sigma"]) / raw_row["robust_sigma"]
    return {
        "best_param": float(best[xcol]),
        "best_sigma": float(best["robust_sigma"]),
        "raw_sigma": float(raw_row["robust_sigma"]),
        "scatter_drop_pct": float(pct_drop),
        "rho_mass_at_best": float(best["rho_mass"]),
        "rho_env_at_best": float(best["rho_env"]) if np.isfinite(best["rho_env"]) else np.nan,
        "used_mass_constraint": bool(used_constraint),
    }


def linear_fit_ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        return np.nan, np.nan
    x0 = float(x.mean())
    y0 = float(y.mean())
    vx = float(np.dot(x - x0, x - x0))
    if vx <= 0:
        return y0, 0.0
    slope = float(np.dot(x - x0, y - y0) / vx)
    intercept = y0 - slope * x0
    return intercept, slope


def linear_fit_huber(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3:
        return np.nan, np.nan

    xm = float(np.median(x))
    xs = float(np.std(x))
    if not np.isfinite(xs) or xs <= 0:
        xs = 1.0
    xn = (x - xm) / xs
    p0 = np.array([float(np.median(y)), 0.0], dtype=float)
    fit = least_squares(
        lambda p: y - (p[0] + p[1] * xn),
        p0,
        method="trf",
        loss="huber",
        f_scale=1.0,
    )
    intercept = float(fit.x[0] - fit.x[1] * xm / xs)
    slope = float(fit.x[1] / xs)
    return intercept, slope


def fit_ratio_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    d = df.copy()
    d = d[np.isfinite(d["log_cs2"]) & np.isfinite(d["log_Mb"])].copy()
    x = d["log_Mb"].to_numpy()
    y = d["log_cs2"].to_numpy()
    if len(d) < 5:
        return d.assign(pred=np.nan, residual=np.nan), {"n": int(len(d)), "status": "too_few"}

    intercept, slope = linear_fit_huber(x, y)
    pred = intercept + slope * x
    resid = y - pred
    d["pred"] = pred
    d["residual"] = resid

    rho_m, p_m = spearman_safe(resid, x)
    rho_e, p_e = spearman_safe(resid, d["env_code"].to_numpy(dtype=float))
    sig_res = robust_sigma(resid)

    out = {
        "n": int(len(d)),
        "slope_robust": float(slope),
        "intercept_robust": float(intercept),
        "resid_sigma": float(sig_res),
        "rho_resid_mass": rho_m,
        "p_resid_mass": p_m,
        "rho_resid_env": rho_e,
        "p_resid_env": p_e,
    }
    return d, out


def em_pca_fill(X_nan: np.ndarray, n_components: int = 2, n_iter: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X_nan, dtype=float)
    mask = ~np.isfinite(X)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    Xf = X.copy()
    Xf[mask] = np.take(col_means, np.where(mask)[1])

    for _ in range(n_iter):
        mu = Xf.mean(axis=0)
        Xc = Xf - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Xc_hat = (U[:, :n_components] * S[:n_components]) @ Vt[:n_components, :]
        Xhat = Xc_hat + mu
        Xf[mask] = Xhat[mask]

    mu = Xf.mean(axis=0)
    Xc = Xf - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S**2) / max(Xc.shape[0] - 1, 1)
    evr = eigvals / eigvals.sum() if eigvals.sum() > 0 else np.full_like(eigvals, np.nan)
    scores = U[:, :n_components] * S[:n_components]
    comps = Vt[:n_components, :]
    return Xf, scores, comps, evr


def mass_matched_env_effect(y: np.ndarray, logm: np.ndarray, env_bin: np.ndarray, n_bins: int = 6) -> Dict[str, float]:
    m = np.isfinite(y) & np.isfinite(logm) & np.isfinite(env_bin)
    y = y[m]
    logm = logm[m]
    env_bin = env_bin[m]
    if len(y) < 10:
        return {"effect": np.nan, "n_used": int(len(y))}

    qs = np.quantile(logm, np.linspace(0, 1, n_bins + 1))
    effects = []
    weights = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mm = (logm >= qs[i]) & (logm < qs[i + 1])
        else:
            mm = (logm >= qs[i]) & (logm <= qs[i + 1])
        if mm.sum() < 4:
            continue
        yi = y[mm]
        ei = env_bin[mm]
        if (ei == 0).sum() < 2 or (ei == 1).sum() < 2:
            continue
        diff = np.mean(yi[ei == 1]) - np.mean(yi[ei == 0])
        effects.append(diff)
        weights.append(mm.sum())
    if len(effects) == 0:
        return {"effect": np.nan, "n_used": int(len(y))}
    effects = np.asarray(effects)
    weights = np.asarray(weights, dtype=float)
    return {"effect": float(np.average(effects, weights=weights)), "n_used": int(len(y))}


def fit_boundary_model(y: np.ndarray, logm: np.ndarray, mean_log_gbar: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y) & np.isfinite(logm) & np.isfinite(mean_log_gbar)
    y = y[m]
    x = logm[m]
    lg = mean_log_gbar[m]
    n = len(y)
    if n < 10:
        return {"n": int(n), "status": "too_few"}

    def null_pred(p):
        return p[0] + p[1] * x

    p0_null = np.array([np.median(y), 0.0], dtype=float)
    fit_null = least_squares(lambda p: y - null_pred(p), p0_null, method="trf")
    rss_null = float(np.sum((y - null_pred(fit_null.x)) ** 2))
    rss_null = max(rss_null, 1e-12)
    k0 = 2
    aic0 = n * np.log(rss_null / n) + 2 * k0
    bic0 = n * np.log(rss_null / n) + k0 * np.log(n)

    def boundary_pred(p):
        a, b, c, log_gdag, s = p
        sigma = 1.0 / (1.0 + np.exp(-s * (lg - log_gdag)))
        return a + b * x + c * sigma

    p0 = np.array([np.median(y), 0.0, 0.0, LOG_G_DAG, 3.0], dtype=float)
    lb = np.array([-np.inf, -np.inf, -np.inf, -12.0, 0.05], dtype=float)
    ub = np.array([np.inf, np.inf, np.inf, -8.0, 30.0], dtype=float)
    fit_b = least_squares(lambda p: y - boundary_pred(p), p0, bounds=(lb, ub), method="trf")
    rss_b = float(np.sum((y - boundary_pred(fit_b.x)) ** 2))
    rss_b = max(rss_b, 1e-12)
    k1 = 5
    aic1 = n * np.log(rss_b / n) + 2 * k1
    bic1 = n * np.log(rss_b / n) + k1 * np.log(n)

    return {
        "n": int(n),
        "a_null": float(fit_null.x[0]),
        "b_null": float(fit_null.x[1]),
        "rss_null": rss_null,
        "aic_null": float(aic0),
        "bic_null": float(bic0),
        "a_boundary": float(fit_b.x[0]),
        "b_boundary": float(fit_b.x[1]),
        "c_boundary": float(fit_b.x[2]),
        "log_gdag_fit": float(fit_b.x[3]),
        "gdag_fit": float(10.0 ** fit_b.x[3]),
        "s_fit": float(fit_b.x[4]),
        "rss_boundary": rss_b,
        "aic_boundary": float(aic1),
        "bic_boundary": float(bic1),
        "delta_aic_boundary_minus_null": float(aic1 - aic0),
        "delta_bic_boundary_minus_null": float(bic1 - bic0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run universality + GPP invariant audit.")
    p.add_argument(
        "--max-control-n",
        type=int,
        default=MAX_CONTROL_N,
        help="Max galaxies per dataset for controls. Use 0 for full no-subsampling controls.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="SPARC,TNG_DEV3000,TNG_BIG48133",
        help="Comma-separated datasets to run from: SPARC,TNG_DEV3000,TNG_BIG48133",
    )
    return p.parse_args()


def alpha_best(y: np.ndarray, x: np.ndarray, grid: np.ndarray) -> Tuple[float, float, float]:
    m = np.isfinite(y) & np.isfinite(x)
    y = y[m]
    x = x[m]
    Z = np.outer(x, grid) + y[:, None]
    med = np.median(Z, axis=0)
    mad = np.median(np.abs(Z - med), axis=0)
    sig = 1.4826 * mad
    idx = int(np.nanargmin(sig))
    raw = sig[np.argmin(np.abs(grid - 0.0))]
    return float(grid[idx]), float(sig[idx]), float(raw)


def run_controls(dataset: str, df: pd.DataFrame, ratio_df: pd.DataFrame, max_control_n: int) -> Dict[str, object]:
    d = df[np.isfinite(df["log_cs2"]) & np.isfinite(df["log_Mb"])].copy()
    if max_control_n > 0 and len(d) > max_control_n:
        d_ctrl = d.sample(n=max_control_n, random_state=42).copy()
        controls_subsampled = True
    else:
        d_ctrl = d.copy()
        controls_subsampled = False

    y = d_ctrl["log_cs2"].to_numpy(dtype=float)
    x = d_ctrl["log_Mb"].to_numpy(dtype=float)
    env = d_ctrl["env_bin"].to_numpy(dtype=float)
    out = {
        "dataset": dataset,
        "n_total": int(len(d)),
        "n_used_controls": int(len(d_ctrl)),
        "controls_subsampled": controls_subsampled,
    }

    # 1) Synthetic injection
    syn_rows = []
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    gamma_vals = [0.0, 0.3, 0.6]
    n_rep = 100
    x0 = float(np.median(x))
    for gamma in gamma_vals:
        for noise in noise_levels:
            for rep in range(n_rep):
                y_syn = 9.5 + gamma * (x - x0) + RNG.normal(0.0, noise, size=len(x))
                _, slope = linear_fit_ols(x, y_syn)
                a_hat, smin, sraw = alpha_best(y_syn, x, ALPHAS)
                syn_rows.append(
                    {
                        "dataset": dataset,
                        "gamma_true": gamma,
                        "noise_sigma": noise,
                        "rep": rep,
                        "gamma_hat": float(slope),
                        "alpha_hat": a_hat,
                        "expected_alpha": -gamma,
                        "alpha_error": a_hat + gamma,
                        "scatter_min": smin,
                        "scatter_raw": sraw,
                    }
                )
    syn_df = pd.DataFrame(syn_rows)
    syn_df.to_csv(OUT_DIR / f"controls_synthetic_injection_{dataset}.csv", index=False)
    out["synthetic"] = {
        "n_rows": int(len(syn_df)),
        "n_rep_per_combo": n_rep,
        "summary": syn_df.groupby(["gamma_true", "noise_sigma"], as_index=False)[["gamma_hat", "alpha_hat", "alpha_error"]]
        .mean()
        .to_dict(orient="records"),
    }

    # 2) Shuffle null (N=1000)
    a_real, s_real, s_raw = alpha_best(y, x, ALPHAS)
    XA = np.outer(x, ALPHAS)
    null_rows = []
    n_perm = 1000
    for i in range(n_perm):
        yp = RNG.permutation(y)
        Z = XA + yp[:, None]
        med = np.median(Z, axis=0)
        mad = np.median(np.abs(Z - med), axis=0)
        sig = 1.4826 * mad
        j = int(np.argmin(sig))
        null_rows.append({"dataset": dataset, "perm": i, "best_alpha": float(ALPHAS[j]), "min_sigma": float(sig[j])})
        if (i + 1) % 200 == 0:
            print(f"  [control:shuffle] {dataset} {i+1}/{n_perm}")
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(OUT_DIR / f"controls_shuffle_null_{dataset}.csv", index=False)
    p_shuffle = float((null_df["min_sigma"] <= s_real).mean())
    out["shuffle_null"] = {
        "n_perm": n_perm,
        "alpha_real": a_real,
        "sigma_real": s_real,
        "sigma_raw": s_raw,
        "p_value": p_shuffle,
    }

    # 3) Split-half stability
    sh_rows = []
    n_split = 100
    idx = np.arange(len(y))
    for i in range(n_split):
        RNG.shuffle(idx)
        h = len(idx) // 2
        i1 = idx[:h]
        i2 = idx[h:]
        a1, s1, _ = alpha_best(y[i1], x[i1], ALPHAS)
        a2, s2, _ = alpha_best(y[i2], x[i2], ALPHAS)
        sh_rows.append({"dataset": dataset, "iter": i, "alpha_half1": a1, "alpha_half2": a2, "sigma_half1": s1, "sigma_half2": s2})
    sh_df = pd.DataFrame(sh_rows)
    sh_df.to_csv(OUT_DIR / f"controls_split_half_{dataset}.csv", index=False)
    out["split_half"] = {
        "n_iter": n_split,
        "alpha_half1_std": float(sh_df["alpha_half1"].std()),
        "alpha_half2_std": float(sh_df["alpha_half2"].std()),
        "alpha_half_diff_mean_abs": float((sh_df["alpha_half1"] - sh_df["alpha_half2"]).abs().mean()),
    }

    # 4) Duplication invariance
    a0, s0, _ = alpha_best(y, x, ALPHAS)
    y_all = np.concatenate([y, y])
    x_all = np.concatenate([x, x])
    a_all, s_all, _ = alpha_best(y_all, x_all, ALPHAS)
    high = x >= np.quantile(x, 0.75)
    y_hi = np.concatenate([y, y[high], y[high], y[high]])
    x_hi = np.concatenate([x, x[high], x[high], x[high]])
    a_hi, s_hi, _ = alpha_best(y_hi, x_hi, ALPHAS)
    dup_df = pd.DataFrame(
        [
            {"dataset": dataset, "case": "original", "alpha": a0, "sigma": s0},
            {"dataset": dataset, "case": "duplicate_all", "alpha": a_all, "sigma": s_all},
            {"dataset": dataset, "case": "duplicate_highmass_x3", "alpha": a_hi, "sigma": s_hi},
        ]
    )
    dup_df.to_csv(OUT_DIR / f"controls_duplication_{dataset}.csv", index=False)
    out["duplication"] = {
        "alpha_original": a0,
        "alpha_duplicate_all": a_all,
        "alpha_duplicate_highmass": a_hi,
        "alpha_shift_all": float(a_all - a0),
        "alpha_shift_highmass": float(a_hi - a0),
    }

    # 5) Environment swap tests on ratio residual effect
    rr = ratio_df.copy()
    rr = rr[np.isfinite(rr["residual"]) & np.isfinite(rr["env_bin"]) & np.isfinite(rr["log_Mb"])].copy()
    if controls_subsampled and "id" in rr.columns:
        rr = rr[rr["id"].astype(str).isin(set(d_ctrl["id"].astype(str)))].copy()
    if len(rr) >= 20 and rr["env_bin"].nunique() >= 2:
        obs = mass_matched_env_effect(rr["residual"].to_numpy(), rr["log_Mb"].to_numpy(), rr["env_bin"].to_numpy(), n_bins=6)["effect"]
        n_perm_env = 1000
        glob = []
        strat = []
        mb = rr["log_Mb"].to_numpy()
        yy = rr["residual"].to_numpy()
        ee = rr["env_bin"].to_numpy().astype(int)
        q = np.quantile(mb, np.linspace(0, 1, 7))
        for i in range(n_perm_env):
            eperm = RNG.permutation(ee)
            glob.append(mass_matched_env_effect(yy, mb, eperm, n_bins=6)["effect"])

            e2 = ee.copy()
            for b in range(6):
                if b < 5:
                    msk = (mb >= q[b]) & (mb < q[b + 1])
                else:
                    msk = (mb >= q[b]) & (mb <= q[b + 1])
                idxb = np.where(msk)[0]
                if len(idxb) > 1:
                    e2[idxb] = RNG.permutation(e2[idxb])
            strat.append(mass_matched_env_effect(yy, mb, e2, n_bins=6)["effect"])
        env_df = pd.DataFrame(
            {
                "dataset": dataset,
                "perm": np.arange(n_perm_env),
                "effect_global_swap": glob,
                "effect_stratified_swap": strat,
            }
        )
        env_df.to_csv(OUT_DIR / f"controls_env_swap_{dataset}.csv", index=False)
        pg = float(np.mean(np.abs(env_df["effect_global_swap"]) >= abs(obs)))
        ps = float(np.mean(np.abs(env_df["effect_stratified_swap"]) >= abs(obs)))
        out["env_swap"] = {
            "observed_effect": float(obs),
            "n_perm": n_perm_env,
            "p_global_swap": pg,
            "p_stratified_swap": ps,
        }
    else:
        out["env_swap"] = {"status": "insufficient_env_data"}

    # plots: shuffle histogram + split-half alpha scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_df["min_sigma"], bins=40, alpha=0.75, color="#4e79a7")
    ax.axvline(s_real, color="crimson", lw=2, label=f"real={s_real:.3f}")
    ax.set_title(f"Shuffle Null: {dataset}")
    ax.set_xlabel("Best robust sigma")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"plot_shuffle_null_{dataset}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sh_df["alpha_half1"], sh_df["alpha_half2"], s=18, alpha=0.7, color="#59a14f")
    lo = min(sh_df["alpha_half1"].min(), sh_df["alpha_half2"].min())
    hi = max(sh_df["alpha_half1"].max(), sh_df["alpha_half2"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"Split-half alpha*: {dataset}")
    ax.set_xlabel("alpha* half 1")
    ax.set_ylabel("alpha* half 2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"plot_split_half_{dataset}.png", dpi=160)
    plt.close(fig)

    return out


def write_compact_comparison(
    sweep_summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    pca_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    controls_summary: Dict[str, object],
) -> None:
    rows = []
    for ds in sorted(sweep_summary["dataset"].unique()):
        sw = sweep_summary[sweep_summary["dataset"] == ds].iloc[0]
        rr = ratio_summary[ratio_summary["dataset"] == ds].iloc[0]
        ps = pca_summary[pca_summary["dataset"] == ds]
        ps_row = ps.iloc[0] if len(ps) and "pc1_explained_variance" in ps.columns and pd.notna(ps.iloc[0].get("pc1_explained_variance")) else None
        bm = boundary_summary[(boundary_summary["dataset"] == ds) & (boundary_summary["target"] == "log_cs2")]
        bm_row = bm.iloc[0] if len(bm) else None
        nt = controls_summary.get(ds, {})
        rows.append(
            {
                "dataset": ds,
                "n_gal": int(sw["n_gal"]),
                "alpha_star": float(sw["alpha_star"]),
                "alpha_scatter_drop_pct": float(sw["alpha_scatter_drop_pct"]),
                "beta_star": float(sw["beta_star"]),
                "beta_scatter_drop_pct": float(sw["beta_scatter_drop_pct"]),
                "ratio_resid_sigma": float(rr["resid_sigma"]),
                "pca_pc1_explained_variance": float(ps_row["pc1_explained_variance"]) if ps_row is not None else np.nan,
                "boundary_delta_aic_log_cs2": float(bm_row["delta_aic_boundary_minus_null"]) if bm_row is not None else np.nan,
                "shuffle_p_value": float(nt.get("shuffle_null", {}).get("p_value", np.nan)),
                "split_half_absdiff_alpha": float(nt.get("split_half", {}).get("alpha_half_diff_mean_abs", np.nan)),
                "duplication_shift_highmass": float(nt.get("duplication", {}).get("alpha_shift_highmass", np.nan)),
                "env_swap_p_global": float(nt.get("env_swap", {}).get("p_global_swap", np.nan)),
                "controls_subsampled": bool(nt.get("controls_subsampled", False)),
                "n_used_controls": int(nt.get("n_used_controls", 0)),
            }
        )
    comp = pd.DataFrame(rows).sort_values("dataset")
    comp.to_csv(OUT_DIR / "comparison_compact_table.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(comp))

    axes[0, 0].bar(x, comp["alpha_scatter_drop_pct"], color="#4e79a7")
    axes[0, 0].set_xticks(x, comp["dataset"], rotation=15)
    axes[0, 0].set_ylabel("Scatter drop (%)")
    axes[0, 0].set_title("Best alpha rescaling")

    axes[0, 1].bar(x, comp["ratio_resid_sigma"], color="#f28e2b")
    axes[0, 1].set_xticks(x, comp["dataset"], rotation=15)
    axes[0, 1].set_ylabel("Robust sigma")
    axes[0, 1].set_title("Ratio residual scatter")

    axes[1, 0].bar(x, comp["pca_pc1_explained_variance"], color="#59a14f")
    axes[1, 0].set_xticks(x, comp["dataset"], rotation=15)
    axes[1, 0].set_ylabel("Explained variance")
    axes[1, 0].set_title("PCA PC1 explained variance")

    axes[1, 1].bar(x, comp["boundary_delta_aic_log_cs2"], color="#e15759")
    axes[1, 1].set_xticks(x, comp["dataset"], rotation=15)
    axes[1, 1].set_ylabel("ΔAIC (boundary - null)")
    axes[1, 1].set_title("Boundary model gain (log_cs2)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_comparison_compact.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_out()
    phase0_project_map()

    selected = {s.strip() for s in args.datasets.split(",") if s.strip()}
    valid = {"SPARC", "TNG_DEV3000", "TNG_BIG48133"}
    unknown = selected - valid
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}")

    # PHASE 1
    datasets = []
    if "SPARC" in selected:
        datasets.append(build_sparc_pack())
    if "TNG_DEV3000" in selected:
        datasets.append(build_tng_dev_pack())
    if "TNG_BIG48133" in selected:
        tng_big = build_tng_big_pack()
        if tng_big is not None:
            datasets.append(tng_big)
        else:
            print("[WARN] TNG_BIG48133 requested but not found; skipping.")

    df_gal = pd.concat([d.df_gal for d in datasets], ignore_index=True)

    # Residual bins shared across datasets
    bin_edges = np.linspace(-13.0, -8.0, 26)
    df_resid = pd.concat(
        [build_resid_df(d.df_points, d.name, d.id_col, bin_edges) for d in datasets],
        ignore_index=True,
    )

    df_gal.to_parquet(OUT_DIR / "df_gal.parquet", index=False)
    df_resid.to_parquet(OUT_DIR / "df_resid.parquet", index=False)

    # PHASE 2 + 3 + 4
    alpha_rows = []
    beta_rows = []
    sweep_summary_rows = []
    ratio_summary_rows = []
    ratio_resid_rows = []
    pca_summary_rows = []
    pca_shape_rows = []
    pca_loading_rows = []
    boundary_rows = []
    controls_summary = {}

    for ds in datasets:
        print(f"\n=== Dataset: {ds.name} ===")
        g = ds.df_gal.copy()
        g = g[np.isfinite(g["log_cs2"]) & np.isfinite(g["log_Mb"]) & np.isfinite(g["log_xi"])].copy()
        y = g["log_cs2"].to_numpy(dtype=float)
        x = g["log_Mb"].to_numpy(dtype=float)
        env = g["env_code"].to_numpy(dtype=float)

        # A) alpha sweep
        a_df = run_sweep(y, x, env, ALPHAS, "alpha", ds.name)
        a_pick = pick_best_from_sweep(a_df, "alpha")
        a_df.to_csv(OUT_DIR / f"alpha_sweep_{ds.name}.csv", index=False)
        alpha_rows.append(a_df)

        # B) beta sweep
        b_df = run_sweep(y, g["log_xi"].to_numpy(dtype=float), env, BETAS, "beta", ds.name)
        b_pick = pick_best_from_sweep(b_df, "beta")
        b_df.to_csv(OUT_DIR / f"beta_sweep_{ds.name}.csv", index=False)
        beta_rows.append(b_df)

        # C) ratio residual
        rr_df, rr_sum = fit_ratio_model(g)
        rr_df["dataset"] = ds.name
        ratio_resid_rows.append(rr_df)
        rr_sum["dataset"] = ds.name
        ratio_summary_rows.append(rr_sum)

        # mass-matched env effect on ratio residual
        mm_eff = mass_matched_env_effect(rr_df["residual"].to_numpy(), rr_df["log_Mb"].to_numpy(), rr_df["env_bin"].to_numpy(), n_bins=6)
        rr_sum["mass_matched_env_effect"] = mm_eff["effect"]

        # PHASE 3: PCA residual-shape
        r = df_resid[df_resid["dataset"] == ds.name].copy()
        X = r.pivot(index="id", columns="bin_idx", values="delta_mean")
        # keep galaxies with enough observed bins
        keep = X.notna().sum(axis=1) >= 5
        Xk = X.loc[keep].copy()
        if len(Xk) >= 10:
            Xf, scores, comps, evr = em_pca_fill(Xk.to_numpy(dtype=float), n_components=2, n_iter=30)
            pc1 = scores[:, 0]
            load_df = pd.DataFrame({"id": Xk.index.astype(str), "pc1_amp": pc1})
            load_df["dataset"] = ds.name
            load_df = load_df.merge(g[["id", "log_Mb", "env_code", "env_bin", "mean_log_gbar", "log_cs2"]], on="id", how="left")

            rho_pc1_m, p_pc1_m = spearman_safe(load_df["pc1_amp"].to_numpy(), load_df["log_Mb"].to_numpy())
            rho_pc1_e, p_pc1_e = spearman_safe(load_df["pc1_amp"].to_numpy(), load_df["env_code"].to_numpy())
            mm_pc1 = mass_matched_env_effect(
                load_df["pc1_amp"].to_numpy(),
                load_df["log_Mb"].to_numpy(),
                load_df["env_bin"].to_numpy(),
                n_bins=6,
            )

            pca_summary_rows.append(
                {
                    "dataset": ds.name,
                    "n_gal_pca": int(len(Xk)),
                    "n_bins": int(Xk.shape[1]),
                    "pc1_explained_variance": float(evr[0]),
                    "pc2_explained_variance": float(evr[1]) if len(evr) > 1 else np.nan,
                    "rho_pc1_mass": rho_pc1_m,
                    "p_pc1_mass": p_pc1_m,
                    "rho_pc1_env": rho_pc1_e,
                    "p_pc1_env": p_pc1_e,
                    "mass_matched_env_effect_pc1": mm_pc1["effect"],
                }
            )

            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            for bi, c in enumerate(centers):
                pca_shape_rows.append({"dataset": ds.name, "bin_idx": bi, "bin_center": float(c), "pc1_loading": float(comps[0, bi])})

            pca_loading_rows.append(load_df)

            # PHASE 4 boundary model on pc1 amplitude
            bm_pc1 = fit_boundary_model(
                load_df["pc1_amp"].to_numpy(dtype=float),
                load_df["log_Mb"].to_numpy(dtype=float),
                load_df["mean_log_gbar"].to_numpy(dtype=float),
            )
            bm_pc1["dataset"] = ds.name
            bm_pc1["target"] = "pc1_amp"
            boundary_rows.append(bm_pc1)
        else:
            pca_summary_rows.append({"dataset": ds.name, "status": "insufficient_data"})

        # PHASE 4 boundary model on log_cs2
        bm_cs2 = fit_boundary_model(
            g["log_cs2"].to_numpy(dtype=float),
            g["log_Mb"].to_numpy(dtype=float),
            g["mean_log_gbar"].to_numpy(dtype=float),
        )
        bm_cs2["dataset"] = ds.name
        bm_cs2["target"] = "log_cs2"
        boundary_rows.append(bm_cs2)

        sweep_summary_rows.append(
            {
                "dataset": ds.name,
                "alpha_star": a_pick["best_param"],
                "alpha_sigma_raw": a_pick["raw_sigma"],
                "alpha_sigma_best": a_pick["best_sigma"],
                "alpha_scatter_drop_pct": a_pick["scatter_drop_pct"],
                "alpha_rho_mass_best": a_pick["rho_mass_at_best"],
                "alpha_rho_env_best": a_pick["rho_env_at_best"],
                "alpha_used_constraint": a_pick["used_mass_constraint"],
                "beta_star": b_pick["best_param"],
                "beta_sigma_raw": b_pick["raw_sigma"],
                "beta_sigma_best": b_pick["best_sigma"],
                "beta_scatter_drop_pct": b_pick["scatter_drop_pct"],
                "beta_rho_mass_best": b_pick["rho_mass_at_best"],
                "beta_rho_env_best": b_pick["rho_env_at_best"],
                "beta_used_constraint": b_pick["used_mass_constraint"],
                "n_gal": int(len(g)),
            }
        )

        # PHASE 5 controls
        controls_summary[ds.name] = run_controls(ds.name, g, rr_df, max_control_n=args.max_control_n)

    # Save aggregates
    alpha_all = pd.concat(alpha_rows, ignore_index=True)
    beta_all = pd.concat(beta_rows, ignore_index=True)
    sweep_summary = pd.DataFrame(sweep_summary_rows)
    ratio_summary = pd.DataFrame(ratio_summary_rows)
    ratio_resid = pd.concat(ratio_resid_rows, ignore_index=True)
    pca_summary = pd.DataFrame(pca_summary_rows)
    pca_shapes = pd.DataFrame(pca_shape_rows)
    pca_loadings = pd.concat(pca_loading_rows, ignore_index=True) if pca_loading_rows else pd.DataFrame()
    boundary_summary = pd.DataFrame(boundary_rows)

    alpha_all.to_csv(OUT_DIR / "alpha_sweep.csv", index=False)
    beta_all.to_csv(OUT_DIR / "beta_sweep.csv", index=False)
    sweep_summary.to_csv(OUT_DIR / "invariant_sweep_summary.csv", index=False)
    ratio_summary.to_csv(OUT_DIR / "ratio_regression_summary.csv", index=False)
    ratio_resid.to_parquet(OUT_DIR / "ratio_residuals.parquet", index=False)
    pca_summary.to_csv(OUT_DIR / "pca_summary.csv", index=False)
    pca_shapes.to_csv(OUT_DIR / "pca_pc1_shape.csv", index=False)
    if len(pca_loadings):
        pca_loadings.to_parquet(OUT_DIR / "pca_loadings.parquet", index=False)
    boundary_summary.to_csv(OUT_DIR / "boundary_model_summary.csv", index=False)
    (OUT_DIR / "null_tests.json").write_text(json.dumps(controls_summary, indent=2), encoding="utf-8")

    # plots: alpha/beta sweeps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ds_name, dsub in alpha_all.groupby("dataset"):
        axes[0].plot(dsub["alpha"], dsub["robust_sigma"], label=ds_name)
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("robust sigma(log_cs2 + alpha*log_Mb)")
    axes[0].set_title("Mass-exponent sweep")
    axes[0].legend()
    for ds_name, dsub in beta_all.groupby("dataset"):
        axes[1].plot(dsub["beta"], dsub["robust_sigma"], label=ds_name)
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("robust sigma(log_cs2 + beta*log_xi)")
    axes[1].set_title("Healing-length sweep")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_alpha_beta_sweeps.png", dpi=170)
    plt.close(fig)

    if len(pca_shapes):
        fig, ax = plt.subplots(figsize=(7, 4))
        for ds_name, dsub in pca_shapes.groupby("dataset"):
            ax.plot(dsub["bin_center"], dsub["pc1_loading"], marker="o", label=ds_name)
        ax.axvline(LOG_G_DAG, color="k", ls="--", lw=1, label="log g†")
        ax.set_xlabel("log gbar bin center")
        ax.set_ylabel("PC1 loading")
        ax.set_title("Residual-shape PC1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "plot_pca_pc1_shape.png", dpi=170)
        plt.close(fig)

    if len(pca_summary):
        fig, ax = plt.subplots(figsize=(6, 4))
        d = pca_summary[pca_summary["dataset"].notna() & pca_summary["pc1_explained_variance"].notna()]
        color_cycle = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(d))]
        ax.bar(d["dataset"], d["pc1_explained_variance"], color=colors)
        ax.set_ylabel("Explained variance ratio")
        ax.set_title("PC1 explained variance")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "plot_pca_explained_variance.png", dpi=170)
        plt.close(fig)

    write_compact_comparison(sweep_summary, ratio_summary, pca_summary, boundary_summary, controls_summary)

    # report
    report = []
    report.append("# Universality Audit Report")
    report.append("")
    report.append("## Data Used")
    report.append(f"- Repo root: `{REPO_ROOT}`")
    report.append(f"- Output dir: `{OUT_DIR}`")
    ds_run = ", ".join([f"`{d.name}`" for d in datasets])
    report.append(f"- Datasets run: {ds_run}.")
    if any(d.name == "TNG_BIG48133" for d in datasets):
        report.append(
            "- `TNG_BIG48133` env proxy uses host-group occupancy (`log10(group_member_count)`), "
            "with median split into low/high density bins."
        )
    else:
        report.append("- Big-base 48,133x50 data not found locally at runtime.")
    report.append("")
    report.append("## Units + Conventions")
    report.append("- `log_cs2 = log10(c_s^2 [m^2/s^2])`.")
    report.append("- `log_Mb = log10(M_baryon_proxy [Msun])`, where `M_baryon_proxy` is from top-3 outer points using `M_b = g_bar r^2 / G`.")
    report.append("- `xi = sqrt(G M_b / g†)` with `g† = 1.2e-10 m/s^2`, stored as `xi_m` and `log_xi`.")
    report.append("- Residual curves use `delta = log_gobs - log_gRAR_pred` binned in log_gbar.")
    report.append("")
    report.append("## Filters")
    report.append("- Canonical `df_gal` keeps rows with finite `log_cs2`, finite `log_Mb`, and finite `mean_log_gbar`.")
    report.append("- TNG `c_s^2` proxy requires at least 3 valid radial points after `g_dm = g_obs - g_bar > 0` and finite-gradient filtering.")
    report.append("- PCA keeps galaxies with at least 5 non-missing residual bins.")
    report.append(
        f"- Controls are run per dataset with `N=1000` shuffle/environment permutations; "
        f"`--max-control-n={args.max_control_n}` was used in this run (`0` means no subsampling)."
    )
    report.append("")
    report.append("## Main Findings")
    for _, row in sweep_summary.iterrows():
        report.append(
            f"- **{row['dataset']}**: alpha*={row['alpha_star']:.3f}, "
            f"alpha scatter drop={row['alpha_scatter_drop_pct']:.1f}%, "
            f"beta*={row['beta_star']:.3f}, beta scatter drop={row['beta_scatter_drop_pct']:.1f}%."
        )
    for _, row in ratio_summary.iterrows():
        if "status" in row and row["status"] == "too_few":
            continue
        report.append(
            f"- Ratio residual `{row['dataset']}`: slope={row['slope_robust']:.4f}, "
            f"resid sigma={row['resid_sigma']:.3f}, rho(resid,mass)={row['rho_resid_mass']:.3f}, "
            f"rho(resid,env)={row['rho_resid_env']:.3f}."
        )
    if len(pca_summary):
        for _, row in pca_summary.iterrows():
            if "pc1_explained_variance" in row and pd.notna(row["pc1_explained_variance"]):
                report.append(
                    f"- PCA `{row['dataset']}`: PC1 explained variance={row['pc1_explained_variance']:.3f}, "
                    f"rho(PC1,mass)={row['rho_pc1_mass']:.3f}, rho(PC1,env)={row['rho_pc1_env']:.3f}."
                )
    report.append("")
    report.append("## Boundary Transparency Model")
    for _, row in boundary_summary.iterrows():
        if "status" in row and row["status"] == "too_few":
            continue
        report.append(
            f"- {row['dataset']} [{row['target']}]: "
            f"log_gdag_fit={row['log_gdag_fit']:.3f}, s={row['s_fit']:.3f}, "
            f"ΔAIC(boundary-null)={row['delta_aic_boundary_minus_null']:.2f}."
        )
    report.append("")
    report.append("## Falsification Controls")
    for ds_name, c in controls_summary.items():
        sh = c.get("shuffle_null", {})
        split = c.get("split_half", {})
        dup = c.get("duplication", {})
        envs = c.get("env_swap", {})
        report.append(
            f"- **{ds_name}**: shuffle p={sh.get('p_value', np.nan):.4f}, "
            f"split |Δalpha*| mean={split.get('alpha_half_diff_mean_abs', np.nan):.3f}, "
            f"dup shifts(all/highmass)=({dup.get('alpha_shift_all', np.nan):.3f}, {dup.get('alpha_shift_highmass', np.nan):.3f}), "
            f"env-swap p(global/strat)=({envs.get('p_global_swap', np.nan)}, {envs.get('p_stratified_swap', np.nan)})."
        )
    report.append("")
    report.append("## Failure Modes / Caveats")
    report.append("- SPARC `log_Mb` is a proxy from `g_bar` and radius, not direct catalog baryonic mass.")
    report.append("- PCA uses EM-style iterative imputation; missing-bin structure can affect mode amplitudes.")
    report.append("- Boundary model is minimal and intended as falsifiable baseline, not full microphysics.")
    if args.max_control_n > 0:
        report.append(
            f"- Controls used at most `{args.max_control_n}` galaxies per dataset for runtime stability; `N=1000` permutations were still used."
        )
    else:
        report.append("- Controls used full available dataset sizes (no control subsampling).")
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    # keep combined control tables for convenience
    shuffle_all = []
    split_all = []
    dup_all = []
    env_all = []
    syn_all = []
    for ds in [d.name for d in datasets]:
        p = OUT_DIR / f"controls_shuffle_null_{ds}.csv"
        if p.exists():
            shuffle_all.append(pd.read_csv(p))
        p = OUT_DIR / f"controls_split_half_{ds}.csv"
        if p.exists():
            split_all.append(pd.read_csv(p))
        p = OUT_DIR / f"controls_duplication_{ds}.csv"
        if p.exists():
            dup_all.append(pd.read_csv(p))
        p = OUT_DIR / f"controls_env_swap_{ds}.csv"
        if p.exists():
            env_all.append(pd.read_csv(p))
        p = OUT_DIR / f"controls_synthetic_injection_{ds}.csv"
        if p.exists():
            syn_all.append(pd.read_csv(p))

    if shuffle_all:
        pd.concat(shuffle_all, ignore_index=True).to_csv(OUT_DIR / "controls_shuffle_null.csv", index=False)
    if split_all:
        pd.concat(split_all, ignore_index=True).to_csv(OUT_DIR / "controls_split_half.csv", index=False)
    if dup_all:
        pd.concat(dup_all, ignore_index=True).to_csv(OUT_DIR / "controls_duplication.csv", index=False)
    if env_all:
        pd.concat(env_all, ignore_index=True).to_csv(OUT_DIR / "controls_env_swap.csv", index=False)
    if syn_all:
        pd.concat(syn_all, ignore_index=True).to_csv(OUT_DIR / "controls_synthetic_injection.csv", index=False)

    print("\n[DONE] Universality audit complete.")
    print(f"  Outputs: {OUT_DIR}")
    print("  Key files:")
    for fn in [
        "df_gal.parquet",
        "df_resid.parquet",
        "alpha_sweep.csv",
        "beta_sweep.csv",
        "ratio_regression_summary.csv",
        "pca_summary.csv",
        "boundary_model_summary.csv",
        "null_tests.json",
        "report.md",
    ]:
        p = OUT_DIR / fn
        print(f"    - {p} ({'OK' if p.exists() else 'MISSING'})")


if __name__ == "__main__":
    main()
