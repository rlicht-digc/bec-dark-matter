#!/usr/bin/env python3
"""
TNG vs SPARC Composition Sweep (Discrimination Search)
======================================================

Systematically searches over composition choices (cuts + aggregation style)
to find which setups best discriminate TNG from SPARC.

Outputs:
  - analysis/results/tng_sparc_composition_sweep/composition_ranking.csv
  - analysis/results/tng_sparc_composition_sweep/summary_tng_sparc_composition_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


G_SI = 6.674e-11
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
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sig = 1.4826 * mad
    if sig <= 0:
        sig = np.std(x, ddof=1) if x.size > 1 else np.nan
    return float(sig)


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float((2.0 * u_stat) / (n1 * n2) - 1.0)


def bootstrap_median_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    na = len(a)
    nb = len(b)
    for i in range(n_boot):
        sa = a[rng.integers(0, na, na)]
        sb = b[rng.integers(0, nb, nb)]
        diffs[i] = np.median(sa) - np.median(sb)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _canonical_points_df(
    raw: pd.DataFrame,
    dataset_name: str,
) -> Optional[pd.DataFrame]:
    cols = {c.lower(): c for c in raw.columns}

    if "log_gbar" in cols:
        log_gbar = pd.to_numeric(raw[cols["log_gbar"]], errors="coerce")
    elif "g_bar" in cols:
        gbar = pd.to_numeric(raw[cols["g_bar"]], errors="coerce")
        log_gbar = np.log10(np.maximum(gbar, 1e-30))
    else:
        return None

    if "log_gobs" in cols:
        log_gobs = pd.to_numeric(raw[cols["log_gobs"]], errors="coerce")
    elif "g_obs" in cols:
        gobs = pd.to_numeric(raw[cols["g_obs"]], errors="coerce")
        log_gobs = np.log10(np.maximum(gobs, 1e-30))
    else:
        return None

    if "r_kpc" in cols:
        r_kpc = pd.to_numeric(raw[cols["r_kpc"]], errors="coerce")
    elif "radius" in cols:
        r_kpc = pd.to_numeric(raw[cols["radius"]], errors="coerce")
    else:
        r_kpc = pd.Series(np.nan, index=raw.index, dtype=float)

    if "subhaloid" in cols:
        galaxy = raw[cols["subhaloid"]].astype(str)
    elif "galaxy" in cols:
        galaxy = raw[cols["galaxy"]].astype(str)
    elif "galaxyid" in cols:
        galaxy = raw[cols["galaxyid"]].astype(str)
    else:
        galaxy = pd.Series([f"{dataset_name}_{i}" for i in range(len(raw))], index=raw.index)

    if "lowres_flag" in cols:
        lowres_flag = pd.to_numeric(raw[cols["lowres_flag"]], errors="coerce").fillna(0).astype(int)
    else:
        lowres_flag = pd.Series(0, index=raw.index, dtype=int)

    out = pd.DataFrame(
        {
            "dataset": dataset_name,
            "galaxy": galaxy,
            "r_kpc": r_kpc,
            "log_gbar": log_gbar,
            "log_gobs": log_gobs,
            "lowres_flag": lowres_flag,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_gbar", "log_gobs", "r_kpc"])
    return out


def load_tng_points(tng_input: Optional[str]) -> pd.DataFrame:
    candidates = []
    if tng_input:
        candidates.append(tng_input)
    candidates.extend(
        [
            os.path.expanduser("~/Desktop/tng_cross_validation/LATEST_GOOD/rar_points.parquet"),
            os.path.expanduser("~/Desktop/tng_cross_validation/tng_gas_comparison/phase1b_rar_data.npz"),
        ]
    )

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in {".parquet", ".pq"}:
            raw = pd.read_parquet(path)
            df = _canonical_points_df(raw, "TNG")
            if df is not None and len(df) > 0:
                print(f"Loaded TNG points from parquet: {path}")
                return df
        elif ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            raw = pd.read_csv(path, sep=sep)
            df = _canonical_points_df(raw, "TNG")
            if df is not None and len(df) > 0:
                print(f"Loaded TNG points from table: {path}")
                return df
        elif ext == ".npz":
            arr = np.load(path, allow_pickle=True)
            # Prefer star+gas points for physical completeness
            if {"g_bar_full", "g_obs_full", "gid_full", "radius"} <= set(arr.files):
                gbar = np.asarray(arr["g_bar_full"], dtype=float)
                gobs = np.asarray(arr["g_obs_full"], dtype=float)
                gid = np.asarray(arr["gid_full"], dtype=int)
                rad = np.asarray(arr["radius"], dtype=float)
            elif {"g_bar_star", "g_obs_star", "gid_star", "radius"} <= set(arr.files):
                gbar = np.asarray(arr["g_bar_star"], dtype=float)
                gobs = np.asarray(arr["g_obs_star"], dtype=float)
                gid = np.asarray(arr["gid_star"], dtype=int)
                rad = np.asarray(arr["radius"], dtype=float)
            else:
                continue

            df = pd.DataFrame(
                {
                    "dataset": "TNG",
                    "galaxy": pd.Series(gid).astype(str),
                    "r_kpc": rad,
                    "log_gbar": np.log10(np.maximum(gbar, 1e-30)),
                    "log_gobs": np.log10(np.maximum(gobs, 1e-30)),
                    "lowres_flag": 0,
                }
            )
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_gbar", "log_gobs", "r_kpc"])
            if len(df) > 0:
                print(f"Loaded TNG points from NPZ: {path}")
                return df

    raise FileNotFoundError(
        "No TNG point file found. Pass --tng-input or place data at "
        "~/Desktop/tng_cross_validation/LATEST_GOOD/rar_points.parquet "
        "or ~/Desktop/tng_cross_validation/tng_gas_comparison/phase1b_rar_data.npz"
    )


def load_sparc_points(project_root: str) -> pd.DataFrame:
    sparc_dir = os.path.join(project_root, "data", "sparc")
    table2_path = os.path.join(sparc_dir, "SPARC_table2_rotmods.dat")
    mrt_path = os.path.join(sparc_dir, "SPARC_Lelli2016c.mrt")
    if not (os.path.exists(table2_path) and os.path.exists(mrt_path)):
        raise FileNotFoundError("SPARC table files not found under data/sparc/")

    rc_data: Dict[str, Dict[str, List[float]]] = {}
    with open(table2_path, "r") as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                if not name:
                    continue
                rad = float(line[19:25].strip())
                vobs = float(line[26:32].strip())
                vgas = float(line[39:45].strip())
                vdisk = float(line[46:52].strip())
                vbul = float(line[53:59].strip())
            except (ValueError, IndexError):
                continue
            if name not in rc_data:
                rc_data[name] = {"R": [], "Vobs": [], "Vgas": [], "Vdisk": [], "Vbul": []}
            rc_data[name]["R"].append(rad)
            rc_data[name]["Vobs"].append(vobs)
            rc_data[name]["Vgas"].append(vgas)
            rc_data[name]["Vdisk"].append(vdisk)
            rc_data[name]["Vbul"].append(vbul)

    for name in rc_data:
        for k in rc_data[name]:
            rc_data[name][k] = np.asarray(rc_data[name][k], dtype=float)

    # Quality metadata: Q and Inclination
    props: Dict[str, Dict[str, float]] = {}
    with open(mrt_path, "r") as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---") and i > 50:
            data_start = i + 1
            break
    for line in lines[data_start:]:
        if not line.strip() or line.startswith("#"):
            continue
        try:
            name = line[0:11].strip()
            parts = line[11:].split()
            if len(parts) < 17:
                continue
            props[name] = {"Inc": float(parts[4]), "Q": int(parts[16])}
        except (ValueError, IndexError):
            continue

    rows = []
    for name, gdata in rc_data.items():
        p = props.get(name)
        if p is None:
            continue
        if p["Q"] > 2 or p["Inc"] < 30.0 or p["Inc"] > 85.0:
            continue

        r = gdata["R"]
        vobs = gdata["Vobs"]
        vgas = gdata["Vgas"]
        vdisk = gdata["Vdisk"]
        vbul = gdata["Vbul"]

        # Standard SPARC conversion used across this codebase
        vbar_sq = 0.5 * vdisk**2 + vgas * np.abs(vgas) + 0.7 * vbul * np.abs(vbul)
        valid = (r > 0) & (vobs > 0) & (vbar_sq > 0)
        if np.sum(valid) < 3:
            continue

        r_use = r[valid]
        gb = (vbar_sq[valid] * 1.0e6) / (r_use * KPC_M)
        go = ((vobs[valid] * 1.0e3) ** 2) / (r_use * KPC_M)
        mask2 = (gb > 1e-15) & (go > 1e-15)

        for rk, gbar, gobs in zip(r_use[mask2], gb[mask2], go[mask2]):
            rows.append(
                {
                    "dataset": "SPARC",
                    "galaxy": name,
                    "r_kpc": float(rk),
                    "log_gbar": float(np.log10(gbar)),
                    "log_gobs": float(np.log10(gobs)),
                    "lowres_flag": 0,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("SPARC point table built empty after quality cuts.")
    print(f"Loaded SPARC points from raw tables: {len(df)} points, {df['galaxy'].nunique()} galaxies")
    return df


def apply_composition(
    df: pd.DataFrame,
    dm_threshold: float,
    rmin_kpc: float,
    min_pts: int,
    tng_require_lowres0: bool,
) -> pd.DataFrame:
    mask = (df["log_gbar"] < dm_threshold) & (df["r_kpc"] >= rmin_kpc)
    if tng_require_lowres0 and df["dataset"].iloc[0] == "TNG":
        mask &= (df["lowres_flag"] == 0)
    out = df[mask].copy()
    if len(out) == 0:
        return out
    counts = out.groupby("galaxy").size()
    keep = counts[counts >= min_pts].index
    return out[out["galaxy"].isin(keep)].copy()


def metric_array(df: pd.DataFrame, style: str) -> np.ndarray:
    if style == "pooled_points":
        arr = df["log_res"].to_numpy(dtype=float)
    elif style == "equal_weight_galaxy_median":
        arr = df.groupby("galaxy")["log_res"].median().to_numpy(dtype=float)
    elif style == "per_gal_scatter":
        arr = (
            df.groupby("galaxy")["log_res"]
            .apply(lambda s: robust_sigma(s.to_numpy(dtype=float)))
            .to_numpy(dtype=float)
        )
    else:
        raise ValueError(f"Unknown style: {style}")

    arr = arr[np.isfinite(arr)]
    return arr


def compare_arrays(
    arr_tng: np.ndarray,
    arr_sparc: np.ndarray,
    bootstrap_n: int,
) -> Dict[str, object]:
    n_tng = len(arr_tng)
    n_sparc = len(arr_sparc)
    if n_tng < 8 or n_sparc < 8:
        return {
            "n_tng": int(n_tng),
            "n_sparc": int(n_sparc),
            "mw_p": None,
            "u_stat": None,
            "cliffs_delta": None,
            "median_tng": None,
            "median_sparc": None,
            "median_diff": None,
            "median_ratio": None,
            "median_diff_ci95": None,
            "score": None,
        }

    u_stat, mw_p = mannwhitneyu(arr_tng, arr_sparc, alternative="two-sided")
    cd = cliffs_delta_from_u(float(u_stat), n_tng, n_sparc)
    med_tng = float(np.median(arr_tng))
    med_sparc = float(np.median(arr_sparc))
    diff = med_tng - med_sparc
    ratio = med_tng / med_sparc if med_sparc != 0 else np.nan

    ci95 = None
    if bootstrap_n > 0:
        lo, hi = bootstrap_median_diff_ci(arr_tng, arr_sparc, n_boot=bootstrap_n, seed=42)
        ci95 = [lo, hi]

    score = abs(cd) * max(0.0, -np.log10(max(float(mw_p), 1e-300)))
    return {
        "n_tng": int(n_tng),
        "n_sparc": int(n_sparc),
        "mw_p": float(mw_p),
        "u_stat": float(u_stat),
        "cliffs_delta": float(cd),
        "median_tng": med_tng,
        "median_sparc": med_sparc,
        "median_diff": float(diff),
        "median_ratio": float(ratio) if np.isfinite(ratio) else None,
        "median_diff_ci95": ci95,
        "score": float(score),
    }


def run_sweep(
    tng_df: pd.DataFrame,
    sparc_df: pd.DataFrame,
    bootstrap_n: int,
) -> pd.DataFrame:
    dm_thresholds = [-10.3, -10.4, -10.5, -10.6, -10.7, -10.8, -10.9]
    rmins = [0.0, 0.5, 1.0, 2.0]
    min_pts_list = [2, 3, 5, 8, 10, 15, 20, 25]
    styles = ["pooled_points", "equal_weight_galaxy_median", "per_gal_scatter"]

    has_lowres = (
        "lowres_flag" in tng_df.columns
        and int(np.sum(pd.to_numeric(tng_df["lowres_flag"], errors="coerce").fillna(0).to_numpy() != 0)) > 0
    )
    lowres_modes = [False, True] if has_lowres else [False]

    rows: List[Dict[str, object]] = []
    total = len(dm_thresholds) * len(rmins) * len(min_pts_list) * len(lowres_modes) * len(styles)
    done = 0

    for dm_cut in dm_thresholds:
        for rmin in rmins:
            for min_pts in min_pts_list:
                for lowres_mode in lowres_modes:
                    t_sub = apply_composition(tng_df, dm_cut, rmin, min_pts, tng_require_lowres0=lowres_mode)
                    s_sub = apply_composition(sparc_df, dm_cut, rmin, min_pts, tng_require_lowres0=False)
                    for style in styles:
                        done += 1
                        arr_t = metric_array(t_sub, style)
                        arr_s = metric_array(s_sub, style)
                        cmp = compare_arrays(arr_t, arr_s, bootstrap_n=bootstrap_n if style != "pooled_points" else 0)

                        row = {
                            "dm_threshold": dm_cut,
                            "rmin_kpc": rmin,
                            "min_pts_per_gal": min_pts,
                            "tng_lowres0_only": lowres_mode,
                            "style": style,
                            "tng_n_points": int(len(t_sub)),
                            "sparc_n_points": int(len(s_sub)),
                            "tng_n_gal": int(t_sub["galaxy"].nunique()) if len(t_sub) else 0,
                            "sparc_n_gal": int(s_sub["galaxy"].nunique()) if len(s_sub) else 0,
                        }
                        row.update(cmp)
                        rows.append(row)

                    if done % 60 == 0 or done == total:
                        print(f"  sweep progress: {done}/{total}")

    rank = pd.DataFrame(rows)
    rank = rank.sort_values(["score", "style"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return rank


def main():
    parser = argparse.ArgumentParser(description="Systematic TNG-vs-SPARC composition discrimination sweep")
    parser.add_argument("--tng-input", type=str, default=None, help="Path to TNG points (.parquet/.csv/.npz)")
    parser.add_argument("--bootstrap", type=int, default=400, help="Bootstrap draws for median-diff CI (non-pooled styles)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    outdir = args.outdir or os.path.join(project_root, "analysis", "results", "tng_sparc_composition_sweep")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 80)
    print("TNG vs SPARC COMPOSITION DISCRIMINATION SWEEP")
    print("=" * 80)

    tng_df = load_tng_points(args.tng_input)
    sparc_df = load_sparc_points(project_root)

    for df in (tng_df, sparc_df):
        df["log_res"] = df["log_gobs"].to_numpy(dtype=float) - rar_pred_log(df["log_gbar"].to_numpy(dtype=float))
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["log_gbar", "log_gobs", "log_res", "r_kpc"], inplace=True)
        # keep RAR dynamic range
        df.query("log_gbar >= -13 and log_gbar <= -8 and log_gobs >= -13 and log_gobs <= -8", inplace=True)

    print(f"TNG base:   {len(tng_df)} points, {tng_df['galaxy'].nunique()} galaxies")
    print(f"SPARC base: {len(sparc_df)} points, {sparc_df['galaxy'].nunique()} galaxies")

    rank = run_sweep(tng_df, sparc_df, bootstrap_n=args.bootstrap)

    csv_path = os.path.join(outdir, "composition_ranking.csv")
    rank.to_csv(csv_path, index=False)

    valid = rank[rank["score"].notna()].copy()
    top_overall = valid.head(20)
    top_by_style = {}
    for style in ["pooled_points", "equal_weight_galaxy_median", "per_gal_scatter"]:
        sub = valid[valid["style"] == style].head(10)
        top_by_style[style] = sub.to_dict(orient="records")

    summary = {
        "description": "TNG vs SPARC composition discrimination sweep",
        "inputs": {
            "tng_input": args.tng_input,
            "bootstrap": args.bootstrap,
            "tng_base_points": int(len(tng_df)),
            "tng_base_galaxies": int(tng_df["galaxy"].nunique()),
            "sparc_base_points": int(len(sparc_df)),
            "sparc_base_galaxies": int(sparc_df["galaxy"].nunique()),
        },
        "ranking_csv": csv_path,
        "n_compositions_evaluated": int(len(rank)),
        "n_valid": int(len(valid)),
        "top_overall": top_overall.to_dict(orient="records"),
        "top_by_style": top_by_style,
    }

    json_path = os.path.join(outdir, "summary_tng_sparc_composition_sweep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nTop 10 compositions (by score):")
    cols = [
        "style",
        "dm_threshold",
        "rmin_kpc",
        "min_pts_per_gal",
        "tng_lowres0_only",
        "tng_n_gal",
        "sparc_n_gal",
        "median_tng",
        "median_sparc",
        "cliffs_delta",
        "mw_p",
        "score",
    ]
    print(top_overall[cols].to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
