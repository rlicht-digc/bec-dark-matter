#!/usr/bin/env python3
"""
Mass-Matched Fairness Sweep: TNG vs SPARC
=========================================

Re-runs the "fairness gap vs threshold" analysis using per-galaxy DM-dominated
robust residual scatter, then compares:
  1) unmatched distributions
  2) 1:1 stellar-mass-matched distributions

Outputs:
  - analysis/results/tng_sparc_composition_sweep/fairness_gap_vs_threshold.csv
  - analysis/results/tng_sparc_composition_sweep/summary_fairness_gap_vs_threshold.json
  - analysis/results/tng_sparc_composition_sweep/fairness_best_ratio_vs_threshold.png
  - analysis/results/tng_sparc_composition_sweep/fairness_best_effect_vs_threshold.png
  - analysis/results/tng_sparc_composition_sweep/fairness_score_heatmaps_by_threshold.png
  - analysis/results/tng_sparc_composition_sweep/fairness_one_page_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


G_DAGGER = 1.2e-10
KPC_M = 3.086e19

THRESHOLDS = [-10.4, -10.5, -10.6, -10.7, -10.8]
MIN_PTS = [10, 15, 20]
RMINS = [0.0, 0.5, 1.0, 2.0]


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


def mannwhitney_cliffs(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    res = mannwhitneyu(a, b, alternative="two-sided")
    p = float(res.pvalue)
    u = float(res.statistic)
    cliffs = (2.0 * u) / (a.size * b.size) - 1.0
    return p, float(cliffs)


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-30:
        return np.nan
    return float(num / den)


def load_tng_points(tng_parquet: str, tng_profiles_npz: str) -> pd.DataFrame:
    pts = pd.read_parquet(tng_parquet).copy()
    npz = np.load(tng_profiles_npz, allow_pickle=True)
    mass_map = dict(
        zip(np.asarray(npz["galaxy_ids"], dtype=int), np.asarray(npz["m_star_total"], dtype=float))
    )

    if "SubhaloID" in pts.columns:
        galaxy = pts["SubhaloID"].astype(int)
    elif "galaxy" in pts.columns:
        galaxy = pts["galaxy"].astype(int)
    else:
        raise ValueError("TNG points require SubhaloID or galaxy column")

    out = pd.DataFrame(
        {
            "galaxy": galaxy,
            "r_kpc": pd.to_numeric(pts["r_kpc"], errors="coerce"),
            "log_gbar": pd.to_numeric(pts["log_gbar"], errors="coerce"),
            "log_gobs": pd.to_numeric(pts["log_gobs"], errors="coerce"),
        }
    )
    out["logMstar"] = out["galaxy"].map(lambda g: np.log10(max(mass_map.get(int(g), np.nan), 1e-30)))
    out["resid"] = out["log_gobs"] - rar_pred_log(out["log_gbar"].values)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["r_kpc", "log_gbar", "log_gobs", "logMstar", "resid"]
    )
    out["galaxy"] = out["galaxy"].astype(str)
    return out


def load_sparc_points(project_root: str) -> pd.DataFrame:
    table2_path = os.path.join(project_root, "data", "sparc", "SPARC_table2_rotmods.dat")
    mrt_path = os.path.join(project_root, "data", "sparc", "SPARC_Lelli2016c.mrt")

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
        for key in rc_data[name]:
            rc_data[name][key] = np.asarray(rc_data[name][key], dtype=float)

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
        name = line[0:11].strip()
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        try:
            props[name] = {
                "Inc": float(parts[4]),
                "Q": int(parts[16]),
                "L36": float(parts[6]),  # 1e9 L_sun at 3.6um
            }
        except (ValueError, IndexError):
            continue

    rows: List[Dict[str, float]] = []
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

        vbar_sq = 0.5 * vdisk**2 + vgas * np.abs(vgas) + 0.7 * vbul * np.abs(vbul)
        valid = (r > 0) & (vobs > 0) & (vbar_sq > 0)
        if np.sum(valid) < 3:
            continue

        r_use = r[valid]
        gb = (vbar_sq[valid] * 1.0e6) / (r_use * KPC_M)
        go = ((vobs[valid] * 1.0e3) ** 2) / (r_use * KPC_M)
        mask2 = (gb > 1e-15) & (go > 1e-15)
        if np.sum(mask2) < 3:
            continue

        logMstar = np.log10(max(0.5 * p["L36"] * 1.0e9, 1e-30))
        lgb = np.log10(gb[mask2])
        lgo = np.log10(go[mask2])
        resid = lgo - rar_pred_log(lgb)

        for rk, log_gbar, log_gobs, rres in zip(r_use[mask2], lgb, lgo, resid):
            rows.append(
                {
                    "galaxy": name,
                    "r_kpc": float(rk),
                    "log_gbar": float(log_gbar),
                    "log_gobs": float(log_gobs),
                    "resid": float(rres),
                    "logMstar": float(logMstar),
                }
            )

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["r_kpc", "log_gbar", "log_gobs", "resid", "logMstar"]
    )
    return out


def per_gal_scatter(df: pd.DataFrame, dm_threshold: float, min_pts: int, rmin_kpc: float) -> pd.DataFrame:
    sub = df[(df["log_gbar"] < dm_threshold) & (df["r_kpc"] >= rmin_kpc)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["galaxy", "n", "sigma", "logMstar"])
    agg = (
        sub.groupby("galaxy")
        .agg(n=("resid", "size"), sigma=("resid", robust_sigma), logMstar=("logMstar", "first"))
        .reset_index()
    )
    agg = agg[(agg["n"] >= min_pts) & np.isfinite(agg["sigma"]) & np.isfinite(agg["logMstar"])].copy()
    return agg


def mass_match_nearest(
    tng: pd.DataFrame,
    sparc: pd.DataFrame,
    caliper_dex: float,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    if tng.empty or sparc.empty:
        return np.array([]), np.array([]), 0, np.nan

    t = tng.sort_values("logMstar").reset_index(drop=True)
    s = sparc.sort_values("logMstar").reset_index(drop=True)
    used = np.zeros(len(t), dtype=bool)
    pairs_t = []
    pairs_s = []
    dlogm = []

    t_mass = t["logMstar"].values
    t_sig = t["sigma"].values

    for _, row in s.iterrows():
        diffs = np.abs(t_mass - row["logMstar"])
        diffs[used] = np.inf
        j = int(np.argmin(diffs))
        if not np.isfinite(diffs[j]) or diffs[j] > caliper_dex:
            continue
        used[j] = True
        pairs_t.append(float(t_sig[j]))
        pairs_s.append(float(row["sigma"]))
        dlogm.append(float(diffs[j]))

    if not pairs_t:
        return np.array([]), np.array([]), 0, np.nan
    return (
        np.asarray(pairs_t, dtype=float),
        np.asarray(pairs_s, dtype=float),
        len(pairs_t),
        float(np.mean(dlogm)),
    )


def build_rows(
    tng_points: pd.DataFrame,
    sparc_points: pd.DataFrame,
    thresholds: List[float],
    min_pts_grid: List[int],
    rmins: List[float],
    caliper_dex: float,
) -> pd.DataFrame:
    rows = []
    total = len(thresholds) * len(min_pts_grid) * len(rmins)
    done = 0

    for dm in thresholds:
        for mp in min_pts_grid:
            for rmin in rmins:
                tng_sc = per_gal_scatter(tng_points, dm, mp, rmin)
                sp_sc = per_gal_scatter(sparc_points, dm, mp, rmin)

                un_a = tng_sc["sigma"].to_numpy(dtype=float)
                un_b = sp_sc["sigma"].to_numpy(dtype=float)
                un_p, un_cd = mannwhitney_cliffs(un_a, un_b)
                un_med_a = float(np.median(un_a)) if un_a.size else np.nan
                un_med_b = float(np.median(un_b)) if un_b.size else np.nan

                mm_a, mm_b, mm_pairs, mm_dlogm = mass_match_nearest(tng_sc, sp_sc, caliper_dex)
                mm_p, mm_cd = mannwhitney_cliffs(mm_a, mm_b)
                mm_med_a = float(np.median(mm_a)) if mm_a.size else np.nan
                mm_med_b = float(np.median(mm_b)) if mm_b.size else np.nan
                if np.isfinite(mm_p) and mm_p > 0 and np.isfinite(mm_cd):
                    mm_score = float(abs(mm_cd) * (-np.log10(mm_p)))
                else:
                    mm_score = np.nan

                rows.append(
                    {
                        "dm_threshold": dm,
                        "min_pts": mp,
                        "rmin_kpc": rmin,
                        "tng_gal": int(tng_sc["galaxy"].nunique()),
                        "sparc_gal": int(sp_sc["galaxy"].nunique()),
                        "massmatch_pairs": int(mm_pairs),
                        "massmatch_mean_abs_dlogM": mm_dlogm,
                        "un_n_tng": int(un_a.size),
                        "un_n_sparc": int(un_b.size),
                        "un_mw_p": un_p,
                        "un_cliffs_delta": un_cd,
                        "un_median_tng": un_med_a,
                        "un_median_sparc": un_med_b,
                        "un_median_diff": un_med_a - un_med_b if np.isfinite(un_med_a) and np.isfinite(un_med_b) else np.nan,
                        "un_median_ratio": safe_ratio(un_med_a, un_med_b),
                        "mm_n_tng": int(mm_a.size),
                        "mm_n_sparc": int(mm_b.size),
                        "mm_mw_p": mm_p,
                        "mm_cliffs_delta": mm_cd,
                        "mm_median_tng": mm_med_a,
                        "mm_median_sparc": mm_med_b,
                        "mm_median_diff": mm_med_a - mm_med_b if np.isfinite(mm_med_a) and np.isfinite(mm_med_b) else np.nan,
                        "mm_median_ratio": safe_ratio(mm_med_a, mm_med_b),
                        "mm_score": mm_score,
                    }
                )

                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  fairness progress: {done}/{total}")

    return pd.DataFrame(rows)


def write_summary_json(
    out_json: str,
    out_csv: str,
    rows: pd.DataFrame,
    tng_points: pd.DataFrame,
    sparc_points: pd.DataFrame,
    caliper_dex: float,
) -> None:
    valid = rows[np.isfinite(rows["mm_score"])].sort_values("mm_score", ascending=False).copy()
    best_per_dm = (
        valid.sort_values("mm_score", ascending=False)
        .groupby("dm_threshold", as_index=False)
        .first()
        .sort_values("dm_threshold")
    )

    summary = {
        "tng_points": int(len(tng_points)),
        "tng_galaxies": int(tng_points["galaxy"].nunique()),
        "sparc_points": int(len(sparc_points)),
        "sparc_galaxies": int(sparc_points["galaxy"].nunique()),
        "thresholds": THRESHOLDS,
        "min_pts": MIN_PTS,
        "rmins": RMINS,
        "massmatch_caliper_dex": float(caliper_dex),
        "best_massmatched": valid.head(15).to_dict(orient="records"),
        "best_per_threshold": best_per_dm.to_dict(orient="records"),
        "output_csv": out_csv,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)


def plot_best_lines(rows: pd.DataFrame, outdir: str) -> None:
    valid = rows[np.isfinite(rows["mm_score"])].copy()
    best = (
        valid.sort_values("mm_score", ascending=False)
        .groupby("dm_threshold", as_index=False)
        .first()
        .sort_values("dm_threshold")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(best["dm_threshold"], best["mm_median_ratio"], marker="o", linewidth=2)
    for _, r in best.iterrows():
        plt.text(r["dm_threshold"], r["mm_median_ratio"], f" N={int(r['massmatch_pairs'])}", fontsize=8)
    plt.xlabel("DM threshold (log gbar < threshold)")
    plt.ylabel("Best matched median ratio (TNG / SPARC)")
    plt.title("Best Fairness Ratio vs DM Threshold")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fairness_best_ratio_vs_threshold.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(best["dm_threshold"], best["mm_cliffs_delta"], marker="o", linewidth=2)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    for _, r in best.iterrows():
        p = r["mm_mw_p"]
        ptxt = f"{p:.2g}" if np.isfinite(p) else "nan"
        plt.text(r["dm_threshold"], r["mm_cliffs_delta"], f" p={ptxt}", fontsize=8)
    plt.xlabel("DM threshold (log gbar < threshold)")
    plt.ylabel("Best matched Cliff's delta")
    plt.title("Best Fairness Effect Size vs DM Threshold")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fairness_best_effect_vs_threshold.png"), dpi=180)
    plt.close()


def plot_heatmaps(rows: pd.DataFrame, outdir: str) -> None:
    dms = sorted(rows["dm_threshold"].unique())
    mins = sorted(rows["min_pts"].unique())
    rmins = sorted(rows["rmin_kpc"].unique())
    fig, axes = plt.subplots(1, len(dms), figsize=(4.5 * len(dms), 4), sharey=True)
    if len(dms) == 1:
        axes = [axes]
    for ax, dm in zip(axes, dms):
        sub = rows[rows["dm_threshold"] == dm]
        piv = sub.pivot(index="min_pts", columns="rmin_kpc", values="mm_score")
        piv = piv.reindex(index=mins, columns=rmins)
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(rmins)))
        ax.set_xticklabels([f"{x:g}" for x in rmins], rotation=45)
        ax.set_yticks(range(len(mins)))
        ax.set_yticklabels([str(x) for x in mins])
        ax.set_title(f"dm={dm:g}")
        ax.set_xlabel("rmin_kpc")
        if ax is axes[0]:
            ax.set_ylabel("min_pts")
        for i, mp in enumerate(mins):
            for j, rm in enumerate(rmins):
                v = piv.loc[mp, rm]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mass-Matched Fairness Score Heatmaps (mm_score)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fairness_score_heatmaps_by_threshold.png"), dpi=180)
    plt.close(fig)


def write_markdown(rows: pd.DataFrame, out_md: str, source_csv: str, outdir: str) -> None:
    valid = rows[np.isfinite(rows["mm_score"])].sort_values("mm_score", ascending=False).copy()
    best_per_dm = (
        valid.sort_values("mm_score", ascending=False)
        .groupby("dm_threshold", as_index=False)
        .first()
        .sort_values("dm_threshold")
    )
    top12 = valid.head(12)

    lines = []
    lines.append("# TNG vs SPARC Fairness Sweep Summary")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Source CSV: `{source_csv}`")
    lines.append(f"- Rows evaluated: {len(rows)}; valid mass-matched rows: {len(valid)}")
    lines.append("- Score: `mm_score = |Cliff's delta| * (-log10(MW p))`")
    lines.append("")
    lines.append("## Best Per DM Threshold")
    lines.append("")
    lines.append("| dm_threshold | min_pts | rmin_kpc | matched_pairs | median_tng | median_sparc | ratio_tng_over_sparc | cliffs_delta | mw_p | mm_score |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in best_per_dm.iterrows():
        lines.append(
            f"| {r['dm_threshold']:.1f} | {int(r['min_pts'])} | {r['rmin_kpc']:.1f} | {int(r['massmatch_pairs'])} | "
            f"{r['mm_median_tng']:.4f} | {r['mm_median_sparc']:.4f} | {r['mm_median_ratio']:.3f} | "
            f"{r['mm_cliffs_delta']:.3f} | {r['mm_mw_p']:.3g} | {r['mm_score']:.3f} |"
        )
    lines.append("")
    lines.append("## Top 12 Configurations (Overall)")
    lines.append("")
    lines.append("| dm_threshold | min_pts | rmin_kpc | matched_pairs | dlogM_mean | ratio | cliffs_delta | mw_p | score |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in top12.iterrows():
        lines.append(
            f"| {r['dm_threshold']:.1f} | {int(r['min_pts'])} | {r['rmin_kpc']:.1f} | {int(r['massmatch_pairs'])} | "
            f"{r['massmatch_mean_abs_dlogM']:.3f} | {r['mm_median_ratio']:.3f} | {r['mm_cliffs_delta']:.3f} | "
            f"{r['mm_mw_p']:.3g} | {r['mm_score']:.3f} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append(f"- `{os.path.join(outdir, 'fairness_best_ratio_vs_threshold.png')}`")
    lines.append(f"- `{os.path.join(outdir, 'fairness_best_effect_vs_threshold.png')}`")
    lines.append(f"- `{os.path.join(outdir, 'fairness_score_heatmaps_by_threshold.png')}`")
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mass-matched fairness sweep for TNG vs SPARC.")
    _repo_root = str(Path(__file__).resolve().parent.parent.parent)
    parser.add_argument(
        "--tng-input",
        default=os.path.join(_repo_root, "rar_points.parquet"),
        help="Path to TNG rar_points.parquet",
    )
    parser.add_argument(
        "--tng-profiles",
        default=os.path.join(_repo_root, "tng_mass_profiles.npz"),
        help="Path to TNG mass profiles NPZ (for stellar-mass matching)",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(_repo_root, "analysis", "results", "tng_sparc_composition_sweep"),
        help="Output directory",
    )
    parser.add_argument(
        "--massmatch-caliper",
        type=float,
        default=0.20,
        help="Max |delta logM*| for 1:1 nearest-neighbor matching (dex)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(project_root)

    print("=" * 80)
    print("TNG vs SPARC FAIRNESS GAP SWEEP (MASS-MATCHED)")
    print("=" * 80)
    print(f"TNG input: {args.tng_input}")
    print(f"TNG profiles: {args.tng_profiles}")
    print(f"Mass-match caliper: {args.massmatch_caliper:.3f} dex")

    tng_points = load_tng_points(args.tng_input, args.tng_profiles)
    sparc_points = load_sparc_points(project_root)
    print(f"Loaded TNG:   {len(tng_points)} points, {tng_points['galaxy'].nunique()} galaxies")
    print(f"Loaded SPARC: {len(sparc_points)} points, {sparc_points['galaxy'].nunique()} galaxies")

    rows = build_rows(
        tng_points=tng_points,
        sparc_points=sparc_points,
        thresholds=THRESHOLDS,
        min_pts_grid=MIN_PTS,
        rmins=RMINS,
        caliper_dex=args.massmatch_caliper,
    )

    out_csv = os.path.join(args.outdir, "fairness_gap_vs_threshold.csv")
    out_json = os.path.join(args.outdir, "summary_fairness_gap_vs_threshold.json")
    out_md = os.path.join(args.outdir, "fairness_one_page_summary.md")
    rows.to_csv(out_csv, index=False)
    write_summary_json(out_json, out_csv, rows, tng_points, sparc_points, args.massmatch_caliper)
    plot_best_lines(rows, args.outdir)
    plot_heatmaps(rows, args.outdir)
    write_markdown(rows, out_md, out_csv, args.outdir)

    valid = rows[np.isfinite(rows["mm_score"])].sort_values("mm_score", ascending=False)
    print("\nTop 10 matched configs by mm_score:")
    cols = [
        "dm_threshold",
        "min_pts",
        "rmin_kpc",
        "massmatch_pairs",
        "massmatch_mean_abs_dlogM",
        "mm_median_tng",
        "mm_median_sparc",
        "mm_median_ratio",
        "mm_cliffs_delta",
        "mm_mw_p",
        "mm_score",
    ]
    print(valid[cols].head(10).to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved MD: {out_md}")
    print(f"Saved plots under: {args.outdir}")


if __name__ == "__main__":
    main()

