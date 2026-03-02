#!/usr/bin/env python3
"""
Nucleation-gap diagnostics for constrained finite-T runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = "/Users/russelllicht/bec-dark-matter/outputs/paper3_finiteT/20260301_032629_constrained"


def parse_window(text: str) -> Tuple[float, float]:
    parts = [x.strip() for x in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("Window must be 'lo,hi'")
    lo = float(parts[0])
    hi = float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid window: {text}")
    return lo, hi


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    low = s.astype(str).str.strip().str.lower()
    return low.isin({"1", "true", "t", "yes", "y"})


def finite_stats(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0.0,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": float(x.size),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def write_report(
    out_md: Path,
    summary_row: Dict[str, object],
    closest: pd.DataFrame,
    tc_window: Tuple[float, float],
    s3_target: float,
) -> None:
    lines = []
    lines.append("# Nucleation Gap Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- run_dir: {summary_row['run_dir']}")
    lines.append(f"- Tc_window: [{tc_window[0]}, {tc_window[1]}] GeV")
    lines.append(f"- S3/T target: {s3_target}")
    lines.append(f"- N_total_eval: {summary_row['N_total_eval']}")
    lines.append(f"- N_first_order: {summary_row['N_first_order']}")
    lines.append(f"- N_Tc_in_window: {summary_row['N_Tc_in_window']}")
    lines.append(f"- N_relevant(first-order + Tc window): {summary_row['N_relevant']}")
    lines.append("")
    lines.append("## min(S3/T) Diagnostics")
    lines.append(f"- median: {summary_row['minS3_median']}")
    lines.append(f"- p10: {summary_row['minS3_p10']}")
    lines.append(f"- p90: {summary_row['minS3_p90']}")
    lines.append(f"- min: {summary_row['minS3_min']}")
    lines.append(f"- max: {summary_row['minS3_max']}")
    lines.append("")
    lines.append("## T_at_min(S3/T) Diagnostics")
    lines.append(f"- median: {summary_row['TminS3_median']}")
    lines.append(f"- min: {summary_row['TminS3_min']}")
    lines.append(f"- max: {summary_row['TminS3_max']}")
    lines.append("")
    lines.append("## Threshold Fractions")
    lines.append(f"- min_S3/T <= 200: {summary_row['n_le_200']} ({summary_row['f_le_200']})")
    lines.append(f"- min_S3/T <= 180: {summary_row['n_le_180']} ({summary_row['f_le_180']})")
    lines.append(f"- min_S3/T <= 160: {summary_row['n_le_160']} ({summary_row['f_le_160']})")
    lines.append(f"- min_S3/T <= 140: {summary_row['n_le_140']} ({summary_row['f_le_140']})")
    lines.append("")
    lines.append("## Gap Score")
    lines.append(f"- gap_min = min_S3/T_min - target = {summary_row['gap_min']}")
    lines.append(f"- interpretation: {summary_row['gap_interpretation']}")
    lines.append("")

    lines.append("## Closest Case")
    if closest.empty:
        lines.append("- No finite min(S3/T) rows in relevant set.")
    else:
        r = closest.iloc[0]
        lines.append(
            f"- candidate_id={r.get('candidate_id', np.nan)}, Tc={r.get('Tc', np.nan)}, "
            f"E_used={r.get('E_used', np.nan)}, g_portal={r.get('g_portal_used', np.nan)}, "
            f"kE={r.get('kE_used', np.nan)}, min_S3/T={r.get('min_S3_over_T', np.nan)}, "
            f"T_at_min={r.get('T_at_min_S3_over_T', np.nan)}"
        )
    lines.append("")

    near_miss = bool(summary_row["n_le_180"] > 0)
    lines.append("## Conclusion")
    if near_miss:
        lines.append("- near-miss: at least one relevant case has min(S3/T) <= 180.")
        lines.append("- recommendation: refine T scan near minima, increase portal strength, and add one-loop thermal terms.")
    else:
        lines.append("- not close: no relevant case reaches min(S3/T) <= 180.")
        lines.append("- recommendation: consider a different potential family or drop the ~20 GeV bubble nucleation claim.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nucleation-gap dashboard for constrained finite-T outputs.")
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--S3_over_T_target", type=float, default=140.0)
    parser.add_argument("--Tc_window", type=str, default="10,50")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()

    expected = [
        run_dir / "constrained_cases_used.csv",
        run_dir / "constrained_finiteT_results.csv",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print("[GAP] Expected files not found. Directory listing:")
        if run_dir.exists():
            for p in sorted(run_dir.iterdir()):
                print(f"  {p.name}")
        else:
            print(f"  run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = repo_root / "outputs" / "paper3_finiteT_gap" / timestamp
    else:
        p = Path(args.out_dir).expanduser()
        out_dir = p if p.is_absolute() else (repo_root / p)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "nucleation_gap_summary.csv"
    closest_csv = out_dir / "nucleation_gap_closest_cases.csv"
    plots_png = out_dir / "nucleation_gap_plots.png"
    report_md = out_dir / "nucleation_gap_report.md"

    df = pd.read_csv(run_dir / "constrained_finiteT_results.csv")
    tc_lo, tc_hi = parse_window(args.Tc_window)
    s3_target = float(args.S3_over_T_target)

    m_first = bool_series(df["first_order_flag"]) if "first_order_flag" in df.columns else pd.Series(False, index=df.index)
    m_tc_found = bool_series(df["Tc_found"]) if "Tc_found" in df.columns else df["Tc"].apply(np.isfinite)
    m_tc_in_window = m_tc_found & np.isfinite(df["Tc"].to_numpy(dtype=float)) & (df["Tc"].to_numpy(dtype=float) >= tc_lo) & (
        df["Tc"].to_numpy(dtype=float) <= tc_hi
    )
    relevant = df[m_first & m_tc_in_window].copy()

    min_s3 = relevant["min_S3_over_T"].to_numpy(dtype=float) if "min_S3_over_T" in relevant.columns else np.array([], dtype=float)
    t_at_min = relevant["T_at_min_S3_over_T"].to_numpy(dtype=float) if "T_at_min_S3_over_T" in relevant.columns else np.array([], dtype=float)
    min_s3_finite = min_s3[np.isfinite(min_s3)]
    t_at_min_finite = t_at_min[np.isfinite(t_at_min)]

    s3_stats = finite_stats(min_s3)
    t_stats = finite_stats(t_at_min)

    denom = float(len(relevant))
    n_le_200 = int(np.sum(min_s3_finite <= 200.0))
    n_le_180 = int(np.sum(min_s3_finite <= 180.0))
    n_le_160 = int(np.sum(min_s3_finite <= 160.0))
    n_le_140 = int(np.sum(min_s3_finite <= 140.0))

    def frac(n: int) -> float:
        if denom <= 0:
            return float("nan")
        return float(n / denom)

    closest = relevant[np.isfinite(relevant["min_S3_over_T"].to_numpy(dtype=float))].sort_values("min_S3_over_T").head(
        int(max(1, args.top_k))
    )
    closest_cols = [
        "candidate_id",
        "log10_V0_ratio",
        "g_portal_used",
        "kE_used",
        "E_used",
        "Tc",
        "barrier_height_Tc",
        "min_S3_over_T",
        "T_at_min_S3_over_T",
        "alpha_PT",
        "beta_over_H",
        "lambda4",
        "lambda6",
        "v_GeV",
        "cT2",
    ]
    closest_keep = [c for c in closest_cols if c in closest.columns]
    closest = closest.loc[:, closest_keep] if not closest.empty else pd.DataFrame(columns=closest_cols)
    closest.to_csv(closest_csv, index=False)

    if min_s3_finite.size > 0:
        best = relevant[np.isfinite(relevant["min_S3_over_T"].to_numpy(dtype=float))].sort_values("min_S3_over_T").iloc[0]
        gap_min = float(best["min_S3_over_T"] - s3_target)
        closest_case = {
            "candidate_id": float(best["candidate_id"]) if "candidate_id" in best else np.nan,
            "Tc": float(best["Tc"]) if np.isfinite(best["Tc"]) else np.nan,
            "E_used": float(best["E_used"]) if "E_used" in best else np.nan,
            "min_S3_over_T": float(best["min_S3_over_T"]),
            "kE_used": float(best["kE_used"]) if "kE_used" in best and np.isfinite(best["kE_used"]) else np.nan,
            "g_portal_used": float(best["g_portal_used"]) if "g_portal_used" in best and np.isfinite(best["g_portal_used"]) else np.nan,
        }
    else:
        gap_min = np.nan
        closest_case = {"candidate_id": np.nan, "Tc": np.nan, "E_used": np.nan, "min_S3_over_T": np.nan, "kE_used": np.nan, "g_portal_used": np.nan}

    if np.isfinite(gap_min):
        if gap_min < 20:
            gap_interp = "very close (likely salvageable)"
        elif gap_min <= 200:
            gap_interp = "possible with model refinement"
        elif gap_min > 500:
            gap_interp = "likely far from nucleation under this EFT"
        else:
            gap_interp = "not yet close"
    else:
        gap_interp = "no finite min_S3_over_T in relevant set"

    summary_row: Dict[str, object] = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "run_dir": str(run_dir),
        "Tc_window_lo": tc_lo,
        "Tc_window_hi": tc_hi,
        "S3_over_T_target": s3_target,
        "N_total_eval": int(len(df)),
        "N_first_order": int(np.sum(m_first)),
        "N_Tc_in_window": int(np.sum(m_tc_in_window)),
        "N_relevant": int(len(relevant)),
        "N_relevant_finite_minS3": int(min_s3_finite.size),
        "minS3_median": s3_stats["median"],
        "minS3_p10": s3_stats["p10"],
        "minS3_p90": s3_stats["p90"],
        "minS3_min": s3_stats["min"],
        "minS3_max": s3_stats["max"],
        "TminS3_median": t_stats["median"],
        "TminS3_min": t_stats["min"],
        "TminS3_max": t_stats["max"],
        "n_le_200": n_le_200,
        "f_le_200": frac(n_le_200),
        "n_le_180": n_le_180,
        "f_le_180": frac(n_le_180),
        "n_le_160": n_le_160,
        "f_le_160": frac(n_le_160),
        "n_le_140": n_le_140,
        "f_le_140": frac(n_le_140),
        "gap_min": gap_min,
        "gap_interpretation": gap_interp,
        "closest_candidate_id": closest_case["candidate_id"],
        "closest_Tc": closest_case["Tc"],
        "closest_E_used": closest_case["E_used"],
        "closest_kE_used": closest_case["kE_used"],
        "closest_g_portal_used": closest_case["g_portal_used"],
        "closest_min_S3_over_T": closest_case["min_S3_over_T"],
    }
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    # Plot panels.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.flat

    if min_s3_finite.size > 0:
        axA.hist(min_s3_finite, bins=20, color="#1f77b4", alpha=0.75)
        axA.axvline(140, color="#d62728", lw=1.5, ls="--")
        axA.axvline(160, color="#ff7f0e", lw=1.5, ls="--")
    else:
        axA.text(0.5, 0.5, "No finite min(S3/T)", ha="center", transform=axA.transAxes)
    axA.set_xlabel("min(S3/T)")
    axA.set_ylabel("Count")
    axA.set_title("A) min(S3/T) histogram")

    def scatter_panel(ax, x, y, c, xlabel):
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        if np.any(m):
            sc = ax.scatter(x[m], y[m], c=c[m], s=28, alpha=0.8, cmap="viridis")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.axhline(140, color="#d62728", lw=1.2, ls="--")
            ax.axhline(160, color="#ff7f0e", lw=1.2, ls="--")
        else:
            ax.text(0.5, 0.5, "No finite points", ha="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("min(S3/T)")

    color_col = "g_portal_used" if "g_portal_used" in relevant.columns and np.any(np.isfinite(relevant["g_portal_used"].to_numpy(dtype=float))) else "E_used"
    cvals = relevant[color_col].to_numpy(dtype=float) if color_col in relevant.columns else np.zeros(len(relevant))
    scatter_panel(axB, relevant["Tc"].to_numpy(dtype=float), relevant["min_S3_over_T"].to_numpy(dtype=float), cvals, "Tc [GeV]")
    axB.set_title(f"B) Tc vs min(S3/T), color={color_col}")

    scatter_panel(
        axC,
        relevant["barrier_height_Tc"].to_numpy(dtype=float) if "barrier_height_Tc" in relevant.columns else np.full(len(relevant), np.nan),
        relevant["min_S3_over_T"].to_numpy(dtype=float),
        cvals,
        "barrier_height_Tc",
    )
    axC.set_title("C) barrier_height_Tc vs min(S3/T)")

    scatter_panel(
        axD,
        relevant["T_at_min_S3_over_T"].to_numpy(dtype=float) if "T_at_min_S3_over_T" in relevant.columns else np.full(len(relevant), np.nan),
        relevant["min_S3_over_T"].to_numpy(dtype=float),
        cvals,
        "T_at_min(S3/T) [GeV]",
    )
    axD.set_title("D) T_at_min(S3/T) vs min(S3/T)")

    fig.tight_layout()
    fig.savefig(plots_png, dpi=180)
    plt.close(fig)

    write_report(report_md, summary_row, closest, tc_window=(tc_lo, tc_hi), s3_target=s3_target)

    print(
        f"gap_min={summary_row['gap_min']}, "
        f"closest_case: Tc={summary_row['closest_Tc']}, E_used={summary_row['closest_E_used']}, "
        f"min_S3/T={summary_row['closest_min_S3_over_T']}"
    )
    print(f"count(min_S3/T<=200)={summary_row['n_le_200']}, count(min_S3/T<=160)={summary_row['n_le_160']}")
    print(f"output_dir={out_dir}")
    print(f"files:\n  {summary_csv}\n  {closest_csv}\n  {plots_png}\n  {report_md}")


if __name__ == "__main__":
    main()
