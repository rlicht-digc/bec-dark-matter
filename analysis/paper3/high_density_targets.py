#!/usr/bin/env python3
"""
Identify high-density systems from RAR points and test residual shifts
against the constant-coupling RAR-BEC prediction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


G_NEWTON = 6.67430e-11
KPC_TO_M = 3.085677581491367e19
DEFAULT_G_DAGGER = 1.286e-10
EPS = np.finfo(float).tiny
RNG_SEED = 314159
N_PERM = 10_000
N_BOOT = 5_000
INNER_MIN_POINTS_FOR_MEDIAN = 3


def _run_and_print(cmd: str) -> None:
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.rstrip("\n"))
    if proc.stderr:
        print(proc.stderr.rstrip("\n"), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd}")


def print_step0_discovery(repo_root: Path) -> None:
    print("\n=== STEP 0: LOCATE INPUT DATA ===")
    ls_cmd = f"ls -la {shlex.quote(str(repo_root))}"
    find_cmd = (
        f"find {shlex.quote(str(repo_root))} -type f "
        r"\( -name '*rar*csv' -o -name '*sparc*csv' -o -name '*points*csv' "
        r"-o -name '*unified*csv' -o -name '*.mrt' \) | head -200"
    )
    _run_and_print(ls_cmd)
    _run_and_print(find_cmd)


def _lower_col_map(columns: Sequence[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}


def _pick_column(lower_map: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    return None


def infer_column_mapping(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    lower_map = _lower_col_map(columns)
    mapping = {
        "galaxy": _pick_column(
            lower_map,
            (
                "galaxy",
                "gal_name",
                "name",
                "galaxy_id",
                "objname",
                "id",
            ),
        ),
        "dataset": _pick_column(lower_map, ("source", "dataset", "survey", "sample")),
        "radius": _pick_column(
            lower_map,
            (
                "r_kpc",
                "r(kpc)",
                "r",
                "radius_kpc",
                "radius",
                "rkpc",
                "radii_kpc",
            ),
        ),
        "gbar_linear": _pick_column(
            lower_map,
            (
                "g_bar",
                "gbar",
                "gbar_ms2",
                "gbary",
                "g_bary",
                "g_baryon",
            ),
        ),
        "gbar_log": _pick_column(
            lower_map,
            (
                "log_gbar",
                "log10_gbar",
                "lgbar",
                "log_g_bar",
            ),
        ),
        "gobs_linear": _pick_column(
            lower_map,
            (
                "g_obs",
                "gobs",
                "gobs_ms2",
                "gtot",
                "g_tot",
                "g_total",
            ),
        ),
        "gobs_log": _pick_column(
            lower_map,
            (
                "log_gobs",
                "log10_gobs",
                "lgobs",
                "log_g_obs",
            ),
        ),
        "v_obs": _pick_column(
            lower_map,
            (
                "vobs",
                "v_obs",
                "vobs_kms",
                "v_obs_kms",
                "vobs_km_s",
                "v_rot",
                "vrot",
                "vcirc",
                "v_circ",
                "vtot",
            ),
        ),
        "v_bar": _pick_column(
            lower_map,
            (
                "vbar",
                "v_bar",
                "vbar_kms",
                "v_bar_kms",
                "vbar_km_s",
                "vbary",
                "v_bary",
                "v_bary_kms",
            ),
        ),
    }
    return mapping


def _is_usable_mapping(mapping: Dict[str, Optional[str]]) -> bool:
    has_key_fields = mapping["galaxy"] is not None and mapping["radius"] is not None
    has_acc = (
        (
            (mapping["gbar_linear"] is not None or mapping["gbar_log"] is not None)
            and (mapping["gobs_linear"] is not None or mapping["gobs_log"] is not None)
        )
        or (mapping["v_obs"] is not None and mapping["v_bar"] is not None)
    )
    return bool(has_key_fields and has_acc)


def _score_candidate(path: Path, mapping: Dict[str, Optional[str]]) -> float:
    score = 0.0
    if mapping["galaxy"] is not None:
        score += 20
    if mapping["radius"] is not None:
        score += 20
    if mapping["gbar_linear"] is not None and mapping["gobs_linear"] is not None:
        score += 30
    elif mapping["gbar_log"] is not None and mapping["gobs_log"] is not None:
        score += 28
    elif mapping["v_obs"] is not None and mapping["v_bar"] is not None:
        score += 15
    if mapping["dataset"] is not None:
        score += 5
    if "rar_points_unified" in path.name.lower():
        score += 15
    if path.suffix.lower() == ".csv":
        score += 5
    score += min(path.stat().st_size / 1e7, 5.0)
    return score


def discover_candidate_files(repo_root: Path) -> List[Path]:
    patterns = ("*rar*csv", "*sparc*csv", "*points*csv", "*unified*csv")
    found: List[Path] = []
    for pattern in patterns:
        found.extend(repo_root.rglob(pattern))
    deduped = sorted({p.resolve() for p in found if p.is_file()})
    return deduped


def choose_rar_points_file(repo_root: Path, user_file: Optional[str]) -> Tuple[Path, Dict[str, Optional[str]]]:
    if user_file:
        path = Path(user_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--rar_points_file does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"--rar_points_file must be a CSV for this pipeline: {path}")
        columns = list(pd.read_csv(path, nrows=1).columns)
        mapping = infer_column_mapping(columns)
        if mapping["radius"] is None:
            raise ValueError(
                f"Radius column missing in {path}. Found columns: {columns}"
            )
        if not _is_usable_mapping(mapping):
            raise ValueError(f"File {path} lacks required acceleration fields. Found columns: {columns}")
        return path, mapping

    preferred = repo_root / "analysis" / "results" / "rar_points_unified.csv"
    if preferred.exists():
        columns = list(pd.read_csv(preferred, nrows=1).columns)
        mapping = infer_column_mapping(columns)
        if _is_usable_mapping(mapping):
            return preferred.resolve(), mapping

    candidates = discover_candidate_files(repo_root)
    scored: List[Tuple[float, Path, Dict[str, Optional[str]]]] = []
    for path in candidates:
        if path.suffix.lower() != ".csv":
            continue
        try:
            columns = list(pd.read_csv(path, nrows=1).columns)
        except Exception:
            continue
        mapping = infer_column_mapping(columns)
        if not _is_usable_mapping(mapping):
            continue
        scored.append((_score_candidate(path, mapping), path, mapping))

    if not scored:
        raise RuntimeError("Could not auto-detect a usable RAR points CSV with galaxy, radius, g_bar, g_obs.")

    scored.sort(key=lambda item: item[0], reverse=True)
    _, chosen, mapping = scored[0]
    return chosen.resolve(), mapping


def print_first_lines(path: Path, n_lines: int = 10) -> None:
    print(f"\n=== FIRST {n_lines} LINES OF CHOSEN FILE ===")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if idx >= n_lines:
                break
            print(line.rstrip("\n"))


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _radius_to_kpc(radius_raw: np.ndarray, radius_column_name: str) -> np.ndarray:
    col = radius_column_name.lower()
    if "kpc" in col:
        return radius_raw
    if "mpc" in col:
        return radius_raw * 1_000.0
    if "pc" in col and "kpc" not in col:
        return radius_raw / 1_000.0
    if "(m)" in col or col.endswith("_m") or "meter" in col or "metre" in col:
        return radius_raw / KPC_TO_M
    return radius_raw


def _velocity_to_m_s(velocity: np.ndarray, column_name: str) -> np.ndarray:
    lower = column_name.lower()
    if "km" in lower:
        return velocity * 1_000.0
    if "m/s" in lower or "ms-1" in lower:
        return velocity
    med = np.nanmedian(np.abs(velocity))
    if not np.isfinite(med):
        return velocity
    # Typical galaxy rotation speeds are O(100 km/s).
    if med < 1.0e4:
        return velocity * 1_000.0
    return velocity


def standardize_points(
    raw_df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    assume_r_kpc: bool,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    galaxy_col = mapping["galaxy"]
    radius_col = mapping["radius"]
    if galaxy_col is None or radius_col is None:
        missing = []
        if galaxy_col is None:
            missing.append("galaxy identifier")
        if radius_col is None:
            missing.append("radius")
        raise ValueError(f"Required field(s) missing: {', '.join(missing)}")

    standardized_notes: Dict[str, str] = {}
    galaxy = raw_df[galaxy_col].astype(str).str.strip()
    dataset = (
        raw_df[mapping["dataset"]].astype(str).str.strip()
        if mapping["dataset"] is not None
        else pd.Series("unknown", index=raw_df.index, dtype=object)
    )

    radius_raw = _to_numeric(raw_df[radius_col])
    r_kpc = _radius_to_kpc(radius_raw, radius_col)
    standardized_notes["radius"] = f"{radius_col} -> r_kpc"

    positive_r = r_kpc[np.isfinite(r_kpc) & (r_kpc > 0)]
    median_r = float(np.nanmedian(positive_r)) if positive_r.size else np.nan
    if np.isfinite(median_r) and median_r > 1000.0 and not assume_r_kpc:
        raise ValueError(
            "Radius unit sanity failed: median r_kpc > 1000. "
            "Use --assume_r_kpc to override."
        )

    if np.isfinite(median_r) and median_r > 1000.0 and assume_r_kpc:
        print(
            "WARNING: median radius > 1000 kpc but --assume_r_kpc provided; proceeding as requested."
        )

    g_bar: Optional[np.ndarray] = None
    g_obs: Optional[np.ndarray] = None

    if mapping["gbar_linear"] is not None:
        g_bar = _to_numeric(raw_df[mapping["gbar_linear"]])
        standardized_notes["g_bar"] = mapping["gbar_linear"]
    elif mapping["gbar_log"] is not None:
        g_bar = np.power(10.0, _to_numeric(raw_df[mapping["gbar_log"]]))
        standardized_notes["g_bar"] = f"10**{mapping['gbar_log']}"

    if mapping["gobs_linear"] is not None:
        g_obs = _to_numeric(raw_df[mapping["gobs_linear"]])
        standardized_notes["g_obs"] = mapping["gobs_linear"]
    elif mapping["gobs_log"] is not None:
        g_obs = np.power(10.0, _to_numeric(raw_df[mapping["gobs_log"]]))
        standardized_notes["g_obs"] = f"10**{mapping['gobs_log']}"

    if g_bar is None or g_obs is None:
        if mapping["v_obs"] is None or mapping["v_bar"] is None:
            raise ValueError(
                "Could not build g_bar/g_obs from columns. Need g_bar/g_obs directly "
                "or both baryonic and observed velocity columns."
            )
        r_m = r_kpc * KPC_TO_M
        v_obs = _velocity_to_m_s(_to_numeric(raw_df[mapping["v_obs"]]), mapping["v_obs"])
        v_bar = _velocity_to_m_s(_to_numeric(raw_df[mapping["v_bar"]]), mapping["v_bar"])
        g_obs = np.square(v_obs) / r_m
        g_bar = np.square(v_bar) / r_m
        standardized_notes["g_obs"] = f"{mapping['v_obs']}^2 / r_m"
        standardized_notes["g_bar"] = f"{mapping['v_bar']}^2 / r_m"

    std = pd.DataFrame(
        {
            "galaxy": galaxy,
            "dataset": dataset,
            "r_kpc": r_kpc,
            "g_bar": g_bar,
            "g_obs": g_obs,
        }
    )
    return std, standardized_notes


def apply_quality_filters(points: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    counts = {"input_rows": int(len(points))}
    filtered = points.copy()
    filtered = filtered.replace([np.inf, -np.inf], np.nan)
    filtered = filtered.dropna(subset=["galaxy", "r_kpc", "g_bar", "g_obs"])
    counts["after_nonfinite_drop"] = int(len(filtered))
    filtered = filtered[
        (filtered["r_kpc"] > 0.0) & (filtered["g_bar"] > 0.0) & (filtered["g_obs"] > 0.0)
    ].copy()
    counts["after_physical_cuts"] = int(len(filtered))
    return filtered, counts


def compute_rar_prediction(g_bar: np.ndarray, g_dagger: float) -> np.ndarray:
    ratio = np.clip(g_bar / g_dagger, a_min=0.0, a_max=None)
    x = np.sqrt(ratio)
    denom = -np.expm1(-x)
    g_pred = np.divide(g_bar, denom, out=np.full_like(g_bar, np.nan, dtype=float), where=denom > EPS)
    return g_pred


def summarize_galaxies(
    points: pd.DataFrame,
    g_dagger: float,
    r_inner_max_kpc: float,
    min_points: int,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    log_gdag = math.log10(g_dagger)

    for galaxy, gdf in points.groupby("galaxy", sort=False):
        n_points = int(len(gdf))
        dataset = (
            gdf["dataset"].mode(dropna=True).iloc[0]
            if not gdf["dataset"].dropna().empty
            else "unknown"
        )

        inner = gdf[gdf["r_kpc"] <= r_inner_max_kpc]
        rho_dyn_max = float(inner["rho_dyn"].max()) if not inner.empty else np.nan
        rho_dyn_p50_inner = float(inner["rho_dyn"].median()) if not inner.empty else np.nan

        idx = (np.abs(np.log10(gdf["g_bar"]) - log_gdag)).idxmin()
        row_gdag = gdf.loc[idx]
        rho_dyn_at_gdag = float(row_gdag["rho_dyn"])
        r_at_gdag_kpc = float(row_gdag["r_kpc"])

        median_log_resid = float(np.median(gdf["log_resid"]))
        p90_abs_log_resid = float(np.percentile(np.abs(gdf["log_resid"]), 90))
        rms_log_resid = float(np.sqrt(np.mean(np.square(gdf["log_resid"]))))

        notes: List[str] = []
        flag_low = n_points < min_points
        if flag_low:
            notes.append(f"N_points<{min_points}")

        if len(inner) >= INNER_MIN_POINTS_FOR_MEDIAN and np.isfinite(rho_dyn_p50_inner):
            rho_score = rho_dyn_p50_inner
        elif np.isfinite(rho_dyn_max):
            rho_score = rho_dyn_max
            notes.append("rho_score_fallback=max_inner")
        else:
            rho_score = float(gdf["rho_dyn"].max())
            notes.append("rho_score_fallback=max_global")

        if len(inner) == 0:
            notes.append("no_points_within_inner_window")
        elif len(inner) < INNER_MIN_POINTS_FOR_MEDIAN:
            notes.append(f"inner_points<{INNER_MIN_POINTS_FOR_MEDIAN}")

        records.append(
            {
                "galaxy": galaxy,
                "dataset": dataset,
                "N_points": n_points,
                "rho_dyn_max": rho_dyn_max,
                "rho_dyn_p50_inner": rho_dyn_p50_inner,
                "rho_dyn_at_gdag": rho_dyn_at_gdag,
                "r_at_gdag_kpc": r_at_gdag_kpc,
                "rho_score": rho_score,
                "median_log_resid": median_log_resid,
                "p90_abs_log_resid": p90_abs_log_resid,
                "rms_log_resid": rms_log_resid,
                "flag_low_points": bool(flag_low),
                "notes": "; ".join(notes) if notes else "",
            }
        )

    summary = pd.DataFrame.from_records(records)
    return summary


def permutation_test_median_difference(
    top_vals: np.ndarray, bottom_vals: np.ndarray, n_perm: int, rng: np.random.Generator
) -> Tuple[float, float]:
    observed = float(np.median(top_vals) - np.median(bottom_vals))
    combined = np.concatenate([top_vals, bottom_vals])
    n_top = len(top_vals)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = float(np.median(perm[:n_top]) - np.median(perm[n_top:]))
        if abs(diff) >= abs(observed):
            extreme += 1
    p_val = (extreme + 1) / (n_perm + 1)
    return observed, float(p_val)


def permutation_spearman_pvalue(
    x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator
) -> Tuple[float, float]:
    corr_obs, _ = stats.spearmanr(x, y)
    if not np.isfinite(corr_obs):
        return np.nan, np.nan
    extreme = 0
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        corr_perm, _ = stats.spearmanr(x, perm_y)
        if np.isfinite(corr_perm) and abs(corr_perm) >= abs(corr_obs):
            extreme += 1
    p_val = (extreme + 1) / (n_perm + 1)
    return float(corr_obs), float(p_val)


def bootstrap_theilsen_slope(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator
) -> Tuple[float, float, float]:
    slopes: List[float] = []
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        if np.nanstd(xb) <= 0:
            continue
        slope, _, _, _ = stats.theilslopes(yb, xb, alpha=0.95)
        if np.isfinite(slope):
            slopes.append(float(slope))
    if not slopes:
        return np.nan, np.nan, np.nan
    arr = np.array(slopes)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p_boot = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    return float(lo), float(hi), float(p_boot)


def compute_group_stats(
    points: pd.DataFrame,
    galaxy_summary: pd.DataFrame,
    top_n: int,
    min_points: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    eligible = galaxy_summary[
        (~galaxy_summary["flag_low_points"])
        & np.isfinite(galaxy_summary["rho_score"])
        & np.isfinite(galaxy_summary["median_log_resid"])
    ].copy()

    eligible = eligible.sort_values("rho_score", ascending=False).reset_index(drop=True)
    top = eligible.head(top_n).copy()

    bottom_pool = eligible.loc[~eligible["galaxy"].isin(top["galaxy"])].copy()
    bottom = bottom_pool.sort_values("rho_score", ascending=True).head(top_n).copy()

    top_galaxies = set(top["galaxy"])
    bottom_galaxies = set(bottom["galaxy"])
    top_resid = points[points["galaxy"].isin(top_galaxies)]["log_resid"].to_numpy(dtype=float)
    bottom_resid = points[points["galaxy"].isin(bottom_galaxies)]["log_resid"].to_numpy(dtype=float)

    result: Dict[str, object] = {
        "eligible": eligible,
        "top": top,
        "bottom": bottom,
        "top_resid": top_resid,
        "bottom_resid": bottom_resid,
        "n_eligible": int(len(eligible)),
        "min_points": min_points,
    }

    if len(top_resid) > 1 and len(bottom_resid) > 1:
        delta_median, perm_p = permutation_test_median_difference(top_resid, bottom_resid, N_PERM, rng)
        ks = stats.ks_2samp(top_resid, bottom_resid, alternative="two-sided", method="auto")
        result["delta_median_top_minus_bottom"] = delta_median
        result["perm_p_median_diff"] = perm_p
        result["ks_stat"] = float(ks.statistic)
        result["ks_p"] = float(ks.pvalue)
    else:
        result["delta_median_top_minus_bottom"] = np.nan
        result["perm_p_median_diff"] = np.nan
        result["ks_stat"] = np.nan
        result["ks_p"] = np.nan

    x = np.log10(eligible["rho_score"].to_numpy(dtype=float))
    y = eligible["median_log_resid"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) >= 2:
        slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
        ci_lo, ci_hi, slope_boot_p = bootstrap_theilsen_slope(x, y, N_BOOT, rng)
        rho_s, rho_perm_p = permutation_spearman_pvalue(x, y, N_PERM, rng)
        result.update(
            {
                "trend_slope": float(slope),
                "trend_intercept": float(intercept),
                "trend_slope_boot_ci_lo": ci_lo,
                "trend_slope_boot_ci_hi": ci_hi,
                "trend_slope_boot_p": slope_boot_p,
                "spearman_rho": rho_s,
                "spearman_perm_p": rho_perm_p,
            }
        )
    else:
        result.update(
            {
                "trend_slope": np.nan,
                "trend_intercept": np.nan,
                "trend_slope_boot_ci_lo": np.nan,
                "trend_slope_boot_ci_hi": np.nan,
                "trend_slope_boot_p": np.nan,
                "spearman_rho": np.nan,
                "spearman_perm_p": np.nan,
            }
        )

    return result


def _save_scatter_plot(
    out_path: Path, summary: pd.DataFrame, trend_slope: float, trend_intercept: float
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = summary[np.isfinite(summary["rho_score"]) & np.isfinite(summary["median_log_resid"])]
    ax.scatter(plot_df["rho_score"], plot_df["median_log_resid"], s=24, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho_{\mathrm{score}}$ (kg m$^{-3}$)")
    ax.set_ylabel("Median log residual (dex)")
    ax.set_title("RAR Residual vs Dynamical Density Proxy")

    if np.isfinite(trend_slope) and np.isfinite(trend_intercept) and not plot_df.empty:
        x_grid = np.geomspace(plot_df["rho_score"].min(), plot_df["rho_score"].max(), 300)
        y_grid = trend_intercept + trend_slope * np.log10(x_grid)
        ax.plot(x_grid, y_grid, color="crimson", lw=2.0)

    ax.axhline(0.0, color="gray", lw=1.0, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_hist_plot(out_path: Path, top_resid: np.ndarray, bottom_resid: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(top_resid) > 0 and len(bottom_resid) > 0:
        all_vals = np.concatenate([top_resid, bottom_resid])
        bins = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 40)
        ax.hist(top_resid, bins=bins, alpha=0.55, label="Top-density", density=True)
        ax.hist(bottom_resid, bins=bins, alpha=0.55, label="Bottom-density", density=True)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient data for both groups", transform=ax.transAxes, ha="center")
    ax.set_xlabel("log residual (dex)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution: High vs Low Density Galaxies")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_residual_vs_gbar_plot(out_path: Path, points_top: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not points_top.empty:
        x = np.log10(points_top["g_bar"].to_numpy(dtype=float))
        y = points_top["log_resid"].to_numpy(dtype=float)
        ax.scatter(x, y, s=12, alpha=0.4)
    else:
        ax.text(0.5, 0.5, "No top-density points available", transform=ax.transAxes, ha="center")
    ax.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax.set_xlabel(r"log10 $g_{\mathrm{bar}}$ (m s$^{-2}$)")
    ax.set_ylabel("log residual (dex)")
    ax.set_title("RAR Residual vs log g_bar (Top Density Galaxies)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_report(
    out_path: Path,
    input_file: Path,
    column_mapping: Dict[str, Optional[str]],
    conversion_notes: Dict[str, str],
    filter_counts: Dict[str, int],
    g_dagger: float,
    top_targets: pd.DataFrame,
    stats_result: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("paper3_density_window_report")
    lines.append("")
    lines.append("Data sources used and column mapping")
    lines.append(f"- Input file: {input_file}")
    lines.append("- Column mapping:")
    for key in (
        "galaxy",
        "dataset",
        "radius",
        "gbar_linear",
        "gbar_log",
        "gobs_linear",
        "gobs_log",
        "v_obs",
        "v_bar",
    ):
        lines.append(f"  {key}: {column_mapping.get(key)}")
    lines.append("- Conversion notes:")
    for key, value in conversion_notes.items():
        lines.append(f"  {key}: {value}")
    lines.append("- Filtering summary:")
    for key, value in filter_counts.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append(f"g_dagger used: {g_dagger:.6e} m/s^2")
    lines.append("")

    lines.append(f"Top {len(top_targets)} systems by rho_score")
    for _, row in top_targets.iterrows():
        lines.append(f"- {row['galaxy']}: rho_score={row['rho_score']:.6e}, N_points={int(row['N_points'])}")
    lines.append("")

    lines.append("Residual shift with density")
    lines.append(f"- Theil-Sen slope (median_log_resid vs log10(rho_score)): {stats_result.get('trend_slope', np.nan):.6f}")
    lines.append(
        "- Bootstrap 95% CI: "
        f"[{stats_result.get('trend_slope_boot_ci_lo', np.nan):.6f}, "
        f"{stats_result.get('trend_slope_boot_ci_hi', np.nan):.6f}]"
    )
    lines.append(f"- Bootstrap sign p-value (slope): {stats_result.get('trend_slope_boot_p', np.nan):.6g}")
    lines.append(f"- Spearman rho: {stats_result.get('spearman_rho', np.nan):.6f}")
    lines.append(f"- Spearman permutation p-value: {stats_result.get('spearman_perm_p', np.nan):.6g}")
    lines.append(
        "- Top-vs-bottom residual median difference (top-bottom): "
        f"{stats_result.get('delta_median_top_minus_bottom', np.nan):.6f} dex"
    )
    lines.append(f"- KS statistic: {stats_result.get('ks_stat', np.nan):.6f}")
    lines.append(f"- KS p-value: {stats_result.get('ks_p', np.nan):.6g}")
    lines.append(f"- Permutation p-value (median shift): {stats_result.get('perm_p_median_diff', np.nan):.6g}")
    lines.append("")

    target_count = min(20, max(10, len(top_targets)))
    recommended = top_targets.head(target_count)
    lines.append(f"Recommended observational target list ({len(recommended)} names)")
    for _, row in recommended.iterrows():
        lines.append(
            f"- {row['galaxy']} (rho_score={row['rho_score']:.6e}, "
            f"median_log_resid={row['median_log_resid']:.4f}, N_points={int(row['N_points'])})"
        )
    lines.append("")

    lines.append("Limitations / missing fields")
    lines.append("- rho_dyn is a proxy from enclosed acceleration and radius, not a full profile inversion.")
    lines.append("- Residual-group tests pool point-level residuals and can be influenced by uneven per-galaxy sampling.")
    lines.append("- If velocity-derived accelerations are used, they depend on velocity unit inference.")
    lines.append("- Galaxies with N_points < min_points are retained in global tables but excluded from eligibility ranking.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-density target selection from RAR points.")
    parser.add_argument("--rar_points_file", type=str, default=None)
    parser.add_argument("--g_dagger", type=float, default=DEFAULT_G_DAGGER)
    parser.add_argument("--r_inner_max_kpc", type=float, default=2.0)
    parser.add_argument("--min_points", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--assume_r_kpc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = repo_root / "outputs" / "paper3_high_density" / timestamp
    else:
        user_out = Path(args.out_dir).expanduser()
        out_dir = user_out if user_out.is_absolute() else (repo_root / user_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_step0_discovery(repo_root)
    input_file, mapping = choose_rar_points_file(repo_root, args.rar_points_file)
    print(f"\nChosen RAR points file: {input_file}")
    print_first_lines(input_file, n_lines=10)

    raw = pd.read_csv(input_file)
    points_std, conversion_notes = standardize_points(raw, mapping, args.assume_r_kpc)
    points, filter_counts = apply_quality_filters(points_std)
    if points.empty:
        raise RuntimeError("No usable points after filtering.")

    g_pred = compute_rar_prediction(points["g_bar"].to_numpy(dtype=float), args.g_dagger)
    points["g_pred"] = g_pred
    points["log_resid"] = np.log10(points["g_obs"]) - np.log10(points["g_pred"])
    points["rho_dyn"] = (3.0 * points["g_obs"]) / (4.0 * np.pi * G_NEWTON * (points["r_kpc"] * KPC_TO_M))
    points = points.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["g_pred", "log_resid", "rho_dyn"]
    )
    if points.empty:
        raise RuntimeError("No usable points after residual/rho computation.")

    galaxy_summary = summarize_galaxies(
        points=points,
        g_dagger=args.g_dagger,
        r_inner_max_kpc=args.r_inner_max_kpc,
        min_points=args.min_points,
    )
    if galaxy_summary.empty:
        raise RuntimeError("No per-galaxy summary rows produced.")

    rng = np.random.default_rng(RNG_SEED)
    stats_result = compute_group_stats(
        points=points,
        galaxy_summary=galaxy_summary,
        top_n=args.top_n,
        min_points=args.min_points,
        rng=rng,
    )

    eligible = stats_result["eligible"]
    top_targets = stats_result["top"]

    high_density_cols = [
        "galaxy",
        "N_points",
        "rho_dyn_max",
        "rho_dyn_p50_inner",
        "rho_dyn_at_gdag",
        "r_at_gdag_kpc",
        "median_log_resid",
        "p90_abs_log_resid",
        "flag_low_points",
        "notes",
    ]
    high_density_csv = out_dir / "high_density_targets.csv"
    top_targets.loc[:, high_density_cols].to_csv(high_density_csv, index=False)

    density_vs_residuals = out_dir / "density_vs_residuals.csv"
    galaxy_summary.loc[
        :,
        ["galaxy", "rho_score", "median_log_resid", "rms_log_resid", "N_points", "dataset"],
    ].to_csv(density_vs_residuals, index=False)

    scatter_plot = out_dir / "rho_vs_residual_scatter.png"
    hist_plot = out_dir / "residual_hist_high_vs_low.png"
    residual_vs_gbar_plot = out_dir / "residual_vs_gbar_high_density.png"

    _save_scatter_plot(
        scatter_plot,
        eligible if not eligible.empty else galaxy_summary,
        float(stats_result.get("trend_slope", np.nan)),
        float(stats_result.get("trend_intercept", np.nan)),
    )
    _save_hist_plot(
        hist_plot,
        np.asarray(stats_result.get("top_resid", []), dtype=float),
        np.asarray(stats_result.get("bottom_resid", []), dtype=float),
    )
    top_names = set(top_targets["galaxy"].tolist())
    _save_residual_vs_gbar_plot(
        residual_vs_gbar_plot,
        points[points["galaxy"].isin(top_names)].copy(),
    )

    report_path = out_dir / "paper3_density_window_report.txt"
    build_report(
        out_path=report_path,
        input_file=input_file,
        column_mapping=mapping,
        conversion_notes=conversion_notes,
        filter_counts=filter_counts,
        g_dagger=args.g_dagger,
        top_targets=top_targets,
        stats_result=stats_result,
    )

    print("\n=== FINAL OUTPUT REQUIRED ===")
    print(f"Input file used: {input_file}")
    print("Top target list (galaxy, rho_score):")
    if top_targets.empty:
        print("  [none]")
    else:
        for _, row in top_targets.iterrows():
            print(f"  {row['galaxy']}, {row['rho_score']:.6e}")
    print(
        "Trend slope and p-values: "
        f"slope={stats_result.get('trend_slope', np.nan):.6f}, "
        f"slope_boot_p={stats_result.get('trend_slope_boot_p', np.nan):.6g}, "
        f"median_shift_perm_p={stats_result.get('perm_p_median_diff', np.nan):.6g}"
    )
    print("Output files:")
    print(f"  {high_density_csv}")
    print(f"  {density_vs_residuals}")
    print(f"  {scatter_plot}")
    print(f"  {hist_plot}")
    print(f"  {residual_vs_gbar_plot}")
    print(f"  {report_path}")
    print("STOP.")


if __name__ == "__main__":
    main()
