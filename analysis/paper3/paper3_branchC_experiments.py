#!/usr/bin/env python3
"""
Branch C decisive experiments (C1-C3) for Paper3 density bridge.

C1: Galaxy-correct pooled tests
C2: log_gbar-distribution matched shifts
C3: Source-stratified replication
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse existing Paper3 bridge machinery for exact definitions.
import paper3_bridge_pack as pb

DEFAULT_REQUIRE_CSV_SHA = "11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c"
DEFAULT_RAR_POINTS_FILE = "analysis/results/rar_points_unified.csv"
DEFAULT_R_INNER_MAX_KPC = 2.0
DEFAULT_MIN_POINTS_PER_GALAXY = 20
DEFAULT_TOP_N = 20
DEFAULT_N_PERM = 10_000
DEFAULT_N_PERM_FALLBACK = 5_000
DEFAULT_N_BINS = 15
DEFAULT_MIN_BIN_GROUP_POINTS = 30
DEFAULT_N_BOOT_BIN = 800
DEFAULT_SEED = 271828
SUBSET_MIN_TOP_N = 5


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_stamp_compact() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _parse_csv_list(raw: str) -> List[str]:
    vals = [v.strip() for v in str(raw).split(",")]
    return [v for v in vals if v]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        return np.nan
    v = v[good]
    w = w[good]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    csum = np.cumsum(w)
    cutoff = 0.5 * float(np.sum(w))
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _safe_median(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.median(a))


def _make_out_dir(repo_root: Path, out_dir_arg: Optional[str]) -> Path:
    if out_dir_arg:
        p = Path(out_dir_arg).expanduser()
        out_dir = p if p.is_absolute() else (repo_root / p)
    else:
        out_dir = repo_root / "outputs" / "paper3_high_density" / f"BRANCHC_C1C2C3_{utc_stamp_compact()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    return out_dir


def _prepare_points(
    repo_root: Path,
    rar_points_file: Optional[str],
    g_dagger: float,
    assume_r_kpc: bool,
    include_ss20: bool,
    require_csv_sha: Optional[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    input_file, mapping = pb.choose_rar_points_file(repo_root, rar_points_file)
    csv_sha = pb.sha256_file(input_file)

    if require_csv_sha is not None:
        expect = str(require_csv_sha).strip().lower()
        got = str(csv_sha).strip().lower()
        if expect != got:
            raise RuntimeError(f"Dataset hash mismatch: expected {expect}, got {got}")

    raw = pd.read_csv(input_file)
    points_std, conversion_notes, all_galaxies = pb.standardize_points(raw, mapping, assume_r_kpc=assume_r_kpc)
    points, filter_counts = pb.apply_quality_filters(points_std)
    if points.empty:
        raise RuntimeError("No valid points after quality filters.")

    ss20_excluded_points = 0
    if not include_ss20:
        ss20_mask = points["dataset"].astype(str).str.match(r"^SS20_")
        ss20_excluded_points = int(np.sum(ss20_mask))
        points = points.loc[~ss20_mask].copy()
        all_galaxies = sorted(points["galaxy_key"].astype(str).unique().tolist())

    if points.empty:
        raise RuntimeError("No valid points remain after SS20 filter.")

    points = points.copy()
    points["g_pred"] = pb.compute_rar_prediction(points["g_bar"].to_numpy(dtype=float), g_dagger)
    points["log_resid"] = np.log10(points["g_obs"]) - np.log10(points["g_pred"])
    points["rho_dyn"] = (3.0 * points["g_obs"]) / (4.0 * np.pi * pb.G_NEWTON * (points["r_kpc"] * pb.KPC_TO_M))
    points["log_gbar"] = np.log10(points["g_bar"].to_numpy(dtype=float))
    points = points.replace([np.inf, -np.inf], np.nan).dropna(subset=["g_pred", "log_resid", "rho_dyn", "log_gbar"])

    if points.empty:
        raise RuntimeError("No valid points after residual/rho computation.")

    prep_meta = {
        "input_file": str(input_file),
        "input_csv_sha256": csv_sha,
        "mapping": mapping,
        "conversion_notes": conversion_notes,
        "filter_counts": filter_counts,
        "include_ss20": bool(include_ss20),
        "ss20_excluded_points": int(ss20_excluded_points),
    }
    return points, all_galaxies, prep_meta


def _select_groups_exact(
    points: pd.DataFrame,
    all_galaxies: List[str],
    g_dagger: float,
    r_inner_max_kpc: float,
    min_points_per_galaxy: int,
    top_n: int,
    shrink_if_needed: bool,
) -> Dict[str, Any]:
    summary = pb.summarize_galaxies(
        points=points,
        all_galaxies=all_galaxies,
        g_dagger=g_dagger,
        r_inner_max_kpc=r_inner_max_kpc,
        min_points=min_points_per_galaxy,
    )
    eligible = summary[
        (~summary["flag_low_points"]) & np.isfinite(summary["rho_score"]) & np.isfinite(summary["median_log_resid"])
    ].copy()
    eligible = eligible.sort_values("rho_score", ascending=False).reset_index(drop=True)

    top_n_eff = int(top_n)
    shrink_reason = None
    if shrink_if_needed and len(eligible) < (2 * top_n_eff):
        top_n_eff = int(len(eligible) // 2)
        shrink_reason = f"eligible<{2*top_n}; using top_n_eff={top_n_eff}"

    top = eligible.head(top_n_eff).copy()
    bottom = (
        eligible.loc[~eligible["galaxy_key"].isin(top["galaxy_key"])]
        .sort_values("rho_score", ascending=True)
        .head(top_n_eff)
        .copy()
    )

    return {
        "summary": summary,
        "eligible": eligible,
        "top": top,
        "bottom": bottom,
        "top_n_eff": int(top_n_eff),
        "shrink_reason": shrink_reason,
    }


def _resid_arrays_by_galaxy(points: pd.DataFrame, galaxy_keys: Sequence[str]) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    keys: List[str] = []
    vals: List[np.ndarray] = []
    wts: List[np.ndarray] = []
    for k in galaxy_keys:
        arr = points.loc[points["galaxy_key"] == k, "log_resid"].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        keys.append(str(k))
        vals.append(arr)
        wts.append(np.full(arr.shape, 1.0 / float(arr.size), dtype=float))
    return keys, vals, wts


def _shift_from_group_arrays(
    vals_top: Sequence[np.ndarray],
    vals_bottom: Sequence[np.ndarray],
    wts_top: Sequence[np.ndarray],
    wts_bottom: Sequence[np.ndarray],
) -> Dict[str, float]:
    if len(vals_top) == 0 or len(vals_bottom) == 0:
        return {
            "median_top_unweighted": np.nan,
            "median_bottom_unweighted": np.nan,
            "shift_unweighted": np.nan,
            "median_top_weighted": np.nan,
            "median_bottom_weighted": np.nan,
            "shift_weighted": np.nan,
        }

    top = np.concatenate(vals_top)
    bottom = np.concatenate(vals_bottom)
    top_w = np.concatenate(wts_top)
    bottom_w = np.concatenate(wts_bottom)

    med_top = _safe_median(top)
    med_bottom = _safe_median(bottom)
    wmed_top = _weighted_median(top, top_w)
    wmed_bottom = _weighted_median(bottom, bottom_w)

    return {
        "median_top_unweighted": med_top,
        "median_bottom_unweighted": med_bottom,
        "shift_unweighted": float(med_top - med_bottom) if np.isfinite(med_top) and np.isfinite(med_bottom) else np.nan,
        "median_top_weighted": wmed_top,
        "median_bottom_weighted": wmed_bottom,
        "shift_weighted": float(wmed_top - wmed_bottom)
        if np.isfinite(wmed_top) and np.isfinite(wmed_bottom)
        else np.nan,
    }


def _perm_pvalue(obs: float, null_arr: np.ndarray) -> float:
    n = int(len(null_arr))
    if not np.isfinite(obs) or n == 0:
        return np.nan
    good = null_arr[np.isfinite(null_arr)]
    if good.size == 0:
        return np.nan
    extreme = int(np.sum(np.abs(good) >= abs(obs)))
    return float((extreme + 1) / (good.size + 1))


def run_c1(
    points: pd.DataFrame,
    top_keys: Sequence[str],
    bottom_keys: Sequence[str],
    n_perm: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    top_keys = [str(k) for k in top_keys]
    bottom_keys = [str(k) for k in bottom_keys]
    union_keys = top_keys + [k for k in bottom_keys if k not in set(top_keys)]

    keys, vals, wts = _resid_arrays_by_galaxy(points, union_keys)
    idx_map = {k: i for i, k in enumerate(keys)}
    top_idx = [idx_map[k] for k in top_keys if k in idx_map]
    bottom_idx = [idx_map[k] for k in bottom_keys if k in idx_map]

    obs = _shift_from_group_arrays(
        vals_top=[vals[i] for i in top_idx],
        vals_bottom=[vals[i] for i in bottom_idx],
        wts_top=[wts[i] for i in top_idx],
        wts_bottom=[wts[i] for i in bottom_idx],
    )

    n_top = len(top_idx)
    n_bottom = len(bottom_idx)
    all_idx = np.array(top_idx + bottom_idx, dtype=int)
    all_idx = np.unique(all_idx)

    null_unweighted = np.full(n_perm, np.nan, dtype=float)
    null_weighted = np.full(n_perm, np.nan, dtype=float)

    if n_top > 0 and n_bottom > 0 and len(all_idx) >= (n_top + n_bottom):
        for i in range(n_perm):
            perm = rng.permutation(all_idx)
            pt = perm[:n_top]
            pbm = perm[n_top : n_top + n_bottom]
            st = _shift_from_group_arrays(
                vals_top=[vals[j] for j in pt],
                vals_bottom=[vals[j] for j in pbm],
                wts_top=[wts[j] for j in pt],
                wts_bottom=[wts[j] for j in pbm],
            )
            null_unweighted[i] = st["shift_unweighted"]
            null_weighted[i] = st["shift_weighted"]

    p_unweighted = _perm_pvalue(obs["shift_unweighted"], null_unweighted)
    p_weighted = _perm_pvalue(obs["shift_weighted"], null_weighted)

    out = {
        "n_top_gal": int(n_top),
        "n_bottom_gal": int(n_bottom),
        "n_perm": int(n_perm),
        "observed": obs,
        "difference_weighted_minus_unweighted": float(obs["shift_weighted"] - obs["shift_unweighted"])
        if np.isfinite(obs["shift_weighted"]) and np.isfinite(obs["shift_unweighted"])
        else np.nan,
        "p_unweighted_block": p_unweighted,
        "p_weighted_block": p_weighted,
        "null_unweighted": null_unweighted,
        "null_weighted": null_weighted,
    }
    return out


def _galaxy_block_bootstrap_shift(
    top_bin: pd.DataFrame,
    bottom_bin: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    top_groups = [
        grp["log_resid"].to_numpy(dtype=float)
        for _, grp in top_bin.groupby("galaxy_key", sort=False)
        if len(grp) > 0
    ]
    bottom_groups = [
        grp["log_resid"].to_numpy(dtype=float)
        for _, grp in bottom_bin.groupby("galaxy_key", sort=False)
        if len(grp) > 0
    ]
    if len(top_groups) == 0 or len(bottom_groups) == 0:
        return np.nan, np.nan

    draws = np.full(n_boot, np.nan, dtype=float)
    nt = len(top_groups)
    nb = len(bottom_groups)
    for i in range(n_boot):
        it = rng.integers(0, nt, size=nt)
        ib = rng.integers(0, nb, size=nb)
        top_arr = np.concatenate([top_groups[j] for j in it])
        bot_arr = np.concatenate([bottom_groups[j] for j in ib])
        draws[i] = _safe_median(top_arr) - _safe_median(bot_arr)
    good = draws[np.isfinite(draws)]
    if good.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(good, [2.5, 97.5])
    return float(lo), float(hi)


def run_c2(
    points: pd.DataFrame,
    top_keys: Sequence[str],
    bottom_keys: Sequence[str],
    n_bins: int,
    min_bin_group_points: int,
    n_boot_bin: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    top = points[points["galaxy_key"].isin(set(top_keys))].copy()
    bottom = points[points["galaxy_key"].isin(set(bottom_keys))].copy()

    top = top[np.isfinite(top["log_gbar"]) & np.isfinite(top["log_resid"])].copy()
    bottom = bottom[np.isfinite(bottom["log_gbar"]) & np.isfinite(bottom["log_resid"])].copy()

    all_log_gbar = np.concatenate(
        [top["log_gbar"].to_numpy(dtype=float), bottom["log_gbar"].to_numpy(dtype=float)]
    )
    all_log_gbar = all_log_gbar[np.isfinite(all_log_gbar)]

    if all_log_gbar.size < 2:
        return {
            "bin_table": pd.DataFrame(),
            "aggregate_equal": np.nan,
            "aggregate_matched": np.nan,
            "n_bins_valid": 0,
            "insufficient_reason": "insufficient_log_gbar_points",
        }

    lo = float(np.min(all_log_gbar))
    hi = float(np.max(all_log_gbar))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {
            "bin_table": pd.DataFrame(),
            "aggregate_equal": np.nan,
            "aggregate_matched": np.nan,
            "n_bins_valid": 0,
            "insufficient_reason": "invalid_log_gbar_range",
        }

    edges = np.linspace(lo, hi, int(n_bins) + 1)
    recs: List[Dict[str, Any]] = []

    for i in range(len(edges) - 1):
        e0 = float(edges[i])
        e1 = float(edges[i + 1])
        if i == len(edges) - 2:
            mt = (top["log_gbar"] >= e0) & (top["log_gbar"] <= e1)
            mb = (bottom["log_gbar"] >= e0) & (bottom["log_gbar"] <= e1)
        else:
            mt = (top["log_gbar"] >= e0) & (top["log_gbar"] < e1)
            mb = (bottom["log_gbar"] >= e0) & (bottom["log_gbar"] < e1)

        top_bin = top.loc[mt]
        bottom_bin = bottom.loc[mb]
        n_top = int(len(top_bin))
        n_bottom = int(len(bottom_bin))
        sufficient = bool((n_top >= min_bin_group_points) and (n_bottom >= min_bin_group_points))

        if sufficient:
            shift = _safe_median(top_bin["log_resid"].to_numpy(dtype=float)) - _safe_median(
                bottom_bin["log_resid"].to_numpy(dtype=float)
            )
            ci_lo, ci_hi = _galaxy_block_bootstrap_shift(
                top_bin=top_bin,
                bottom_bin=bottom_bin,
                n_boot=n_boot_bin,
                rng=rng,
            )
        else:
            shift = np.nan
            ci_lo, ci_hi = np.nan, np.nan

        recs.append(
            {
                "bin_index": int(i),
                "log_gbar_lo": e0,
                "log_gbar_hi": e1,
                "log_gbar_center": 0.5 * (e0 + e1),
                "n_top": n_top,
                "n_bottom": n_bottom,
                "sufficient": sufficient,
                "shift_median_top_minus_bottom": shift,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )

    bdf = pd.DataFrame.from_records(recs)
    valid = bdf[bdf["sufficient"] & np.isfinite(bdf["shift_median_top_minus_bottom"])].copy()

    if valid.empty:
        agg_eq = np.nan
        agg_match = np.nan
    else:
        agg_eq = float(valid["shift_median_top_minus_bottom"].mean())
        w = np.minimum(valid["n_top"].to_numpy(dtype=float), valid["n_bottom"].to_numpy(dtype=float))
        if np.sum(w) > 0:
            w = w / np.sum(w)
            agg_match = float(np.sum(w * valid["shift_median_top_minus_bottom"].to_numpy(dtype=float)))
        else:
            agg_match = np.nan

    return {
        "bin_table": bdf,
        "aggregate_equal": agg_eq,
        "aggregate_matched": agg_match,
        "n_bins_valid": int(len(valid)),
        "insufficient_reason": None,
    }


def _save_c1_figure(out_path: Path, c1: Dict[str, Any], title_suffix: str) -> None:
    null_u = np.asarray(c1["null_unweighted"], dtype=float)
    null_w = np.asarray(c1["null_weighted"], dtype=float)
    obs_u = float(c1["observed"]["shift_unweighted"])
    obs_w = float(c1["observed"]["shift_weighted"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for ax, arr, obs, lab in [
        (axes[0], null_u, obs_u, "unweighted"),
        (axes[1], null_w, obs_w, "galaxy-weighted"),
    ]:
        good = arr[np.isfinite(arr)]
        if good.size > 0:
            bins = np.linspace(np.min(good), np.max(good), 60)
            ax.hist(good, bins=bins, density=True, alpha=0.7, color="#4e79a7")
        ax.axvline(obs, color="crimson", lw=2.0, label="observed")
        ax.axvline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xlabel("median shift (top-bottom) [dex]")
        ax.set_title(f"C1 block-null: {lab}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes[0].set_ylabel("density")
    stitle = "C1 Galaxy-Block Permutation Null"
    if title_suffix:
        stitle += f" - {title_suffix}"
    fig.suptitle(stitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_c2_figure(out_path: Path, c2: Dict[str, Any], title_suffix: str) -> None:
    bdf = c2["bin_table"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.1]})

    if bdf is None or len(bdf) == 0:
        ax1.text(0.5, 0.5, "No valid C2 bins", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No occupancy data", ha="center", va="center", transform=ax2.transAxes)
    else:
        centers = bdf["log_gbar_center"].to_numpy(dtype=float)
        shifts = bdf["shift_median_top_minus_bottom"].to_numpy(dtype=float)
        ci_lo = bdf["ci_lo"].to_numpy(dtype=float)
        ci_hi = bdf["ci_hi"].to_numpy(dtype=float)
        good = np.isfinite(shifts)
        yerr_low = shifts - ci_lo
        yerr_high = ci_hi - shifts
        yerr = np.vstack([np.where(np.isfinite(yerr_low), yerr_low, np.nan), np.where(np.isfinite(yerr_high), yerr_high, np.nan)])

        ax1.axhline(0.0, color="black", lw=1.0, ls="--")
        if np.any(good):
            ax1.errorbar(
                centers[good],
                shifts[good],
                yerr=yerr[:, good],
                fmt="o-",
                lw=1.6,
                ms=4.5,
                color="#e15759",
                capsize=2.5,
                label="bin-wise shift",
            )
        bad = ~good
        if np.any(bad):
            ax1.plot(centers[bad], np.zeros(np.sum(bad)), "x", color="gray", label="insufficient bins")
        ax1.set_ylabel("median shift [dex]")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best")

        width = (float(np.nanmedian(np.diff(centers))) if len(centers) > 1 else 0.1) * 0.4
        ax2.bar(centers - width / 2.0, bdf["n_top"].to_numpy(dtype=float), width=width, alpha=0.7, label="top", color="#59a14f")
        ax2.bar(centers + width / 2.0, bdf["n_bottom"].to_numpy(dtype=float), width=width, alpha=0.7, label="bottom", color="#4e79a7")
        ax2.set_ylabel("points/bin")
        ax2.set_xlabel("log10(g_bar) [m s^-2]")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="best")

    title = "C2 Within-bin Shift vs log_gbar"
    if title_suffix:
        title += f" - {title_suffix}"
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_c3_figure(out_path: Path, c3_rows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    if c3_rows.empty:
        ax.text(0.5, 0.5, "No source subsets evaluated", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    rows = c3_rows.copy()
    rows["x"] = np.arange(len(rows), dtype=float)

    colors = []
    for v in rows["replication_verdict"].astype(str):
        if v == "consistent direction":
            colors.append("#59a14f")
        elif v.startswith("inconsistent"):
            colors.append("#e15759")
        else:
            colors.append("#9c9c9c")

    ax.axhline(0.0, color="black", lw=1.0, ls="--")
    ax.bar(rows["x"], rows["c1_shift_weighted"], color=colors, alpha=0.8)
    for _, r in rows.iterrows():
        ptxt = "nan" if not np.isfinite(r["c1_p_weighted"]) else f"{r['c1_p_weighted']:.3g}"
        ax.text(float(r["x"]), float(r["c1_shift_weighted"]), f" p={ptxt}", fontsize=8, rotation=90, va="bottom")

    ax.set_xticks(rows["x"])
    ax.set_xticklabels(rows["subset"], rotation=20, ha="right")
    ax.set_ylabel("C1 weighted shift [dex]")
    ax.set_title("C3 Source-Stratified Replication")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _decision_label(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    c3_rows: pd.DataFrame,
) -> str:
    c1_shift_u = float(c1["observed"]["shift_unweighted"])
    c1_shift_w = float(c1["observed"]["shift_weighted"])
    c1_p_w = float(c1["p_weighted_block"]) if np.isfinite(c1["p_weighted_block"]) else np.nan
    c2_eq = float(c2["aggregate_equal"]) if np.isfinite(c2["aggregate_equal"]) else np.nan
    c2_match = float(c2["aggregate_matched"]) if np.isfinite(c2["aggregate_matched"]) else np.nan

    collapse = False
    if np.isfinite(c1_shift_u) and abs(c1_shift_u) > 0:
        ratio = abs(c1_shift_w) / abs(c1_shift_u) if np.isfinite(c1_shift_w) else 0.0
        collapse = ratio < 0.35

    within_bin_small = (np.isfinite(c2_eq) and np.isfinite(c2_match) and abs(c2_eq) < 0.005 and abs(c2_match) < 0.005)

    inconsistent = False
    if not c3_rows.empty:
        inconsistent = bool(np.any(c3_rows["replication_verdict"].astype(str).str.startswith("inconsistent")))

    if inconsistent:
        return "source-specific"
    if (not np.isfinite(c1_p_w) or c1_p_w >= 0.01) or collapse or within_bin_small:
        return "sampling artifact"
    return "candidate physical effect"


def _source_subset_definitions() -> List[Tuple[str, List[str]]]:
    return [
        ("SPARC-only", ["SPARC"]),
        ("GHASP-only", ["GHASP"]),
        ("deBlok2002-only", ["deBlok2002"]),
        ("WALLABY-only", ["WALLABY"]),
        ("WALLABY+WALLABY_DR2", ["WALLABY", "WALLABY_DR2"]),
    ]


def _run_subset_c3(
    subset_name: str,
    source_values: List[str],
    points_all: pd.DataFrame,
    g_dagger: float,
    r_inner_max_kpc: float,
    min_points_per_galaxy: int,
    top_n: int,
    n_perm: int,
    n_bins: int,
    min_bin_group_points: int,
    n_boot_bin: int,
    rng: np.random.Generator,
    reference_sign: float,
) -> Dict[str, Any]:
    subset_points = points_all[points_all["dataset"].astype(str).isin(source_values)].copy()
    n_pts = int(len(subset_points))
    n_gal = int(subset_points["galaxy_key"].nunique()) if n_pts > 0 else 0

    out: Dict[str, Any] = {
        "subset": subset_name,
        "sources": source_values,
        "n_points": n_pts,
        "n_galaxies": n_gal,
        "status": "ok",
        "reason": None,
    }

    if n_pts == 0 or n_gal == 0:
        out["status"] = "skipped"
        out["reason"] = "no_points"
        out["replication_verdict"] = "inconclusive (N too small)"
        return out

    all_gals = sorted(subset_points["galaxy_key"].astype(str).unique().tolist())
    sel = _select_groups_exact(
        points=subset_points,
        all_galaxies=all_gals,
        g_dagger=g_dagger,
        r_inner_max_kpc=r_inner_max_kpc,
        min_points_per_galaxy=min_points_per_galaxy,
        top_n=top_n,
        shrink_if_needed=True,
    )
    top = sel["top"]
    bottom = sel["bottom"]
    top_n_eff = int(sel["top_n_eff"])

    if top_n_eff < SUBSET_MIN_TOP_N or len(top) < SUBSET_MIN_TOP_N or len(bottom) < SUBSET_MIN_TOP_N:
        out["status"] = "skipped"
        out["reason"] = (
            f"insufficient_eligible_for_split: top_n_eff={top_n_eff}, "
            f"n_top={len(top)}, n_bottom={len(bottom)}"
        )
        out["replication_verdict"] = "inconclusive (N too small)"
        return out

    c1 = run_c1(
        points=subset_points,
        top_keys=top["galaxy_key"].astype(str).tolist(),
        bottom_keys=bottom["galaxy_key"].astype(str).tolist(),
        n_perm=n_perm,
        rng=rng,
    )

    c2 = run_c2(
        points=subset_points,
        top_keys=top["galaxy_key"].astype(str).tolist(),
        bottom_keys=bottom["galaxy_key"].astype(str).tolist(),
        n_bins=n_bins,
        min_bin_group_points=min_bin_group_points,
        n_boot_bin=n_boot_bin,
        rng=rng,
    )

    shift_w = float(c1["observed"]["shift_weighted"])
    p_w = float(c1["p_weighted_block"]) if np.isfinite(c1["p_weighted_block"]) else np.nan
    c2_match = float(c2["aggregate_matched"]) if np.isfinite(c2["aggregate_matched"]) else np.nan

    if not np.isfinite(shift_w):
        verdict = "inconclusive (N too small)"
    elif np.isfinite(p_w) and p_w > 0.1:
        verdict = "inconclusive (N too small)"
    elif np.sign(shift_w) == np.sign(reference_sign) and (not np.isfinite(c2_match) or np.sign(c2_match) == np.sign(reference_sign)):
        verdict = "consistent direction"
    else:
        verdict = "inconsistent / likely source-specific systematic"

    out.update(
        {
            "top_n_eff": top_n_eff,
            "n_top_gal": int(len(top)),
            "n_bottom_gal": int(len(bottom)),
            "c1_shift_unweighted": float(c1["observed"]["shift_unweighted"]),
            "c1_shift_weighted": shift_w,
            "c1_p_unweighted": float(c1["p_unweighted_block"]),
            "c1_p_weighted": p_w,
            "c2_aggregate_equal": float(c2["aggregate_equal"]),
            "c2_aggregate_matched": c2_match,
            "c2_n_bins_valid": int(c2["n_bins_valid"]),
            "replication_verdict": verdict,
            "shrink_reason": sel["shrink_reason"],
        }
    )
    return out


def _write_report(
    out_path: Path,
    summary: Dict[str, Any],
    c2_bins: pd.DataFrame,
    c3_df: pd.DataFrame,
) -> None:
    c1 = summary["C1"]
    c2 = summary["C2"]
    final_decision = summary["final_decision"]
    meta = summary["metadata"]

    lines: List[str] = []
    lines.append("# Branch C (C1-C3) Report")
    lines.append("")
    lines.append("## 1) Executive Summary")
    lines.append(
        f"- Decision: **{final_decision}**"
    )
    lines.append(
        f"- C1 weighted pooled shift: {c1['observed']['shift_weighted']:.6f} dex, "
        f"galaxy-block p={c1['p_weighted_block']:.6g}"
    )
    lines.append(
        f"- C2 aggregate shifts: equal-bin={c2['aggregate_equal']:.6f} dex, "
        f"matched-bin={c2['aggregate_matched']:.6f} dex"
    )
    lines.append("")

    lines.append("## 2) C1 Galaxy-Correct Pooled Tests")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Unweighted pooled shift (dex) | {c1['observed']['shift_unweighted']:.6f} |")
    lines.append(f"| Galaxy-weighted pooled shift (dex) | {c1['observed']['shift_weighted']:.6f} |")
    lines.append(f"| Weighted - unweighted (dex) | {c1['difference_weighted_minus_unweighted']:.6f} |")
    lines.append(f"| Galaxy-block p (unweighted) | {c1['p_unweighted_block']:.6g} |")
    lines.append(f"| Galaxy-block p (weighted) | {c1['p_weighted_block']:.6g} |")
    lines.append("")
    lines.append("Figure: `figures/C1_block_nulls.png`")
    lines.append("")

    lines.append("## 3) C2 gbar-Matched Shift")
    lines.append(
        f"- Valid bins: {int(c2['n_bins_valid'])} / {int(meta['parameters']['n_bins'])} "
        f"(minimum per-group bin count={int(meta['parameters']['min_bin_group_points'])})"
    )
    lines.append(f"- Aggregate C2A (equal-bin): {c2['aggregate_equal']:.6f} dex")
    lines.append(f"- Aggregate C2B (matched-bin): {c2['aggregate_matched']:.6f} dex")
    lines.append("")
    lines.append("Bin-wise results (sufficient bins):")
    lines.append("| bin | log_gbar_center | n_top | n_bottom | shift_dex | ci_lo | ci_hi |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    if c2_bins is not None and len(c2_bins) > 0:
        for _, r in c2_bins.iterrows():
            if bool(r.get("sufficient", False)):
                lines.append(
                    f"| {int(r['bin_index'])} | {r['log_gbar_center']:.4f} | {int(r['n_top'])} | {int(r['n_bottom'])} | "
                    f"{r['shift_median_top_minus_bottom']:.6f} | {r['ci_lo']:.6f} | {r['ci_hi']:.6f} |"
                )
    lines.append("")
    lines.append("Figure: `figures/C2_binwise_shift.png`")
    lines.append("")

    lines.append("## 4) C3 Source-Stratified Replication")
    lines.append("| subset | N_gal | N_pts | weighted_shift | p_weighted | C2_matched | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    if c3_df is not None and len(c3_df) > 0:
        for _, r in c3_df.iterrows():
            w = r.get("c1_shift_weighted", np.nan)
            p = r.get("c1_p_weighted", np.nan)
            m = r.get("c2_aggregate_matched", np.nan)
            lines.append(
                f"| {r['subset']} | {int(r.get('n_galaxies', 0))} | {int(r.get('n_points', 0))} | "
                f"{(f'{w:.6f}' if np.isfinite(w) else 'nan')} | "
                f"{(f'{p:.4g}' if np.isfinite(p) else 'nan')} | "
                f"{(f'{m:.6f}' if np.isfinite(m) else 'nan')} | {r.get('replication_verdict', 'n/a')} |"
            )
    lines.append("")
    lines.append("Figure: `figures/C3_source_replication.png`")
    lines.append("")

    lines.append("## 5) Final Decision")
    lines.append(f"- **{final_decision}**")
    lines.append("")

    lines.append("## 6) Stamps")
    lines.append(f"- CSV SHA256: `{meta['input_csv_sha256']}`")
    lines.append(f"- Git HEAD: `{meta['git_head']}`")
    lines.append(f"- Seed: `{meta['seed']}`")
    lines.append(f"- Parameters: `{json.dumps(meta['parameters'], sort_keys=True)}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_metrics_records(records: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    records.append(row)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Branch C C1-C3 decisive experiments for Paper3 density bridge.")
    p.add_argument("--rar_points_file", type=str, default=DEFAULT_RAR_POINTS_FILE)
    p.add_argument("--g_dagger", type=float, default=pb.DEFAULT_G_DAGGER)
    p.add_argument("--r_inner_max_kpc", type=float, default=DEFAULT_R_INNER_MAX_KPC)
    p.add_argument("--min_points_per_galaxy", type=int, default=DEFAULT_MIN_POINTS_PER_GALAXY)
    p.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--n_perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--n_perm_fallback", type=int, default=DEFAULT_N_PERM_FALLBACK)
    p.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS)
    p.add_argument("--min_bin_group_points", type=int, default=DEFAULT_MIN_BIN_GROUP_POINTS)
    p.add_argument("--n_boot_bin", type=int, default=DEFAULT_N_BOOT_BIN)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--assume_r_kpc", action="store_true")
    p.add_argument("--include_ss20", action="store_true")
    p.add_argument("--require_csv_sha", type=str, default=DEFAULT_REQUIRE_CSV_SHA)
    p.add_argument(
        "--runtime_limit_sec",
        type=float,
        default=900.0,
        help="Soft runtime budget for deciding 10k vs 5k permutations.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    t_start = time.time()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = _make_out_dir(repo_root=repo_root, out_dir_arg=args.out_dir)
    figs_dir = out_dir / "figures"

    rng = np.random.default_rng(int(args.seed))

    points, all_galaxies, prep_meta = _prepare_points(
        repo_root=repo_root,
        rar_points_file=args.rar_points_file,
        g_dagger=float(args.g_dagger),
        assume_r_kpc=bool(args.assume_r_kpc),
        include_ss20=bool(args.include_ss20),
        require_csv_sha=args.require_csv_sha,
    )

    # Top/bottom split follows Paper3 ranking logic exactly (rho_score descending for top, ascending for bottom).
    sel_main = _select_groups_exact(
        points=points,
        all_galaxies=all_galaxies,
        g_dagger=float(args.g_dagger),
        r_inner_max_kpc=float(args.r_inner_max_kpc),
        min_points_per_galaxy=int(args.min_points_per_galaxy),
        top_n=int(args.top_n),
        shrink_if_needed=False,
    )
    top_main = sel_main["top"]
    bottom_main = sel_main["bottom"]

    if len(top_main) == 0 or len(bottom_main) == 0:
        raise RuntimeError("Unable to form top/bottom groups with current filters.")

    # Runtime-aware permutation count: attempt 10k by default, fallback to 5k if near runtime budget.
    elapsed = time.time() - t_start
    n_perm_eff = int(args.n_perm)
    runtime_note = "used_default_n_perm"
    if elapsed > float(args.runtime_limit_sec) * 0.2 and int(args.n_perm) > int(args.n_perm_fallback):
        n_perm_eff = int(args.n_perm_fallback)
        runtime_note = "fallback_to_n_perm_fallback_due_to_elapsed"

    c1_main = run_c1(
        points=points,
        top_keys=top_main["galaxy_key"].astype(str).tolist(),
        bottom_keys=bottom_main["galaxy_key"].astype(str).tolist(),
        n_perm=n_perm_eff,
        rng=rng,
    )

    c2_main = run_c2(
        points=points,
        top_keys=top_main["galaxy_key"].astype(str).tolist(),
        bottom_keys=bottom_main["galaxy_key"].astype(str).tolist(),
        n_bins=int(args.n_bins),
        min_bin_group_points=int(args.min_bin_group_points),
        n_boot_bin=int(args.n_boot_bin),
        rng=rng,
    )

    # C3 source-stratified reruns.
    ref_sign = np.sign(float(c1_main["observed"]["shift_weighted"]))
    c3_rows: List[Dict[str, Any]] = []
    for subset_name, source_vals in _source_subset_definitions():
        c3_rows.append(
            _run_subset_c3(
                subset_name=subset_name,
                source_values=source_vals,
                points_all=points,
                g_dagger=float(args.g_dagger),
                r_inner_max_kpc=float(args.r_inner_max_kpc),
                min_points_per_galaxy=int(args.min_points_per_galaxy),
                top_n=int(args.top_n),
                n_perm=n_perm_eff,
                n_bins=int(args.n_bins),
                min_bin_group_points=int(args.min_bin_group_points),
                n_boot_bin=int(args.n_boot_bin),
                rng=rng,
                reference_sign=ref_sign if ref_sign != 0 else 1.0,
            )
        )

    c3_df = pd.DataFrame.from_records(c3_rows)

    # Figures
    c1_fig = figs_dir / "C1_block_nulls.png"
    c2_fig = figs_dir / "C2_binwise_shift.png"
    c3_fig = figs_dir / "C3_source_replication.png"
    _save_c1_figure(c1_fig, c1_main, title_suffix="full sample")
    _save_c2_figure(c2_fig, c2_main, title_suffix="full sample")
    _save_c3_figure(c3_fig, c3_df)

    # Metrics parquet (tidy long form)
    metric_rows: List[Dict[str, Any]] = []

    for k, v in c1_main["observed"].items():
        _append_metrics_records(
            metric_rows,
            {
                "experiment": "C1",
                "subset": "full",
                "kind": "observed",
                "metric": k,
                "value": float(v) if np.isfinite(v) else np.nan,
                "bin_index": np.nan,
                "log_gbar_center": np.nan,
                "permutation_index": np.nan,
                "status": "ok",
            },
        )
    for k in ["difference_weighted_minus_unweighted", "p_unweighted_block", "p_weighted_block", "n_perm"]:
        v = c1_main[k]
        _append_metrics_records(
            metric_rows,
            {
                "experiment": "C1",
                "subset": "full",
                "kind": "summary",
                "metric": k,
                "value": float(v) if np.isfinite(v) else np.nan,
                "bin_index": np.nan,
                "log_gbar_center": np.nan,
                "permutation_index": np.nan,
                "status": "ok",
            },
        )

    for i, v in enumerate(np.asarray(c1_main["null_unweighted"], dtype=float)):
        _append_metrics_records(
            metric_rows,
            {
                "experiment": "C1",
                "subset": "full",
                "kind": "null",
                "metric": "null_unweighted_shift",
                "value": float(v) if np.isfinite(v) else np.nan,
                "bin_index": np.nan,
                "log_gbar_center": np.nan,
                "permutation_index": int(i),
                "status": "ok",
            },
        )
    for i, v in enumerate(np.asarray(c1_main["null_weighted"], dtype=float)):
        _append_metrics_records(
            metric_rows,
            {
                "experiment": "C1",
                "subset": "full",
                "kind": "null",
                "metric": "null_weighted_shift",
                "value": float(v) if np.isfinite(v) else np.nan,
                "bin_index": np.nan,
                "log_gbar_center": np.nan,
                "permutation_index": int(i),
                "status": "ok",
            },
        )

    c2_bins = c2_main["bin_table"] if isinstance(c2_main.get("bin_table"), pd.DataFrame) else pd.DataFrame()
    if len(c2_bins) > 0:
        for _, r in c2_bins.iterrows():
            _append_metrics_records(
                metric_rows,
                {
                    "experiment": "C2",
                    "subset": "full",
                    "kind": "bin",
                    "metric": "shift_median_top_minus_bottom",
                    "value": float(r["shift_median_top_minus_bottom"]) if np.isfinite(r["shift_median_top_minus_bottom"]) else np.nan,
                    "bin_index": int(r["bin_index"]),
                    "log_gbar_center": float(r["log_gbar_center"]),
                    "permutation_index": np.nan,
                    "status": "ok" if bool(r["sufficient"]) else "insufficient",
                },
            )
            for m in ["n_top", "n_bottom", "ci_lo", "ci_hi"]:
                _append_metrics_records(
                    metric_rows,
                    {
                        "experiment": "C2",
                        "subset": "full",
                        "kind": "bin_aux",
                        "metric": m,
                        "value": float(r[m]) if np.isfinite(r[m]) else np.nan,
                        "bin_index": int(r["bin_index"]),
                        "log_gbar_center": float(r["log_gbar_center"]),
                        "permutation_index": np.nan,
                        "status": "ok" if bool(r["sufficient"]) else "insufficient",
                    },
                )

    for m in ["aggregate_equal", "aggregate_matched", "n_bins_valid"]:
        v = c2_main[m]
        _append_metrics_records(
            metric_rows,
            {
                "experiment": "C2",
                "subset": "full",
                "kind": "aggregate",
                "metric": m,
                "value": float(v) if np.isfinite(v) else np.nan,
                "bin_index": np.nan,
                "log_gbar_center": np.nan,
                "permutation_index": np.nan,
                "status": "ok",
            },
        )

    if len(c3_df) > 0:
        for _, r in c3_df.iterrows():
            subset = str(r["subset"])
            for m in [
                "n_points",
                "n_galaxies",
                "top_n_eff",
                "n_top_gal",
                "n_bottom_gal",
                "c1_shift_unweighted",
                "c1_shift_weighted",
                "c1_p_unweighted",
                "c1_p_weighted",
                "c2_aggregate_equal",
                "c2_aggregate_matched",
                "c2_n_bins_valid",
            ]:
                if m not in r:
                    continue
                val = r[m]
                _append_metrics_records(
                    metric_rows,
                    {
                        "experiment": "C3",
                        "subset": subset,
                        "kind": "subset_summary",
                        "metric": m,
                        "value": float(val) if val is not None and np.isfinite(val) else np.nan,
                        "bin_index": np.nan,
                        "log_gbar_center": np.nan,
                        "permutation_index": np.nan,
                        "status": str(r.get("status", "ok")),
                    },
                )

    metrics_df = pd.DataFrame.from_records(metric_rows)
    metrics_path = out_dir / "metrics_C1C2C3.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    # Final decision and summary JSON.
    final_decision = _decision_label(c1=c1_main, c2=c2_main, c3_rows=c3_df)

    assumptions = [
        "Requested WORKDIR (/Users/russelllicht/bh-singularity) did not contain Paper3 files; proceeded in /Users/russelllicht/bec-dark-matter where requested inputs/code exist.",
        "Top/bottom ranking uses exact Paper3 rho_score eligibility/ranking logic from summarize_galaxies + rank split.",
        "C1 block permutation shuffles galaxy labels across the combined selected top+bottom galaxy set while preserving observed group sizes.",
        "C2 uses 15 equal-width bins in log_gbar over the combined top/bottom range; bins require >=30 points/group.",
        "C2 uncertainty uses galaxy-block bootstrap (95% CI) per bin.",
        "C3 subset top_n is shrunk only when eligible galaxies are insufficient to form equal-sized top/bottom groups.",
    ]

    summary: Dict[str, Any] = {
        "metadata": {
            "timestamp_utc": utc_now().isoformat().replace("+00:00", "Z"),
            "repo_root": str(repo_root),
            "output_dir": str(out_dir),
            "git_head": pb.git_head_sha(repo_root),
            "seed": int(args.seed),
            "input_file": prep_meta["input_file"],
            "input_csv_sha256": prep_meta["input_csv_sha256"],
            "require_csv_sha": args.require_csv_sha,
            "assumptions": assumptions,
            "runtime_note": runtime_note,
            "parameters": {
                "g_dagger": float(args.g_dagger),
                "r_inner_max_kpc": float(args.r_inner_max_kpc),
                "min_points_per_galaxy": int(args.min_points_per_galaxy),
                "top_n": int(args.top_n),
                "n_perm_requested": int(args.n_perm),
                "n_perm_used": int(n_perm_eff),
                "n_perm_fallback": int(args.n_perm_fallback),
                "n_bins": int(args.n_bins),
                "min_bin_group_points": int(args.min_bin_group_points),
                "n_boot_bin": int(args.n_boot_bin),
                "include_ss20": bool(args.include_ss20),
            },
            "prep": prep_meta,
        },
        "C1": {
            "observed": c1_main["observed"],
            "difference_weighted_minus_unweighted": c1_main["difference_weighted_minus_unweighted"],
            "p_unweighted_block": c1_main["p_unweighted_block"],
            "p_weighted_block": c1_main["p_weighted_block"],
            "n_perm": c1_main["n_perm"],
            "n_top_gal": c1_main["n_top_gal"],
            "n_bottom_gal": c1_main["n_bottom_gal"],
            "null_unweighted": np.asarray(c1_main["null_unweighted"], dtype=float).tolist(),
            "null_weighted": np.asarray(c1_main["null_weighted"], dtype=float).tolist(),
        },
        "C2": {
            "aggregate_equal": float(c2_main["aggregate_equal"]),
            "aggregate_matched": float(c2_main["aggregate_matched"]),
            "n_bins_valid": int(c2_main["n_bins_valid"]),
            "insufficient_reason": c2_main.get("insufficient_reason"),
            "bins": c2_bins.to_dict(orient="records") if len(c2_bins) > 0 else [],
        },
        "C3": {
            "subsets": c3_rows,
        },
    }
    summary["final_decision"] = final_decision

    summary_path = out_dir / "summary_C1C2C3.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")

    report_path = out_dir / "report_C1C2C3.md"
    _write_report(report_path, summary=summary, c2_bins=c2_bins, c3_df=c3_df)

    logs_path = out_dir / "logs.txt"
    logs_lines = [
        f"timestamp_utc={summary['metadata']['timestamp_utc']}",
        f"repo_root={repo_root}",
        f"git_head={summary['metadata']['git_head']}",
        f"input_file={prep_meta['input_file']}",
        f"input_sha256={prep_meta['input_csv_sha256']}",
        f"runtime_note={runtime_note}",
        "assumptions:",
    ]
    logs_lines.extend([f"- {a}" for a in assumptions])
    logs_lines.append("output_files:")
    for p in [summary_path, metrics_path, report_path, logs_path, c1_fig, c2_fig, c3_fig]:
        logs_lines.append(f"- {p}")
    logs_path.write_text("\n".join(logs_lines) + "\n", encoding="utf-8")

    # Console summary
    print(f"[BRANCHC] output_dir={out_dir}")
    print(
        "[BRANCHC] C1 full: "
        f"unweighted={c1_main['observed']['shift_unweighted']:.6f} dex, "
        f"weighted={c1_main['observed']['shift_weighted']:.6f} dex, "
        f"p_unweighted={c1_main['p_unweighted_block']:.6g}, "
        f"p_weighted={c1_main['p_weighted_block']:.6g}"
    )
    print(
        "[BRANCHC] C2 full: "
        f"agg_equal={c2_main['aggregate_equal']:.6f} dex, "
        f"agg_matched={c2_main['aggregate_matched']:.6f} dex, "
        f"valid_bins={int(c2_main['n_bins_valid'])}"
    )
    print(f"[BRANCHC] C3 subsets evaluated={len(c3_rows)}")
    print(f"[BRANCHC] final_decision={final_decision}")


if __name__ == "__main__":
    main()
