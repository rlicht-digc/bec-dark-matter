#!/usr/bin/env python3
"""Strict apples-to-apples TNG↔SPARC feature reproduction driver."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import kurtosis

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from analysis.pipeline.analysis_tools import LOG_G_DAGGER, g_dagger, find_zero_crossings, numerical_derivative
from analysis.tng.adapters import load_sparc_points, load_tng_points
from analysis.tng.matching import (
    MatchConfig,
    balance_table,
    build_matched_points,
    nearest_neighbor_match,
    sample_pairwise_k_points,
    summarize_galaxies,
)


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        return out
    except Exception:
        return "UNKNOWN"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _fit_nonparametric_mean(log_gbar: np.ndarray, log_gobs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Spline on binned means, shared residual definition for both datasets."""
    x = np.asarray(log_gbar, dtype=float)
    y = np.asarray(log_gobs, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    n_support = int(np.clip(len(x) // 8, 24, 120))
    edges = np.linspace(float(x.min()), float(x.max()), n_support + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    for i in range(len(centers)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        c = int(np.sum(m))
        counts[i] = c
        if c >= 3:
            means[i] = float(np.mean(y[m]))

    m_valid = np.isfinite(means)
    xc = centers[m_valid]
    yc = means[m_valid]
    if len(xc) < 4:
        coef = np.polyfit(x, y, deg=1)

        def pred_fn(x_new: np.ndarray) -> np.ndarray:
            return np.polyval(coef, np.asarray(x_new, dtype=float))

        return x, y, pred_fn

    k = min(3, len(xc) - 1)
    smooth = max(1e-6, len(xc) * float(np.var(yc)) * 0.02)
    spl = UnivariateSpline(xc, yc, s=smooth, k=k)

    x_lo = float(xc.min())
    x_hi = float(xc.max())

    def pred_fn(x_new: np.ndarray) -> np.ndarray:
        xx = np.asarray(x_new, dtype=float)
        xx = np.clip(xx, x_lo, x_hi)
        return spl(xx)

    return xc, yc, pred_fn


def add_residuals(points: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x = points["log_gbar"].to_numpy(dtype=float)
    y = points["log_gobs"].to_numpy(dtype=float)
    x_support, y_support, pred_fn = _fit_nonparametric_mean(x, y)
    pred = pred_fn(x)
    out = points.copy()
    out["mean_log_gobs_pred"] = pred
    out["log_resid"] = y - pred
    meta = {
        "n_support": int(len(x_support)),
        "support_x_min": float(np.min(x_support)) if len(x_support) else None,
        "support_x_max": float(np.max(x_support)) if len(x_support) else None,
    }
    return out, meta


def _common_log_gbar_range(sparc: pd.DataFrame, tng: pd.DataFrame) -> Tuple[float, float]:
    s = sparc["log_gbar"].to_numpy(dtype=float)
    t = tng["log_gbar"].to_numpy(dtype=float)
    s_lo, s_hi = np.percentile(s, [1.0, 99.0])
    t_lo, t_hi = np.percentile(t, [1.0, 99.0])
    lo = max(float(s_lo), float(t_lo))
    hi = min(float(s_hi), float(t_hi))
    if hi - lo < 0.8:
        lo = max(float(np.min(s)), float(np.min(t)))
        hi = min(float(np.max(s)), float(np.max(t)))
    if hi <= lo:
        raise ValueError("No overlapping log_gbar range between SPARC and TNG matched samples.")
    return lo, hi


def _fixed_edges(gmin: float, gmax: float, n_bins: int) -> np.ndarray:
    return np.linspace(gmin, gmax, int(n_bins) + 1)


def _offset_edges(gmin: float, gmax: float, n_bins: int, offset_idx: int, n_offsets: int) -> np.ndarray:
    width = (gmax - gmin) / float(n_bins)
    shift = (float(offset_idx) / float(n_offsets)) * width
    return np.arange(gmin + shift, gmax + 1e-12, width)


def _curve_from_edges(points_resid: pd.DataFrame, edges: np.ndarray, min_bin_points: int) -> Dict[str, Any]:
    x = points_resid["log_gbar"].to_numpy(dtype=float)
    r = points_resid["log_resid"].to_numpy(dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma = np.full(len(centers), np.nan, dtype=float)
    k4 = np.full(len(centers), np.nan, dtype=float)
    counts = np.zeros(len(centers), dtype=int)

    for i in range(len(centers)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        n = int(np.sum(m))
        counts[i] = n
        if n >= min_bin_points:
            rr = r[m]
            sigma[i] = float(np.std(rr))
            if n >= max(8, min_bin_points):
                k4[i] = float(kurtosis(rr, fisher=True, bias=False))

    inv_loc = None
    d_sigma = np.full(len(centers), np.nan, dtype=float)
    v = np.isfinite(sigma)
    if np.sum(v) >= 4:
        c_v = centers[v]
        s_v = sigma[v]
        d_v = numerical_derivative(c_v, s_v)
        d_sigma[v] = d_v
        crossings = find_zero_crossings(c_v, d_v, direction="pos_to_neg")
        if crossings:
            inv_loc = float(crossings[int(np.argmin(np.abs(np.asarray(crossings) - LOG_G_DAGGER)))])

    peak_loc = None
    peak_val = None
    k_valid = np.isfinite(k4)
    if np.any(k_valid):
        idx_peak = int(np.nanargmax(k4))
        peak_loc = float(centers[idx_peak])
        peak_val = float(k4[idx_peak])

    kurt_at_gdagger = None
    if np.any(k_valid):
        idx_g = int(np.argmin(np.abs(centers - LOG_G_DAGGER)))
        if np.isfinite(k4[idx_g]):
            kurt_at_gdagger = float(k4[idx_g])

    return {
        "centers": centers,
        "sigma": sigma,
        "kurtosis": k4,
        "counts": counts,
        "d_sigma": d_sigma,
        "inversion_log_g": inv_loc,
        "inversion_delta_from_gdagger": (float(abs(inv_loc - LOG_G_DAGGER)) if inv_loc is not None else None),
        "kurtosis_peak_log_gbar": peak_loc,
        "kurtosis_peak_value": peak_val,
        "kurtosis_peak_delta_from_gdagger": (float(abs(peak_loc - LOG_G_DAGGER)) if peak_loc is not None else None),
        "kurtosis_at_gdagger": kurt_at_gdagger,
    }


def compute_features(
    points_resid: pd.DataFrame,
    gmin: float,
    gmax: float,
    n_bins: int,
    n_offsets: int,
    min_bin_points: int,
) -> Dict[str, Any]:
    edges = _fixed_edges(gmin, gmax, n_bins)
    base = _curve_from_edges(points_resid, edges, min_bin_points=min_bin_points)

    inv_offsets: List[float] = []
    for j in range(int(n_offsets)):
        e = _offset_edges(gmin, gmax, n_bins, j, int(n_offsets))
        if len(e) < 5:
            continue
        c = _curve_from_edges(points_resid, e, min_bin_points=min_bin_points)
        if c["inversion_log_g"] is not None:
            inv_offsets.append(float(c["inversion_log_g"]))

    out = dict(base)
    out["edges"] = edges
    out["inversion_offsets"] = inv_offsets
    out["inversion_offsets_near_gdagger_0p10"] = int(np.sum(np.abs(np.asarray(inv_offsets, dtype=float) - LOG_G_DAGGER) <= 0.10)) if inv_offsets else 0
    out["inversion_offsets_near_gdagger_0p05"] = int(np.sum(np.abs(np.asarray(inv_offsets, dtype=float) - LOG_G_DAGGER) <= 0.05)) if inv_offsets else 0
    out["inversion_offsets_count"] = int(len(inv_offsets))
    return out


def _ci_from_values(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "ci95": [None, None]}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
    }


def bootstrap_dataset(
    points: pd.DataFrame,
    gmin: float,
    gmax: float,
    n_bins: int,
    n_offsets: int,
    min_bin_points: int,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    grouped = {k: g.copy() for k, g in points.groupby("galaxy_key", sort=False)}
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)

    inv_vals: List[float] = []
    peak_vals: List[float] = []
    kurt_gd_vals: List[float] = []
    kurt_curves: List[np.ndarray] = []

    for _ in range(int(n_bootstrap)):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        boot_df = pd.concat([grouped[k] for k in sample_keys], ignore_index=True)
        boot_resid, _ = add_residuals(boot_df)
        feat = compute_features(
            boot_resid,
            gmin=gmin,
            gmax=gmax,
            n_bins=n_bins,
            n_offsets=n_offsets,
            min_bin_points=min_bin_points,
        )
        if feat["inversion_log_g"] is not None:
            inv_vals.append(float(feat["inversion_log_g"]))
        if feat["kurtosis_peak_log_gbar"] is not None:
            peak_vals.append(float(feat["kurtosis_peak_log_gbar"]))
        if feat["kurtosis_at_gdagger"] is not None:
            kurt_gd_vals.append(float(feat["kurtosis_at_gdagger"]))
        kurt_curves.append(np.asarray(feat["kurtosis"], dtype=float))

    kstack = np.vstack(kurt_curves) if kurt_curves else np.empty((0, n_bins), dtype=float)
    if kstack.size:
        with np.errstate(invalid="ignore"):
            k_lo = np.nanpercentile(kstack, 2.5, axis=0)
            k_hi = np.nanpercentile(kstack, 97.5, axis=0)
    else:
        k_lo = np.full(n_bins, np.nan)
        k_hi = np.full(n_bins, np.nan)

    return {
        "inversion_samples": inv_vals,
        "kurtosis_peak_samples": peak_vals,
        "kurtosis_at_gdagger_samples": kurt_gd_vals,
        "inversion_ci": _ci_from_values(inv_vals),
        "kurtosis_peak_ci": _ci_from_values(peak_vals),
        "kurtosis_at_gdagger_ci": _ci_from_values(kurt_gd_vals),
        "kurtosis_curve_ci_lo": k_lo.tolist(),
        "kurtosis_curve_ci_hi": k_hi.tolist(),
    }


def split_half_replication(
    points: pd.DataFrame,
    gmin: float,
    gmax: float,
    n_bins: int,
    n_offsets: int,
    min_bin_points: int,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    keys = sorted(points["galaxy_key"].astype(str).unique().tolist())
    if len(keys) < 6:
        return {"status": "insufficient_galaxies", "n_galaxies": int(len(keys))}

    rng = np.random.default_rng(seed)
    half = len(keys) // 2
    inv_a: List[float] = []
    inv_b: List[float] = []
    peak_a: List[float] = []
    peak_b: List[float] = []
    both_near = 0
    both_peak_near = 0

    for _ in range(int(n_splits)):
        perm = rng.permutation(keys)
        a_keys = set(perm[:half].tolist())
        b_keys = set(perm[half:].tolist())
        a = points.loc[points["galaxy_key"].astype(str).isin(a_keys)].copy()
        b = points.loc[points["galaxy_key"].astype(str).isin(b_keys)].copy()
        a_resid, _ = add_residuals(a)
        b_resid, _ = add_residuals(b)
        fa = compute_features(a_resid, gmin, gmax, n_bins, n_offsets, min_bin_points)
        fb = compute_features(b_resid, gmin, gmax, n_bins, n_offsets, min_bin_points)
        ia = fa["inversion_log_g"]
        ib = fb["inversion_log_g"]
        pa = fa["kurtosis_peak_log_gbar"]
        pb = fb["kurtosis_peak_log_gbar"]
        if ia is not None:
            inv_a.append(float(ia))
        if ib is not None:
            inv_b.append(float(ib))
        if pa is not None:
            peak_a.append(float(pa))
        if pb is not None:
            peak_b.append(float(pb))
        if ia is not None and ib is not None and abs(ia - LOG_G_DAGGER) <= 0.10 and abs(ib - LOG_G_DAGGER) <= 0.10:
            both_near += 1
        if pa is not None and pb is not None and abs(pa - LOG_G_DAGGER) <= 0.10 and abs(pb - LOG_G_DAGGER) <= 0.10:
            both_peak_near += 1

    return {
        "status": "ok",
        "n_splits": int(n_splits),
        "inversion_halfA_ci": _ci_from_values(inv_a),
        "inversion_halfB_ci": _ci_from_values(inv_b),
        "kurtosis_peak_halfA_ci": _ci_from_values(peak_a),
        "kurtosis_peak_halfB_ci": _ci_from_values(peak_b),
        "both_halves_inversion_near_gdagger_0p10": int(both_near),
        "both_halves_kurtosis_peak_near_gdagger_0p10": int(both_peak_near),
    }


def shuffle_null_inversion(
    points_resid: pd.DataFrame,
    edges: np.ndarray,
    min_bin_points: int,
    n_shuffles: int,
    seed: int,
) -> Dict[str, Any]:
    """Galaxy-block residual shuffle null for inversion proximity."""
    keys = sorted(points_resid["galaxy_key"].astype(str).unique().tolist())
    grouped = {k: g.copy() for k, g in points_resid.groupby("galaxy_key", sort=False)}
    res_map = {k: grouped[k]["log_resid"].to_numpy(dtype=float) for k in grouped}
    idx_map = {k: grouped[k].index.to_numpy() for k in grouped}
    rng = np.random.default_rng(seed)
    inv_samples: List[float] = []

    for _ in range(int(n_shuffles)):
        perm = rng.permutation(keys)
        out_resid = np.full(len(points_resid), np.nan, dtype=float)
        for src_key, dst_key in zip(keys, perm):
            idx = idx_map[src_key]
            src_res = res_map[dst_key]
            if len(src_res) == len(idx):
                vals = rng.permutation(src_res)
            else:
                vals = rng.choice(src_res, size=len(idx), replace=True)
            out_resid[idx] = vals
        shuf = points_resid.copy()
        shuf["log_resid"] = out_resid
        curve = _curve_from_edges(shuf, edges=edges, min_bin_points=min_bin_points)
        if curve["inversion_log_g"] is not None:
            inv_samples.append(float(curve["inversion_log_g"]))

    arr = np.asarray(inv_samples, dtype=float)
    p10 = float(np.mean(np.abs(arr - LOG_G_DAGGER) <= 0.10)) if arr.size else None
    p05 = float(np.mean(np.abs(arr - LOG_G_DAGGER) <= 0.05)) if arr.size else None
    return {
        "n_shuffles": int(n_shuffles),
        "n_valid": int(arr.size),
        "inversion_log_g_samples": inv_samples,
        "p_within_0p10_dex": p10,
        "p_within_0p05_dex": p05,
    }


def _plot_inversion_curve(fig_path: Path, sparc_feat: Dict[str, Any], tng_feat: Dict[str, Any]) -> None:
    plt.figure(figsize=(8.5, 5.5))
    for feat, color, label in [
        (sparc_feat, "#1f77b4", "SPARC"),
        (tng_feat, "#d62728", "TNG matched"),
    ]:
        c = np.asarray(feat["centers"], dtype=float)
        s = np.asarray(feat["sigma"], dtype=float)
        m = np.isfinite(c) & np.isfinite(s)
        if np.any(m):
            plt.plot(c[m], s[m], marker="o", color=color, linewidth=2, label=label)
        inv = feat.get("inversion_log_g")
        if inv is not None:
            plt.axvline(float(inv), color=color, linestyle="--", alpha=0.55)
    plt.axvline(LOG_G_DAGGER, color="black", linestyle=":", linewidth=1.5, label="log g†")
    plt.xlabel("log10(g_bar) [m s^-2]")
    plt.ylabel("σ(log residual)")
    plt.title("Scatter-Derivative Inversion Curves (same pipeline)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()


def _plot_kurtosis_curve(
    fig_path: Path,
    sparc_feat: Dict[str, Any],
    tng_feat: Dict[str, Any],
    sparc_boot: Dict[str, Any],
    tng_boot: Dict[str, Any],
) -> None:
    plt.figure(figsize=(8.5, 5.5))
    for feat, boot, color, label in [
        (sparc_feat, sparc_boot, "#1f77b4", "SPARC"),
        (tng_feat, tng_boot, "#d62728", "TNG matched"),
    ]:
        c = np.asarray(feat["centers"], dtype=float)
        k4 = np.asarray(feat["kurtosis"], dtype=float)
        lo = np.asarray(boot.get("kurtosis_curve_ci_lo", []), dtype=float)
        hi = np.asarray(boot.get("kurtosis_curve_ci_hi", []), dtype=float)
        m = np.isfinite(c) & np.isfinite(k4)
        if np.any(m):
            plt.plot(c[m], k4[m], marker="o", color=color, linewidth=2, label=label)
            if lo.size == k4.size and hi.size == k4.size:
                m2 = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(c)
                if np.any(m2):
                    plt.fill_between(c[m2], lo[m2], hi[m2], color=color, alpha=0.15)
    plt.axvline(LOG_G_DAGGER, color="black", linestyle=":", linewidth=1.5, label="log g†")
    plt.xlabel("log10(g_bar) [m s^-2]")
    plt.ylabel("κ4 (excess kurtosis of residuals)")
    plt.title("Kurtosis Spike Curves (same pipeline)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()


def _plot_matching_diagnostics(fig_path: Path, s_sum: pd.DataFrame, t_sum: pd.DataFrame, bal: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = [
        ("log_gbar_med", "(a) median log g_bar"),
        ("log_gbar_iqr", "(b) IQR log g_bar"),
        ("R_med", "(c) median R_kpc"),
    ]
    for ax, (metric, title) in zip(axes.flat[:3], metrics):
        s = s_sum[metric].to_numpy(dtype=float)
        t = t_sum[metric].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        t = t[np.isfinite(t)]
        bins = 18
        ax.hist(s, bins=bins, alpha=0.55, color="#1f77b4", label="SPARC")
        ax.hist(t, bins=bins, alpha=0.55, color="#d62728", label="TNG")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    ax = axes.flat[3]
    if not bal.empty:
        y = np.arange(len(bal))
        ax.barh(y + 0.15, bal["smd_all"].to_numpy(dtype=float), height=0.28, color="#999999", label="SMD pre-match")
        ax.barh(y - 0.15, bal["smd_matched"].to_numpy(dtype=float), height=0.28, color="#2ca02c", label="SMD matched")
        ax.set_yticks(y)
        ax.set_yticklabels(bal["metric"].tolist(), fontsize=8)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title("(d) Balance diagnostics")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def _contains(ci: Dict[str, Any], value: float | None) -> bool:
    if value is None:
        return False
    lo, hi = ci.get("ci95", [None, None])
    if lo is None or hi is None:
        return False
    return float(lo) <= float(value) <= float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict TNG↔SPARC feature reproduction (inversion + kurtosis).")
    parser.add_argument("--tng-input", required=True, help="Path to TNG points table (parquet/csv).")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--n-shuffles", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--bin-offsets", type=int, default=5)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sparc-input", default=None, help="Optional override for SPARC unified CSV path.")
    parser.add_argument("--caliper-gbar-med", type=float, default=0.35)
    parser.add_argument("--caliper-gbar-iqr", type=float, default=0.30)
    parser.add_argument("--caliper-R-med", type=float, default=6.0)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on matched pairs.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    n_bootstrap = int(args.n_bootstrap) if args.n_bootstrap is not None else (500 if args.mode == "smoke" else 10000)
    n_shuffles = int(args.n_shuffles) if args.n_shuffles is not None else (200 if args.mode == "smoke" else 1000)
    n_splits = 80 if args.mode == "smoke" else 300
    min_bin_points = 8 if args.mode == "smoke" else 12
    max_pairs = args.max_pairs if args.max_pairs is not None else (24 if args.mode == "smoke" else 0)

    if args.outdir:
        run_dir = Path(args.outdir).resolve()
    else:
        run_dir = repo_root / "analysis" / "results" / f"tng_sparc_feature_repro_{args.mode}_{_now_tag()}"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sparc = load_sparc_points(str(repo_root), source_filter="SPARC", sparc_input_path=args.sparc_input)
    tng = load_tng_points(args.tng_input)

    s_sum_all = summarize_galaxies(sparc)
    t_sum_all = summarize_galaxies(tng)
    match_cfg = MatchConfig(
        caliper_gbar_med=float(args.caliper_gbar_med),
        caliper_gbar_iqr=float(args.caliper_gbar_iqr),
        caliper_r_med=float(args.caliper_R_med),
        min_pair_points=5,
    )
    match_table, unmatched = nearest_neighbor_match(s_sum_all, t_sum_all, match_cfg)
    if max_pairs and len(match_table) > int(max_pairs):
        match_table = match_table.sort_values("distance").head(int(max_pairs)).reset_index(drop=True)

    if match_table.empty:
        raise RuntimeError("No SPARC↔TNG matches survived calipers.")

    sparc_m, tng_m = build_matched_points(sparc, tng, match_table)
    sparc_k, tng_k, pair_table = sample_pairwise_k_points(
        sparc_m,
        tng_m,
        match_table,
        k=int(args.K),
        seed=int(args.seed),
        min_pair_points=match_cfg.min_pair_points,
    )
    if sparc_k.empty or tng_k.empty:
        raise RuntimeError("Pairwise K sampling produced zero points. Relax K/calipers.")

    s_sum = summarize_galaxies(sparc_k)
    t_sum = summarize_galaxies(tng_k)
    bal = balance_table(s_sum_all, t_sum_all, match_table)

    sparc_resid, sparc_resid_meta = add_residuals(sparc_k)
    tng_resid, tng_resid_meta = add_residuals(tng_k)
    gmin, gmax = _common_log_gbar_range(sparc_resid, tng_resid)

    sparc_feat = compute_features(
        sparc_resid,
        gmin=gmin,
        gmax=gmax,
        n_bins=int(args.n_bins),
        n_offsets=int(args.bin_offsets),
        min_bin_points=min_bin_points,
    )
    tng_feat = compute_features(
        tng_resid,
        gmin=gmin,
        gmax=gmax,
        n_bins=int(args.n_bins),
        n_offsets=int(args.bin_offsets),
        min_bin_points=min_bin_points,
    )

    sparc_boot = bootstrap_dataset(
        sparc_k,
        gmin=gmin,
        gmax=gmax,
        n_bins=int(args.n_bins),
        n_offsets=int(args.bin_offsets),
        min_bin_points=min_bin_points,
        n_bootstrap=n_bootstrap,
        seed=int(args.seed) + 11,
    )
    tng_boot = bootstrap_dataset(
        tng_k,
        gmin=gmin,
        gmax=gmax,
        n_bins=int(args.n_bins),
        n_offsets=int(args.bin_offsets),
        min_bin_points=min_bin_points,
        n_bootstrap=n_bootstrap,
        seed=int(args.seed) + 37,
    )

    split_half = {
        "sparc": split_half_replication(
            sparc_k, gmin, gmax, int(args.n_bins), int(args.bin_offsets), min_bin_points, n_splits, int(args.seed) + 101
        ),
        "tng": split_half_replication(
            tng_k, gmin, gmax, int(args.n_bins), int(args.bin_offsets), min_bin_points, n_splits, int(args.seed) + 151
        ),
    }

    edges = np.asarray(sparc_feat["edges"], dtype=float)
    shuf = {
        "sparc": shuffle_null_inversion(
            sparc_resid,
            edges=edges,
            min_bin_points=min_bin_points,
            n_shuffles=n_shuffles,
            seed=int(args.seed) + 211,
        ),
        "tng": shuffle_null_inversion(
            tng_resid,
            edges=edges,
            min_bin_points=min_bin_points,
            n_shuffles=n_shuffles,
            seed=int(args.seed) + 241,
        ),
    }

    inv_tng = tng_feat.get("inversion_log_g")
    peak_tng = tng_feat.get("kurtosis_peak_log_gbar")
    inv_in_ci = _contains(sparc_boot["inversion_ci"], inv_tng)
    peak_in_ci = _contains(sparc_boot["kurtosis_peak_ci"], peak_tng)
    inv_near = inv_tng is not None and abs(float(inv_tng) - LOG_G_DAGGER) <= 0.10
    peak_near = peak_tng is not None and abs(float(peak_tng) - LOG_G_DAGGER) <= 0.10
    reproduces = bool(inv_in_ci and peak_in_ci and inv_near and peak_near)

    matched_samples = pd.concat(
        [sparc_k.assign(dataset="SPARC"), tng_k.assign(dataset="TNG")],
        ignore_index=True,
    )
    matched_samples.to_parquet(run_dir / "matched_samples.parquet", index=False)

    _plot_inversion_curve(fig_dir / "inversion_curve.png", sparc_feat, tng_feat)
    _plot_kurtosis_curve(fig_dir / "kurtosis_curve.png", sparc_feat, tng_feat, sparc_boot, tng_boot)
    _plot_matching_diagnostics(fig_dir / "matching_diagnostics.png", s_sum, t_sum, bal)

    sparc_path = Path(args.sparc_input).resolve() if args.sparc_input else repo_root / "analysis" / "results" / "rar_points_unified.csv"
    tng_path = Path(args.tng_input).resolve()
    run_stamp = {
        "timestamp_utc": _now_iso_utc(),
        "git_head": _git_head(repo_root),
        "repo_root": str(repo_root),
        "mode": args.mode,
        "output_dir": str(run_dir),
        "input_paths": {
            "sparc_unified_csv": str(sparc_path),
            "tng_input": str(tng_path),
        },
        "input_sha256": {
            "sparc_unified_csv": _sha256_file(sparc_path),
            "tng_input": _sha256_file(tng_path),
        },
        "params": {
            "seed": int(args.seed),
            "n_bootstrap": int(n_bootstrap),
            "n_shuffles": int(n_shuffles),
            "n_splits": int(n_splits),
            "n_bins": int(args.n_bins),
            "bin_offsets": int(args.bin_offsets),
            "K": int(args.K),
            "max_pairs": int(max_pairs),
            "min_bin_points": int(min_bin_points),
            "calipers": {
                "gbar_med": float(args.caliper_gbar_med),
                "gbar_iqr": float(args.caliper_gbar_iqr),
                "R_med": float(args.caliper_R_med),
            },
        },
    }
    (run_dir / "run_stamp.json").write_text(json.dumps(_jsonable(run_stamp), indent=2))

    summary = {
        "test_name": "tng_sparc_feature_repro_strict",
        "description": "Strict apples-to-apples TNG↔SPARC feature reproduction using matched sampling and identical residual/binning codepath.",
        "g_dagger_log10": float(LOG_G_DAGGER),
        "g_dagger_si": float(g_dagger),
        "mode": args.mode,
        "params": run_stamp["params"],
        "matching": {
            "n_sparc_galaxies_all": int(s_sum_all["galaxy_key"].nunique()),
            "n_tng_galaxies_all": int(t_sum_all["galaxy_key"].nunique()),
            "n_pairs_matched": int(len(match_table)),
            "n_pairs_after_k_sampling": int(len(pair_table)),
            "n_unmatched_sparc": int(len(unmatched)),
            "match_table_head": match_table.head(20).to_dict(orient="records"),
            "unmatched_head": unmatched.head(20).to_dict(orient="records"),
            "balance_table": bal.to_dict(orient="records"),
        },
        "samples": {
            "sparc_matched": {"n_galaxies": int(s_sum["galaxy_key"].nunique()), "n_points": int(len(sparc_k))},
            "tng_matched": {"n_galaxies": int(t_sum["galaxy_key"].nunique()), "n_points": int(len(tng_k))},
            "common_log_gbar_range": [float(gmin), float(gmax)],
        },
        "sparc": {
            "residual_model": sparc_resid_meta,
            "features": _jsonable(sparc_feat),
            "bootstrap": _jsonable(sparc_boot),
        },
        "tng": {
            "residual_model": tng_resid_meta,
            "features": _jsonable(tng_feat),
            "bootstrap": _jsonable(tng_boot),
        },
        "split_half_replication": _jsonable(split_half),
        "shuffle_null": _jsonable(shuf),
        "comparison": {
            "criterion": "reproduces if TNG inversion and kurtosis-peak locations are both inside SPARC bootstrap CI and within ±0.10 dex of log g†",
            "inversion_tng_in_sparc_ci": bool(inv_in_ci),
            "kurtosis_peak_tng_in_sparc_ci": bool(peak_in_ci),
            "inversion_tng_within_0p10_dex_of_gdagger": bool(inv_near),
            "kurtosis_peak_tng_within_0p10_dex_of_gdagger": bool(peak_near),
            "reproduces": bool(reproduces),
            "statement": (
                "TNG reproduces SPARC features under matched sampling and identical pipeline."
                if reproduces
                else "TNG does not reproduce SPARC features under matched sampling and identical pipeline."
            ),
        },
        "artifacts": {
            "run_stamp": str(run_dir / "run_stamp.json"),
            "matched_samples": str(run_dir / "matched_samples.parquet"),
            "inversion_curve": str(fig_dir / "inversion_curve.png"),
            "kurtosis_curve": str(fig_dir / "kurtosis_curve.png"),
            "matching_diagnostics": str(fig_dir / "matching_diagnostics.png"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    report_lines = [
        "# TNG↔SPARC Strict Feature Reproduction Report",
        "",
        f"- Timestamp (UTC): {_now_iso_utc()}",
        f"- Mode: `{args.mode}`",
        f"- g†: log10 = {LOG_G_DAGGER:.4f}",
        "",
        "## Matched Sample",
        f"- Matched pairs (raw): {len(match_table)}",
        f"- Matched pairs after K-sampling: {len(pair_table)}",
        f"- SPARC matched: {len(sparc_k)} points across {s_sum['galaxy_key'].nunique()} galaxies",
        f"- TNG matched: {len(tng_k)} points across {t_sum['galaxy_key'].nunique()} galaxies",
        "",
        "## Feature Results",
        f"- SPARC inversion: {sparc_feat.get('inversion_log_g')}",
        f"- TNG inversion: {tng_feat.get('inversion_log_g')}",
        f"- SPARC kurtosis peak: {sparc_feat.get('kurtosis_peak_log_gbar')}",
        f"- TNG kurtosis peak: {tng_feat.get('kurtosis_peak_log_gbar')}",
        "",
        "## Pass/Fail Rule",
        "- Reproduces only if TNG inversion and kurtosis-peak locations are both inside SPARC bootstrap CI and both within ±0.10 dex of log g†.",
        f"- Verdict: **{'REPRODUCES' if reproduces else 'DOES NOT REPRODUCE'}**",
        "",
        "## Secondary Diagnostics",
        "- Split-half replication: included for SPARC and TNG.",
        "- Shuffle null proximity to g†: included (galaxy-block residual shuffle).",
        "",
        "## Note",
        "- This run uses identical residual definition (non-parametric spline mean relation), identical binning parameters, and identical inversion/kurtosis codepaths for both datasets.",
    ]
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    print(f"RUN_DIR={run_dir}")
    print(f"SPARC inversion={sparc_feat.get('inversion_log_g')}  kurtosis_peak={sparc_feat.get('kurtosis_peak_log_gbar')}")
    print(f"TNG inversion={tng_feat.get('inversion_log_g')}  kurtosis_peak={tng_feat.get('kurtosis_peak_log_gbar')}")
    print(f"REPRODUCES={reproduces}")


if __name__ == "__main__":
    main()
