#!/usr/bin/env python3
"""
r_s vs xi discriminating test for scale-hierarchy outputs.

Runs two analyses on the SPARC structured subset:
1) Matched split test (high vs low) using r_s as splitter and Lc_over_xi as outcome.
   Also reruns xi split for a direct head-to-head comparison.
2) Horse-race OLS model for per-galaxy scatter:
      log_sigma_dm_robust ~ log_xi + log_rs + logMstar + logMh + log_g_ext_proxy
   with baseline/+rs/+xi/+both comparisons, plus 5-fold CV.

Outputs:
  - analysis/results/scale_hierarchy/rs_discriminator_summary.json
  - analysis/results/scale_hierarchy/rs_vs_xi_split_by_caliper.csv
  - analysis/results/scale_hierarchy/rs_split_pairs_primary.parquet
  - analysis/results/scale_hierarchy/xi_split_pairs_primary.parquet
  - analysis/results/scale_hierarchy/horse_race_sigma_dm_robust.csv
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


G_DAGGER = 1.2e-10
H0 = 67.74
H_HUBBLE = H0 / 100.0
RHO_CRIT = 1.27e11 * H_HUBBLE**2  # Msun / Mpc^3

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results", "scale_hierarchy")
RESULTS_TABLE = os.path.join(RESULTS_DIR, "results_scale_hierarchy.parquet")
RAR_POINTS = os.path.join(PROJECT_ROOT, "analysis", "results", "rar_points_unified.csv")


def rar_pred_log(log_gbar: np.ndarray, gdagger: float = G_DAGGER) -> np.ndarray:
    gbar = 10.0 ** np.asarray(log_gbar, dtype=float)
    eps = np.sqrt(np.clip(gbar / gdagger, 1e-300, None))
    denom = 1.0 - np.exp(-eps)
    gobs = gbar / np.clip(denom, 1e-300, None)
    return np.log10(np.clip(gobs, 1e-300, None))


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        if x.size < 2:
            return np.nan
        return float(np.std(x, ddof=1))
    return float(1.4826 * mad)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
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
    return (
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
        float(np.std(vals, ddof=1)),
    )


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


def high_mask_with_tie_fallback(v: np.ndarray, min_group: int = 5) -> Tuple[np.ndarray, float, str]:
    """
    Median split with deterministic tie fallback.
    Primary rule: high = (v >= median), low = (v < median).
    Fallback (if a side is too small): high = (v > median), low = (v <= median).
    """
    x = np.asarray(v, dtype=float)
    med = float(np.median(x))
    mask = x >= med
    n_hi = int(np.sum(mask))
    n_lo = int(mask.size - n_hi)
    rule = "ge_median"
    if min(n_hi, n_lo) < min_group:
        mask = x > med
        n_hi = int(np.sum(mask))
        n_lo = int(mask.size - n_hi)
        rule = "gt_median_tie_fallback"
    return mask, med, rule


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    idx = np.argsort(x)
    ranks = np.empty_like(idx, dtype=float)
    i = 0
    n = x.size
    while i < n:
        j = i
        while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[idx[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = x.size
    if n < 3:
        return np.nan, np.nan
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
    if denom <= 0:
        return np.nan, np.nan
    r = float(np.sum(x0 * y0) / denom)
    # normal approximation for p-value
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(max(n - 3, 1))
    from math import erf, sqrt

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return r, float(p)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan, np.nan
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearsonr_np(rx, ry)


def derive_rs_kpc(mh_msun: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mh = np.asarray(mh_msun, dtype=float)
    log_mh = np.log10(np.clip(mh, 1e-300, None))
    log_c200 = 0.905 - 0.101 * (log_mh - 12.0 + np.log10(H_HUBBLE))
    c200 = 10.0 ** log_c200
    rho_200 = 200.0 * RHO_CRIT
    r200_kpc = (3.0 * mh / (4.0 * np.pi * rho_200)) ** (1.0 / 3.0) * 1000.0
    r_s = r200_kpc / np.clip(c200, 1e-300, None)
    return r_s, c200, r200_kpc


@dataclass
class MatchResult:
    stats: Dict[str, float]
    pairs_df: pd.DataFrame


def match_high_low_by_confounds(
    df: pd.DataFrame,
    outcome_col: str,
    split_col: str,
    confounds: Sequence[str],
    caliper: float,
    seed: int = 42,
) -> MatchResult:
    d = df.copy()
    d = d[np.isfinite(d[outcome_col]) & np.isfinite(d[split_col])].copy()
    if len(d) < 12:
        return MatchResult(stats={"pairs": 0}, pairs_df=pd.DataFrame())

    high_mask, med_split, split_rule = high_mask_with_tie_fallback(d[split_col].to_numpy(dtype=float), min_group=5)
    d["high_group"] = high_mask

    for c in confounds:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(np.nanmedian(d[c]))

    d = zscore_columns(d, confounds)
    zcols = [c + "_z" for c in confounds]
    d = d.dropna(subset=zcols + [outcome_col, "high_group"])
    if len(d) < 12:
        return MatchResult(stats={"pairs": 0}, pairs_df=pd.DataFrame())

    hi = d[d["high_group"]].copy().reset_index(drop=True)
    lo = d[~d["high_group"]].copy().reset_index(drop=True)
    if len(hi) < 5 or len(lo) < 5:
        return MatchResult(stats={"pairs": 0}, pairs_df=pd.DataFrame())

    if len(hi) <= len(lo):
        A, B = hi, lo
        a_is_high = True
    else:
        A, B = lo, hi
        a_is_high = False

    used = np.zeros(len(B), dtype=bool)
    pairs: List[Tuple[int, int]] = []

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
        return MatchResult(stats={"pairs": len(pairs)}, pairs_df=pd.DataFrame())

    A_vals = np.array([A.loc[i, outcome_col] for i, _ in pairs], dtype=float)
    B_vals = np.array([B.loc[j, outcome_col] for _, j in pairs], dtype=float)

    if a_is_high:
        high_vals = A_vals
        low_vals = B_vals
    else:
        high_vals = B_vals
        low_vals = A_vals

    cd = cliffs_delta(high_vals, low_vals)
    ci_lo, ci_hi, cd_sd = bootstrap_cliffs_ci(high_vals, low_vals, n_boot=1000, seed=seed)
    p_perm = median_diff_perm_p(high_vals, low_vals, n_perm=1000, seed=seed)

    pair_rows = []
    for i, j in pairs:
        if a_is_high:
            hi_row = A.loc[i]
            lo_row = B.loc[j]
        else:
            hi_row = B.loc[j]
            lo_row = A.loc[i]
        d_metric = float(hi_row[outcome_col] - lo_row[outcome_col])
        d_gext = np.nan
        if "log_g_ext_proxy" in hi_row.index and "log_g_ext_proxy" in lo_row.index:
            if np.isfinite(hi_row["log_g_ext_proxy"]) and np.isfinite(lo_row["log_g_ext_proxy"]):
                d_gext = float(hi_row["log_g_ext_proxy"] - lo_row["log_g_ext_proxy"])
        pair_rows.append(
            {
                "gal_high": hi_row.get("gal_id", np.nan),
                "gal_low": lo_row.get("gal_id", np.nan),
                "delta_metric_high_minus_low": d_metric,
                "delta_log_gext_high_minus_low": d_gext,
            }
        )
    pairs_df = pd.DataFrame(pair_rows)

    rho_sp, p_sp = spearmanr_np(
        pairs_df["delta_metric_high_minus_low"].to_numpy(dtype=float),
        pairs_df["delta_log_gext_high_minus_low"].to_numpy(dtype=float),
    )
    r_pe, p_pe = pearsonr_np(
        pairs_df["delta_metric_high_minus_low"].to_numpy(dtype=float),
        pairs_df["delta_log_gext_high_minus_low"].to_numpy(dtype=float),
    )

    stats = {
        "split_rule": split_rule,
        "split_median": med_split,
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
        "median_delta_metric_high_minus_low": float(np.median(pairs_df["delta_metric_high_minus_low"])),
        "fraction_delta_lt_0": float(np.mean(pairs_df["delta_metric_high_minus_low"] < 0)),
        "spearman_deltaMetric_vs_deltaLogGext_rho": float(rho_sp) if np.isfinite(rho_sp) else np.nan,
        "spearman_p": float(p_sp) if np.isfinite(p_sp) else np.nan,
        "pearson_r": float(r_pe) if np.isfinite(r_pe) else np.nan,
        "pearson_p": float(p_pe) if np.isfinite(p_pe) else np.nan,
    }
    return MatchResult(stats=stats, pairs_df=pairs_df)


def run_split_suite(
    df: pd.DataFrame,
    outcome_col: str,
    split_col: str,
    confounds: Sequence[str],
    calipers: Sequence[float],
    seed: int = 42,
) -> Dict[str, object]:
    d = df.copy()
    d = d[np.isfinite(d[outcome_col]) & np.isfinite(d[split_col])].copy()
    if len(d) < 12:
        return {"n_sample": int(len(d)), "unmatched": {}, "matched_by_caliper": [], "primary_matched": {}}

    high_mask, med_split, split_rule = high_mask_with_tie_fallback(d[split_col].to_numpy(dtype=float), min_group=5)
    hi = d.loc[high_mask, outcome_col].to_numpy(dtype=float)
    lo = d.loc[~high_mask, outcome_col].to_numpy(dtype=float)
    cd = cliffs_delta(hi, lo)
    ci_lo, ci_hi, cd_sd = bootstrap_cliffs_ci(hi, lo, n_boot=1000, seed=seed)
    p_perm = median_diff_perm_p(hi, lo, n_perm=1000, seed=seed)
    unmatched = {
        "split_rule": split_rule,
        "split_median": med_split,
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

    matched = []
    pair_frames = {}
    for c in calipers:
        mr = match_high_low_by_confounds(
            d, outcome_col=outcome_col, split_col=split_col, confounds=confounds, caliper=float(c), seed=seed
        )
        row = dict(mr.stats)
        row["caliper_z"] = float(c)
        matched.append(row)
        pair_frames[float(c)] = mr.pairs_df

    mdf = pd.DataFrame(matched)
    if "pairs" in mdf.columns and (pd.to_numeric(mdf["pairs"], errors="coerce") > 0).any():
        m2 = mdf[pd.to_numeric(mdf["pairs"], errors="coerce") > 0].copy()
        m2["perm_p_median_diff"] = pd.to_numeric(m2["perm_p_median_diff"], errors="coerce").fillna(1.0)
        m2["abs_cd"] = np.abs(pd.to_numeric(m2["cliffs_delta"], errors="coerce"))
        m2 = m2.sort_values(["pairs", "perm_p_median_diff", "abs_cd"], ascending=[False, True, False])
        primary = m2.iloc[0].to_dict()
        primary_cal = float(primary["caliper_z"])
        primary_pairs = pair_frames.get(primary_cal, pd.DataFrame())
    else:
        primary = {}
        primary_pairs = pd.DataFrame()

    return {
        "n_sample": int(len(d)),
        "outcome": outcome_col,
        "split_col": split_col,
        "confounds": list(confounds),
        "unmatched": unmatched,
        "matched_by_caliper": matched,
        "primary_matched": primary,
        "primary_pairs_df": primary_pairs,
    }


def compute_sigma_dm_robust(dm_threshold: float = -10.5, min_pts: int = 10) -> pd.DataFrame:
    pts = pd.read_csv(RAR_POINTS, usecols=["galaxy", "source", "log_gbar", "log_gobs"])
    pts = pts[pts["source"] == "SPARC"].copy()
    pts["log_rar_pred"] = rar_pred_log(pts["log_gbar"].to_numpy(dtype=float))
    pts["resid"] = pts["log_gobs"] - pts["log_rar_pred"]
    pts = pts[np.isfinite(pts["resid"])].copy()
    dm = pts[pts["log_gbar"] < dm_threshold].copy()
    agg = (
        dm.groupby("galaxy")["resid"]
        .agg(n_dm="size", sigma_dm_robust=robust_sigma, sigma_dm_std="std")
        .reset_index()
    )
    agg = agg[agg["n_dm"] >= min_pts].copy()
    agg["log_sigma_dm_robust"] = np.log10(np.clip(agg["sigma_dm_robust"], 1e-300, None))
    return agg


def fit_ols(y: np.ndarray, X: pd.DataFrame) -> Dict[str, object]:
    y = np.asarray(y, dtype=float)
    Xv = np.asarray(X, dtype=float)
    n = y.size
    p = Xv.shape[1]
    Xd = np.column_stack([np.ones(n), Xv])
    beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    resid = y - yhat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    k = p + 1
    if n > k and rss > 0:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    else:
        adj_r2 = np.nan
    if rss > 0 and n > 0:
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
    else:
        aic = np.nan
        bic = np.nan
    rmse = float(np.sqrt(np.mean(resid**2))) if n > 0 else np.nan
    return {
        "beta": beta,
        "rss": rss,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
        "aic": float(aic) if np.isfinite(aic) else np.nan,
        "bic": float(bic) if np.isfinite(bic) else np.nan,
        "rmse": rmse,
    }


def cross_val_metrics(y: np.ndarray, X: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    n = y.size
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    y_pred = np.full(n, np.nan, dtype=float)

    for test_idx in folds:
        if test_idx.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        if train_idx.size < X.shape[1] + 2:
            continue
        fit = fit_ols(y[train_idx], X.iloc[train_idx])
        beta = fit["beta"]
        Xt = np.asarray(X.iloc[test_idx], dtype=float)
        Xd_t = np.column_stack([np.ones(test_idx.size), Xt])
        y_pred[test_idx] = Xd_t @ beta

    ok = np.isfinite(y_pred)
    if np.sum(ok) < 3:
        return {"cv_r2": np.nan, "cv_rmse": np.nan, "cv_n": int(np.sum(ok))}
    sse = float(np.sum((y[ok] - y_pred[ok]) ** 2))
    tss = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
    cv_r2 = 1.0 - sse / tss if tss > 0 else np.nan
    cv_rmse = float(np.sqrt(np.mean((y[ok] - y_pred[ok]) ** 2)))
    return {"cv_r2": float(cv_r2) if np.isfinite(cv_r2) else np.nan, "cv_rmse": cv_rmse, "cv_n": int(np.sum(ok))}


def bootstrap_coef(
    df: pd.DataFrame,
    y_col: str,
    predictors: Sequence[str],
    coef_name: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    d = df.dropna(subset=[y_col] + list(predictors)).copy()
    n = len(d)
    if n < max(12, len(predictors) + 3):
        return {"coef": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "p_two_sided_sign": np.nan}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = d.iloc[rng.integers(0, n, size=n)].copy()
        fit = fit_ols(s[y_col].to_numpy(dtype=float), s[list(predictors)])
        names = ["intercept"] + list(predictors)
        b = dict(zip(names, fit["beta"]))
        vals.append(float(b.get(coef_name, np.nan)))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return {"coef": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "p_two_sided_sign": np.nan}
    p_sign = 2.0 * min(np.mean(vals <= 0), np.mean(vals >= 0))
    return {
        "coef": float(np.median(vals)),
        "ci95_lo": float(np.percentile(vals, 2.5)),
        "ci95_hi": float(np.percentile(vals, 97.5)),
        "p_two_sided_sign": float(p_sign),
    }


def run_horse_race(df: pd.DataFrame) -> Dict[str, object]:
    required = ["log_sigma_dm_robust", "logMstar", "logMh", "log_g_ext_proxy", "log_rs", "log_xi"]
    d = df.dropna(subset=required).copy()
    if len(d) < 20:
        return {"n_model_sample": int(len(d)), "model_rows": [], "coef_checks": {}, "corr_log_xi_log_rs": np.nan}

    y = d["log_sigma_dm_robust"].to_numpy(dtype=float)
    models = {
        "baseline_mass_gext": ["logMstar", "logMh", "log_g_ext_proxy"],
        "plus_rs": ["logMstar", "logMh", "log_g_ext_proxy", "log_rs"],
        "plus_xi": ["logMstar", "logMh", "log_g_ext_proxy", "log_xi"],
        "plus_both": ["logMstar", "logMh", "log_g_ext_proxy", "log_rs", "log_xi"],
    }

    rows = []
    for name, predictors in models.items():
        X = d[predictors]
        fit = fit_ols(y, X)
        cv = cross_val_metrics(y, X, n_splits=5, seed=42)
        row = {
            "model": name,
            "predictors": ",".join(predictors),
            "n": int(len(d)),
            "k_params": int(len(predictors) + 1),
            "r2": fit["r2"],
            "adj_r2": fit["adj_r2"],
            "aic": fit["aic"],
            "bic": fit["bic"],
            "rmse": fit["rmse"],
            "cv_r2": cv["cv_r2"],
            "cv_rmse": cv["cv_rmse"],
            "cv_n": cv["cv_n"],
        }
        names = ["intercept"] + predictors
        for ncoef, b in zip(names, fit["beta"]):
            row[f"beta_{ncoef}"] = float(b)
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("aic").reset_index(drop=True)

    coef_checks = {
        "plus_both_log_xi": bootstrap_coef(d, "log_sigma_dm_robust", models["plus_both"], "log_xi"),
        "plus_both_log_rs": bootstrap_coef(d, "log_sigma_dm_robust", models["plus_both"], "log_rs"),
        "plus_xi_log_xi": bootstrap_coef(d, "log_sigma_dm_robust", models["plus_xi"], "log_xi"),
        "plus_rs_log_rs": bootstrap_coef(d, "log_sigma_dm_robust", models["plus_rs"], "log_rs"),
    }

    corr, corr_p = pearsonr_np(d["log_xi"].to_numpy(dtype=float), d["log_rs"].to_numpy(dtype=float))

    return {
        "n_model_sample": int(len(d)),
        "model_rows": out.to_dict(orient="records"),
        "coef_checks": coef_checks,
        "corr_log_xi_log_rs": float(corr) if np.isfinite(corr) else np.nan,
        "corr_log_xi_log_rs_p": float(corr_p) if np.isfinite(corr_p) else np.nan,
    }


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gal = pd.read_parquet(RESULTS_TABLE)
    d = gal[(gal["source"] == "SPARC") & (gal["has_structure_metrics"] == True)].copy()
    d = d[np.isfinite(d["Lc_over_xi"]) & np.isfinite(d["xi"]) & np.isfinite(d["Mh"])].copy()

    rs_kpc, c200, r200 = derive_rs_kpc(d["Mh"].to_numpy(dtype=float))
    d["r_s_kpc"] = rs_kpc
    d["c200_dm14"] = c200
    d["R200_kpc_dm14"] = r200
    d["log_rs"] = np.log10(np.clip(d["r_s_kpc"], 1e-300, None))
    d["log_xi"] = np.log10(np.clip(d["xi"], 1e-300, None))
    d["logMstar"] = np.log10(np.clip(d["Mstar"], 1e-300, None))
    d["logMh"] = np.log10(np.clip(d["Mh"], 1e-300, None))
    d["g_ext_proxy"] = d["Mh"] / np.clip(d["Rext"], 1e-300, None) ** 2
    d["log_g_ext_proxy"] = np.log10(np.clip(d["g_ext_proxy"], 1e-300, None))

    scat = compute_sigma_dm_robust(dm_threshold=-10.5, min_pts=10)
    d = d.merge(
        scat[["galaxy", "n_dm", "sigma_dm_robust", "log_sigma_dm_robust"]],
        left_on="gal_id",
        right_on="galaxy",
        how="left",
    )

    confounds = ["Rext", "Npts", "dR", "incl", "dist_err_frac"]
    calipers = [0.75, 1.00, 1.25, 1.50, 2.00]

    xi_suite = run_split_suite(
        d, outcome_col="Lc_over_xi", split_col="xi", confounds=confounds, calipers=calipers, seed=42
    )
    rs_suite = run_split_suite(
        d, outcome_col="Lc_over_xi", split_col="r_s_kpc", confounds=confounds, calipers=calipers, seed=42
    )

    xi_pairs = xi_suite.pop("primary_pairs_df", pd.DataFrame())
    rs_pairs = rs_suite.pop("primary_pairs_df", pd.DataFrame())

    out_xi_pairs = os.path.join(RESULTS_DIR, "xi_split_pairs_primary.parquet")
    out_rs_pairs = os.path.join(RESULTS_DIR, "rs_split_pairs_primary.parquet")
    if not xi_pairs.empty:
        xi_pairs.to_parquet(out_xi_pairs, index=False)
    if not rs_pairs.empty:
        rs_pairs.to_parquet(out_rs_pairs, index=False)

    comp_rows = []
    xi_by_cal = {float(r["caliper_z"]): r for r in xi_suite.get("matched_by_caliper", []) if "caliper_z" in r}
    rs_by_cal = {float(r["caliper_z"]): r for r in rs_suite.get("matched_by_caliper", []) if "caliper_z" in r}
    for c in sorted(set(xi_by_cal.keys()) | set(rs_by_cal.keys())):
        row = {"caliper_z": c}
        for pref, src in [("xi", xi_by_cal.get(c, {})), ("rs", rs_by_cal.get(c, {}))]:
            row[f"{pref}_pairs"] = src.get("pairs", np.nan)
            row[f"{pref}_cliffs_delta"] = src.get("cliffs_delta", np.nan)
            row[f"{pref}_perm_p"] = src.get("perm_p_median_diff", np.nan)
            row[f"{pref}_median_delta"] = src.get("median_delta_metric_high_minus_low", np.nan)
            row[f"{pref}_frac_delta_lt0"] = src.get("fraction_delta_lt_0", np.nan)
            row[f"{pref}_spearman_dMetric_dGext"] = src.get("spearman_deltaMetric_vs_deltaLogGext_rho", np.nan)
            row[f"{pref}_spearman_p"] = src.get("spearman_p", np.nan)
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    out_comp = os.path.join(RESULTS_DIR, "rs_vs_xi_split_by_caliper.csv")
    comp_df.to_csv(out_comp, index=False)

    horse = run_horse_race(d)
    horse_df = pd.DataFrame(horse.get("model_rows", []))
    out_horse = os.path.join(RESULTS_DIR, "horse_race_sigma_dm_robust.csv")
    if not horse_df.empty:
        horse_df.to_csv(out_horse, index=False)

    xi_primary = xi_suite.get("primary_matched", {})
    rs_primary = rs_suite.get("primary_matched", {})

    summary = {
        "analysis": "r_s_vs_xi_discriminator",
        "inputs": {
            "results_scale_hierarchy": RESULTS_TABLE,
            "rar_points_unified": RAR_POINTS,
        },
        "sample_counts": {
            "sparc_structured_with_lc_xi": int(len(d[np.isfinite(d["Lc_over_xi"]) & np.isfinite(d["xi"])])),
            "sparc_structured_with_sigma_dm_robust": int(np.isfinite(d["sigma_dm_robust"]).sum()),
        },
        "assumptions": {
            "rs_derivation": "Dutton-Maccio 2014 c(M) mean relation with rho_crit from analysis_tools.py constants",
            "g_ext_proxy": "Mh / Rext^2 (proxy only; not direct host-field reconstruction)",
            "sigma_dm_robust": "MAD-based robust sigma of RAR residuals for SPARC points with log_gbar < -10.5 and n_dm >= 10",
        },
        "split_test": {
            "outcome": "Lc_over_xi",
            "confounds": confounds,
            "xi_split": xi_suite,
            "rs_split": rs_suite,
            "head_to_head_primary": {
                "xi_primary_caliper": xi_primary.get("caliper_z"),
                "xi_primary_pairs": xi_primary.get("pairs"),
                "xi_primary_cliffs_delta": xi_primary.get("cliffs_delta"),
                "xi_primary_perm_p": xi_primary.get("perm_p_median_diff"),
                "rs_primary_caliper": rs_primary.get("caliper_z"),
                "rs_primary_pairs": rs_primary.get("pairs"),
                "rs_primary_cliffs_delta": rs_primary.get("cliffs_delta"),
                "rs_primary_perm_p": rs_primary.get("perm_p_median_diff"),
            },
        },
        "horse_race_sigma_dm_robust": horse,
        "outputs": {
            "summary_json": os.path.join(RESULTS_DIR, "rs_discriminator_summary.json"),
            "split_comparison_csv": out_comp,
            "rs_primary_pairs": out_rs_pairs,
            "xi_primary_pairs": out_xi_pairs,
            "horse_race_csv": out_horse,
        },
    }

    out_json = os.path.join(RESULTS_DIR, "rs_discriminator_summary.json")
    with open(out_json, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print("=" * 72)
    print("r_s VS xi DISCRIMINATOR SUMMARY")
    print("=" * 72)
    print(f"SPARC structured sample (Lc_over_xi): {summary['sample_counts']['sparc_structured_with_lc_xi']}")
    print(f"SPARC with sigma_dm_robust: {summary['sample_counts']['sparc_structured_with_sigma_dm_robust']}")
    print("\nPrimary matched split (outcome = Lc_over_xi):")
    if xi_primary:
        print(
            f"  xi split: caliper={xi_primary.get('caliper_z')}, pairs={int(xi_primary.get('pairs', 0))}, "
            f"Cliff's={xi_primary.get('cliffs_delta'):.3f}, perm_p={xi_primary.get('perm_p_median_diff'):.4f}"
        )
    if rs_primary:
        print(
            f"  r_s split: caliper={rs_primary.get('caliper_z')}, pairs={int(rs_primary.get('pairs', 0))}, "
            f"Cliff's={rs_primary.get('cliffs_delta'):.3f}, perm_p={rs_primary.get('perm_p_median_diff'):.4f}"
        )

    if not horse_df.empty:
        best = horse_df.sort_values("aic").iloc[0]
        print("\nHorse-race (outcome = log_sigma_dm_robust):")
        print(
            f"  Best AIC model: {best['model']} | AIC={best['aic']:.3f}, BIC={best['bic']:.3f}, "
            f"R2={best['r2']:.3f}, CV_R2={best['cv_r2']:.3f}"
        )
        print(f"  log_xi-log_rs corr: {horse.get('corr_log_xi_log_rs', np.nan):.3f}")
        cb = horse.get("coef_checks", {})
        if "plus_both_log_xi" in cb:
            x = cb["plus_both_log_xi"]
            r = cb["plus_both_log_rs"]
            print(
                "  plus_both coef bootstrap: "
                f"log_xi median={x.get('coef', np.nan):.3f} "
                f"[{x.get('ci95_lo', np.nan):.3f}, {x.get('ci95_hi', np.nan):.3f}], "
                f"log_rs median={r.get('coef', np.nan):.3f} "
                f"[{r.get('ci95_lo', np.nan):.3f}, {r.get('ci95_hi', np.nan):.3f}]"
            )

    print("\nSaved:")
    print(f"  {out_json}")
    print(f"  {out_comp}")
    print(f"  {out_horse}")
    if os.path.exists(out_rs_pairs):
        print(f"  {out_rs_pairs}")
    if os.path.exists(out_xi_pairs):
        print(f"  {out_xi_pairs}")


if __name__ == "__main__":
    main()
