#!/usr/bin/env python3
"""
Referee-required test runner for BEC dark matter RAR analyses.

This script executes five publication-gating tests and writes the exact
JSON/figure artifacts requested by the referee prompt:

1) phase_peak_null_distribution
2) mass_matched_phase
3) xi_organizing
4) alpha_star_convergence
5) dataset_lineage_audit
"""

from __future__ import annotations

import os
import sys


def _pin_blas(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


_blas_n = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--blas-threads" and _i + 1 < len(sys.argv):
        try:
            _blas_n = int(sys.argv[_i + 1])
        except ValueError:
            pass
_pin_blas(_blas_n)

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ks_2samp, wilcoxon

warnings.filterwarnings("ignore")
plt.style.use("default")

# Physical/constants per referee prompt
LOG_G_DAGGER = -9.921
G_DAGGER = 1.2e-10  # m/s^2
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
MSUN = 1.989e30  # kg
KPC = 3.086e19  # m

BIN_WIDTH = 0.25
BIN_EDGES = np.arange(-13.5, -8.0 + 1e-12, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + 0.5 * BIN_WIDTH


def rar_bec(log_gbar: np.ndarray, log_gd: float = LOG_G_DAGGER) -> np.ndarray:
    gbar = 10.0 ** log_gbar
    gd = 10.0 ** log_gd
    x = np.sqrt(np.maximum(gbar / gd, 1e-300))
    denom = np.maximum(1.0 - np.exp(-x), 1e-300)
    return np.log10(gbar / denom)


def healing_length_kpc(M_total_Msun: np.ndarray) -> np.ndarray:
    return np.sqrt(G_SI * M_total_Msun * MSUN / G_DAGGER) / KPC


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, (np.ndarray,)):
        return [sanitize_json(v) for v in obj.tolist()]
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(obj), indent=2))


def robust_percentiles(x: np.ndarray, q: Sequence[float]) -> List[Optional[float]]:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return [None for _ in q]
    return [float(v) for v in np.percentile(y, q)]


def pick_id_col(cols: Iterable[str]) -> Optional[str]:
    c = set(cols)
    for cand in ["SubhaloID", "subhalo_id", "galaxy_id", "galaxy", "id"]:
        if cand in c:
            return cand
    return None


def ensure_repo_structure(root: Path) -> Dict[str, Any]:
    required_dirs = ["analysis", "analysis/pipeline", "analysis/results", "datasets"]
    required_files = [
        "analysis/results/rar_points_unified.csv",
        "analysis/results/galaxy_results_unified.csv",
    ]
    top_level = sorted([p.name for p in root.iterdir() if p.is_dir()])
    checks = {
        "repo_root": str(root),
        "top_level_dirs": top_level,
        "required_dirs_present": {d: (root / d).exists() for d in required_dirs},
        "required_files_present": {f: (root / f).exists() for f in required_files},
    }
    checks["ok"] = all(checks["required_dirs_present"].values()) and all(
        checks["required_files_present"].values()
    )
    return checks


def discover_tng_candidates(root: Path) -> Dict[str, Optional[Path]]:
    out: Dict[str, Optional[Path]] = {
        "clean_3000_points": None,
        "clean_3000_master": None,
        "clean_48133_points": None,
        "clean_48133_master": None,
        "contaminated_20899_points": None,
    }
    cands = {
        "clean_3000_points": root
        / "datasets/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points_CLEAN.parquet",
        "clean_3000_master": root
        / "datasets/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/meta/master_catalog.csv",
        "clean_48133_points": root
        / "datasets/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/rar_points.parquet",
        "clean_48133_master": root
        / "datasets/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/meta/master_catalog.csv",
        "contaminated_20899_points": root
        / "datasets/quarantine_contaminated/rar_points_20899x15_CONTAMINATED.parquet",
    }
    for k, p in cands.items():
        if p.exists():
            out[k] = p
    return out


def load_sparc_points(root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = root / "analysis/results/rar_points_unified.csv"
    df = pd.read_csv(p)
    df = df[df["source"] == "SPARC"].copy()
    for col in ["galaxy", "log_gbar", "log_gobs", "R_kpc"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing SPARC column: {col}")
    df = df[np.isfinite(df["log_gbar"]) & np.isfinite(df["log_gobs"]) & np.isfinite(df["R_kpc"])].copy()

    recomputed = df["log_gobs"].to_numpy() - rar_bec(df["log_gbar"].to_numpy())
    has_log_res = "log_res" in df.columns
    if has_log_res:
        diff = np.abs(df["log_res"].to_numpy() - recomputed)
        med_abs_diff = float(np.nanmedian(diff))
        max_abs_diff = float(np.nanmax(diff))
        use_existing = med_abs_diff <= 0.01
    else:
        med_abs_diff = None
        max_abs_diff = None
        use_existing = False

    if use_existing:
        df["log_res_use"] = df["log_res"].astype(float)
    else:
        df["log_res_use"] = recomputed

    meta = {
        "file": str(p),
        "n_points": int(len(df)),
        "n_galaxies": int(df["galaxy"].nunique()),
        "log_gbar_min": float(df["log_gbar"].min()),
        "log_gbar_max": float(df["log_gbar"].max()),
        "used_existing_log_res": bool(use_existing),
        "log_res_median_abs_diff_recomputed": med_abs_diff,
        "log_res_max_abs_diff_recomputed": max_abs_diff,
    }
    return df, meta


def load_tng_points(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    id_col = pick_id_col(df.columns)
    if id_col is None:
        raise RuntimeError(f"No galaxy ID column found in {path}")
    if "r_kpc" not in df.columns:
        if "R_kpc" in df.columns:
            df = df.rename(columns={"R_kpc": "r_kpc"})
        else:
            raise RuntimeError(f"No radius column (r_kpc) found in {path}")
    for c in ["log_gbar", "log_gobs"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} in {path}")

    df = df.rename(columns={id_col: "id"}).copy()
    df = df[np.isfinite(df["log_gbar"]) & np.isfinite(df["log_gobs"]) & np.isfinite(df["r_kpc"])].copy()
    # Remove placeholder floor values used in some extraction runs.
    df = df[(df["log_gbar"] > -20.0) & (df["log_gobs"] > -20.0) & (df["r_kpc"] > 0)].copy()
    df["log_res_use"] = df["log_gobs"].to_numpy() - rar_bec(df["log_gbar"].to_numpy())
    return df


def compute_galaxy_mass_table(
    df: pd.DataFrame, id_col: str, r_col: str, min_points: int = 1
) -> pd.DataFrame:
    rows = []
    for gid, g in df.groupby(id_col, sort=False):
        g2 = g.sort_values(r_col)
        n = int(len(g2))
        if n < min_points:
            continue
        r_out = float(g2[r_col].iloc[-1])
        lgb_out = float(g2["log_gbar"].iloc[-1])
        lgo_out = float(g2["log_gobs"].iloc[-1])
        M_bar = (10.0**lgb_out) * (r_out * KPC) ** 2 / G_SI / MSUN
        M_dyn = (10.0**lgo_out) * (r_out * KPC) ** 2 / G_SI / MSUN
        if not np.isfinite(M_bar) or M_bar <= 0 or not np.isfinite(M_dyn) or M_dyn <= 0:
            continue
        rows.append(
            {
                "id": gid,
                "n_points": n,
                "R_out_kpc": r_out,
                "log_gbar_out": lgb_out,
                "log_gobs_out": lgo_out,
                "log_Mb": float(np.log10(M_bar)),
                "log_Mdyn": float(np.log10(M_dyn)),
                "xi_kpc": float(healing_length_kpc(np.array([M_dyn]))[0]),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BinPrep:
    bin_idx: np.ndarray
    counts: np.ndarray
    valid_bins: np.ndarray
    centers_valid: np.ndarray


def prepare_binning(log_gbar: np.ndarray, min_points: int = 10) -> BinPrep:
    idx = np.digitize(log_gbar, BIN_EDGES) - 1
    good = (idx >= 0) & (idx < len(BIN_CENTERS))
    idx2 = np.full_like(idx, -1)
    idx2[good] = idx[good]
    counts = np.bincount(idx2[good], minlength=len(BIN_CENTERS))
    valid_bins = np.where(counts >= min_points)[0]
    return BinPrep(bin_idx=idx2, counts=counts, valid_bins=valid_bins, centers_valid=BIN_CENTERS[valid_bins])


def variance_profile_from_prebinned(residuals: np.ndarray, prep: BinPrep) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    good = prep.bin_idx >= 0
    idx = prep.bin_idx[good]
    r = residuals[good]
    n_bins = len(BIN_CENTERS)
    sum_r = np.bincount(idx, weights=r, minlength=n_bins)
    sum_r2 = np.bincount(idx, weights=r * r, minlength=n_bins)
    n = prep.counts.astype(float)

    var = np.full(n_bins, np.nan)
    ok = n > 1
    var[ok] = (sum_r2[ok] - (sum_r[ok] ** 2) / n[ok]) / (n[ok] - 1.0)
    var = np.maximum(var, 1e-12)
    err = np.full(n_bins, np.nan)
    err[ok] = var[ok] * np.sqrt(2.0 / (n[ok] - 1.0))
    err = np.maximum(err, 1e-12)

    vb = prep.valid_bins
    return prep.centers_valid.copy(), var[vb], err[vb]


def nll_from_model(y: np.ndarray, yhat: np.ndarray, yerr: np.ndarray) -> float:
    if np.any(~np.isfinite(yhat)):
        return 1e30
    # Smooth positivity penalty: keeps optimizer gradients informative
    # instead of hard-failing invalid starts.
    neg = np.minimum(yhat, 0.0)
    penalty = 1e7 * float(np.sum(neg * neg))
    yhat = np.where(yhat <= 1e-12, 1e-12, yhat)
    sig2 = np.maximum(yerr * yerr, 1e-18)
    base = 0.5 * float(np.sum(((y - yhat) ** 2) / sig2 + np.log(2.0 * np.pi * sig2)))
    return base + penalty


def fit_m1_linear(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, k_for_aic: int = 3) -> Dict[str, Any]:
    w = 1.0 / np.maximum(yerr * yerr, 1e-18)
    A = np.column_stack([np.ones_like(x), x])
    ATW = A.T * w
    ATA = ATW @ A
    ATy = ATW @ y
    try:
        beta = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = beta[0] + beta[1] * x
    yhat = np.maximum(yhat, 1e-12)
    nll = nll_from_model(y, yhat, yerr)
    aic = 2 * k_for_aic + 2 * nll
    return {"params": beta, "nll": nll, "aic": float(aic), "yhat": yhat, "k": int(k_for_aic)}


def gauss(x: np.ndarray, mu: float, w: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / np.maximum(w, 1e-9)) ** 2)


def model_edge(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    s0, s1, Ap, mup, wp, Ad, mud, wd, E, xe, de = params
    edge = 1.0 / (1.0 + np.exp(-(x - xe) / np.maximum(de, 1e-6)))
    return s0 + s1 * x + Ap * gauss(x, mup, wp) + Ad * gauss(x, mud, wd) + E * edge


def model_peak_dip(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    s0, s1, Ap, mup, wp, Ad, mud, wd = params
    return s0 + s1 * x + Ap * gauss(x, mup, wp) + Ad * gauss(x, mud, wd)


def fit_edge_model(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 30,
    maxiter: int = 2000,
) -> Dict[str, Any]:
    bounds = [
        (1e-8, 5.0),  # s0
        (-2.0, 2.0),  # s1
        (1e-6, 5.0),  # Ap > 0
        (-12.0, -8.0),  # mup
        (0.05, 2.0),  # wp
        (-5.0, -1e-6),  # Ad < 0
        (-12.0, -8.0),  # mud
        (0.05, 2.0),  # wd
        (-5.0, 5.0),  # E
        (-12.0, -8.0),  # xe
        (0.01, 1.0),  # de
    ]

    def obj(p: np.ndarray) -> float:
        return nll_from_model(y, model_edge(p, x), yerr)

    starts: List[np.ndarray] = []
    y_med = float(np.median(y))
    y_span = float(max(np.max(y) - np.min(y), 0.01))
    starts.append(
        np.array(
            [
                max(y_med, 1e-4),
                0.0,
                0.5 * y_span,
                LOG_G_DAGGER,
                0.35,
                -0.25 * y_span,
                LOG_G_DAGGER - 0.3,
                0.25,
                0.0,
                -9.0,
                0.15,
            ],
            dtype=float,
        )
    )
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    for _ in range(max(0, n_starts - 1)):
        p = lb + rng.random(len(bounds)) * (ub - lb)
        # Make random starts feasible for variance positivity by lifting baseline.
        y0 = model_edge(p, x)
        if np.min(y0) <= 1e-5:
            p[0] += float(1e-3 - np.min(y0))
        starts.append(p)

    best = None
    for p0 in starts:
        try:
            res = minimize(
                obj,
                p0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter)},
            )
            if not np.isfinite(res.fun):
                continue
            if best is None or float(res.fun) < float(best.fun):
                best = res
        except Exception:
            continue

    if best is None:
        return {"ok": False, "model": "M2b_edge"}
    nll = float(best.fun)
    mup = float(best.x[3]); mud = float(best.x[6])
    mu_peak = mup if abs(mup - LOG_G_DAGGER) <= abs(mud - LOG_G_DAGGER) else mud
    return {
        "ok": True,
        "model": "M2b_edge",
        "params": np.asarray(best.x, dtype=float),
        "nll": nll,
        "aic": float(2 * 11 + 2 * nll),  # referee-specified k=11
        "mu_peak": mu_peak, "mup_raw": mup, "mud_raw": mud,
        "result_success": bool(getattr(best, "success", False)),
    }


def fit_peak_dip_fallback(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 15,
    maxiter: int = 1500,
) -> Dict[str, Any]:
    bounds = [
        (1e-8, 5.0),  # s0
        (-2.0, 2.0),  # s1
        (1e-6, 5.0),  # Ap > 0
        (-12.0, -8.0),  # mup
        (0.05, 2.0),  # wp
        (-5.0, -1e-6),  # Ad < 0
        (-12.0, -8.0),  # mud
        (0.05, 2.0),  # wd
    ]

    def obj(p: np.ndarray) -> float:
        return nll_from_model(y, model_peak_dip(p, x), yerr)

    y_med = float(np.median(y))
    y_span = float(max(np.max(y) - np.min(y), 0.01))
    starts = [
        np.array(
            [
                max(y_med, 1e-4),
                0.0,
                0.4 * y_span,
                LOG_G_DAGGER,
                0.35,
                -0.2 * y_span,
                LOG_G_DAGGER - 0.25,
                0.25,
            ],
            dtype=float,
        )
    ]
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    for _ in range(max(0, n_starts - 1)):
        p = lb + rng.random(len(bounds)) * (ub - lb)
        y0 = model_peak_dip(p, x)
        if np.min(y0) <= 1e-5:
            p[0] += float(1e-3 - np.min(y0))
        starts.append(p)

    best = None
    for p0 in starts:
        try:
            res = minimize(
                obj,
                p0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter)},
            )
            if not np.isfinite(res.fun):
                continue
            if best is None or float(res.fun) < float(best.fun):
                best = res
        except Exception:
            continue

    if best is None:
        return {"ok": False, "model": "M2b_peak_dip_fallback"}
    nll = float(best.fun)
    mup = float(best.x[3]); mud = float(best.x[6])
    mu_peak = mup if abs(mup - LOG_G_DAGGER) <= abs(mud - LOG_G_DAGGER) else mud
    return {
        "ok": True,
        "model": "M2b_peak_dip_fallback",
        "params": np.asarray(best.x, dtype=float),
        "nll": nll,
        "aic": float(2 * 9 + 2 * nll),  # technical-note k=9 fallback
        "mu_peak": mu_peak, "mup_raw": mup, "mud_raw": mud,
        "result_success": bool(getattr(best, "success", False)),
    }


def fit_phase_profile_models(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts_edge: int = 30,
    n_starts_null: int = 5,
    for_null: bool = False,
) -> Dict[str, Any]:
    m1 = fit_m1_linear(x, y, yerr, k_for_aic=3)
    starts = n_starts_null if for_null else n_starts_edge
    maxiter = 900 if for_null else 2500
    edge = fit_edge_model(x, y, yerr, rng=rng, n_starts=starts, maxiter=maxiter)
    used_fallback = False

    if not edge.get("ok", False):
        edge = fit_peak_dip_fallback(
            x, y, yerr, rng=rng, n_starts=max(6, starts), maxiter=1200 if for_null else 1800
        )
        used_fallback = True

    if not edge.get("ok", False):
        return {
            "ok": False,
            "m1": m1,
            "edge": edge,
            "used_fallback": True,
        }

    return {
        "ok": True,
        "m1": m1,
        "edge": edge,
        "used_fallback": used_fallback,
        "mu_peak": float(edge["mu_peak"]),
        "aic_m1": float(m1["aic"]),
        "aic_edge": float(edge["aic"]),
        "daic": float(edge["aic"] - m1["aic"]),
    }


def eval_fit_curve(fit: Dict[str, Any], xgrid: np.ndarray) -> np.ndarray:
    if fit["model"] == "M2b_edge":
        return model_edge(np.asarray(fit["params"], dtype=float), xgrid)
    return model_peak_dip(np.asarray(fit["params"], dtype=float), xgrid)


def shuffle_within_galaxy(
    residuals: np.ndarray, groups: List[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    out = residuals.copy()
    for idx in groups:
        out[idx] = residuals[idx][rng.permutation(len(idx))]
    return out


def _resample_rank(values: np.ndarray, n_out: int) -> np.ndarray:
    n_in = len(values)
    if n_in == n_out:
        return values.copy()
    if n_in <= 1:
        return np.repeat(values[0], n_out)
    xp = np.linspace(0.0, 1.0, n_in)
    xnew = np.linspace(0.0, 1.0, n_out)
    return np.interp(xnew, xp, values)


def shuffle_galaxy_label(
    residuals: np.ndarray, groups: List[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    n_g = len(groups)
    order = rng.permutation(n_g)
    source_vecs = [residuals[idx] for idx in groups]
    unique_lengths = sorted({len(idx) for idx in groups})
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    for src in range(n_g):
        for L in unique_lengths:
            cache[(src, L)] = _resample_rank(source_vecs[src], L)

    out = residuals.copy()
    for tgt_i, src_i in enumerate(order):
        idx_t = groups[tgt_i]
        out[idx_t] = cache[(int(src_i), len(idx_t))]
    return out


# --- Test 1 parallel worker infrastructure ---

_W1_LOG_GBAR: Optional[np.ndarray] = None
_W1_RESIDUALS: Optional[np.ndarray] = None
_W1_PREP: Optional[BinPrep] = None
_W1_GROUPS: List[np.ndarray] = []


def _phase_worker_init(
    log_gbar: np.ndarray, residuals: np.ndarray,
    groups: List[np.ndarray], prep: BinPrep,
) -> None:
    global _W1_LOG_GBAR, _W1_RESIDUALS, _W1_PREP, _W1_GROUPS
    _W1_LOG_GBAR = log_gbar
    _W1_RESIDUALS = residuals
    _W1_PREP = prep
    _W1_GROUPS = groups


def _phase_shuffle_worker(args: Tuple[int, str]) -> dict:
    seed, mode_code = args
    rng = np.random.default_rng(seed)
    if mode_code == "A":
        rs = shuffle_within_galaxy(_W1_RESIDUALS, _W1_GROUPS, rng)
    else:
        rs = shuffle_galaxy_label(_W1_RESIDUALS, _W1_GROUPS, rng)
    xb, vb, eb = variance_profile_from_prebinned(rs, _W1_PREP)
    fit = fit_phase_profile_models(
        xb, vb, eb, rng=rng,
        n_starts_edge=5, n_starts_null=5, for_null=True,
    )
    if not fit["ok"]:
        return {"ok": False}
    return {
        "ok": True,
        "mu_peak": float(fit["mu_peak"]),
        "delta": float(abs(fit["mu_peak"] - LOG_G_DAGGER)),
        "daic": float(fit["daic"]),
        "fallback": bool(fit.get("used_fallback", False)),
    }


def run_phase_null_shuffles(
    log_gbar: np.ndarray,
    residuals: np.ndarray,
    galaxies: np.ndarray,
    delta_real: float,
    daic_real: float,
    rng: np.random.Generator,
    n_shuffles: int = 1000,
    n_jobs: int = 1,
    chunksize: int = 10,
    num_shards: int = 1,
    shard_id: int = 0,
    base_seed: int = 42,
) -> Dict[str, Any]:
    prep = prepare_binning(log_gbar, min_points=10)
    groups = [np.where(galaxies == g)[0] for g in pd.unique(galaxies)]
    groups = [g for g in groups if len(g) > 0]

    shard_indices = [i for i in range(n_shuffles) if (i % num_shards) == shard_id]
    n_shard = len(shard_indices)

    out: Dict[str, Any] = {}
    for mode, mode_code in [("A_within_galaxy", "A"), ("B_galaxy_label", "B")]:
        tasks = [(base_seed + i, mode_code) for i in shard_indices]

        all_results: List[dict] = []
        if n_jobs > 1:
            import multiprocessing as mp
            with mp.Pool(n_jobs, initializer=_phase_worker_init,
                         initargs=(log_gbar, residuals, groups, prep)) as pool:
                done = 0
                for result in pool.imap_unordered(_phase_shuffle_worker, tasks, chunksize=chunksize):
                    all_results.append(result)
                    done += 1
                    if done % 100 == 0:
                        print(f"    [{mode}] {done}/{n_shard}")
        else:
            _phase_worker_init(log_gbar, residuals, groups, prep)
            for j, task in enumerate(tasks):
                result = _phase_shuffle_worker(task)
                all_results.append(result)
                if (j + 1) % 100 == 0:
                    print(f"    [{mode}] {j+1}/{n_shard}")

        mu_null = np.full(n_shard, np.nan, dtype=float)
        delta_null = np.full(n_shard, np.nan, dtype=float)
        daic_null = np.full(n_shard, np.nan, dtype=float)
        n_fail = 0
        n_fallback = 0
        k = 0
        for result in all_results:
            if not result["ok"]:
                n_fail += 1
            else:
                if result.get("fallback", False):
                    n_fallback += 1
                mu_null[k] = result["mu_peak"]
                delta_null[k] = result["delta"]
                daic_null[k] = result["daic"]
            k += 1

        mode_result: Dict[str, Any] = {
            "delta_null": delta_null,
            "daic_null": daic_null,
            "mu_null": mu_null,
            "n_fail": int(n_fail),
            "n_fallback": int(n_fallback),
        }

        if num_shards == 1:
            p_delta = float(np.mean(np.where(np.isfinite(delta_null), delta_null <= delta_real, False)))
            p_daic = float(np.mean(np.where(np.isfinite(daic_null), daic_null <= daic_real, False)))
            d50, d05, d95 = robust_percentiles(delta_null, [50, 5, 95])
            a50, a05, a95 = robust_percentiles(daic_null, [50, 5, 95])
            mode_result.update({
                "p_delta": p_delta,
                "p_daic": p_daic,
                "delta_null_median": d50,
                "delta_null_5pct": d05,
                "delta_null_95pct": d95,
                "daic_null_median": a50,
                "daic_null_5pct": a05,
                "daic_null_95pct": a95,
            })

        out[mode] = mode_result
    out["_shard_indices"] = shard_indices
    return out


def run_test1_phase_peak_null(
    sparc_df: pd.DataFrame, out_dir: Path, rng: np.random.Generator, n_shuffles: int,
    n_jobs: int = 1, chunksize: int = 10, shard_id: int = 0, num_shards: int = 1,
    base_seed: int = 42,
) -> Dict[str, Any]:
    print("\n[TEST 1] Phase peak null distribution")
    x = sparc_df["log_gbar"].to_numpy(dtype=float)
    r = sparc_df["log_res_use"].to_numpy(dtype=float)
    g = sparc_df["galaxy"].astype(str).to_numpy()

    prep = prepare_binning(x, min_points=10)
    xb, vb, eb = variance_profile_from_prebinned(r, prep)
    fit_real = fit_phase_profile_models(xb, vb, eb, rng=rng, n_starts_edge=30, n_starts_null=5, for_null=False)
    if not fit_real["ok"]:
        raise RuntimeError("Failed to fit real phase profile for Test 1.")

    mu_real = float(fit_real["mu_peak"])
    delta_real = float(abs(mu_real - LOG_G_DAGGER))
    daic_real = float(fit_real["daic"])
    nulls = run_phase_null_shuffles(
        x, r, g, delta_real, daic_real, rng=rng, n_shuffles=n_shuffles,
        n_jobs=n_jobs, chunksize=chunksize,
        num_shards=num_shards, shard_id=shard_id, base_seed=base_seed,
    )

    if num_shards > 1:
        shard_dir = out_dir / "shuffle_shards" / f"shard_{shard_id}_of_{num_shards}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for key, mode in [("A", "A_within_galaxy"), ("B", "B_galaxy_label")]:
            np.savez_compressed(
                shard_dir / f"test1_{key}_shuffle_results.npz",
                mu_null=nulls[mode]["mu_null"],
                delta_null=nulls[mode]["delta_null"],
                daic_null=nulls[mode]["daic_null"],
                indices=np.array(nulls["_shard_indices"]),
            )
        meta = {
            "seed": int(base_seed), "shard_id": int(shard_id),
            "num_shards": int(num_shards), "n_shuffles": int(n_shuffles),
            "indices": nulls["_shard_indices"],
            "delta_real": delta_real, "daic_real": daic_real, "mu_real": mu_real,
        }
        write_json(shard_dir / "shard_metadata.json", meta)
        print(f"  [TEST 1] Shard {shard_id}/{num_shards} saved to {shard_dir}")
        return {"status": "SHARD_SAVED", "shard_dir": str(shard_dir)}

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    configs = [
        ("A_within_galaxy", "Δ |μ_peak - log g†|"),
        ("B_galaxy_label", "Δ |μ_peak - log g†|"),
        ("A_within_galaxy", "ΔAIC (edge - M1)"),
        ("B_galaxy_label", "ΔAIC (edge - M1)"),
    ]
    arrays = [
        nulls["A_within_galaxy"]["delta_null"],
        nulls["B_galaxy_label"]["delta_null"],
        nulls["A_within_galaxy"]["daic_null"],
        nulls["B_galaxy_label"]["daic_null"],
    ]
    reals = [delta_real, delta_real, daic_real, daic_real]
    pvals = [
        nulls["A_within_galaxy"]["p_delta"],
        nulls["B_galaxy_label"]["p_delta"],
        nulls["A_within_galaxy"]["p_daic"],
        nulls["B_galaxy_label"]["p_daic"],
    ]
    titles = [
        "Shuffle A: within-galaxy Δ-null",
        "Shuffle B: galaxy-label Δ-null",
        "Shuffle A: within-galaxy ΔAIC-null",
        "Shuffle B: galaxy-label ΔAIC-null",
    ]
    for ax, arr, realv, p, title, xlab in zip(axes.flat, arrays, reals, pvals, titles, [c[1] for c in configs]):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        ax.hist(arr, bins=35, color="#9ecae1", edgecolor="black", alpha=0.9)
        ax.axvline(realv, color="red", linewidth=2.0, label="real")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("count")
        ax.legend(loc="best", frameon=False)
        ax.text(
            0.98,
            0.95,
            f"N={len(arr)}\np={p:.4g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    fig.tight_layout()
    fig_path = out_dir / "fig_phase_null.png"
    fig.savefig(fig_path, facecolor="white")
    plt.close(fig)

    summary = {
        "test": "phase_peak_null_distribution",
        "_provenance": {
            "x_col": "log_gbar", "log_base": "log10", "units": "m/s^2",
            "peak_definition": "M2b_edge Gaussian center nearest log10(g_dagger)",
            "log_g_dagger": LOG_G_DAGGER,
        },
        "n_sparc_points": int(len(sparc_df)),
        "n_sparc_galaxies": int(sparc_df["galaxy"].nunique()),
        "bin_width_dex": BIN_WIDTH,
        "n_bins_used": int(len(xb)),
        "n_shuffles": int(n_shuffles),
        "real": {
            "mu_peak": mu_real,
            "mup_raw": fit_real["edge"].get("mup_raw"),
            "mud_raw": fit_real["edge"].get("mud_raw"),
            "delta_from_gdagger": delta_real,
            "aic_M1": float(fit_real["aic_m1"]),
            "aic_edge": float(fit_real["aic_edge"]),
            "daic": daic_real,
            "fit_model": fit_real["edge"]["model"],
        },
        "shuffle_A_within_galaxy": {
            "p_delta": nulls["A_within_galaxy"]["p_delta"],
            "p_daic": nulls["A_within_galaxy"]["p_daic"],
            "delta_null_median": nulls["A_within_galaxy"]["delta_null_median"],
            "delta_null_5pct": nulls["A_within_galaxy"]["delta_null_5pct"],
            "delta_null_95pct": nulls["A_within_galaxy"]["delta_null_95pct"],
            "daic_null_median": nulls["A_within_galaxy"]["daic_null_median"],
            "daic_null_5pct": nulls["A_within_galaxy"]["daic_null_5pct"],
            "daic_null_95pct": nulls["A_within_galaxy"]["daic_null_95pct"],
            "n_fail": nulls["A_within_galaxy"]["n_fail"],
            "n_fallback": nulls["A_within_galaxy"]["n_fallback"],
        },
        "shuffle_B_galaxy_label": {
            "p_delta": nulls["B_galaxy_label"]["p_delta"],
            "p_daic": nulls["B_galaxy_label"]["p_daic"],
            "delta_null_median": nulls["B_galaxy_label"]["delta_null_median"],
            "delta_null_5pct": nulls["B_galaxy_label"]["delta_null_5pct"],
            "delta_null_95pct": nulls["B_galaxy_label"]["delta_null_95pct"],
            "daic_null_median": nulls["B_galaxy_label"]["daic_null_median"],
            "daic_null_5pct": nulls["B_galaxy_label"]["daic_null_5pct"],
            "daic_null_95pct": nulls["B_galaxy_label"]["daic_null_95pct"],
            "n_fail": nulls["B_galaxy_label"]["n_fail"],
            "n_fallback": nulls["B_galaxy_label"]["n_fallback"],
        },
    }
    write_json(out_dir / "summary_phase_peak_null.json", summary)
    return summary


def mass_match_galaxies(
    sparc_mass: pd.DataFrame, tng_mass: pd.DataFrame, caliper: float = 0.3
) -> pd.DataFrame:
    s = sparc_mass.copy().reset_index(drop=True)
    t = tng_mass.copy().reset_index(drop=True)
    used = np.zeros(len(t), dtype=bool)
    pairs = []
    t_vals = t["log_Mb"].to_numpy(dtype=float)

    for _, row in s.sort_values("log_Mb").iterrows():
        d = np.abs(t_vals - float(row["log_Mb"]))
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > caliper:
            continue
        used[j] = True
        pairs.append(
            {
                "sparc_id": row["id"],
                "tng_id": t.iloc[j]["id"],
                "sparc_log_Mb": float(row["log_Mb"]),
                "tng_log_Mb": float(t.iloc[j]["log_Mb"]),
                "abs_dlogM": float(abs(row["log_Mb"] - t.iloc[j]["log_Mb"])),
            }
        )
    return pd.DataFrame(pairs)


def build_gal_dict(df: pd.DataFrame, id_col: str, x_col: str, r_col: str) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for gid, g in df.groupby(id_col, sort=False):
        x = g[x_col].to_numpy(dtype=float)
        r = g[r_col].to_numpy(dtype=float)
        out[gid] = (x, r)
    return out


def concat_sample_from_dict(gal_dict: Dict[Any, Tuple[np.ndarray, np.ndarray]], ids: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    rs: List[np.ndarray] = []
    for gid in ids:
        x, r = gal_dict[gid]
        xs.append(x)
        rs.append(r)
    return np.concatenate(xs), np.concatenate(rs)


def phase_fit_from_points(
    x: np.ndarray,
    r: np.ndarray,
    rng: np.random.Generator,
    n_starts_edge: int = 30,
    for_null: bool = False,
) -> Dict[str, Any]:
    prep = prepare_binning(x, min_points=10)
    xb, vb, eb = variance_profile_from_prebinned(r, prep)
    fit = fit_phase_profile_models(
        xb,
        vb,
        eb,
        rng=rng,
        n_starts_edge=n_starts_edge,
        n_starts_null=max(5, min(10, n_starts_edge)),
        for_null=for_null,
    )
    return {
        "x_bins": xb,
        "var_bins": vb,
        "var_err": eb,
        "n_bins_used": int(len(xb)),
        "fit": fit,
    }


def run_test2_mass_matched_phase(
    root: Path,
    out_dir: Path,
    sparc_df: pd.DataFrame,
    tng_points_path: Optional[Path],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    print("\n[TEST 2] Mass-matched SPARC vs TNG phase diagram")
    searched = [
        str(root / "datasets/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points_CLEAN.parquet"),
        str(root / "datasets/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/rar_points.parquet"),
        str(Path.home() / "analysis"),
        str(Path.home() / "data"),
        str(Path.home() / "tng_data"),
    ]
    if tng_points_path is None or not tng_points_path.exists():
        blocked = {
            "status": "BLOCKED",
            "reason": "TNG per-point data not found",
            "paths_searched": searched,
        }
        write_json(out_dir / "summary_mass_matched_phase.json", blocked)
        return blocked

    # SPARC mass table
    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sp_mass = compute_galaxy_mass_table(sp, id_col="id", r_col="r_kpc", min_points=5)
    sp = sp[sp["id"].isin(set(sp_mass["id"]))].copy()

    # TNG
    tng = load_tng_points(tng_points_path)
    tng_mass = compute_galaxy_mass_table(tng, id_col="id", r_col="r_kpc", min_points=5)
    tng = tng[tng["id"].isin(set(tng_mass["id"]))].copy()

    pairs = mass_match_galaxies(sp_mass, tng_mass, caliper=0.3)
    if len(pairs) == 0:
        blocked = {
            "status": "BLOCKED",
            "reason": "No SPARC-TNG mass-matched pairs found within ±0.3 dex",
            "tng_file_used": str(tng_points_path),
            "paths_searched": searched,
        }
        write_json(out_dir / "summary_mass_matched_phase.json", blocked)
        return blocked

    sp_ids = set(pairs["sparc_id"])
    tng_ids = set(pairs["tng_id"])
    sp_m = sp[sp["id"].isin(sp_ids)].copy()
    tng_m = tng[tng["id"].isin(tng_ids)].copy()

    # Fits
    fit_sp = phase_fit_from_points(
        sp_m["log_gbar"].to_numpy(dtype=float),
        sp_m["log_res_use"].to_numpy(dtype=float),
        rng=rng,
        n_starts_edge=30,
    )
    fit_tng = phase_fit_from_points(
        tng_m["log_gbar"].to_numpy(dtype=float),
        tng_m["log_res_use"].to_numpy(dtype=float),
        rng=rng,
        n_starts_edge=30,
    )
    if (not fit_sp["fit"]["ok"]) or (not fit_tng["fit"]["ok"]):
        raise RuntimeError("Failed to fit phase models on matched SPARC/TNG.")

    # Bootstrap by galaxy
    n_boot = 200
    sp_dict = build_gal_dict(sp_m, "id", "log_gbar", "log_res_use")
    tng_dict = build_gal_dict(tng_m, "id", "log_gbar", "log_res_use")
    sp_ids_list = np.array(list(sp_dict.keys()))
    tng_ids_list = np.array(list(tng_dict.keys()))

    mu_sp = np.full(n_boot, np.nan)
    mu_tng = np.full(n_boot, np.nan)
    daic_sp = np.full(n_boot, np.nan)
    daic_tng = np.full(n_boot, np.nan)

    for i in range(n_boot):
        ids_sp = rng.choice(sp_ids_list, size=len(sp_ids_list), replace=True)
        ids_t = rng.choice(tng_ids_list, size=len(tng_ids_list), replace=True)
        x_sp, r_sp = concat_sample_from_dict(sp_dict, ids_sp)
        x_t, r_t = concat_sample_from_dict(tng_dict, ids_t)
        bs_sp = phase_fit_from_points(x_sp, r_sp, rng=rng, n_starts_edge=10, for_null=True)
        bs_t = phase_fit_from_points(x_t, r_t, rng=rng, n_starts_edge=10, for_null=True)
        if bs_sp["fit"]["ok"]:
            mu_sp[i] = bs_sp["fit"]["mu_peak"]
            daic_sp[i] = bs_sp["fit"]["daic"]
        if bs_t["fit"]["ok"]:
            mu_tng[i] = bs_t["fit"]["mu_peak"]
            daic_tng[i] = bs_t["fit"]["daic"]
        if (i + 1) % 50 == 0:
            print(f"    [bootstrap] {i+1}/{n_boot}")

    sp_mass_vals = pairs["sparc_log_Mb"].to_numpy(dtype=float)
    tng_mass_vals = pairs["tng_log_Mb"].to_numpy(dtype=float)
    ks = ks_2samp(sp_mass_vals, tng_mass_vals)

    # Figure
    fig = plt.figure(figsize=(14, 4.6), dpi=180)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ymax = max(
        np.nanmax(fit_sp["var_bins"] + fit_sp["var_err"]),
        np.nanmax(fit_tng["var_bins"] + fit_tng["var_err"]),
    )
    xgrid = np.linspace(BIN_CENTERS.min(), BIN_CENTERS.max(), 400)

    for ax, label, fit_obj, color in [
        (ax1, "SPARC (matched)", fit_sp, "#1f77b4"),
        (ax2, "TNG (matched)", fit_tng, "#2ca02c"),
    ]:
        ax.errorbar(
            fit_obj["x_bins"],
            fit_obj["var_bins"],
            yerr=fit_obj["var_err"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            label="binned variance",
        )
        curve = eval_fit_curve(fit_obj["fit"]["edge"], xgrid)
        ax.plot(xgrid, curve, color="black", linewidth=1.5, label=fit_obj["fit"]["edge"]["model"])
        ax.axvline(LOG_G_DAGGER, color="red", linestyle="--", linewidth=1.2, label="log g†")
        ax.set_ylim(0, 1.15 * ymax)
        ax.set_xlabel("log gbar")
        ax.set_ylabel("variance of log residual")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=8)

    bins_mass = np.linspace(
        min(sp_mass_vals.min(), tng_mass_vals.min()),
        max(sp_mass_vals.max(), tng_mass_vals.max()),
        20,
    )
    ax3.hist(sp_mass_vals, bins=bins_mass, alpha=0.6, label="SPARC", color="#1f77b4", edgecolor="black")
    ax3.hist(tng_mass_vals, bins=bins_mass, alpha=0.6, label="TNG", color="#2ca02c", edgecolor="black")
    ax3.set_xlabel("log(Mbar/Msun)")
    ax3.set_ylabel("count")
    ax3.set_title("Mass distributions (matched)")
    ax3.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mass_matched_phase.png", facecolor="white")
    plt.close(fig)

    mu_sp_ci = robust_percentiles(mu_sp, [2.5, 97.5])
    mu_tng_ci = robust_percentiles(mu_tng, [2.5, 97.5])
    daic_sp_ci = robust_percentiles(daic_sp, [2.5, 97.5])
    daic_tng_ci = robust_percentiles(daic_tng, [2.5, 97.5])

    summary = {
        "status": "OK",
        "test": "mass_matched_phase",
        "_provenance": {
            "x_col": "log_gbar", "log_base": "log10", "units": "m/s^2",
            "peak_definition": "M2b_edge Gaussian center nearest log10(g_dagger)",
            "log_g_dagger": LOG_G_DAGGER,
        },
        "tng_file_used": str(tng_points_path),
        "N_matched": int(len(pairs)),
        "mass_range": [float(min(sp_mass_vals.min(), tng_mass_vals.min())), float(max(sp_mass_vals.max(), tng_mass_vals.max()))],
        "ks_pvalue_mass": float(ks.pvalue),
        "sparc": {
            "n_points": int(len(sp_m)),
            "n_galaxies": int(sp_m["id"].nunique()),
            "mu_peak": float(fit_sp["fit"]["mu_peak"]),
            "mup_raw": fit_sp["fit"]["edge"].get("mup_raw"),
            "mud_raw": fit_sp["fit"]["edge"].get("mud_raw"),
            "mu_peak_ci95": mu_sp_ci,
            "daic": float(fit_sp["fit"]["daic"]),
            "daic_ci95": daic_sp_ci,
            "fit_model": fit_sp["fit"]["edge"]["model"],
        },
        "tng": {
            "n_points": int(len(tng_m)),
            "n_galaxies": int(tng_m["id"].nunique()),
            "mu_peak": float(fit_tng["fit"]["mu_peak"]),
            "mup_raw": fit_tng["fit"]["edge"].get("mup_raw"),
            "mud_raw": fit_tng["fit"]["edge"].get("mud_raw"),
            "mu_peak_ci95": mu_tng_ci,
            "daic": float(fit_tng["fit"]["daic"]),
            "daic_ci95": daic_tng_ci,
            "fit_model": fit_tng["fit"]["edge"]["model"],
        },
    }
    write_json(out_dir / "summary_mass_matched_phase.json", summary)
    return summary


def per_galaxy_xi_payload(df: pd.DataFrame, id_col: str, r_col: str, res_col: str) -> List[Dict[str, Any]]:
    payload = []
    for gid, g in df.groupby(id_col, sort=False):
        g2 = g.sort_values(r_col)
        if len(g2) < 8:
            continue
        r = g2[r_col].to_numpy(dtype=float)
        res = g2[res_col].to_numpy(dtype=float)
        lgo = g2["log_gobs"].to_numpy(dtype=float)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(res)) or not np.all(np.isfinite(lgo)):
            continue
        j = int(np.argmax(r))
        M_dyn = (10.0 ** lgo[j]) * (r[j] * KPC) ** 2 / G_SI / MSUN
        if not np.isfinite(M_dyn) or M_dyn <= 0:
            continue
        xi = float(healing_length_kpc(np.array([M_dyn]))[0])
        if not np.isfinite(xi) or xi <= 0:
            continue
        x = r / xi
        lx = np.log10(np.maximum(x, 1e-12))
        payload.append({"id": gid, "logX": lx, "res": res, "xi_kpc": xi, "n_points": len(r)})
    return payload


def stacked_variance_profile(logx_list: List[np.ndarray], res_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-2.0, 1.5, 9)
    n_bins = len(edges) - 1
    vmat = np.full((len(logx_list), n_bins), np.nan, dtype=float)
    for i, (lx, rr) in enumerate(zip(logx_list, res_list)):
        idx = np.digitize(lx, edges) - 1
        for b in range(n_bins):
            m = idx == b
            if int(m.sum()) >= 2:
                vmat[i, b] = float(np.var(rr[m], ddof=1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, vmat


def concentration_from_profile(centers: np.ndarray, mean_var: np.ndarray) -> float:
    x = 10.0 ** centers
    m = np.isfinite(mean_var)
    if m.sum() == 0:
        return np.nan
    core = m & (x >= 0.3) & (x <= 3.0)
    if core.sum() == 0:
        return np.nan
    num = np.nanmean(mean_var[core])
    den = np.nanmean(mean_var[m])
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


# --- Test 3 parallel worker infrastructure ---

_W3_LOGX: Optional[List[np.ndarray]] = None
_W3_RES: Optional[List[np.ndarray]] = None
_W3_CENTERS: Optional[np.ndarray] = None


def _xi_worker_init(
    logx_list: List[np.ndarray], res_list: List[np.ndarray], centers: np.ndarray,
) -> None:
    global _W3_LOGX, _W3_RES, _W3_CENTERS
    _W3_LOGX = logx_list
    _W3_RES = res_list
    _W3_CENTERS = centers


def _xi_worker_chunk(seeds: List[int]) -> List[float]:
    results: List[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rr = [r[rng.permutation(len(r))] for r in _W3_RES]
        _, vmat = stacked_variance_profile(_W3_LOGX, rr)
        mean_prof = np.nanmean(vmat, axis=0)
        results.append(concentration_from_profile(_W3_CENTERS, mean_prof))
    return results


def xi_permutation_null(
    payload: List[Dict[str, Any]],
    centers: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 1000,
    n_jobs: int = 1,
    chunksize: int = 10,
    num_shards: int = 1,
    shard_id: int = 0,
    base_seed: int = 42,
) -> np.ndarray:
    # For large fixed-length TNG payloads, use vectorized permutation path.
    n_g = len(payload)
    if n_g == 0:
        return np.array([], dtype=float)

    shard_indices = [i for i in range(n_perm) if (i % num_shards) == shard_id]
    n_shard = len(shard_indices)

    lengths = np.array([len(p["res"]) for p in payload], dtype=int)
    fixed_len = np.all(lengths == lengths[0])
    large = n_g > 1000 and fixed_len and lengths[0] <= 64
    edges = np.linspace(-2.0, 1.5, 9)
    n_bins = len(edges) - 1
    cnull = np.full(n_shard, np.nan, dtype=float)

    if large:
        # Vectorized path — keep sequential
        res_arr = np.vstack([p["res"] for p in payload]).astype(float)
        logx_arr = np.vstack([p["logX"] for p in payload]).astype(float)
        bin_arr = np.digitize(logx_arr, edges) - 1
        mask_bins = [(bin_arr == b).astype(float) for b in range(n_bins)]
        counts = np.stack([m.sum(axis=1) for m in mask_bins], axis=1)  # G x K
        valid = counts >= 2

        for j, perm_idx in enumerate(shard_indices):
            perm_rng = np.random.default_rng(base_seed + perm_idx)
            rp = perm_rng.permuted(res_arr, axis=1)
            rp2 = rp * rp
            vmat = np.full((n_g, n_bins), np.nan, dtype=float)
            for b in range(n_bins):
                m = mask_bins[b]
                s1 = np.sum(rp * m, axis=1)
                s2 = np.sum(rp2 * m, axis=1)
                n = counts[:, b]
                ok = valid[:, b]
                vb = np.full(n_g, np.nan, dtype=float)
                vb[ok] = (s2[ok] - (s1[ok] ** 2) / n[ok]) / (n[ok] - 1.0)
                vmat[:, b] = vb
            mean_prof = np.nanmean(vmat, axis=0)
            cnull[j] = concentration_from_profile(centers, mean_prof)
            if (j + 1) % 100 == 0:
                print(f"    [xi perm vectorized] {j+1}/{n_shard}")
        return cnull

    # Generic path (SPARC and small datasets) — parallel
    logx_list = [p["logX"] for p in payload]
    res_list = [p["res"] for p in payload]

    if n_jobs > 1:
        import multiprocessing as mp
        chunk_tasks: List[List[int]] = []
        for ci in range(0, n_shard, chunksize):
            chunk_idxs = shard_indices[ci:ci + chunksize]
            chunk_tasks.append([base_seed + idx for idx in chunk_idxs])
        with mp.Pool(n_jobs, initializer=_xi_worker_init,
                     initargs=(logx_list, res_list, centers)) as pool:
            chunk_results = pool.map(_xi_worker_chunk, chunk_tasks)
        k = 0
        for chunk_result in chunk_results:
            for val in chunk_result:
                cnull[k] = val
                k += 1
    else:
        for j, perm_idx in enumerate(shard_indices):
            perm_rng = np.random.default_rng(base_seed + perm_idx)
            rr = [r[perm_rng.permutation(len(r))] for r in res_list]
            _, vmat = stacked_variance_profile(logx_list, rr)
            mean_prof = np.nanmean(vmat, axis=0)
            cnull[j] = concentration_from_profile(centers, mean_prof)
            if (j + 1) % 100 == 0:
                print(f"    [xi perm] {j+1}/{n_shard}")
    return cnull


def xi_dataset_summary(
    name: str,
    df: pd.DataFrame,
    id_col: str,
    r_col: str,
    res_col: str,
    rng: np.random.Generator,
    n_boot: int = 500,
    n_perm: int = 1000,
    n_jobs: int = 1,
    chunksize: int = 10,
    num_shards: int = 1,
    shard_id: int = 0,
    base_seed: int = 42,
) -> Dict[str, Any]:
    payload = per_galaxy_xi_payload(df, id_col=id_col, r_col=r_col, res_col=res_col)
    if len(payload) == 0:
        return {"status": "BLOCKED", "reason": f"{name}: no galaxies with >=8 points for xi test"}

    logx_list = [p["logX"] for p in payload]
    res_list = [p["res"] for p in payload]
    centers, vmat = stacked_variance_profile(logx_list, res_list)
    mean_prof = np.nanmean(vmat, axis=0)

    # Bootstrap CIs (galaxy resampling)
    boot = np.full((n_boot, len(centers)), np.nan, dtype=float)
    n_g = len(payload)
    for i in range(n_boot):
        idx = rng.integers(0, n_g, size=n_g)
        boot[i] = np.nanmean(vmat[idx], axis=0)
        if (i + 1) % 100 == 0:
            print(f"    [{name} bootstrap] {i+1}/{n_boot}")
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    C_real = concentration_from_profile(centers, mean_prof)
    cnull = xi_permutation_null(
        payload, centers, rng=rng, n_perm=n_perm,
        n_jobs=n_jobs, chunksize=chunksize,
        num_shards=num_shards, shard_id=shard_id, base_seed=base_seed,
    )
    p_c = float(np.mean(np.where(np.isfinite(cnull), cnull >= C_real, False))) if num_shards == 1 else None

    # Per-galaxy X_peak
    peaks = []
    for p in payload:
        if p["n_points"] < 10:
            continue
        c2, v2 = stacked_variance_profile([p["logX"]], [p["res"]])
        vv = v2[0]
        ok = np.isfinite(vv)
        if ok.sum() == 0:
            continue
        j = int(np.nanargmax(vv))
        peaks.append(float(c2[j]))
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) > 0:
        try:
            w = wilcoxon(peaks - 0.0, alternative="two-sided")
            p_w = float(w.pvalue)
        except Exception:
            p_w = None
        peak_stats = {
            "n_galaxies": int(len(peaks)),
            "median_log10_X_peak": float(np.nanmedian(peaks)),
            "mean_log10_X_peak": float(np.nanmean(peaks)),
            "std_log10_X_peak": float(np.nanstd(peaks)),
            "wilcoxon_pvalue_median_eq_0": p_w,
        }
    else:
        peak_stats = {
            "n_galaxies": 0,
            "median_log10_X_peak": None,
            "mean_log10_X_peak": None,
            "std_log10_X_peak": None,
            "wilcoxon_pvalue_median_eq_0": None,
        }

    result: Dict[str, Any] = {
        "status": "OK",
        "dataset": name,
        "n_galaxies": int(n_g),
        "X_bins_log10_center": centers,
        "stacked_variance": mean_prof,
        "stacked_variance_ci95_low": lo,
        "stacked_variance_ci95_high": hi,
        "concentration_C": C_real,
        "X_peak_stats": peak_stats,
        "concentration_null_samples": cnull,
    }

    if num_shards == 1:
        c50, c5, c95 = robust_percentiles(cnull, [50, 5, 95])
        result.update({
            "concentration_null_median": c50,
            "concentration_null_5pct": c5,
            "concentration_null_95pct": c95,
            "concentration_pvalue": p_c,
        })

    return result


def run_test3_xi_organizing(
    out_dir: Path,
    sparc_df: pd.DataFrame,
    tng_points_path: Optional[Path],
    rng: np.random.Generator,
    n_jobs: int = 1,
    chunksize: int = 10,
    num_shards: int = 1,
    shard_id: int = 0,
    base_seed: int = 42,
) -> Dict[str, Any]:
    print("\n[TEST 3] xi organizing on clean TNG + SPARC")

    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sp["log_res_use"] = sp["log_res_use"].astype(float)
    sparc_res = xi_dataset_summary(
        "SPARC",
        sp,
        id_col="id",
        r_col="r_kpc",
        res_col="log_res_use",
        rng=rng,
        n_boot=500,
        n_perm=1000,
        n_jobs=n_jobs,
        chunksize=chunksize,
        num_shards=num_shards,
        shard_id=shard_id,
        base_seed=base_seed + 100000,
    )

    if tng_points_path is None or not tng_points_path.exists():
        if num_shards > 1:
            shard_dir = out_dir / "shuffle_shards" / f"shard_{shard_id}_of_{num_shards}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            if sparc_res.get("status") == "OK":
                np.savez_compressed(
                    shard_dir / "test3_sparc_cnull.npz",
                    cnull=np.asarray(sparc_res["concentration_null_samples"], dtype=float),
                    indices=np.arange(len(sparc_res["concentration_null_samples"])),
                )
                # Update shard metadata with Test 3 real-data stats
                meta_path = shard_dir / "shard_metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                else:
                    meta = {}
                meta["C_real_sparc"] = sparc_res["concentration_C"]
                write_json(meta_path, meta)
            print(f"  [TEST 3] Shard {shard_id}/{num_shards} saved (TNG blocked)")
            return {"status": "SHARD_SAVED", "sparc": sparc_res, "tng_status": "BLOCKED"}
        summary = {"sparc": sparc_res, "tng_status": "BLOCKED", "reason": "TNG per-point data not found"}
        write_json(out_dir / "summary_xi_organizing.json", summary)
        return summary

    tng = load_tng_points(tng_points_path)
    tng_res = xi_dataset_summary(
        "TNG",
        tng,
        id_col="id",
        r_col="r_kpc",
        res_col="log_res_use",
        rng=rng,
        n_boot=500,
        n_perm=1000,
        n_jobs=n_jobs,
        chunksize=chunksize,
        num_shards=num_shards,
        shard_id=shard_id,
        base_seed=base_seed + 200000,
    )

    if num_shards > 1:
        shard_dir = out_dir / "shuffle_shards" / f"shard_{shard_id}_of_{num_shards}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for tag, res in [("sparc", sparc_res), ("tng", tng_res)]:
            if res.get("status") == "OK":
                np.savez_compressed(
                    shard_dir / f"test3_{tag}_cnull.npz",
                    cnull=np.asarray(res["concentration_null_samples"], dtype=float),
                    indices=np.arange(len(res["concentration_null_samples"])),
                )
        # Update shard metadata with Test 3 real-data stats
        meta_path = shard_dir / "shard_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}
        if sparc_res.get("status") == "OK":
            meta["C_real_sparc"] = sparc_res["concentration_C"]
        if tng_res.get("status") == "OK":
            meta["C_real_tng"] = tng_res["concentration_C"]
        write_json(meta_path, meta)
        print(f"  [TEST 3] Shard {shard_id}/{num_shards} saved to {shard_dir}")
        return {"status": "SHARD_SAVED", "shard_dir": str(shard_dir)}

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    ds = [sparc_res, tng_res]
    names = ["SPARC", "TNG"]
    for j in range(2):
        d = ds[j]
        ax = axes[0, j]
        if d.get("status") == "OK":
            x = 10.0 ** np.asarray(d["X_bins_log10_center"], dtype=float)
            y = np.asarray(d["stacked_variance"], dtype=float)
            lo = np.asarray(d["stacked_variance_ci95_low"], dtype=float)
            hi = np.asarray(d["stacked_variance_ci95_high"], dtype=float)
            ax.plot(x, y, color="#1f77b4", marker="o")
            ax.fill_between(x, lo, hi, color="#9ecae1", alpha=0.5)
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2)
            ax.set_xscale("log")
            ax.set_xlabel("X = R/xi")
            ax.set_ylabel("stacked variance")
            ax.set_title(f"{names[j]} stacked σ²(X)")
        else:
            ax.text(0.5, 0.5, "BLOCKED", ha="center", va="center")
            ax.set_title(f"{names[j]} stacked σ²(X)")
        ax.set_facecolor("white")

    for j in range(2):
        d = ds[j]
        ax = axes[1, j]
        if d.get("status") == "OK":
            cnull = np.asarray(d["concentration_null_samples"], dtype=float)
            cnull = cnull[np.isfinite(cnull)]
            ax.hist(cnull, bins=35, color="#9ecae1", edgecolor="black")
            ax.axvline(float(d["concentration_C"]), color="red", linewidth=2)
            ax.set_xlabel("C null")
            ax.set_ylabel("count")
            ax.set_title(f"{names[j]} concentration null (p={float(d['concentration_pvalue']):.4g})")
        else:
            ax.text(0.5, 0.5, "BLOCKED", ha="center", va="center")
            ax.set_title(f"{names[j]} concentration null")
        ax.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_xi_organizing.png", facecolor="white")
    plt.close(fig)

    summary = {"sparc": sparc_res, "tng": tng_res, "tng_file_used": str(tng_points_path)}
    write_json(out_dir / "summary_xi_organizing.json", summary)
    return summary


def per_galaxy_alpha_table(points: pd.DataFrame, id_col: str, r_col: str, res_col: str) -> pd.DataFrame:
    mass = compute_galaxy_mass_table(points, id_col=id_col, r_col=r_col, min_points=5)
    sig_rows = []
    for gid, g in points.groupby(id_col, sort=False):
        rr = g[res_col].to_numpy(dtype=float)
        rr = rr[np.isfinite(rr)]
        if len(rr) < 5:
            continue
        s = float(np.std(rr, ddof=1))
        if not np.isfinite(s) or s <= 0:
            continue
        q = float(np.log10(s * s))
        sig_rows.append({"id": gid, "q": q, "sigma_res": s, "n_points": int(len(rr))})
    sig = pd.DataFrame(sig_rows)
    d = mass.merge(sig, on=["id", "n_points"], how="inner")
    d = d[np.isfinite(d["q"]) & np.isfinite(d["log_Mb"])].copy()
    return d


def alpha_star_closed_form(q: np.ndarray, m: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=float)
    ok = np.isfinite(q) & np.isfinite(m)
    q = q[ok]
    m = m[ok]
    if len(q) < 3:
        return np.nan
    vm = float(np.var(m))
    if vm <= 0:
        return np.nan
    cov = float(np.mean((q - np.mean(q)) * (m - np.mean(m))))
    return float(-cov / vm)


def alpha_condition_stats(df: pd.DataFrame, rng: np.random.Generator, n_boot: int = 200) -> Dict[str, Any]:
    if len(df) < 5:
        return {"status": "BLOCKED", "n_gal": int(len(df))}
    q = df["q"].to_numpy(dtype=float)
    m = df["log_Mb"].to_numpy(dtype=float)
    a = alpha_star_closed_form(q, m)
    if not np.isfinite(a):
        return {"status": "BLOCKED", "n_gal": int(len(df))}
    z = q + a * m
    sc = float(np.std(z, ddof=1))

    boots = np.full(n_boot, np.nan, dtype=float)
    n = len(df)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = alpha_star_closed_form(q[idx], m[idx])
    ci = robust_percentiles(boots, [2.5, 97.5])
    return {
        "status": "OK",
        "n_gal": int(len(df)),
        "alpha_star": float(a),
        "alpha_ci95": ci,
        "scatter_z": sc,
    }


def run_test4_alpha_convergence(
    out_dir: Path,
    sparc_df: pd.DataFrame,
    tng_3k_path: Optional[Path],
    tng_48k_path: Optional[Path],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    print("\n[TEST 4] alpha* convergence under matching")

    datasets: Dict[str, Optional[pd.DataFrame]] = {}
    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    datasets["SPARC"] = per_galaxy_alpha_table(sp, id_col="id", r_col="r_kpc", res_col="log_res_use")

    if tng_3k_path is not None and tng_3k_path.exists():
        t3 = load_tng_points(tng_3k_path)
        datasets["TNG_3k"] = per_galaxy_alpha_table(t3, id_col="id", r_col="r_kpc", res_col="log_res_use")
    else:
        datasets["TNG_3k"] = None

    if tng_48k_path is not None and tng_48k_path.exists():
        t48 = load_tng_points(tng_48k_path)
        datasets["TNG_48k"] = per_galaxy_alpha_table(t48, id_col="id", r_col="r_kpc", res_col="log_res_use")
    else:
        datasets["TNG_48k"] = None

    available = {k: v for k, v in datasets.items() if v is not None and len(v) >= 5}
    if len(available) == 0:
        blocked = {"status": "BLOCKED", "reason": "No datasets available for alpha* convergence"}
        write_json(out_dir / "summary_alpha_star_convergence.json", blocked)
        return blocked

    # Define condition filters
    min_mass = max(float(v["log_Mb"].min()) for v in available.values())
    max_mass = min(float(v["log_Mb"].max()) for v in available.values())

    mins = [int(v["n_points"].min()) for v in available.values()]
    n_res = int(max(mins)) if len(mins) else 0

    conditions = [
        ("Full", {"mass": None, "nmin": None}),
        ("Mass-matched", {"mass": (min_mass, max_mass), "nmin": None}),
        ("Resolution-matched", {"mass": None, "nmin": n_res}),
        ("Mass+Resolution", {"mass": (min_mass, max_mass), "nmin": n_res}),
    ]

    rows = []
    by_condition: Dict[str, Dict[str, Any]] = {}
    for cname, cdef in conditions:
        cres: Dict[str, Any] = {}
        for dname, ddf in datasets.items():
            if ddf is None:
                cres[dname] = {"status": "BLOCKED", "reason": "dataset missing"}
                continue
            x = ddf.copy()
            if cdef["mass"] is not None:
                lo, hi = cdef["mass"]
                x = x[(x["log_Mb"] >= lo) & (x["log_Mb"] <= hi)].copy()
            if cdef["nmin"] is not None:
                x = x[x["n_points"] >= int(cdef["nmin"])].copy()
            st = alpha_condition_stats(x, rng=rng, n_boot=200)
            cres[dname] = st
            if st.get("status") == "OK":
                rows.append(
                    {
                        "condition": cname,
                        "dataset": dname,
                        "alpha_star": st["alpha_star"],
                        "alpha_ci_low": st["alpha_ci95"][0],
                        "alpha_ci_high": st["alpha_ci95"][1],
                        "n_gal": st["n_gal"],
                        "scatter_z": st["scatter_z"],
                    }
                )
        vals = [cres[d]["alpha_star"] for d in cres if cres[d].get("status") == "OK"]
        cres["delta_alpha_max"] = float(np.max(vals) - np.min(vals)) if len(vals) >= 2 else None
        by_condition[cname] = cres

    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "summary_alpha_star_convergence_table.csv", index=False)

    # Forest plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    cond_order = [c[0] for c in conditions]
    dnames = ["SPARC", "TNG_3k", "TNG_48k"]
    colors = {"SPARC": "#1f77b4", "TNG_3k": "#2ca02c", "TNG_48k": "#ff7f0e"}
    offsets = {"SPARC": -0.2, "TNG_3k": 0.0, "TNG_48k": 0.2}
    for i, cname in enumerate(cond_order):
        for d in dnames:
            s = by_condition[cname].get(d, {})
            if s.get("status") != "OK":
                continue
            a = float(s["alpha_star"])
            lo, hi = s["alpha_ci95"]
            y = i + offsets[d]
            ax.errorbar(
                a,
                y,
                xerr=[[a - lo], [hi - a]],
                fmt="o",
                color=colors[d],
                capsize=3,
                label=d if i == 0 else None,
            )
    ax.set_yticks(range(len(cond_order)))
    ax.set_yticklabels(cond_order)
    ax.set_xlabel("alpha*")
    ax.set_title("alpha* convergence by matching condition")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_alpha_star.png", facecolor="white")
    plt.close(fig)

    full_da = by_condition["Full"]["delta_alpha_max"]
    mr_da = by_condition["Mass+Resolution"]["delta_alpha_max"]
    if full_da is not None and mr_da is not None and np.isfinite(full_da) and full_da > 0:
        shrink = 1.0 - (mr_da / full_da)
        if shrink > 0.3:
            verdict = "Convergence improves substantially under matching (selection-driven component likely)."
        else:
            verdict = "alpha* differences persist under matching (definition/physics differences remain plausible)."
    else:
        verdict = "Insufficient overlapping datasets to judge convergence."

    summary = {
        "status": "OK",
        "test": "alpha_star_convergence",
        "conditions": by_condition,
        "delta_alpha_max": {k: by_condition[k]["delta_alpha_max"] for k in by_condition},
        "verdict": verdict,
        "tng_status": {
            "TNG_3k": "OK" if datasets["TNG_3k"] is not None else "BLOCKED",
            "TNG_48k": "OK" if datasets["TNG_48k"] is not None else "BLOCKED",
        },
    }
    write_json(out_dir / "summary_alpha_star_convergence.json", summary)
    return summary


def source_stats_from_points(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for src, g in df.groupby("source", sort=True):
        ppg = g.groupby("galaxy").size()
        env = g.get("env_dense", pd.Series(index=g.index, dtype=object)).astype(str).str.lower()
        n_field = int((env == "field").sum())
        n_dense = int((env == "dense").sum())
        n_missing = int((env == "nan").sum() + env.isna().sum())
        out[str(src)] = {
            "n_galaxies": int(g["galaxy"].nunique()),
            "n_points": int(len(g)),
            "points_per_galaxy": {
                "min": int(ppg.min()) if len(ppg) else None,
                "median": float(ppg.median()) if len(ppg) else None,
                "max": int(ppg.max()) if len(ppg) else None,
            },
            "log_gbar_range": [float(g["log_gbar"].min()), float(g["log_gbar"].max())],
            "env_dense_breakdown": {"n_field": n_field, "n_dense": n_dense, "n_missing": n_missing},
        }
    return out


def source_stats_from_galaxy(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for src, g in df.groupby("source", sort=True):
        q1 = float(g["sigma_res"].quantile(0.25))
        q3 = float(g["sigma_res"].quantile(0.75))
        out[str(src)] = {
            "n_galaxies": int(g["galaxy"].nunique()),
            "n_rows": int(len(g)),
            "sigma_res_median": float(g["sigma_res"].median()),
            "sigma_res_IQR": float(q3 - q1),
            "logMh_range": [
                float(g["logMh"].min()) if "logMh" in g.columns else None,
                float(g["logMh"].max()) if "logMh" in g.columns else None,
            ],
        }
    return out


def tng_dataset_quick_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    df = load_tng_points(path)
    ppg = df.groupby("id").size()
    return {
        "path": str(path),
        "status": "present",
        "n_points": int(len(df)),
        "n_galaxies": int(df["id"].nunique()),
        "points_per_galaxy": {
            "min": int(ppg.min()) if len(ppg) else None,
            "median": float(ppg.median()) if len(ppg) else None,
            "max": int(ppg.max()) if len(ppg) else None,
        },
    }


def compute_massmatched_sigma_ratio(
    sparc_df: pd.DataFrame, tng_path: Optional[Path], caliper: float = 0.3
) -> Optional[float]:
    if tng_path is None or not tng_path.exists():
        return None
    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sp_mass = compute_galaxy_mass_table(sp, "id", "r_kpc", min_points=5)
    sp_sig = (
        sp.groupby("id")["log_res_use"]
        .agg(lambda x: float(np.std(x.to_numpy(dtype=float), ddof=1)))
        .reset_index(name="sigma_res")
    )
    sp_tab = sp_mass.merge(sp_sig, on="id", how="inner")

    tng = load_tng_points(tng_path)
    t_mass = compute_galaxy_mass_table(tng, "id", "r_kpc", min_points=5)
    t_sig = (
        tng.groupby("id")["log_res_use"]
        .agg(lambda x: float(np.std(x.to_numpy(dtype=float), ddof=1)))
        .reset_index(name="sigma_res")
    )
    t_tab = t_mass.merge(t_sig, on="id", how="inner")
    pairs = mass_match_galaxies(
        sp_tab[["id", "log_Mb", "n_points"]], t_tab[["id", "log_Mb", "n_points"]], caliper=caliper
    )
    if len(pairs) == 0:
        return None
    spm = sp_tab.set_index("id")
    tm = t_tab.set_index("id")
    ratios = []
    for _, row in pairs.iterrows():
        s = float(spm.loc[row["sparc_id"], "sigma_res"])
        t = float(tm.loc[row["tng_id"], "sigma_res"])
        if s > 0 and np.isfinite(s) and np.isfinite(t):
            ratios.append(t / s)
    if len(ratios) == 0:
        return None
    return float(np.median(ratios))


def run_test5_dataset_lineage(
    root: Path,
    out_dir: Path,
    sparc_df: pd.DataFrame,
    tng_paths: Dict[str, Optional[Path]],
) -> Dict[str, Any]:
    print("\n[TEST 5] Dataset lineage audit")
    p_points = root / "analysis/results/rar_points_unified.csv"
    p_gal = root / "analysis/results/galaxy_results_unified.csv"
    rp = pd.read_csv(p_points)
    gr = pd.read_csv(p_gal)

    points_stats = source_stats_from_points(rp)
    galaxy_stats = source_stats_from_galaxy(gr)

    tng_meta = {
        "verified_clean_3000x50": tng_dataset_quick_meta(tng_paths.get("clean_3000_points") or Path("missing")),
        "verified_clean_48133x50": tng_dataset_quick_meta(tng_paths.get("clean_48133_points") or Path("missing")),
        "contaminated_20899x15": tng_dataset_quick_meta(tng_paths.get("contaminated_20899_points") or Path("missing")),
    }

    fairness_path = root / "analysis/results/tng_sparc_composition_sweep/fairness_gap_vs_threshold.csv"
    contaminated_ratio = None
    if fairness_path.exists():
        try:
            fd = pd.read_csv(fairness_path)
            if "mm_median_ratio" in fd.columns:
                contaminated_ratio = float(fd["mm_median_ratio"].max())
        except Exception:
            contaminated_ratio = None

    clean_ratio = compute_massmatched_sigma_ratio(sparc_df, tng_paths.get("clean_48133_points"), caliper=0.3)

    contamination_note = {
        "text": [
            "A mixed TNG dataset was initially used for SPARC-vs-TNG scatter comparisons.",
            "That mixed set reported TNG scatter substantially higher than SPARC (historical ratio near 4.13 in archived fairness sweeps).",
            "After verification/cleanup, clean TNG datasets reverse the direction (mass-matched median sigma ratio is near 0.5 in current clean-base checks).",
            "All current analyses use verified clean datasets only (3000x50 and/or 48133x50).",
            "Affected earlier outputs include TNG-vs-SPARC composition/fairness sweeps based on 20899x15 mixed extraction outputs.",
        ],
        "historical_ratio_contaminated": contaminated_ratio,
        "current_ratio_clean_massmatched": clean_ratio,
        "known_contaminated_file": str(tng_paths["contaminated_20899_points"]) if tng_paths.get("contaminated_20899_points") else None,
        "known_issue": "mixed extraction passes produced 20899 galaxies x 15 points (non-uniform provenance); quarantined in datasets/quarantine_contaminated",
        "unknowns": [],
    }

    summary = {
        "test": "dataset_lineage_audit",
        "points_file": str(p_points),
        "galaxy_file": str(p_gal),
        "per_source_points": points_stats,
        "per_source_galaxy": galaxy_stats,
        "tng_datasets": tng_meta,
        "contamination_note": contamination_note,
    }
    write_json(out_dir / "summary_dataset_lineage.json", summary)

    # Human-readable markdown
    lines: List[str] = []
    lines.append("# Dataset Lineage Audit")
    lines.append("")
    lines.append("## Source Table (Point-level)")
    lines.append("")
    lines.append("| source | n_galaxies | n_points | ppg_min | ppg_med | ppg_max | log_gbar_min | log_gbar_max | n_field | n_dense | n_missing |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for src, st in sorted(points_stats.items()):
        ppg = st["points_per_galaxy"]
        env = st["env_dense_breakdown"]
        lg = st["log_gbar_range"]
        lines.append(
            f"| {src} | {st['n_galaxies']} | {st['n_points']} | {ppg['min']} | {ppg['median']:.1f} | {ppg['max']} | {lg[0]:.3f} | {lg[1]:.3f} | {env['n_field']} | {env['n_dense']} | {env['n_missing']} |"
        )
    lines.append("")
    lines.append("## Source Table (Galaxy-level)")
    lines.append("")
    lines.append("| source | n_galaxies | sigma_res_median | sigma_res_IQR | logMh_min | logMh_max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for src, st in sorted(galaxy_stats.items()):
        lmh = st["logMh_range"]
        lines.append(
            f"| {src} | {st['n_galaxies']} | {st['sigma_res_median']:.4f} | {st['sigma_res_IQR']:.4f} | {lmh[0]:.3f} | {lmh[1]:.3f} |"
        )
    lines.append("")
    lines.append("## TNG Dataset Lineage")
    lines.append("")
    lines.append("| tag | path | n_galaxies | n_points | ppg_median | status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for tag, st in tng_meta.items():
        if st["status"] != "present":
            lines.append(f"| {tag} | {st['path']} | - | - | - | missing |")
        else:
            lines.append(
                f"| {tag} | {st['path']} | {st['n_galaxies']} | {st['n_points']} | {st['points_per_galaxy']['median']:.1f} | present |"
            )
    lines.append("")
    lines.append("## Contamination Note")
    lines.append("")
    lines.append("CONTAMINATION NOTE:")
    for t in contamination_note["text"]:
        lines.append(f"- {t}")
    if contaminated_ratio is not None:
        lines.append(f"- Historical contaminated scatter-ratio proxy: {contaminated_ratio:.3f}")
    if clean_ratio is not None:
        lines.append(f"- Current clean mass-matched scatter-ratio proxy: {clean_ratio:.3f}")
    lines.append(f"- Contaminated file: {contamination_note['known_contaminated_file']}")
    lines.append(f"- Diagnosed issue: {contamination_note['known_issue']}")
    (out_dir / "dataset_lineage.md").write_text("\n".join(lines) + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run referee-required RAR tests.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--auto-jobs", action="store_true")
    parser.add_argument("--reserve-cpus", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=10)
    parser.add_argument("--shuffle-num-shards", type=int, default=1)
    parser.add_argument("--shuffle-shard-id", type=int, default=0)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    root = Path(args.project_root).resolve() if args.project_root else default_root
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-jobs resolution
    n_jobs = args.n_jobs
    if args.auto_jobs:
        import multiprocessing as mp
        total = mp.cpu_count() or 1
        n_jobs = max(1, total - args.reserve_cpus)
        print(f"[auto-jobs] {total} CPUs -> {n_jobs} workers")

    num_shards = args.shuffle_num_shards
    shard_id = args.shuffle_shard_id
    base_seed = args.seed

    rng = np.random.default_rng(args.seed)
    print("=" * 72)
    print("REFEREE-REQUIRED TEST RUNNER")
    print("=" * 72)
    print(f"Project root: {root}")
    print(f"Output dir:   {out_dir}")
    print(f"Seed:         {args.seed}")
    print(f"n_jobs:       {n_jobs}")
    if num_shards > 1:
        print(f"Shard:        {shard_id} of {num_shards}")

    # Pre-run structure verification
    structure = ensure_repo_structure(root)
    print("\n[STRUCTURE CHECK]")
    print(f"  ok: {structure['ok']}")
    print(f"  repo_root: {structure['repo_root']}")
    print(f"  top-level dirs: {', '.join(structure['top_level_dirs'])}")
    for k, v in structure["required_dirs_present"].items():
        print(f"  dir {k}: {v}")
    for k, v in structure["required_files_present"].items():
        print(f"  file {k}: {v}")
    if not structure["ok"]:
        raise RuntimeError("Repository structure check failed; aborting.")

    # Core data load
    sparc_df, sparc_meta = load_sparc_points(root)
    print("\n[SPARC CHECK]")
    print(f"  n_points={sparc_meta['n_points']}, n_galaxies={sparc_meta['n_galaxies']}")
    print(
        f"  log_res source: {'existing column' if sparc_meta['used_existing_log_res'] else 'recomputed'} "
        f"(median abs diff={sparc_meta['log_res_median_abs_diff_recomputed']})"
    )

    tng_paths = discover_tng_candidates(root)
    print("\n[TNG DISCOVERY]")
    for k, v in tng_paths.items():
        print(f"  {k}: {v if v is not None else 'MISSING'}")

    # Execution order required by referee prompt:
    # 1) Test 1, 5) Test 5, 2) Test 2, 3) Test 3, 4) Test 4
    t1 = run_test1_phase_peak_null(
        sparc_df, out_dir=out_dir, rng=rng, n_shuffles=int(args.n_shuffles),
        n_jobs=n_jobs, chunksize=args.chunksize,
        shard_id=shard_id, num_shards=num_shards, base_seed=base_seed,
    )
    _ = t1
    run_test5_dataset_lineage(root, out_dir=out_dir, sparc_df=sparc_df, tng_paths=tng_paths)
    run_test2_mass_matched_phase(
        root,
        out_dir=out_dir,
        sparc_df=sparc_df,
        tng_points_path=tng_paths.get("clean_48133_points"),
        rng=rng,
    )
    run_test3_xi_organizing(
        out_dir=out_dir,
        sparc_df=sparc_df,
        tng_points_path=tng_paths.get("clean_3000_points"),
        rng=rng,
        n_jobs=n_jobs, chunksize=args.chunksize,
        num_shards=num_shards, shard_id=shard_id, base_seed=base_seed,
    )
    run_test4_alpha_convergence(
        out_dir=out_dir,
        sparc_df=sparc_df,
        tng_3k_path=tng_paths.get("clean_3000_points"),
        tng_48k_path=tng_paths.get("clean_48133_points"),
        rng=rng,
    )

    print("\nDone.")
    print("Wrote:")
    for name in [
        "summary_phase_peak_null.json",
        "fig_phase_null.png",
        "summary_mass_matched_phase.json",
        "fig_mass_matched_phase.png",
        "summary_xi_organizing.json",
        "fig_xi_organizing.png",
        "summary_alpha_star_convergence.json",
        "fig_alpha_star.png",
        "summary_dataset_lineage.json",
        "dataset_lineage.md",
    ]:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
