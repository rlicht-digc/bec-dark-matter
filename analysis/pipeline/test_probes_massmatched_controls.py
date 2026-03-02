#!/usr/bin/env python3
"""
PROBES Mass-Matched Controls — Phase peak and xi-organizing tests.

Matches SPARC and TNG logMbar distributions to PROBES, then reruns:
  (a) phase peak profile test (variance/scatter feature)
  (b) xi-organizing stacked variance-vs-X test
on those matched subsets.  Optionally builds 1:1 triads.

Outputs (to --out-dir):
  summary_probes_massmatched.json
  probes_mass_bins.csv
  fig_mass_distributions.png
  fig_phase_profiles_matched.png
  fig_xi_profiles_matched.png
  report_probes_massmatched.md
  run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")
plt.style.use("default")

# ── Physical constants ────────────────────────────────────────────────
G_SI = 6.674e-11       # m^3 kg^-1 s^-2
MSUN = 1.989e30         # kg
KPC  = 3.086e19         # m
LOG_G_DAGGER = -9.921   # log10(1.2e-10)
G_DAGGER = 1.2e-10      # m/s^2  (default)
ARCSEC_RAD = 4.8481e-6  # radians per arcsec

BIN_WIDTH_PHASE = 0.25
BIN_EDGES_PHASE = np.arange(-13.5, -8.0 + 1e-12, BIN_WIDTH_PHASE)
BIN_CENTERS_PHASE = BIN_EDGES_PHASE[:-1] + 0.5 * BIN_WIDTH_PHASE


# =====================================================================
# Reusable functions (from run_referee_required_tests.py / test6)
# =====================================================================

def rar_bec(log_gbar: np.ndarray, log_gd: float = LOG_G_DAGGER) -> np.ndarray:
    gbar = 10.0 ** log_gbar
    gd   = 10.0 ** log_gd
    x    = np.sqrt(np.maximum(gbar / gd, 1e-300))
    denom = np.maximum(1.0 - np.exp(-x), 1e-300)
    return np.log10(gbar / denom)


def healing_length_kpc(M_total_Msun: np.ndarray,
                       g_dagger: float = G_DAGGER) -> np.ndarray:
    return np.sqrt(G_SI * M_total_Msun * MSUN / g_dagger) / KPC


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
    for cand in ("SubhaloID", "subhalo_id", "galaxy_id", "galaxy", "id"):
        if cand in c:
            return cand
    return None


# ── Data loaders ──────────────────────────────────────────────────────

def load_sparc_points(root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = root / "analysis/results/rar_points_unified.csv"
    df = pd.read_csv(p)
    df = df[df["source"] == "SPARC"].copy()
    for col in ("galaxy", "log_gbar", "log_gobs", "R_kpc"):
        if col not in df.columns:
            raise RuntimeError(f"Missing SPARC column: {col}")
    df = df[
        np.isfinite(df["log_gbar"])
        & np.isfinite(df["log_gobs"])
        & np.isfinite(df["R_kpc"])
    ].copy()

    recomputed = df["log_gobs"].to_numpy() - rar_bec(df["log_gbar"].to_numpy())
    has_log_res = "log_res" in df.columns
    if has_log_res:
        diff = np.abs(df["log_res"].to_numpy() - recomputed)
        use_existing = float(np.nanmedian(diff)) <= 0.01
    else:
        use_existing = False

    df["log_res_use"] = df["log_res"].astype(float) if use_existing else recomputed
    meta = {
        "file": str(p),
        "n_points": int(len(df)),
        "n_galaxies": int(df["galaxy"].nunique()),
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
    for c in ("log_gbar", "log_gobs"):
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} in {path}")

    df = df.rename(columns={id_col: "id"}).copy()
    df = df[
        np.isfinite(df["log_gbar"])
        & np.isfinite(df["log_gobs"])
        & np.isfinite(df["r_kpc"])
    ].copy()
    df = df[
        (df["log_gbar"] > -20.0) & (df["log_gobs"] > -20.0) & (df["r_kpc"] > 0)
    ].copy()
    df["log_res_use"] = df["log_gobs"].to_numpy() - rar_bec(df["log_gbar"].to_numpy())
    return df


def discover_tng_points(root: Path) -> Optional[Path]:
    candidates = [
        root / "datasets/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/rar_points.parquet",
        root / "datasets/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points_CLEAN.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── PROBES loader ─────────────────────────────────────────────────────

def fold_rotation_curve(R_arcsec: np.ndarray,
                        V_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fold a two-sided RC into one-sided: |R|, |V|."""
    return np.abs(R_arcsec), np.abs(V_raw)


def load_probes_points(
    probes_dir: Path,
    min_points: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load PROBES galaxies and convert to RAR points (log_gbar, log_gobs,
    R_kpc, log_res_use) using the same strategy as
    test_rar_tightness_probes.py.

    Returns a DataFrame with columns:
        galaxy, log_gbar, log_gobs, R_kpc, log_res_use
    and a metadata dict.
    """
    profiles_dir = probes_dir / "profiles" / "profiles"

    # ── Main table ────────────────────────────────────────────────────
    mt = pd.read_csv(probes_dir / "main_table.csv", skiprows=1)

    # ── Structural parameters (only needed columns) ──────────────────
    needed_cols = [
        "name", "Mstar|Rlast:rc", "Mstar:E|Rlast:rc",
        "inclination|Rlast:rc", "inclination:E|Rlast:rc",
        "physR|Rlast:rc", "physR|Rp50:r",
    ]
    header_df = pd.read_csv(
        probes_dir / "structural_parameters.csv", skiprows=1, nrows=0,
    )
    usecols = [c for c in needed_cols if c in header_df.columns]
    sp = pd.read_csv(
        probes_dir / "structural_parameters.csv", skiprows=1, usecols=usecols,
    )
    sp_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in sp.iterrows():
        sp_dict[row["name"]] = row.to_dict()

    # ── Build RAR points per galaxy ──────────────────────────────────
    all_rows: List[Dict[str, Any]] = []
    n_loaded = 0
    n_skip_struct = 0
    n_skip_inc = 0
    n_skip_pts = 0
    n_skip_norc = 0
    n_skip_outlier = 0

    for _, mt_row in mt.iterrows():
        name = mt_row["name"]

        # Structural parameters
        if name not in sp_dict:
            n_skip_struct += 1
            continue
        sp_row = sp_dict[name]
        Mstar_total = sp_row.get("Mstar|Rlast:rc", np.nan)
        ba_ratio = sp_row.get("inclination|Rlast:rc", np.nan)
        R_phys_last = sp_row.get("physR|Rlast:rc", np.nan)

        if not np.isfinite(Mstar_total) or Mstar_total <= 0:
            n_skip_struct += 1
            continue
        if not np.isfinite(ba_ratio) or ba_ratio <= 0:
            n_skip_struct += 1
            continue

        # Inclination from b/a
        ba_clamped = min(max(ba_ratio, 0.1), 1.0)
        inc_rad = np.arccos(ba_clamped)
        inc_deg = np.degrees(inc_rad)
        if inc_deg < 30 or inc_deg > 85:
            n_skip_inc += 1
            continue

        # Load RC
        rc_path = profiles_dir / f"{name}_rc.prof"
        if not rc_path.exists():
            n_skip_norc += 1
            continue
        try:
            rc_lines = rc_path.read_text().splitlines()
            data = []
            for line in rc_lines[2:]:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    data.append((float(parts[0]), float(parts[1]), float(parts[2])))
            if len(data) == 0:
                n_skip_norc += 1
                continue
            arr = np.array(data)
        except Exception:
            n_skip_norc += 1
            continue

        R_arcsec = arr[:, 0]
        V_raw = arr[:, 1]

        # Fold two-sided RC
        R_fold, V_fold = fold_rotation_curve(R_arcsec, V_raw)

        # Convert R from arcsec to kpc
        dist_mpc = mt_row.get("distance", np.nan)
        if not np.isfinite(dist_mpc) or dist_mpc <= 0:
            n_skip_struct += 1
            continue
        R_kpc = R_fold * ARCSEC_RAD * dist_mpc * 1e3

        # Correct velocity for inclination
        sin_i = np.sin(inc_rad)
        V_rot = V_fold / max(sin_i, 0.3)

        # Compute gobs
        valid = (R_kpc > 0.1) & (V_rot > 5) & np.isfinite(V_rot) & np.isfinite(R_kpc)
        if np.sum(valid) < min_points:
            n_skip_pts += 1
            continue

        R_valid = R_kpc[valid]
        V_valid = V_rot[valid]
        gobs_SI = (V_valid * 1e3) ** 2 / (R_valid * KPC)

        # Compute gbar from total stellar mass + exponential disk
        R50 = sp_row.get("physR|Rp50:r", np.nan)
        if not np.isfinite(R50) or R50 <= 0:
            R50 = R_phys_last / 3.0 if np.isfinite(R_phys_last) and R_phys_last > 0 else 5.0
        Rd = R50 / 1.678

        x = R_valid / Rd
        M_enc_frac = 1.0 - (1.0 + x) * np.exp(-x)
        M_enc = Mstar_total * MSUN * M_enc_frac
        gbar_SI = G_SI * M_enc / (R_valid * KPC) ** 2

        # Validity check
        valid2 = (gbar_SI > 1e-15) & (gobs_SI > 1e-15)
        if np.sum(valid2) < min_points:
            n_skip_pts += 1
            continue

        log_gbar = np.log10(gbar_SI[valid2])
        log_gobs = np.log10(gobs_SI[valid2])
        log_gobs_rar = rar_bec(log_gbar)
        log_res = log_gobs - log_gobs_rar

        # Outlier filter
        if np.abs(np.median(log_res)) > 1.0:
            n_skip_outlier += 1
            continue

        R_out = R_valid[valid2]

        for i in range(len(log_gbar)):
            all_rows.append({
                "galaxy": name,
                "log_gbar": float(log_gbar[i]),
                "log_gobs": float(log_gobs[i]),
                "R_kpc": float(R_out[i]),
                "log_res_use": float(log_res[i]),
            })
        n_loaded += 1

    df = pd.DataFrame(all_rows)
    meta = {
        "n_galaxies": n_loaded,
        "n_points": len(df),
        "n_skip_struct": n_skip_struct,
        "n_skip_inc": n_skip_inc,
        "n_skip_pts": n_skip_pts,
        "n_skip_norc": n_skip_norc,
        "n_skip_outlier": n_skip_outlier,
    }
    return df, meta


# ── Galaxy mass table ─────────────────────────────────────────────────

def compute_galaxy_mass_table(
    df: pd.DataFrame, id_col: str, r_col: str, min_points: int = 1,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gid, g in df.groupby(id_col, sort=False):
        g2 = g.sort_values(r_col)
        n = int(len(g2))
        if n < min_points:
            continue
        r_out = float(g2[r_col].iloc[-1])
        lgb_out = float(g2["log_gbar"].iloc[-1])
        lgo_out = float(g2["log_gobs"].iloc[-1])
        M_bar = (10.0 ** lgb_out) * (r_out * KPC) ** 2 / G_SI / MSUN
        M_dyn = (10.0 ** lgo_out) * (r_out * KPC) ** 2 / G_SI / MSUN
        if (
            not np.isfinite(M_bar) or M_bar <= 0
            or not np.isfinite(M_dyn) or M_dyn <= 0
        ):
            continue
        rows.append({
            "id": gid,
            "n_points": n,
            "R_out_kpc": r_out,
            "log_gbar_out": lgb_out,
            "log_gobs_out": lgo_out,
            "log_Mb": float(np.log10(M_bar)),
            "log_Mdyn": float(np.log10(M_dyn)),
            "xi_kpc": float(healing_length_kpc(np.array([M_dyn]))[0]),
        })
    return pd.DataFrame(rows)


# ── Distribution matching ─────────────────────────────────────────────

def distribution_match_to_target(
    target_logMb: np.ndarray,
    pool_ids: np.ndarray,
    pool_logMb: np.ndarray,
    bin_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Downsample pool to match the logMbar distribution of target.
    Returns array of selected pool IDs.
    """
    lo = min(target_logMb.min(), pool_logMb.min())
    hi = max(target_logMb.max(), pool_logMb.max())
    edges = np.arange(np.floor(lo * 10) / 10, hi + bin_width, bin_width)

    target_hist, _ = np.histogram(target_logMb, bins=edges)
    pool_hist, _ = np.histogram(pool_logMb, bins=edges)
    pool_bin_idx = np.digitize(pool_logMb, edges) - 1

    selected: List[Any] = []
    for b in range(len(edges) - 1):
        n_target = target_hist[b]
        if n_target == 0:
            continue
        in_bin = np.where(pool_bin_idx == b)[0]
        n_pool = len(in_bin)
        if n_pool == 0:
            continue
        n_pick = min(n_target, n_pool)
        chosen = rng.choice(in_bin, size=n_pick, replace=False)
        selected.extend(pool_ids[chosen].tolist())

    return np.array(selected)


# ── Phase diagram functions ───────────────────────────────────────────

class BinPrep:
    __slots__ = ("bin_idx", "counts", "valid_bins", "centers_valid")
    def __init__(self, bin_idx, counts, valid_bins, centers_valid):
        self.bin_idx = bin_idx
        self.counts = counts
        self.valid_bins = valid_bins
        self.centers_valid = centers_valid


def prepare_binning(log_gbar: np.ndarray, min_points: int = 10) -> BinPrep:
    idx = np.digitize(log_gbar, BIN_EDGES_PHASE) - 1
    good = (idx >= 0) & (idx < len(BIN_CENTERS_PHASE))
    idx2 = np.full_like(idx, -1)
    idx2[good] = idx[good]
    counts = np.bincount(idx2[good], minlength=len(BIN_CENTERS_PHASE))
    valid_bins = np.where(counts >= min_points)[0]
    return BinPrep(
        bin_idx=idx2, counts=counts,
        valid_bins=valid_bins,
        centers_valid=BIN_CENTERS_PHASE[valid_bins],
    )


def variance_profile_from_prebinned(
    residuals: np.ndarray, prep: BinPrep,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    good = prep.bin_idx >= 0
    idx = prep.bin_idx[good]
    r = residuals[good]
    n_bins = len(BIN_CENTERS_PHASE)
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
    neg = np.minimum(yhat, 0.0)
    penalty = 1e7 * float(np.sum(neg * neg))
    yhat = np.where(yhat <= 1e-12, 1e-12, yhat)
    sig2 = np.maximum(yerr * yerr, 1e-18)
    base = 0.5 * float(np.sum(((y - yhat) ** 2) / sig2 + np.log(2.0 * np.pi * sig2)))
    return base + penalty


def fit_m1_linear(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                  k_for_aic: int = 3) -> Dict[str, Any]:
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
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 30, maxiter: int = 2000,
) -> Dict[str, Any]:
    bounds = [
        (1e-8, 5.0), (-2.0, 2.0),
        (1e-6, 5.0), (-12.0, -8.0), (0.05, 2.0),
        (-5.0, -1e-6), (-12.0, -8.0), (0.05, 2.0),
        (-5.0, 5.0), (-12.0, -8.0), (0.01, 1.0),
    ]
    def obj(p):
        return nll_from_model(y, model_edge(p, x), yerr)

    y_med = float(np.median(y))
    y_span = float(max(np.max(y) - np.min(y), 0.01))
    starts = [np.array([
        max(y_med, 1e-4), 0.0, 0.5 * y_span, LOG_G_DAGGER, 0.35,
        -0.25 * y_span, LOG_G_DAGGER - 0.3, 0.25, 0.0, -9.0, 0.15,
    ], dtype=float)]
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    for _ in range(max(0, n_starts - 1)):
        p = lb + rng.random(len(bounds)) * (ub - lb)
        y0 = model_edge(p, x)
        if np.min(y0) <= 1e-5:
            p[0] += float(1e-3 - np.min(y0))
        starts.append(p)

    best = None
    for p0 in starts:
        try:
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": int(maxiter)})
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
        "ok": True, "model": "M2b_edge",
        "params": np.asarray(best.x, dtype=float),
        "nll": nll,
        "aic": float(2 * 11 + 2 * nll),
        "mu_peak": mu_peak, "mup_raw": mup, "mud_raw": mud,
    }


def fit_peak_dip_fallback(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 15, maxiter: int = 1500,
) -> Dict[str, Any]:
    bounds = [
        (1e-8, 5.0), (-2.0, 2.0),
        (1e-6, 5.0), (-12.0, -8.0), (0.05, 2.0),
        (-5.0, -1e-6), (-12.0, -8.0), (0.05, 2.0),
    ]
    def obj(p):
        return nll_from_model(y, model_peak_dip(p, x), yerr)

    y_med = float(np.median(y))
    y_span = float(max(np.max(y) - np.min(y), 0.01))
    starts = [np.array([
        max(y_med, 1e-4), 0.0, 0.4 * y_span, LOG_G_DAGGER, 0.35,
        -0.2 * y_span, LOG_G_DAGGER - 0.25, 0.25,
    ], dtype=float)]
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
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": int(maxiter)})
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
        "ok": True, "model": "M2b_peak_dip_fallback",
        "params": np.asarray(best.x, dtype=float),
        "nll": nll,
        "aic": float(2 * 9 + 2 * nll),
        "mu_peak": mu_peak, "mup_raw": mup, "mud_raw": mud,
    }


def fit_phase_profile_models(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts_edge: int = 30,
    for_null: bool = False,
) -> Dict[str, Any]:
    m1 = fit_m1_linear(x, y, yerr, k_for_aic=3)
    starts = max(5, n_starts_edge) if not for_null else max(5, min(10, n_starts_edge))
    maxiter = 2500 if not for_null else 900
    edge = fit_edge_model(x, y, yerr, rng=rng, n_starts=starts, maxiter=maxiter)
    used_fallback = False

    if not edge.get("ok", False):
        edge = fit_peak_dip_fallback(
            x, y, yerr, rng=rng,
            n_starts=max(6, starts),
            maxiter=1200 if for_null else 1800,
        )
        used_fallback = True

    if not edge.get("ok", False):
        return {"ok": False, "m1": m1, "edge": edge, "used_fallback": True}

    return {
        "ok": True, "m1": m1, "edge": edge,
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


def phase_fit_from_points(
    x: np.ndarray, r: np.ndarray, rng: np.random.Generator,
    n_starts_edge: int = 30, for_null: bool = False,
) -> Dict[str, Any]:
    prep = prepare_binning(x, min_points=10)
    xb, vb, eb = variance_profile_from_prebinned(r, prep)
    fit = fit_phase_profile_models(
        xb, vb, eb, rng=rng,
        n_starts_edge=n_starts_edge, for_null=for_null,
    )
    return {"x_bins": xb, "var_bins": vb, "var_err": eb,
            "n_bins_used": int(len(xb)), "fit": fit}


# ── Xi-organizing functions ───────────────────────────────────────────

def per_galaxy_xi_payload(
    df: pd.DataFrame, id_col: str, r_col: str, res_col: str,
    g_dagger: float = G_DAGGER,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for gid, g in df.groupby(id_col, sort=False):
        g2 = g.sort_values(r_col)
        if len(g2) < 8:
            continue
        r   = g2[r_col].to_numpy(dtype=float)
        res = g2[res_col].to_numpy(dtype=float)
        lgo = g2["log_gobs"].to_numpy(dtype=float)
        if (
            not np.all(np.isfinite(r))
            or not np.all(np.isfinite(res))
            or not np.all(np.isfinite(lgo))
        ):
            continue
        j = int(np.argmax(r))
        M_dyn = (10.0 ** lgo[j]) * (r[j] * KPC) ** 2 / G_SI / MSUN
        if not np.isfinite(M_dyn) or M_dyn <= 0:
            continue
        xi = float(healing_length_kpc(np.array([M_dyn]), g_dagger=g_dagger)[0])
        if not np.isfinite(xi) or xi <= 0:
            continue
        x  = r / xi
        lx = np.log10(np.maximum(x, 1e-12))
        payload.append({
            "id": gid, "logX": lx, "res": res,
            "xi_kpc": xi, "n_points": len(r),
        })
    return payload


def stacked_variance_profile(
    logx_list: List[np.ndarray], res_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    edges  = np.linspace(-2.0, 1.5, 9)
    n_bins = len(edges) - 1
    vmat   = np.full((len(logx_list), n_bins), np.nan, dtype=float)
    for i, (lx, rr) in enumerate(zip(logx_list, res_list)):
        idx = np.digitize(lx, edges) - 1
        for b in range(n_bins):
            m = idx == b
            if int(m.sum()) >= 2:
                vmat[i, b] = float(np.var(rr[m], ddof=1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, vmat


def concentration_from_profile(
    centers: np.ndarray, mean_var: np.ndarray,
) -> float:
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


def xi_permutation_null(
    payload: List[Dict[str, Any]],
    centers: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 1000,
) -> np.ndarray:
    n_g = len(payload)
    if n_g == 0:
        return np.array([], dtype=float)

    cnull = np.full(n_perm, np.nan, dtype=float)
    logx_list = [p["logX"] for p in payload]
    res_list  = [p["res"]  for p in payload]
    for i in range(n_perm):
        rr = [r[rng.permutation(len(r))] for r in res_list]
        _, vmat = stacked_variance_profile(logx_list, rr)
        mean_prof = np.nanmean(vmat, axis=0)
        cnull[i] = concentration_from_profile(centers, mean_prof)
        if (i + 1) % 200 == 0:
            print(f"      [xi perm] {i+1}/{n_perm}")
    return cnull


# ── 1:1 Triad matching ───────────────────────────────────────────────

def build_triads(
    probes_mass: pd.DataFrame,
    sparc_mass: pd.DataFrame,
    tng_mass: pd.DataFrame,
    caliper: float,
) -> pd.DataFrame:
    """For each PROBES galaxy, find nearest SPARC and TNG by logMbar."""
    s_vals = sparc_mass["log_Mb"].to_numpy(dtype=float)
    t_vals = tng_mass["log_Mb"].to_numpy(dtype=float)
    s_ids  = sparc_mass["id"].to_numpy()
    t_ids  = tng_mass["id"].to_numpy()
    used_s = np.zeros(len(s_vals), dtype=bool)
    used_t = np.zeros(len(t_vals), dtype=bool)

    triads: List[Dict[str, Any]] = []
    for _, pr in probes_mass.sort_values("log_Mb").iterrows():
        pm = float(pr["log_Mb"])

        ds = np.abs(s_vals - pm)
        ds[used_s] = np.inf
        js = int(np.argmin(ds))
        if not np.isfinite(ds[js]) or ds[js] > caliper:
            continue

        dt = np.abs(t_vals - pm)
        dt[used_t] = np.inf
        jt = int(np.argmin(dt))
        if not np.isfinite(dt[jt]) or dt[jt] > caliper:
            continue

        used_s[js] = True
        used_t[jt] = True
        triads.append({
            "probes_id": pr["id"],
            "sparc_id": s_ids[js],
            "tng_id": t_ids[jt],
            "probes_logMb": pm,
            "sparc_logMb": float(s_vals[js]),
            "tng_logMb": float(t_vals[jt]),
        })
    return pd.DataFrame(triads)


# =====================================================================
# Main analysis
# =====================================================================

def run_analysis(
    root: Path,
    out_dir: Path,
    g_dagger: float,
    min_points: int,
    bin_width_dex: float,
    caliper_dex: float,
    n_perm: int,
    seed: int,
    do_triads: bool,
) -> Dict[str, Any]:
    print("=" * 72)
    print("PROBES Mass-Matched Controls: Phase + Xi tests")
    print("=" * 72)

    t0 = time.time()
    rng = np.random.default_rng(seed)

    # ── STEP 0: Load datasets ────────────────────────────────────────
    print("\n[0] Loading datasets...")

    # SPARC
    sparc_df, sparc_meta = load_sparc_points(root)
    print(f"  SPARC: {sparc_meta['n_points']} points, "
          f"{sparc_meta['n_galaxies']} galaxies")

    # TNG
    tng_path = discover_tng_points(root)
    if tng_path is None:
        summary = {"status": "BLOCKED", "reason": "TNG per-point data not found"}
        write_json(out_dir / "summary_probes_massmatched.json", summary)
        return summary
    tng_df = load_tng_points(tng_path)
    print(f"  TNG:   {len(tng_df)} points, {tng_df['id'].nunique()} galaxies "
          f"(from {tng_path.name})")

    # PROBES
    probes_dir = root / "raw_data/observational/probes"
    if not probes_dir.exists():
        # Try alternate location
        probes_dir = root / "analysis/pipeline/data/probes"
    if not probes_dir.exists():
        summary = {"status": "BLOCKED", "reason": "PROBES directory not found"}
        write_json(out_dir / "summary_probes_massmatched.json", summary)
        return summary

    print(f"  Loading PROBES from {probes_dir} (min_points={min_points})...")
    probes_df, probes_meta = load_probes_points(probes_dir, min_points=min_points)
    print(f"  PROBES: {probes_meta['n_points']} points, "
          f"{probes_meta['n_galaxies']} galaxies")
    print(f"    Skipped: struct={probes_meta['n_skip_struct']}, "
          f"inc={probes_meta['n_skip_inc']}, pts={probes_meta['n_skip_pts']}, "
          f"norc={probes_meta['n_skip_norc']}, outlier={probes_meta['n_skip_outlier']}")

    if probes_meta["n_galaxies"] < 10:
        summary = {"status": "BLOCKED", "reason": "Too few PROBES galaxies loaded"}
        write_json(out_dir / "summary_probes_massmatched.json", summary)
        return summary

    # ── STEP 1: Build per-galaxy mass tables ─────────────────────────
    print("\n[1] Building per-galaxy mass tables...")

    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sp_mass = compute_galaxy_mass_table(sp, "id", "r_kpc", min_points=5)

    tng_mass = compute_galaxy_mass_table(tng_df, "id", "r_kpc", min_points=5)

    probes_renamed = probes_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    probes_mass = compute_galaxy_mass_table(probes_renamed, "id", "r_kpc", min_points=min_points)

    print(f"  SPARC mass table:  {len(sp_mass)} galaxies")
    print(f"  TNG mass table:    {len(tng_mass)} galaxies")
    print(f"  PROBES mass table: {len(probes_mass)} galaxies")

    N_probes = len(probes_mass)
    N_probes_points = int(probes_df["galaxy"].isin(set(probes_mass["id"])).sum())

    # ── STEP 2: Distribution-match SPARC and TNG to PROBES ───────────
    print(f"\n[2] Distribution-matching to PROBES (bin_width={bin_width_dex} dex)...")

    probes_logMb = probes_mass["log_Mb"].to_numpy()

    matched_sparc_ids = distribution_match_to_target(
        probes_logMb, sp_mass["id"].to_numpy(), sp_mass["log_Mb"].to_numpy(),
        bin_width_dex, rng,
    )
    matched_tng_ids = distribution_match_to_target(
        probes_logMb, tng_mass["id"].to_numpy(), tng_mass["log_Mb"].to_numpy(),
        bin_width_dex, rng,
    )

    N_sparc_matched = len(matched_sparc_ids)
    N_tng_matched = len(matched_tng_ids)
    print(f"  Matched SPARC: {N_sparc_matched} galaxies")
    print(f"  Matched TNG:   {N_tng_matched} galaxies")

    if N_sparc_matched < 5 or N_tng_matched < 5:
        summary = {
            "status": "BLOCKED",
            "reason": "Too few galaxies after distribution matching",
            "N_sparc_matched": N_sparc_matched,
            "N_tng_matched": N_tng_matched,
        }
        write_json(out_dir / "summary_probes_massmatched.json", summary)
        return summary

    # KS tests
    sp_matched_logMb = sp_mass[sp_mass["id"].isin(set(matched_sparc_ids))]["log_Mb"].to_numpy()
    tng_matched_logMb = tng_mass[tng_mass["id"].isin(set(matched_tng_ids))]["log_Mb"].to_numpy()

    ks_sp = ks_2samp(probes_logMb, sp_matched_logMb)
    ks_tng = ks_2samp(probes_logMb, tng_matched_logMb)
    print(f"  KS p (SPARC vs PROBES): {ks_sp.pvalue:.4f}")
    print(f"  KS p (TNG vs PROBES):   {ks_tng.pvalue:.4f}")

    # Filter point DataFrames to matched galaxies
    sp_matched_df = sp[sp["id"].isin(set(matched_sparc_ids))].copy()
    tng_matched_df = tng_df[tng_df["id"].isin(set(matched_tng_ids))].copy()

    # Mass bins CSV
    lo_edge = min(probes_logMb.min(), sp_matched_logMb.min(), tng_matched_logMb.min())
    hi_edge = max(probes_logMb.max(), sp_matched_logMb.max(), tng_matched_logMb.max())
    mb_edges = np.arange(np.floor(lo_edge * 10) / 10, hi_edge + bin_width_dex, bin_width_dex)
    probes_h, _ = np.histogram(probes_logMb, bins=mb_edges)
    sparc_h, _  = np.histogram(sp_matched_logMb, bins=mb_edges)
    tng_h, _    = np.histogram(tng_matched_logMb, bins=mb_edges)
    bin_df = pd.DataFrame({
        "bin_lo": mb_edges[:-1],
        "bin_hi": mb_edges[1:],
        "n_probes": probes_h,
        "n_sparc_matched": sparc_h,
        "n_tng_matched": tng_h,
    })
    bin_df.to_csv(out_dir / "probes_mass_bins.csv", index=False)

    # ── STEP 3: Phase peak test on matched subsets ───────────────────
    print("\n[3] Phase peak test on matched subsets...")

    fit_probes = phase_fit_from_points(
        probes_df["log_gbar"].to_numpy(dtype=float),
        probes_df["log_res_use"].to_numpy(dtype=float),
        rng=rng, n_starts_edge=30,
    )
    fit_sparc_m = phase_fit_from_points(
        sp_matched_df["log_gbar"].to_numpy(dtype=float),
        sp_matched_df["log_res_use"].to_numpy(dtype=float),
        rng=rng, n_starts_edge=30,
    )
    fit_tng_m = phase_fit_from_points(
        tng_matched_df["log_gbar"].to_numpy(dtype=float),
        tng_matched_df["log_res_use"].to_numpy(dtype=float),
        rng=rng, n_starts_edge=30,
    )

    phase_results: Dict[str, Any] = {"_provenance": {
        "x_col": "log_gbar", "log_base": "log10", "units": "m/s^2",
        "peak_definition": "M2b_edge Gaussian center nearest log10(g_dagger)",
        "log_g_dagger": LOG_G_DAGGER,
    }}
    for label, fit_obj in [("PROBES", fit_probes),
                           ("SPARC_matched", fit_sparc_m),
                           ("TNG_matched", fit_tng_m)]:
        if fit_obj["fit"]["ok"]:
            edge = fit_obj["fit"]["edge"]
            phase_results[label] = {
                "mu_peak": float(fit_obj["fit"]["mu_peak"]),
                "daic": float(fit_obj["fit"]["daic"]),
                "aic_m1": float(fit_obj["fit"]["aic_m1"]),
                "aic_edge": float(fit_obj["fit"]["aic_edge"]),
                "n_bins_used": fit_obj["n_bins_used"],
                "model": edge["model"],
                "mup_raw": edge.get("mup_raw"),
                "mud_raw": edge.get("mud_raw"),
            }
            print(f"  {label:16s}: μ_peak={fit_obj['fit']['mu_peak']:.3f}, "
                  f"ΔAIC={fit_obj['fit']['daic']:.1f}"
                  f"  (mup={edge.get('mup_raw', 0):.3f}, mud={edge.get('mud_raw', 0):.3f})")
        else:
            phase_results[label] = {"mu_peak": None, "daic": None, "fit_ok": False}
            print(f"  {label:16s}: FIT FAILED")

    # ── STEP 4: Xi-organizing on matched subsets ─────────────────────
    print(f"\n[4] Xi-organizing ({n_perm} permutations)...")

    xi_results: Dict[str, Any] = {}
    for label, df_sub, id_col, r_col in [
        ("PROBES", probes_renamed, "id", "r_kpc"),
        ("SPARC_matched", sp_matched_df, "id", "r_kpc"),
        ("TNG_matched", tng_matched_df, "id", "r_kpc"),
    ]:
        print(f"  [{label}]")
        payload = per_galaxy_xi_payload(
            df_sub, id_col=id_col, r_col=r_col,
            res_col="log_res_use", g_dagger=g_dagger,
        )
        if len(payload) < 3:
            xi_results[label] = {
                "n_galaxies": len(payload),
                "C": None, "perm_p": None,
                "reason": "too few galaxies",
            }
            print(f"    Too few galaxies ({len(payload)}) for xi test")
            continue

        logx_list = [p["logX"] for p in payload]
        res_list  = [p["res"]  for p in payload]
        centers, vmat = stacked_variance_profile(logx_list, res_list)
        mean_prof = np.nanmean(vmat, axis=0)
        C_real = concentration_from_profile(centers, mean_prof)

        # Bootstrap CIs
        n_boot = 500
        n_g = len(payload)
        boot_C = np.full(n_boot, np.nan)
        boot_prof = np.full((n_boot, len(centers)), np.nan)
        for i in range(n_boot):
            idx = rng.integers(0, n_g, size=n_g)
            bp = np.nanmean(vmat[idx], axis=0)
            boot_prof[i] = bp
            boot_C[i] = concentration_from_profile(centers, bp)
        ci_C = [float(np.nanpercentile(boot_C, 2.5)),
                float(np.nanpercentile(boot_C, 97.5))]
        prof_lo = np.nanpercentile(boot_prof, 2.5, axis=0)
        prof_hi = np.nanpercentile(boot_prof, 97.5, axis=0)

        # Permutation null
        cnull = xi_permutation_null(payload, centers, rng, n_perm=n_perm)
        p_c = float(np.mean(np.where(np.isfinite(cnull), cnull >= C_real, False)))

        xi_results[label] = {
            "n_galaxies": n_g,
            "C": float(C_real),
            "C_ci95": ci_C,
            "perm_p": p_c,
            "centers": centers.tolist(),
            "mean_profile": mean_prof.tolist(),
            "profile_ci_lo": prof_lo.tolist(),
            "profile_ci_hi": prof_hi.tolist(),
        }
        print(f"    N={n_g}, C={C_real:.4f}, p={p_c:.4f}")

    # ── STEP 5: Optional triads ──────────────────────────────────────
    triad_results: Optional[Dict[str, Any]] = None
    if do_triads:
        print("\n[5] Building 1:1 triads...")
        triads = build_triads(probes_mass, sp_mass, tng_mass, caliper=caliper_dex)
        print(f"  Triads: {len(triads)}")

        if len(triads) >= 5:
            triad_probes_ids = set(triads["probes_id"])
            triad_sparc_ids = set(triads["sparc_id"])
            triad_tng_ids = set(triads["tng_id"])

            triad_results = {"N_triads": len(triads)}

            # Phase on triads
            for tlabel, tdf, tid_set in [
                ("PROBES_triad", probes_renamed, triad_probes_ids),
                ("SPARC_triad", sp, triad_sparc_ids),
                ("TNG_triad", tng_df, triad_tng_ids),
            ]:
                sub = tdf[tdf["id"].isin(tid_set)].copy()
                if len(sub) < 50:
                    triad_results[f"phase_{tlabel}"] = {"mu_peak": None, "n_points": len(sub)}
                    continue
                tfit = phase_fit_from_points(
                    sub["log_gbar"].to_numpy(dtype=float),
                    sub["log_res_use"].to_numpy(dtype=float),
                    rng=rng, n_starts_edge=20,
                )
                if tfit["fit"]["ok"]:
                    tedge = tfit["fit"]["edge"]
                    triad_results[f"phase_{tlabel}"] = {
                        "mu_peak": float(tfit["fit"]["mu_peak"]),
                        "daic": float(tfit["fit"]["daic"]),
                        "mup_raw": tedge.get("mup_raw"),
                        "mud_raw": tedge.get("mud_raw"),
                    }
                    print(f"    {tlabel}: μ={tfit['fit']['mu_peak']:.3f}, "
                          f"ΔAIC={tfit['fit']['daic']:.1f}"
                          f"  (mup={tedge.get('mup_raw', 0):.3f}, mud={tedge.get('mud_raw', 0):.3f})")
                else:
                    triad_results[f"phase_{tlabel}"] = {"mu_peak": None, "fit_ok": False}

            # Xi on triads
            for tlabel, tdf, tid_set in [
                ("PROBES_triad", probes_renamed, triad_probes_ids),
                ("SPARC_triad", sp, triad_sparc_ids),
                ("TNG_triad", tng_df, triad_tng_ids),
            ]:
                sub = tdf[tdf["id"].isin(tid_set)].copy()
                tpay = per_galaxy_xi_payload(
                    sub, "id", "r_kpc", "log_res_use", g_dagger=g_dagger,
                )
                if len(tpay) < 3:
                    triad_results[f"xi_{tlabel}"] = {"C": None, "n_galaxies": len(tpay)}
                    continue
                tlogx = [p["logX"] for p in tpay]
                tres  = [p["res"]  for p in tpay]
                tctrs, tvmat = stacked_variance_profile(tlogx, tres)
                tprof = np.nanmean(tvmat, axis=0)
                tC = concentration_from_profile(tctrs, tprof)
                tcnull = xi_permutation_null(tpay, tctrs, rng, n_perm=n_perm)
                tp = float(np.mean(np.where(np.isfinite(tcnull), tcnull >= tC, False)))
                triad_results[f"xi_{tlabel}"] = {
                    "C": float(tC), "perm_p": tp, "n_galaxies": len(tpay),
                }
                print(f"    {tlabel} xi: C={tC:.4f}, p={tp:.4f}")

            triads.to_csv(out_dir / "triads.csv", index=False)
        else:
            triad_results = {"N_triads": len(triads), "reason": "too few triads"}

    # ── Figures ───────────────────────────────────────────────────────
    print("\n[6] Generating figures...")

    # Fig 1: Mass distributions
    fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=180)
    mb_fine = np.arange(
        min(probes_logMb.min(), sp_matched_logMb.min(), tng_matched_logMb.min()) - 0.3,
        max(probes_logMb.max(), sp_matched_logMb.max(), tng_matched_logMb.max()) + 0.3,
        0.2,
    )
    ax1.hist(probes_logMb, bins=mb_fine, alpha=0.5, label=f"PROBES (N={N_probes})",
             color="#d62728", edgecolor="black", linewidth=0.5)
    ax1.hist(sp_matched_logMb, bins=mb_fine, alpha=0.5,
             label=f"SPARC matched (N={N_sparc_matched})",
             color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax1.hist(tng_matched_logMb, bins=mb_fine, alpha=0.5,
             label=f"TNG matched (N={N_tng_matched})",
             color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("log$_{10}$(M$_{\\rm bar}$ / M$_\\odot$)")
    ax1.set_ylabel("Count")
    ax1.set_title("Mass distributions after matching to PROBES")
    ax1.legend(fontsize=9)
    ax1.set_facecolor("white")
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig_mass_distributions.png", facecolor="white")
    plt.close(fig1)

    # Fig 2: Phase profiles
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), dpi=180)
    xgrid = np.linspace(BIN_CENTERS_PHASE.min(), BIN_CENTERS_PHASE.max(), 400)
    configs = [
        ("PROBES", fit_probes, "#d62728"),
        ("SPARC matched", fit_sparc_m, "#1f77b4"),
        ("TNG matched", fit_tng_m, "#2ca02c"),
    ]
    ymax = 0
    for _, fo, _ in configs:
        if fo["fit"]["ok"]:
            ymax = max(ymax, np.nanmax(fo["var_bins"] + fo["var_err"]))

    for ax, (label, fit_obj, color) in zip(axes2, configs):
        ax.errorbar(
            fit_obj["x_bins"], fit_obj["var_bins"], yerr=fit_obj["var_err"],
            fmt="o", color=color, ecolor=color, capsize=3, label="binned var",
        )
        if fit_obj["fit"]["ok"]:
            curve = eval_fit_curve(fit_obj["fit"]["edge"], xgrid)
            ax.plot(xgrid, curve, color="black", linewidth=1.5,
                    label=fit_obj["fit"]["edge"]["model"])
            mu = fit_obj["fit"]["mu_peak"]
            daic = fit_obj["fit"]["daic"]
            ax.set_title(f"{label}\nμ={mu:.3f}, ΔAIC={daic:.1f}")
        else:
            ax.set_title(f"{label}\nFIT FAILED")
        ax.axvline(LOG_G_DAGGER, color="red", linestyle="--", linewidth=1.2,
                    label="log g†")
        if ymax > 0:
            ax.set_ylim(0, 1.15 * ymax)
        ax.set_xlabel("log g$_{\\rm bar}$")
        ax.set_ylabel("variance of log residual")
        ax.legend(frameon=False, fontsize=7)
        ax.set_facecolor("white")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_phase_profiles_matched.png", facecolor="white")
    plt.close(fig2)

    # Fig 3: Xi profiles overlay
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), dpi=180)
    xi_colors = {"PROBES": "#d62728", "SPARC_matched": "#1f77b4",
                 "TNG_matched": "#2ca02c"}

    for ax, label in zip(axes3[:3], ["PROBES", "SPARC_matched", "TNG_matched"]):
        xr = xi_results.get(label, {})
        if xr.get("C") is not None:
            ctrs = np.array(xr["centers"])
            prof = np.array(xr["mean_profile"])
            x_plot = 10.0 ** ctrs
            ok = np.isfinite(prof)
            ax.plot(x_plot[ok], prof[ok], "o-", color=xi_colors[label])
            if "profile_ci_lo" in xr:
                lo = np.array(xr["profile_ci_lo"])
                hi = np.array(xr["profile_ci_hi"])
                ax.fill_between(x_plot, lo, hi, alpha=0.3, color=xi_colors[label])
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_title(f"{label}\nC={xr['C']:.4f}, p={xr['perm_p']:.4f}")
        else:
            ax.set_title(f"{label}\nNO DATA")
        ax.set_xlabel("X = R / ξ")
        ax.set_ylabel("stacked variance σ²")
        ax.set_facecolor("white")
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_xi_profiles_matched.png", facecolor="white")
    plt.close(fig3)

    # ── Summary JSON ──────────────────────────────────────────────────
    dt = time.time() - t0

    summary: Dict[str, Any] = {
        "test": "probes_massmatched_controls",
        "status": "OK",
        "seed": seed,
        "g_dagger": g_dagger,
        "min_points": min_points,
        "bin_width_dex": bin_width_dex,
        "caliper_dex": caliper_dex,
        "n_perm": n_perm,
        "N_probes": N_probes,
        "N_probes_points": N_probes_points,
        "N_sparc_matched": N_sparc_matched,
        "N_tng_matched": N_tng_matched,
        "ks_p_sparc_vs_probes": float(ks_sp.pvalue),
        "ks_p_tng_vs_probes": float(ks_tng.pvalue),
        "probes_logMb_range": [float(probes_logMb.min()), float(probes_logMb.max())],
        "phase": phase_results,
        "xi": xi_results,
        "triads": triad_results,
        "tng_file": str(tng_path),
        "probes_dir": str(probes_dir),
        "elapsed_seconds": round(dt, 1),
    }
    write_json(out_dir / "summary_probes_massmatched.json", summary)

    # ── Report ────────────────────────────────────────────────────────
    lines: List[str] = [
        "# PROBES Mass-Matched Controls",
        "",
        "## Purpose",
        "Match SPARC and TNG logMbar distributions to PROBES, then rerun the",
        "phase peak profile test and xi-organizing test on matched subsets.",
        "",
        "## Configuration",
        f"- g† = {g_dagger:.4e} m/s²",
        f"- min_points = {min_points}",
        f"- bin_width = {bin_width_dex} dex",
        f"- caliper = {caliper_dex} dex",
        f"- n_perm = {n_perm}",
        f"- seed = {seed}",
        f"- TNG file: `{tng_path}`",
        f"- PROBES dir: `{probes_dir}`",
        "",
        "## Sample Sizes",
        f"- PROBES: {N_probes} galaxies, {N_probes_points} points",
        f"- SPARC matched: {N_sparc_matched} galaxies",
        f"- TNG matched:   {N_tng_matched} galaxies",
        "",
        "## Mass Distribution KS Tests",
        f"- SPARC vs PROBES: p = {ks_sp.pvalue:.4f}",
        f"- TNG vs PROBES:   p = {ks_tng.pvalue:.4f}",
        "",
        "## Phase Peak Results",
        "| Dataset | μ_peak | ΔAIC | Model |",
        "|---------|--------|------|-------|",
    ]
    for label in ["PROBES", "SPARC_matched", "TNG_matched"]:
        pr = phase_results.get(label, {})
        mu = pr.get("mu_peak")
        daic = pr.get("daic")
        model = pr.get("model", "—")
        if mu is not None:
            lines.append(f"| {label} | {mu:.3f} | {daic:.1f} | {model} |")
        else:
            lines.append(f"| {label} | FAILED | — | — |")

    lines += [
        "",
        "## Xi-Organizing Results",
        "| Dataset | N galaxies | C | 95% CI | perm p |",
        "|---------|-----------|---|--------|--------|",
    ]
    for label in ["PROBES", "SPARC_matched", "TNG_matched"]:
        xr = xi_results.get(label, {})
        ng = xr.get("n_galaxies", 0)
        C = xr.get("C")
        ci = xr.get("C_ci95", [None, None])
        pp = xr.get("perm_p")
        if C is not None:
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "—"
            lines.append(f"| {label} | {ng} | {C:.4f} | {ci_str} | {pp:.4f} |")
        else:
            lines.append(f"| {label} | {ng} | — | — | — |")

    if triad_results is not None:
        nt = triad_results.get("N_triads", 0)
        lines += ["", f"## Triads: {nt}"]
        if nt >= 5:
            for k in ["PROBES_triad", "SPARC_triad", "TNG_triad"]:
                pk = triad_results.get(f"phase_{k}", {})
                xk = triad_results.get(f"xi_{k}", {})
                mu = pk.get("mu_peak")
                C = xk.get("C")
                pp = xk.get("perm_p")
                lines.append(f"- {k}: μ={mu}, C={C}, p={pp}")

    lines += [
        "",
        f"Elapsed: {dt:.1f}s",
    ]
    (out_dir / "report_probes_massmatched.md").write_text("\n".join(lines) + "\n")

    # ── Run metadata ──────────────────────────────────────────────────
    run_meta = {
        "script": "test_probes_massmatched_controls.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "args": {
            "project_root": str(root),
            "out_dir": str(out_dir),
            "g_dagger": g_dagger,
            "min_points": min_points,
            "bin_width_dex": bin_width_dex,
            "caliper_dex": caliper_dex,
            "n_perm": n_perm,
            "seed": seed,
            "triads": do_triads,
        },
        "elapsed_seconds": round(dt, 1),
    }
    write_json(out_dir / "run_metadata.json", run_meta)

    # ── Final print ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  N_probes:        {N_probes}")
    print(f"  N_sparc_matched: {N_sparc_matched}")
    print(f"  N_tng_matched:   {N_tng_matched}")
    print(f"  KS p (SPARC vs PROBES): {ks_sp.pvalue:.4f}")
    print(f"  KS p (TNG vs PROBES):   {ks_tng.pvalue:.4f}")
    print()
    print("  Phase peak μ:")
    for label in ["PROBES", "SPARC_matched", "TNG_matched"]:
        pr = phase_results.get(label, {})
        mu = pr.get("mu_peak")
        print(f"    {label:16s}: {mu}")
    print()
    print("  Xi concentration C (perm p):")
    for label in ["PROBES", "SPARC_matched", "TNG_matched"]:
        xr = xi_results.get(label, {})
        C = xr.get("C")
        pp = xr.get("perm_p")
        print(f"    {label:16s}: C={C}, p={pp}")
    print(f"\n  Output folder: {out_dir}")
    print(f"  Elapsed: {dt:.1f}s")

    return summary


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROBES mass-matched controls: phase peak + xi tests")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--g-dagger", type=float, default=1.2e-10)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--bin-width-dex", type=float, default=0.3)
    parser.add_argument("--caliper-dex", type=float, default=0.3)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--triads", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    root = Path(args.project_root).resolve() if args.project_root else default_root

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "outputs" / "probes_massmatched" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    run_analysis(
        root=root,
        out_dir=out_dir,
        g_dagger=args.g_dagger,
        min_points=args.min_points,
        bin_width_dex=args.bin_width_dex,
        caliper_dex=args.caliper_dex,
        n_perm=args.n_perm,
        seed=args.seed,
        do_triads=args.triads,
    )


if __name__ == "__main__":
    main()
