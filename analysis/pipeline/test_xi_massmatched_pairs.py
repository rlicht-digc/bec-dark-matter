#!/usr/bin/env python3
"""
Test 6 — Xi mass-matched pairs comparison.

Test 3 showed TNG has strong xi-organizing (C=1.549, p=0.000) but SPARC does
not (C=0.687, p=0.571).  This discrepancy may be confounded by the different
mass distributions of the two samples.  This test uses Test 2's mass-matching
procedure to build SPARC–TNG pairs of comparable baryon mass, then compares
per-galaxy xi-concentration on the *paired* sub-samples.

Outputs (to --out-dir):
  xi_pairs.csv
  summary_xi_massmatched.json
  fig_xi_pairs_scatter.png
  fig_xi_ratio_vs_mass.png
  fig_xi_matched_X_profiles.png
  report_xi_massmatched.md
  run_metadata.json
"""

from __future__ import annotations

# ── BLAS thread pinning (must come before numpy) ──────────────────────
import os
import sys

def _pin_blas(n: int) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)

# Default: pin to 1 so multiprocessing workers don't oversubscribe.
# Overridden by --blas-threads if provided early via sys.argv peek.
_blas_n = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--blas-threads" and _i + 1 < len(sys.argv):
        try:
            _blas_n = int(sys.argv[_i + 1])
        except ValueError:
            pass
_pin_blas(_blas_n)

# ── Standard / third-party imports ────────────────────────────────────
import argparse
import json
import multiprocessing as mp
import platform
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
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
plt.style.use("default")

# ── Physical constants ────────────────────────────────────────────────
G_SI = 6.674e-11      # m^3 kg^-1 s^-2
MSUN = 1.989e30        # kg
KPC  = 3.086e19        # m
LOG_G_DAGGER = -9.921  # log10(1.2e-10)
G_DAGGER = 1.2e-10     # m/s^2  (default)


# =====================================================================
# Reusable functions from run_referee_required_tests.py
# =====================================================================

def rar_bec(log_gbar: np.ndarray, log_gd: float = LOG_G_DAGGER) -> np.ndarray:
    gbar = 10.0 ** log_gbar
    gd   = 10.0 ** log_gd
    x    = np.sqrt(np.maximum(gbar / gd, 1e-300))
    denom = np.maximum(1.0 - np.exp(-x), 1e-300)
    return np.log10(gbar / denom)


def healing_length_kpc(M_total_Msun: np.ndarray,
                       g_dagger: float = G_DAGGER) -> np.ndarray:
    """Healing length in kpc — accepts custom g_dagger."""
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


def pick_id_col(cols: Iterable[str]) -> Optional[str]:
    c = set(cols)
    for cand in ("SubhaloID", "subhalo_id", "galaxy_id", "galaxy", "id"):
        if cand in c:
            return cand
    return None


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


def compute_galaxy_mass_table(
    df: pd.DataFrame, id_col: str, r_col: str, min_points: int = 1
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


def mass_match_galaxies(
    sparc_mass: pd.DataFrame, tng_mass: pd.DataFrame, caliper: float = 0.3
) -> pd.DataFrame:
    s = sparc_mass.copy().reset_index(drop=True)
    t = tng_mass.copy().reset_index(drop=True)
    used = np.zeros(len(t), dtype=bool)
    pairs: List[Dict[str, Any]] = []
    t_vals = t["log_Mb"].to_numpy(dtype=float)

    for _, row in s.sort_values("log_Mb").iterrows():
        d = np.abs(t_vals - float(row["log_Mb"]))
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > caliper:
            continue
        used[j] = True
        pairs.append({
            "sparc_id": row["id"],
            "tng_id": t.iloc[j]["id"],
            "sparc_log_Mb": float(row["log_Mb"]),
            "tng_log_Mb": float(t.iloc[j]["log_Mb"]),
            "abs_dlogM": float(abs(row["log_Mb"] - t.iloc[j]["log_Mb"])),
        })
    return pd.DataFrame(pairs)


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
    centers: np.ndarray, mean_var: np.ndarray
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


# ── Permutation null (single-process version) ────────────────────────

def xi_permutation_null(
    payload: List[Dict[str, Any]],
    centers: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 1000,
) -> np.ndarray:
    n_g = len(payload)
    if n_g == 0:
        return np.array([], dtype=float)

    lengths = np.array([len(p["res"]) for p in payload], dtype=int)
    fixed_len = np.all(lengths == lengths[0])
    large = n_g > 1000 and fixed_len and lengths[0] <= 64
    edges  = np.linspace(-2.0, 1.5, 9)
    n_bins = len(edges) - 1
    cnull  = np.full(n_perm, np.nan, dtype=float)

    if large:
        res_arr  = np.vstack([p["res"]  for p in payload]).astype(float)
        logx_arr = np.vstack([p["logX"] for p in payload]).astype(float)
        bin_arr  = np.digitize(logx_arr, edges) - 1
        mask_bins = [(bin_arr == b).astype(float) for b in range(n_bins)]
        counts    = np.stack([m.sum(axis=1) for m in mask_bins], axis=1)
        valid     = counts >= 2
        core      = (10.0 ** centers >= 0.3) & (10.0 ** centers <= 3.0)

        for i in range(n_perm):
            rp  = rng.permuted(res_arr, axis=1)
            rp2 = rp * rp
            vmat = np.full((n_g, n_bins), np.nan, dtype=float)
            for b in range(n_bins):
                m  = mask_bins[b]
                s1 = np.sum(rp * m, axis=1)
                s2 = np.sum(rp2 * m, axis=1)
                n  = counts[:, b]
                ok = valid[:, b]
                vb = np.full(n_g, np.nan, dtype=float)
                vb[ok] = (s2[ok] - (s1[ok] ** 2) / n[ok]) / (n[ok] - 1.0)
                vmat[:, b] = vb
            mean_prof = np.nanmean(vmat, axis=0)
            cnull[i] = concentration_from_profile(centers, mean_prof)
        return cnull

    logx_list = [p["logX"] for p in payload]
    res_list  = [p["res"]  for p in payload]
    for i in range(n_perm):
        rr = [r[rng.permutation(len(r))] for r in res_list]
        _, vmat = stacked_variance_profile(logx_list, rr)
        mean_prof = np.nanmean(vmat, axis=0)
        cnull[i] = concentration_from_profile(centers, mean_prof)
    return cnull


# ── Multiprocessing permutation worker ────────────────────────────────

# Module-level globals set by pool initializer
_WORKER_LOGX: List[np.ndarray] = []
_WORKER_RES:  List[np.ndarray] = []
_WORKER_CENTERS: np.ndarray = np.array([])


def _worker_init(logx_list: List[np.ndarray],
                 res_list: List[np.ndarray],
                 centers: np.ndarray) -> None:
    global _WORKER_LOGX, _WORKER_RES, _WORKER_CENTERS
    _WORKER_LOGX = logx_list
    _WORKER_RES  = res_list
    _WORKER_CENTERS = centers


def _worker_chunk(args: Tuple[int, int]) -> List[float]:
    """Run a chunk of permutations in a worker process."""
    seed, n_perm = args
    rng = np.random.default_rng(seed)
    results: List[float] = []
    for _ in range(n_perm):
        rr = [r[rng.permutation(len(r))] for r in _WORKER_RES]
        _, vmat = stacked_variance_profile(_WORKER_LOGX, rr)
        mean_prof = np.nanmean(vmat, axis=0)
        results.append(concentration_from_profile(_WORKER_CENTERS, mean_prof))
    return results


def xi_permutation_null_parallel(
    payload: List[Dict[str, Any]],
    centers: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 1000,
    n_jobs: int = 1,
    chunksize: int = 50,
) -> np.ndarray:
    """Parallelized permutation null using multiprocessing.Pool."""
    if n_jobs <= 1 or len(payload) == 0:
        return xi_permutation_null(payload, centers, rng, n_perm)

    logx_list = [p["logX"] for p in payload]
    res_list  = [p["res"]  for p in payload]

    # Divide permutations into chunks
    n_chunks = max(1, (n_perm + chunksize - 1) // chunksize)
    chunk_sizes: List[int] = []
    remaining = n_perm
    for _ in range(n_chunks):
        cs = min(chunksize, remaining)
        chunk_sizes.append(cs)
        remaining -= cs
        if remaining <= 0:
            break

    # Generate independent seeds for each chunk
    seeds = rng.integers(0, 2**63, size=len(chunk_sizes)).tolist()
    tasks = list(zip(seeds, chunk_sizes))

    print(f"    [xi perm parallel] {n_perm} perms across {n_jobs} workers, "
          f"{len(tasks)} chunks of ~{chunksize}")

    with mp.Pool(
        processes=n_jobs,
        initializer=_worker_init,
        initargs=(logx_list, res_list, centers),
    ) as pool:
        results = pool.map(_worker_chunk, tasks)

    cnull = np.array([v for chunk in results for v in chunk], dtype=float)
    return cnull[:n_perm]


# =====================================================================
# Per-galaxy concentration helper
# =====================================================================

def per_galaxy_concentration(payload: List[Dict[str, Any]],
                             ) -> List[Dict[str, Any]]:
    """Compute xi-concentration C for each individual galaxy."""
    rows: List[Dict[str, Any]] = []
    for p in payload:
        if p["n_points"] < 8:
            continue
        centers, vmat = stacked_variance_profile([p["logX"]], [p["res"]])
        mean_var = vmat[0]
        C = concentration_from_profile(centers, mean_var)
        if np.isfinite(C):
            rows.append({
                "id": p["id"],
                "xi_kpc": p["xi_kpc"],
                "n_points": p["n_points"],
                "C": C,
            })
    return rows


# =====================================================================
# Main analysis
# =====================================================================

def discover_tng_points(root: Path) -> Optional[Path]:
    candidates = [
        root / "datasets/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/rar_points.parquet",
        root / "datasets/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points_CLEAN.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_test6(
    root: Path,
    out_dir: Path,
    g_dagger: float,
    seed: int,
    n_perm: int,
    n_boot: int,
    n_jobs: int,
    chunksize: int,
    caliper: float,
) -> Dict[str, Any]:
    print("=" * 72)
    print("TEST 6 — Xi mass-matched pairs comparison")
    print("=" * 72)

    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    sparc_df, sparc_meta = load_sparc_points(root)
    print(f"  SPARC: {sparc_meta['n_points']} points, "
          f"{sparc_meta['n_galaxies']} galaxies")

    tng_path = discover_tng_points(root)
    if tng_path is None:
        msg = "TNG per-point data not found"
        print(f"  BLOCKED: {msg}")
        summary = {"status": "BLOCKED", "reason": msg}
        write_json(out_dir / "summary_xi_massmatched.json", summary)
        return summary
    print(f"  TNG file: {tng_path}")
    tng_df = load_tng_points(tng_path)
    print(f"  TNG: {len(tng_df)} points, {tng_df['id'].nunique()} galaxies")

    # ── Mass tables ───────────────────────────────────────────────────
    print("\n[2/6] Computing mass tables & matching...")
    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sp_mass = compute_galaxy_mass_table(sp, id_col="id", r_col="r_kpc",
                                        min_points=5)
    tng_mass = compute_galaxy_mass_table(tng_df, id_col="id", r_col="r_kpc",
                                         min_points=5)

    pairs = mass_match_galaxies(sp_mass, tng_mass, caliper=caliper)
    n_pairs = len(pairs)
    print(f"  Matched pairs: {n_pairs} (caliper={caliper} dex)")
    if n_pairs == 0:
        summary = {
            "status": "BLOCKED",
            "reason": f"No mass-matched pairs within ±{caliper} dex",
        }
        write_json(out_dir / "summary_xi_massmatched.json", summary)
        return summary

    # Enrich pairs with ξ and Mdyn from mass tables
    sp_xi_map  = dict(zip(sp_mass["id"],  sp_mass["xi_kpc"]))
    tng_xi_map = dict(zip(tng_mass["id"], tng_mass["xi_kpc"]))
    sp_md_map  = dict(zip(sp_mass["id"],  sp_mass["log_Mdyn"]))
    tng_md_map = dict(zip(tng_mass["id"], tng_mass["log_Mdyn"]))

    pairs["sparc_xi_kpc"]  = pairs["sparc_id"].map(sp_xi_map)
    pairs["tng_xi_kpc"]    = pairs["tng_id"].map(tng_xi_map)
    pairs["sparc_log_Mdyn"] = pairs["sparc_id"].map(sp_md_map)
    pairs["tng_log_Mdyn"]   = pairs["tng_id"].map(tng_md_map)
    pairs["log10_xi_ratio"]   = np.log10(pairs["tng_xi_kpc"] / pairs["sparc_xi_kpc"])
    pairs["log10_Mdyn_ratio"] = pairs["tng_log_Mdyn"] - pairs["sparc_log_Mdyn"]

    pairs.to_csv(out_dir / "xi_pairs.csv", index=False)

    # ── ξ and Mdyn ratio headline statistics ──────────────────────────
    xi_ratios   = pairs["log10_xi_ratio"].dropna().to_numpy()
    mdyn_ratios = pairs["log10_Mdyn_ratio"].dropna().to_numpy()
    median_log10_xi_ratio   = float(np.median(xi_ratios))   if len(xi_ratios)   > 0 else None
    median_log10_Mdyn_ratio = float(np.median(mdyn_ratios)) if len(mdyn_ratios) > 0 else None
    print(f"  Median log10(ξ_TNG / ξ_SPARC)     = {median_log10_xi_ratio}")
    print(f"  Median log10(Mdyn_TNG / Mdyn_SPARC) = {median_log10_Mdyn_ratio}")

    # ── Filter to matched galaxies ────────────────────────────────────
    sparc_ids = set(pairs["sparc_id"])
    tng_ids   = set(pairs["tng_id"])
    sp_matched  = sp[sp["id"].isin(sparc_ids)].copy()
    tng_matched = tng_df[tng_df["id"].isin(tng_ids)].copy()

    # ── Per-galaxy xi payloads (with custom g†) ───────────────────────
    print("\n[3/6] Computing per-galaxy xi profiles...")
    rng = np.random.default_rng(seed)

    sp_payload = per_galaxy_xi_payload(
        sp_matched, id_col="id", r_col="r_kpc",
        res_col="log_res_use", g_dagger=g_dagger,
    )
    tng_payload = per_galaxy_xi_payload(
        tng_matched, id_col="id", r_col="r_kpc",
        res_col="log_res_use", g_dagger=g_dagger,
    )
    print(f"  SPARC payload: {len(sp_payload)} galaxies (≥8 pts)")
    print(f"  TNG   payload: {len(tng_payload)} galaxies (≥8 pts)")

    # ── Stacked variance profiles ─────────────────────────────────────
    sp_logx  = [p["logX"] for p in sp_payload]
    sp_res   = [p["res"]  for p in sp_payload]
    tng_logx = [p["logX"] for p in tng_payload]
    tng_res  = [p["res"]  for p in tng_payload]

    centers, sp_vmat  = stacked_variance_profile(sp_logx, sp_res)
    _,       tng_vmat = stacked_variance_profile(tng_logx, tng_res)

    sp_mean  = np.nanmean(sp_vmat, axis=0)
    tng_mean = np.nanmean(tng_vmat, axis=0)

    C_sparc = concentration_from_profile(centers, sp_mean)
    C_tng   = concentration_from_profile(centers, tng_mean)
    print(f"  C_SPARC = {C_sparc:.4f}")
    print(f"  C_TNG   = {C_tng:.4f}")

    # ── Bootstrap CIs (galaxy resampling) ─────────────────────────────
    print(f"\n[4/6] Bootstrap CIs ({n_boot} iterations)...")
    sp_boot  = np.full(n_boot, np.nan)
    tng_boot = np.full(n_boot, np.nan)
    n_sp  = len(sp_payload)
    n_tng = len(tng_payload)
    for i in range(n_boot):
        idx_s = rng.integers(0, n_sp, size=n_sp)
        idx_t = rng.integers(0, n_tng, size=n_tng)
        sp_boot[i]  = concentration_from_profile(
            centers, np.nanmean(sp_vmat[idx_s], axis=0))
        tng_boot[i] = concentration_from_profile(
            centers, np.nanmean(tng_vmat[idx_t], axis=0))
        if (i + 1) % 100 == 0:
            print(f"    bootstrap {i+1}/{n_boot}")

    sp_ci  = [float(np.nanpercentile(sp_boot, 2.5)),
              float(np.nanpercentile(sp_boot, 97.5))]
    tng_ci = [float(np.nanpercentile(tng_boot, 2.5)),
              float(np.nanpercentile(tng_boot, 97.5))]

    # ── Stacked variance bootstrap CIs for profiles ───────────────────
    sp_prof_lo  = np.nanpercentile(
        np.array([np.nanmean(sp_vmat[rng.integers(0, n_sp, size=n_sp)], axis=0)
                  for _ in range(n_boot)]), 2.5, axis=0)
    sp_prof_hi  = np.nanpercentile(
        np.array([np.nanmean(sp_vmat[rng.integers(0, n_sp, size=n_sp)], axis=0)
                  for _ in range(n_boot)]), 97.5, axis=0)
    tng_prof_lo = np.nanpercentile(
        np.array([np.nanmean(tng_vmat[rng.integers(0, n_tng, size=n_tng)], axis=0)
                  for _ in range(n_boot)]), 2.5, axis=0)
    tng_prof_hi = np.nanpercentile(
        np.array([np.nanmean(tng_vmat[rng.integers(0, n_tng, size=n_tng)], axis=0)
                  for _ in range(n_boot)]), 97.5, axis=0)

    # ── Permutation nulls ─────────────────────────────────────────────
    print(f"\n[5/6] Permutation null ({n_perm} perms, {n_jobs} jobs)...")
    t_perm = time.time()
    sp_cnull  = xi_permutation_null_parallel(
        sp_payload, centers, rng, n_perm=n_perm,
        n_jobs=n_jobs, chunksize=chunksize)
    tng_cnull = xi_permutation_null_parallel(
        tng_payload, centers, rng, n_perm=n_perm,
        n_jobs=n_jobs, chunksize=chunksize)
    dt_perm = time.time() - t_perm
    print(f"  Permutations done in {dt_perm:.1f}s")

    sp_p  = float(np.mean(np.where(np.isfinite(sp_cnull),
                                    sp_cnull >= C_sparc, False)))
    tng_p = float(np.mean(np.where(np.isfinite(tng_cnull),
                                    tng_cnull >= C_tng, False)))
    print(f"  SPARC perm p = {sp_p:.4f}")
    print(f"  TNG   perm p = {tng_p:.4f}")

    # ── Per-galaxy concentration ──────────────────────────────────────
    sp_per_gal  = per_galaxy_concentration(sp_payload)
    tng_per_gal = per_galaxy_concentration(tng_payload)

    # Build paired table (matched by pair ordering)
    sp_c_map  = {r["id"]: r["C"] for r in sp_per_gal}
    tng_c_map = {r["id"]: r["C"] for r in tng_per_gal}

    paired_rows: List[Dict[str, Any]] = []
    for _, pr in pairs.iterrows():
        sc = sp_c_map.get(pr["sparc_id"])
        tc = tng_c_map.get(pr["tng_id"])
        if sc is not None and tc is not None:
            paired_rows.append({
                "sparc_id": pr["sparc_id"],
                "tng_id": pr["tng_id"],
                "sparc_log_Mb": pr["sparc_log_Mb"],
                "tng_log_Mb": pr["tng_log_Mb"],
                "sparc_C": sc,
                "tng_C": tc,
                "log10_ratio": float(np.log10(tc / sc)) if sc > 0 else np.nan,
            })

    n_paired_C = len(paired_rows)
    print(f"\n  Paired galaxies with valid C: {n_paired_C}")

    if n_paired_C > 0:
        log10_ratios = np.array([r["log10_ratio"] for r in paired_rows])
        log10_ratios = log10_ratios[np.isfinite(log10_ratios)]
        median_log10_ratio = float(np.median(log10_ratios))
        mean_log10_ratio   = float(np.mean(log10_ratios))

        try:
            w = wilcoxon(log10_ratios, alternative="two-sided")
            wilcoxon_p = float(w.pvalue)
            wilcoxon_stat = float(w.statistic)
        except Exception:
            wilcoxon_p = None
            wilcoxon_stat = None

        sparc_Cs = np.array([r["sparc_C"] for r in paired_rows])
        tng_Cs   = np.array([r["tng_C"]   for r in paired_rows])
        print(f"  Median log10(C_TNG/C_SPARC) = {median_log10_ratio:.4f}")
        print(f"  Wilcoxon signed-rank p = {wilcoxon_p}")
    else:
        median_log10_ratio = None
        mean_log10_ratio   = None
        wilcoxon_p = None
        wilcoxon_stat = None
        sparc_Cs = np.array([])
        tng_Cs   = np.array([])

    # ── Figures ───────────────────────────────────────────────────────
    print("\n[6/6] Generating figures & report...")

    # Fig 1: Paired C scatter
    fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=180)
    if n_paired_C > 0:
        ax1.scatter(sparc_Cs, tng_Cs, s=20, alpha=0.6, color="#1f77b4",
                    edgecolors="black", linewidths=0.3)
        lims = [
            min(sparc_Cs.min(), tng_Cs.min()) * 0.8,
            max(sparc_Cs.max(), tng_Cs.max()) * 1.2,
        ]
        ax1.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
    ax1.set_xlabel("SPARC per-galaxy C")
    ax1.set_ylabel("TNG per-galaxy C")
    ax1.set_title(f"Mass-matched xi concentration (N={n_paired_C})")
    ax1.set_facecolor("white")
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig_xi_pairs_scatter.png", facecolor="white")
    plt.close(fig1)

    # Fig 2: log10 ratio vs mass
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=180)
    if n_paired_C > 0:
        masses = np.array([r["sparc_log_Mb"] for r in paired_rows])
        ratios = np.array([r["log10_ratio"]  for r in paired_rows])
        ok = np.isfinite(ratios)
        ax2.scatter(masses[ok], ratios[ok], s=20, alpha=0.6, color="#2ca02c",
                    edgecolors="black", linewidths=0.3)
        ax2.axhline(0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("log$_{10}$(M$_{\\rm bar}$ / M$_\\odot$)")
        ax2.set_ylabel("log$_{10}$(C$_{\\rm TNG}$ / C$_{\\rm SPARC}$)")
        if median_log10_ratio is not None:
            ax2.axhline(median_log10_ratio, color="blue", linestyle=":",
                        linewidth=1, label=f"median={median_log10_ratio:.3f}")
            ax2.legend(fontsize=9)
    ax2.set_title("C ratio vs baryon mass")
    ax2.set_facecolor("white")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_xi_ratio_vs_mass.png", facecolor="white")
    plt.close(fig2)

    # Fig 3: Matched stacked variance profiles
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), dpi=180)

    # Panel A: SPARC profile
    ax = axes3[0]
    x_plot = 10.0 ** centers
    ok_s = np.isfinite(sp_mean)
    ax.plot(x_plot[ok_s], sp_mean[ok_s], "o-", color="#1f77b4", label="SPARC")
    ax.fill_between(x_plot, sp_prof_lo, sp_prof_hi, color="#9ecae1", alpha=0.4)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("X = R / ξ")
    ax.set_ylabel("stacked variance σ²")
    ax.set_title(f"SPARC matched (N={len(sp_payload)}, C={C_sparc:.3f})")
    ax.set_facecolor("white")

    # Panel B: TNG profile
    ax = axes3[1]
    ok_t = np.isfinite(tng_mean)
    ax.plot(x_plot[ok_t], tng_mean[ok_t], "s-", color="#ff7f0e", label="TNG")
    ax.fill_between(x_plot, tng_prof_lo, tng_prof_hi, color="#ffbb78", alpha=0.4)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("X = R / ξ")
    ax.set_ylabel("stacked variance σ²")
    ax.set_title(f"TNG matched (N={len(tng_payload)}, C={C_tng:.3f})")
    ax.set_facecolor("white")

    # Panel C: Overlay
    ax = axes3[2]
    if ok_s.any():
        ax.plot(x_plot[ok_s], sp_mean[ok_s], "o-", color="#1f77b4",
                label=f"SPARC (C={C_sparc:.3f})")
    if ok_t.any():
        ax.plot(x_plot[ok_t], tng_mean[ok_t], "s-", color="#ff7f0e",
                label=f"TNG (C={C_tng:.3f})")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("X = R / ξ")
    ax.set_ylabel("stacked variance σ²")
    ax.set_title("Overlay")
    ax.legend(fontsize=9)
    ax.set_facecolor("white")

    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_xi_matched_X_profiles.png", facecolor="white")
    plt.close(fig3)

    # ── Summary JSON ──────────────────────────────────────────────────
    dt = time.time() - t0

    summary: Dict[str, Any] = {
        "test": "xi_massmatched_pairs",
        "status": "OK",
        "g_dagger": g_dagger,
        "caliper_dex": caliper,
        "seed": seed,
        "n_perm": n_perm,
        "n_boot": n_boot,
        "N_pairs_mass": n_pairs,
        "N_pairs_with_C": n_paired_C,
        "tng_file": str(tng_path),
        "sparc": {
            "n_payload_galaxies": len(sp_payload),
            "C_stacked": C_sparc,
            "C_ci95": sp_ci,
            "perm_p": sp_p,
        },
        "tng": {
            "n_payload_galaxies": len(tng_payload),
            "C_stacked": C_tng,
            "C_ci95": tng_ci,
            "perm_p": tng_p,
        },
        "paired_comparison": {
            "N": n_paired_C,
            "median_log10_C_ratio": median_log10_ratio,
            "mean_log10_C_ratio": mean_log10_ratio,
            "wilcoxon_signed_rank_p": wilcoxon_p,
            "wilcoxon_statistic": wilcoxon_stat,
        },
        "healing_length_comparison": {
            "N_pairs": n_pairs,
            "median_log10_xi_ratio": median_log10_xi_ratio,
            "median_log10_Mdyn_ratio": median_log10_Mdyn_ratio,
        },
        "elapsed_seconds": round(dt, 1),
    }
    write_json(out_dir / "summary_xi_massmatched.json", summary)

    # ── Markdown report ───────────────────────────────────────────────
    lines: List[str] = [
        "# Test 6 — Xi Mass-Matched Pairs Comparison",
        "",
        "## Purpose",
        "Test 3 showed TNG has strong xi-organizing (C=1.549, p=0.000) but "
        "SPARC does not (C=0.687, p=0.571). This test controls for mass by "
        "using Test 2's mass-matching to build paired sub-samples.",
        "",
        "## Configuration",
        f"- g† = {g_dagger:.4e} m/s²",
        f"- Caliper = {caliper} dex",
        f"- N_perm = {n_perm}, N_boot = {n_boot}",
        f"- Seed = {seed}",
        f"- TNG file: `{tng_path}`",
        "",
        "## Results",
        f"- Mass-matched pairs: {n_pairs}",
        f"- Pairs with valid per-galaxy C: {n_paired_C}",
        "",
        "### Stacked concentration (matched sub-samples)",
        f"| Dataset | N galaxies | C_stacked | 95% CI | perm p |",
        f"|---------|-----------|-----------|--------|--------|",
        f"| SPARC   | {len(sp_payload)} | {C_sparc:.4f} | [{sp_ci[0]:.4f}, {sp_ci[1]:.4f}] | {sp_p:.4f} |",
        f"| TNG     | {len(tng_payload)} | {C_tng:.4f} | [{tng_ci[0]:.4f}, {tng_ci[1]:.4f}] | {tng_p:.4f} |",
        "",
        "### Paired comparison",
    ]
    if n_paired_C > 0:
        lines += [
            f"- Median log10(C_TNG / C_SPARC) = {median_log10_ratio:.4f}",
            f"- Mean log10(C_TNG / C_SPARC) = {mean_log10_ratio:.4f}",
            f"- Wilcoxon signed-rank p = {wilcoxon_p}",
            f"- Wilcoxon statistic = {wilcoxon_stat}",
        ]
    else:
        lines.append("- No valid paired concentrations to compare.")
    lines += [
        "",
        "### Healing length and dynamical mass at fixed baryon mass",
        f"- N pairs: {n_pairs}",
        f"- Median log10(ξ_TNG / ξ_SPARC) = {median_log10_xi_ratio}",
        f"- Median log10(Mdyn_TNG / Mdyn_SPARC) = {median_log10_Mdyn_ratio}",
        "- Since ξ ∝ sqrt(Mdyn), a systematic Mdyn offset drives ξ differences.",
        "",
        f"Elapsed: {dt:.1f}s",
    ]
    (out_dir / "report_xi_massmatched.md").write_text("\n".join(lines) + "\n")

    # ── Run metadata ──────────────────────────────────────────────────
    meta = {
        "script": "test_xi_massmatched_pairs.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "args": {
            "project_root": str(root),
            "out_dir": str(out_dir),
            "g_dagger": g_dagger,
            "seed": seed,
            "n_perm": n_perm,
            "n_boot": n_boot,
            "n_jobs": n_jobs,
            "chunksize": chunksize,
            "caliper": caliper,
        },
        "elapsed_seconds": round(dt, 1),
    }
    write_json(out_dir / "run_metadata.json", meta)

    print(f"\nDone in {dt:.1f}s.  Outputs in {out_dir}")
    return summary


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 6: Xi mass-matched pairs comparison")
    parser.add_argument("--project-root", type=str, default=None,
                        help="BEC dark matter repo root")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: outputs/xi_massmatched/<ts>)")
    parser.add_argument("--g-dagger", type=float, default=1.2e-10,
                        help="Condensation scale g† in m/s² (default: 1.2e-10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers for permutation null")
    parser.add_argument("--auto-jobs", action="store_true",
                        help="Auto-detect n_jobs from cpu_count()")
    parser.add_argument("--reserve-cpus", type=int, default=2,
                        help="CPUs to reserve when using --auto-jobs")
    parser.add_argument("--blas-threads", type=int, default=1,
                        help="BLAS threads per process")
    parser.add_argument("--chunksize", type=int, default=50,
                        help="Permutations per worker chunk")
    parser.add_argument("--caliper", type=float, default=0.3,
                        help="Mass-matching caliper in dex (default: 0.3)")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    root = Path(args.project_root).resolve() if args.project_root else default_root

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "outputs" / "xi_massmatched" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = args.n_jobs
    if args.auto_jobs:
        total = mp.cpu_count() or 1
        n_jobs = max(1, total - args.reserve_cpus)
        print(f"[auto-jobs] {total} CPUs detected, reserving {args.reserve_cpus} "
              f"→ {n_jobs} workers")

    run_test6(
        root=root,
        out_dir=out_dir,
        g_dagger=args.g_dagger,
        seed=args.seed,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        n_jobs=n_jobs,
        chunksize=args.chunksize,
        caliper=args.caliper,
    )


if __name__ == "__main__":
    main()
