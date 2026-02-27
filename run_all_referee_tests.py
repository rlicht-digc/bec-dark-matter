#!/usr/bin/env python3
"""Clean-room full rerun of referee battery with provenance and BLOCKED fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ks_2samp, wilcoxon

from common_rar import (
    G_SI,
    KPC,
    LOG_G_DAGGER,
    MSUN,
    healing_length_kpc,
    phase_bin_edges_centers,
    rar_bec,
    write_json,
)


# ---------------------------
# Shared config
# ---------------------------

PHASE_BIN_WIDTH = 0.25
PHASE_MIN_POINTS = 10
PHASE_BIN_START = -13.5
PHASE_BIN_STOP = -8.0

M2B_EDGE_BOUNDS: List[Tuple[float, float]] = [
    (1e-8, 5.0),   # s0
    (-2.0, 2.0),   # s1
    (1e-6, 5.0),   # Ap > 0
    (-12.0, -8.0), # mu_peak
    (0.05, 2.0),   # w_peak
    (-5.0, -1e-6), # Ad < 0
    (-12.0, -8.0), # mu_dip
    (0.05, 2.0),   # w_dip
    (-5.0, 5.0),   # E
    (-12.0, -8.0), # x_edge
    (0.01, 1.0),   # d_edge
]


# ---------------------------
# Logging / shell helpers
# ---------------------------


def log_line(runlog_path: Path, message: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with runlog_path.open("a") as f:
        f.write(line + "\n")


def run_shell(cmd: str) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/zsh",
        capture_output=True,
        text=True,
    )
    out_lines = proc.stdout.splitlines() if proc.stdout else []
    err_lines = proc.stderr.splitlines() if proc.stderr else []
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_lines": out_lines,
        "stderr_lines": err_lines,
    }


def robust_percentiles(x: np.ndarray, q: Sequence[float]) -> List[Optional[float]]:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return [None for _ in q]
    return [float(v) for v in np.percentile(y, list(q))]


def save_blocked_figure(path: Path, title: str, reason: str, dpi: int = 180) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, f"BLOCKED\n{reason}", ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---------------------------
# Discovery and loading
# ---------------------------


def path_score_generic(path: str, target_name: str) -> int:
    p = path.lower()
    score = 0
    if p.endswith(target_name.lower()):
        score += 200
    if "/analysis/results/" in p:
        score += 60
    if "bec-dark-matter" in p:
        score += 40
    if "archive" in p or "backup" in p or "old" in p:
        score -= 20
    return score


def choose_best_path(paths: List[str], target_name: str) -> Optional[str]:
    if not paths:
        return None
    scored = sorted(paths, key=lambda x: (path_score_generic(x, target_name), -len(x)), reverse=True)
    return scored[0]


def pick_id_col(cols: Iterable[str]) -> Optional[str]:
    c = set(cols)
    for cand in ["SubhaloID", "subhalo_id", "galaxy_id", "galaxy", "id", "ID"]:
        if cand in c:
            return cand
    return None


def load_tng_points(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        fmt = "parquet"
    else:
        df = pd.read_csv(path)
        fmt = "csv"

    id_col = pick_id_col(df.columns)
    if id_col is None:
        raise RuntimeError(f"No galaxy-id column found in {path}")

    if "r_kpc" in df.columns:
        r_col = "r_kpc"
    elif "R_kpc" in df.columns:
        r_col = "R_kpc"
    else:
        raise RuntimeError(f"No radius column (r_kpc/R_kpc) found in {path}")

    log_gbar_formula = None
    if "log_gbar" in df.columns:
        lgbar = pd.to_numeric(df["log_gbar"], errors="coerce").to_numpy(dtype=float)
        log_gbar_formula = "used existing log_gbar column"
    else:
        raw_col = None
        for c in ["gbar", "g_bar", "gbar_si", "gbar_SI"]:
            if c in df.columns:
                raw_col = c
                break
        if raw_col is None:
            raise RuntimeError(f"No log_gbar or gbar-like column in {path}")
        raw = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(dtype=float)
        lgbar = np.log10(np.maximum(raw, 1e-300))
        log_gbar_formula = f"log_gbar = log10(max({raw_col}, 1e-300))"

    log_gobs_formula = None
    if "log_gobs" in df.columns:
        lgobs = pd.to_numeric(df["log_gobs"], errors="coerce").to_numpy(dtype=float)
        log_gobs_formula = "used existing log_gobs column"
    else:
        raw_col = None
        for c in ["gobs", "g_obs", "gobs_si", "gobs_SI"]:
            if c in df.columns:
                raw_col = c
                break
        if raw_col is None:
            raise RuntimeError(f"No log_gobs or gobs-like column in {path}")
        raw = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(dtype=float)
        lgobs = np.log10(np.maximum(raw, 1e-300))
        log_gobs_formula = f"log_gobs = log10(max({raw_col}, 1e-300))"

    r = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)
    gid = df[id_col].astype(str).to_numpy()

    out = pd.DataFrame(
        {
            "id": gid,
            "r_kpc": r,
            "log_gbar": lgbar,
            "log_gobs": lgobs,
        }
    )

    if "log_res" in df.columns:
        log_res = pd.to_numeric(df["log_res"], errors="coerce").to_numpy(dtype=float)
        recomputed = out["log_gobs"].to_numpy(dtype=float) - rar_bec(out["log_gbar"].to_numpy(dtype=float))
        use_existing = np.nanmedian(np.abs(log_res - recomputed)) <= 0.01
        out["log_res_use"] = log_res if use_existing else recomputed
        residual_formula = "used log_res column" if use_existing else "recomputed log_res = log_gobs - rar_bec(log_gbar)"
    else:
        out["log_res_use"] = out["log_gobs"].to_numpy(dtype=float) - rar_bec(out["log_gbar"].to_numpy(dtype=float))
        residual_formula = "recomputed log_res = log_gobs - rar_bec(log_gbar)"

    mask = (
        np.isfinite(out["r_kpc"])
        & np.isfinite(out["log_gbar"])
        & np.isfinite(out["log_gobs"])
        & np.isfinite(out["log_res_use"])
        & (out["r_kpc"] > 0)
        & (out["log_gbar"] > -20)
        & (out["log_gobs"] > -20)
    )
    out = out[mask].copy()

    meta = {
        "input_format": fmt,
        "id_col": id_col,
        "radius_col": r_col,
        "log_gbar_formula": log_gbar_formula,
        "log_gobs_formula": log_gobs_formula,
        "residual_formula": residual_formula,
        "n_points_after_filters": int(len(out)),
        "n_galaxies_after_filters": int(out["id"].nunique()),
    }
    return out, meta


def score_tng_candidate(path: str) -> int:
    p = path.lower()
    s = 0
    if "verified" in p:
        s += 100
    if "clean" in p:
        s += 50
    if "3000x50" in p:
        s += 40
    if "48133" in p:
        s += 35
    if "rar_points" in p:
        s += 30
    if p.endswith(".parquet"):
        s += 8
    if p.endswith(".csv"):
        s += 5
    if "contaminated" in p or "quarantine" in p:
        s -= 150
    return s


def choose_tng_path(candidates: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    notes: Dict[str, Any] = {"ranked": [], "validation_failures": {}}
    if not candidates:
        return None, notes

    ranked = sorted(set(candidates), key=lambda p: (score_tng_candidate(p), -len(p)), reverse=True)
    notes["ranked"] = [{"path": p, "score": score_tng_candidate(p)} for p in ranked[:50]]

    for p in ranked:
        pp = Path(p)
        try:
            df, meta = load_tng_points(pp)
            if len(df) == 0:
                notes["validation_failures"][p] = "loaded but empty after filters"
                continue
            notes["chosen_meta_preview"] = meta
            return p, notes
        except Exception as exc:
            notes["validation_failures"][p] = str(exc)
            continue
    return None, notes


def discover_data(runlog_path: Path) -> Dict[str, Any]:
    cmd_rar_gal = 'find ~ -name "*rar_points_unified.csv" -o -name "*galaxy_results_unified.csv" 2>/dev/null'
    cmd_tng_csv = 'find ~ -iname "*tng*" -iname "*.csv" 2>/dev/null'
    cmd_tng_all = 'find ~ -iname "*tng*" \\( -iname "*.csv" -o -iname "*.parquet" \\) 2>/dev/null'

    res_rar_gal = run_shell(cmd_rar_gal)
    res_tng_csv = run_shell(cmd_tng_csv)
    res_tng_all = run_shell(cmd_tng_all)

    log_line(runlog_path, f"Discovery cmd: {cmd_rar_gal} -> rc={res_rar_gal['returncode']} lines={len(res_rar_gal['stdout_lines'])}")
    log_line(runlog_path, f"Discovery cmd: {cmd_tng_csv} -> rc={res_tng_csv['returncode']} lines={len(res_tng_csv['stdout_lines'])}")
    log_line(runlog_path, f"Discovery cmd: {cmd_tng_all} -> rc={res_tng_all['returncode']} lines={len(res_tng_all['stdout_lines'])}")

    rg_lines = sorted(set([ln.strip() for ln in res_rar_gal["stdout_lines"] if ln.strip()]))

    tng_candidates = sorted(
        set([ln.strip() for ln in (res_tng_csv["stdout_lines"] + res_tng_all["stdout_lines"]) if ln.strip()])
    )

    # Fallback walk for environments where find from "~" yields empty results.
    fallback_hits: Dict[str, List[str]] = {
        "rar_points": [],
        "galaxy_results": [],
        "tng_points": [],
    }
    if len(rg_lines) == 0 or len(tng_candidates) == 0:
        log_line(runlog_path, "Discovery fallback: starting os.walk scan under home directory")
        home = str(Path.home())
        skip_dirs = {
            "Library",
            ".Trash",
            ".cache",
            ".vscode",
            ".git",
            ".venv",
            "venv",
            "myenv",
            ".pytest_cache",
            ".m2",
            ".gradle",
            ".npm",
            ".cargo",
            ".rustup",
            ".swiftpm",
            ".local",
        }
        for dirpath, dirnames, filenames in os.walk(home, topdown=True):
            base = os.path.basename(dirpath)
            if base in skip_dirs:
                dirnames[:] = []
                continue

            # Prune hidden directories except a few that may contain project data.
            keep_hidden = {".codex", ".claude", ".claude-worktrees"}
            dirnames[:] = [
                d for d in dirnames if (not d.startswith(".")) or (d in keep_hidden)
            ]

            for fn in filenames:
                full = os.path.join(dirpath, fn)
                lf = fn.lower()
                if fn == "rar_points_unified.csv":
                    fallback_hits["rar_points"].append(full)
                if fn == "galaxy_results_unified.csv":
                    fallback_hits["galaxy_results"].append(full)
                if ("tng" in full.lower()) and (lf.endswith(".csv") or lf.endswith(".parquet")):
                    fallback_hits["tng_points"].append(full)

        if len(rg_lines) == 0:
            rg_lines = sorted(set(fallback_hits["rar_points"] + fallback_hits["galaxy_results"]))
        if len(tng_candidates) == 0:
            tng_candidates = sorted(set(fallback_hits["tng_points"]))
        log_line(
            runlog_path,
            f"Discovery fallback complete: rar_hits={len(fallback_hits['rar_points'])}, "
            f"gal_hits={len(fallback_hits['galaxy_results'])}, tng_hits={len(fallback_hits['tng_points'])}",
        )

    # Candidate extraction after all discovery passes.
    rar_candidates = [p for p in rg_lines if p.endswith("rar_points_unified.csv")]
    gal_candidates = [p for p in rg_lines if p.endswith("galaxy_results_unified.csv")]

    chosen_rar = choose_best_path(rar_candidates, "rar_points_unified.csv")
    chosen_gal = choose_best_path(gal_candidates, "galaxy_results_unified.csv")
    chosen_tng, tng_notes = choose_tng_path(tng_candidates)

    dir_listings: Dict[str, Any] = {}
    for d in [Path.home() / "analysis", Path.home() / "data", Path.home() / "tng_data"]:
        key = str(d)
        if d.exists() and d.is_dir():
            items = sorted([p.name for p in d.iterdir()])
            dir_listings[key] = {"exists": True, "n_items": len(items), "first_50": items[:50]}
            log_line(runlog_path, f"Listed {key}: {len(items)} entries")
        else:
            dir_listings[key] = {"exists": False}
            log_line(runlog_path, f"Listed {key}: missing")

    manifest = {
        "rar_points_path": chosen_rar,
        "galaxy_results_path": chosen_gal,
        "tng_candidate_paths": tng_candidates,
        "chosen_tng_path": chosen_tng,
        "notes": {
            "rar_candidates": rar_candidates,
            "galaxy_candidates": gal_candidates,
            "discovery_commands": [cmd_rar_gal, cmd_tng_csv, cmd_tng_all],
            "directory_listings": dir_listings,
            "tng_selection": tng_notes,
            "fallback_hits": fallback_hits,
        },
    }
    return manifest


# ---------------------------
# Input validation
# ---------------------------


def validate_inputs(
    rar_points_path: Optional[str],
    sample_seed: int,
    runlog_path: Path,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if rar_points_path is None:
        return {
            "status": "BLOCKED",
            "reason": "rar_points_unified.csv not found",
        }, None, None

    p = Path(rar_points_path)
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        return {"status": "BLOCKED", "reason": f"Failed to read {p}: {exc}"}, None, None

    required_cols = ["galaxy", "source", "log_gbar", "log_gobs", "log_res", "R_kpc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {
            "status": "BLOCKED",
            "reason": f"Missing required columns: {missing}",
            "required_cols": required_cols,
            "present_cols": list(df.columns),
        }, None, None

    for c in ["log_gbar", "log_gobs", "log_res", "R_kpc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    source_counts = {str(k): int(v) for k, v in df["source"].value_counts(dropna=False).to_dict().items()}

    sparc = df[df["source"] == "SPARC"].copy()
    sparc = sparc[np.isfinite(sparc["log_gbar"]) & np.isfinite(sparc["log_gobs"]) & np.isfinite(sparc["log_res"]) & np.isfinite(sparc["R_kpc"])].copy()

    n_sparc_points = int(len(sparc))
    n_sparc_galaxies = int(sparc["galaxy"].nunique()) if len(sparc) else 0

    if n_sparc_points == 0:
        return {
            "status": "BLOCKED",
            "reason": "No valid SPARC rows after filtering",
            "source_counts": source_counts,
        }, df, None

    rng = np.random.default_rng(sample_seed)
    sample_n = min(200, n_sparc_points)
    sample_idx = rng.choice(n_sparc_points, size=sample_n, replace=False)
    sample = sparc.iloc[sample_idx]

    res_check_sample = sample["log_gobs"].to_numpy(dtype=float) - rar_bec(sample["log_gbar"].to_numpy(dtype=float))
    diff_sample = res_check_sample - sample["log_res"].to_numpy(dtype=float)

    full_res_check = sparc["log_gobs"].to_numpy(dtype=float) - rar_bec(sparc["log_gbar"].to_numpy(dtype=float))
    full_diff = full_res_check - sparc["log_res"].to_numpy(dtype=float)
    mismatch_frac_full = float(np.mean(np.abs(full_diff) > 0.01))
    recompute_all = bool(mismatch_frac_full > 0.05)

    if recompute_all:
        sparc["log_res_use"] = full_res_check
        residual_definition = "recomputed log_res = log_gobs - rar_bec(log_gbar)"
        log_line(runlog_path, "Residual mismatch >5%; recomputing SPARC residuals globally")
    else:
        sparc["log_res_use"] = sparc["log_res"].to_numpy(dtype=float)
        residual_definition = "used log_res column"

    summary = {
        "status": "OK",
        "rar_points_path": str(p),
        "required_columns": required_cols,
        "source_counts": source_counts,
        "sparc_counts": {
            "n_points": n_sparc_points,
            "n_galaxies": n_sparc_galaxies,
        },
        "residual_consistency_sample": {
            "sample_size": int(sample_n),
            "sample_seed": int(sample_seed),
            "max_abs_diff": float(np.max(np.abs(diff_sample))),
            "median_abs_diff": float(np.median(np.abs(diff_sample))),
            "fraction_abs_diff_gt_0p01": float(np.mean(np.abs(diff_sample) > 0.01)),
        },
        "residual_consistency_full": {
            "max_abs_diff": float(np.max(np.abs(full_diff))),
            "median_abs_diff": float(np.median(np.abs(full_diff))),
            "fraction_abs_diff_gt_0p01": mismatch_frac_full,
            "mismatch_threshold": 0.05,
            "recomputed": recompute_all,
            "residual_definition_used": residual_definition,
        },
    }

    return summary, df, sparc


# ---------------------------
# Phase model helpers (Test 1/2)
# ---------------------------


def prepare_phase_binning(log_gbar: np.ndarray, width: float, min_points: int) -> Dict[str, Any]:
    edges, centers = phase_bin_edges_centers(width=width, start_edge=PHASE_BIN_START, stop_edge=PHASE_BIN_STOP)
    idx = np.digitize(log_gbar, edges) - 1
    good = (idx >= 0) & (idx < len(centers))
    idx2 = np.full(len(log_gbar), -1, dtype=int)
    idx2[good] = idx[good]
    counts = np.bincount(idx2[good], minlength=len(centers))
    valid_bins = np.where(counts >= min_points)[0]
    dropped_bins = np.where(counts < min_points)[0]
    return {
        "edges": edges,
        "centers": centers,
        "idx": idx2,
        "counts": counts,
        "valid_bins": valid_bins,
        "dropped_bins": dropped_bins,
    }


def variance_profile_from_prebinned(
    residuals: np.ndarray,
    prep: Dict[str, Any],
    ddof: int = 1,
) -> Dict[str, Any]:
    centers = prep["centers"]
    idx = prep["idx"]
    counts = prep["counts"]
    valid_bins = prep["valid_bins"]

    x_bins: List[float] = []
    var_bins: List[float] = []
    var_err: List[float] = []
    used_bins: List[int] = []
    used_counts: List[int] = []

    for b in valid_bins:
        m = idx == int(b)
        n = int(np.sum(m))
        if n <= ddof:
            continue
        r = residuals[m]
        var = float(np.var(r, ddof=ddof))
        var = max(var, 1e-12)
        err = float(var * math.sqrt(2.0 / max(n - 1, 1)))
        err = max(err, 1e-12)
        x_bins.append(float(centers[b]))
        var_bins.append(var)
        var_err.append(err)
        used_bins.append(int(b))
        used_counts.append(int(n))

    return {
        "x_bins": np.asarray(x_bins, dtype=float),
        "var_bins": np.asarray(var_bins, dtype=float),
        "var_err": np.asarray(var_err, dtype=float),
        "used_bins": np.asarray(used_bins, dtype=int),
        "used_counts": np.asarray(used_counts, dtype=int),
        "n_bins_used": int(len(x_bins)),
    }


def nll_from_model(y: np.ndarray, yhat: np.ndarray, yerr: np.ndarray) -> float:
    if np.any(~np.isfinite(yhat)):
        return 1e30
    neg = np.minimum(yhat, 0.0)
    penalty = 1e7 * float(np.sum(neg * neg))
    yhat = np.where(yhat <= 1e-12, 1e-12, yhat)
    sig2 = np.maximum(yerr * yerr, 1e-18)
    base = 0.5 * float(np.sum(((y - yhat) ** 2) / sig2 + np.log(2.0 * np.pi * sig2)))
    return base + penalty


def fit_m1_linear(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> Dict[str, Any]:
    w = 1.0 / np.maximum(yerr * yerr, 1e-18)
    a = np.column_stack([np.ones_like(x), x])
    atw = a.T * w
    ata = atw @ a
    aty = atw @ y
    try:
        beta = np.linalg.solve(ata, aty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(a, y, rcond=None)[0]
    yhat = np.maximum(beta[0] + beta[1] * x, 1e-12)
    nll = nll_from_model(y, yhat, yerr)
    aic = float(2 * 3 + 2 * nll)
    return {
        "ok": True,
        "model": "M1",
        "k": 3,
        "params": np.asarray(beta, dtype=float),
        "nll": float(nll),
        "aic": aic,
    }


def gauss(x: np.ndarray, mu: float, w: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / np.maximum(w, 1e-9)) ** 2)


def model_m2b_edge(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    s0, s1, ap, mup, wp, ad, mud, wd, e, xe, de = params
    z = np.clip((x - xe) / np.maximum(de, 1e-6), -50.0, 50.0)
    edge = 1.0 / (1.0 + np.exp(-z))
    return s0 + s1 * x + ap * gauss(x, mup, wp) + ad * gauss(x, mud, wd) + e * edge


def fit_m2b_edge(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    rng: np.random.Generator,
    n_starts: int,
    maxiter: int,
) -> Dict[str, Any]:
    bounds = list(M2B_EDGE_BOUNDS)

    def obj(p: np.ndarray) -> float:
        return nll_from_model(y, model_m2b_edge(p, x), yerr)

    y_med = float(np.median(y))
    y_span = float(max(np.max(y) - np.min(y), 0.01))
    starts: List[np.ndarray] = [
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
    ]

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    for _ in range(max(0, n_starts - 1)):
        p = lb + rng.random(len(bounds)) * (ub - lb)
        y0 = model_m2b_edge(p, x)
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
        return {"ok": False, "reason": "M2b_edge fit failed"}

    nll = float(best.fun)
    return {
        "ok": True,
        "model": "M2b_edge",
        "k": 11,
        "params": np.asarray(best.x, dtype=float),
        "nll": nll,
        "aic": float(2 * 11 + 2 * nll),
        "mu_peak": float(best.x[3]),
    }


def fit_phase_profile_models(
    x_bins: np.ndarray,
    var_bins: np.ndarray,
    var_err: np.ndarray,
    rng: np.random.Generator,
    n_starts_edge: int,
    maxiter_edge: int,
) -> Dict[str, Any]:
    m1 = fit_m1_linear(x_bins, var_bins, var_err)
    edge = fit_m2b_edge(x_bins, var_bins, var_err, rng=rng, n_starts=n_starts_edge, maxiter=maxiter_edge)
    if not edge.get("ok", False):
        return {
            "ok": False,
            "reason": edge.get("reason", "M2b_edge failed"),
            "M1": m1,
            "M2b_edge": edge,
        }
    daic = float(edge["aic"] - m1["aic"])
    mu_peak = float(edge["mu_peak"])
    delta = float(abs(mu_peak - LOG_G_DAGGER))
    return {
        "ok": True,
        "M1": m1,
        "M2b_edge": edge,
        "mu_peak": mu_peak,
        "delta": delta,
        "daic": daic,
        "aic_m1": float(m1["aic"]),
        "aic_edge": float(edge["aic"]),
    }


def shuffle_within_galaxy(
    residuals: np.ndarray,
    groups: List[np.ndarray],
    rng: np.random.Generator,
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
    residuals: np.ndarray,
    groups: List[np.ndarray],
    rng: np.random.Generator,
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


def run_phase_null_shuffles(
    x: np.ndarray,
    residuals: np.ndarray,
    galaxies: np.ndarray,
    prep: Dict[str, Any],
    delta_real: float,
    daic_real: float,
    rng: np.random.Generator,
    n_shuffles: int,
    maxiter_null: int,
    include_trials: bool,
    runlog_path: Path,
) -> Dict[str, Any]:
    groups = [np.where(galaxies == g)[0] for g in pd.unique(galaxies)]
    groups = [g for g in groups if len(g) > 0]

    out: Dict[str, Any] = {}
    for mode in ["A_within_galaxy", "B_galaxy_label"]:
        delta_null = np.full(n_shuffles, np.nan, dtype=float)
        daic_null = np.full(n_shuffles, np.nan, dtype=float)
        mu_null = np.full(n_shuffles, np.nan, dtype=float)
        n_fail = 0

        for i in range(n_shuffles):
            if mode == "A_within_galaxy":
                rs = shuffle_within_galaxy(residuals, groups, rng)
            else:
                rs = shuffle_galaxy_label(residuals, groups, rng)

            var_pack = variance_profile_from_prebinned(rs, prep, ddof=1)
            if var_pack["n_bins_used"] < 5:
                n_fail += 1
                if (i + 1) % 100 == 0:
                    log_line(runlog_path, f"[{mode}] {i+1}/{n_shuffles}")
                continue

            fit = fit_phase_profile_models(
                var_pack["x_bins"],
                var_pack["var_bins"],
                var_pack["var_err"],
                rng=rng,
                n_starts_edge=5,
                maxiter_edge=maxiter_null,
            )
            if not fit["ok"]:
                n_fail += 1
                if (i + 1) % 100 == 0:
                    log_line(runlog_path, f"[{mode}] {i+1}/{n_shuffles}")
                continue

            mu_null[i] = fit["mu_peak"]
            delta_null[i] = abs(fit["mu_peak"] - LOG_G_DAGGER)
            daic_null[i] = fit["daic"]

            if (i + 1) % 100 == 0:
                log_line(runlog_path, f"[{mode}] {i+1}/{n_shuffles}")

        p_delta = float(np.mean(np.where(np.isfinite(delta_null), delta_null <= delta_real, False)))
        p_daic = float(np.mean(np.where(np.isfinite(daic_null), daic_null <= daic_real, False)))
        d50, d05, d95 = robust_percentiles(delta_null, [50, 5, 95])
        a50, a05, a95 = robust_percentiles(daic_null, [50, 5, 95])

        entry: Dict[str, Any] = {
            "status": "OK",
            "n_shuffles": int(n_shuffles),
            "n_fail": int(n_fail),
            "p_delta": p_delta,
            "p_daic": p_daic,
            "delta_null_median": d50,
            "delta_null_5pct": d05,
            "delta_null_95pct": d95,
            "daic_null_median": a50,
            "daic_null_5pct": a05,
            "daic_null_95pct": a95,
        }
        if include_trials:
            entry["delta_trials"] = delta_null
            entry["daic_trials"] = daic_null
            entry["mu_peak_trials"] = mu_null
        out[mode] = entry

    return out


def run_test1_phase_peak(
    sparc_df: pd.DataFrame,
    out_dir: Path,
    seed_real: int,
    seed_null: int,
    n_shuffles: int,
    write_artifacts: bool,
    include_trials: bool,
    runlog_path: Path,
) -> Dict[str, Any]:
    x = sparc_df["log_gbar"].to_numpy(dtype=float)
    r = sparc_df["log_res_use"].to_numpy(dtype=float)
    g = sparc_df["galaxy"].astype(str).to_numpy()

    prep = prepare_phase_binning(x, width=PHASE_BIN_WIDTH, min_points=PHASE_MIN_POINTS)
    var_pack = variance_profile_from_prebinned(r, prep, ddof=1)

    if var_pack["n_bins_used"] < 5:
        summary = {
            "status": "BLOCKED",
            "reason": "Too few bins with >=10 points",
            "n_bins_used": int(var_pack["n_bins_used"]),
        }
        if write_artifacts:
            write_json(out_dir / "summary_phase_peak_null.json", summary)
            save_blocked_figure(out_dir / "fig_phase_null.png", "Phase Peak Null", summary["reason"])
        return summary

    rng_real = np.random.default_rng(seed_real)
    fit_real = fit_phase_profile_models(
        var_pack["x_bins"],
        var_pack["var_bins"],
        var_pack["var_err"],
        rng=rng_real,
        n_starts_edge=30,
        maxiter_edge=2500,
    )
    if not fit_real["ok"]:
        summary = {"status": "BLOCKED", "reason": fit_real.get("reason", "Real fit failed")}
        if write_artifacts:
            write_json(out_dir / "summary_phase_peak_null.json", summary)
            save_blocked_figure(out_dir / "fig_phase_null.png", "Phase Peak Null", summary["reason"])
        return summary

    delta_real = float(fit_real["delta"])
    daic_real = float(fit_real["daic"])

    rng_null = np.random.default_rng(seed_null)
    nulls = run_phase_null_shuffles(
        x=x,
        residuals=r,
        galaxies=g,
        prep=prep,
        delta_real=delta_real,
        daic_real=daic_real,
        rng=rng_null,
        n_shuffles=n_shuffles,
        maxiter_null=300,
        include_trials=include_trials,
        runlog_path=runlog_path,
    )

    summary = {
        "status": "OK",
        "test": "phase_peak_null_distribution",
        "seeds": {
            "real_fit_seed": int(seed_real),
            "null_shuffle_seed": int(seed_null),
        },
        "n_sparc_points": int(len(sparc_df)),
        "n_sparc_galaxies": int(sparc_df["galaxy"].nunique()),
        "residual_definition": str(sparc_df.attrs.get("residual_definition_used", "unknown")),
        "spec": {
            "bin_width_dex": PHASE_BIN_WIDTH,
            "min_points_per_bin": PHASE_MIN_POINTS,
            "centers_required": [-13.375, -8.125],
            "var_definition": "np.var(res, ddof=1)",
            "var_err_definition": "var * sqrt(2/(N-1))",
            "M1_k": 3,
            "M2b_edge_k": 11,
            "M2b_edge_bounds": M2B_EDGE_BOUNDS,
            "optimizer": {
                "method": "L-BFGS-B",
                "real_random_starts": 30,
                "null_random_starts": 5,
                "maxiter_real": 2500,
                "maxiter_null": 300,
            },
            "likelihood": "weighted Gaussian NLL on binned variance",
        },
        "binning": {
            "edges": prep["edges"],
            "centers": prep["centers"],
            "dropped_bins_n_lt_10": [
                {
                    "bin_index": int(b),
                    "center": float(prep["centers"][int(b)]),
                    "count": int(prep["counts"][int(b)]),
                }
                for b in prep["dropped_bins"]
            ],
            "n_bins_used": int(var_pack["n_bins_used"]),
            "bins_used": [int(b) for b in var_pack["used_bins"]],
            "counts_used": [int(n) for n in var_pack["used_counts"]],
            "centers_span_match_required": bool(
                np.isclose(prep["centers"][0], -13.375, atol=1e-12)
                and np.isclose(prep["centers"][-1], -8.125, atol=1e-12)
            ),
        },
        "var_profile": {
            "x_bins": var_pack["x_bins"],
            "var_bins": var_pack["var_bins"],
            "var_err": var_pack["var_err"],
        },
        "real": {
            "mu_peak": float(fit_real["mu_peak"]),
            "delta_from_gdagger": delta_real,
            "daic": daic_real,
            "M1": {
                "aic": float(fit_real["M1"]["aic"]),
                "nll": float(fit_real["M1"]["nll"]),
                "params": fit_real["M1"]["params"],
            },
            "M2b_edge": {
                "aic": float(fit_real["M2b_edge"]["aic"]),
                "nll": float(fit_real["M2b_edge"]["nll"]),
                "params": fit_real["M2b_edge"]["params"],
            },
        },
        "shuffle_A_within_galaxy": nulls["A_within_galaxy"],
        "shuffle_B_galaxy_label": nulls["B_galaxy_label"],
    }

    if write_artifacts:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
        arrays = [
            np.asarray(nulls["A_within_galaxy"].get("delta_trials", []), dtype=float),
            np.asarray(nulls["B_galaxy_label"].get("delta_trials", []), dtype=float),
            np.asarray(nulls["A_within_galaxy"].get("daic_trials", []), dtype=float),
            np.asarray(nulls["B_galaxy_label"].get("daic_trials", []), dtype=float),
        ]
        reals = [delta_real, delta_real, daic_real, daic_real]
        pvals = [
            float(nulls["A_within_galaxy"]["p_delta"]),
            float(nulls["B_galaxy_label"]["p_delta"]),
            float(nulls["A_within_galaxy"]["p_daic"]),
            float(nulls["B_galaxy_label"]["p_daic"]),
        ]
        titles = [
            "Shuffle A: within-galaxy Delta-null",
            "Shuffle B: galaxy-label Delta-null",
            "Shuffle A: within-galaxy DeltaAIC-null",
            "Shuffle B: galaxy-label DeltaAIC-null",
        ]
        xlabels = [
            "Delta = |mu_peak - log g_dagger|",
            "Delta = |mu_peak - log g_dagger|",
            "DeltaAIC (edge - M1)",
            "DeltaAIC (edge - M1)",
        ]

        for ax, arr, rv, pv, title, xlab in zip(axes.flat, arrays, reals, pvals, titles, xlabels):
            vals = arr[np.isfinite(arr)]
            ax.hist(vals, bins=35, color="#9ecae1", edgecolor="black", alpha=0.9)
            ax.axvline(rv, color="red", linewidth=2.0)
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel("count")
            ax.text(
                0.98,
                0.95,
                f"N={len(vals)}\\np={pv:.4g}",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(out_dir / "fig_phase_null.png", facecolor="white")
        plt.close(fig)

        write_json(out_dir / "summary_phase_peak_null.json", summary)

    return summary


# ---------------------------
# Mass-matched (Test 2)
# ---------------------------


def compute_galaxy_mass_table(
    df: pd.DataFrame,
    id_col: str,
    r_col: str,
    min_points: int = 1,
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
        M_bar = (10.0 ** lgb_out) * (r_out * KPC) ** 2 / G_SI / MSUN
        M_dyn = (10.0 ** lgo_out) * (r_out * KPC) ** 2 / G_SI / MSUN
        if not np.isfinite(M_bar) or M_bar <= 0 or not np.isfinite(M_dyn) or M_dyn <= 0:
            continue
        rows.append(
            {
                "id": str(gid),
                "n_points": n,
                "R_out_kpc": r_out,
                "log_gbar_out": lgb_out,
                "log_gobs_out": lgo_out,
                "log_Mb": float(np.log10(M_bar)),
                "log_Mdyn": float(np.log10(M_dyn)),
            }
        )
    return pd.DataFrame(rows)


def mass_match_galaxies(
    sparc_mass: pd.DataFrame,
    tng_mass: pd.DataFrame,
    caliper: float = 0.3,
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
                "sparc_id": str(row["id"]),
                "tng_id": str(t.iloc[j]["id"]),
                "sparc_log_Mb": float(row["log_Mb"]),
                "tng_log_Mb": float(t.iloc[j]["log_Mb"]),
                "abs_dlogM": float(abs(row["log_Mb"] - t.iloc[j]["log_Mb"])),
            }
        )
    return pd.DataFrame(pairs)


def eval_fit_curve(fit_edge: Dict[str, Any], xgrid: np.ndarray) -> np.ndarray:
    return model_m2b_edge(np.asarray(fit_edge["params"], dtype=float), xgrid)


def phase_fit_from_points(
    x: np.ndarray,
    r: np.ndarray,
    rng: np.random.Generator,
    n_starts_edge: int,
    min_points: int = 10,
) -> Dict[str, Any]:
    prep = prepare_phase_binning(x, width=PHASE_BIN_WIDTH, min_points=min_points)
    var_pack = variance_profile_from_prebinned(r, prep, ddof=1)
    fit = fit_phase_profile_models(
        var_pack["x_bins"],
        var_pack["var_bins"],
        var_pack["var_err"],
        rng=rng,
        n_starts_edge=n_starts_edge,
        maxiter_edge=2500,
    )
    return {
        "prep": prep,
        "var": var_pack,
        "fit": fit,
    }


def build_gal_dict(df: pd.DataFrame, id_col: str, x_col: str, r_col: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for gid, g in df.groupby(id_col, sort=False):
        out[str(gid)] = (
            g[x_col].to_numpy(dtype=float),
            g[r_col].to_numpy(dtype=float),
        )
    return out


def concat_sample_from_dict(gal_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    rs: List[np.ndarray] = []
    for gid in ids:
        x, r = gal_dict[str(gid)]
        xs.append(x)
        rs.append(r)
    return np.concatenate(xs), np.concatenate(rs)


def run_test2_mass_matched_phase(
    sparc_df: Optional[pd.DataFrame],
    tng_df: Optional[pd.DataFrame],
    tng_meta: Dict[str, Any],
    out_dir: Path,
    seed: int,
    runlog_path: Path,
) -> Dict[str, Any]:
    if sparc_df is None:
        summary = {"status": "BLOCKED", "reason": "SPARC input unavailable"}
        write_json(out_dir / "summary_mass_matched_phase.json", summary)
        save_blocked_figure(out_dir / "fig_mass_matched_phase.png", "Mass-Matched Phase", summary["reason"])
        return summary

    if tng_df is None:
        summary = {
            "status": "BLOCKED",
            "reason": "Verified clean TNG per-point dataset not available",
            "tng_validation": tng_meta,
        }
        write_json(out_dir / "summary_mass_matched_phase.json", summary)
        save_blocked_figure(out_dir / "fig_mass_matched_phase.png", "Mass-Matched Phase", summary["reason"])
        return summary

    rng = np.random.default_rng(seed)
    log_line(runlog_path, f"Test2 seed={seed}")

    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sp_mass = compute_galaxy_mass_table(sp, id_col="id", r_col="r_kpc", min_points=5)
    sp = sp[sp["id"].isin(set(sp_mass["id"]))].copy()

    tng_mass = compute_galaxy_mass_table(tng_df, id_col="id", r_col="r_kpc", min_points=5)
    tng_use = tng_df[tng_df["id"].isin(set(tng_mass["id"]))].copy()

    pairs = mass_match_galaxies(sp_mass, tng_mass, caliper=0.3)
    if len(pairs) == 0:
        summary = {
            "status": "BLOCKED",
            "reason": "No SPARC-TNG mass matched pairs within ±0.3 dex",
            "caliper_dex": 0.3,
        }
        write_json(out_dir / "summary_mass_matched_phase.json", summary)
        save_blocked_figure(out_dir / "fig_mass_matched_phase.png", "Mass-Matched Phase", summary["reason"])
        return summary

    sp_ids = set(pairs["sparc_id"])
    t_ids = set(pairs["tng_id"])
    sp_m = sp[sp["id"].isin(sp_ids)].copy()
    t_m = tng_use[tng_use["id"].isin(t_ids)].copy()

    fit_sp = phase_fit_from_points(
        sp_m["log_gbar"].to_numpy(dtype=float),
        sp_m["log_res_use"].to_numpy(dtype=float),
        rng=rng,
        n_starts_edge=30,
    )
    fit_t = phase_fit_from_points(
        t_m["log_gbar"].to_numpy(dtype=float),
        t_m["log_res_use"].to_numpy(dtype=float),
        rng=rng,
        n_starts_edge=30,
    )

    if (not fit_sp["fit"].get("ok", False)) or (not fit_t["fit"].get("ok", False)):
        summary = {
            "status": "BLOCKED",
            "reason": "Phase model fit failed on matched sets",
            "sparc_fit_ok": bool(fit_sp["fit"].get("ok", False)),
            "tng_fit_ok": bool(fit_t["fit"].get("ok", False)),
        }
        write_json(out_dir / "summary_mass_matched_phase.json", summary)
        save_blocked_figure(out_dir / "fig_mass_matched_phase.png", "Mass-Matched Phase", summary["reason"])
        return summary

    # Bootstrap 200 galaxy resamples
    n_boot = 200
    sp_dict = build_gal_dict(sp_m, "id", "log_gbar", "log_res_use")
    t_dict = build_gal_dict(t_m, "id", "log_gbar", "log_res_use")
    sp_ids_list = np.array(list(sp_dict.keys()), dtype=object)
    t_ids_list = np.array(list(t_dict.keys()), dtype=object)

    mu_sp = np.full(n_boot, np.nan, dtype=float)
    mu_t = np.full(n_boot, np.nan, dtype=float)
    daic_sp = np.full(n_boot, np.nan, dtype=float)
    daic_t = np.full(n_boot, np.nan, dtype=float)

    for i in range(n_boot):
        ids_sp = rng.choice(sp_ids_list, size=len(sp_ids_list), replace=True)
        ids_t = rng.choice(t_ids_list, size=len(t_ids_list), replace=True)
        x_sp, r_sp = concat_sample_from_dict(sp_dict, ids_sp)
        x_t, r_t = concat_sample_from_dict(t_dict, ids_t)
        bs_sp = phase_fit_from_points(x_sp, r_sp, rng=rng, n_starts_edge=8)
        bs_t = phase_fit_from_points(x_t, r_t, rng=rng, n_starts_edge=8)
        if bs_sp["fit"].get("ok", False):
            mu_sp[i] = bs_sp["fit"]["mu_peak"]
            daic_sp[i] = bs_sp["fit"]["daic"]
        if bs_t["fit"].get("ok", False):
            mu_t[i] = bs_t["fit"]["mu_peak"]
            daic_t[i] = bs_t["fit"]["daic"]
        if (i + 1) % 50 == 0:
            log_line(runlog_path, f"[test2 bootstrap] {i+1}/{n_boot}")

    sp_mass_vals = pairs["sparc_log_Mb"].to_numpy(dtype=float)
    t_mass_vals = pairs["tng_log_Mb"].to_numpy(dtype=float)
    ks = ks_2samp(sp_mass_vals, t_mass_vals)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    ax0, ax1 = axes

    xgrid = np.linspace(-13.375, -8.125, 400)
    for label, fit_obj, color, marker in [
        ("SPARC", fit_sp, "#1f77b4", "o"),
        ("TNG", fit_t, "#2ca02c", "s"),
    ]:
        ax0.errorbar(
            fit_obj["var"]["x_bins"],
            fit_obj["var"]["var_bins"],
            yerr=fit_obj["var"]["var_err"],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
            label=f"{label} binned var",
        )
        ax0.plot(xgrid, eval_fit_curve(fit_obj["fit"]["M2b_edge"], xgrid), color=color, linewidth=2.0, alpha=0.8)
    ax0.axvline(LOG_G_DAGGER, color="red", linestyle="--", linewidth=1.2, label="log g_dagger")
    ax0.set_xlabel("log_gbar")
    ax0.set_ylabel("variance(log_res)")
    ax0.set_title("Mass-matched phase profiles")
    ax0.legend(frameon=False, fontsize=8)

    bins = np.linspace(min(sp_mass_vals.min(), t_mass_vals.min()), max(sp_mass_vals.max(), t_mass_vals.max()), 20)
    ax1.hist(sp_mass_vals, bins=bins, alpha=0.6, label="SPARC", color="#1f77b4", edgecolor="black")
    ax1.hist(t_mass_vals, bins=bins, alpha=0.6, label="TNG", color="#2ca02c", edgecolor="black")
    ax1.set_xlabel("log10(M_bar/Msun)")
    ax1.set_ylabel("count")
    ax1.set_title("Matched mass distributions")
    ax1.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_mass_matched_phase.png", facecolor="white")
    plt.close(fig)

    summary = {
        "status": "OK",
        "test": "mass_matched_phase",
        "seed": int(seed),
        "caliper_dex": 0.3,
        "N_matched": int(len(pairs)),
        "tng_source": tng_meta,
        "mass_range": [float(min(sp_mass_vals.min(), t_mass_vals.min())), float(max(sp_mass_vals.max(), t_mass_vals.max()))],
        "ks_pvalue_mass": float(ks.pvalue),
        "sparc": {
            "n_points": int(len(sp_m)),
            "n_galaxies": int(sp_m["id"].nunique()),
            "mu_peak": float(fit_sp["fit"]["mu_peak"]),
            "mu_peak_ci95": robust_percentiles(mu_sp, [2.5, 97.5]),
            "daic": float(fit_sp["fit"]["daic"]),
            "daic_ci95": robust_percentiles(daic_sp, [2.5, 97.5]),
            "n_bins_used": int(fit_sp["var"]["n_bins_used"]),
        },
        "tng": {
            "n_points": int(len(t_m)),
            "n_galaxies": int(t_m["id"].nunique()),
            "mu_peak": float(fit_t["fit"]["mu_peak"]),
            "mu_peak_ci95": robust_percentiles(mu_t, [2.5, 97.5]),
            "daic": float(fit_t["fit"]["daic"]),
            "daic_ci95": robust_percentiles(daic_t, [2.5, 97.5]),
            "n_bins_used": int(fit_t["var"]["n_bins_used"]),
        },
    }
    write_json(out_dir / "summary_mass_matched_phase.json", summary)
    return summary


# ---------------------------
# Xi organizing (Test 3)
# ---------------------------


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
        payload.append({"id": str(gid), "logX": lx, "res": res, "xi_kpc": xi, "n_points": int(len(r))})
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


def xi_permutation_null(
    payload: List[Dict[str, Any]],
    centers: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    runlog_path: Path,
    name: str,
) -> np.ndarray:
    n_g = len(payload)
    if n_g == 0:
        return np.array([], dtype=float)

    lengths = np.array([len(p["res"]) for p in payload], dtype=int)
    fixed_len = np.all(lengths == lengths[0])
    large = n_g > 1000 and fixed_len and lengths[0] <= 64
    edges = np.linspace(-2.0, 1.5, 9)
    n_bins = len(edges) - 1
    cnull = np.full(n_perm, np.nan, dtype=float)

    if large:
        res_arr = np.vstack([p["res"] for p in payload]).astype(float)
        logx_arr = np.vstack([p["logX"] for p in payload]).astype(float)
        bin_arr = np.digitize(logx_arr, edges) - 1
        mask_bins = [(bin_arr == b).astype(float) for b in range(n_bins)]
        counts = np.stack([m.sum(axis=1) for m in mask_bins], axis=1)
        valid = counts >= 2

        for i in range(n_perm):
            rp = rng.permuted(res_arr, axis=1)
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
            cnull[i] = concentration_from_profile(centers, mean_prof)
            if (i + 1) % 100 == 0:
                log_line(runlog_path, f"[{name} xi perm vectorized] {i+1}/{n_perm}")
        return cnull

    logx_list = [p["logX"] for p in payload]
    res_list = [p["res"] for p in payload]
    for i in range(n_perm):
        rr = [r[rng.permutation(len(r))] for r in res_list]
        _, vmat = stacked_variance_profile(logx_list, rr)
        mean_prof = np.nanmean(vmat, axis=0)
        cnull[i] = concentration_from_profile(centers, mean_prof)
        if (i + 1) % 100 == 0:
            log_line(runlog_path, f"[{name} xi perm] {i+1}/{n_perm}")
    return cnull


def xi_dataset_summary(
    name: str,
    df: pd.DataFrame,
    id_col: str,
    r_col: str,
    res_col: str,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
    runlog_path: Path,
) -> Dict[str, Any]:
    payload = per_galaxy_xi_payload(df, id_col=id_col, r_col=r_col, res_col=res_col)
    if len(payload) == 0:
        return {"status": "BLOCKED", "reason": f"{name}: no galaxies with >=8 points"}

    logx_list = [p["logX"] for p in payload]
    res_list = [p["res"] for p in payload]
    centers, vmat = stacked_variance_profile(logx_list, res_list)
    mean_prof = np.nanmean(vmat, axis=0)

    boot = np.full((n_boot, len(centers)), np.nan, dtype=float)
    n_g = len(payload)
    for i in range(n_boot):
        idx = rng.integers(0, n_g, size=n_g)
        boot[i] = np.nanmean(vmat[idx], axis=0)
        if (i + 1) % 100 == 0:
            log_line(runlog_path, f"[{name} xi bootstrap] {i+1}/{n_boot}")
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    c_real = concentration_from_profile(centers, mean_prof)
    cnull = xi_permutation_null(payload, centers, rng=rng, n_perm=n_perm, runlog_path=runlog_path, name=name)
    p_c = float(np.mean(np.where(np.isfinite(cnull), cnull >= c_real, False)))

    peaks = []
    for p in payload:
        if p["n_points"] < 10:
            continue
        c2, v2 = stacked_variance_profile([p["logX"]], [p["res"]])
        vv = v2[0]
        if np.isfinite(vv).sum() == 0:
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

    c50, c5, c95 = robust_percentiles(cnull, [50, 5, 95])
    return {
        "status": "OK",
        "dataset": name,
        "n_galaxies": int(n_g),
        "X_bins_log10_center": centers,
        "stacked_variance": mean_prof,
        "stacked_variance_ci95_low": lo,
        "stacked_variance_ci95_high": hi,
        "concentration_C": c_real,
        "concentration_null_median": c50,
        "concentration_null_5pct": c5,
        "concentration_null_95pct": c95,
        "concentration_pvalue": p_c,
        "X_peak_stats": peak_stats,
        "concentration_null_samples": cnull,
    }


def run_test3_xi_organizing(
    sparc_df: Optional[pd.DataFrame],
    tng_df: Optional[pd.DataFrame],
    out_dir: Path,
    seed: int,
    runlog_path: Path,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    log_line(runlog_path, f"Test3 seed={seed}")

    if sparc_df is None:
        summary = {"status": "BLOCKED", "reason": "SPARC input unavailable"}
        write_json(out_dir / "summary_xi_organizing.json", summary)
        save_blocked_figure(out_dir / "fig_xi_organizing.png", "Xi Organizing", summary["reason"])
        return summary

    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    sparc_res = xi_dataset_summary(
        "SPARC",
        sp,
        id_col="id",
        r_col="r_kpc",
        res_col="log_res_use",
        rng=rng,
        n_boot=500,
        n_perm=1000,
        runlog_path=runlog_path,
    )

    if tng_df is None:
        tng_res = {"status": "BLOCKED", "reason": "Verified clean TNG per-point dataset not available"}
    else:
        tng_res = xi_dataset_summary(
            "TNG",
            tng_df,
            id_col="id",
            r_col="r_kpc",
            res_col="log_res_use",
            rng=rng,
            n_boot=500,
            n_perm=1000,
            runlog_path=runlog_path,
        )

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    datasets = [("SPARC", sparc_res), ("TNG", tng_res)]
    for j, (name, d) in enumerate(datasets):
        ax = axes[0, j]
        if d.get("status") == "OK":
            x = 10.0 ** np.asarray(d["X_bins_log10_center"], dtype=float)
            y = np.asarray(d["stacked_variance"], dtype=float)
            lo = np.asarray(d["stacked_variance_ci95_low"], dtype=float)
            hi = np.asarray(d["stacked_variance_ci95_high"], dtype=float)
            ax.plot(x, y, marker="o", color="#1f77b4")
            ax.fill_between(x, lo, hi, color="#9ecae1", alpha=0.5)
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2)
            ax.set_xscale("log")
            ax.set_xlabel("X = R/xi")
            ax.set_ylabel("stacked variance")
            ax.set_title(f"{name} stacked variance vs X")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"BLOCKED\\n{d.get('reason', '')}", ha="center", va="center")
            ax.set_title(f"{name} stacked variance vs X")

    for j, (name, d) in enumerate(datasets):
        ax = axes[1, j]
        if d.get("status") == "OK":
            cnull = np.asarray(d["concentration_null_samples"], dtype=float)
            cnull = cnull[np.isfinite(cnull)]
            ax.hist(cnull, bins=35, color="#9ecae1", edgecolor="black")
            ax.axvline(float(d["concentration_C"]), color="red", linewidth=2)
            ax.set_xlabel("C null")
            ax.set_ylabel("count")
            ax.set_title(f"{name} C-null (p={float(d['concentration_pvalue']):.4g})")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"BLOCKED\\n{d.get('reason', '')}", ha="center", va="center")
            ax.set_title(f"{name} C-null")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_xi_organizing.png", facecolor="white")
    plt.close(fig)

    summary = {
        "status": "OK" if sparc_res.get("status") == "OK" else "BLOCKED",
        "test": "xi_organizing",
        "seed": int(seed),
        "spec": {
            "x_bins_log10_edges": np.linspace(-2.0, 1.5, 9),
            "x_bins_log10_count": 8,
            "bootstrap_galaxy_resamples": 500,
            "permutation_nulls": 1000,
            "concentration_definition": "C = mean(var|0.3<=X<=3.0) / mean(var overall)",
            "perm_definition": "within each galaxy shuffle R_kpc <-> log_res",
        },
        "sparc": sparc_res,
        "tng": tng_res,
    }
    write_json(out_dir / "summary_xi_organizing.json", summary)
    return summary


# ---------------------------
# alpha* convergence (Test 4)
# ---------------------------


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
        return {"status": "BLOCKED", "reason": "Too few galaxies", "n_gal": int(len(df))}

    q = df["q"].to_numpy(dtype=float)
    m = df["m"].to_numpy(dtype=float)
    a = alpha_star_closed_form(q, m)
    if not np.isfinite(a):
        return {"status": "BLOCKED", "reason": "alpha* undefined", "n_gal": int(len(df))}

    z = q + a * m
    sc = float(np.std(z, ddof=1)) if len(z) >= 2 else None

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


def build_sparc_alpha_table(gal_df: pd.DataFrame, sparc_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if "sigma_res" not in gal_df.columns:
        return None, "sigma_res column missing in galaxy_results_unified.csv"

    g = gal_df.copy()
    if "source" in g.columns:
        g = g[g["source"] == "SPARC"].copy()
    if "galaxy" not in g.columns:
        return None, "galaxy column missing in galaxy_results_unified.csv"

    g["sigma_res"] = pd.to_numeric(g["sigma_res"], errors="coerce")
    g = g[np.isfinite(g["sigma_res"]) & (g["sigma_res"] > 0)].copy()
    g["q"] = np.log10(g["sigma_res"].to_numpy(dtype=float) ** 2)

    sp = sparc_df.rename(columns={"galaxy": "id", "R_kpc": "r_kpc"}).copy()
    mass = compute_galaxy_mass_table(sp, id_col="id", r_col="r_kpc", min_points=5)
    mass = mass.rename(columns={"log_Mb": "m"})

    mass_use = mass[["id", "m", "n_points"]].rename(columns={"n_points": "n_points_mass"})
    merged = g.merge(mass_use, left_on="galaxy", right_on="id", how="inner")
    merged = merged[np.isfinite(merged["q"]) & np.isfinite(merged["m"])].copy()

    out = (
        merged[["galaxy", "q", "m", "n_points_mass"]]
        .rename(columns={"galaxy": "id", "n_points_mass": "n_points"})
        .copy()
    )
    out["id"] = out["id"].astype(str)
    return out, None


def build_tng_alpha_table(tng_df: pd.DataFrame) -> pd.DataFrame:
    mass = compute_galaxy_mass_table(tng_df, id_col="id", r_col="r_kpc", min_points=5)
    sig_rows = []
    for gid, g in tng_df.groupby("id", sort=False):
        rr = g["log_res_use"].to_numpy(dtype=float)
        rr = rr[np.isfinite(rr)]
        if len(rr) < 5:
            continue
        s = float(np.std(rr, ddof=1))
        if not np.isfinite(s) or s <= 0:
            continue
        sig_rows.append({"id": str(gid), "q": float(np.log10(s * s)), "n_points": int(len(rr))})
    sig = pd.DataFrame(sig_rows)
    if len(sig) == 0 or len(mass) == 0:
        return pd.DataFrame(columns=["id", "q", "m", "n_points"])
    out = mass[["id", "log_Mb", "n_points"]].rename(columns={"log_Mb": "m"}).merge(sig, on=["id", "n_points"], how="inner")
    out = out[["id", "q", "m", "n_points"]].copy()
    out["id"] = out["id"].astype(str)
    return out


def run_test4_alpha_convergence(
    gal_df: Optional[pd.DataFrame],
    sparc_df: Optional[pd.DataFrame],
    tng_df: Optional[pd.DataFrame],
    out_dir: Path,
    seed: int,
    runlog_path: Path,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    log_line(runlog_path, f"Test4 seed={seed}")

    if gal_df is None or sparc_df is None:
        summary = {"status": "BLOCKED", "reason": "Required SPARC/galaxy inputs unavailable"}
        write_json(out_dir / "summary_alpha_star_convergence.json", summary)
        save_blocked_figure(out_dir / "fig_alpha_star.png", "alpha* Convergence", summary["reason"])
        return summary

    sp_table, sp_err = build_sparc_alpha_table(gal_df, sparc_df)
    if sp_table is None:
        summary = {"status": "BLOCKED", "reason": f"SPARC alpha table build failed: {sp_err}"}
        write_json(out_dir / "summary_alpha_star_convergence.json", summary)
        save_blocked_figure(out_dir / "fig_alpha_star.png", "alpha* Convergence", summary["reason"])
        return summary

    tng_table = build_tng_alpha_table(tng_df) if tng_df is not None else pd.DataFrame(columns=["id", "q", "m", "n_points"])

    datasets: Dict[str, Optional[pd.DataFrame]] = {
        "SPARC": sp_table,
        "TNG": tng_table if len(tng_table) > 0 else None,
    }

    # Condition definitions
    n_threshold = 10
    has_tng = datasets["TNG"] is not None

    if has_tng:
        spm = datasets["SPARC"]["m"]
        tm = datasets["TNG"]["m"]
        mass_overlap = (max(float(spm.min()), float(tm.min())), min(float(spm.max()), float(tm.max())))
        if mass_overlap[1] <= mass_overlap[0]:
            mass_overlap = None
    else:
        mass_overlap = None

    conditions = [
        ("Full", {"mass": None, "nmin": None}),
        ("Mass-matched overlap", {"mass": mass_overlap, "nmin": None}),
        ("Resolution-matched", {"mass": None, "nmin": n_threshold}),
        ("Mass+Resolution", {"mass": mass_overlap, "nmin": n_threshold}),
    ]

    by_condition: Dict[str, Any] = {}
    table_rows = []

    for cname, cdef in conditions:
        csum: Dict[str, Any] = {
            "definition": {
                "mass_overlap": cdef["mass"],
                "resolution_nmin": cdef["nmin"],
            },
            "datasets": {},
        }
        for dname, ddf in datasets.items():
            if ddf is None:
                csum["datasets"][dname] = {"status": "BLOCKED", "reason": "dataset unavailable"}
                continue
            x = ddf.copy()
            if cdef["mass"] is not None:
                lo, hi = cdef["mass"]
                x = x[(x["m"] >= lo) & (x["m"] <= hi)].copy()
            elif cname in ["Mass-matched overlap", "Mass+Resolution"] and cdef["mass"] is None:
                csum["datasets"][dname] = {
                    "status": "BLOCKED",
                    "reason": "mass overlap unavailable (no TNG overlap)",
                }
                continue
            if cdef["nmin"] is not None:
                x = x[x["n_points"] >= int(cdef["nmin"])].copy()

            st = alpha_condition_stats(x, rng=rng, n_boot=200)
            csum["datasets"][dname] = st
            if st.get("status") == "OK":
                table_rows.append(
                    {
                        "condition": cname,
                        "dataset": dname,
                        "n_gal": int(st["n_gal"]),
                        "alpha_star": float(st["alpha_star"]),
                        "alpha_ci_low": st["alpha_ci95"][0],
                        "alpha_ci_high": st["alpha_ci95"][1],
                        "scatter_z": st["scatter_z"],
                    }
                )

        vals = [
            csum["datasets"][d]["alpha_star"]
            for d in csum["datasets"]
            if csum["datasets"][d].get("status") == "OK"
        ]
        csum["delta_alpha_max"] = float(np.max(vals) - np.min(vals)) if len(vals) >= 2 else None
        by_condition[cname] = csum

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    cond_names = [c[0] for c in conditions]
    colors = {"SPARC": "#1f77b4", "TNG": "#2ca02c"}
    offsets = {"SPARC": -0.12, "TNG": 0.12}

    has_any = False
    for i, cname in enumerate(cond_names):
        for dname in ["SPARC", "TNG"]:
            d = by_condition[cname]["datasets"].get(dname, {})
            if d.get("status") != "OK":
                continue
            has_any = True
            a = float(d["alpha_star"])
            lo, hi = d["alpha_ci95"]
            y = i + offsets[dname]
            ax.errorbar(
                a,
                y,
                xerr=[[a - lo], [hi - a]],
                fmt="o",
                color=colors[dname],
                capsize=3,
                label=dname if i == 0 else None,
            )

    ax.set_yticks(range(len(cond_names)))
    ax.set_yticklabels(cond_names)
    ax.set_xlabel("alpha*")
    ax.set_title("alpha* convergence under matching")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    if has_any:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "BLOCKED", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_alpha_star.png", facecolor="white")
    plt.close(fig)

    summary = {
        "status": "OK" if len(table_rows) > 0 else "BLOCKED",
        "test": "alpha_star_convergence",
        "seed": int(seed),
        "alpha_definition": {
            "q_i": "log10(sigma_res_i^2)",
            "m_i": "log10(M_bar_i) from outermost-radius mass in per-point data",
            "alpha_star": "-Cov(q,m)/Var(m)",
            "z_i": "q_i + alpha_star * m_i",
            "resolution_threshold_N": int(n_threshold),
        },
        "conditions": by_condition,
        "table": table_rows,
    }

    write_json(out_dir / "summary_alpha_star_convergence.json", summary)
    return summary


# ---------------------------
# Dataset lineage (Test 5)
# ---------------------------


def source_stats_from_points(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for src, g in df.groupby("source", sort=True):
        ppg = g.groupby("galaxy").size()
        env_col = "env_dense" if "env_dense" in g.columns else None
        if env_col is not None:
            env = g[env_col].astype(str).str.lower()
            n_field = int((env == "field").sum())
            n_dense = int((env == "dense").sum())
            n_missing = int((env == "nan").sum() + env.isna().sum())
        else:
            n_field = None
            n_dense = None
            n_missing = int(len(g))

        out[str(src)] = {
            "n_galaxies": int(g["galaxy"].nunique()),
            "n_points": int(len(g)),
            "points_per_galaxy": {
                "min": int(ppg.min()) if len(ppg) else None,
                "median": float(ppg.median()) if len(ppg) else None,
                "max": int(ppg.max()) if len(ppg) else None,
            },
            "log_gbar_range": [float(g["log_gbar"].min()), float(g["log_gbar"].max())],
            "env_dense_breakdown": {
                "n_field": n_field,
                "n_dense": n_dense,
                "n_missing": n_missing,
            },
        }
    return out


def source_stats_from_galaxy(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for src, g in df.groupby("source", sort=True):
        g2 = g.copy()
        if "sigma_res" in g2.columns:
            g2["sigma_res"] = pd.to_numeric(g2["sigma_res"], errors="coerce")
            q1 = float(g2["sigma_res"].quantile(0.25)) if g2["sigma_res"].notna().any() else np.nan
            q3 = float(g2["sigma_res"].quantile(0.75)) if g2["sigma_res"].notna().any() else np.nan
            sig_med = float(g2["sigma_res"].median()) if g2["sigma_res"].notna().any() else np.nan
            sig_iqr = float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else np.nan
        else:
            sig_med = np.nan
            sig_iqr = np.nan

        if "logMh" in g2.columns:
            g2["logMh"] = pd.to_numeric(g2["logMh"], errors="coerce")
            logmh_range = [
                float(g2["logMh"].min()) if g2["logMh"].notna().any() else None,
                float(g2["logMh"].max()) if g2["logMh"].notna().any() else None,
            ]
        else:
            logmh_range = [None, None]

        out[str(src)] = {
            "n_galaxies": int(g2["galaxy"].nunique()) if "galaxy" in g2.columns else None,
            "n_rows": int(len(g2)),
            "sigma_res_median": sig_med,
            "sigma_res_IQR": sig_iqr,
            "logMh_range": logmh_range,
        }
    return out


def extract_historical_ratios(rar_points_path: Optional[str]) -> Dict[str, Any]:
    result = {
        "historical_ratio_values": None,
        "status": "unknown",
        "source_files": [],
        "explanation": "No known historical ratio table found",
    }
    if rar_points_path is None:
        return result

    rp = Path(rar_points_path).resolve()
    candidates: List[Path] = []
    for parent in rp.parents:
        c = parent / "analysis/results/tng_sparc_composition_sweep/fairness_gap_vs_threshold.csv"
        if c.exists():
            candidates.append(c)
        if len(candidates) >= 5:
            break

    ratios: List[float] = []
    for c in candidates:
        try:
            df = pd.read_csv(c)
            ratio_cols = [col for col in df.columns if "ratio" in col.lower()]
            for col in ratio_cols:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                ratios.extend([float(v) for v in vals])
            result["source_files"].append(str(c))
        except Exception:
            continue

    if len(ratios) > 0:
        uniq = sorted(set([round(v, 6) for v in ratios]))
        result["historical_ratio_values"] = uniq[:30]
        result["status"] = "present"
        result["explanation"] = "Extracted from discovered fairness/composition ratio columns"
    return result


def run_test5_dataset_lineage(
    rar_points_path: Optional[str],
    galaxy_results_path: Optional[str],
    tng_manifest: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    if rar_points_path is None or galaxy_results_path is None:
        summary = {
            "status": "BLOCKED",
            "reason": "Required input file missing",
            "rar_points_path": rar_points_path,
            "galaxy_results_path": galaxy_results_path,
        }
        write_json(out_dir / "summary_dataset_lineage.json", summary)
        (out_dir / "dataset_lineage.md").write_text("# Dataset Lineage\n\nBLOCKED: required input file missing.\n")
        return summary

    try:
        rp = pd.read_csv(rar_points_path)
        gr = pd.read_csv(galaxy_results_path)
    except Exception as exc:
        summary = {
            "status": "BLOCKED",
            "reason": f"Failed reading inputs: {exc}",
            "rar_points_path": rar_points_path,
            "galaxy_results_path": galaxy_results_path,
        }
        write_json(out_dir / "summary_dataset_lineage.json", summary)
        (out_dir / "dataset_lineage.md").write_text(f"# Dataset Lineage\n\nBLOCKED: {exc}\n")
        return summary

    points_stats = source_stats_from_points(rp)
    galaxy_stats = source_stats_from_galaxy(gr)

    hist = extract_historical_ratios(rar_points_path)
    if hist["status"] == "present":
        contamination_note = {
            "status": "present",
            "historical_ratio_values": hist["historical_ratio_values"],
            "source_files": hist["source_files"],
            "explanation": hist["explanation"],
        }
    else:
        contamination_note = {
            "status": "unknown",
            "historical_ratio_values": "unknown",
            "source_files": hist["source_files"],
            "explanation": "Known historical ratio values were not found in discoverable fairness tables; leaving as unknown.",
        }

    summary = {
        "status": "OK",
        "test": "dataset_lineage_audit",
        "rar_points_path": rar_points_path,
        "galaxy_results_path": galaxy_results_path,
        "per_source_points": points_stats,
        "per_source_galaxy": galaxy_stats,
        "tng_manifest": tng_manifest,
        "contamination_note": contamination_note,
    }
    write_json(out_dir / "summary_dataset_lineage.json", summary)

    # Markdown report
    lines: List[str] = []
    lines.append("# Dataset Lineage Audit")
    lines.append("")
    lines.append("## Point-level Sources")
    lines.append("")
    lines.append("| source | n_galaxies | n_points | ppg_min | ppg_med | ppg_max | log_gbar_min | log_gbar_max | n_field | n_dense | n_missing |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for src, st in sorted(points_stats.items()):
        ppg = st["points_per_galaxy"]
        env = st["env_dense_breakdown"]
        lg = st["log_gbar_range"]
        ppg_med = ppg["median"] if ppg["median"] is not None else float("nan")
        n_field = env["n_field"] if env["n_field"] is not None else "NA"
        n_dense = env["n_dense"] if env["n_dense"] is not None else "NA"
        lines.append(
            f"| {src} | {st['n_galaxies']} | {st['n_points']} | {ppg['min']} | {ppg_med:.1f} | {ppg['max']} | {lg[0]:.3f} | {lg[1]:.3f} | {n_field} | {n_dense} | {env['n_missing']} |"
        )

    lines.append("")
    lines.append("## Galaxy-level Sources")
    lines.append("")
    lines.append("| source | n_galaxies | sigma_res_median | sigma_res_IQR | logMh_min | logMh_max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for src, st in sorted(galaxy_stats.items()):
        lmh = st["logMh_range"]
        s_med = st["sigma_res_median"]
        s_iqr = st["sigma_res_IQR"]
        lines.append(
            f"| {src} | {st['n_galaxies']} | {s_med if s_med is not None else 'NA'} | {s_iqr if s_iqr is not None else 'NA'} | {lmh[0]} | {lmh[1]} |"
        )

    lines.append("")
    lines.append("## Contamination Note")
    lines.append("")
    if contamination_note["status"] == "present":
        lines.append(f"- Historical ratio values (discovered): {contamination_note['historical_ratio_values']}")
        lines.append(f"- Source files: {contamination_note['source_files']}")
    else:
        lines.append("- Historical ratio values: unknown")
        lines.append(f"- Explanation: {contamination_note['explanation']}")

    (out_dir / "dataset_lineage.md").write_text("\n".join(lines) + "\n")
    return summary


# ---------------------------
# TNG gate (Step 6)
# ---------------------------


def run_tng_validation(
    chosen_tng_path: Optional[str],
    out_dir: Path,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    if chosen_tng_path is None:
        summary = {
            "status": "BLOCKED",
            "reason": "No verified clean TNG per-point dataset discovered",
            "chosen_tng_path": None,
        }
        write_json(out_dir / "tng_validation.json", summary)
        return summary, None

    p = Path(chosen_tng_path)
    if not p.exists():
        summary = {
            "status": "BLOCKED",
            "reason": "Chosen TNG path does not exist",
            "chosen_tng_path": str(p),
        }
        write_json(out_dir / "tng_validation.json", summary)
        return summary, None

    try:
        df, meta = load_tng_points(p)
    except Exception as exc:
        summary = {
            "status": "BLOCKED",
            "reason": str(exc),
            "chosen_tng_path": str(p),
        }
        write_json(out_dir / "tng_validation.json", summary)
        return summary, None

    ppg = df.groupby("id").size()
    keep_ids = set(ppg[ppg >= 5].index.astype(str))
    df_use = df[df["id"].isin(keep_ids)].copy()

    summary = {
        "status": "OK",
        "chosen_tng_path": str(p),
        "meta": meta,
        "inclusion_rule": ">=5 points per galaxy",
        "n_points_before_nmin5": int(len(df)),
        "n_galaxies_before_nmin5": int(df["id"].nunique()),
        "n_points_after_nmin5": int(len(df_use)),
        "n_galaxies_after_nmin5": int(df_use["id"].nunique()),
        "conversion_formula_doc": {
            "log_gbar": meta.get("log_gbar_formula"),
            "log_gobs": meta.get("log_gobs_formula"),
            "residual": meta.get("residual_formula"),
        },
    }
    write_json(out_dir / "tng_validation.json", summary)
    return summary, df_use


# ---------------------------
# Consistency checks
# ---------------------------


def run_consistency_checks(
    out_dir: Path,
    sparc_df: Optional[pd.DataFrame],
    test1_summary: Dict[str, Any],
    test2_summary: Dict[str, Any],
    seed_real: int,
    seed_null: int,
    n_shuffles: int,
    runlog_path: Path,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "OK",
        "residual_definition_used": test1_summary.get("residual_definition"),
        "phase_bins": test1_summary.get("binning", {}),
        "constants": {
            "log_gdagger": LOG_G_DAGGER,
            "gdagger_SI": 1.2e-10,
            "G_SI": G_SI,
            "MSUN": MSUN,
            "KPC_m": KPC,
        },
        "match_counts_and_ranges": {
            "test2_status": test2_summary.get("status"),
            "N_matched": test2_summary.get("N_matched"),
            "mass_range": test2_summary.get("mass_range"),
        },
    }

    if sparc_df is None or test1_summary.get("status") != "OK":
        summary["determinism_check"] = {
            "status": "BLOCKED",
            "reason": "Test1 baseline unavailable",
        }
        write_json(out_dir / "summary_consistency_checks.json", summary)
        return summary

    log_line(runlog_path, "Starting determinism rerun of Test1 with identical seeds")
    rerun = run_test1_phase_peak(
        sparc_df=sparc_df,
        out_dir=out_dir,
        seed_real=seed_real,
        seed_null=seed_null,
        n_shuffles=n_shuffles,
        write_artifacts=False,
        include_trials=False,
        runlog_path=runlog_path,
    )

    keys = [
        ("real.mu_peak", test1_summary["real"]["mu_peak"], rerun.get("real", {}).get("mu_peak")),
        ("real.daic", test1_summary["real"]["daic"], rerun.get("real", {}).get("daic")),
        (
            "A.p_delta",
            test1_summary["shuffle_A_within_galaxy"]["p_delta"],
            rerun.get("shuffle_A_within_galaxy", {}).get("p_delta"),
        ),
        (
            "A.p_daic",
            test1_summary["shuffle_A_within_galaxy"]["p_daic"],
            rerun.get("shuffle_A_within_galaxy", {}).get("p_daic"),
        ),
        (
            "B.p_delta",
            test1_summary["shuffle_B_galaxy_label"]["p_delta"],
            rerun.get("shuffle_B_galaxy_label", {}).get("p_delta"),
        ),
        (
            "B.p_daic",
            test1_summary["shuffle_B_galaxy_label"]["p_daic"],
            rerun.get("shuffle_B_galaxy_label", {}).get("p_daic"),
        ),
    ]

    diffs = []
    deterministic = True
    for k, a, b in keys:
        if a is None or b is None:
            deterministic = False
            diffs.append({"metric": k, "first": a, "second": b, "match": False})
            continue
        match = bool(np.isclose(float(a), float(b), atol=0.0, rtol=0.0))
        if not match:
            deterministic = False
        diffs.append(
            {
                "metric": k,
                "first": float(a),
                "second": float(b),
                "abs_diff": float(abs(float(a) - float(b))),
                "match": match,
            }
        )

    summary["determinism_check"] = {
        "status": "OK" if deterministic else "FAIL",
        "seed_real": int(seed_real),
        "seed_null": int(seed_null),
        "n_shuffles": int(n_shuffles),
        "metrics": diffs,
    }
    if not deterministic:
        summary["status"] = "BLOCKED"

    write_json(out_dir / "summary_consistency_checks.json", summary)
    return summary


# ---------------------------
# Output index
# ---------------------------


def file_sha256_short(path: Path, n_hex: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def write_outputs_index(out_dir: Path) -> Dict[str, Any]:
    rows = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rows.append(
                {
                    "path": str(p.relative_to(out_dir)),
                    "size_bytes": int(p.stat().st_size),
                    "sha256_short": file_sha256_short(p),
                }
            )
    payload = {
        "status": "OK",
        "output_dir": str(out_dir),
        "n_files": int(len(rows)),
        "files": rows,
    }
    write_json(out_dir / "outputs_index.json", payload)
    return payload


# ---------------------------
# Main orchestrator
# ---------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-room full rerun of referee battery")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    args = parser.parse_args()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / "rerun_outputs" / run_stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    runlog_path = out_dir / "runlog.txt"

    # Step 0: environment + provenance
    seeds = {
        "global": int(args.seed),
        "input_validation_sample": int(args.seed + 1),
        "test1_real": int(args.seed + 10),
        "test1_null": int(args.seed + 11),
        "test2": int(args.seed + 20),
        "test3": int(args.seed + 30),
        "test4": int(args.seed + 40),
        "consistency_test1_real": int(args.seed + 10),
        "consistency_test1_null": int(args.seed + 11),
    }

    log_line(runlog_path, f"timestamp={datetime.now().isoformat(timespec='seconds')}")
    log_line(runlog_path, f"hostname={socket.gethostname()}")
    log_line(runlog_path, f"working_directory={Path.cwd()}")
    log_line(runlog_path, f"python_version={sys.version.replace(chr(10), ' ')}")

    try:
        import numpy
        import scipy
        import pandas
        import matplotlib

        log_line(runlog_path, f"numpy={numpy.__version__}")
        log_line(runlog_path, f"scipy={scipy.__version__}")
        log_line(runlog_path, f"pandas={pandas.__version__}")
        log_line(runlog_path, f"matplotlib={matplotlib.__version__}")
    except Exception as exc:
        log_line(runlog_path, f"package_version_logging_failed={exc}")

    git_hash = None
    git_status = None
    if (Path.cwd() / ".git").exists():
        g1 = run_shell("git rev-parse HEAD")
        g2 = run_shell("git status --short")
        if g1["returncode"] == 0 and g1["stdout_lines"]:
            git_hash = g1["stdout_lines"][0].strip()
        git_status = g2["stdout_lines"] if g2["returncode"] == 0 else None
    log_line(runlog_path, f"git_hash={git_hash}")
    log_line(runlog_path, f"git_status_lines={len(git_status) if git_status is not None else None}")
    log_line(runlog_path, f"rng_seeds={seeds}")

    # Step 1: data discovery
    manifest = discover_data(runlog_path)
    write_json(out_dir / "data_manifest.json", manifest)

    # Step 2: input validation
    summary_input, rar_df, sparc_df = validate_inputs(
        manifest.get("rar_points_path"),
        sample_seed=seeds["input_validation_sample"],
        runlog_path=runlog_path,
    )
    write_json(out_dir / "summary_input_validation.json", summary_input)

    if sparc_df is not None:
        sparc_df.attrs["residual_definition_used"] = summary_input.get("residual_consistency_full", {}).get(
            "residual_definition_used", "unknown"
        )
        log_line(
            runlog_path,
            f"SPARC counts: n_points={len(sparc_df)}, n_galaxies={sparc_df['galaxy'].nunique()}",
        )

    # Step 3: common functions are provided via common_rar.py (imported)
    log_line(runlog_path, "Common module loaded: common_rar.py")

    # Step 4: Test 1
    if sparc_df is None:
        test1_summary = {"status": "BLOCKED", "reason": "SPARC validation unavailable"}
        write_json(out_dir / "summary_phase_peak_null.json", test1_summary)
        save_blocked_figure(out_dir / "fig_phase_null.png", "Phase Peak Null", test1_summary["reason"])
    else:
        try:
            log_line(runlog_path, f"Running Test1 with seeds real={seeds['test1_real']} null={seeds['test1_null']}")
            test1_summary = run_test1_phase_peak(
                sparc_df=sparc_df,
                out_dir=out_dir,
                seed_real=seeds["test1_real"],
                seed_null=seeds["test1_null"],
                n_shuffles=int(args.n_shuffles),
                write_artifacts=True,
                include_trials=True,
                runlog_path=runlog_path,
            )
        except Exception as exc:
            test1_summary = {"status": "BLOCKED", "reason": f"Test1 exception: {exc}"}
            write_json(out_dir / "summary_phase_peak_null.json", test1_summary)
            save_blocked_figure(out_dir / "fig_phase_null.png", "Phase Peak Null", test1_summary["reason"])
            log_line(runlog_path, f"Test1 exception captured: {exc}")

    # Step 5: dataset lineage
    try:
        test5_summary = run_test5_dataset_lineage(
            rar_points_path=manifest.get("rar_points_path"),
            galaxy_results_path=manifest.get("galaxy_results_path"),
            tng_manifest=manifest,
            out_dir=out_dir,
        )
    except Exception as exc:
        test5_summary = {"status": "BLOCKED", "reason": f"Test5 exception: {exc}"}
        write_json(out_dir / "summary_dataset_lineage.json", test5_summary)
        (out_dir / "dataset_lineage.md").write_text(f"# Dataset Lineage Audit\\n\\nBLOCKED: {exc}\\n")
        log_line(runlog_path, f"Test5 exception captured: {exc}")

    # Step 6: TNG gate
    try:
        tng_validation, tng_df = run_tng_validation(manifest.get("chosen_tng_path"), out_dir=out_dir)
    except Exception as exc:
        tng_validation = {"status": "BLOCKED", "reason": f"TNG validation exception: {exc}"}
        tng_df = None
        write_json(out_dir / "tng_validation.json", tng_validation)
        log_line(runlog_path, f"TNG validation exception captured: {exc}")

    # Step 7: Test 2
    try:
        test2_summary = run_test2_mass_matched_phase(
            sparc_df=sparc_df,
            tng_df=tng_df,
            tng_meta=tng_validation,
            out_dir=out_dir,
            seed=seeds["test2"],
            runlog_path=runlog_path,
        )
    except Exception as exc:
        test2_summary = {"status": "BLOCKED", "reason": f"Test2 exception: {exc}"}
        write_json(out_dir / "summary_mass_matched_phase.json", test2_summary)
        save_blocked_figure(out_dir / "fig_mass_matched_phase.png", "Mass-Matched Phase", test2_summary["reason"])
        log_line(runlog_path, f"Test2 exception captured: {exc}")

    # Step 8: Test 3
    try:
        test3_summary = run_test3_xi_organizing(
            sparc_df=sparc_df,
            tng_df=tng_df,
            out_dir=out_dir,
            seed=seeds["test3"],
            runlog_path=runlog_path,
        )
    except Exception as exc:
        test3_summary = {"status": "BLOCKED", "reason": f"Test3 exception: {exc}"}
        write_json(out_dir / "summary_xi_organizing.json", test3_summary)
        save_blocked_figure(out_dir / "fig_xi_organizing.png", "Xi Organizing", test3_summary["reason"])
        log_line(runlog_path, f"Test3 exception captured: {exc}")

    # Step 9: Test 4
    galaxy_df = None
    if manifest.get("galaxy_results_path") is not None:
        try:
            galaxy_df = pd.read_csv(manifest["galaxy_results_path"])
        except Exception:
            galaxy_df = None

    try:
        test4_summary = run_test4_alpha_convergence(
            gal_df=galaxy_df,
            sparc_df=sparc_df,
            tng_df=tng_df,
            out_dir=out_dir,
            seed=seeds["test4"],
            runlog_path=runlog_path,
        )
    except Exception as exc:
        test4_summary = {"status": "BLOCKED", "reason": f"Test4 exception: {exc}"}
        write_json(out_dir / "summary_alpha_star_convergence.json", test4_summary)
        save_blocked_figure(out_dir / "fig_alpha_star.png", "alpha* Convergence", test4_summary["reason"])
        log_line(runlog_path, f"Test4 exception captured: {exc}")

    # Step 10: consistency dashboard (includes deterministic rerun of Test1)
    try:
        consistency_summary = run_consistency_checks(
            out_dir=out_dir,
            sparc_df=sparc_df,
            test1_summary=test1_summary,
            test2_summary=test2_summary,
            seed_real=seeds["consistency_test1_real"],
            seed_null=seeds["consistency_test1_null"],
            n_shuffles=int(args.n_shuffles),
            runlog_path=runlog_path,
        )
    except Exception as exc:
        consistency_summary = {"status": "BLOCKED", "reason": f"Consistency exception: {exc}"}
        write_json(out_dir / "summary_consistency_checks.json", consistency_summary)
        log_line(runlog_path, f"Consistency exception captured: {exc}")

    # Step 11: outputs index
    outputs_index = write_outputs_index(out_dir)
    _ = outputs_index

    # Final console report (no narrative)
    status_map = {
        "test1_phase_peak_null": test1_summary.get("status", "BLOCKED"),
        "test2_mass_matched_phase": test2_summary.get("status", "BLOCKED"),
        "test3_xi_organizing": test3_summary.get("status", "BLOCKED"),
        "test4_alpha_star_convergence": test4_summary.get("status", "BLOCKED"),
        "test5_dataset_lineage": test5_summary.get("status", "BLOCKED"),
        "consistency_checks": consistency_summary.get("status", "BLOCKED"),
        "tng_validation": tng_validation.get("status", "BLOCKED"),
    }

    key = {
        "mu_peak": test1_summary.get("real", {}).get("mu_peak") if isinstance(test1_summary, dict) else None,
        "delta_aic": test1_summary.get("real", {}).get("daic") if isinstance(test1_summary, dict) else None,
        "p_delta_A": test1_summary.get("shuffle_A_within_galaxy", {}).get("p_delta") if isinstance(test1_summary, dict) else None,
        "p_daic_A": test1_summary.get("shuffle_A_within_galaxy", {}).get("p_daic") if isinstance(test1_summary, dict) else None,
        "p_delta_B": test1_summary.get("shuffle_B_galaxy_label", {}).get("p_delta") if isinstance(test1_summary, dict) else None,
        "p_daic_B": test1_summary.get("shuffle_B_galaxy_label", {}).get("p_daic") if isinstance(test1_summary, dict) else None,
        "C_sparc": test3_summary.get("sparc", {}).get("concentration_C") if isinstance(test3_summary, dict) else None,
        "C_p_sparc": test3_summary.get("sparc", {}).get("concentration_pvalue") if isinstance(test3_summary, dict) else None,
        "alpha_table_rows": len(test4_summary.get("table", [])) if isinstance(test4_summary, dict) else None,
    }

    print(f"output_folder={out_dir}")
    for k, v in status_map.items():
        print(f"{k}={v}")
    for k, v in key.items():
        print(f"{k}={v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
