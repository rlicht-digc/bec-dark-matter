#!/usr/bin/env python3
"""
Paper-3 bridge pack:
High-density target selection + residual-shift statistics + robustness suite.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
try:
    from galaxy_naming import canonicalize_galaxy_name
except Exception:
    def canonicalize_galaxy_name(name: str) -> str:
        s = str(name).strip().upper()
        s = " ".join(s.split())
        if s.startswith("WALLABY"):
            return s
        return s.replace(" ", "")


G_NEWTON = 6.67430e-11
KPC_TO_M = 3.085677581491367e19
DEFAULT_G_DAGGER = 1.286e-10
DEFAULT_PRIMARY_POINTS = "analysis/results/rar_points_unified.csv"
DEFAULT_SUMMARY_FILE = "analysis/results/summary_unified.json"
EPS = np.finfo(float).tiny
RNG_SEED = 314159
DEFAULT_N_PERM = 10_000
DEFAULT_N_BOOT = 5_000
INNER_MIN_POINTS_FOR_MEDIAN = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _lower_col_map(columns: Sequence[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}


def _pick_column(lower_map: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    return None


def infer_column_mapping(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    lower_map = _lower_col_map(columns)
    return {
        "galaxy": _pick_column(
            lower_map,
            ("galaxy", "gal_name", "name", "galaxy_id", "objname", "id"),
        ),
        "galaxy_key": _pick_column(
            lower_map,
            ("galaxy_key", "galaxy_norm", "gal_key", "name_key"),
        ),
        "dataset": _pick_column(lower_map, ("source", "dataset", "survey", "sample")),
        "radius": _pick_column(
            lower_map,
            ("r_kpc", "r(kpc)", "r", "radius_kpc", "radius", "rkpc", "radii_kpc"),
        ),
        "gbar_linear": _pick_column(
            lower_map,
            ("g_bar", "gbar", "gbar_ms2", "gbary", "g_bary", "g_baryon"),
        ),
        "gbar_log": _pick_column(
            lower_map,
            ("log_gbar", "log10_gbar", "lgbar", "log_g_bar"),
        ),
        "gobs_linear": _pick_column(
            lower_map,
            ("g_obs", "gobs", "gobs_ms2", "gtot", "g_tot", "g_total"),
        ),
        "gobs_log": _pick_column(
            lower_map,
            ("log_gobs", "log10_gobs", "lgobs", "log_g_obs"),
        ),
        "v_obs": _pick_column(
            lower_map,
            ("vobs", "v_obs", "vobs_kms", "v_obs_kms", "vobs_km_s", "vrot", "v_rot", "vtot"),
        ),
        "v_bar": _pick_column(
            lower_map,
            ("vbar", "v_bar", "vbar_kms", "v_bar_kms", "vbar_km_s", "vbary", "v_bary"),
        ),
    }


def _is_usable_mapping(mapping: Dict[str, Optional[str]]) -> bool:
    has_key_fields = mapping["galaxy"] is not None and mapping["radius"] is not None
    has_acc = (
        (mapping["gbar_linear"] is not None or mapping["gbar_log"] is not None)
        and (mapping["gobs_linear"] is not None or mapping["gobs_log"] is not None)
    ) or (mapping["v_obs"] is not None and mapping["v_bar"] is not None)
    return bool(has_key_fields and has_acc)


def choose_rar_points_file(repo_root: Path, user_file: Optional[str]) -> Tuple[Path, Dict[str, Optional[str]]]:
    if user_file:
        path = Path(user_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--rar_points_file does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"--rar_points_file must be CSV: {path}")
        mapping = infer_column_mapping(pd.read_csv(path, nrows=1).columns)
        if not _is_usable_mapping(mapping):
            raise ValueError(f"Input file does not expose required columns: {path}")
        return path, mapping

    preferred = (repo_root / DEFAULT_PRIMARY_POINTS).resolve()
    if preferred.exists():
        mapping = infer_column_mapping(pd.read_csv(preferred, nrows=1).columns)
        if _is_usable_mapping(mapping):
            return preferred, mapping

    raise RuntimeError(
        f"Could not find usable points file. Expected: {preferred}"
    )


def load_sigma_int_from_summary(repo_root: Path) -> Dict[str, object]:
    summary_path = (repo_root / DEFAULT_SUMMARY_FILE).resolve()
    info: Dict[str, object] = {
        "used": False,
        "sigma_int_z": np.nan,
        "source": "summary_unified.json",
        "summary_path": str(summary_path),
        "reason": "not_requested",
    }
    if not summary_path.exists():
        info["reason"] = "summary_file_missing"
        return info

    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        info["reason"] = f"summary_read_failed:{exc}"
        return info

    node = (
        obj.get("refined_bec_tests", {})
        .get("bec_transition_function", {})
    )
    shared = node.get("sigma_int_shared_z")
    bec_only = node.get("bec_fit", {}).get("sigma_int_bec_z")

    if shared is not None and np.isfinite(shared):
        info["used"] = True
        info["sigma_int_z"] = float(shared)
        info["reason"] = "loaded_shared_sigma"
    elif bec_only is not None and np.isfinite(bec_only):
        info["used"] = True
        info["sigma_int_z"] = float(bec_only)
        info["reason"] = "loaded_bec_sigma"
    else:
        info["reason"] = "sigma_int_not_present_in_summary"
    return info


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _radius_to_kpc(radius_raw: np.ndarray, radius_column_name: str) -> np.ndarray:
    name = radius_column_name.lower()
    if "kpc" in name:
        return radius_raw
    if "mpc" in name:
        return radius_raw * 1_000.0
    if "pc" in name and "kpc" not in name:
        return radius_raw / 1_000.0
    if "(m)" in name or name.endswith("_m") or "meter" in name or "metre" in name:
        return radius_raw / KPC_TO_M
    return radius_raw


def _velocity_to_m_s(velocity: np.ndarray, col_name: str) -> np.ndarray:
    name = col_name.lower()
    if "km" in name:
        return velocity * 1_000.0
    if "m/s" in name or "ms-1" in name:
        return velocity
    med = np.nanmedian(np.abs(velocity))
    if not np.isfinite(med):
        return velocity
    if med < 1.0e4:
        return velocity * 1_000.0
    return velocity


def standardize_points(
    raw_df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    assume_r_kpc: bool,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    if mapping["galaxy"] is None or mapping["radius"] is None:
        raise ValueError("Missing required columns: galaxy and/or radius.")

    galaxy = raw_df[mapping["galaxy"]].astype(str).str.strip()
    if mapping.get("galaxy_key") is not None:
        galaxy_key = raw_df[mapping["galaxy_key"]].astype(str).map(canonicalize_galaxy_name)
    else:
        galaxy_key = galaxy.map(canonicalize_galaxy_name)
    dataset = (
        raw_df[mapping["dataset"]].astype(str).str.strip()
        if mapping["dataset"] is not None
        else pd.Series("unknown", index=raw_df.index, dtype=object)
    )
    r_raw = _to_numeric(raw_df[mapping["radius"]])
    r_kpc = _radius_to_kpc(r_raw, mapping["radius"])

    good_r = r_kpc[np.isfinite(r_kpc) & (r_kpc > 0)]
    r_med = float(np.nanmedian(good_r)) if good_r.size else np.nan
    if np.isfinite(r_med) and r_med > 1000.0 and not assume_r_kpc:
        raise ValueError(
            "Radius unit sanity failed: median radius > 1000 kpc. "
            "Use --assume_r_kpc to override."
        )

    g_bar: Optional[np.ndarray] = None
    g_obs: Optional[np.ndarray] = None
    notes: Dict[str, str] = {"radius": f"{mapping['radius']} -> r_kpc"}

    if mapping["gbar_linear"] is not None:
        g_bar = _to_numeric(raw_df[mapping["gbar_linear"]])
        notes["g_bar"] = mapping["gbar_linear"]
    elif mapping["gbar_log"] is not None:
        g_bar = np.power(10.0, _to_numeric(raw_df[mapping["gbar_log"]]))
        notes["g_bar"] = f"10**{mapping['gbar_log']}"

    if mapping["gobs_linear"] is not None:
        g_obs = _to_numeric(raw_df[mapping["gobs_linear"]])
        notes["g_obs"] = mapping["gobs_linear"]
    elif mapping["gobs_log"] is not None:
        g_obs = np.power(10.0, _to_numeric(raw_df[mapping["gobs_log"]]))
        notes["g_obs"] = f"10**{mapping['gobs_log']}"

    if g_bar is None or g_obs is None:
        if mapping["v_obs"] is None or mapping["v_bar"] is None:
            raise ValueError("Need g_bar/g_obs or both velocity columns to derive them.")
        r_m = r_kpc * KPC_TO_M
        v_obs = _velocity_to_m_s(_to_numeric(raw_df[mapping["v_obs"]]), mapping["v_obs"])
        v_bar = _velocity_to_m_s(_to_numeric(raw_df[mapping["v_bar"]]), mapping["v_bar"])
        g_obs = np.square(v_obs) / r_m
        g_bar = np.square(v_bar) / r_m
        notes["g_obs"] = f"{mapping['v_obs']}^2 / r_m"
        notes["g_bar"] = f"{mapping['v_bar']}^2 / r_m"

    points = pd.DataFrame(
        {
            "galaxy": galaxy,
            "galaxy_key": galaxy_key,
            "dataset": dataset,
            "r_kpc": r_kpc,
            "g_bar": g_bar,
            "g_obs": g_obs,
        }
    )
    all_galaxies = sorted(galaxy_key.dropna().astype(str).str.strip().unique().tolist())
    return points, notes, all_galaxies


def apply_quality_filters(points: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    counts = {"input_rows": int(len(points))}
    out = points.replace([np.inf, -np.inf], np.nan).copy()
    out = out.dropna(subset=["galaxy", "galaxy_key", "r_kpc", "g_bar", "g_obs"])
    counts["after_nonfinite_drop"] = int(len(out))
    out = out[(out["r_kpc"] > 0.0) & (out["g_bar"] > 0.0) & (out["g_obs"] > 0.0)].copy()
    counts["after_physical_cuts"] = int(len(out))
    return out, counts


def source_histogram(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["source", "n_points", "n_galaxies"])
    hist = (
        points.groupby("dataset", dropna=False)
        .agg(
            n_points=("dataset", "size"),
            n_galaxies=("galaxy_key", "nunique"),
        )
        .reset_index()
        .rename(columns={"dataset": "source"})
        .sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )
    hist["source"] = hist["source"].fillna("unknown").astype(str)
    return hist


def compute_rar_prediction(g_bar: np.ndarray, g_dagger: float) -> np.ndarray:
    x = np.sqrt(np.clip(g_bar / g_dagger, a_min=0.0, a_max=None))
    denom = -np.expm1(-x)
    return np.divide(
        g_bar,
        denom,
        out=np.full_like(g_bar, np.nan, dtype=float),
        where=denom > EPS,
    )


def permutation_test_median_difference(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    if len(vals_a) < 1 or len(vals_b) < 1:
        return np.nan, np.nan
    observed = float(np.median(vals_a) - np.median(vals_b))
    combined = np.concatenate([vals_a, vals_b])
    n_a = len(vals_a)
    extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(combined)
        diff = float(np.median(shuffled[:n_a]) - np.median(shuffled[n_a:]))
        if abs(diff) >= abs(observed):
            extreme += 1
    p_val = (extreme + 1) / (n_perm + 1)
    return observed, float(p_val)


def bootstrap_theilsen_slope(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    slopes: List[float] = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if np.nanstd(xb) <= 0:
            continue
        slope, _, _, _ = stats.theilslopes(yb, xb, alpha=0.95)
        if np.isfinite(slope):
            slopes.append(float(slope))
    if not slopes:
        return np.nan, np.nan, np.nan
    arr = np.asarray(slopes)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p_boot = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    return float(lo), float(hi), float(p_boot)


def summarize_galaxies(
    points: pd.DataFrame,
    all_galaxies: List[str],
    g_dagger: float,
    r_inner_max_kpc: float,
    min_points: int,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    log_gdag = math.log10(g_dagger)
    grouped = {name: frame for name, frame in points.groupby("galaxy_key", sort=False)}

    for galaxy_key in all_galaxies:
        gdf = grouped.get(galaxy_key)
        if gdf is None or gdf.empty:
            records.append(
                {
                    "galaxy": galaxy_key,
                    "galaxy_key": galaxy_key,
                    "N_points": 0,
                    "rho_score": np.nan,
                    "rho_dyn_max": np.nan,
                    "rho_dyn_p50_inner": np.nan,
                    "rho_dyn_at_gdag": np.nan,
                    "r_at_gdag_kpc": np.nan,
                    "median_log_resid": np.nan,
                    "rms_log_resid": np.nan,
                    "p90_abs_log_resid": np.nan,
                    "dataset": "unknown",
                    "flag_low_points": True,
                    "notes": "no_valid_points_after_filter; N_points<min_points",
                }
            )
            continue

        n_points = int(len(gdf))
        galaxy_display = (
            gdf["galaxy"].mode(dropna=True).iloc[0]
            if "galaxy" in gdf.columns and not gdf["galaxy"].dropna().empty
            else galaxy_key
        )
        dataset = gdf["dataset"].mode(dropna=True).iloc[0] if not gdf["dataset"].dropna().empty else "unknown"
        inner = gdf[gdf["r_kpc"] <= r_inner_max_kpc]
        rho_dyn_max = float(inner["rho_dyn"].max()) if not inner.empty else np.nan
        rho_dyn_p50_inner = float(inner["rho_dyn"].median()) if not inner.empty else np.nan
        idx = (np.abs(np.log10(gdf["g_bar"]) - log_gdag)).idxmin()
        row_gdag = gdf.loc[idx]
        rho_dyn_at_gdag = float(row_gdag["rho_dyn"])
        r_at_gdag_kpc = float(row_gdag["r_kpc"])
        median_log_resid = float(np.median(gdf["log_resid"]))
        rms_log_resid = float(np.sqrt(np.mean(np.square(gdf["log_resid"]))))
        p90_abs_log_resid = float(np.percentile(np.abs(gdf["log_resid"]), 90))

        notes: List[str] = []
        flag_low = n_points < min_points
        if flag_low:
            notes.append("N_points<min_points")
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
                "galaxy": galaxy_display,
                "galaxy_key": galaxy_key,
                "N_points": n_points,
                "rho_score": rho_score,
                "rho_dyn_max": rho_dyn_max,
                "rho_dyn_p50_inner": rho_dyn_p50_inner,
                "rho_dyn_at_gdag": rho_dyn_at_gdag,
                "r_at_gdag_kpc": r_at_gdag_kpc,
                "median_log_resid": median_log_resid,
                "rms_log_resid": rms_log_resid,
                "p90_abs_log_resid": p90_abs_log_resid,
                "dataset": dataset,
                "flag_low_points": bool(flag_low),
                "notes": "; ".join(notes),
            }
        )

    return pd.DataFrame.from_records(records)


def run_single_density_window(
    points: pd.DataFrame,
    all_galaxies: List[str],
    g_dagger: float,
    r_inner_max_kpc: float,
    min_points: int,
    top_n: int,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    print(f"[P3] Computing rho_score with r_inner_max_kpc={r_inner_max_kpc:.1f}", flush=True)
    summary = summarize_galaxies(points, all_galaxies, g_dagger, r_inner_max_kpc, min_points)
    eligible = summary[
        (~summary["flag_low_points"])
        & np.isfinite(summary["rho_score"])
        & np.isfinite(summary["median_log_resid"])
    ].copy()
    eligible = eligible.sort_values("rho_score", ascending=False).reset_index(drop=True)
    top = eligible.head(top_n).copy()
    bottom = eligible.loc[~eligible["galaxy_key"].isin(top["galaxy_key"])].sort_values("rho_score").head(top_n).copy()

    first_five = ", ".join(top["galaxy"].head(5).tolist()) if not top.empty else "[none]"
    print(f"[P3] Top20 generated (list first 5): {first_five}", flush=True)

    print("[P3] Fitting trend (Theil–Sen)...", flush=True)
    x = np.log10(eligible["rho_score"].to_numpy(dtype=float))
    y = eligible["median_log_resid"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) >= 2:
        slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
        ci_lo, ci_hi, boot_p = bootstrap_theilsen_slope(x, y, n_boot, rng)
    else:
        slope, intercept, ci_lo, ci_hi, boot_p = (np.nan, np.nan, np.nan, np.nan, np.nan)

    top_names = set(top["galaxy_key"].tolist())
    bottom_names = set(bottom["galaxy_key"].tolist())
    top_points_resid = points[points["galaxy_key"].isin(top_names)]["log_resid"].to_numpy(dtype=float)
    bottom_points_resid = points[points["galaxy_key"].isin(bottom_names)]["log_resid"].to_numpy(dtype=float)
    top_gal_median = top["median_log_resid"].to_numpy(dtype=float)
    bottom_gal_median = bottom["median_log_resid"].to_numpy(dtype=float)

    perm_label = "10k" if n_perm == 10_000 else str(n_perm)
    print(f"[P3] Permutation test ({perm_label})...", flush=True)
    pooled_shift, pooled_perm_p = permutation_test_median_difference(
        top_points_resid, bottom_points_resid, n_perm, rng
    )
    gal_shift, gal_perm_p = permutation_test_median_difference(
        top_gal_median, bottom_gal_median, n_perm, rng
    )

    done_p = pooled_perm_p if np.isfinite(pooled_perm_p) else gal_perm_p
    print(
        f"[P3] Done run for r_inner_max_kpc={r_inner_max_kpc:.1f} slope={slope:.6f}, p={done_p:.6g}",
        flush=True,
    )

    return {
        "r_inner_max_kpc": r_inner_max_kpc,
        "summary": summary,
        "eligible": eligible,
        "top": top,
        "bottom": bottom,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci_lo": float(ci_lo),
        "slope_ci_hi": float(ci_hi),
        "bootstrap_p": float(boot_p),
        "perm_p_pooled": float(pooled_perm_p),
        "perm_p_gal": float(gal_perm_p),
        "shift_pooled": float(pooled_shift),
        "shift_gal": float(gal_shift),
        "top_points_resid": top_points_resid,
        "bottom_points_resid": bottom_points_resid,
        "top_gal_median": top_gal_median,
        "bottom_gal_median": bottom_gal_median,
    }


def save_scatter_plot(out_path: Path, eligible: pd.DataFrame, slope: float, intercept: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.log10(eligible["rho_score"].to_numpy(dtype=float))
    y = eligible["median_log_resid"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) > 0:
        ax.scatter(x, y, s=26, alpha=0.8, color="#1f77b4")
    if np.isfinite(slope) and np.isfinite(intercept) and len(x) > 1:
        x_grid = np.linspace(np.min(x), np.max(x), 300)
        y_grid = intercept + slope * x_grid
        ax.plot(x_grid, y_grid, color="crimson", lw=2.0, label="Theil-Sen")
        ax.legend(loc="best")
    ax.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax.set_xlabel("log10(rho_score) [kg m^-3]")
    ax.set_ylabel("Median log residual (dex)")
    ax.set_title("rho_score vs Median Residual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_hist_plot(
    out_path: Path,
    top_points_resid: np.ndarray,
    bottom_points_resid: np.ndarray,
    top_gal_median: np.ndarray,
    bottom_gal_median: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    pooled_all = np.concatenate([top_points_resid, bottom_points_resid]) if len(top_points_resid) and len(bottom_points_resid) else np.array([])
    if pooled_all.size > 0:
        bins = np.linspace(np.nanmin(pooled_all), np.nanmax(pooled_all), 45)
        axes[0].hist(top_points_resid, bins=bins, alpha=0.55, density=True, label="Top20 pooled points")
        axes[0].hist(bottom_points_resid, bins=bins, alpha=0.55, density=True, label="Bottom20 pooled points")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Insufficient pooled data", ha="center", transform=axes[0].transAxes)
    axes[0].set_title("Pooled Point Residuals")
    axes[0].set_xlabel("log residual (dex)")
    axes[0].set_ylabel("Density")

    gal_all = np.concatenate([top_gal_median, bottom_gal_median]) if len(top_gal_median) and len(bottom_gal_median) else np.array([])
    if gal_all.size > 0:
        bins = np.linspace(np.nanmin(gal_all), np.nanmax(gal_all), 25)
        axes[1].hist(top_gal_median, bins=bins, alpha=0.55, density=True, label="Top20 galaxy medians")
        axes[1].hist(bottom_gal_median, bins=bins, alpha=0.55, density=True, label="Bottom20 galaxy medians")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Insufficient galaxy-level data", ha="center", transform=axes[1].transAxes)
    axes[1].set_title("Per-Galaxy Median Residuals")
    axes[1].set_xlabel("median log residual (dex)")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_residual_vs_gbar_plot(out_path: Path, points_top: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if points_top.empty:
        ax.text(0.5, 0.5, "No top-density points", ha="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    x = np.log10(points_top["g_bar"].to_numpy(dtype=float))
    y = points_top["log_resid"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    ax.scatter(x, y, s=10, alpha=0.35, color="#2a9d8f", label="Points")
    if len(x) >= 12:
        edges = np.linspace(np.min(x), np.max(x), 14)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binned = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x < hi)
            if np.sum(mask) >= 3:
                binned.append(np.median(y[mask]))
            else:
                binned.append(np.nan)
        binned_arr = np.asarray(binned)
        good = np.isfinite(binned_arr)
        if np.any(good):
            ax.plot(centers[good], binned_arr[good], color="crimson", lw=2.0, marker="o", ms=3, label="Binned median")
    ax.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax.set_xlabel("log10(g_bar) [m s^-2]")
    ax.set_ylabel("log residual (dex)")
    ax.set_title("Residual vs log g_bar (Top20 Density Systems)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_bridge_report(
    out_path: Path,
    input_file: Path,
    rar_points_sha256: str,
    require_csv_sha: Optional[str],
    git_head: Optional[str],
    g_dagger: float,
    baseline_r_inner: float,
    min_points: int,
    top_n: int,
    include_ss20: bool,
    ss20_excluded_points: int,
    top_targets: pd.DataFrame,
    bottom_targets: pd.DataFrame,
    baseline_eligible: pd.DataFrame,
    baseline_result: Dict[str, object],
    robustness_df: pd.DataFrame,
    source_hist: pd.DataFrame,
    point_counts: Dict[str, float],
    mapping: Dict[str, Optional[str]],
    conversion_notes: Dict[str, str],
    filter_counts: Dict[str, int],
    sigma_cal_info: Dict[str, object],
    weighted_stats: Optional[Dict[str, float]],
) -> None:
    slope = baseline_result["slope"]
    slope_ci_lo = baseline_result["slope_ci_lo"]
    slope_ci_hi = baseline_result["slope_ci_hi"]
    bootstrap_p = baseline_result["bootstrap_p"]
    perm_p = baseline_result["perm_p_pooled"]
    shift = baseline_result["shift_pooled"]

    lines: List[str] = []
    lines.append("paper3_density_bridge_report")
    lines.append("")
    lines.append(f"file used: {input_file}")
    lines.append(f"actual_csv_sha: {rar_points_sha256}")
    lines.append(f"require_csv_sha: {require_csv_sha if require_csv_sha else 'None'}")
    lines.append(f"git_head: {git_head if git_head else 'None'}")
    lines.append(f"g_dagger: {g_dagger:.6e}")
    lines.append(f"r_inner_max_kpc (baseline): {baseline_r_inner:.1f}")
    lines.append(f"min_points: {min_points}")
    lines.append(f"include_ss20: {include_ss20}")
    lines.append(f"ss20_excluded_points: {ss20_excluded_points}")
    use_txt = "yes" if bool(sigma_cal_info.get("used", False)) else "no"
    sig_val = sigma_cal_info.get("sigma_int_z")
    sig_txt = f"{float(sig_val):.6g}" if sig_val is not None and np.isfinite(sig_val) else "None"
    lines.append(
        f"σ_int calibration used: {use_txt}; sigma_int value: {sig_txt}; source: summary_unified.json"
    )
    lines.append(f"σ_int calibration detail: {sigma_cal_info.get('reason')}")
    lines.append("")
    lines.append("source histogram (analysis set)")
    for _, row in source_hist.iterrows():
        lines.append(
            f"- {row['source']}: n_points={int(row['n_points'])}, n_galaxies={int(row['n_galaxies'])}"
        )
    lines.append("")
    lines.append("analysis header diagnostics")
    top_n_pts = int(len(baseline_result.get("top_points_resid", [])))
    bottom_n_pts = int(len(baseline_result.get("bottom_points_resid", [])))
    eligible_n_gal = int(len(baseline_eligible))
    eligible_n_pts = int(np.nansum(baseline_eligible["N_points"].to_numpy(dtype=float))) if len(baseline_eligible) else 0
    eligible_n_arr = baseline_eligible["N_points"].to_numpy(dtype=float) if len(baseline_eligible) else np.array([])
    eligible_med = float(np.nanmedian(eligible_n_arr)) if len(eligible_n_arr) else np.nan
    eligible_max = float(np.nanmax(eligible_n_arr)) if len(eligible_n_arr) else np.nan
    top_med = float(point_counts["top20_median"]) if np.isfinite(point_counts["top20_median"]) else np.nan
    top_max = float(np.nanmax(top_targets["N_points"].to_numpy(dtype=float))) if len(top_targets) else np.nan
    bot_med = float(point_counts["bottom20_median"]) if np.isfinite(point_counts["bottom20_median"]) else np.nan
    bot_max = float(np.nanmax(bottom_targets["N_points"].to_numpy(dtype=float))) if len(bottom_targets) else np.nan
    lines.append(
        f"- N_gal eligible={eligible_n_gal}, top={len(top_targets)}, bottom={len(bottom_targets)}; "
        f"N_pts pooled top={top_n_pts}, bottom={bottom_n_pts}, eligible≈{eligible_n_pts}"
    )
    lines.append(
        f"- points-per-galaxy (eligible median/max): {eligible_med:.2f}/{eligible_max:.0f}; "
        f"(top median/max): {top_med:.2f}/{top_max:.0f}; "
        f"(bottom median/max): {bot_med:.2f}/{bot_max:.0f}"
    )
    lines.append(
        f"- eligibility for ranking: N_points >= {min_points}, finite rho_score, finite median_log_resid"
    )
    lines.append(
        f"- top/bottom construction: global per-galaxy ranking by rho_score over eligible galaxies (grouped by galaxy_key); "
        f"top={top_n} highest rho_score, bottom={top_n} lowest rho_score after excluding top set"
    )
    lines.append(
        "- pooled vs per-gal shifts: pooled uses all point-level residuals from top/bottom groups; "
        "per-gal uses one median residual per galaxy"
    )
    lines.append("- quantile thresholds: none (rank-based cut only)")
    lines.append(
        "- histogram bin edges: each panel uses np.linspace(min(sample), max(sample), 31) "
        "(30 equal-width bins, computed separately for pooled-point and per-galaxy-median samples)"
    )
    lines.append(
        "- residual space for trend/permutation/shift: dex "
        "(log_resid = log10(g_obs) - log10(g_pred))"
    )
    if weighted_stats is not None:
        lines.append(
            "- residual space for calibrated-weight section: z "
            "(z_resid = (log_resid - mean_all_points)/std_all_points)"
        )
    else:
        lines.append("- residual space for calibrated-weight section: not used in this run")
    gbar_note = conversion_notes.get("g_bar", "unknown")
    if gbar_note.startswith("10**"):
        lines.append(f"- g_bar derivation: g_bar = 10**({gbar_note.replace('10**', '')})")
    else:
        lines.append(f"- g_bar derivation: {gbar_note}")
    lines.append("")
    lines.append("column mapping")
    for key, val in mapping.items():
        lines.append(f"- {key}: {val}")
    lines.append("conversion notes")
    for key, val in conversion_notes.items():
        lines.append(f"- {key}: {val}")
    lines.append("filtering summary")
    for key, val in filter_counts.items():
        lines.append(f"- {key}: {val}")
    lines.append("")
    lines.append("top 20 target list with rho_score")
    for _, row in top_targets.iterrows():
        lines.append(f"- {row['galaxy']}: rho_score={row['rho_score']:.6e}, N_points={int(row['N_points'])}")
    lines.append("")
    lines.append("trend statistics")
    lines.append(f"- Theil-Sen slope: {slope:.6f}")
    lines.append(f"- bootstrap CI: [{slope_ci_lo:.6f}, {slope_ci_hi:.6f}]")
    lines.append(f"- bootstrap p-value: {bootstrap_p:.6g}")
    lines.append("")
    lines.append("top-vs-bottom residual shift")
    lines.append(f"- pooled-point median shift (top-bottom): {shift:.6f} dex")
    lines.append(f"- permutation p-value (10k): {perm_p:.6g}")
    lines.append(
        f"- per-galaxy median shift (top-bottom): {baseline_result['shift_gal']:.6f} dex; "
        f"permutation p-value: {baseline_result['perm_p_gal']:.6g}"
    )
    lines.append("")
    lines.append("red-team robustness summary (r_inner_max_kpc = 1, 2, 3)")
    for _, row in robustness_df.iterrows():
        lines.append(
            f"- r_inner={row['r_inner_max_kpc']:.1f}: overlap_vs_2kpc={row['top20_overlap_vs_2kpc']:.3f}, "
            f"slope={row['slope']:.6f}, bootstrap_p={row['bootstrap_p']:.6g}, perm_p={row['perm_p']:.6g}, N_gal={int(row['N_gal'])}"
        )
    baseline_sign = np.sign(float(robustness_df.loc[robustness_df["r_inner_max_kpc"] == 2.0, "slope"].iloc[0]))
    same_sign = bool(np.all(np.sign(robustness_df["slope"].to_numpy(dtype=float)) == baseline_sign))
    perm_stable = bool(np.all(robustness_df["perm_p"].to_numpy(dtype=float) < 0.05))
    lines.append(f"- slope sign consistent across windows: {same_sign}")
    lines.append(f"- permutation p-value < 0.05 across windows: {perm_stable}")
    lines.append("")
    if same_sign and perm_stable:
        lines.append(
            'interpretation: "high-density systems show a systematic residual shift under constant-coupling RAR-BEC"'
        )
        lines.append(
            'interpretation: "this provides an empirical handle on the critical density ρ_c"'
        )
    else:
        lines.append(
            'interpretation: "residual shifts are suggestive but not robustly stable across density windows under current cuts"'
        )
        lines.append(
            'interpretation: "this run provides candidate targets; constraints on critical density ρ_c remain provisional"'
        )
    lines.append("")
    lines.append("point-count confound check")
    lines.append(
        f"- top20 N_points median: {point_counts['top20_median']:.2f}; "
        f"bottom20 N_points median: {point_counts['bottom20_median']:.2f}"
    )
    lines.append("")
    if weighted_stats is not None:
        lines.append("calibrated-weight residual stats (z-space)")
        lines.append(
            f"- sigma_tot_z = {weighted_stats['sigma_tot_z']:.6f} "
            f"(from sigma_meas_z=1 and sigma_int_z={weighted_stats['sigma_int_z']:.6f})"
        )
        lines.append(
            f"- top20 weighted mean/std z_resid: "
            f"{weighted_stats['top_weighted_mean_z']:.6f} / {weighted_stats['top_weighted_std_z']:.6f} "
            f"(N={int(weighted_stats['top_n'])})"
        )
        lines.append(
            f"- bottom20 weighted mean/std z_resid: "
            f"{weighted_stats['bottom_weighted_mean_z']:.6f} / {weighted_stats['bottom_weighted_std_z']:.6f} "
            f"(N={int(weighted_stats['bottom_n'])})"
        )
        lines.append("")

    lines.append("limitations / caveats")
    lines.append("- rho_dyn is a dynamical proxy and not a full deprojection of the mass profile.")
    lines.append("- residual distributions can be influenced by heterogeneous radial sampling.")
    lines.append("- no morphology assumptions are used; systems are ranked only by measured proxy.")
    lines.append("- this analysis does not fit (rho_c, alpha); it provides empirical targeting constraints.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_writeup_stub(
    out_path: Path,
    baseline_r_inner: float,
    top_targets: pd.DataFrame,
    slope: float,
    slope_ci_lo: float,
    slope_ci_hi: float,
    bootstrap_p: float,
    perm_p: float,
    pooled_shift: float,
) -> None:
    lines: List[str] = []
    lines.append("# Paper 3 Bridge: Density-Dependent Coupling Observational Handle")
    lines.append("")
    lines.append("## Motivation")
    lines.append(
        "Paper 1 established an empirical RAR structure under a constant-coupling baseline. "
        "Paper 3 addresses compact-object regimes where density-dependent coupling is expected to matter. "
        "This bridge analysis defines a data-driven density proxy from RAR points and tests whether residual structure "
        "against the constant-coupling RAR-BEC prediction shifts first in high-density systems."
    )
    lines.append("")
    lines.append("## Method")
    lines.append(
        f"We define a per-point dynamical proxy rho_dyn(r)=3*g_obs/(4*pi*G*r), with r in meters, "
        f"then summarize each galaxy with rho_score = median(rho_dyn) for r_kpc <= {baseline_r_inner:.1f} "
        "(fallback to inner max where needed). Residuals are computed as "
        "log_resid = log10(g_obs) - log10(g_pred), with "
        "g_pred = g_bar/(1-exp(-sqrt(g_bar/g_dagger)))."
    )
    lines.append("")
    lines.append("## Results")
    lines.append(
        f"A robust Theil-Sen trend fit gives slope={slope:.4f} "
        f"(bootstrap 95% CI [{slope_ci_lo:.4f}, {slope_ci_hi:.4f}], bootstrap p={bootstrap_p:.3g}) "
        "for median residual versus log10(rho_score). "
        f"Top-vs-bottom density groups show a pooled median residual shift of {pooled_shift:.4f} dex "
        f"(permutation p={perm_p:.3g})."
    )
    lines.append("")
    lines.append("### Baseline Top-Density Targets")
    for _, row in top_targets.iterrows():
        lines.append(f"- {row['galaxy']} (rho_score={row['rho_score']:.3e}, N_points={int(row['N_points'])})")
    lines.append("")
    lines.append("## Prediction")
    lines.append(
        "If coupling varies with density, deviations from the constant-coupling baseline should emerge first in the "
        "densest DM-dominated systems identified by rho_score."
    )
    lines.append("")
    lines.append("## How This Constrains rho_c")
    lines.append(
        "This target-ranked residual shift provides an empirical threshold handle: "
        "the onset of systematic residual drift across density windows brackets the transition scale rho_c "
        "before committing to a full parametric (rho_c, alpha) fit."
    )
    lines.append("")
    lines.append("## Next Tests")
    lines.append("- Dense x-sweep over coupling-transition models in simulation-informed priors.")
    lines.append("- Alternative baseline relations to test model dependence of residual shifts.")
    lines.append("- Focused follow-up on high-density low-mass systems with improved inner-point sampling.")
    lines.append("")
    lines.append("## Figure Index")
    lines.append("- Figure 1: rho_vs_residual_scatter.png")
    lines.append("- Figure 2: residual_hist_high_vs_low.png")
    lines.append("- Figure 3: residual_vs_gbar_high_density.png")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-3 bridge artifact pack.")
    parser.add_argument("--rar_points_file", type=str, default=None)
    parser.add_argument("--g_dagger", type=float, default=DEFAULT_G_DAGGER)
    parser.add_argument("--r_inner_max_kpc", type=float, default=2.0)
    parser.add_argument("--min_points_per_galaxy", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--assume_r_kpc", action="store_true")
    parser.add_argument(
        "--include_ss20",
        action="store_true",
        help="Include SS20_* single-point sources in baseline analysis (excluded by default).",
    )
    parser.add_argument(
        "--require_csv_sha",
        type=str,
        default=None,
        help="Require exact SHA256 hash match for input CSV; abort on mismatch.",
    )
    parser.add_argument(
        "--require_min_sparc_points",
        type=int,
        default=None,
        help="Require at least this many SPARC points in input dataset; abort if below threshold.",
    )
    parser.add_argument("--use_sigma_int_calibration", action="store_true")
    parser.add_argument("--n_perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    git_head = git_head_sha(repo_root)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = repo_root / "outputs" / "paper3_high_density" / timestamp
    else:
        user_out = Path(args.out_dir).expanduser()
        out_dir = user_out if user_out.is_absolute() else (repo_root / user_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_file, mapping = choose_rar_points_file(repo_root, args.rar_points_file)
    print(f"[P3] Using input file: {input_file}", flush=True)
    input_sha256 = sha256_file(input_file)
    print(f"[P3] Input SHA256: {input_sha256}", flush=True)
    if args.require_csv_sha is not None:
        expected_sha = str(args.require_csv_sha).strip().lower()
        actual_sha = str(input_sha256).strip().lower()
        if expected_sha != actual_sha:
            raise RuntimeError(
                f"Dataset hash mismatch: expected {expected_sha}, got {actual_sha}. Refusing to run."
            )
        print("[P3] CSV SHA requirement satisfied.", flush=True)

    raw = pd.read_csv(input_file)
    points_std, conversion_notes, all_galaxies = standardize_points(raw, mapping, args.assume_r_kpc)
    if args.require_min_sparc_points is not None:
        sparc_points = int(np.sum(points_std["dataset"].astype(str) == "SPARC"))
        if sparc_points < int(args.require_min_sparc_points):
            raise RuntimeError(
                f"SPARC point requirement failed: require_min_sparc_points={int(args.require_min_sparc_points)}, "
                f"found={sparc_points}. Refusing to run."
            )
        print(
            f"[P3] SPARC point requirement satisfied: found={sparc_points} "
            f"(threshold={int(args.require_min_sparc_points)})",
            flush=True,
        )
    points, filter_counts = apply_quality_filters(points_std)
    if points.empty:
        raise RuntimeError("No valid points after filtering.")

    ss20_excluded_points = 0
    if not args.include_ss20:
        ss20_mask = points["dataset"].astype(str).str.match(r"^SS20_")
        ss20_excluded_points = int(np.sum(ss20_mask))
        points = points.loc[~ss20_mask].copy()
        all_galaxies = sorted(points["galaxy_key"].astype(str).unique().tolist())
        print(
            f"[P3] Excluding SS20_* sources by default: removed_points={ss20_excluded_points}",
            flush=True,
        )
    else:
        all_galaxies = sorted(points["galaxy_key"].astype(str).unique().tolist())
        print("[P3] include_ss20=True: SS20_* sources retained", flush=True)

    if points.empty:
        raise RuntimeError("No valid points remain after SS20 source filter.")

    points = points.copy()
    points["g_pred"] = compute_rar_prediction(points["g_bar"].to_numpy(dtype=float), args.g_dagger)
    points["log_resid"] = np.log10(points["g_obs"]) - np.log10(points["g_pred"])
    points["rho_dyn"] = (3.0 * points["g_obs"]) / (4.0 * np.pi * G_NEWTON * (points["r_kpc"] * KPC_TO_M))
    points = points.replace([np.inf, -np.inf], np.nan).dropna(subset=["g_pred", "log_resid", "rho_dyn"])
    if points.empty:
        raise RuntimeError("No valid points after residual and rho computation.")

    rng = np.random.default_rng(RNG_SEED)
    r_grid = [1.0, 2.0, 3.0]
    if args.r_inner_max_kpc not in r_grid:
        r_grid = sorted(set(r_grid + [float(args.r_inner_max_kpc)]))

    run_results: Dict[float, Dict[str, object]] = {}
    for r_inner in r_grid:
        run_results[r_inner] = run_single_density_window(
            points=points,
            all_galaxies=all_galaxies,
            g_dagger=args.g_dagger,
            r_inner_max_kpc=r_inner,
            min_points=args.min_points_per_galaxy,
            top_n=args.top_n,
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            rng=rng,
        )

    baseline_r = 2.0
    if baseline_r not in run_results:
        raise RuntimeError("Baseline run for r_inner_max_kpc=2.0 was not computed.")
    baseline = run_results[baseline_r]
    baseline_top = baseline["top"].copy()
    baseline_summary = baseline["summary"].copy()
    baseline_eligible = baseline["eligible"].copy()
    baseline_bottom = baseline["bottom"].copy()

    sigma_cal_info: Dict[str, object] = {
        "used": False,
        "sigma_int_z": np.nan,
        "source": "summary_unified.json",
        "summary_path": str((repo_root / DEFAULT_SUMMARY_FILE).resolve()),
        "reason": "not_requested",
    }
    weighted_stats: Optional[Dict[str, float]] = None
    if args.use_sigma_int_calibration:
        sigma_cal_info = load_sigma_int_from_summary(repo_root)
        print(
            f"[P3] sigma_int calibration requested: used={sigma_cal_info['used']} "
            f"value={sigma_cal_info.get('sigma_int_z')} reason={sigma_cal_info.get('reason')}",
            flush=True,
        )
        if bool(sigma_cal_info.get("used", False)) and np.isfinite(float(sigma_cal_info.get("sigma_int_z"))):
            sigma_int_z = float(sigma_cal_info["sigma_int_z"])
            mu_res = float(np.nanmean(points["log_resid"].to_numpy(dtype=float)))
            std_res = float(np.nanstd(points["log_resid"].to_numpy(dtype=float)))
            if np.isfinite(std_res) and std_res > 0:
                points = points.copy()
                points["z_resid"] = (points["log_resid"] - mu_res) / std_res
                sigma_tot_z = float(np.sqrt(1.0 + sigma_int_z**2))
                top_names_tmp = set(baseline_top["galaxy_key"].tolist())
                bottom_names_tmp = set(baseline_bottom["galaxy_key"].tolist())
                top_z = points[points["galaxy_key"].isin(top_names_tmp)]["z_resid"].to_numpy(dtype=float)
                bottom_z = points[points["galaxy_key"].isin(bottom_names_tmp)]["z_resid"].to_numpy(dtype=float)

                def _weighted_mean_std(arr: np.ndarray, sigma_tot: float) -> Tuple[float, float]:
                    good = arr[np.isfinite(arr)]
                    if len(good) == 0:
                        return np.nan, np.nan
                    w = np.full_like(good, 1.0 / (sigma_tot**2), dtype=float)
                    mean = float(np.average(good, weights=w))
                    var = float(np.average((good - mean) ** 2, weights=w))
                    return mean, float(np.sqrt(max(var, 0.0)))

                top_w_mean, top_w_std = _weighted_mean_std(top_z, sigma_tot_z)
                bot_w_mean, bot_w_std = _weighted_mean_std(bottom_z, sigma_tot_z)
                weighted_stats = {
                    "sigma_int_z": sigma_int_z,
                    "sigma_tot_z": sigma_tot_z,
                    "top_weighted_mean_z": top_w_mean,
                    "top_weighted_std_z": top_w_std,
                    "bottom_weighted_mean_z": bot_w_mean,
                    "bottom_weighted_std_z": bot_w_std,
                    "top_n": float(np.sum(np.isfinite(top_z))),
                    "bottom_n": float(np.sum(np.isfinite(bottom_z))),
                }
            else:
                sigma_cal_info["reason"] = "z_resid_std_nonpositive"

    base_names = set(baseline_top["galaxy_key"].tolist())
    denom = max(len(base_names), 1)
    robustness_rows = []
    for r_inner in [1.0, 2.0, 3.0]:
        if r_inner not in run_results:
            continue
        res = run_results[r_inner]
        names = set(res["top"]["galaxy_key"].tolist())
        overlap = len(base_names.intersection(names)) / denom
        robustness_rows.append(
            {
                "r_inner_max_kpc": r_inner,
                "top20_overlap_vs_2kpc": overlap,
                "slope": res["slope"],
                "slope_ci_lo": res["slope_ci_lo"],
                "slope_ci_hi": res["slope_ci_hi"],
                "bootstrap_p": res["bootstrap_p"],
                "perm_p": res["perm_p_pooled"],
                "N_gal": len(res["eligible"]),
            }
        )
    robustness_df = pd.DataFrame.from_records(robustness_rows).sort_values("r_inner_max_kpc").reset_index(drop=True)

    high_density_targets = out_dir / "high_density_targets.csv"
    density_vs_residuals = out_dir / "density_vs_residuals.csv"
    scatter_plot = out_dir / "rho_vs_residual_scatter.png"
    hist_plot = out_dir / "residual_hist_high_vs_low.png"
    residual_vs_gbar_plot = out_dir / "residual_vs_gbar_high_density.png"
    report_path = out_dir / "paper3_density_bridge_report.txt"
    writeup_path = out_dir / "paper3_bridge_writeup.md"
    robustness_csv = out_dir / "robustness_summary.csv"

    high_density_cols = [
        "galaxy",
        "galaxy_key",
        "N_points",
        "rho_score",
        "rho_dyn_max",
        "rho_dyn_p50_inner",
        "rho_dyn_at_gdag",
        "r_at_gdag_kpc",
        "median_log_resid",
        "rms_log_resid",
        "p90_abs_log_resid",
        "flag_low_points",
        "notes",
    ]
    baseline_top.loc[:, high_density_cols].to_csv(high_density_targets, index=False)

    density_cols = [
        "galaxy",
        "galaxy_key",
        "N_points",
        "rho_score",
        "median_log_resid",
        "rms_log_resid",
        "p90_abs_log_resid",
        "dataset",
        "flag_low_points",
    ]
    density_df = baseline_summary.loc[:, density_cols].copy()
    density_df["log10_rho_score"] = np.log10(density_df["rho_score"].to_numpy(dtype=float))
    density_df = density_df[
        [
            "galaxy",
            "galaxy_key",
            "N_points",
            "rho_score",
            "log10_rho_score",
            "median_log_resid",
            "rms_log_resid",
            "p90_abs_log_resid",
            "dataset",
            "flag_low_points",
        ]
    ]
    density_df.to_csv(density_vs_residuals, index=False)

    top_names = set(baseline_top["galaxy_key"].tolist())
    points_top = points[points["galaxy_key"].isin(top_names)].copy()
    save_scatter_plot(scatter_plot, baseline_eligible, baseline["slope"], baseline["intercept"])
    save_hist_plot(
        hist_plot,
        baseline["top_points_resid"],
        baseline["bottom_points_resid"],
        baseline["top_gal_median"],
        baseline["bottom_gal_median"],
    )
    save_residual_vs_gbar_plot(residual_vs_gbar_plot, points_top)
    robustness_df.to_csv(robustness_csv, index=False)

    top20_n_points = baseline_top["N_points"].to_numpy(dtype=float)
    bottom20_n_points = baseline_bottom["N_points"].to_numpy(dtype=float)
    source_hist_df = source_histogram(points)
    point_counts = {
        "top20_median": float(np.nanmedian(top20_n_points)) if len(top20_n_points) else np.nan,
        "bottom20_median": float(np.nanmedian(bottom20_n_points)) if len(bottom20_n_points) else np.nan,
    }
    write_bridge_report(
        out_path=report_path,
        input_file=input_file,
        rar_points_sha256=input_sha256,
        require_csv_sha=str(args.require_csv_sha) if args.require_csv_sha is not None else None,
        git_head=git_head,
        g_dagger=args.g_dagger,
        baseline_r_inner=baseline_r,
        min_points=args.min_points_per_galaxy,
        top_n=args.top_n,
        include_ss20=bool(args.include_ss20),
        ss20_excluded_points=int(ss20_excluded_points),
        top_targets=baseline_top,
        bottom_targets=baseline_bottom,
        baseline_eligible=baseline_eligible,
        baseline_result=baseline,
        robustness_df=robustness_df,
        source_hist=source_hist_df,
        point_counts=point_counts,
        mapping=mapping,
        conversion_notes=conversion_notes,
        filter_counts=filter_counts,
        sigma_cal_info=sigma_cal_info,
        weighted_stats=weighted_stats,
    )
    write_writeup_stub(
        out_path=writeup_path,
        baseline_r_inner=baseline_r,
        top_targets=baseline_top,
        slope=baseline["slope"],
        slope_ci_lo=baseline["slope_ci_lo"],
        slope_ci_hi=baseline["slope_ci_hi"],
        bootstrap_p=baseline["bootstrap_p"],
        perm_p=baseline["perm_p_pooled"],
        pooled_shift=baseline["shift_pooled"],
    )

    print(f"[P3] Output folder: {out_dir}", flush=True)
    print("[P3] Baseline top 20 list (r_inner_max_kpc=2.0):", flush=True)
    for _, row in baseline_top.iterrows():
        print(f"  {row['galaxy']}, {row['rho_score']:.6e}", flush=True)
    print(
        f"[P3] Key stats: slope={baseline['slope']:.6f}, bootstrap_p={baseline['bootstrap_p']:.6g}, "
        f"permutation_p={baseline['perm_p_pooled']:.6g}",
        flush=True,
    )
    print("[P3] Output files:", flush=True)
    for path in (
        high_density_targets,
        density_vs_residuals,
        scatter_plot,
        hist_plot,
        residual_vs_gbar_plot,
        report_path,
        writeup_path,
        robustness_csv,
    ):
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
