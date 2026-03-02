#!/usr/bin/env python3
"""
Paper 3 rescue: g_dagger-anchored rho_c lever with systematics controls.

Upgrades:
1) Resolution/systematics cuts on r_gdag and window support.
2) Robust per-galaxy residual summaries (median, trimmed mean, Huber mean).
3) Galaxy-bootstrap confidence intervals for top-vs-bottom effect sizes.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
DEFAULT_RAR_POINTS = "analysis/results/rar_points_unified.csv"
RNG_SEED = 314159
EPS = np.finfo(float).tiny

AGGREGATORS = ("resid_median", "resid_trimmed_mean", "resid_huber_mean")


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


def _check_required_fields(mapping: Dict[str, Optional[str]]) -> None:
    missing: List[str] = []
    if mapping["galaxy"] is None:
        missing.append("galaxy identifier")
    if mapping["radius"] is None:
        missing.append("radius/r_kpc")

    has_gbar = (mapping["gbar_linear"] is not None) or (mapping["gbar_log"] is not None)
    has_gobs = (mapping["gobs_linear"] is not None) or (mapping["gobs_log"] is not None)
    has_vel = (mapping["v_obs"] is not None) and (mapping["v_bar"] is not None)

    if not has_gbar and not has_vel:
        missing.append("g_bar (or v_bar + radius)")
    if not has_gobs and not has_vel:
        missing.append("g_obs (or v_obs + radius)")

    if missing:
        raise RuntimeError("STOP: missing required per-point fields: " + ", ".join(missing))


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _radius_to_kpc(radius_raw: np.ndarray, radius_col: str) -> np.ndarray:
    name = radius_col.lower()
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
    lower = col_name.lower()
    if "km" in lower:
        return velocity * 1_000.0
    if "m/s" in lower or "ms-1" in lower:
        return velocity
    med = np.nanmedian(np.abs(velocity))
    if not np.isfinite(med):
        return velocity
    return velocity * 1_000.0 if med < 1.0e4 else velocity


def load_points(repo_root: Path, rar_points_file: str, g_dagger: float) -> Tuple[pd.DataFrame, Path]:
    input_path = Path(rar_points_file).expanduser()
    if not input_path.is_absolute():
        input_path = (repo_root / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--rar_points_file does not exist: {input_path}")

    raw = pd.read_csv(input_path)
    mapping = infer_column_mapping(raw.columns)
    _check_required_fields(mapping)

    galaxy_col = mapping["galaxy"]
    radius_col = mapping["radius"]
    if galaxy_col is None or radius_col is None:
        raise RuntimeError("STOP: failed to resolve galaxy/radius columns")

    points = pd.DataFrame()
    points["galaxy"] = raw[galaxy_col].astype(str).str.strip()
    points["r_kpc"] = _radius_to_kpc(_to_numeric(raw[radius_col]), radius_col)

    g_bar: Optional[np.ndarray] = None
    g_obs: Optional[np.ndarray] = None

    if mapping["gbar_linear"] is not None:
        g_bar = _to_numeric(raw[mapping["gbar_linear"]])
    elif mapping["gbar_log"] is not None:
        g_bar = np.power(10.0, _to_numeric(raw[mapping["gbar_log"]]))

    if mapping["gobs_linear"] is not None:
        g_obs = _to_numeric(raw[mapping["gobs_linear"]])
    elif mapping["gobs_log"] is not None:
        g_obs = np.power(10.0, _to_numeric(raw[mapping["gobs_log"]]))

    if g_bar is None or g_obs is None:
        if mapping["v_obs"] is None or mapping["v_bar"] is None:
            raise RuntimeError("STOP: cannot compute g_bar/g_obs from available columns")
        r_m = points["r_kpc"].to_numpy(dtype=float) * KPC_TO_M
        v_obs = _velocity_to_m_s(_to_numeric(raw[mapping["v_obs"]]), mapping["v_obs"])
        v_bar = _velocity_to_m_s(_to_numeric(raw[mapping["v_bar"]]), mapping["v_bar"])
        g_obs = np.square(v_obs) / r_m
        g_bar = np.square(v_bar) / r_m

    points["g_bar"] = g_bar
    points["g_obs"] = g_obs
    points = points.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["galaxy", "r_kpc", "g_bar", "g_obs"]
    )
    points = points[
        (points["r_kpc"] > 0.0) & (points["g_bar"] > 0.0) & (points["g_obs"] > 0.0)
    ].copy()
    if points.empty:
        raise RuntimeError("STOP: no usable rows after quality filters")

    ratio = np.clip(points["g_bar"].to_numpy(dtype=float) / g_dagger, a_min=0.0, a_max=None)
    x = np.sqrt(ratio)
    denom = -np.expm1(-x)
    points["g_pred"] = np.divide(
        points["g_bar"].to_numpy(dtype=float),
        denom,
        out=np.full(len(points), np.nan, dtype=float),
        where=denom > EPS,
    )
    points["log_resid"] = (
        np.log10(points["g_obs"].to_numpy(dtype=float))
        - np.log10(points["g_pred"].to_numpy(dtype=float))
    )
    points["rho_dyn"] = (
        3.0
        * points["g_obs"].to_numpy(dtype=float)
        / (4.0 * np.pi * G_NEWTON * (points["r_kpc"].to_numpy(dtype=float) * KPC_TO_M))
    )
    points = points.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_resid", "rho_dyn"])
    if points.empty:
        raise RuntimeError("STOP: no usable rows after residual/rho calculations")
    return points, input_path


def trimmed_mean(values: np.ndarray, trim_frac: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if trim_frac <= 0:
        return float(np.mean(arr))
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    n_trim = int(np.floor(trim_frac * n))
    if (2 * n_trim) >= n:
        return float(np.mean(arr_sorted))
    trimmed = arr_sorted[n_trim : n - n_trim]
    return float(np.mean(trimmed))


def huber_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mu = float(np.median(arr))
    mad = float(np.median(np.abs(arr - mu)))
    if not np.isfinite(mad) or mad <= EPS:
        return float(np.mean(arr))
    k = 1.5 * mad
    if k <= EPS:
        return float(np.mean(arr))

    for _ in range(30):
        resid = arr - mu
        w = np.ones_like(arr)
        mask = np.abs(resid) > k
        w[mask] = k / np.abs(resid[mask])
        mu_new = float(np.sum(w * arr) / np.sum(w))
        if abs(mu_new - mu) < 1.0e-10:
            mu = mu_new
            break
        mu = mu_new
    return mu


def summarize_galaxies(
    points: pd.DataFrame,
    g_dagger: float,
    min_points: int,
    window_dex: float,
    min_window_points: int,
    trim_frac: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    log_gdag = np.log10(g_dagger)

    for galaxy, gdf in points.groupby("galaxy", sort=False):
        n_points = int(len(gdf))
        log_gbar = np.log10(gdf["g_bar"].to_numpy(dtype=float))
        abs_delta = np.abs(log_gbar - log_gdag)

        i_star = int(np.argmin(abs_delta))
        closest = gdf.iloc[i_star]
        r_gdag_kpc = float(closest["r_kpc"])
        rho_gdag = float(closest["rho_dyn"])
        resid_gdag = float(closest["log_resid"])

        in_window = abs_delta <= window_dex
        n_window = int(np.sum(in_window))
        if n_window >= min_window_points:
            win_rho = gdf.loc[in_window, "rho_dyn"].to_numpy(dtype=float)
            win_resid = gdf.loc[in_window, "log_resid"].to_numpy(dtype=float)
            rho_score = float(np.median(win_rho))
            flag_low_window = False
        else:
            win_rho = np.array([rho_gdag], dtype=float)
            win_resid = np.array([resid_gdag], dtype=float)
            rho_score = float(rho_gdag)
            flag_low_window = True

        rows.append(
            {
                "galaxy": galaxy,
                "N_points": n_points,
                "r_gdag_kpc": r_gdag_kpc,
                "rho_gdag": rho_gdag,
                "resid_gdag": resid_gdag,
                "n_window": n_window,
                "rho_score": rho_score,
                "resid_median": float(np.median(win_resid)),
                "resid_trimmed_mean": trimmed_mean(win_resid, trim_frac=trim_frac),
                "resid_huber_mean": huber_mean(win_resid),
                "flag_low_points": bool(n_points < min_points),
                "flag_low_window": bool(flag_low_window),
            }
        )

    summary = pd.DataFrame(rows)
    summary["pass_min_points"] = summary["N_points"] >= min_points
    return summary


def select_top_bottom(
    df: pd.DataFrame, score_col: str, top_n: int
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    n = len(df)
    if n < 2:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), 0
    top_n_eff = min(top_n, n // 2)
    if top_n_eff < 1:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), 0

    ranked = df.sort_values(score_col, ascending=False)
    top = ranked.head(top_n_eff).copy()
    bottom_pool = df.loc[~df.index.isin(top.index)].copy()
    bottom = bottom_pool.sort_values(score_col, ascending=True).head(top_n_eff).copy()
    return top, bottom, top_n_eff


def permutation_delta(
    values: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    observed = float(np.median(values[top_mask]) - np.median(values[bottom_mask]))
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(values)
        diff = float(np.median(perm[top_mask]) - np.median(perm[bottom_mask]))
        if abs(diff) >= abs(observed):
            extreme += 1
    p_val = (extreme + 1) / (n_perm + 1)
    return observed, float(p_val)


def bootstrap_delta(
    eligible: pd.DataFrame,
    agg_col: str,
    top_n: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(eligible)
    if n < 2:
        return np.array([], dtype=float)
    boot_vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = eligible.iloc[idx].copy()
        top_b, bottom_b, top_n_eff_b = select_top_bottom(boot, score_col="rho_score", top_n=top_n)
        if top_n_eff_b < 1 or top_b.empty or bottom_b.empty:
            continue
        t = top_b[agg_col].to_numpy(dtype=float)
        b = bottom_b[agg_col].to_numpy(dtype=float)
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(b))):
            continue
        boot_vals.append(float(np.median(t) - np.median(b)))
    return np.array(boot_vals, dtype=float)


def compute_effects(
    eligible: pd.DataFrame,
    top_n: int,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    top, bottom, top_n_eff = select_top_bottom(eligible, score_col="rho_score", top_n=top_n)

    effects_rows: List[Dict[str, object]] = []
    if top_n_eff < 1 or top.empty or bottom.empty:
        for agg in AGGREGATORS:
            effects_rows.append(
                {
                    "aggregator": agg,
                    "n_gal_eligible": int(len(eligible)),
                    "top_n_used": int(top_n_eff),
                    "bottom_n_used": int(top_n_eff),
                    "delta": np.nan,
                    "perm_p": np.nan,
                    "delta_boot_median": np.nan,
                    "delta_boot_ci_lo_95": np.nan,
                    "delta_boot_ci_hi_95": np.nan,
                    "frac_boot_delta_gt0": np.nan,
                    "boot_n_success": 0,
                }
            )
        return pd.DataFrame(effects_rows), top, bottom, top_n_eff

    top_mask = np.asarray(eligible.index.isin(top.index), dtype=bool)
    bottom_mask = np.asarray(eligible.index.isin(bottom.index), dtype=bool)

    for agg in AGGREGATORS:
        vals = eligible[agg].to_numpy(dtype=float)
        delta = np.nan
        perm_p = np.nan
        if np.all(np.isfinite(vals[top_mask])) and np.all(np.isfinite(vals[bottom_mask])):
            delta, perm_p = permutation_delta(
                values=vals,
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                n_perm=n_perm,
                rng=rng,
            )

        boots = bootstrap_delta(
            eligible=eligible,
            agg_col=agg,
            top_n=top_n_eff,
            n_boot=n_boot,
            rng=rng,
        )
        if boots.size:
            b_med = float(np.median(boots))
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            frac_pos = float(np.mean(boots > 0.0))
            n_succ = int(boots.size)
        else:
            b_med = np.nan
            ci_lo = np.nan
            ci_hi = np.nan
            frac_pos = np.nan
            n_succ = 0

        effects_rows.append(
            {
                "aggregator": agg,
                "n_gal_eligible": int(len(eligible)),
                "top_n_used": int(top_n_eff),
                "bottom_n_used": int(top_n_eff),
                "delta": float(delta) if np.isfinite(delta) else np.nan,
                "perm_p": float(perm_p) if np.isfinite(perm_p) else np.nan,
                "delta_boot_median": b_med,
                "delta_boot_ci_lo_95": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                "delta_boot_ci_hi_95": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
                "frac_boot_delta_gt0": frac_pos,
                "boot_n_success": n_succ,
            }
        )
    return pd.DataFrame(effects_rows), top, bottom, top_n_eff


def theilsen_scatter_fit(eligible: pd.DataFrame) -> Tuple[float, float]:
    x = np.log10(eligible["rho_score"].to_numpy(dtype=float))
    y = eligible["resid_median"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.nanstd(x) <= 0:
        return np.nan, np.nan
    slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
    return float(slope), float(intercept)


def save_plot(
    out_path: Path,
    cutflow_labels: List[str],
    cutflow_counts: List[int],
    effects: pd.DataFrame,
    eligible: pd.DataFrame,
    slope: float,
    intercept: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel A: cutflow
    ax1.bar(np.arange(len(cutflow_labels)), cutflow_counts, color="#334155")
    ax1.set_xticks(np.arange(len(cutflow_labels)))
    ax1.set_xticklabels(cutflow_labels, rotation=25, ha="right")
    ax1.set_ylabel("N galaxies")
    ax1.set_title("Panel A: Cutflow")
    ax1.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Panel B: delta with bootstrap CI by aggregator
    order = list(AGGREGATORS)
    eplot = effects.set_index("aggregator").reindex(order).reset_index()
    x = np.arange(len(eplot))
    delta = eplot["delta"].to_numpy(dtype=float)
    ci_lo = eplot["delta_boot_ci_lo_95"].to_numpy(dtype=float)
    ci_hi = eplot["delta_boot_ci_hi_95"].to_numpy(dtype=float)
    yerr = np.vstack([delta - ci_lo, ci_hi - delta])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    ax2.errorbar(x, delta, yerr=yerr, fmt="o", color="#0b6e4f", capsize=4, lw=1.5)
    ax2.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["median", "trim10", "huber"], rotation=0)
    ax2.set_ylabel("delta (top-bottom)")
    ax2.set_title("Panel B: Effect sizes + bootstrap 95% CI")
    ax2.grid(True, alpha=0.3, linestyle=":")

    # Panel C: permutation p-values
    perm = eplot["perm_p"].to_numpy(dtype=float)
    ax3.bar(x, perm, color="#1d4ed8")
    ax3.axhline(0.05, color="crimson", lw=1.2, ls="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["median", "trim10", "huber"], rotation=0)
    ax3.set_ylabel("perm p-value")
    ax3.set_ylim(0.0, max(0.1, float(np.nanmax(perm) * 1.15) if np.any(np.isfinite(perm)) else 0.1))
    ax3.set_title("Panel C: Permutation p-values")
    ax3.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Panel D: post-cut scatter with Theil-Sen fit
    if not eligible.empty:
        sx = np.log10(eligible["rho_score"].to_numpy(dtype=float))
        sy = eligible["resid_median"].to_numpy(dtype=float)
        ax4.scatter(sx, sy, s=22, alpha=0.6, color="#475569")
        if np.isfinite(slope) and np.isfinite(intercept):
            xg = np.linspace(np.nanmin(sx), np.nanmax(sx), 200)
            yg = intercept + slope * xg
            ax4.plot(xg, yg, color="black", lw=2.0)
    else:
        ax4.text(0.5, 0.5, "No eligible galaxies after cuts", transform=ax4.transAxes, ha="center")
    ax4.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax4.set_xlabel("log10(rho_score)")
    ax4.set_ylabel("resid_median")
    ax4.set_title("Panel D: Post-cut rho_score vs resid_median")
    ax4.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_report(
    out_path: Path,
    input_file: Path,
    cutflow: Dict[str, int],
    effects: pd.DataFrame,
    top_n_requested: int,
    top_n_used: int,
    robust: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Paper 3 g_dagger rescue report")
    lines.append("")
    lines.append(f"Input file: `{input_file}`")
    lines.append("")
    lines.append("Cutflow")
    lines.append(f"- Galaxies with finite summary metrics: {cutflow['finite_summary']}")
    lines.append(f"- After `N_points >= min_points`: {cutflow['after_min_points']} (removed {cutflow['removed_min_points']})")
    lines.append(f"- After `r_gdag_kpc >= r_gdag_min_kpc`: {cutflow['after_r_gdag']} (removed {cutflow['removed_r_gdag']})")
    lines.append(f"- After `n_window >= min_window_points`: {cutflow['after_window']} (removed {cutflow['removed_window']})")
    lines.append("")
    if top_n_used < top_n_requested:
        lines.append(
            f"Grouping note: reduced `top_n` from {top_n_requested} to {top_n_used} "
            "because fewer than 2*top_n galaxies passed cuts."
        )
        lines.append("")

    lines.append("Aggregator results")
    for _, row in effects.iterrows():
        lines.append(
            f"- `{row['aggregator']}`: delta={row['delta']:.6f}, "
            f"perm_p={row['perm_p']:.6g}, "
            f"boot_med={row['delta_boot_median']:.6f}, "
            f"CI95=[{row['delta_boot_ci_lo_95']:.6f}, {row['delta_boot_ci_hi_95']:.6f}], "
            f"frac(delta>0)={row['frac_boot_delta_gt0']:.3f}"
        )
    lines.append("")

    if robust:
        lines.append(
            "Conclusion: **robust enough** as a candidate `rho_c` lever "
            "(at least one aggregator has perm_p < 0.05 and bootstrap CI excludes 0)."
        )
    else:
        lines.append(
            "Conclusion: **still not robust**; treat this as a target-prioritization list only."
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescue attempt for g_dagger-anchored rho_c lever with robust stats."
    )
    parser.add_argument("--rar_points_file", type=str, default=DEFAULT_RAR_POINTS)
    parser.add_argument("--g_dagger", type=float, default=DEFAULT_G_DAGGER)
    parser.add_argument("--window_dex", type=float, default=0.2)
    parser.add_argument("--min_points", type=int, default=20)
    parser.add_argument("--min_window_points", type=int, default=5)
    parser.add_argument("--r_gdag_min_kpc", type=float, default=0.20)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--trim_frac", type=float, default=0.10)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(RNG_SEED)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = repo_root / "outputs" / "paper3_gdag_anchor_rescue" / timestamp
    else:
        out_user = Path(args.out_dir).expanduser()
        out_dir = out_user if out_user.is_absolute() else (repo_root / out_user)
    out_dir.mkdir(parents=True, exist_ok=True)

    points, input_file = load_points(repo_root, args.rar_points_file, args.g_dagger)
    summary = summarize_galaxies(
        points=points,
        g_dagger=args.g_dagger,
        min_points=args.min_points,
        window_dex=args.window_dex,
        min_window_points=args.min_window_points,
        trim_frac=args.trim_frac,
    )

    finite_mask = (
        np.isfinite(summary["rho_score"])
        & np.isfinite(summary["resid_median"])
        & np.isfinite(summary["resid_trimmed_mean"])
        & np.isfinite(summary["resid_huber_mean"])
        & np.isfinite(summary["r_gdag_kpc"])
        & np.isfinite(summary["n_window"])
    )
    finite = summary[finite_mask].copy()

    after_min_points = finite[finite["N_points"] >= args.min_points].copy()
    after_r_gdag = after_min_points[after_min_points["r_gdag_kpc"] >= args.r_gdag_min_kpc].copy()
    eligible = after_r_gdag[after_r_gdag["n_window"] >= args.min_window_points].copy()
    eligible = eligible.sort_values("rho_score", ascending=False)

    cutflow = {
        "finite_summary": int(len(finite)),
        "after_min_points": int(len(after_min_points)),
        "after_r_gdag": int(len(after_r_gdag)),
        "after_window": int(len(eligible)),
        "removed_min_points": int(len(finite) - len(after_min_points)),
        "removed_r_gdag": int(len(after_min_points) - len(after_r_gdag)),
        "removed_window": int(len(after_r_gdag) - len(eligible)),
    }

    # Add pass flags to the full summary table for traceability.
    summary["pass_finite"] = finite_mask
    summary["pass_min_points"] = summary["N_points"] >= args.min_points
    summary["pass_r_gdag"] = summary["r_gdag_kpc"] >= args.r_gdag_min_kpc
    summary["pass_window"] = summary["n_window"] >= args.min_window_points
    summary["pass_all_cuts"] = (
        summary["pass_finite"]
        & summary["pass_min_points"]
        & summary["pass_r_gdag"]
        & summary["pass_window"]
    )

    effects, top, bottom, top_n_used = compute_effects(
        eligible=eligible,
        top_n=args.top_n,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        rng=rng,
    )

    slope, intercept = theilsen_scatter_fit(eligible)

    robust_flags = []
    for _, row in effects.iterrows():
        ci_excludes_zero = np.isfinite(row["delta_boot_ci_lo_95"]) and np.isfinite(row["delta_boot_ci_hi_95"]) and (
            (row["delta_boot_ci_lo_95"] > 0.0) or (row["delta_boot_ci_hi_95"] < 0.0)
        )
        sig_perm = np.isfinite(row["perm_p"]) and (row["perm_p"] < 0.05)
        robust_flags.append(bool(ci_excludes_zero and sig_perm))
    robust = bool(any(robust_flags))
    effects["robust_candidate"] = robust_flags

    summary_csv = out_dir / "rescue_summary.csv"
    effects_csv = out_dir / "rescue_effects.csv"
    plots_png = out_dir / "rescue_plots.png"
    report_md = out_dir / "rescue_report.md"

    summary[
        [
            "galaxy",
            "N_points",
            "r_gdag_kpc",
            "rho_gdag",
            "resid_gdag",
            "n_window",
            "rho_score",
            "resid_median",
            "resid_trimmed_mean",
            "resid_huber_mean",
            "flag_low_points",
            "flag_low_window",
            "pass_finite",
            "pass_min_points",
            "pass_r_gdag",
            "pass_window",
            "pass_all_cuts",
        ]
    ].to_csv(summary_csv, index=False)

    effects.to_csv(effects_csv, index=False)

    cut_labels = ["finite", "N>=min", "r_gdag", "n_window"]
    cut_counts = [
        cutflow["finite_summary"],
        cutflow["after_min_points"],
        cutflow["after_r_gdag"],
        cutflow["after_window"],
    ]
    save_plot(
        out_path=plots_png,
        cutflow_labels=cut_labels,
        cutflow_counts=cut_counts,
        effects=effects,
        eligible=eligible,
        slope=slope,
        intercept=intercept,
    )

    build_report(
        out_path=report_md,
        input_file=input_file,
        cutflow=cutflow,
        effects=effects,
        top_n_requested=args.top_n,
        top_n_used=top_n_used,
        robust=robust,
    )

    print(f"[P3RSC] Input file: {input_file}")
    print(f"[P3RSC] Eligible galaxies after cuts: {len(eligible)}")
    print(f"[P3RSC] top_n requested/used: {args.top_n}/{top_n_used}")
    for _, row in effects.iterrows():
        print(
            f"[P3RSC] {row['aggregator']}: "
            f"delta={row['delta']:.6f}, perm_p={row['perm_p']:.6g}, "
            f"CI95=[{row['delta_boot_ci_lo_95']:.6f}, {row['delta_boot_ci_hi_95']:.6f}]"
        )
    print(f"[P3RSC] Robust enough: {robust}")
    print(f"[P3RSC] Output folder: {out_dir}")
    print(f"[P3RSC] Files:\n  {summary_csv}\n  {effects_csv}\n  {plots_png}\n  {report_md}")


if __name__ == "__main__":
    main()
