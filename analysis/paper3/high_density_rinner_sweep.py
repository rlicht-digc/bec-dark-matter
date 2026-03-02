#!/usr/bin/env python3
"""
Paper 3: r_inner_max_kpc sweep for high-density residual separation.

This diagnostic quantifies how the density-residual effect changes with inner
window choice and reports effect size, significance, and top-target stability.
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
DEFAULT_N_BOOT = 2000
EPS = np.finfo(float).tiny
RNG_SEED = 314159


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
    return velocity * 1_000.0 if med < 1.0e4 else velocity


def parse_r_inner_grid(spec: str) -> List[float]:
    toks = [x.strip() for x in spec.split(":")]
    if len(toks) != 3:
        raise ValueError("--r_inner_grid must use start:end:step format")
    start, stop, step = map(float, toks)
    if step <= 0.0:
        raise ValueError("r_inner_grid step must be > 0")
    if stop < start:
        raise ValueError("r_inner_grid stop must be >= start")

    arr = np.arange(start, stop + 0.5 * step, step, dtype=float)
    arr = np.round(arr, 6)
    arr = arr[arr <= stop + 1e-9]
    if arr.size == 0:
        raise ValueError("r_inner_grid produced no values")
    return [float(x) for x in arr]


def load_and_prepare_points(
    repo_root: Path, rar_points_file: str, g_dagger: float
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Path]:
    input_path = Path(rar_points_file).expanduser()
    if not input_path.is_absolute():
        input_path = (repo_root / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--rar_points_file does not exist: {input_path}")

    raw = pd.read_csv(input_path)
    mapping = infer_column_mapping(raw.columns)
    if not _is_usable_mapping(mapping):
        raise ValueError(f"Input file lacks required fields: {input_path}")

    galaxy_col = mapping["galaxy"]
    radius_col = mapping["radius"]
    if galaxy_col is None or radius_col is None:
        raise ValueError("Missing required galaxy or radius column")

    points = pd.DataFrame()
    points["galaxy"] = raw[galaxy_col].astype(str).str.strip()
    points["dataset"] = (
        raw[mapping["dataset"]].astype(str).str.strip()
        if mapping["dataset"] is not None
        else "unknown"
    )
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
            raise ValueError(
                "Need either direct g_bar/g_obs columns or both v_obs/v_bar columns"
            )
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
        raise RuntimeError("No usable points after quality filters")

    ratio = np.clip(points["g_bar"].to_numpy(dtype=float) / g_dagger, a_min=0.0, a_max=None)
    denom = -np.expm1(-np.sqrt(ratio))
    g_pred = np.divide(
        points["g_bar"].to_numpy(dtype=float),
        denom,
        out=np.full(len(points), np.nan, dtype=float),
        where=denom > EPS,
    )
    points["g_pred"] = g_pred
    points["log_resid"] = np.log10(points["g_obs"].to_numpy(dtype=float)) - np.log10(g_pred)
    points["rho_dyn"] = (
        3.0
        * points["g_obs"].to_numpy(dtype=float)
        / (4.0 * np.pi * G_NEWTON * (points["r_kpc"].to_numpy(dtype=float) * KPC_TO_M))
    )
    points = points.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_resid", "rho_dyn"])
    if points.empty:
        raise RuntimeError("No usable points after residual/density construction")

    return points, mapping, input_path


def summarize_galaxies(points: pd.DataFrame, r_inner_max_kpc: float, min_points: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for galaxy, gdf in points.groupby("galaxy", sort=False):
        n_points = int(len(gdf))
        inner = gdf[gdf["r_kpc"] <= r_inner_max_kpc]

        rho_score = float(inner["rho_dyn"].median()) if len(inner) > 0 else np.nan
        used_fallback = False
        if not np.isfinite(rho_score):
            rho_score = float(gdf["rho_dyn"].max())
            used_fallback = True

        rows.append(
            {
                "galaxy": galaxy,
                "N_points": n_points,
                "rho_score": rho_score,
                "median_log_resid": float(np.median(gdf["log_resid"].to_numpy(dtype=float))),
                "flag_low_points": bool(n_points < min_points),
                "used_rho_fallback_max": bool(used_fallback),
                "n_inner_points": int(len(inner)),
            }
        )
    return pd.DataFrame(rows)


def permutation_test_delta_median(
    top_vals: np.ndarray, bottom_vals: np.ndarray, n_perm: int, rng: np.random.Generator
) -> Tuple[float, float]:
    # Red-team check requirement: use per-galaxy medians only, not pooled points.
    obs = float(np.median(top_vals) - np.median(bottom_vals))
    combined = np.concatenate([top_vals, bottom_vals])
    n_top = len(top_vals)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = float(np.median(perm[:n_top]) - np.median(perm[n_top:]))
        if abs(diff) >= abs(obs):
            extreme += 1
    return obs, float((extreme + 1) / (n_perm + 1))


def bootstrap_slope_pvalue(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator
) -> float:
    n = len(x)
    if n < 3:
        return np.nan
    slopes: List[float] = []
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
        return np.nan
    arr = np.array(slopes, dtype=float)
    return float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))


def jaccard_index(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return np.nan
    union = sa | sb
    if not union:
        return np.nan
    return float(len(sa & sb) / len(union))


def classify_robustness(summary_df: pd.DataFrame) -> Tuple[str, str]:
    df = summary_df.sort_values("r_inner_max_kpc").reset_index(drop=True)
    low_p = df["perm_p"].to_numpy(dtype=float) <= 0.05
    jacc = df["top20_jaccard_vs_2kpc"].to_numpy(dtype=float)

    longest = 0
    current = 0
    for flag in low_p:
        if bool(flag):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    med_jacc_low_p = float(np.nanmedian(jacc[low_p])) if np.any(low_p) else np.nan
    if longest >= 3 and np.isfinite(med_jacc_low_p) and med_jacc_low_p >= 0.35:
        return (
            "robust plateau",
            "Low p-values persist across multiple adjacent windows with stable top-target overlap.",
        )
    return (
        "window-specific spike",
        "Low p-values are narrow or accompanied by weak top-target overlap stability.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep r_inner_max_kpc to test Paper 3 density-residual separation stability."
    )
    parser.add_argument("--rar_points_file", type=str, default=DEFAULT_RAR_POINTS)
    parser.add_argument("--g_dagger", type=float, default=DEFAULT_G_DAGGER)
    parser.add_argument("--min_points", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--r_inner_grid", type=str, default="0.5:5.0:0.25")
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def save_plots(out_path: Path, summary_df: pd.DataFrame) -> None:
    sdf = summary_df.sort_values("r_inner_max_kpc").reset_index(drop=True)
    r = sdf["r_inner_max_kpc"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    perm_p = np.clip(sdf["perm_p"].to_numpy(dtype=float), 1.0e-6, 1.0)
    ax1.plot(r, perm_p, marker="o", lw=1.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("r_inner_max_kpc")
    ax1.set_ylabel("Permutation p-value")
    ax1.set_title("Panel A: perm_p vs r_inner")
    ax1.grid(True, alpha=0.35, linestyle=":")

    ax2.plot(r, sdf["delta_median"].to_numpy(dtype=float), marker="o", lw=1.5, color="#0b6e4f")
    ax2.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax2.set_xlabel("r_inner_max_kpc")
    ax2.set_ylabel("delta_median (dex)")
    ax2.set_title("Panel B: delta_median vs r_inner")
    ax2.grid(True, alpha=0.35, linestyle=":")

    ax3.plot(
        r,
        sdf["top20_jaccard_vs_2kpc"].to_numpy(dtype=float),
        marker="o",
        lw=1.5,
        color="#7c3aed",
    )
    ax3.set_xlabel("r_inner_max_kpc")
    ax3.set_ylabel("Jaccard overlap vs 2.0 kpc")
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_title("Panel C: top20 stability")
    ax3.grid(True, alpha=0.35, linestyle=":")

    ax4.plot(
        r,
        sdf["n_gal_eligible"].to_numpy(dtype=float),
        marker="o",
        lw=1.5,
        color="#b91c1c",
    )
    ax4.set_xlabel("r_inner_max_kpc")
    ax4.set_ylabel("n_gal_eligible")
    ax4.set_title("Panel D: eligible galaxies vs r_inner")
    ax4.grid(True, alpha=0.35, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_report(
    out_path: Path,
    input_file: Path,
    summary_df: pd.DataFrame,
    top20_df: pd.DataFrame,
    best_p_row: pd.Series,
    best_effect_row: pd.Series,
    classification: str,
    class_reason: str,
) -> None:
    srt = summary_df.sort_values("r_inner_max_kpc").reset_index(drop=True)
    rvals = srt["r_inner_max_kpc"].to_numpy(dtype=float)
    nvals = srt["n_gal_eligible"].to_numpy(dtype=float)
    pvals = srt["perm_p"].to_numpy(dtype=float)
    jvals = srt["top20_jaccard_vs_2kpc"].to_numpy(dtype=float)

    small_drop_warn = ""
    if len(nvals) > 1:
        max_n = np.nanmax(nvals)
        min_r_idx = int(np.nanargmin(rvals))
        min_r_n = nvals[min_r_idx]
        if np.isfinite(max_n) and max_n > 0 and min_r_n / max_n < 0.7:
            small_drop_warn = (
                f"Selection warning: eligible galaxies at smallest window are {min_r_n:.0f}, "
                f"only {100.0 * min_r_n / max_n:.1f}% of the max across windows."
            )

    near_zero_overlap_warn = ""
    if np.isfinite(np.nanmedian(jvals)) and np.nanmedian(jvals) < 0.1:
        near_zero_overlap_warn = (
            "Stability warning: top20 overlap is near zero across most windows, "
            "so the targets list is unstable."
        )

    outlier_2kpc_note = ""
    row_2 = srt[np.isclose(srt["r_inner_max_kpc"], 2.0, atol=1e-9)]
    if not row_2.empty:
        p2 = float(row_2.iloc[0]["perm_p"])
        idx = row_2.index[0]
        neighbor_idx = [i for i in (idx - 1, idx + 1) if 0 <= i < len(srt)]
        neighbor_p = [float(srt.iloc[i]["perm_p"]) for i in neighbor_idx if np.isfinite(srt.iloc[i]["perm_p"])]
        global_min = float(np.nanmin(pvals)) if np.any(np.isfinite(pvals)) else np.nan
        if neighbor_p and np.isfinite(p2):
            if p2 > 2.0 * np.nanmedian(neighbor_p) and np.isfinite(global_min) and p2 > 2.0 * global_min:
                outlier_2kpc_note = (
                    "r_inner=2.0 kpc appears as an isolated outlier in significance relative to neighboring windows."
                )
            else:
                outlier_2kpc_note = "r_inner=2.0 kpc does not appear as an isolated significance outlier."
    else:
        outlier_2kpc_note = "r_inner=2.0 kpc was not in the sweep grid; outlier check not applicable."

    top10_2k = top20_df[np.isclose(top20_df["r_inner_max_kpc"], 2.0, atol=1e-9)].copy()
    top10_2k = top10_2k.sort_values("rho_score", ascending=False).head(10)
    top10_best = top20_df[
        np.isclose(top20_df["r_inner_max_kpc"], float(best_p_row["r_inner_max_kpc"]), atol=1e-9)
    ].copy()
    top10_best = top10_best.sort_values("rho_score", ascending=False).head(10)

    lines: List[str] = []
    lines.append("# Paper 3 r_inner window sweep report")
    lines.append("")
    lines.append(f"Input file: `{input_file}`")
    lines.append("")
    lines.append(
        "Window dependence is expected because the density proxy is built from inner-radius points, "
        "so changing `r_inner_max_kpc` changes which radii dominate `rho_score`; this can move galaxies "
        "between top/bottom density ranks even if their full-profile residual medians are unchanged."
    )
    lines.append("")
    lines.append(
        f"Strongest separation by significance occurs at `r_inner*={float(best_p_row['r_inner_max_kpc']):.2f} kpc` "
        f"with `perm_p={float(best_p_row['perm_p']):.3g}`, "
        f"`delta_median={float(best_p_row['delta_median']):.4f}`, "
        f"`Jaccard={float(best_p_row['top20_jaccard_vs_2kpc']):.3f}`."
    )
    lines.append(
        f"Peak effect size (|delta_median|) occurs at `r_inner={float(best_effect_row['r_inner_max_kpc']):.2f} kpc` "
        f"with `delta_median={float(best_effect_row['delta_median']):.4f}` and "
        f"`perm_p={float(best_effect_row['perm_p']):.3g}`."
    )
    lines.append("")
    lines.append(f"Signal classification: **{classification}**. {class_reason}")
    lines.append("")
    lines.append("Paper 3 guidance:")
    lines.append("- Treat `r_inner_max_kpc` as a hyperparameter (or fit jointly) instead of fixing it a priori.")
    lines.append("- Interpret the sweep as a scale-dependent transition test first, while guarding for window-induced systematics.")
    lines.append("")
    if small_drop_warn:
        lines.append(f"- {small_drop_warn}")
    if near_zero_overlap_warn:
        lines.append(f"- {near_zero_overlap_warn}")
    lines.append(f"- {outlier_2kpc_note}")
    lines.append("")
    lines.append("Top 10 galaxies at r_inner=2.0 kpc")
    if top10_2k.empty:
        lines.append("- [none]")
    else:
        for _, row in top10_2k.iterrows():
            lines.append(
                f"- {row['galaxy']}: rho_score={row['rho_score']:.3e}, "
                f"median_log_resid={row['median_log_resid']:.4f}, N_points={int(row['N_points'])}"
            )
    lines.append("")
    lines.append(f"Top 10 galaxies at r_inner*={float(best_p_row['r_inner_max_kpc']):.2f} kpc")
    if top10_best.empty:
        lines.append("- [none]")
    else:
        for _, row in top10_best.iterrows():
            lines.append(
                f"- {row['galaxy']}: rho_score={row['rho_score']:.3e}, "
                f"median_log_resid={row['median_log_resid']:.4f}, N_points={int(row['N_points'])}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    r_grid = parse_r_inner_grid(args.r_inner_grid)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = repo_root / "outputs" / "paper3_rinner_sweep" / timestamp
    else:
        o = Path(args.out_dir).expanduser()
        out_dir = o if o.is_absolute() else (repo_root / o)
    out_dir.mkdir(parents=True, exist_ok=True)

    points, mapping, input_path = load_and_prepare_points(
        repo_root=repo_root,
        rar_points_file=args.rar_points_file,
        g_dagger=args.g_dagger,
    )

    print(f"[P3R] Input points: {input_path}")
    print(f"[P3R] Rows after filters: {len(points)}")
    print(f"[P3R] r_inner grid ({len(r_grid)}): {r_grid[0]:.2f}..{r_grid[-1]:.2f} step {r_grid[1]-r_grid[0]:.2f}")

    rng = np.random.default_rng(RNG_SEED)
    sweep_rows: List[Dict[str, object]] = []
    top20_rows: List[Dict[str, object]] = []
    top_sets: Dict[float, List[str]] = {}

    for r_inner in r_grid:
        gal = summarize_galaxies(points=points, r_inner_max_kpc=r_inner, min_points=args.min_points)
        eligible = gal[
            (~gal["flag_low_points"])
            & np.isfinite(gal["rho_score"])
            & np.isfinite(gal["median_log_resid"])
        ].copy()
        eligible = eligible.sort_values("rho_score", ascending=False).reset_index(drop=True)

        top = eligible.head(args.top_n).copy()
        bottom = eligible.loc[~eligible["galaxy"].isin(top["galaxy"])].copy()
        bottom = bottom.sort_values("rho_score", ascending=True).head(args.top_n).copy()

        top_vals = top["median_log_resid"].to_numpy(dtype=float)
        bottom_vals = bottom["median_log_resid"].to_numpy(dtype=float)

        delta_median = np.nan
        perm_p = np.nan
        delta_mean = np.nan
        if len(top_vals) >= 2 and len(bottom_vals) >= 2:
            delta_median, perm_p = permutation_test_delta_median(
                top_vals=top_vals, bottom_vals=bottom_vals, n_perm=args.n_perm, rng=rng
            )
            delta_mean = float(np.mean(top_vals) - np.mean(bottom_vals))

        x = np.log10(eligible["rho_score"].to_numpy(dtype=float))
        y = eligible["median_log_resid"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        theilsen_slope = np.nan
        slope_boot_p = np.nan
        if len(x) >= 3 and np.nanstd(x) > 0:
            theilsen_slope, _, _, _ = stats.theilslopes(y, x, alpha=0.95)
            slope_boot_p = bootstrap_slope_pvalue(
                x=x, y=y, n_boot=DEFAULT_N_BOOT, rng=rng
            )

        top_set = top["galaxy"].tolist()
        top_sets[float(r_inner)] = top_set

        for _, row in top.iterrows():
            top20_rows.append(
                {
                    "r_inner_max_kpc": float(r_inner),
                    "galaxy": row["galaxy"],
                    "rho_score": float(row["rho_score"]),
                    "median_log_resid": float(row["median_log_resid"]),
                    "N_points": int(row["N_points"]),
                    "flag_low_points": bool(row["flag_low_points"]),
                }
            )

        sweep_rows.append(
            {
                "r_inner_max_kpc": float(r_inner),
                "n_gal_eligible": int(len(eligible)),
                "delta_median": float(delta_median) if np.isfinite(delta_median) else np.nan,
                "perm_p": float(perm_p) if np.isfinite(perm_p) else np.nan,
                "theilsen_slope": float(theilsen_slope) if np.isfinite(theilsen_slope) else np.nan,
                "slope_boot_p": float(slope_boot_p) if np.isfinite(slope_boot_p) else np.nan,
                "top20_jaccard_vs_2kpc": np.nan,  # filled after baseline is known
                "top_n_used": int(len(top)),
                "bottom_n_used": int(len(bottom)),
                "top20_median_Npoints": float(np.median(top["N_points"])) if len(top) else np.nan,
                "bottom20_median_Npoints": float(np.median(bottom["N_points"])) if len(bottom) else np.nan,
                "delta_mean": float(delta_mean) if np.isfinite(delta_mean) else np.nan,
                "n_flagged_low_points_total": int(gal["flag_low_points"].sum()),
                "n_rho_fallback_max": int(gal["used_rho_fallback_max"].sum()),
            }
        )

    summary = pd.DataFrame(sweep_rows).sort_values("r_inner_max_kpc").reset_index(drop=True)
    top20 = pd.DataFrame(top20_rows)

    # Jaccard baseline anchored at r_inner=2.0 kpc.
    if np.any(np.isclose(np.array(r_grid), 2.0, atol=1e-9)):
        baseline_key = float(np.array(r_grid)[np.argmin(np.abs(np.array(r_grid) - 2.0))])
        baseline_set = top_sets.get(baseline_key, [])
    else:
        gal2 = summarize_galaxies(points=points, r_inner_max_kpc=2.0, min_points=args.min_points)
        eligible2 = gal2[
            (~gal2["flag_low_points"])
            & np.isfinite(gal2["rho_score"])
            & np.isfinite(gal2["median_log_resid"])
        ].copy()
        eligible2 = eligible2.sort_values("rho_score", ascending=False).reset_index(drop=True)
        baseline_set = eligible2.head(args.top_n)["galaxy"].tolist()

    jacc = [
        jaccard_index(top_sets.get(float(r), []), baseline_set)
        for r in summary["r_inner_max_kpc"].to_numpy(dtype=float)
    ]
    summary["top20_jaccard_vs_2kpc"] = jacc

    # Keep requested summary columns first (extras omitted in final CSV).
    summary_out = summary[
        [
            "r_inner_max_kpc",
            "n_gal_eligible",
            "delta_median",
            "perm_p",
            "theilsen_slope",
            "slope_boot_p",
            "top20_jaccard_vs_2kpc",
            "top_n_used",
            "bottom_n_used",
            "top20_median_Npoints",
            "bottom20_median_Npoints",
        ]
    ].copy()

    # Identify best windows.
    finite_p = summary_out[np.isfinite(summary_out["perm_p"])].copy()
    if finite_p.empty:
        best_p_row = summary_out.iloc[0]
    else:
        min_p = float(finite_p["perm_p"].min())
        tied = finite_p[np.isclose(finite_p["perm_p"], min_p)]
        best_p_row = tied.loc[tied["delta_median"].abs().idxmax()]

    finite_delta = summary_out[np.isfinite(summary_out["delta_median"])].copy()
    if finite_delta.empty:
        best_effect_row = summary_out.iloc[0]
    else:
        best_effect_row = finite_delta.loc[finite_delta["delta_median"].abs().idxmax()]

    classification, class_reason = classify_robustness(summary_out)

    summary_csv = out_dir / "rinner_sweep_summary.csv"
    top20_csv = out_dir / "rinner_sweep_targets_top20.csv"
    plot_png = out_dir / "rinner_sweep_plots.png"
    report_md = out_dir / "rinner_sweep_report.md"

    summary_out.to_csv(summary_csv, index=False)
    top20[
        ["r_inner_max_kpc", "galaxy", "rho_score", "median_log_resid", "N_points", "flag_low_points"]
    ].to_csv(top20_csv, index=False)
    save_plots(plot_png, summary_out)
    build_report(
        out_path=report_md,
        input_file=input_path,
        summary_df=summary_out,
        top20_df=top20,
        best_p_row=best_p_row,
        best_effect_row=best_effect_row,
        classification=classification,
        class_reason=class_reason,
    )

    print("[P3R] Output folder:", out_dir)
    print(
        "[P3R] Best perm_p window: "
        f"r_inner*={float(best_p_row['r_inner_max_kpc']):.2f} kpc, "
        f"perm_p={float(best_p_row['perm_p']):.4g}, "
        f"delta_median={float(best_p_row['delta_median']):.4f}, "
        f"Jaccard={float(best_p_row['top20_jaccard_vs_2kpc']):.3f}"
    )
    print(f"[P3R] Classification: {classification}")
    print("[P3R] Files:")
    print(f"  {summary_csv}")
    print(f"  {top20_csv}")
    print(f"  {plot_png}")
    print(f"  {report_md}")


if __name__ == "__main__":
    main()
