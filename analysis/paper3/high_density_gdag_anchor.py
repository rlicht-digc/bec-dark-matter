#!/usr/bin/env python3
"""
Paper 3: g_dagger-anchored density proxy for residual-separation diagnostics.

This removes r_inner window hyperparameter dependence by anchoring each galaxy
to the local region where g_bar is nearest to g_dagger.
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
DEFAULT_N_BOOT = 2000
EPS = np.finfo(float).tiny


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


def _check_required_fields(mapping: Dict[str, Optional[str]]) -> None:
    missing: List[str] = []
    if mapping["galaxy"] is None:
        missing.append("galaxy identifier")
    if mapping["radius"] is None:
        missing.append("r_kpc/radius")

    can_build_gbar = (mapping["gbar_linear"] is not None) or (mapping["gbar_log"] is not None)
    can_build_gobs = (mapping["gobs_linear"] is not None) or (mapping["gobs_log"] is not None)
    can_build_from_vel = (mapping["v_obs"] is not None) and (mapping["v_bar"] is not None)

    if not can_build_gbar and not can_build_from_vel:
        missing.append("g_bar (or v_bar + radius)")
    if not can_build_gobs and not can_build_from_vel:
        missing.append("g_obs (or v_obs + radius)")

    if missing:
        msg = "STOP: missing required per-point fields: " + ", ".join(missing)
        raise RuntimeError(msg)


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
        (points["r_kpc"] > 0.0)
        & (points["g_bar"] > 0.0)
        & (points["g_obs"] > 0.0)
    ].copy()
    if points.empty:
        raise RuntimeError("STOP: no usable rows after basic quality filters")

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
    points = points.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log_resid", "rho_dyn"]
    )
    if points.empty:
        raise RuntimeError("STOP: no usable rows after residual/rho calculations")

    return points, input_path


def summarize_per_galaxy(
    points: pd.DataFrame,
    g_dagger: float,
    min_points: int,
    window_dex: float,
    min_window_points: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    log_gdag = np.log10(g_dagger)

    for galaxy, gdf in points.groupby("galaxy", sort=False):
        n_points = int(len(gdf))
        gbar_log = np.log10(gdf["g_bar"].to_numpy(dtype=float))
        abs_delta = np.abs(gbar_log - log_gdag)

        idx_local = int(np.argmin(abs_delta))
        g_closest = gdf.iloc[idx_local]

        r_gdag_kpc = float(g_closest["r_kpc"])
        rho_gdag = float(g_closest["rho_dyn"])
        resid_gdag = float(g_closest["log_resid"])

        in_window = abs_delta <= window_dex
        n_window = int(np.sum(in_window))

        if n_window >= min_window_points:
            rho_score = float(np.median(gdf.loc[in_window, "rho_dyn"].to_numpy(dtype=float)))
            resid_score = float(np.median(gdf.loc[in_window, "log_resid"].to_numpy(dtype=float)))
            flag_low_window = False
        else:
            rho_score = rho_gdag
            resid_score = resid_gdag
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
                "resid_score": resid_score,
                "flag_low_points": bool(n_points < min_points),
                "flag_low_window": bool(flag_low_window),
            }
        )

    return pd.DataFrame(rows)


def permutation_delta_median(
    top_vals: np.ndarray, bottom_vals: np.ndarray, n_perm: int, rng: np.random.Generator
) -> Tuple[float, float]:
    # Uses per-galaxy medians as required (no pooled point-level weighting).
    observed = float(np.median(top_vals) - np.median(bottom_vals))
    combined = np.concatenate([top_vals, bottom_vals])
    n_top = len(top_vals)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = float(np.median(perm[:n_top]) - np.median(perm[n_top:]))
        if abs(diff) >= abs(observed):
            extreme += 1
    p_val = (extreme + 1) / (n_perm + 1)
    return observed, float(p_val)


def bootstrap_theilsen_pvalue(
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


def save_plot(
    out_path: Path,
    eligible: pd.DataFrame,
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    slope: float,
    intercept: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    e = eligible.copy()
    top_names = set(top["galaxy"].tolist())
    bottom_names = set(bottom["galaxy"].tolist())

    x_all = np.log10(e["rho_score"].to_numpy(dtype=float))
    y_all = e["resid_score"].to_numpy(dtype=float)
    ax1.scatter(x_all, y_all, s=20, alpha=0.45, color="#475569", label="eligible")
    if len(top):
        xt = np.log10(top["rho_score"].to_numpy(dtype=float))
        yt = top["resid_score"].to_numpy(dtype=float)
        ax1.scatter(xt, yt, s=38, alpha=0.9, color="#b91c1c", label="top20")
    if len(bottom):
        xb = np.log10(bottom["rho_score"].to_numpy(dtype=float))
        yb = bottom["resid_score"].to_numpy(dtype=float)
        ax1.scatter(xb, yb, s=38, alpha=0.9, color="#1d4ed8", label="bottom20")
    if np.isfinite(slope) and np.isfinite(intercept) and len(e) > 0:
        xg = np.linspace(np.nanmin(x_all), np.nanmax(x_all), 200)
        yg = intercept + slope * xg
        ax1.plot(xg, yg, color="black", lw=2.0, label="Theil-Sen")
    ax1.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax1.set_xlabel("log10(rho_score)")
    ax1.set_ylabel("resid_score (dex)")
    ax1.set_title("Panel A: log10(rho_score) vs resid_score")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3, linestyle=":")

    top_res = top["resid_score"].to_numpy(dtype=float)
    bot_res = bottom["resid_score"].to_numpy(dtype=float)
    if len(top_res) and len(bot_res):
        bins = np.linspace(
            min(np.nanmin(top_res), np.nanmin(bot_res)),
            max(np.nanmax(top_res), np.nanmax(bot_res)),
            20,
        )
        ax2.hist(top_res, bins=bins, alpha=0.55, density=True, label="top20", color="#b91c1c")
        ax2.hist(bot_res, bins=bins, alpha=0.55, density=True, label="bottom20", color="#1d4ed8")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Insufficient groups", transform=ax2.transAxes, ha="center")
    ax2.set_xlabel("resid_score (dex)")
    ax2.set_ylabel("density")
    ax2.set_title("Panel B: resid_score top20 vs bottom20")
    ax2.grid(True, alpha=0.3, linestyle=":")

    top_r = top["r_gdag_kpc"].to_numpy(dtype=float)
    bot_r = bottom["r_gdag_kpc"].to_numpy(dtype=float)
    if len(top_r) and len(bot_r):
        bins_r = np.linspace(
            min(np.nanmin(top_r), np.nanmin(bot_r)),
            max(np.nanmax(top_r), np.nanmax(bot_r)),
            20,
        )
        ax3.hist(top_r, bins=bins_r, alpha=0.55, density=True, label="top20", color="#b91c1c")
        ax3.hist(bot_r, bins=bins_r, alpha=0.55, density=True, label="bottom20", color="#1d4ed8")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Insufficient groups", transform=ax3.transAxes, ha="center")
    ax3.set_xlabel("r_gdag_kpc")
    ax3.set_ylabel("density")
    ax3.set_title("Panel C: r_gdag_kpc top20 vs bottom20")
    ax3.grid(True, alpha=0.3, linestyle=":")

    colors = []
    for gal in e["galaxy"]:
        if gal in top_names:
            colors.append("#b91c1c")
        elif gal in bottom_names:
            colors.append("#1d4ed8")
        else:
            colors.append("#475569")
    ax4.scatter(
        e["rho_score"].to_numpy(dtype=float),
        e["N_points"].to_numpy(dtype=float),
        s=22,
        alpha=0.75,
        c=colors,
    )
    ax4.set_xscale("log")
    ax4.set_xlabel("rho_score")
    ax4.set_ylabel("N_points")
    ax4.set_title("Panel D: rho_score vs N_points")
    ax4.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_report(
    out_path: Path,
    input_file: Path,
    summary: pd.DataFrame,
    eligible: pd.DataFrame,
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    delta_median: float,
    perm_p: float,
    slope: float,
    slope_boot_p: float,
    robust: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Paper 3 g_dagger-anchored density report")
    lines.append("")
    lines.append(f"Input: `{input_file}`")
    lines.append("")
    lines.append("Definitions")
    lines.append("- `r_gdag`: radius where `|log10(g_bar)-log10(g_dagger)|` is minimized for each galaxy.")
    lines.append(
        "- `rho_score`: median `rho_dyn` over points with "
        "`|log10(g_bar)-log10(g_dagger)| <= window_dex`; fallback to single closest point if window is sparse."
    )
    lines.append("- `resid_score`: median `log_resid` over the same window (or single-point fallback).")
    lines.append("")
    lines.append("Core statistics")
    lines.append(f"- n_gal_total: {len(summary)}")
    lines.append(f"- n_gal_eligible (N_points >= threshold): {len(eligible)}")
    lines.append(f"- top_n_used: {len(top)}")
    lines.append(f"- bottom_n_used: {len(bottom)}")
    lines.append(f"- delta_median (top-bottom): {delta_median:.6f} dex")
    lines.append(f"- permutation p-value (10k shuffles): {perm_p:.6g}")
    lines.append(f"- Theil-Sen slope (resid_score vs log10(rho_score)): {slope:.6f}")
    lines.append(f"- slope bootstrap p-value: {slope_boot_p:.6g}")
    lines.append("")
    if robust:
        lines.append(
            "Conclusion: **robust enough (provisionally)** to treat as a `rho_c` lever under this dataset, "
            "because the top-bottom separation passes permutation significance."
        )
    else:
        lines.append(
            "Conclusion: **not robust**; with current SPARC-level lever we do not constrain `rho_c` from this test."
        )
    lines.append("")
    lines.append("Top 10 targets by rho_score")
    top10 = top.sort_values("rho_score", ascending=False).head(10)
    if top10.empty:
        lines.append("- [none]")
    else:
        for _, row in top10.iterrows():
            lines.append(
                f"- {row['galaxy']}: rho_score={row['rho_score']:.3e}, "
                f"r_gdag_kpc={row['r_gdag_kpc']:.3f}, resid_score={row['resid_score']:.4f}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="g_dagger-anchored high-density target diagnostic."
    )
    parser.add_argument("--rar_points_file", type=str, default=DEFAULT_RAR_POINTS)
    parser.add_argument("--g_dagger", type=float, default=DEFAULT_G_DAGGER)
    parser.add_argument("--min_points", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--window_dex", type=float, default=0.2)
    parser.add_argument("--min_window_points", type=int, default=3)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = repo_root / "outputs" / "paper3_gdag_anchor" / timestamp
    else:
        user_out = Path(args.out_dir).expanduser()
        out_dir = user_out if user_out.is_absolute() else (repo_root / user_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    points, input_file = load_points(
        repo_root=repo_root,
        rar_points_file=args.rar_points_file,
        g_dagger=args.g_dagger,
    )

    summary = summarize_per_galaxy(
        points=points,
        g_dagger=args.g_dagger,
        min_points=args.min_points,
        window_dex=args.window_dex,
        min_window_points=args.min_window_points,
    )
    summary = summary.sort_values("galaxy").reset_index(drop=True)

    eligible = summary[
        (~summary["flag_low_points"])
        & np.isfinite(summary["rho_score"])
        & np.isfinite(summary["resid_score"])
    ].copy()
    eligible = eligible.sort_values("rho_score", ascending=False).reset_index(drop=True)

    top = eligible.head(args.top_n).copy()
    bottom_pool = eligible.loc[~eligible["galaxy"].isin(top["galaxy"])].copy()
    bottom = bottom_pool.sort_values("rho_score", ascending=True).head(args.top_n).copy()

    top_vals = top["resid_score"].to_numpy(dtype=float)
    bottom_vals = bottom["resid_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(RNG_SEED)

    delta_median = np.nan
    perm_p = np.nan
    if len(top_vals) >= 2 and len(bottom_vals) >= 2:
        delta_median, perm_p = permutation_delta_median(
            top_vals=top_vals, bottom_vals=bottom_vals, n_perm=args.n_perm, rng=rng
        )

    x = np.log10(eligible["rho_score"].to_numpy(dtype=float))
    y = eligible["resid_score"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    slope = np.nan
    intercept = np.nan
    slope_boot_p = np.nan
    if len(x) >= 3 and np.nanstd(x) > 0:
        slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
        slope_boot_p = bootstrap_theilsen_pvalue(
            x=x, y=y, n_boot=DEFAULT_N_BOOT, rng=rng
        )

    robust = bool(np.isfinite(perm_p) and perm_p < 0.05)

    top_bottom = pd.concat(
        [
            top.assign(group="top20"),
            bottom.assign(group="bottom20"),
        ],
        ignore_index=True,
    )

    summary_csv = out_dir / "gdag_anchor_summary.csv"
    top20_csv = out_dir / "gdag_anchor_top20.csv"
    plot_png = out_dir / "gdag_anchor_plots.png"
    report_md = out_dir / "gdag_anchor_report.md"

    summary[
        [
            "galaxy",
            "N_points",
            "r_gdag_kpc",
            "rho_gdag",
            "resid_gdag",
            "n_window",
            "rho_score",
            "resid_score",
            "flag_low_points",
            "flag_low_window",
        ]
    ].to_csv(summary_csv, index=False)

    top_bottom[
        ["group", "galaxy", "rho_score", "resid_score", "r_gdag_kpc", "n_window", "N_points"]
    ].to_csv(top20_csv, index=False)

    save_plot(
        out_path=plot_png,
        eligible=eligible,
        top=top,
        bottom=bottom,
        slope=float(slope) if np.isfinite(slope) else np.nan,
        intercept=float(intercept) if np.isfinite(intercept) else np.nan,
    )
    build_report(
        out_path=report_md,
        input_file=input_file,
        summary=summary,
        eligible=eligible,
        top=top,
        bottom=bottom,
        delta_median=float(delta_median) if np.isfinite(delta_median) else np.nan,
        perm_p=float(perm_p) if np.isfinite(perm_p) else np.nan,
        slope=float(slope) if np.isfinite(slope) else np.nan,
        slope_boot_p=float(slope_boot_p) if np.isfinite(slope_boot_p) else np.nan,
        robust=robust,
    )

    print(f"[P3G] Input file: {input_file}")
    print(f"[P3G] n_points_used: {len(points)}")
    print(f"[P3G] n_galaxies_summary: {len(summary)}")
    print(f"[P3G] n_gal_eligible: {len(eligible)}")
    print(
        "[P3G] Separation: "
        f"delta_median={float(delta_median):.6f} dex, perm_p={float(perm_p):.6g}"
    )
    print(f"[P3G] Robust lever: {robust}")
    print(f"[P3G] Output folder: {out_dir}")
    print("[P3G] Files:")
    print(f"  {summary_csv}")
    print(f"  {top20_csv}")
    print(f"  {plot_png}")
    print(f"  {report_md}")


if __name__ == "__main__":
    main()
