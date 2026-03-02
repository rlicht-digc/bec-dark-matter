#!/usr/bin/env python3
"""
Paper 3 asymmetry test:
Does a dense↔dilute two-minima potential predict V0 ~ rhoLambda without extra tuning?
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RHO_LAMBDA_J_M3 = 6.0e-10
DEFAULT_M_LIST = "1e-22,1e-21,1e-20,1e-19"
DEFAULT_A_S_LIST_M = "1e-36,1e-30,1e-24,1e-18,1e-15"
DEFAULT_V_MIN_EV = 1.0e10
DEFAULT_V_MAX_EV = 1.0e13
DEFAULT_V_NPTS = 40
DEFAULT_MAP = "real_scalar"
REFERENCE_V0_QUARTIC_MEV = 2.316061

# Unit conversion constants.
JOULE_PER_EV = 1.602176634e-19
HBAR_C_EV_M = 1.973269804e-7  # eV * m
J_M3_TO_EV4 = (1.0 / JOULE_PER_EV) * (HBAR_C_EV_M ** 3)


def j_m3_to_ev4(val: float) -> float:
    return float(val) * J_M3_TO_EV4


def parse_list(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Parsed empty list from input string")
    return vals


def lambda4_from_map(m_eV: float, a_s_m: float, map_name: str) -> float:
    # Convert scattering length to eV^-1 using 1 eV^-1 = hbar*c [m].
    a_s_eV_inv = a_s_m / HBAR_C_EV_M
    if map_name == "real_scalar":
        # a_s ~= lambda4 / (32 pi m)
        return 32.0 * np.pi * m_eV * a_s_eV_inv
    if map_name == "complex_scalar":
        # a_s ~= lambda4 / (8 pi m)
        return 8.0 * np.pi * m_eV * a_s_eV_inv
    raise ValueError(f"Unsupported lambda4_map: {map_name}")


def check_structure_and_compute(
    m_eV: float,
    a_s_m: float,
    v_eV: float,
    lambda4: float,
    V0_obs_eV4: float,
) -> Dict[str, object]:
    """
    Build potential:
      V(phi) = V0 + 1/2 m^2 phi^2 - (lambda4/4) phi^4 + (lambda6/6) phi^6
    with lambda6 fixed by requiring V'(v)=0.
    """
    row: Dict[str, object] = {}
    row["m_eV"] = m_eV
    row["a_s_m"] = a_s_m
    row["v_eV"] = v_eV
    row["lambda4"] = lambda4
    row["V0_obs_eV4"] = V0_obs_eV4

    reasons: List[str] = []
    valid = True

    m2 = m_eV ** 2
    if lambda4 <= 0.0:
        valid = False
        reasons.append("lambda4<=0")

    # Stationary condition at phi=v:
    # m^2 - lambda4 v^2 + lambda6 v^4 = 0.
    lambda6 = (lambda4 * v_eV**2 - m2) / v_eV**4
    row["lambda6"] = lambda6
    if lambda6 <= 0.0:
        valid = False
        reasons.append("lambda6<=0")

    # Check that phi=v is actually a local minimum (not a maximum/saddle).
    # V''(phi) = m^2 - 3 lambda4 phi^2 + 5 lambda6 phi^4.
    d2_v = m2 - 3.0 * lambda4 * v_eV**2 + 5.0 * lambda6 * v_eV**4
    if d2_v <= 0.0:
        valid = False
        reasons.append("dense_extremum_not_min")

    # Barrier check: in y=phi^2, equation is lambda6 y^2 - lambda4 y + m^2 = 0.
    # Need two positive roots (one is y=v^2), with the smaller root inside (0, v^2).
    if valid:
        roots = np.roots([lambda6, -lambda4, m2])
        y_real = sorted([float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0.0])
        if len(y_real) < 2:
            valid = False
            reasons.append("no_two_positive_extrema")
        else:
            y1, y2 = y_real[0], y_real[-1]
            if not (y1 < y2):
                valid = False
                reasons.append("degenerate_extrema")
            else:
                # Expect larger root near v^2; smaller one should be barrier location.
                if not (abs(y2 - v_eV**2) / max(v_eV**2, 1.0) < 1e-5):
                    # If labeling swapped due to numerics, try the other root check.
                    if abs(y1 - v_eV**2) / max(v_eV**2, 1.0) < 1e-5:
                        y1, y2 = y2, y1
                y_bar = min(y_real)
                if not (0.0 < y_bar < v_eV**2):
                    valid = False
                    reasons.append("no_barrier_between_minima")
                else:
                    phi_bar = np.sqrt(y_bar)
                    d2_bar = m2 - 3.0 * lambda4 * phi_bar**2 + 5.0 * lambda6 * phi_bar**4
                    if d2_bar >= 0.0:
                        valid = False
                        reasons.append("barrier_not_maximum")

    # Predicted V0 from convention V(v_dense)=0.
    V_without_V0_at_v = (
        0.5 * m2 * v_eV**2
        - (lambda4 / 4.0) * v_eV**4
        + (lambda6 / 6.0) * v_eV**6
    )
    V0_pred = -V_without_V0_at_v
    row["V0_pred_eV4"] = V0_pred

    ratio = V0_pred / V0_obs_eV4
    if ratio > 0:
        log10_ratio = np.log10(ratio)
    else:
        log10_ratio = np.nan
        reasons.append("V0_pred<=0")
    row["log10(V0_pred/V0_obs)"] = log10_ratio

    if abs(V0_pred) > 0:
        log10_tuning_dense = np.log10(abs(V0_pred) / (v_eV**4))
    else:
        log10_tuning_dense = -np.inf
    row["log10(|V0_pred|/v^4)"] = log10_tuning_dense

    row["valid_flag"] = bool(valid)
    row["reason"] = ";".join(sorted(set(reasons))) if reasons else "ok"
    return row


def make_plots(out_png: Path, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax1, ax2, ax3 = axes

    valid = df[df["valid_flag"]].copy()
    finite_ratio = valid[np.isfinite(valid["log10(V0_pred/V0_obs)"])].copy()

    # Panel A: log10 ratio vs v with slices by (m, a_s).
    if not finite_ratio.empty:
        for (m_eV, a_s_m), g in finite_ratio.groupby(["m_eV", "a_s_m"], sort=False):
            ax1.plot(
                g["v_eV"].to_numpy(dtype=float),
                g["log10(V0_pred/V0_obs)"].to_numpy(dtype=float),
                alpha=0.45,
                lw=1.1,
            )
        ax1.scatter(
            finite_ratio["v_eV"].to_numpy(dtype=float),
            finite_ratio["log10(V0_pred/V0_obs)"].to_numpy(dtype=float),
            c=np.log10(finite_ratio["a_s_m"].to_numpy(dtype=float)),
            cmap="viridis",
            s=14,
            alpha=0.85,
        )
    else:
        ax1.text(0.5, 0.5, "No finite positive-ratio points", transform=ax1.transAxes, ha="center")
    for y in [0, 1, -1, 2, -2, 3, -3]:
        ax1.axhline(y, color="gray", lw=0.8 if y != 0 else 1.1, ls="--" if y != 0 else "-")
    ax1.set_xscale("log")
    ax1.set_xlabel("v_eV")
    ax1.set_ylabel("log10(V0_pred / V0_obs)")
    ax1.set_title("Panel A: Asymmetry prediction vs dense scale")
    ax1.grid(True, which="both", alpha=0.3, linestyle=":")

    # Panel B: dense tuning vs v.
    if not valid.empty:
        ax2.scatter(
            valid["v_eV"].to_numpy(dtype=float),
            valid["log10(|V0_pred|/v^4)"].to_numpy(dtype=float),
            c=np.log10(valid["m_eV"].to_numpy(dtype=float)),
            cmap="plasma",
            s=14,
            alpha=0.8,
        )
    else:
        ax2.text(0.5, 0.5, "No valid points", transform=ax2.transAxes, ha="center")
    ax2.set_xscale("log")
    ax2.set_xlabel("v_eV")
    ax2.set_ylabel("log10(|V0_pred|/v^4)")
    ax2.set_title("Panel B: EFT tuning proxy")
    ax2.grid(True, which="both", alpha=0.3, linestyle=":")

    # Panel C: mark points within 3 dex.
    if not finite_ratio.empty:
        within3 = np.abs(finite_ratio["log10(V0_pred/V0_obs)"].to_numpy(dtype=float)) <= 3.0
        ax3.scatter(
            finite_ratio.loc[~within3, "v_eV"].to_numpy(dtype=float),
            finite_ratio.loc[~within3, "log10(V0_pred/V0_obs)"].to_numpy(dtype=float),
            s=12,
            color="#94a3b8",
            alpha=0.6,
            label="outside 3 dex",
        )
        ax3.scatter(
            finite_ratio.loc[within3, "v_eV"].to_numpy(dtype=float),
            finite_ratio.loc[within3, "log10(V0_pred/V0_obs)"].to_numpy(dtype=float),
            s=24,
            color="#16a34a",
            alpha=0.9,
            label="within 3 dex",
        )
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No finite positive-ratio points", transform=ax3.transAxes, ha="center")
    ax3.axhline(0.0, color="gray", lw=1.0)
    ax3.axhline(3.0, color="gray", lw=0.8, ls="--")
    ax3.axhline(-3.0, color="gray", lw=0.8, ls="--")
    ax3.set_xscale("log")
    ax3.set_xlabel("v_eV")
    ax3.set_ylabel("log10(V0_pred / V0_obs)")
    ax3.set_title("Panel C: Near-Λ matches")
    ax3.grid(True, which="both", alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_report(
    out_md: Path,
    df: pd.DataFrame,
    rhoLambda_J_m3: float,
    V0_obs_eV4: float,
    V0_obs_quartic_meV: float,
    lambda4_map: str,
) -> Tuple[int, int, int, int, pd.Series]:
    valid = df[df["valid_flag"]].copy()
    finite_ratio = valid[np.isfinite(valid["log10(V0_pred/V0_obs)"])].copy()

    n_valid = int(len(valid))
    within1 = int(np.sum(np.abs(finite_ratio["log10(V0_pred/V0_obs)"].to_numpy(dtype=float)) <= 1.0))
    within2 = int(np.sum(np.abs(finite_ratio["log10(V0_pred/V0_obs)"].to_numpy(dtype=float)) <= 2.0))
    within3 = int(np.sum(np.abs(finite_ratio["log10(V0_pred/V0_obs)"].to_numpy(dtype=float)) <= 3.0))

    if finite_ratio.empty:
        best = df.iloc[0]
    else:
        idx = np.abs(finite_ratio["log10(V0_pred/V0_obs)"].to_numpy(dtype=float)).argmin()
        best = finite_ratio.iloc[int(idx)]

    lines: List[str] = []
    lines.append("# Potential Asymmetry Test (Paper 3)")
    lines.append("")
    lines.append(
        "Contact interaction energy alone has w>=0 and cannot source cosmic acceleration; "
        "this test instead checks whether vacuum offset from potential asymmetry can land near rhoLambda."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("- Potential: V(phi)=V0+1/2 m^2 phi^2-(lambda4/4)phi^4+(lambda6/6)phi^6")
    lines.append("- Convention tested: set dense minimum vacuum to zero, V(v_dense)=0")
    lines.append("- Prediction: V(0)=V0_pred is the dilute-phase vacuum offset")
    lines.append(f"- lambda4 mapping convention: `{lambda4_map}`")
    lines.append("")
    lines.append("## Observed vacuum reference")
    lines.append(f"- rhoLambda_J_m3 = {rhoLambda_J_m3:.3e}")
    lines.append(f"- V0_obs_eV4 = {V0_obs_eV4:.6e}")
    lines.append(
        f"- V0_obs^(1/4) = {V0_obs_quartic_meV:.6f} meV (cross-check target {REFERENCE_V0_QUARTIC_MEV:.6f} meV)"
    )
    lines.append("")
    lines.append("## Scan outcome")
    lines.append(f"- total rows = {len(df)}")
    lines.append(f"- valid structural two-minima rows = {n_valid}")
    lines.append(f"- rows within 1 dex of Lambda = {within1}")
    lines.append(f"- rows within 2 dex of Lambda = {within2}")
    lines.append(f"- rows within 3 dex of Lambda = {within3}")
    lines.append("")
    lines.append("## Best-match row (minimum |log10(V0_pred/V0_obs)|)")
    lines.append(
        f"- m_eV={best['m_eV']:.3e}, a_s_m={best['a_s_m']:.3e}, v_eV={best['v_eV']:.3e}, "
        f"lambda4={best['lambda4']:.3e}, lambda6={best['lambda6']:.3e}"
    )
    lines.append(
        f"- V0_pred_eV4={best['V0_pred_eV4']:.6e}, "
        f"log10(V0_pred/V0_obs)={best['log10(V0_pred/V0_obs)']:.6f}, "
        f"log10(|V0_pred|/v^4)={best['log10(|V0_pred|/v^4)']:.6f}"
    )
    lines.append("")
    lines.append(
        "This is a consistency/tuning test in one concrete potential family, not a cosmological-constant solution."
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_valid, within1, within2, within3, best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Potential asymmetry scan for Paper 3 cosmology module.")
    p.add_argument("--rhoLambda_J_m3", type=float, default=DEFAULT_RHO_LAMBDA_J_M3)
    p.add_argument("--lambda4_map", type=str, choices=["real_scalar", "complex_scalar"], default=DEFAULT_MAP)
    p.add_argument("--m_eV_list", type=str, default=DEFAULT_M_LIST)
    p.add_argument("--a_s_list_m", type=str, default=DEFAULT_A_S_LIST_M)
    p.add_argument("--v_min_eV", type=float, default=DEFAULT_V_MIN_EV)
    p.add_argument("--v_max_eV", type=float, default=DEFAULT_V_MAX_EV)
    p.add_argument("--v_npts", type=int, default=DEFAULT_V_NPTS)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rhoLambda_eV4 = j_m3_to_ev4(args.rhoLambda_J_m3)
    V0_obs_quartic_meV = (rhoLambda_eV4 ** 0.25) * 1e3

    m_list = parse_list(args.m_eV_list)
    a_s_list = parse_list(args.a_s_list_m)
    if args.v_min_eV <= 0 or args.v_max_eV <= 0 or args.v_max_eV <= args.v_min_eV:
        raise ValueError("Require 0 < v_min_eV < v_max_eV")
    if args.v_npts < 2:
        raise ValueError("--v_npts must be >= 2")
    v_grid = np.logspace(np.log10(args.v_min_eV), np.log10(args.v_max_eV), args.v_npts)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = REPO_ROOT / "outputs" / "paper3_cosmology_asymmetry" / timestamp
    else:
        o = Path(args.out_dir).expanduser()
        out_dir = o if o.is_absolute() else (REPO_ROOT / o)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for m_eV in m_list:
        for a_s_m in a_s_list:
            lambda4 = lambda4_from_map(m_eV=m_eV, a_s_m=a_s_m, map_name=args.lambda4_map)
            for v_eV in v_grid:
                res = check_structure_and_compute(
                    m_eV=m_eV,
                    a_s_m=a_s_m,
                    v_eV=float(v_eV),
                    lambda4=float(lambda4),
                    V0_obs_eV4=float(rhoLambda_eV4),
                )
                res["lambda4_map"] = args.lambda4_map
                rows.append(res)

    df = pd.DataFrame(rows)
    # Order columns exactly as requested.
    df = df[
        [
            "lambda4_map",
            "m_eV",
            "a_s_m",
            "v_eV",
            "lambda4",
            "lambda6",
            "V0_pred_eV4",
            "V0_obs_eV4",
            "log10(V0_pred/V0_obs)",
            "log10(|V0_pred|/v^4)",
            "valid_flag",
            "reason",
        ]
    ]

    csv_path = out_dir / "asymmetry_scan.csv"
    report_path = out_dir / "asymmetry_report.md"
    plot_path = out_dir / "asymmetry_plots.png"

    df.to_csv(csv_path, index=False)
    make_plots(plot_path, df)
    n_valid, within1, within2, within3, best = build_report(
        out_md=report_path,
        df=df,
        rhoLambda_J_m3=args.rhoLambda_J_m3,
        V0_obs_eV4=rhoLambda_eV4,
        V0_obs_quartic_meV=V0_obs_quartic_meV,
        lambda4_map=args.lambda4_map,
    )

    print(f"[P3A] V0_obs^(1/4) = {V0_obs_quartic_meV:.6f} meV")
    print(f"[P3A] valid_structural_points = {n_valid}")
    print(f"[P3A] within_1_dex = {within1}")
    print(f"[P3A] within_2_dex = {within2}")
    print(f"[P3A] within_3_dex = {within3}")
    print(
        "[P3A] best_match: "
        f"m_eV={best['m_eV']:.3e}, a_s_m={best['a_s_m']:.3e}, v_eV={best['v_eV']:.3e}, "
        f"log10_ratio={best['log10(V0_pred/V0_obs)']:.6f}"
    )
    print(f"[P3A] output_folder = {out_dir}")
    print(f"[P3A] files:\n  {csv_path}\n  {report_path}\n  {plot_path}")


if __name__ == "__main__":
    main()
