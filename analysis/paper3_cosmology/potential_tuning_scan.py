#!/usr/bin/env python3
"""
Paper 3 cosmology tuning scan:
two-phase potential + vacuum-offset naturalness bookkeeping.
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

JOULE_PER_EV = 1.602176634e-19
HBAR_C_EV_M = 1.973269804e-7  # eV*m
J_M3_TO_EV4 = (1.0 / JOULE_PER_EV) * (HBAR_C_EV_M ** 3)
EV4_TO_J_M3 = 1.0 / J_M3_TO_EV4

DEFAULT_RHO_LAMBDA_J_M3 = 6.0e-10
DEFAULT_EDENSE_LIST_GEV = "10,100,1000,10000"
DEFAULT_RHOL_OVER_RHODM = 2.5
REFERENCE_V0_QUARTIC_EV = 2.316061e-3

OPTIONAL_DENSE_ANCHOR = Path(
    "/Users/russelllicht/bh-singularity/outputs/alpha_to_xi_dense/20260228_034158/dense_report.md"
)


def j_m3_to_ev4(val: float) -> float:
    return float(val) * J_M3_TO_EV4


def ev4_to_j_m3(val: float) -> float:
    return float(val) * EV4_TO_J_M3


def parse_dense_list(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("--E_dense_list_GeV produced an empty list")
    return vals


def potential_params_for_scale(E_dense_eV: float, v_dense_eV: float) -> Dict[str, float]:
    """
    Construct one illustrative sextic potential instance:

    V(phi) = V0 + a phi^2 + b phi^4 + c phi^6

    using scaled form:
      U(phi) = E^4 [alpha x^2 - beta x^4 + gamma x^6], x = phi/v_dense
    with (alpha,beta,gamma)=(1,3.5,2), giving:
      - local dilute min at x=0,
      - barrier at x=sqrt(1/6),
      - dense min at x=1 with U(1)=-0.5 E^4.
    """
    alpha = 1.0
    beta = 3.5
    gamma = 2.0
    E4 = E_dense_eV ** 4

    a = alpha * E4 / (v_dense_eV ** 2)
    b = -beta * E4 / (v_dense_eV ** 4)
    c = gamma * E4 / (v_dense_eV ** 6)
    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


def evaluate_potential_features(
    V0_ev4: float,
    E_dense_eV: float,
    v_dense_eV: float,
    coeffs: Dict[str, float],
) -> Dict[str, float]:
    a = coeffs["a"]
    b = coeffs["b"]
    c = coeffs["c"]

    def V(phi: float) -> float:
        return V0_ev4 + a * phi**2 + b * phi**4 + c * phi**6

    V0_val = V(0.0)
    V_dense = V(v_dense_eV)

    # In scaled coordinates, barrier is at x = sqrt(1/6).
    x_bar = np.sqrt(1.0 / 6.0)
    phi_bar = x_bar * v_dense_eV
    V_bar = V(phi_bar)

    barrier_above_dilute = V_bar - V0_val
    dense_depth = abs(V_dense - V0_val)

    return {
        "V_at_zero": V0_val,
        "V_at_dense": V_dense,
        "V_at_barrier": V_bar,
        "barrier_height_eV4": barrier_above_dilute,
        "dense_depth_eV4": dense_depth,
        "phi_barrier_eV": phi_bar,
        "v_dense_eV": v_dense_eV,
    }


def make_plots(
    out_png: Path,
    summary: pd.DataFrame,
    V0_quartic_root_eV: float,
    curve_examples: List[Tuple[float, np.ndarray, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax1, ax2, ax3 = axes

    E_dense_GeV = summary["E_dense_GeV"].to_numpy(dtype=float)
    E_dense_eV = E_dense_GeV * 1.0e9
    ratio = E_dense_eV / V0_quartic_root_eV

    ax1.plot(E_dense_GeV, np.full_like(E_dense_GeV, V0_quartic_root_eV), marker="o", lw=1.8, label=r"$V_0^{1/4}$")
    ax1.plot(E_dense_GeV, E_dense_eV, marker="s", lw=1.8, label=r"$E_{\rm dense}$")
    for x, y in zip(E_dense_GeV, ratio):
        ax1.text(x, V0_quartic_root_eV * 1.7, f"{y:.1e}", ha="center", va="bottom", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$E_{\rm dense}$ [GeV]")
    ax1.set_ylabel("Energy scale [eV]")
    ax1.set_title(r"Panel A: meV vacuum vs EW/TeV dense scale")
    ax1.grid(True, which="both", alpha=0.35, linestyle=":")
    ax1.legend(fontsize=8)

    ax2.plot(E_dense_GeV, summary["log10_tuning"].to_numpy(dtype=float), marker="o", lw=1.8, color="#0b6e4f")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$E_{\rm dense}$ [GeV]")
    ax2.set_ylabel(r"$\log_{10}(V_0 / E_{\rm dense}^4)$")
    ax2.set_title("Panel B: tuning vs dense scale")
    ax2.grid(True, which="both", alpha=0.35, linestyle=":")

    for E_geV, xgrid, ygrid in curve_examples:
        ax3.plot(xgrid, ygrid, lw=1.8, label=f"{E_geV:.0f} GeV")
    ax3.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax3.set_xlabel(r"$\phi / v_{\rm dense}$")
    ax3.set_ylabel(r"$(V(\phi)-V_0)/E_{\rm dense}^4$")
    ax3.set_title("Panel C: schematic two-phase potentials")
    ax3.grid(True, alpha=0.35, linestyle=":")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_report(
    out_md: Path,
    rhoLambda_J_m3: float,
    rhoLambda_eV4: float,
    V0_quartic_root_eV: float,
    dense_vals_GeV: List[float],
    summary: pd.DataFrame,
    rhoLambda_over_rhoDM: float,
    anchor_exists: bool,
) -> None:
    log_t_min = float(np.nanmin(summary["log10_tuning"]))
    log_t_max = float(np.nanmax(summary["log10_tuning"]))

    lines: List[str] = []
    lines.append("# Paper 3 Cosmology Tuning Scan")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "This module computes vacuum-offset tuning bookkeeping for a two-phase scalar potential. "
        "It does **not** claim a cosmological-constant solution."
    )
    lines.append("")
    lines.append("## Inputs and conversion checks")
    lines.append(f"- rhoLambda_J_m3 = {rhoLambda_J_m3:.3e}")
    lines.append(f"- rhoLambda_eV4 = {rhoLambda_eV4:.6e}")
    lines.append(f"- V0^(1/4) = {V0_quartic_root_eV:.6e} eV ({V0_quartic_root_eV*1e3:.6f} meV)")
    lines.append(f"- Reference prior value from earlier run = {REFERENCE_V0_QUARTIC_EV:.6e} eV")
    lines.append(
        f"- Consistency delta = {abs(V0_quartic_root_eV - REFERENCE_V0_QUARTIC_EV):.3e} eV"
    )
    lines.append("")
    lines.append("## Dense-phase scan and tuning")
    lines.append(f"- E_dense scan [GeV]: {', '.join(f'{x:g}' for x in dense_vals_GeV)}")
    lines.append(
        f"- Unprotected tuning band from this scan: log10(V0/E_dense^4) in [{log_t_min:.2f}, {log_t_max:.2f}]"
    )
    lines.append(
        "- Numerically, meV vacuum versus EW/TeV-ish dense scales implies severe hierarchy "
        "(roughly 10^-52 to 10^-64 across this broad band; 100 GeV-10 TeV alone sits in the expected extreme range)."
    )
    lines.append("")
    lines.append("## Potential interpretation")
    lines.append(
        "Each E_dense point uses an illustrative sextic potential with dilute and dense minima and an explicit barrier, "
        "showing V0 is tiny compared to dense-phase depth/barrier scales."
    )
    lines.append("")
    lines.append("## Coincidence ratio")
    lines.append(f"- Input ratio rhoLambda/rhoDM_today = {rhoLambda_over_rhoDM:.3f}")
    lines.append(
        "- This script only reports coexistence of scales. Solving coincidence would require a dynamical dark-sector "
        "tracking/coupling model, not implemented here."
    )
    lines.append("")
    lines.append("## Program interface notes")
    if anchor_exists:
        lines.append(
            f"- Read-only context anchor found: `{OPTIONAL_DENSE_ANCHOR}` "
            "(used as qualitative EW/TeV mapping context only)."
        )
    else:
        lines.append(
            f"- Optional anchor not found: `{OPTIONAL_DENSE_ANCHOR}`; scan still valid without it."
        )
    lines.append(
        "- Interface to throat-budget program: dense EW/TeV band motivates the high scale, "
        "while Paper 1/2 phenomenology constrains where dense-phase effects may appear."
    )
    lines.append("")
    lines.append("## What is required to improve beyond bookkeeping")
    lines.append("- Symmetry protection / sequestering mechanism for vacuum offset stability.")
    lines.append("- A dynamical dark-sector model linking rhoDM and rhoLambda to address coincidence.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-phase potential tuning scan for Paper 3 cosmology module.")
    p.add_argument("--rhoLambda_J_m3", type=float, default=DEFAULT_RHO_LAMBDA_J_M3)
    p.add_argument("--E_dense_list_GeV", type=str, default=DEFAULT_EDENSE_LIST_GEV)
    p.add_argument("--rhoLambda_over_rhoDM", type=float, default=DEFAULT_RHOL_OVER_RHODM)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dense_vals_GeV = parse_dense_list(args.E_dense_list_GeV)

    rhoLambda_eV4 = j_m3_to_ev4(args.rhoLambda_J_m3)
    V0_quartic_root_eV = rhoLambda_eV4 ** 0.25

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = REPO_ROOT / "outputs" / "paper3_cosmology_tuning" / timestamp
    else:
        o = Path(args.out_dir).expanduser()
        out_dir = o if o.is_absolute() else (REPO_ROOT / o)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    curve_examples: List[Tuple[float, np.ndarray, np.ndarray]] = []
    example_energies = {dense_vals_GeV[0], dense_vals_GeV[min(2, len(dense_vals_GeV)-1)]}

    V0 = rhoLambda_eV4

    for E_dense_GeV in dense_vals_GeV:
        E_dense_eV = E_dense_GeV * 1.0e9
        V_dense_scale = E_dense_eV ** 4
        tuning = V0 / V_dense_scale
        log10_tuning = np.log10(tuning)

        # Representative potential instance: set v_dense ~ E_dense in eV.
        v_dense_eV = E_dense_eV
        coeffs = potential_params_for_scale(E_dense_eV=E_dense_eV, v_dense_eV=v_dense_eV)
        feats = evaluate_potential_features(
            V0_ev4=V0,
            E_dense_eV=E_dense_eV,
            v_dense_eV=v_dense_eV,
            coeffs=coeffs,
        )

        summary_rows.append(
            {
                "E_dense_GeV": float(E_dense_GeV),
                "V0_quartic_root_eV": float(V0_quartic_root_eV),
                "log10_tuning": float(log10_tuning),
                "tuning": float(tuning),
                "example_v_dense": float(v_dense_eV),
                "example_barrier_eV4": float(feats["barrier_height_eV4"]),
                "example_dense_depth_eV4": float(feats["dense_depth_eV4"]),
                "notes": (
                    f"V(0)=V0, V(v_dense)={feats['V_at_dense']:.3e}, "
                    f"barrier={feats['V_at_barrier']:.3e} eV^4"
                ),
            }
        )

        if E_dense_GeV in example_energies:
            x = np.linspace(0.0, 1.6, 400)
            alpha = coeffs["alpha"]
            beta = coeffs["beta"]
            gamma = coeffs["gamma"]
            y = alpha * x**2 - beta * x**4 + gamma * x**6  # (V-V0)/E^4
            curve_examples.append((E_dense_GeV, x, y))

    summary = pd.DataFrame(summary_rows).sort_values("E_dense_GeV").reset_index(drop=True)

    summary_csv = out_dir / "tuning_scan_summary.csv"
    plot_png = out_dir / "tuning_scan_plots.png"
    report_md = out_dir / "tuning_scan_report.md"

    summary[
        [
            "E_dense_GeV",
            "V0_quartic_root_eV",
            "log10_tuning",
            "tuning",
            "example_v_dense",
            "example_barrier_eV4",
            "example_dense_depth_eV4",
            "notes",
        ]
    ].to_csv(summary_csv, index=False)

    make_plots(
        out_png=plot_png,
        summary=summary,
        V0_quartic_root_eV=V0_quartic_root_eV,
        curve_examples=curve_examples,
    )

    build_report(
        out_md=report_md,
        rhoLambda_J_m3=args.rhoLambda_J_m3,
        rhoLambda_eV4=rhoLambda_eV4,
        V0_quartic_root_eV=V0_quartic_root_eV,
        dense_vals_GeV=dense_vals_GeV,
        summary=summary,
        rhoLambda_over_rhoDM=args.rhoLambda_over_rhoDM,
        anchor_exists=OPTIONAL_DENSE_ANCHOR.exists(),
    )

    print(f"[P3T] V0^(1/4) = {V0_quartic_root_eV*1e3:.6f} meV")
    for _, row in summary.iterrows():
        print(
            f"[P3T] E_dense={row['E_dense_GeV']:.0f} GeV: "
            f"log10_tuning={row['log10_tuning']:.6f}"
        )
    print(f"[P3T] Output folder: {out_dir}")
    print(f"[P3T] Files:\n  {summary_csv}\n  {report_md}\n  {plot_png}")


if __name__ == "__main__":
    main()
