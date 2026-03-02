#!/usr/bin/env python3
"""
Paper 3 starter module: cosmology + phase-potential bridge constraints.

This script builds two toy potential families, anchors their interpretation to
throat-budget proxy outputs, and writes summary artifacts for Paper 3 drafting.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "outputs" / "paper3_cosmology"

RHO_LAMBDA_J_M3 = 6.0e-10
W_TARGET = -1.0

# SI <-> natural-unit conversion constants.
JOULE_PER_EV = 1.602176634e-19
HBAR_C_EV_M = 1.973269804e-7  # eV*m
J_M3_TO_EV4 = (1.0 / JOULE_PER_EV) * (HBAR_C_EV_M ** 3)
EV4_TO_J_M3 = 1.0 / J_M3_TO_EV4

MEV_EV = 1.0e-3
GEV_EV = 1.0e9
TEV_EV = 1.0e12

ANCHOR_INPUTS = {
    "stageA_M": Path(
        "/Users/russelllicht/bh-singularity/outputs/thick_shell_budget_sweep/"
        "20260228_025306/M/alpha_vs_delta.csv"
    ),
    "stageA_H": Path(
        "/Users/russelllicht/bh-singularity/outputs/thick_shell_budget_sweep/"
        "20260228_025306/H/alpha_vs_delta.csv"
    ),
    "kappa_summary": Path(
        "/Users/russelllicht/bh-singularity/outputs/kappa_estimate/"
        "20260228_050842/kappa_estimate_summary.csv"
    ),
    "A1_band": Path(
        "/Users/russelllicht/bh-singularity/outputs/kappa_bands/"
        "20260228_052709/A1_band_table.csv"
    ),
}


def j_m3_to_ev4(rho_j_m3: float) -> float:
    return float(rho_j_m3) * J_M3_TO_EV4


def ev4_to_j_m3(rho_ev4: float) -> float:
    return float(rho_ev4) * EV4_TO_J_M3


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required anchor input: {path}")
    return path


def load_throat_anchors() -> Dict[str, object]:
    stageA_M = pd.read_csv(_require_file(ANCHOR_INPUTS["stageA_M"]))
    stageA_H = pd.read_csv(_require_file(ANCHOR_INPUTS["stageA_H"]))
    kappa = pd.read_csv(_require_file(ANCHOR_INPUTS["kappa_summary"]))
    a1_band = pd.read_csv(_require_file(ANCHOR_INPUTS["A1_band"]))

    m_star = stageA_M.loc[stageA_M["alpha_min"].idxmin()]
    h_star = stageA_H.loc[stageA_H["alpha_min"].idxmin()]

    kappa = kappa.copy()
    kappa["kappa_over_A1_qnm"] = kappa["kappa_inv_delta"] / kappa["A1_qnm"]
    kappa["kappa_over_A1_echo"] = kappa["kappa_inv_delta"] / kappa["A1_echo"]

    kappa_ratio = {}
    for tier, sub in kappa.groupby("tier"):
        kappa_ratio[tier] = {
            "qnm_min": float(sub["kappa_over_A1_qnm"].min()),
            "qnm_max": float(sub["kappa_over_A1_qnm"].max()),
            "echo_min": float(sub["kappa_over_A1_echo"].min()),
            "echo_max": float(sub["kappa_over_A1_echo"].max()),
            "kappa_inv_delta_min": float(sub["kappa_inv_delta"].min()),
            "kappa_inv_delta_max": float(sub["kappa_inv_delta"].max()),
            "c_map_min": float(sub["c_map"].min()),
            "c_map_max": float(sub["c_map"].max()),
        }

    a1_primary = a1_band[a1_band["band"].astype(str).str.upper() == "PRIMARY"].copy()
    a1_primary_summary = {}
    for _, row in a1_primary.iterrows():
        tier = str(row["tier"])
        a1_primary_summary[tier] = {
            "A1_qnm_min": float(row["A1_qnm_min"]),
            "A1_qnm_max": float(row["A1_qnm_max"]),
            "A1_echo_min": float(row["A1_echo_min"]),
            "A1_echo_max": float(row["A1_echo_max"]),
        }

    return {
        "delta_alpha_star": {
            "M": {"Delta_star": float(m_star["Delta"]), "alpha_star": float(m_star["alpha_min"])} ,
            "H": {"Delta_star": float(h_star["Delta"]), "alpha_star": float(h_star["alpha_min"])} ,
        },
        "kappa_ratio": kappa_ratio,
        "a1_primary": a1_primary_summary,
    }


def family1_result(v_eV: float, lambda_dimless: float, rho_lambda_ev4: float) -> Dict[str, object]:
    if lambda_dimless <= 0.0:
        raise ValueError("Family 1 requires lambda > 0")
    if v_eV <= 0.0:
        raise ValueError("Family 1 requires v > 0")

    # V = lambda/4 (phi^2-v^2)^2 + V0; minima at phi=+-v with Vmin=V0.
    V0 = rho_lambda_ev4
    Vmin = V0
    barrier = 0.25 * lambda_dimless * (v_eV ** 4)

    return {
        "family": "family1_symmetry_breaking_offset",
        "params": json.dumps({"lambda": lambda_dimless, "v_eV": v_eV}, sort_keys=True),
        "v": v_eV,
        "lambda": lambda_dimless,
        "V0": V0,
        "Vmin": Vmin,
        "barrier_height": barrier,
        "V0_quartic_root_eV": V0 ** 0.25,
        "notes": f"minima at phi=+-{v_eV:.3e} eV; barrier at phi=0",
        "V0_J_m3": ev4_to_j_m3(V0),
        "Vmin_J_m3": ev4_to_j_m3(Vmin),
    }


def _family2_stationary_points(a: float, b: float, c: float) -> List[float]:
    """
    Stationary points for x >= 0 using x = |phi| as the order parameter.

    dV/dx = 2 a x + 4 b x^3 + 6 c x^5 = 0
    => x = 0 or y = x^2 solves 3 c y^2 + 2 b y + a = 0.
    """
    if c <= 0.0:
        raise ValueError("Family 2 requires c > 0 for stability")

    y_roots = np.roots([3.0 * c, 2.0 * b, a])
    points = [0.0]
    for root in y_roots:
        if abs(root.imag) < 1e-9 and root.real > 0.0:
            points.append(float(np.sqrt(root.real)))
    points = sorted(set(points))
    return points


def family2_result(a: float, b: float, c: float, rho_lambda_ev4: float) -> Dict[str, object]:
    stationary = _family2_stationary_points(a=a, b=b, c=c)

    def U(x: float) -> float:
        return a * (x ** 2) + b * (x ** 4) + c * (x ** 6)

    def d2U(x: float) -> float:
        return 2.0 * a + 12.0 * b * (x ** 2) + 30.0 * c * (x ** 4)

    minima: List[Tuple[float, float]] = []
    maxima: List[Tuple[float, float]] = []
    for x in stationary:
        val = U(x)
        curv = d2U(x)
        if curv > 0.0:
            minima.append((x, val))
        elif curv < 0.0:
            maxima.append((x, val))

    if len(minima) < 2:
        raise RuntimeError(
            "Family 2 parameter set does not produce two minima in |phi| domain"
        )

    minima = sorted(minima, key=lambda t: t[0])
    low_min = minima[0]
    dense_min = minima[-1]

    between_barriers = [m for m in maxima if low_min[0] < m[0] < dense_min[0]]
    if not between_barriers:
        raise RuntimeError(
            "Family 2 parameter set has minima but no barrier between dilute and dense minima"
        )
    barrier_pt = between_barriers[0]

    # Enforce Vmin (dilute cosmological phase) = rhoLambda.
    V0 = rho_lambda_ev4 - low_min[1]

    V_low = low_min[1] + V0
    V_dense = dense_min[1] + V0
    V_barrier = barrier_pt[1] + V0

    return {
        "family": "family2_two_minima_sextic",
        "params": json.dumps({"a_eV2": a, "b": b, "c_inv_eV2": c}, sort_keys=True),
        "v": dense_min[0],
        "lambda": np.nan,
        "V0": V0,
        "Vmin": V_low,
        "barrier_height": V_barrier - V_low,
        "V0_quartic_root_eV": math.pow(max(V0, 0.0), 0.25),
        "notes": (
            f"|phi| minima at {low_min[0]:.3e}, {dense_min[0]:.3e} eV; "
            f"barrier at {barrier_pt[0]:.3e} eV; dense-phase DeltaV={V_dense - V_low:.3e} eV^4"
        ),
        "V0_J_m3": ev4_to_j_m3(V0),
        "Vmin_J_m3": ev4_to_j_m3(V_low),
    }


def make_scale_plot(path: Path, v0_quartic_root_eV: float) -> None:
    labels = [r"$V_0^{1/4}$", "1 meV", "1 GeV", "1 TeV"]
    values = [v0_quartic_root_eV, MEV_EV, GEV_EV, TEV_EV]
    colors = ["#0b4f6c", "#1f9d55", "#d97706", "#b91c1c"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(labels))
    ax.scatter(x, values, color=colors, s=95, zorder=3)

    for xi, yi, label in zip(x, values, labels):
        ax.vlines(x=xi, ymin=values[0], ymax=yi, color="#94a3b8", lw=1.0, zorder=1)
        ax.text(xi, yi * 1.25, f"{label}\n{yi:.3e} eV", ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.set_ylabel("Energy scale [eV]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(r"Vacuum-offset scale vs meV/GeV/TeV reference markers")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _summary_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "family",
        "v",
        "V0",
        "Vmin",
        "barrier_height",
        "V0_quartic_root_eV",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [
            str(row["family"]),
            f"{row['v']:.3e}",
            f"{row['V0']:.3e}",
            f"{row['Vmin']:.3e}",
            f"{row['barrier_height']:.3e}",
            f"{row['V0_quartic_root_eV']:.3e}",
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    anchors: Dict[str, object],
    summary_df: pd.DataFrame,
    rho_lambda_ev4: float,
    v0_quartic_root_eV: float,
) -> None:
    stars = anchors["delta_alpha_star"]
    kappa_ratio = anchors["kappa_ratio"]
    a1_primary = anchors["a1_primary"]

    lines: List[str] = []
    lines.append("# Paper 3 Cosmology + Phase Potential Bridge (Starter)")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Inputs (treated as fixed here)")
    lines.append(f"- `rhoLambda_J_m3 = {RHO_LAMBDA_J_M3:.3e}`")
    lines.append(f"- `rhoLambda_eV4 = {rho_lambda_ev4:.3e}`")
    lines.append(f"- `w_target ~= {W_TARGET:.1f}` (enforced as vacuum-dominance requirement only)")
    lines.append("")
    lines.append("## Throat-Scale Anchors Loaded (read-only from bh-singularity)")
    lines.append(
        "- Stage-A minima: "
        f"M `(Delta*, alpha*)=({stars['M']['Delta_star']:.6g}, {stars['M']['alpha_star']:.6g})`, "
        f"H `(Delta*, alpha*)=({stars['H']['Delta_star']:.6g}, {stars['H']['alpha_star']:.6g})`"
    )
    lines.append(
        "- Stage-B proxy kappa/A1 bands: "
        f"M qnm [{kappa_ratio['M']['qnm_min']:.3f}, {kappa_ratio['M']['qnm_max']:.3f}], "
        f"H qnm [{kappa_ratio['H']['qnm_min']:.3f}, {kappa_ratio['H']['qnm_max']:.3f}]"
    )
    lines.append(
        "- PRIMARY A1 bands: "
        f"M qnm [{a1_primary['M']['A1_qnm_min']:.3e}, {a1_primary['M']['A1_qnm_max']:.3e}], "
        f"H qnm [{a1_primary['H']['A1_qnm_min']:.3e}, {a1_primary['H']['A1_qnm_max']:.3e}]"
    )
    lines.append("")
    lines.append("## Why contact BEC interaction energy cannot be Lambda")
    lines.append(
        "For a contact-interaction BEC component, the interaction pressure is non-negative "
        "(`w >= 0` in effective-fluid language), so it redshifts with expansion and cannot "
        "produce late-time acceleration (`w < -1/3` needed)."
    )
    lines.append("")
    lines.append("## Minimal Potential Structure Demonstrated")
    lines.append("- Family 1 (`lambda/4 (phi^2-v^2)^2 + V0`) makes explicit that a vacuum offset `V0` is needed.")
    lines.append("- Family 2 (`a phi^2 + b phi^4 + c phi^6 + V0`) shows two-phase minima with an explicit barrier.")
    lines.append("")
    lines.append(_summary_markdown_table(summary_df))
    lines.append("")
    lines.append("## Scale Comparison")
    lines.append(
        f"- `V0^(1/4)` for the cosmological vacuum requirement is `{v0_quartic_root_eV:.3e} eV` "
        f"(~{v0_quartic_root_eV / MEV_EV:.3f} meV)."
    )
    lines.append(
        "- This module treats throat-budget anchors as constraints on gradient/impedance structure; "
        "they do not independently fix the absolute cosmological offset `V0`."
    )
    lines.append(
        "- Practical interpretation: a two-scale EFT is needed (meV vacuum offset + denser "
        "phase structure at much higher field scales, often EW/TeV-ish in toy examples)."
    )
    lines.append("")
    lines.append("## Deferred (explicitly not solved here)")
    lines.append("- No fit to `w(z)` or full background/perturbation cosmology.")
    lines.append("- No microphysical derivation of potential coefficients from UV theory.")
    lines.append("- No solved cosmological-constant problem; this is a bookkeeping bridge.")
    lines.append(
        "- Galaxy-side lever for `rho_c`: constrain transition density using high-density systems "
        "(inner-region residual shifts / phase-indicator onset versus inferred local density)."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    anchors = load_throat_anchors()
    print("[P3C] Loaded throat-scale anchors (Δ*, α*, κ/A1 bands) from bh-singularity outputs")

    rho_lambda_ev4 = j_m3_to_ev4(RHO_LAMBDA_J_M3)
    v0_quartic_root_eV = rho_lambda_ev4 ** 0.25

    print("[P3C] Contact BEC interaction energy has w>=0; cannot source acceleration")

    family1_params = [
        {"v_eV": 246.0e9, "lambda": 0.13},
        {"v_eV": 1.0e12, "lambda": 0.50},
    ]

    family2_params = [
        {"a": 4.2e22, "b": -0.40, "c": 1.0e-24},
        {"a": 1.8e23, "b": -0.40, "c": 2.5e-25},
    ]

    rows: List[Dict[str, object]] = []
    for fp in family1_params:
        rows.append(
            family1_result(
                v_eV=fp["v_eV"],
                lambda_dimless=fp["lambda"],
                rho_lambda_ev4=rho_lambda_ev4,
            )
        )

    print(
        "[P3C] Potential family 1: V0 set to match rhoLambda; "
        f"V0^(1/4)={v0_quartic_root_eV:.6e} eV"
    )

    family2_first_barrier = float("nan")
    for idx, fp in enumerate(family2_params):
        res = family2_result(
            a=fp["a"],
            b=fp["b"],
            c=fp["c"],
            rho_lambda_ev4=rho_lambda_ev4,
        )
        rows.append(res)
        if idx == 0:
            family2_first_barrier = float(res["barrier_height"])

    print(
        "[P3C] Potential family 2: two minima exist; "
        f"barrier={family2_first_barrier:.6e} eV^4, "
        f"V0^(1/4)={v0_quartic_root_eV:.6e} eV"
    )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[
        [
            "family",
            "params",
            "v",
            "lambda",
            "V0",
            "Vmin",
            "barrier_height",
            "V0_quartic_root_eV",
            "notes",
            "V0_J_m3",
            "Vmin_J_m3",
        ]
    ]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    summary_csv = out_dir / "potential_scan_summary.csv"
    report_md = out_dir / "potential_scan_report.md"
    plot_png = out_dir / "potential_scale_plots.png"

    summary_df.to_csv(summary_csv, index=False)
    write_report(
        path=report_md,
        anchors=anchors,
        summary_df=summary_df,
        rho_lambda_ev4=rho_lambda_ev4,
        v0_quartic_root_eV=v0_quartic_root_eV,
    )
    make_scale_plot(path=plot_png, v0_quartic_root_eV=v0_quartic_root_eV)

    print(f"[P3C] Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
