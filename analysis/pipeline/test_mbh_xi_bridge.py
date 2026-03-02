#!/usr/bin/env python3
"""
MBH ↔ xi bridge test.

Joins SPARC kinematic data (Mdyn, xi, Mstar=V^4/(G g†)) to an external
MBH catalog and fits scaling relations:
  A) log MBH ~ log Mdyn
  B) log MBH ~ log xi
  C) log MBH ~ log Vout
  D) log MBH ~ log Mstar  (Mstar = Vout^4 / (G g†))

Outputs plots, a match CSV, a fit summary JSON, and a markdown report.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

warnings.filterwarnings("ignore")
plt.style.use("default")

# ---------- physical constants ----------
G_SI = 6.674e-11      # m^3 kg^-1 s^-2
MSUN = 1.989e30        # kg
KPC = 3.086e19         # m
KMS_TO_MS = 1e3        # km/s → m/s


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(obj), indent=2))


def canon(name: str) -> str:
    """Canonical galaxy name for fuzzy matching."""
    s = str(name).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^NGC0*", "NGC", s)
    s = re.sub(r"^UGC0*", "UGC", s)
    s = re.sub(r"^IC0*", "IC", s)
    return s


def load_sparc_kinematics(root: Path, g_dagger: float) -> pd.DataFrame:
    """Build per-galaxy table from SPARC RAR points: Mdyn, Vout, xi, Mstar."""
    pts = pd.read_csv(root / "analysis/results/rar_points_unified.csv")
    pts = pts[pts["source"] == "SPARC"].copy()
    pts = pts[np.isfinite(pts["log_gbar"]) & np.isfinite(pts["log_gobs"])
              & np.isfinite(pts["R_kpc"]) & (pts["R_kpc"] > 0)].copy()

    rows: List[Dict[str, Any]] = []
    for gal, g in pts.groupby("galaxy", sort=False):
        g2 = g.sort_values("R_kpc")
        if len(g2) < 5:
            continue
        r_kpc = g2["R_kpc"].to_numpy(dtype=float)
        log_gbar = g2["log_gbar"].to_numpy(dtype=float)
        log_gobs = g2["log_gobs"].to_numpy(dtype=float)

        # Outermost point
        j = len(r_kpc) - 1
        R_out = r_kpc[j]
        gobs_out = 10.0 ** log_gobs[j]
        gbar_out = 10.0 ** log_gbar[j]

        # Vout from gobs: V = sqrt(gobs * R)
        V_out_ms = np.sqrt(gobs_out * R_out * KPC)  # m/s
        V_out_kms = V_out_ms / KMS_TO_MS

        # Mdyn = gobs * R^2 / G
        M_dyn = gobs_out * (R_out * KPC) ** 2 / G_SI / MSUN
        if not np.isfinite(M_dyn) or M_dyn <= 0:
            continue

        # Mstar_BEC = V^4 / (G * g†)
        M_star = V_out_ms ** 4 / (G_SI * g_dagger) / MSUN
        if not np.isfinite(M_star) or M_star <= 0:
            continue

        # xi = healing length = sqrt(G Mdyn Msun / g†) / kpc
        xi_kpc = np.sqrt(G_SI * M_dyn * MSUN / g_dagger) / KPC

        rows.append({
            "galaxy": str(gal),
            "canon": canon(str(gal)),
            "n_points": int(len(g2)),
            "R_out_kpc": float(R_out),
            "V_out_kms": float(V_out_kms),
            "log_Mdyn": float(np.log10(M_dyn)),
            "log_Mstar": float(np.log10(M_star)),
            "xi_kpc": float(xi_kpc),
            "log_xi": float(np.log10(xi_kpc)),
            "log_Vout": float(np.log10(V_out_kms)),
        })
    return pd.DataFrame(rows)


def load_mbh_catalog(root: Path) -> pd.DataFrame:
    p = root / "analysis/data/mbh_catalog.csv"
    if not p.exists():
        raise FileNotFoundError(f"MBH catalog not found: {p}")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise ValueError(
            f"MBH catalog is empty (0 rows): {p}\n"
            "Please populate it with columns: galaxy,log10_MBH_Msun,MBH_sigma_dex,ref,notes"
        )
    df["canon"] = df["galaxy"].apply(canon)
    return df


def match_catalogs(sparc: pd.DataFrame, mbh: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    merged = sparc.merge(mbh, on="canon", how="inner", suffixes=("_sparc", "_mbh"))
    # Use SPARC galaxy name as primary
    if "galaxy_sparc" in merged.columns:
        merged["galaxy"] = merged["galaxy_sparc"]
    n_unmatched = len(mbh) - len(merged)
    return merged, n_unmatched


def classify_measurement_method(ref: Any, notes: Any) -> str:
    """Classify MBH estimate provenance for split reporting."""
    txt = f"{str(ref)} {str(notes)}".lower()
    txt = txt.replace("–", "-").replace("—", "-")
    if ("m-sigma" in txt) or ("m sigma" in txt) or ("msigma" in txt):
        return "M-sigma"
    if ("dyn" in txt) or ("dynamical" in txt) or ("reverb" in txt) or re.search(r"\brm\b", txt):
        return "dyn/RM"
    return "unknown"


def ols_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """OLS linear fit: y = a + b*x."""
    A = np.column_stack([np.ones_like(x), x])
    beta, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    resid = y - yhat
    rms = float(np.sqrt(np.mean(resid ** 2)))
    mad = float(median_abs_deviation(resid, scale="normal"))
    return {"a": float(beta[0]), "b": float(beta[1]), "rms": rms, "mad": mad,
            "yhat": yhat.tolist(), "resid": resid.tolist()}


def huber_fit(x: np.ndarray, y: np.ndarray, delta: float = 1.345) -> Dict[str, float]:
    """Iteratively reweighted least squares with Huber loss."""
    A = np.column_stack([np.ones_like(x), x])
    w = np.ones(len(y))
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(50):
        resid = y - A @ beta
        sigma = float(median_abs_deviation(resid, scale="normal")) or 1e-10
        u = np.abs(resid) / sigma
        w = np.where(u <= delta, 1.0, delta / u)
        Aw = A * w[:, None]
        beta = np.linalg.lstsq(Aw, y * w, rcond=None)[0]
    yhat = A @ beta
    resid = y - yhat
    rms = float(np.sqrt(np.mean(resid ** 2)))
    mad = float(median_abs_deviation(resid, scale="normal"))
    return {"a": float(beta[0]), "b": float(beta[1]), "rms": rms, "mad": mad,
            "yhat": yhat.tolist(), "resid": resid.tolist()}


def run_model(name: str, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    o = ols_fit(x, y)
    h = huber_fit(x, y)
    return {
        "model": name,
        "N": int(len(x)),
        "OLS": {"a": o["a"], "b": o["b"], "rms": o["rms"], "mad": o["mad"]},
        "Huber": {"a": h["a"], "b": h["b"], "rms": h["rms"], "mad": h["mad"]},
        "_ols_yhat": o["yhat"], "_ols_resid": o["resid"],
        "_huber_yhat": h["yhat"], "_huber_resid": h["resid"],
    }


def make_scatter_plot(ax, x, y, xlabel, ylabel, title, fit_result):
    ax.scatter(x, y, s=40, c="#1f77b4", edgecolors="black", linewidths=0.5, zorder=3)
    xgrid = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
    ols = fit_result["OLS"]
    hub = fit_result["Huber"]
    ax.plot(xgrid, ols["a"] + ols["b"] * xgrid, "r-", lw=1.5,
            label=f"OLS: {ols['b']:.2f}x+{ols['a']:.2f}  rms={ols['rms']:.3f}")
    ax.plot(xgrid, hub["a"] + hub["b"] * xgrid, "g--", lw=1.5,
            label=f"Huber: {hub['b']:.2f}x+{hub['a']:.2f}  rms={hub['rms']:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)


def run_model_bundle(df: pd.DataFrame, models: Dict[str, Tuple[str, str]]) -> Optional[Dict[str, Any]]:
    """Fit all bridge models for a subset. Returns None if too few points."""
    if len(df) < 3:
        return None
    y = df["log10_MBH_Msun"].to_numpy(dtype=float)
    out: Dict[str, Any] = {}
    for key, (xcol, _xlabel) in models.items():
        x = df[xcol].to_numpy(dtype=float)
        out[key] = run_model(key, x, y)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MBH-xi bridge test")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--g_dagger", type=float, default=1.2e-10,
                        help="Acceleration scale g† in m/s^2")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = Path(args.project_root).resolve() if args.project_root else script_path.parents[2]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "outputs" / "mbh_xi_bridge" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    g_dagger = args.g_dagger
    print(f"g† = {g_dagger:.4e} m/s²")
    print(f"Output: {out_dir}")

    # Load data
    sparc = load_sparc_kinematics(root, g_dagger)
    print(f"SPARC galaxies with kinematics: {len(sparc)}")

    mbh = load_mbh_catalog(root)
    print(f"MBH catalog entries: {len(mbh)}")

    matched, n_unmatched = match_catalogs(sparc, mbh)
    print(f"Matched galaxies: {len(matched)}")
    print(f"Unmatched MBH entries: {n_unmatched}")

    if len(matched) < 3:
        raise RuntimeError(f"Only {len(matched)} matches — need at least 3 to fit.")

    matched["method_group"] = matched.apply(
        lambda r: classify_measurement_method(r.get("ref", ""), r.get("notes", "")),
        axis=1,
    )
    method_counts = {
        str(k): int(v)
        for k, v in matched["method_group"].value_counts(dropna=False).to_dict().items()
    }

    # Save match table
    match_cols = ["galaxy", "log10_MBH_Msun", "MBH_sigma_dex", "ref", "notes", "method_group",
                  "log_Mdyn", "log_Mstar", "log_xi", "log_Vout",
                  "V_out_kms", "xi_kpc", "R_out_kpc", "n_points"]
    matched[match_cols].to_csv(out_dir / "mbh_matches.csv", index=False)

    # Run four models
    models = {
        "A_MBH_vs_Mdyn": ("log_Mdyn", "log₁₀ Mdyn [M☉]"),
        "B_MBH_vs_xi": ("log_xi", "log₁₀ ξ [kpc]"),
        "C_MBH_vs_Vout": ("log_Vout", "log₁₀ Vout [km/s]"),
        "D_MBH_vs_Mstar": ("log_Mstar", "log₁₀ M★ [M☉]"),
    }
    subset_frames: Dict[str, pd.DataFrame] = {
        "m_sigma_only": matched[matched["method_group"] == "M-sigma"].copy(),
        "dyn_rm_only": matched[matched["method_group"] == "dyn/RM"].copy(),
        "combined": matched.copy(),
    }
    subset_results: Dict[str, Optional[Dict[str, Any]]] = {
        key: run_model_bundle(df_sub, models) for key, df_sub in subset_frames.items()
    }
    if subset_results["combined"] is None:
        raise RuntimeError("Combined sample has <3 rows; cannot fit bridge models.")
    results = subset_results["combined"]
    y = matched["log10_MBH_Msun"].to_numpy(dtype=float)

    # --- Plots ---
    # Individual scatter plots
    plot_specs = [
        ("A_MBH_vs_Mdyn", "log_Mdyn", "log₁₀ Mdyn [M☉]", "fig_mbh_vs_mdyn.png"),
        ("B_MBH_vs_xi", "log_xi", "log₁₀ ξ [kpc]", "fig_mbh_vs_xi.png"),
        ("C_MBH_vs_Vout", "log_Vout", "log₁₀ Vout [km/s]", "fig_mbh_vs_vout.png"),
        ("D_MBH_vs_Mstar", "log_Mstar", "log₁₀ M★ [M☉]", "fig_mbh_vs_mstar.png"),
    ]
    for key, xcol, xlabel, fname in plot_specs:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        x = matched[xcol].to_numpy(dtype=float)
        make_scatter_plot(ax, x, y, xlabel, "log₁₀ MBH [M☉]",
                          f"MBH vs {xcol.replace('log_', '')}", results[key])
        for i, gal in enumerate(matched["galaxy"]):
            ax.annotate(gal, (x[i], y[i]), fontsize=5, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(out_dir / fname, facecolor="white")
        plt.close(fig)

    # Residuals panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    for ax, (key, (xcol, xlabel)) in zip(axes.flat, models.items()):
        resid = np.array(results[key]["_ols_resid"])
        ax.bar(range(len(resid)), resid, color="#9ecae1", edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_title(f"{key} OLS residuals (rms={results[key]['OLS']['rms']:.3f})")
        ax.set_ylabel("Δ log MBH")
        ax.set_xlabel("galaxy index")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_residuals.png", facecolor="white")
    plt.close(fig)

    # --- Summary JSON ---
    summary: Dict[str, Any] = {
        "test": "mbh_xi_bridge",
        "g_dagger": g_dagger,
        "n_matched": int(len(matched)),
        "n_unmatched_mbh": int(n_unmatched),
        "n_sparc_total": int(len(sparc)),
        "method_group_counts": method_counts,
        "models": {},
        "models_by_subset": {},
    }
    for key, r in results.items():
        summary["models"][key] = {
            "N": r["N"],
            "OLS": r["OLS"],
            "Huber": r["Huber"],
        }
    for subset_key in ["m_sigma_only", "dyn_rm_only", "combined"]:
        df_sub = subset_frames[subset_key]
        fitted = subset_results[subset_key]
        if fitted is None:
            summary["models_by_subset"][subset_key] = {
                "N": int(len(df_sub)),
                "status": "insufficient_n_for_fit",
                "models": {},
            }
            continue
        summary["models_by_subset"][subset_key] = {
            "N": int(len(df_sub)),
            "status": "ok",
            "models": {
                model_key: {
                    "N": model_result["N"],
                    "OLS": model_result["OLS"],
                    "Huber": model_result["Huber"],
                }
                for model_key, model_result in fitted.items()
            },
        }
    write_json(out_dir / "mbh_fit_summary.json", summary)

    # --- Markdown report ---
    lines = [
        "# MBH ↔ ξ Bridge Test Report",
        f"",
        f"**Date**: {ts}",
        f"**g†**: {g_dagger:.4e} m/s²",
        f"**Matched galaxies**: {len(matched)}",
        f"**Unmatched MBH entries**: {n_unmatched}",
        "",
        "## Method split",
        "",
        f"- M-sigma-only: {len(subset_frames['m_sigma_only'])}",
        f"- dyn/RM-only: {len(subset_frames['dyn_rm_only'])}",
        f"- combined: {len(subset_frames['combined'])}",
        f"- unknown tags: {method_counts.get('unknown', 0)}",
        "",
        "## Scaling Relations (by subset)",
        "",
    ]
    subset_titles = [
        ("M-sigma-only", "m_sigma_only"),
        ("dyn/RM-only", "dyn_rm_only"),
        ("combined", "combined"),
    ]
    for subset_label, subset_key in subset_titles:
        lines += [
            f"### {subset_label}",
            "",
        ]
        subset_fit = subset_results[subset_key]
        if subset_fit is None:
            lines += [
                f"- N={len(subset_frames[subset_key])}; insufficient N for regression fits (need N ≥ 3).",
                "",
            ]
            continue
        lines += [
            "| Model | N | OLS slope | OLS rms | OLS MAD | Huber slope | Huber rms | Huber MAD |",
            "|-------|---|-----------|---------|---------|-------------|-----------|-----------|",
        ]
        for key, r in subset_fit.items():
            o, h = r["OLS"], r["Huber"]
            lines.append(
                f"| {key} | {r['N']} | {o['b']:.3f} | {o['rms']:.3f} | {o['mad']:.3f} "
                f"| {h['b']:.3f} | {h['rms']:.3f} | {h['mad']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Scaling Relations (combined, compatibility view)",
        "",
        "| Model | N | OLS slope | OLS rms | OLS MAD | Huber slope | Huber rms | Huber MAD |",
        "|-------|---|-----------|---------|---------|-------------|-----------|-----------|",
    ]
    for key, r in results.items():
        o, h = r["OLS"], r["Huber"]
        lines.append(
            f"| {key} | {r['N']} | {o['b']:.3f} | {o['rms']:.3f} | {o['mad']:.3f} "
            f"| {h['b']:.3f} | {h['rms']:.3f} | {h['mad']:.3f} |"
        )
    lines += ["", "## Matched Galaxies", ""]
    for _, row in matched.iterrows():
        lines.append(f"- **{row['galaxy']}**: log MBH={row['log10_MBH_Msun']:.2f}, "
                      f"log Mdyn={row['log_Mdyn']:.2f}, log ξ={row['log_xi']:.2f}, "
                      f"log M★={row['log_Mstar']:.2f}")
    lines.append("")
    (out_dir / "report_mbh_xi_bridge.md").write_text("\n".join(lines))

    # --- Print results block ---
    print("\n" + "=" * 60)
    print("MBH ↔ ξ BRIDGE TEST — RESULTS")
    print("=" * 60)
    print(f"N matched galaxies: {len(matched)}")
    print(f"Match issues (unmatched MBH): {n_unmatched}")
    print(
        "Method groups: "
        f"M-sigma-only={len(subset_frames['m_sigma_only'])}, "
        f"dyn/RM-only={len(subset_frames['dyn_rm_only'])}, "
        f"combined={len(subset_frames['combined'])}, "
        f"unknown={method_counts.get('unknown', 0)}"
    )
    print()
    for subset_label, subset_key in [
        ("M-sigma-only", "m_sigma_only"),
        ("dyn/RM-only", "dyn_rm_only"),
        ("combined", "combined"),
    ]:
        print(f"  [{subset_label}] N={len(subset_frames[subset_key])}")
        subset_fit = subset_results[subset_key]
        if subset_fit is None:
            print("    insufficient N for regression fits (need N >= 3)")
            continue
        for key, r in subset_fit.items():
            o, h = r["OLS"], r["Huber"]
            print(f"    {key}:")
            print(f"      OLS   slope={o['b']:.3f}  rms={o['rms']:.3f}  MAD={o['mad']:.3f}")
            print(f"      Huber slope={h['b']:.3f}  rms={h['rms']:.3f}  MAD={h['mad']:.3f}")
    print(f"\nOutput folder: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
