#!/usr/bin/env python3
"""
Direct-only MBH <-> xi/Mdyn bridge rerun using BHcompilation.fits.

Produces two subsets against SPARC kinematics:
  1) strict detections: SELECTED=1 and UPPERLIMIT=0
  2) larger-N sensitivity: SELECTED=1 and finite MBH, including upper limits
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from astropy.io import fits

# Reuse canonical kinematics builder + fit models via path-based import
BRIDGE_PATH = Path(__file__).resolve().parent / "test_mbh_xi_bridge.py"
spec = importlib.util.spec_from_file_location("mbh_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(bridge)


def canon(name: str) -> str:
    s = str(name).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^NGC0*", "NGC", s)
    s = re.sub(r"^UGC0*", "UGC", s)
    s = re.sub(r"^IC0*", "IC", s)
    return s


def run_two_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    y = df["log10_MBH_Msun"].to_numpy(dtype=float)
    x_mdyn = df["log_Mdyn"].to_numpy(dtype=float)
    x_xi = df["log_xi"].to_numpy(dtype=float)
    return {
        "A_MBH_vs_Mdyn": bridge.run_model("A_MBH_vs_Mdyn", x_mdyn, y),
        "B_MBH_vs_xi": bridge.run_model("B_MBH_vs_xi", x_xi, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-only MBH bridge rerun")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--g_dagger", type=float, default=1.286e-10)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "outputs" / "mbh_xi_bridge_direct" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # SPARC kinematics
    kin = bridge.load_sparc_kinematics(root, args.g_dagger).copy()
    kin["canon"] = kin["galaxy"].apply(canon)

    # BHcompilation rows
    bh_path = root / "raw_data" / "observational" / "bh_masses" / "BHcompilation.fits"
    with fits.open(bh_path) as hdul:
        data = hdul[1].data

    rows = []
    for i in range(len(data)):
        rows.append(
            {
                "name_bh": str(data["NAME"][i]).strip(),
                "canon": canon(str(data["NAME"][i]).strip()),
                "log10_MBH_Msun": float(data["MBH"][i]) if np.isfinite(data["MBH"][i]) else np.nan,
                "MBH_sigma_dex": float(data["DMBH"][i]) if np.isfinite(data["DMBH"][i]) else np.nan,
                "SELECTED": int(data["SELECTED"][i]) if np.isfinite(data["SELECTED"][i]) else -999,
                "UPPERLIMIT": int(data["UPPERLIMIT"][i]) if np.isfinite(data["UPPERLIMIT"][i]) else 1,
                "TYPE": str(data["TYPE"][i]).strip(),
                "REF": str(data["REF"][i]).strip(),
            }
        )

    bh = pd.DataFrame(rows)
    bh = bh[np.isfinite(bh["log10_MBH_Msun"]) & (bh["SELECTED"] == 1)].copy()

    matched = kin.merge(bh, on="canon", how="inner")
    matched["is_upper_limit"] = matched["UPPERLIMIT"].astype(int) == 1

    strict = matched[~matched["is_upper_limit"]].copy()
    extended = matched.copy()

    strict.to_csv(out_dir / "direct_detected_matches.csv", index=False)
    extended.to_csv(out_dir / "direct_plus_upper_matches.csv", index=False)

    summary: Dict[str, Any] = {
        "test": "mbh_xi_bridge_direct_only",
        "timestamp": ts,
        "g_dagger": args.g_dagger,
        "n_sparc_kinematics": int(len(kin)),
        "n_matched_selected_all": int(len(matched)),
        "n_direct_detected": int(len(strict)),
        "n_direct_upper_limits": int(matched["is_upper_limit"].sum()),
        "subsets": {},
    }

    for subset_key, subset_df in [
        ("direct_detected_only", strict),
        ("direct_plus_upper_sensitivity", extended),
    ]:
        if len(subset_df) < 3:
            summary["subsets"][subset_key] = {"N": int(len(subset_df)), "status": "insufficient_n"}
            continue
        fits_out = run_two_models(subset_df)
        summary["subsets"][subset_key] = {
            "N": int(len(subset_df)),
            "contains_upper_limits": bool(subset_df["is_upper_limit"].any()),
            "models": {
                model_key: {
                    "N": model_result["N"],
                    "OLS": model_result["OLS"],
                    "Huber": model_result["Huber"],
                }
                for model_key, model_result in fits_out.items()
            },
        }

    (out_dir / "summary_direct_bridge.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Direct MBH (non-M-sigma) vs xi/Mdyn bridge rerun",
        "",
        f"- Timestamp: {ts}",
        f"- g\u2020: {args.g_dagger:.4e} m/s^2",
        f"- SPARC kinematics galaxies: {len(kin)}",
        f"- Matched selected direct-method BH entries (all, including upper limits): {len(matched)}",
        f"- Direct detections (UPPERLIMIT=0): {len(strict)}",
        f"- Direct upper limits (UPPERLIMIT=1): {int(matched['is_upper_limit'].sum())}",
        "",
        "## Matched galaxy lists",
        "",
        "- Direct detections:",
    ]
    for galaxy in sorted(strict["galaxy"].unique().tolist()):
        lines.append(f"  - {galaxy}")
    lines += ["", "- Direct upper-limit-only additions (selected=1):"]
    for galaxy in sorted(matched[matched["is_upper_limit"]]["galaxy"].unique().tolist()):
        lines.append(f"  - {galaxy}")

    for label, subset_key in [
        ("Strict detections only", "direct_detected_only"),
        ("Sensitivity incl. upper limits as reported values", "direct_plus_upper_sensitivity"),
    ]:
        lines += ["", f"## {label}", ""]
        subset = summary["subsets"][subset_key]
        if "models" not in subset:
            lines.append(f"- N={subset['N']}; insufficient for fit")
            continue
        lines += [
            f"- N={subset['N']}",
            f"- Contains upper limits: {subset.get('contains_upper_limits', False)}",
            "",
            "| Model | OLS slope | OLS RMS | OLS MAD | Huber slope | Huber RMS | Huber MAD |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model in ["A_MBH_vs_Mdyn", "B_MBH_vs_xi"]:
            result = subset["models"][model]
            ols = result["OLS"]
            huber = result["Huber"]
            lines.append(
                f"| {model} | {ols['b']:.3f} | {ols['rms']:.3f} | {ols['mad']:.3f} | "
                f"{huber['b']:.3f} | {huber['rms']:.3f} | {huber['mad']:.3f} |"
            )

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "- The strict inference set is direct detections only (UPPERLIMIT=0).",
        "- The larger-N sensitivity block includes upper limits at tabulated values and is not a censored-likelihood fit.",
    ]
    (out_dir / "report_direct_bridge.md").write_text("\n".join(lines) + "\n")

    print(f"Output folder: {out_dir}")
    print(f"N strict detections: {len(strict)}")
    print(f"N extended (incl upper limits): {len(extended)}")
    print(f"N upper limits in extended: {int(matched['is_upper_limit'].sum())}")


if __name__ == "__main__":
    main()
