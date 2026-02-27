#!/usr/bin/env python3
"""
Phase 4: Unified Model Adjudication
===================================

Builds a single decision report across key diagnostics:
  1) Mass-split bunching trend (X = R/xi prediction)
  2) Forward-model per-galaxy fits (BEC vs NFW)
  3) Healing-length scaling (raw vs controlled)
  4) Phase-diagram discrimination around g_dagger
  5) Hierarchical galaxy-to-cluster consistency

Outputs:
  - analysis/results/summary_phase4_unified_adjudication.json
  - analysis/results/report_phase4_unified_adjudication.md
"""

import json
import os
from datetime import datetime, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results")
PIPELINE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "pipeline", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def source_entry(path):
    if not os.path.exists(path):
        return {"path": path, "exists": False, "mtime_utc": None}
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()
    return {"path": path, "exists": True, "mtime_utc": mtime}


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def evaluate_mass_split(summary):
    claim = "Mass-dependent bunching should strengthen toward high mass / low X"
    criteria = {
        "rho_logMs_vs_daic": "> 0",
        "rho_X_vs_daic": "< 0",
        "trend_p_max": 0.05,
        "inner_minus_outer_daic": "> 0",
    }
    if not summary:
        return {
            "claim": claim,
            "status": "missing",
            "criteria": criteria,
            "metrics": {},
            "rationale": ["summary_mass_split_bunching.json missing"],
        }

    mt = summary.get("mass_trend", {})
    rt = summary.get("radial_inner_outer_test", {})
    rho_m = safe_float(mt.get("spearman_rho_logMs_vs_daic"))
    p_m = safe_float(mt.get("spearman_p_logMs_vs_daic"))
    rho_x = safe_float(mt.get("spearman_rho_X_vs_daic"))
    p_x = safe_float(mt.get("spearman_p_X_vs_daic"))
    inner_minus_outer = safe_float(rt.get("inner_minus_outer_delta_aic"))

    trend_support = (
        rho_m is not None and p_m is not None and
        rho_x is not None and p_x is not None and
        rho_m > 0 and p_m < 0.05 and rho_x < 0 and p_x < 0.05
    )
    radial_support = (inner_minus_outer is not None and inner_minus_outer > 0)

    if trend_support and radial_support:
        status = "supported"
    elif (
        (rho_m is not None and rho_m < 0) or
        (rho_x is not None and rho_x > 0) or
        (inner_minus_outer is not None and inner_minus_outer < 0)
    ):
        status = "rejected"
    else:
        status = "weak"

    rationale = [
        f"rho(logMs,ΔAIC)={rho_m}, p={p_m}",
        f"rho(X,ΔAIC)={rho_x}, p={p_x}",
        f"inner_minus_outer_ΔAIC={inner_minus_outer}",
        f"script_overall={summary.get('overall_verdict')}",
    ]
    metrics = {
        "rho_logMs_vs_daic": rho_m,
        "p_logMs_vs_daic": p_m,
        "rho_X_vs_daic": rho_x,
        "p_X_vs_daic": p_x,
        "inner_minus_outer_daic": inner_minus_outer,
        "n_total_points": summary.get("n_total_points"),
        "n_total_galaxies": summary.get("n_total_galaxies"),
    }
    return {
        "claim": claim,
        "status": status,
        "criteria": criteria,
        "metrics": metrics,
        "rationale": rationale,
    }


def evaluate_forward_model(summary):
    claim = "Forward-model preference should become more BEC-like at higher stellar mass"
    criteria = {
        "rho_logMs_vs_daic": "> 0",
        "rho_Xmed_vs_daic": "< 0",
        "trend_p_max": 0.05,
    }
    if not summary:
        return {
            "claim": claim,
            "status": "missing",
            "criteria": criteria,
            "metrics": {},
            "rationale": ["summary_forward_model_bunching.json missing"],
        }

    sm = summary.get("spearman_logMs", {})
    sx = summary.get("spearman_Xmed", {})
    rho_m = safe_float(sm.get("rho"))
    p_m = safe_float(sm.get("p"))
    rho_x = safe_float(sx.get("rho"))
    p_x = safe_float(sx.get("p"))

    trend_support = (
        rho_m is not None and p_m is not None and
        rho_x is not None and p_x is not None and
        rho_m > 0 and p_m < 0.05 and rho_x < 0 and p_x < 0.05
    )
    significant_opposite = (
        (rho_m is not None and p_m is not None and rho_m < 0 and p_m < 0.05) or
        (rho_x is not None and p_x is not None and rho_x > 0 and p_x < 0.05)
    )

    if trend_support:
        status = "supported"
    elif significant_opposite or summary.get("overall_verdict") == "NFW-FAVORED":
        status = "rejected"
    else:
        status = "weak"

    rationale = [
        f"rho(logMs,ΔAIC)={rho_m}, p={p_m}",
        f"rho(X_med,ΔAIC)={rho_x}, p={p_x}",
        f"overall_verdict={summary.get('overall_verdict')}",
        f"mass_trend={summary.get('mass_trend')}",
    ]
    metrics = {
        "rho_logMs_vs_daic": rho_m,
        "p_logMs_vs_daic": p_m,
        "rho_Xmed_vs_daic": rho_x,
        "p_Xmed_vs_daic": p_x,
        "mean_daic": safe_float(summary.get("mean_daic")),
        "median_daic": safe_float(summary.get("median_daic")),
        "sum_daic": safe_float(summary.get("sum_daic")),
        "n_galaxies": summary.get("n_galaxies"),
    }
    return {
        "claim": claim,
        "status": status,
        "criteria": criteria,
        "metrics": metrics,
        "rationale": rationale,
    }


def evaluate_healing_scaling(summary):
    claim = "Lc–xi scaling should remain significant after size controls"
    criteria = {
        "raw_Lc_vs_xi_p_max": 0.01,
        "multiple_regression_log_xi_p_max": 0.05,
        "normalized_Lc_over_Rext_vs_xi_over_Rext_p_max": 0.05,
        "all_effects_positive": True,
    }
    if not summary:
        return {
            "claim": claim,
            "status": "missing",
            "criteria": criteria,
            "metrics": {},
            "rationale": ["summary_healing_length_scaling.json missing"],
        }

    corr = summary.get("correlations", {})
    raw = corr.get("Lc_vs_xi", {})
    norm = corr.get("Lc_norm_vs_xi_norm", {})
    reg_simple = summary.get("regression_simple", {})
    reg_multi = summary.get("regression_multiple", {}) or {}
    multi_xi = reg_multi.get("coefficients", {}).get("log_xi", {})

    raw_rho = safe_float(raw.get("spearman_rho"))
    raw_p = safe_float(raw.get("spearman_p"))
    norm_rho = safe_float(norm.get("spearman_rho"))
    norm_p = safe_float(norm.get("spearman_p"))
    simple_slope = safe_float(reg_simple.get("slope"))
    multi_beta = safe_float(multi_xi.get("beta"))
    multi_p = safe_float(multi_xi.get("p"))

    support = (
        raw_p is not None and raw_p < 0.01 and
        multi_p is not None and multi_p < 0.05 and
        norm_p is not None and norm_p < 0.05 and
        (raw_rho is None or raw_rho > 0) and
        (norm_rho is None or norm_rho > 0) and
        (simple_slope is None or simple_slope > 0) and
        (multi_beta is None or multi_beta > 0)
    )

    raw_only = (
        raw_p is not None and raw_p < 0.01 and
        (raw_rho is None or raw_rho > 0)
    )

    if support:
        status = "supported"
    elif raw_only:
        status = "weak"
    else:
        status = "rejected"

    rationale = [
        f"raw: rho={raw_rho}, p={raw_p}",
        f"normalized: rho={norm_rho}, p={norm_p}",
        f"simple slope={simple_slope}",
        f"controlled log_xi beta={multi_beta}, p={multi_p}",
        f"script verdict={summary.get('verdict')}",
    ]
    metrics = {
        "raw_rho": raw_rho,
        "raw_p": raw_p,
        "normalized_rho": norm_rho,
        "normalized_p": norm_p,
        "simple_slope": simple_slope,
        "controlled_log_xi_beta": multi_beta,
        "controlled_log_xi_p": multi_p,
        "n_valid_Lc": summary.get("sample", {}).get("n_valid_Lc"),
    }
    return {
        "claim": claim,
        "status": status,
        "criteria": criteria,
        "metrics": metrics,
        "rationale": rationale,
    }


def evaluate_phase_diagram(summary):
    claim = "Phase-diagram peak near g_dagger should discriminate from null/LCDM and hold up in validation"
    criteria = {
        "verdict_contains": "DISCRIMINATING",
        "permutation_p_max": 0.05,
        "cv_delta_mean_min": 0.0,
        "edge_wins_frac_min": 0.5,
        "holdout_delta_mean_min": 0.0,
    }
    if not summary:
        return {
            "claim": claim,
            "status": "missing",
            "criteria": criteria,
            "metrics": {},
            "rationale": ["summary_phase_diagram_model.json missing"],
        }

    verdict = str(summary.get("verdict", ""))
    perm_p = safe_float(summary.get("permutation_test", {}).get("p_value"))
    cv_delta = safe_float(summary.get("cross_validation", {}).get("delta_mean"))
    cv_edge_wins = safe_float(summary.get("cross_validation", {}).get("edge_wins_frac"))
    holdout_delta = safe_float(summary.get("holdout_prediction", {}).get("delta_nll_per_pt_mean"))
    suite_pass = bool(summary.get("boolean_suite", {}).get("suite_pass", False))

    core_detect = ("DISCRIMINATING" in verdict.upper()) and (perm_p is not None and perm_p < 0.05)
    validation_ok = (
        cv_delta is not None and cv_delta >= 0 and
        cv_edge_wins is not None and cv_edge_wins >= 0.5 and
        holdout_delta is not None and holdout_delta >= 0
    )

    if core_detect and validation_ok and suite_pass:
        status = "supported"
    elif core_detect:
        status = "weak"
    else:
        status = "rejected"

    rationale = [
        f"verdict={verdict}",
        f"perm_p={perm_p}",
        f"cv_delta_mean={cv_delta}, edge_wins_frac={cv_edge_wins}",
        f"holdout_delta_mean={holdout_delta}",
        f"boolean_suite_pass={suite_pass}",
    ]
    metrics = {
        "verdict": verdict,
        "perm_p": perm_p,
        "cv_delta_mean": cv_delta,
        "cv_edge_wins_frac": cv_edge_wins,
        "holdout_delta_mean": holdout_delta,
        "boolean_suite_pass": suite_pass,
    }
    return {
        "claim": claim,
        "status": status,
        "criteria": criteria,
        "metrics": metrics,
        "rationale": rationale,
    }


def evaluate_hierarchical(summary):
    claim = "Galaxy-to-cluster scaling should be internally consistent with xi~sqrt(M) and g_dagger,eff mass trend"
    criteria = {
        "mass_scaling_alpha_p_max": 0.05,
        "hierarchical_dev_sigma_max": 2.0,
        "scatter_min_scale_factor_max": 2.0,
    }
    if not summary:
        return {
            "claim": claim,
            "status": "missing",
            "criteria": criteria,
            "metrics": {},
            "rationale": ["summary_hierarchical_healing_length.json missing"],
        }

    t3a = summary.get("test_3a_mass_scaling", {})
    t3c = summary.get("test_3c_hierarchical", {})
    s4 = summary.get("step4_scatter_profile", {})

    alpha = safe_float(t3a.get("combined_fit", {}).get("alpha"))
    alpha_p = safe_float(t3a.get("combined_fit", {}).get("pearson_p"))
    dev_sigma = safe_float(t3c.get("deviation_sigma"))
    closest_factor = safe_float(s4.get("closest_scale_factor"))

    support = (
        alpha is not None and alpha > 0 and
        alpha_p is not None and alpha_p < 0.05 and
        dev_sigma is not None and dev_sigma <= 2.0 and
        closest_factor is not None and closest_factor <= 2.0
    )
    partial = (
        alpha is not None and alpha > 0 and
        alpha_p is not None and alpha_p < 0.05 and
        dev_sigma is not None and dev_sigma <= 2.0
    )

    if support:
        status = "supported"
    elif partial:
        status = "weak"
    else:
        status = "rejected"

    rationale = [
        f"alpha={alpha}, p={alpha_p}",
        f"hierarchical_dev_sigma={dev_sigma}",
        f"closest_scale={s4.get('closest_scale_to_scatter_min')}, factor={closest_factor}",
        f"test_3c_verdict={t3c.get('verdict')}",
    ]
    metrics = {
        "alpha": alpha,
        "alpha_p": alpha_p,
        "hierarchical_dev_sigma": dev_sigma,
        "closest_scale_to_scatter_min": s4.get("closest_scale_to_scatter_min"),
        "closest_scale_factor": closest_factor,
    }
    return {
        "claim": claim,
        "status": status,
        "criteria": criteria,
        "metrics": metrics,
        "rationale": rationale,
    }


def render_markdown(payload):
    lines = []
    lines.append("# Phase 4 Unified Model Adjudication")
    lines.append("")
    lines.append(f"Generated (UTC): {payload['generated_utc']}")
    lines.append("")
    lines.append("## Claim Decisions")
    lines.append("")
    lines.append("| Claim | Status | Key metrics |")
    lines.append("|---|---|---|")
    for c in payload["claims"]:
        m = c.get("metrics", {})
        keybits = []
        for key in ["rho_logMs_vs_daic", "rho_X_vs_daic", "raw_p", "controlled_log_xi_p", "perm_p", "alpha_p"]:
            if key in m and m.get(key) is not None:
                keybits.append(f"{key}={m.get(key)}")
        metric_text = "; ".join(keybits) if keybits else "see JSON"
        lines.append(f"| {c['claim']} | **{c['status']}** | {metric_text} |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Supported: {payload['status_counts'].get('supported', 0)}")
    lines.append(f"- Weak: {payload['status_counts'].get('weak', 0)}")
    lines.append(f"- Rejected: {payload['status_counts'].get('rejected', 0)}")
    lines.append(f"- Missing: {payload['status_counts'].get('missing', 0)}")
    lines.append(f"- Overall: **{payload['overall_conclusion']}**")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    for name, src in payload["sources"].items():
        lines.append(f"- `{name}`: `{src['path']}` (exists={src['exists']}, mtime_utc={src['mtime_utc']})")
    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 72)
    print("PHASE 4 UNIFIED MODEL ADJUDICATION")
    print("=" * 72)

    source_map = {
        "mass_split_bunching": os.path.join(RESULTS_DIR, "summary_mass_split_bunching.json"),
        "forward_model_bunching": os.path.join(RESULTS_DIR, "summary_forward_model_bunching.json"),
        "healing_length_scaling": os.path.join(RESULTS_DIR, "summary_healing_length_scaling.json"),
        "hierarchical_healing_length": os.path.join(RESULTS_DIR, "summary_hierarchical_healing_length.json"),
        "phase_diagram_model": os.path.join(PIPELINE_RESULTS_DIR, "summary_phase_diagram_model.json"),
    }

    summaries = {name: load_json(path) for name, path in source_map.items()}
    sources = {name: source_entry(path) for name, path in source_map.items()}

    claims = [
        evaluate_mass_split(summaries["mass_split_bunching"]),
        evaluate_forward_model(summaries["forward_model_bunching"]),
        evaluate_healing_scaling(summaries["healing_length_scaling"]),
        evaluate_phase_diagram(summaries["phase_diagram_model"]),
        evaluate_hierarchical(summaries["hierarchical_healing_length"]),
    ]

    status_counts = {"supported": 0, "weak": 0, "rejected": 0, "missing": 0}
    for c in claims:
        s = c.get("status", "missing")
        status_counts[s] = status_counts.get(s, 0) + 1

    if status_counts["missing"] > 0:
        overall = "INCOMPLETE_EVIDENCE"
    elif status_counts["rejected"] >= 2:
        overall = "NOT_SUPPORTED_BY_COMBINED_TESTS"
    elif status_counts["supported"] >= 3 and status_counts["rejected"] == 0:
        overall = "PROVISIONALLY_SUPPORTED"
    else:
        overall = "MIXED_OR_INCONCLUSIVE"

    notes = []
    phase_src = sources["phase_diagram_model"]
    if phase_src["exists"]:
        notes.append(
            "Phase-diagram summary is read from analysis/pipeline/results; "
            "if a new long-running phase-diagram job is active, rerun this script after it completes."
        )
    if not sources["mass_split_bunching"]["exists"]:
        notes.append("Mass-split summary missing: run test_mass_split_bunching.py first.")

    payload = {
        "test_name": "phase4_unified_model_adjudication",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_conclusion": overall,
        "status_counts": status_counts,
        "claims": claims,
        "sources": sources,
        "notes": notes,
    }

    out_json = os.path.join(RESULTS_DIR, "summary_phase4_unified_adjudication.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    out_md = os.path.join(RESULTS_DIR, "report_phase4_unified_adjudication.md")
    with open(out_md, "w") as f:
        f.write(render_markdown(payload))

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Overall: {overall}")
    print("=" * 72)


if __name__ == "__main__":
    main()

