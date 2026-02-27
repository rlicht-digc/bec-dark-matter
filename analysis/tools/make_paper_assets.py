#!/usr/bin/env python3
"""
make_paper_assets.py — Read refereeproof run folder and generate
paper-ready LaTeX table snippets, value macros, copy figures, and write a
submission checklist sourced from summary.json.

Usage:
  python3 analysis/tools/make_paper_assets.py [run_folder]

If run_folder is omitted, the newest outputs/gdagger_hunt/*_refereeproof
folder is used.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from scipy.stats import beta as scipy_beta
except Exception:
    scipy_beta = None

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
TABLE_DIR = os.path.join(PAPER_DIR, "tables")
FIG_DIR = os.path.join(PAPER_DIR, "figures")


def load_summary(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(val: Any, decimals: int = 4) -> str:
    """Format a value for LaTeX text/math cells."""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    if isinstance(val, int):
        return str(val)
    return str(val)


def tex_escape_text(text: Any) -> str:
    """Minimal LaTeX escaping for plain-text table cells."""
    s = str(text)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    return s


def find_latest_refereeproof_run() -> str:
    pattern = os.path.join(PROJECT_ROOT, "outputs", "gdagger_hunt",
                           "*_refereeproof")
    runs = sorted(glob.glob(pattern))
    if not runs:
        raise FileNotFoundError(
            f"No refereeproof runs found at pattern: {pattern}"
        )
    return runs[-1]


def resolve_run_dir(run_arg: Optional[str]) -> str:
    if run_arg:
        run_dir = run_arg
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(PROJECT_ROOT, run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    return find_latest_refereeproof_run()


def _cp_upper_onesided_exact(k: int, n: int, alpha: float = 0.05) -> Optional[float]:
    """One-sided exact (Clopper-Pearson) binomial upper bound."""
    if n <= 0 or k < 0 or k > n:
        return None
    if k == n:
        return 1.0

    # Exact CP: Beta inverse quantile.
    if scipy_beta is not None:
        return float(scipy_beta.ppf(1.0 - alpha, k + 1, n - k))

    # For k=0, exact CP reduces to the closed form below.
    if k == 0:
        return float(1.0 - alpha ** (1.0 / n))

    # For k>0 without SciPy, bound omitted rather than approximated.
    return None


def _window_stats(d: Dict[str, Any], window: str) -> Tuple[int, int, float, Optional[float]]:
    """Return (N, hits, p, cp_ub) for a proximity window suffix."""
    n = int(d.get("n_shuffles", 0))

    if window == "0p10":
        k = d.get("n_hits_0p10", d.get("n_hits", None))
        p = d.get("p_within_0p10_dex", d.get("p_within_0p1_dex", None))
    elif window == "0p05":
        k = d.get("n_hits_0p05", None)
        p = d.get("p_within_0p05_dex", None)
    else:
        raise ValueError(f"Unknown window suffix: {window}")

    # Fallback for older summaries lacking explicit hit counts.
    if k is None and p is not None and n > 0:
        k = int(round(float(p) * n))
    if p is None and k is not None and n > 0:
        p = float(k) / float(n)

    if k is None:
        k = -1
    if p is None:
        p = float("nan")

    ub = _cp_upper_onesided_exact(int(k), n, alpha=0.05)
    return n, int(k), float(p), ub


def _hits_prop_cell(k: int, n: int, p: float) -> str:
    if k < 0 or n <= 0 or p != p:  # NaN check for p
        return "N/A"
    return f"{k}/{n} ({p:.4f})"


def _cp_cell(k: int, ub: Optional[float]) -> str:
    if k == 0 and ub is not None:
        return f"{ub:.4f}"
    return "---"


def make_table_baseline(s: Dict[str, Any]) -> str:
    """Table 1: Baseline SPARC RAR results."""
    b = s.get("baseline", {})
    scan = b.get("scale_scan", {})
    best = b.get("best_kernel", {})

    best_name = tex_escape_text(best.get("kernel_name", "N/A"))
    best_log = float(best.get("log_scale_best", 0.0))
    d_from = abs(best_log - (-9.9208))

    lines = [
        r"\begin{deluxetable}{lc}",
        r"\tablecaption{Baseline SPARC RAR kernel+scale identification"
        r"\label{tab:baseline}}",
        r"\tablehead{\colhead{Metric} & \colhead{Value}}",
        r"\startdata",
        f"Best kernel & {best_name} \\\\",
        f"$N_{{\\rm data}}$ & {b.get('n_data', 'N/A')} \\\\",
        f"Best $\\log_{{10}} s$ & ${fmt(best_log)}$ \\\\",
        f"$\\Delta$ from $\\log_{{10}} g^\\dagger$ & "
        f"${fmt(d_from, 4)}$ dex \\\\",
        f"$\\Delta$AIC/N at $\\pm 0.1$ dex & "
        f"${fmt(scan.get('delta_aic_pm_0p1_per_dof', {}).get('mean_0p1', 0))}$ \\\\",
        f"Peak sharpness ($d^2$AIC/$d(\\log s)^2$) & "
        f"${scan.get('peak_sharpness', 0):.0f}$ \\\\",
        f"AIC & ${fmt(best.get('aic', 0), 1)}$ \\\\",
        f"CV RMSE & ${fmt(best.get('cv_rmse', 0))}$ \\\\",
        r"\enddata",
        r"\end{deluxetable}",
    ]
    return "\n".join(lines)


def make_table_nulls(s: Dict[str, Any]) -> str:
    """Table 2: Null tests + control with dual-window hit/proportion stats."""
    sa = s.get("suite_a", {})

    lines = [
        r"\begin{deluxetable}{llcccc}",
        r"\tablecaption{Permutation procedures: dual-window proximity statistics around $g^\dagger$"
        r"\label{tab:nulls}}",
        r"\tablehead{",
        r"  \colhead{Test} &",
        r"  \colhead{Type} &",
        r"  \colhead{$N$} &",
        r"  \colhead{Hits/$N$ ($p_{\pm 0.10}$)} &",
        r"  \colhead{Hits/$N$ ($p_{\pm 0.05}$)} &",
        r"  \colhead{95\% CP UB ($\pm 0.10 / \pm 0.05$)}",
        r"}",
        r"\startdata",
    ]

    for key, label, test_type in [
        ("A1_global", "Global shuffle", "Null"),
        ("A2b_block_permute_bins", "Block-permute bins (0.5 dex)", "Null"),
        ("A3_within_galaxy", "Galaxy circular shift", "Null"),
        ("A2_within_bin", "Within-bin shuffle (0.5 dex)", "Control"),
    ]:
        d = sa.get(key, {})
        n10, k10, p10, ub10 = _window_stats(d, "0p10")
        n05, k05, p05, ub05 = _window_stats(d, "0p05")
        n = n10 if n10 > 0 else n05

        c10 = _hits_prop_cell(k10, n10, p10)
        c05 = _hits_prop_cell(k05, n05, p05)
        cp10 = _cp_cell(k10, ub10)
        cp05 = _cp_cell(k05, ub05)
        cp = f"{cp10} / {cp05}" if (cp10 != "---" or cp05 != "---") else "---"

        lines.append(
            f"{label} & {test_type} & {n} & {c10} & {c05} & {cp} \\\\")

    lines.extend([
        r"\enddata",
        r"\tablecomments{"
        r"For each permutation procedure, Hits/$N$ gives the count and empirical fraction of realizations with best scale within the stated window of $\log_{10}g^\dagger$. "
        r"Rows marked Null are destructive tests expected to drive proximity rates toward zero. "
        r"The block-permute bins null preserves within-bin heteroskedasticity while disrupting the cross-bin monotonic mapping, so intermediate behavior is expected. "
        r"The within-bin shuffle row is a structure-preserving Control row: it approximately preserves within-bin conditional structure $p(y\mid x)$ at the chosen bin width, so $p\approx 1$ is expected up to finite-binning and refit variability. "
        r"Clopper--Pearson (CP) one-sided 95\% upper bounds are reported only for windows with $k=0$ hits; otherwise CP entries are shown as ---. "
        r"CP bounds use the exact binomial inverse-beta form $p_{\rm UB}=\mathrm{BetaInv}(1-\alpha; k+1, n-k)$, which for $k=0$ reduces to $1-\alpha^{1/n}$.",
        r"}",
        r"\end{deluxetable}",
    ])
    return "\n".join(lines)


def make_table_cv(s: Dict[str, Any]) -> str:
    """Table 3: GroupKFold results."""
    sb = s.get("suite_b", {})

    lines = [
        r"\begin{deluxetable}{cccc}",
        r"\tablecaption{Galaxy-grouped 5-fold cross-validation"
        r"\label{tab:groupcv}}",
        r"\tablehead{",
        r"  \colhead{Fold} &",
        r"  \colhead{Best kernel} &",
        r"  \colhead{$\log_{10} s$} &",
        r"  \colhead{OOS RMSE}",
        r"}",
        r"\startdata",
    ]
    for fd in sb.get("fold_details", []):
        lines.append(
            f"{fd['fold']} & {tex_escape_text(fd['best_kernel'])} & "
            f"${fmt(fd['best_log_scale'])}$ & "
            f"${fmt(fd['test_rms'])}$ \\\\")
    lines.extend([
        r"\hline",
        f"Mean & --- & ${fmt(sb.get('best_log_scale_mean', 0))}$ "
        f"$\\pm$ ${fmt(sb.get('best_log_scale_std', 0))}$ & "
        f"${fmt(sb.get('oos_rms_mean', 0))}$ \\\\",
        r"\enddata",
        r"\tablecomments{Group = galaxy. No galaxy appears in both "
        r"train and test within any fold.}",
        r"\end{deluxetable}",
    ])
    return "\n".join(lines)


def make_table_negative(s: Dict[str, Any]) -> str:
    """Table 4: Negative control results."""
    se = s.get("suite_e", {})

    lines = [
        r"\begin{deluxetable}{lccc}",
        r"\tablecaption{Negative controls (break tests): signal "
        r"degradation under destructive perturbations"
        r"\label{tab:negative}}",
        r"\tablehead{",
        r"  \colhead{Perturbation} &",
        r"  \colhead{Sharpness / baseline} &",
        r"  \colhead{$|\Delta \log_{10} s - \log_{10} g^\dagger|$} &",
        r"  \colhead{$\Delta$AIC/N}",
        r"}",
        r"\startdata",
    ]
    for key, label in [
        ("E1_noise_sigma_0.1", r"Noise $\sigma=0.1$ dex"),
        ("E1_noise_sigma_0.3", r"Noise $\sigma=0.3$ dex"),
        ("E2_warp_alpha_0.7", r"Warp $\alpha=0.7$"),
        ("E2_warp_alpha_1.3", r"Warp $\alpha=1.3$"),
        ("E3_galaxy_swap", "Galaxy label swap"),
    ]:
        d = se.get(key, {})
        sr = d.get("sharpness_ratio", 0)
        dg = d.get("delta_from_gdagger", 0)
        an = d.get("aic_per_n_0p1", 0)
        lines.append(f"{label} & ${fmt(sr, 3)}$ & ${fmt(dg, 3)}$ & "
                     f"${fmt(an, 4)}$ \\\\")

    lines.extend([
        r"\enddata",
        r"\tablecomments{Warping the acceleration variable completely "
        r"destroys the optimum (sharpness $\to 0$, scale drifts "
        r"$>1$~dex). Galaxy label swaps reduce sharpness to "
        r"$\sim 0.23\times$ baseline.}",
        r"\end{deluxetable}",
    ])
    return "\n".join(lines)


def make_table_nearby(s: Dict[str, Any]) -> str:
    """Table 5: Nearby-scale comparison."""
    sf = s.get("suite_f", {})

    lines = [
        r"\begin{deluxetable}{lccc}",
        r"\tablecaption{Nearby-scale comparison: $g^\dagger$ vs.\ "
        r"characteristic acceleration scales\label{tab:nearby}}",
        r"\tablehead{",
        r"  \colhead{Scale} &",
        r"  \colhead{$\log_{10}(s)$} &",
        r"  \colhead{AIC ($\eta=1$)} &",
        r"  \colhead{AIC ($\eta$ free)}",
        r"}",
        r"\startdata",
    ]
    f1 = sf.get("F1_eta_fixed", {})
    f2 = sf.get("F2_eta_free", {})
    for name in ["g_dagger", "cH0_over_6", "a_Lambda", "cH0", "cH0_over_2pi"]:
        d1 = f1.get(name, {})
        d2 = f2.get(name, {})
        if not d1:
            continue
        nice_name = {
            "g_dagger": r"$g^\dagger$",
            "a_Lambda": r"$a_\Lambda$",
            "cH0": r"$cH_0$",
            "cH0_over_6": r"$cH_0/6$",
            "cH0_over_2pi": r"$cH_0/(2\pi)$",
        }.get(name, tex_escape_text(name))
        lines.append(
            f"{nice_name} & ${fmt(d1.get('log_scale', 0), 3)}$ & "
            f"${fmt(d1.get('aic', 0), 1)}$ & "
            f"${fmt(d2.get('aic', 0), 1)}$ \\\\")

    lines.extend([
        r"\enddata",
        r"\tablecomments{$\eta=1$: diagnostic direct substitution with scale fixed (no optimization) and likelihood evaluated directly. "
        r"$\eta$ free: $y = \eta \cdot K(x, s)$ with one free parameter per model (matched DoF). "
        r"AIC values in $\eta=1$ mode are not directly comparable to the $\eta$-free block because the parameter count differs; model-selection statements use matched-DoF comparisons.}",
        r"\end{deluxetable}",
    ])
    return "\n".join(lines)


def make_repro_appendix(s: Dict[str, Any], run_dir: str) -> str:
    """Reproducibility appendix snippet."""
    repro = s.get("reproducibility", {})
    hashes = s.get("sha256_hashes", {})

    lines = [
        r"\subsection{Reproducibility}\label{app:repro}",
        r"",
        r"\begin{itemize}",
        f"  \\item Git commit: \\texttt{{{tex_escape_text(repro.get('git_commit', 'N/A'))}}}",
        f"  \\item RNG seed: {s.get('seed', 42)}",
        f"  \\item Shuffle count: {s.get('n_shuffles', 'N/A')}",
        f"  \\item Run folder: \\texttt{{{tex_escape_text(os.path.basename(run_dir))}}}",
        r"\end{itemize}",
        r"",
        r"SHA-256 hashes of primary outputs:",
        r"\begin{verbatim}",
    ]
    for fname, h in hashes.items():
        lines.append(f"  {fname}: {h[:32]}...")
    lines.extend([
        r"\end{verbatim}",
        r"",
        r"Environment: see \texttt{pip\_freeze.txt} in the run folder.",
    ])
    return "\n".join(lines)


def _fraction_str(k: int, n: int) -> str:
    if k < 0 or n <= 0:
        return "N/A"
    return f"{k}/{n}"


def make_values_macros(s: Dict[str, Any], run_dir: str) -> str:
    """Emit manuscript values as TeX macros to prevent manual drift."""
    b = s.get("baseline", {})
    best = b.get("best_kernel", {})
    scan = b.get("scale_scan", {})
    sb = s.get("suite_b", {})
    sc = s.get("suite_c", {})
    sd = s.get("suite_d", {})
    sg = s.get("suite_g", {})
    sa = s.get("suite_a", {})
    fold_details = sb.get("fold_details", [])

    def m(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    best_log = float(best.get("log_scale_best", 0.0))
    delta = abs(best_log - (-9.9208))
    cv_total_folds = len(fold_details)
    cv_counts: Dict[str, int] = {}
    for fd in fold_details:
        kname = str(fd.get("best_kernel", "N/A"))
        cv_counts[kname] = cv_counts.get(kname, 0) + 1
    best_kernel_name = str(best.get("kernel_name", "N/A"))
    cv_best_kernel_count = cv_counts.get(best_kernel_name, 0)
    other = sorted(
        [(k, v) for k, v in cv_counts.items() if k != best_kernel_name],
        key=lambda kv: (-kv[1], kv[0]),
    )
    cv_alt_kernel_name = other[0][0] if other else "N/A"
    cv_alt_kernel_count = other[0][1] if other else 0

    lines = [
        "% Auto-generated by analysis/tools/make_paper_assets.py",
        m("ValRunFolder", tex_escape_text(os.path.basename(run_dir))),
        m("ValNData", str(int(b.get("n_data", 0)))),
        m("ValBestKernel", tex_escape_text(best.get("kernel_name", "N/A"))),
        m("ValBestLogScale", f"{best_log:.4f}"),
        m("ValDeltaFromGdag", f"{delta:.4f}"),
        m("ValDeltaAICPerN", f"{float(scan.get('delta_aic_pm_0p1_per_dof', {}).get('mean_0p1', 0.0)):.4f}"),
        m("ValPeakSharpness", f"{float(scan.get('peak_sharpness', 0.0)):.0f}"),
        m("ValCvMean", f"{float(sb.get('best_log_scale_mean', 0.0)):.4f}"),
        m("ValCvStd", f"{float(sb.get('best_log_scale_std', 0.0)):.4f}"),
        m("ValCvFoldsTotal", str(cv_total_folds)),
        m("ValCvBestKernelCount", str(cv_best_kernel_count)),
        m("ValCvAltKernel", tex_escape_text(cv_alt_kernel_name)),
        m("ValCvAltKernelCount", str(cv_alt_kernel_count)),
        m("ValCutMaxDelta", f"{float(sc.get('max_delta_log_scale', 0.0)):.4f}"),
        m("ValGridMaxDelta", f"{float(sd.get('max_delta_log_scale', 0.0)):.4f}"),
        m("ValClusterBestLogScale", f"{float(sg.get('best_log_scale', 0.0)):.4f}"),
        m("ValClusterDelta", f"{float(sg.get('delta_from_gdagger', 0.0)):.4f}"),
    ]

    for prefix, key in [
        ("AOne", "A1_global"),
        ("ATwoB", "A2b_block_permute_bins"),
        ("AThree", "A3_within_galaxy"),
        ("ATwoCtrl", "A2_within_bin"),
    ]:
        d = sa.get(key, {})
        n10, k10, p10, ub10 = _window_stats(d, "0p10")
        n05, k05, p05, ub05 = _window_stats(d, "0p05")
        n = n10 if n10 > 0 else n05

        lines.extend([
            m(f"Val{prefix}N", str(n)),
            m(f"Val{prefix}HitsTen", str(k10)),
            m(f"Val{prefix}HitsFive", str(k05)),
            m(f"Val{prefix}PTen", f"{p10:.4f}" if p10 == p10 else "N/A"),
            m(f"Val{prefix}PFive", f"{p05:.4f}" if p05 == p05 else "N/A"),
            m(f"Val{prefix}FracTen", _fraction_str(k10, n10)),
            m(f"Val{prefix}FracFive", _fraction_str(k05, n05)),
            m(
                f"Val{prefix}CPUpperTen",
                f"{ub10:.4f}" if (k10 == 0 and ub10 is not None) else "---",
            ),
            m(
                f"Val{prefix}CPUpperFive",
                f"{ub05:.4f}" if (k05 == 0 and ub05 is not None) else "---",
            ),
        ])

    return "\n".join(lines) + "\n"


def make_submission_checklist(s: Dict[str, Any], run_dir: str) -> str:
    """Rebuild SUBMISSION_CHECKLIST.md from summary.json values."""
    b = s.get("baseline", {})
    best = b.get("best_kernel", {})
    scan = b.get("scale_scan", {})
    sa = s.get("suite_a", {})
    sb = s.get("suite_b", {})
    sc = s.get("suite_c", {})
    sd = s.get("suite_d", {})
    sg = s.get("suite_g", {})
    fold_details = sb.get("fold_details", [])

    run_rel = os.path.relpath(run_dir, PROJECT_ROOT)

    def null_row(label: str, key: str, row_type: str) -> str:
        d = sa.get(key, {})
        n10, k10, p10, ub10 = _window_stats(d, "0p10")
        n05, k05, p05, ub05 = _window_stats(d, "0p05")

        cp10 = f"{ub10:.4f}" if (k10 == 0 and ub10 is not None) else "---"
        cp05 = f"{ub05:.4f}" if (k05 == 0 and ub05 is not None) else "---"
        cp = f"{cp10} / {cp05}" if (cp10 != "---" or cp05 != "---") else "---"

        return (
            f"| {label} | {row_type} | {k10}/{n10} ({p10:.4f}) | "
            f"{k05}/{n05} ({p05:.4f}) | {cp} |"
        )

    cv_counts: Dict[str, int] = {}
    for fd in fold_details:
        kname = str(fd.get("best_kernel", "N/A"))
        cv_counts[kname] = cv_counts.get(kname, 0) + 1
    cv_total = len(fold_details)
    cv_split = "; ".join(
        f"{k}: {v}/{cv_total}" for k, v in sorted(cv_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ) if cv_total > 0 else "N/A"

    lines = [
        "# Submission Checklist — g† Hunt Paper",
        "",
        "Mapping claims in the manuscript to source fields in the latest refereeproof summary.",
        "",
        f"**Run folder:** `{run_rel}`",
        "",
        "## Baseline SPARC (Table 1)",
        "",
        "| Metric | summary.json key | Value |",
        "|--------|------------------|-------|",
        f"| N data | `baseline.n_data` | {b.get('n_data', 'N/A')} |",
        f"| Best kernel | `baseline.best_kernel_name` | {best.get('kernel_name', 'N/A')} |",
        f"| Best log10(s) | `baseline.best_kernel.log_scale_best` | {float(best.get('log_scale_best', 0.0)):.4f} |",
        f"| Δ from log10(g†) | computed from key above | {abs(float(best.get('log_scale_best', 0.0)) - (-9.9208)):.4f} dex |",
        f"| ΔAIC/N at ±0.1 dex | `baseline.scale_scan.delta_aic_pm_0p1_per_dof.mean_0p1` | {float(scan.get('delta_aic_pm_0p1_per_dof', {}).get('mean_0p1', 0.0)):.4f} |",
        f"| Peak sharpness | `baseline.scale_scan.peak_sharpness` | {float(scan.get('peak_sharpness', 0.0)):.0f} |",
        "",
        "## Null/Control Dual-Window Stats (Table 2)",
        "",
        "| Test | Type | Hits/N (±0.10) | Hits/N (±0.05) | 95% CP UB (±0.10 / ±0.05, only when k=0) |",
        "|------|------|----------------|----------------|-------------------------------------------|",
        null_row("A1 Global shuffle", "A1_global", "Null"),
        null_row("A2b Block-permute bins", "A2b_block_permute_bins", "Null"),
        null_row("A3 Galaxy circular shift", "A3_within_galaxy", "Null"),
        null_row("A2 Within-bin shuffle", "A2_within_bin", "Control"),
        "",
        "## Validation Stability",
        "",
        "| Metric | summary.json key | Value |",
        "|--------|------------------|-------|",
        f"| GroupKFold mean log10(s) | `suite_b.best_log_scale_mean` | {float(sb.get('best_log_scale_mean', 0.0)):.4f} |",
        f"| GroupKFold std | `suite_b.best_log_scale_std` | {float(sb.get('best_log_scale_std', 0.0)):.4f} |",
        f"| Fold-wise best-kernel split | `suite_b.fold_details[*].best_kernel` | {cv_split} |",
        f"| Cut sensitivity max Δ | `suite_c.max_delta_log_scale` | {float(sc.get('max_delta_log_scale', 0.0)):.4f} dex |",
        f"| Grid sensitivity max Δ | `suite_d.max_delta_log_scale` | {float(sd.get('max_delta_log_scale', 0.0)):.4f} dex |",
        "",
        "## Cluster Pilot",
        "",
        "| Metric | summary.json key | Value |",
        "|--------|------------------|-------|",
        f"| Best log10(s) | `suite_g.best_log_scale` | {float(sg.get('best_log_scale', 0.0)):.4f} |",
        f"| Δ from g† | `suite_g.delta_from_gdagger` | {float(sg.get('delta_from_gdagger', 0.0)):.4f} dex |",
        "",
        "## Figure Asset Paths",
        "",
        "| Figure | Source in run folder | Paper destination |",
        "|--------|----------------------|-------------------|",
        "| Fig 1 | `figures/fig1_three_panel.png` | `paper/figures/fig1_three_panel.png` |",
        "| Fig 2 | `figures/fig2_validation_stability.png` | `paper/figures/fig2_validation_stability.png` |",
        "| Fig 3 | `figures/fig3_negative_controls.png` | `paper/figures/fig3_negative_controls.png` |",
        "",
        "## Reproducibility",
        "",
        "| Item | Location |",
        "|------|----------|",
        "| Git commit | `summary.json -> reproducibility.git_commit` |",
        "| RNG seed | `summary.json -> seed` |",
        "| SHA256 hashes | `summary.json -> sha256_hashes` |",
        "| Pip freeze | `<run_folder>/pip_freeze.txt` |",
        "| Asset command | `python3 analysis/tools/make_paper_assets.py` (omit arg to use newest refereeproof run) |",
        "",
        f"*Last updated: {dt.date.today().isoformat()}*",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    run_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    run_dir = resolve_run_dir(run_arg)

    print(f"Reading: {run_dir}")
    s = load_summary(run_dir)

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    tables = {
        "tab_baseline.tex": make_table_baseline(s),
        "tab_nulls.tex": make_table_nulls(s),
        "tab_groupcv.tex": make_table_cv(s),
        "tab_negative.tex": make_table_negative(s),
        "tab_nearby.tex": make_table_nearby(s),
        "appendix_repro.tex": make_repro_appendix(s, run_dir),
        "values_from_summary.tex": make_values_macros(s, run_dir),
    }

    for fname, content in tables.items():
        path = os.path.join(TABLE_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Wrote: {path}")

    # Copy figures used by manuscript.
    fig_map = {
        "fig1_three_panel.png": "fig1_three_panel.png",
        "fig2_validation_stability.png": "fig2_validation_stability.png",
        "fig3_negative_controls.png": "fig3_negative_controls.png",
    }
    src_fig_dir = os.path.join(run_dir, "figures")
    for src_name, dst_name in fig_map.items():
        src = os.path.join(src_fig_dir, src_name)
        dst = os.path.join(FIG_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {src_name} -> {dst}")
        else:
            print(f"  WARNING: missing figure: {src}")

    # Copy selected baseline figures from the baseline run when present.
    baseline_dir = s.get("baseline", {}).get("output_dir", "")
    if baseline_dir and os.path.isdir(os.path.join(baseline_dir, "figures")):
        for fname in ["data_overview.png", "scale_scan_aic.png", "kernel_comparison.png"]:
            src = os.path.join(baseline_dir, "figures", fname)
            if os.path.exists(src):
                dst = os.path.join(FIG_DIR, f"baseline_{fname}")
                shutil.copy2(src, dst)
                print(f"  Copied: baseline {fname}")

    checklist_path = os.path.join(PAPER_DIR, "SUBMISSION_CHECKLIST.md")
    with open(checklist_path, "w", encoding="utf-8") as f:
        f.write(make_submission_checklist(s, run_dir))
    print(f"  Wrote: {checklist_path}")

    print(f"\nDone. Paper assets in: {PAPER_DIR}")


if __name__ == "__main__":
    main()
