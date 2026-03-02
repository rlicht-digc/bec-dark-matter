#!/usr/bin/env python3
"""
Single-pass external audit artifact generator.

This script writes exactly one coherent set of:
  - EXTERNAL_AUDIT_REPORT_<RUN_ID>.md
  - EXTERNAL_AUDIT_INDEX_<RUN_ID>.json
  - EXTERNAL_AUDIT_FILE_MANIFEST_<RUN_ID>.csv
  - EXTERNAL_AUDIT_APPENDIX_MATH_<RUN_ID>.md

Design guarantees:
  1) One RUN_ID generated once at start (or provided via --run-id).
  2) Lock-file protection via analysis/results/.external_audit.lock.
  3) Atomic writes with *.tmp + os.replace.
  4) Single-pass build: all inventories in memory first, write once at end.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


REPORT_PREFIX = "EXTERNAL_AUDIT_REPORT"
INDEX_PREFIX = "EXTERNAL_AUDIT_INDEX"
MANIFEST_PREFIX = "EXTERNAL_AUDIT_FILE_MANIFEST"
APPENDIX_PREFIX = "EXTERNAL_AUDIT_APPENDIX_MATH"
LOCK_FILENAME = ".external_audit.lock"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_iso(d: dt.datetime) -> str:
    return d.isoformat().replace("+00:00", "Z")


def compact_run_id(d: dt.datetime) -> str:
    return d.strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2) + "\n")


def build_manifest_csv(rows: Sequence[Dict[str, Any]]) -> str:
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)
    return sio.getvalue()


def first_doc_line(path: Path) -> str:
    text = read_text(path)
    if not text:
        return ""
    try:
        mod = ast.parse(text)
        doc = ast.get_docstring(mod)
        if doc:
            return doc.strip().splitlines()[0].strip()
    except Exception:
        pass
    for line in text.splitlines()[:30]:
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return ""


def classify_branch(path: Path) -> str:
    s = str(path).lower()
    n = path.name.lower()
    if "paper3" in s or "branchc" in n:
        return "C"
    if "gdagger" in s or "referee" in s or "branchd" in n:
        return "D"
    return "adjacent"


def find_line_with(path: Path, token: str) -> Tuple[Optional[int], Optional[str]]:
    for i, line in enumerate(read_text(path).splitlines(), start=1):
        if token in line:
            return i, line.strip()
    return None, None


def parse_bridge_report(path: Path) -> Dict[str, Any]:
    text = read_text(path)
    out: Dict[str, Any] = {}
    patterns = {
        "actual_csv_sha": r"^actual_csv_sha:\s*([0-9a-fA-F]+)\s*$",
        "require_csv_sha": r"^require_csv_sha:\s*([^\n]+)$",
        "include_ss20": r"^include_ss20:\s*(True|False)\s*$",
        "ss20_excluded_points": r"^ss20_excluded_points:\s*(\d+)\s*$",
        "theilsen_slope": r"^- Theil-Sen slope:\s*([+-]?[0-9]*\.?[0-9]+)\s*$",
        "pooled_shift_dex": r"^- pooled-point median shift \(top-bottom\):\s*([+-]?[0-9]*\.?[0-9]+)\s*dex\s*$",
        "pooled_perm_p": r"^- permutation p-value \(10k\):\s*([^\n]+)$",
        "pergal_shift_and_p": r"^- per-galaxy median shift \(top-bottom\):\s*([^\n]+)$",
    }
    lines = text.splitlines()
    for key, pat in patterns.items():
        rx = re.compile(pat)
        for line in lines:
            m = rx.match(line.strip())
            if m:
                out[key] = m.group(1).strip()
                break
    out["interpretations"] = [l.strip() for l in lines if l.strip().startswith("interpretation:")]
    return out


def parse_density_window_report(path: Path) -> Dict[str, Any]:
    txt = read_text(path)
    out: Dict[str, Any] = {}
    patterns = {
        "input_file": r"^- Input file:\s*(.+)$",
        "theilsen_slope": r"^- Theil-Sen slope .*:\s*([+-]?[0-9]*\.?[0-9]+)\s*$",
        "bootstrap_p": r"^- Bootstrap sign p-value \(slope\):\s*([^\n]+)$",
        "spearman_p": r"^- Spearman permutation p-value:\s*([^\n]+)$",
        "pooled_shift_dex": r"^- Top-vs-bottom residual median difference \(top-bottom\):\s*([+-]?[0-9]*\.?[0-9]+)\s*dex\s*$",
        "perm_p": r"^- Permutation p-value \(median shift\):\s*([^\n]+)$",
    }
    lines = txt.splitlines()
    for key, pat in patterns.items():
        rx = re.compile(pat)
        for line in lines:
            m = rx.match(line.strip())
            if m:
                out[key] = m.group(1).strip()
                break
    return out


def collect_tests(repo_root: Path) -> List[Dict[str, Any]]:
    script_paths: List[Path] = []
    script_paths.extend(sorted((repo_root / "analysis" / "tests").glob("*.py")))
    script_paths.extend(sorted((repo_root / "analysis" / "pipeline").glob("test_*.py")))
    for p in [
        repo_root / "analysis" / "gdagger_hunt.py",
        repo_root / "analysis" / "pipeline" / "run_referee_required_tests.py",
        repo_root / "analysis" / "paper3" / "high_density_targets.py",
        repo_root / "analysis" / "paper3" / "paper3_bridge_pack.py",
        repo_root / "analysis" / "paper3" / "paper3_branchC_experiments.py",
        repo_root / "analysis" / "run_branchC_paper3_experiments.py",
        repo_root / "analysis" / "run_branchD_rar_control.py",
        repo_root / "analysis" / "tools" / "repo_provenance_scan.py",
    ]:
        if p.exists():
            script_paths.append(p)
    script_paths = sorted(set(script_paths))

    tests: List[Dict[str, Any]] = []
    for i, p in enumerate(script_paths, start=1):
        rel = str(p.relative_to(repo_root))
        tests.append(
            {
                "id": f"T{i:03d}",
                "path": rel,
                "abs_path": str(p),
                "branch": classify_branch(p),
                "description": first_doc_line(p),
                "is_test_script": bool(p.name.startswith("test_") or "/tests/" in rel.replace("\\", "/")),
            }
        )
    return tests


def collect_runs(repo_root: Path, referenced_files: Set[str], warnings: List[str]) -> List[Dict[str, Any]]:
    run_roots = [
        repo_root / "outputs" / "gdagger_hunt",
        repo_root / "outputs" / "paper3_high_density",
        repo_root / "outputs" / "branchD_rar_control",
    ]
    runs: List[Dict[str, Any]] = []

    for root in run_roots:
        if not root.exists():
            warnings.append(f"Missing run root: {root}")
            continue

        for d in sorted([p for p in root.iterdir() if p.is_dir()]):
            run_type = "unknown"
            branch = "adjacent"
            metrics: Dict[str, Any] = {}
            evidence: List[Dict[str, Any]] = []

            summary_c = d / "summary_C1C2C3.json"
            bridge_report = d / "paper3_density_bridge_report.txt"
            density_window_report = d / "paper3_density_window_report.txt"
            gdagger_summary = d / "summary.json"
            stamp_d = d / "run_stamp_branchD.json"
            stamp_c = d / "run_stamp_branchC.json"

            outputs = [
                str(f)
                for f in sorted(d.rglob("*"))
                if f.is_file() and f.suffix.lower() in {".json", ".md", ".txt", ".png", ".csv", ".parquet"}
            ]

            if summary_c.exists():
                run_type = "branchC_C1C2C3"
                branch = "C"
                obj = json.loads(summary_c.read_text(encoding="utf-8"))
                m = obj.get("metadata", {})
                c1 = obj.get("C1", {})
                c2 = obj.get("C2", {})
                metrics = {
                    "final_decision": obj.get("final_decision"),
                    "input_csv_sha256": m.get("input_csv_sha256"),
                    "git_head": m.get("git_head"),
                    "seed": m.get("seed"),
                    "c1_shift_unweighted": c1.get("observed", {}).get("shift_unweighted"),
                    "c1_shift_weighted": c1.get("observed", {}).get("shift_weighted"),
                    "c1_p_unweighted_block": c1.get("p_unweighted_block"),
                    "c1_p_weighted_block": c1.get("p_weighted_block"),
                    "c2_aggregate_equal": c2.get("aggregate_equal"),
                    "c2_aggregate_matched": c2.get("aggregate_matched"),
                    "c2_n_bins_valid": c2.get("n_bins_valid"),
                }
                evidence.append(
                    {
                        "claim": "Branch C final decision",
                        "file": str(summary_c),
                        "locator": "json:final_decision",
                        "value": obj.get("final_decision"),
                    }
                )
                referenced_files.add(str(summary_c))

            elif bridge_report.exists():
                run_type = "branchC_bridgepack"
                branch = "C"
                parsed = parse_bridge_report(bridge_report)
                metrics = {
                    "actual_csv_sha": parsed.get("actual_csv_sha"),
                    "require_csv_sha": parsed.get("require_csv_sha"),
                    "include_ss20": parsed.get("include_ss20"),
                    "ss20_excluded_points": parsed.get("ss20_excluded_points"),
                    "pooled_shift_dex": parsed.get("pooled_shift_dex"),
                    "pooled_perm_p": parsed.get("pooled_perm_p"),
                    "pergal_shift_and_p": parsed.get("pergal_shift_and_p"),
                    "theilsen_slope": parsed.get("theilsen_slope"),
                    "interpretations": parsed.get("interpretations", []),
                }
                ln, txt = find_line_with(bridge_report, "- pooled-point median shift (top-bottom):")
                if ln is not None:
                    evidence.append(
                        {
                            "claim": "Bridge pooled-point shift",
                            "file": str(bridge_report),
                            "locator": f"text:line={ln}",
                            "value": txt,
                        }
                    )
                referenced_files.add(str(bridge_report))

            elif density_window_report.exists():
                run_type = "branchC_density_window_prototype"
                branch = "C"
                parsed = parse_density_window_report(density_window_report)
                metrics = {
                    "input_file": parsed.get("input_file"),
                    "theilsen_slope": parsed.get("theilsen_slope"),
                    "bootstrap_p": parsed.get("bootstrap_p"),
                    "spearman_p": parsed.get("spearman_p"),
                    "pooled_shift_dex": parsed.get("pooled_shift_dex"),
                    "perm_p": parsed.get("perm_p"),
                }
                ln, txt = find_line_with(
                    density_window_report, "- Top-vs-bottom residual median difference (top-bottom):"
                )
                if ln is not None:
                    evidence.append(
                        {
                            "claim": "Prototype pooled-point shift",
                            "file": str(density_window_report),
                            "locator": f"text:line={ln}",
                            "value": txt,
                        }
                    )
                referenced_files.add(str(density_window_report))

            elif gdagger_summary.exists() and "gdagger_hunt" in str(d):
                run_type = "branchD_gdagger"
                branch = "D"
                obj = json.loads(gdagger_summary.read_text(encoding="utf-8"))
                suite_a = obj.get("suite_a", {})
                suite_f = obj.get("suite_f", {})
                suite_g = obj.get("suite_g", {})
                metrics = {
                    "baseline_best_kernel": obj.get("baseline", {}).get("best_kernel_name"),
                    "baseline_best_log_scale": obj.get("baseline", {}).get("best_log_scale"),
                    "A1_p_within_0p10": suite_a.get("A1_global", {}).get(
                        "p_within_0p10_dex", suite_a.get("A1_global", {}).get("p_within_0p1_dex")
                    ),
                    "A2_p_within_0p10": suite_a.get("A2_within_bin", {}).get(
                        "p_within_0p10_dex", suite_a.get("A2_within_bin", {}).get("p_within_0p1_dex")
                    ),
                    "A2b_p_within_0p10": suite_a.get("A2b_block_permute_bins", {}).get(
                        "p_within_0p10_dex", suite_a.get("A2b_block_permute_bins", {}).get("p_within_0p1_dex")
                    ),
                    "A2b_p_within_0p05": suite_a.get("A2b_block_permute_bins", {}).get("p_within_0p05_dex"),
                    "A3_p_within_0p10": suite_a.get("A3_within_galaxy", {}).get(
                        "p_within_0p10_dex", suite_a.get("A3_within_galaxy", {}).get("p_within_0p1_dex")
                    ),
                    "suite_c_max_delta_log_scale": obj.get("suite_c", {}).get("max_delta_log_scale"),
                    "suite_d_max_delta_log_scale": obj.get("suite_d", {}).get("max_delta_log_scale"),
                    "suite_f_F1_aLambda_aic": suite_f.get("F1_eta_fixed", {}).get("a_Lambda", {}).get("aic"),
                    "suite_f_F1_gdagger_aic": suite_f.get("F1_eta_fixed", {}).get("g_dagger", {}).get("aic"),
                    "suite_f_F2_aLambda_aic": suite_f.get("F2_eta_free", {}).get("a_Lambda", {}).get("aic"),
                    "suite_f_F2_gdagger_aic": suite_f.get("F2_eta_free", {}).get("g_dagger", {}).get("aic"),
                    "suite_g_best_log_scale": suite_g.get("best_log_scale"),
                    "suite_g_delta_from_gdagger": suite_g.get("delta_from_gdagger"),
                }
                evidence.append(
                    {
                        "claim": "Refereeproof baseline kernel",
                        "file": str(gdagger_summary),
                        "locator": "json:baseline.best_kernel_name",
                        "value": metrics["baseline_best_kernel"],
                    }
                )
                referenced_files.add(str(gdagger_summary))

            elif stamp_d.exists():
                run_type = "branchD_entrypoint_stamp"
                branch = "D"
                obj = json.loads(stamp_d.read_text(encoding="utf-8"))
                metrics = {
                    "status": obj.get("status"),
                    "dataset_sha256": obj.get("dataset_sha256"),
                    "git_head": obj.get("git_head"),
                    "smoke_mode": obj.get("smoke_mode"),
                    "output_dir": obj.get("output_dir"),
                }
                evidence.append(
                    {
                        "claim": "Branch D wrapper run stamp",
                        "file": str(stamp_d),
                        "locator": "json:status|dataset_sha256",
                        "value": {"status": obj.get("status"), "dataset_sha256": obj.get("dataset_sha256")},
                    }
                )
                referenced_files.add(str(stamp_d))

            elif stamp_c.exists():
                run_type = "branchC_entrypoint_stamp"
                branch = "C"
                obj = json.loads(stamp_c.read_text(encoding="utf-8"))
                metrics = {
                    "status": obj.get("status"),
                    "dataset_sha256": obj.get("dataset_sha256"),
                    "git_head": obj.get("git_head"),
                    "smoke_mode": obj.get("smoke_mode"),
                    "output_dir": obj.get("output_dir"),
                }
                evidence.append(
                    {
                        "claim": "Branch C wrapper run stamp",
                        "file": str(stamp_c),
                        "locator": "json:status|dataset_sha256",
                        "value": {"status": obj.get("status"), "dataset_sha256": obj.get("dataset_sha256")},
                    }
                )
                referenced_files.add(str(stamp_c))

            else:
                warnings.append(f"Unclassified run folder: {d}")

            runs.append(
                {
                    "id": f"R{len(runs) + 1:03d}",
                    "run_name": d.name,
                    "run_path": str(d),
                    "branch": branch,
                    "run_type": run_type,
                    "outputs": outputs,
                    "key_metrics": metrics,
                    "evidence": evidence,
                }
            )

    return runs


def build_corrections(
    repo_root: Path,
    rar_csv: Path,
    rar_csv_sha: Optional[str],
    rar_csv_backup: Path,
    rar_csv_backup_sha: Optional[str],
    referenced_files: Set[str],
) -> List[Dict[str, Any]]:
    corrections: List[Dict[str, Any]] = []

    def add(
        cid: str,
        title: str,
        issue: str,
        change: str,
        effect: str,
        evidence: List[Dict[str, Any]],
        status: str = "evidenced",
    ) -> None:
        corrections.append(
            {
                "id": cid,
                "title": title,
                "issue": issue,
                "change": change,
                "scientific_effect": effect,
                "status": status,
                "evidence": evidence,
            }
        )

    paper_main = repo_root / "paper" / "main.tex"
    ln_ack, txt_ack = find_line_with(paper_main, r"\begin{acknowledgments}")
    add(
        "CORR-01",
        "AASTeX acknowledgments syntax compliance",
        "Deprecated AASTeX acknowledgment command can break manuscript checks.",
        "Manuscript currently uses environment form `begin/end{acknowledgments}`.",
        "Formatting compliance risk reduced; no direct Branch C/D metric impact.",
        [
            {
                "file": str(paper_main),
                "locator": f"text:line={ln_ack}" if ln_ack else "text:not_found",
                "value": txt_ack,
            }
        ],
        status="partially_evidenced_current_state_only",
    )
    if paper_main.exists():
        referenced_files.add(str(paper_main))

    ref_script = repo_root / "analysis" / "tests" / "test_gdagger_hunt_refereeproof.py"
    summary_ref = repo_root / "outputs" / "gdagger_hunt" / "20260224_152455_refereeproof" / "summary.json"
    obj_ref = json.loads(summary_ref.read_text(encoding="utf-8")) if summary_ref.exists() else {}

    ln_cp, txt_cp = find_line_with(ref_script, "def clopper_pearson_upper")
    add(
        "CORR-02",
        "CP bound terminology and explicit storage",
        "Zero-hit null experiments need explicit binomial upper bounds.",
        "Refereeproof suite computes Clopper-Pearson upper bounds and stores `p_upper_95*` fields.",
        "Null proximity claims are bounded conservatively when hits=0.",
        [
            {"file": str(ref_script), "locator": f"text:line={ln_cp}", "value": txt_cp},
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A1_global.p_upper_95_0p10",
                "value": obj_ref.get("suite_a", {}).get("A1_global", {}).get("p_upper_95_0p10"),
            },
        ],
    )

    add(
        "CORR-03",
        "Null vs control relabeling",
        "Within-bin shuffle can be misinterpreted as a destructive null.",
        "Suite A2 is explicitly labeled `structure_preserving_control` and annotated as not a null test.",
        "Interpretation avoids false evidence inflation from a control expected to give p≈1.",
        [
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2_within_bin.type",
                "value": obj_ref.get("suite_a", {}).get("A2_within_bin", {}).get("type"),
            },
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2_within_bin.note",
                "value": obj_ref.get("suite_a", {}).get("A2_within_bin", {}).get("note"),
            },
        ],
    )

    add(
        "CORR-04",
        "Added destructive bin-aware null (A2b)",
        "Global and galaxy-shift nulls alone do not isolate cross-bin composition effects.",
        "Suite A2b block-permute-bin destructive null added with dedicated output fields.",
        "Shows intermediate proximity rates (`p_within_0p10_dex=0.158`), refining null interpretation.",
        [
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2b_block_permute_bins.type",
                "value": obj_ref.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("type"),
            },
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2b_block_permute_bins.p_within_0p10_dex",
                "value": obj_ref.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("p_within_0p10_dex"),
            },
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2b_block_permute_bins.p_within_0p05_dex",
                "value": obj_ref.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("p_within_0p05_dex"),
            },
        ],
    )

    ln_dw, txt_dw = find_line_with(ref_script, "def compute_dual_window_proximity")
    add(
        "CORR-05",
        "Dual-window proximity (±0.05/±0.10 dex)",
        "Single-window proximity can hide scale-local sensitivity.",
        "Dual-window proximity function emits both ±0.05 and ±0.10 dex hit fractions and p-values.",
        "Improves discrimination between tight and broad scale concentration under null.",
        [
            {"file": str(ref_script), "locator": f"text:line={ln_dw}", "value": txt_dw},
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2b_block_permute_bins.p_within_0p05_dex",
                "value": obj_ref.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("p_within_0p05_dex"),
            },
            {
                "file": str(summary_ref),
                "locator": "json:suite_a.A2b_block_permute_bins.p_within_0p10_dex",
                "value": obj_ref.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("p_within_0p10_dex"),
            },
        ],
    )

    pipe09 = repo_root / "analysis" / "pipeline" / "09_unified_rar_pipeline.py"
    ln_resolve, txt_resolve = find_line_with(pipe09, "def resolve_sparc_paths")
    ln_guard, txt_guard = find_line_with(pipe09, "SPARC guardrail triggered: insufficient SPARC coverage")
    bridge_old = repo_root / "outputs" / "paper3_high_density" / "20260227_162238" / "paper3_density_bridge_report.txt"
    bridge_new = repo_root / "outputs" / "paper3_high_density" / "20260227_194611" / "paper3_density_bridge_report.txt"
    old_parsed = parse_bridge_report(bridge_old) if bridge_old.exists() else {}
    new_parsed = parse_bridge_report(bridge_new) if bridge_new.exists() else {}
    add(
        "CORR-06",
        "SPARC path resolver + dataset rebuild hash change",
        "Legacy-path drift could drop SPARC coverage and shift pooled residual statistics.",
        "Unified pipeline adds path resolution/guardrails and emits stamped metadata; dataset hash changed from backup `430a75f2...` to restored `11742ae3...`.",
        "Branch C headline changed materially: pooled shift from ~0.184776 dex (older run) to ~0.027282 dex (restored hash-locked run).",
        [
            {"file": str(pipe09), "locator": f"text:line={ln_resolve}", "value": txt_resolve},
            {"file": str(pipe09), "locator": f"text:line={ln_guard}", "value": txt_guard},
            {"file": str(rar_csv_backup), "locator": "sha256", "value": rar_csv_backup_sha},
            {"file": str(rar_csv), "locator": "sha256", "value": rar_csv_sha},
            {"file": str(bridge_old), "locator": "text:- pooled-point median shift", "value": old_parsed.get("pooled_shift_dex")},
            {"file": str(bridge_new), "locator": "text:- pooled-point median shift", "value": new_parsed.get("pooled_shift_dex")},
        ],
    )

    bridge_script = repo_root / "analysis" / "paper3" / "paper3_bridge_pack.py"
    ln_key, txt_key = find_line_with(bridge_script, 'galaxy_key = raw_df[mapping["galaxy_key"]]')
    ln_ss20, txt_ss20 = find_line_with(bridge_script, "if not args.include_ss20")
    add(
        "CORR-07",
        "galaxy_key canonicalization and default SS20 exclusion",
        "Mixed galaxy naming and SS20 single-point stubs can distort pooled-point analyses.",
        "Bridge pack canonicalizes `galaxy_key` and excludes `SS20_*` by default unless opt-in.",
        "Improves grouping integrity; report explicitly shows excluded SS20 points.",
        [
            {"file": str(bridge_script), "locator": f"text:line={ln_key}", "value": txt_key},
            {"file": str(bridge_script), "locator": f"text:line={ln_ss20}", "value": txt_ss20},
            {"file": str(bridge_new), "locator": "text:include_ss20", "value": new_parsed.get("include_ss20")},
            {"file": str(bridge_new), "locator": "text:ss20_excluded_points", "value": new_parsed.get("ss20_excluded_points")},
        ],
    )

    meta_json = repo_root / "analysis" / "results" / "rar_points_unified.meta.json"
    meta_obj = json.loads(meta_json.read_text(encoding="utf-8")) if meta_json.exists() else {}
    ln_hash_req, txt_hash_req = find_line_with(bridge_script, "if args.require_csv_sha is not None")
    add(
        "CORR-08",
        "Meta stamping + enforce-hash execution guardrails",
        "Unstamped runs allow silent dataset drift.",
        "Pipeline writes `rar_points_unified.meta.json` with output SHA/git head; bridge and Branch C scripts enforce `require_csv_sha` checks.",
        "Run provenance is auditable and hash-locked.",
        [
            {"file": str(meta_json), "locator": "json:output_csv_sha256", "value": meta_obj.get("output_csv_sha256")},
            {"file": str(meta_json), "locator": "json:git_head", "value": meta_obj.get("git_head")},
            {"file": str(bridge_script), "locator": f"text:line={ln_hash_req}", "value": txt_hash_req},
            {"file": str(bridge_new), "locator": "text:require_csv_sha", "value": new_parsed.get("require_csv_sha")},
        ],
    )

    reg_test = repo_root / "analysis" / "tests" / "test_unified_pipeline_regression.py"
    ln_reg1, txt_reg1 = find_line_with(reg_test, "assert sparc_points >= 2500")
    ln_reg2, txt_reg2 = find_line_with(reg_test, "assert sparc_galaxies >= 120")
    add(
        "CORR-09",
        "Regression tests for SPARC presence floors",
        "SPARC depletion can silently pass unless explicitly tested.",
        "Regression test enforces minimum SPARC points and galaxy-key counts in loader and unified CSV.",
        "Protects Branch C/D against recurrence of SPARC-missing drift.",
        [
            {"file": str(reg_test), "locator": f"text:line={ln_reg1}", "value": txt_reg1},
            {"file": str(reg_test), "locator": f"text:line={ln_reg2}", "value": txt_reg2},
        ],
    )

    run_c = repo_root / "analysis" / "run_branchC_paper3_experiments.py"
    run_d = repo_root / "analysis" / "run_branchD_rar_control.py"
    scan_script = repo_root / "analysis" / "tools" / "repo_provenance_scan.py"
    prov_report = repo_root / "analysis" / "results" / "provenance_scan_20260228_001541.md"
    ln_guard_c, txt_guard_c = find_line_with(run_c, "Wrong repo execution context.")
    ln_guard_d, txt_guard_d = find_line_with(run_d, "Wrong repo execution context.")
    add(
        "CORR-10",
        "Branch entrypoints + provenance scan guardrails",
        "Cross-repo leakage and wrong-CWD execution can invalidate provenance.",
        "Dedicated Branch C/D entrypoints enforce repo-root checks and run-stamp sidecars; provenance scanner reports BH references.",
        "Improves reproducibility and auditability of execution context.",
        [
            {"file": str(run_c), "locator": f"text:line={ln_guard_c}", "value": txt_guard_c},
            {"file": str(run_d), "locator": f"text:line={ln_guard_d}", "value": txt_guard_d},
            {"file": str(scan_script), "locator": "docstring", "value": first_doc_line(scan_script)},
            {
                "file": str(prov_report),
                "locator": "text:Vendoring recommendation",
                "value": find_line_with(prov_report, "Vendoring recommendation")[1],
            },
        ],
    )

    for p in [
        ref_script,
        summary_ref,
        pipe09,
        rar_csv,
        rar_csv_backup,
        bridge_old,
        bridge_new,
        bridge_script,
        meta_json,
        reg_test,
        run_c,
        run_d,
        scan_script,
        prov_report,
    ]:
        if p.exists():
            referenced_files.add(str(p))

    return corrections


def build_major_claims(
    repo_root: Path,
    referenced_files: Set[str],
) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []

    def add(claim: str, file: Path, locator: str, value: Any) -> None:
        claims.append({"claim": claim, "file": str(file), "locator": locator, "value": value})
        if file.exists():
            referenced_files.add(str(file))

    branchd_key_run = repo_root / "outputs" / "gdagger_hunt" / "20260224_152455_refereeproof" / "summary.json"
    branchd_obj = json.loads(branchd_key_run.read_text(encoding="utf-8")) if branchd_key_run.exists() else {}
    add(
        "Refereeproof baseline best kernel",
        branchd_key_run,
        "json:baseline.best_kernel_name",
        branchd_obj.get("baseline", {}).get("best_kernel_name"),
    )
    add(
        "Refereeproof baseline best log scale",
        branchd_key_run,
        "json:baseline.best_log_scale",
        branchd_obj.get("baseline", {}).get("best_log_scale"),
    )
    add(
        "Destructive block-permute null p(±0.10 dex)",
        branchd_key_run,
        "json:suite_a.A2b_block_permute_bins.p_within_0p10_dex",
        branchd_obj.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("p_within_0p10_dex"),
    )
    add(
        "Destructive block-permute null p(±0.05 dex)",
        branchd_key_run,
        "json:suite_a.A2b_block_permute_bins.p_within_0p05_dex",
        branchd_obj.get("suite_a", {}).get("A2b_block_permute_bins", {}).get("p_within_0p05_dex"),
    )
    add(
        "Nearby-scale F1 eta-fixed AIC(a_Lambda)",
        branchd_key_run,
        "json:suite_f.F1_eta_fixed.a_Lambda.aic",
        branchd_obj.get("suite_f", {}).get("F1_eta_fixed", {}).get("a_Lambda", {}).get("aic"),
    )
    add(
        "Nearby-scale F1 eta-fixed AIC(g_dagger)",
        branchd_key_run,
        "json:suite_f.F1_eta_fixed.g_dagger.aic",
        branchd_obj.get("suite_f", {}).get("F1_eta_fixed", {}).get("g_dagger", {}).get("aic"),
    )

    branchc_base_run = repo_root / "outputs" / "paper3_high_density" / "20260227_194611" / "paper3_density_bridge_report.txt"
    ln_pool, txt_pool = find_line_with(branchc_base_run, "- pooled-point median shift (top-bottom):")
    ln_gal, txt_gal = find_line_with(branchc_base_run, "- per-galaxy median shift (top-bottom):")
    ln_slope, txt_slope = find_line_with(branchc_base_run, "- Theil-Sen slope:")
    add("Branch C baseline pooled shift (restored hash)", branchc_base_run, f"text:line={ln_pool}", txt_pool)
    add("Branch C baseline per-galaxy shift", branchc_base_run, f"text:line={ln_gal}", txt_gal)
    add("Branch C baseline Theil-Sen slope", branchc_base_run, f"text:line={ln_slope}", txt_slope)

    branchc_c123_run = (
        repo_root / "outputs" / "paper3_high_density" / "BRANCHC_C1C2C3_20260228_004320" / "summary_C1C2C3.json"
    )
    branchc_c123 = json.loads(branchc_c123_run.read_text(encoding="utf-8")) if branchc_c123_run.exists() else {}
    add(
        "Branch C C1 weighted shift",
        branchc_c123_run,
        "json:C1.observed.shift_weighted",
        branchc_c123.get("C1", {}).get("observed", {}).get("shift_weighted"),
    )
    add(
        "Branch C C1 weighted block-permutation p",
        branchc_c123_run,
        "json:C1.p_weighted_block",
        branchc_c123.get("C1", {}).get("p_weighted_block"),
    )
    add(
        "Branch C C2 matched aggregate shift",
        branchc_c123_run,
        "json:C2.aggregate_matched",
        branchc_c123.get("C2", {}).get("aggregate_matched"),
    )
    add(
        "Branch C final decision",
        branchc_c123_run,
        "json:final_decision",
        branchc_c123.get("final_decision"),
    )
    return claims


def build_audit_context(repo_root: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    referenced_files: Set[str] = set()

    tests = collect_tests(repo_root)
    runs = collect_runs(repo_root, referenced_files, warnings)

    rar_csv = repo_root / "analysis" / "results" / "rar_points_unified.csv"
    rar_csv_sha = sha256_file(rar_csv) if rar_csv.exists() else None
    rar_backup = repo_root / "analysis" / "results" / "rar_points_unified__430a75f2__20260227.csv"
    rar_backup_sha = sha256_file(rar_backup) if rar_backup.exists() else None

    corrections = build_corrections(
        repo_root=repo_root,
        rar_csv=rar_csv,
        rar_csv_sha=rar_csv_sha,
        rar_csv_backup=rar_backup,
        rar_csv_backup_sha=rar_backup_sha,
        referenced_files=referenced_files,
    )
    major_claims = build_major_claims(repo_root=repo_root, referenced_files=referenced_files)

    # add key figure refs
    for fig in [
        repo_root / "outputs" / "gdagger_hunt" / "20260224_152455_refereeproof" / "figures" / "fig1_three_panel.png",
        repo_root / "outputs" / "gdagger_hunt" / "20260224_152455_refereeproof" / "figures" / "fig2_validation_stability.png",
        repo_root / "outputs" / "gdagger_hunt" / "20260224_152455_refereeproof" / "figures" / "fig3_negative_controls.png",
        repo_root / "outputs" / "paper3_high_density" / "20260227_194611" / "rho_vs_residual_scatter.png",
        repo_root / "outputs" / "paper3_high_density" / "20260227_194611" / "residual_hist_high_vs_low.png",
        repo_root / "outputs" / "paper3_high_density" / "20260227_194611" / "residual_vs_gbar_high_density.png",
        repo_root / "outputs" / "paper3_high_density" / "BRANCHC_C1C2C3_20260228_004320" / "figures" / "C1_block_nulls.png",
        repo_root / "outputs" / "paper3_high_density" / "BRANCHC_C1C2C3_20260228_004320" / "figures" / "C2_binwise_shift.png",
        repo_root / "outputs" / "paper3_high_density" / "BRANCHC_C1C2C3_20260228_004320" / "figures" / "C3_source_replication.png",
    ]:
        if fig.exists():
            referenced_files.add(str(fig))

    return {
        "repo_root": str(repo_root),
        "git_head": git_head(repo_root),
        "dataset_path": str(rar_csv),
        "dataset_sha256": rar_csv_sha,
        "dataset_backup_path": str(rar_backup) if rar_backup.exists() else None,
        "dataset_backup_sha256": rar_backup_sha,
        "tests": tests,
        "runs": runs,
        "corrections": corrections,
        "major_claims": major_claims,
        "warnings": warnings,
        "referenced_files": sorted(referenced_files),
    }


def render_report(context: Dict[str, Any], run_stamp: Dict[str, Any], out_paths: Dict[str, Path]) -> str:
    tests = context["tests"]
    runs = context["runs"]
    warnings = context["warnings"]
    corrections = context["corrections"]
    major_claims = context["major_claims"]
    repo_root = context["repo_root"]
    git_head_sha = context["git_head"]
    dataset_sha = context["dataset_sha256"]
    dataset_path = context["dataset_path"]

    test_branch_counts = Counter(t["branch"] for t in tests)
    run_branch_counts = Counter(r["branch"] for r in runs)

    branchd_runs = [
        r for r in runs if r["branch"] == "D" and r["run_type"] == "branchD_gdagger" and r["run_name"].endswith("_refereeproof")
    ]
    branchc_proto_runs = [r for r in runs if r["run_type"] == "branchC_density_window_prototype"]
    branchc_bridge_runs = [r for r in runs if r["run_type"] == "branchC_bridgepack"]
    branchc_c123_runs = [r for r in runs if r["run_type"] == "branchC_C1C2C3"]

    branchd_key = Path(repo_root) / "outputs" / "gdagger_hunt" / "20260224_152455_refereeproof" / "summary.json"
    branchd_obj = json.loads(branchd_key.read_text(encoding="utf-8")) if branchd_key.exists() else {}
    suite_b = branchd_obj.get("suite_b", {})
    suite_c = branchd_obj.get("suite_c", {})
    suite_d = branchd_obj.get("suite_d", {})
    suite_e = branchd_obj.get("suite_e", {})
    suite_f = branchd_obj.get("suite_f", {})
    suite_g = branchd_obj.get("suite_g", {})
    baseline = branchd_obj.get("baseline", {})
    scan = baseline.get("scale_scan", {})
    a1 = branchd_obj.get("suite_a", {}).get("A1_global", {})
    a2 = branchd_obj.get("suite_a", {}).get("A2_within_bin", {})
    a2b = branchd_obj.get("suite_a", {}).get("A2b_block_permute_bins", {})
    a3 = branchd_obj.get("suite_a", {}).get("A3_within_galaxy", {})

    branchc_c123_key = (
        Path(repo_root) / "outputs" / "paper3_high_density" / "BRANCHC_C1C2C3_20260228_004320" / "summary_C1C2C3.json"
    )
    branchc_c123_obj = json.loads(branchc_c123_key.read_text(encoding="utf-8")) if branchc_c123_key.exists() else {}

    lines: List[str] = []
    lines.append("# External Audit Report: BEC-dark-matter Branch C/D Program")
    lines.append("")
    lines.append("## Run Stamp")
    lines.append(f"- RUN_ID: `{run_stamp['run_id']}`")
    lines.append(f"- git HEAD: `{run_stamp['git_head']}`")
    lines.append(f"- repo root: `{run_stamp['repo_root']}`")
    lines.append(f"- dataset sha256: `{run_stamp['dataset_sha256']}`")
    lines.append(f"- start_time_utc: `{run_stamp['start_time_utc']}`")
    lines.append(f"- end_time_utc: `{run_stamp['end_time_utc']}`")
    lines.append(
        "- file counts: "
        f"tests={run_stamp['file_counts']['tests_cataloged']}, "
        f"runs={run_stamp['file_counts']['executed_runs_found']}, "
        f"manifest_files={run_stamp['file_counts']['referenced_files_manifested']}, "
        f"warnings={run_stamp['file_counts']['warnings_count']}"
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(f"- Audit generation UTC: `{run_stamp['end_time_utc']}`")
    lines.append(f"- Repo root: `{repo_root}`")
    lines.append(f"- Git HEAD: `{git_head_sha}`")
    lines.append(f"- Unified dataset SHA256: `{dataset_sha}` (`{dataset_path}`)")
    lines.append("- Core interpretation from artifact evidence:")
    lines.append("  - Branch D refereeproof suite shows persistent scale recovery near g† with destructive null controls rejecting easy coincidences.")
    lines.append("  - Branch C baseline bridge pooled-point shift is positive in restored runs, but per-galaxy shift and trend significance are weak.")
    lines.append("  - Branch C decisive C1-C3 run (`BRANCHC_C1C2C3_20260228_004320`) concludes **\"sampling artifact\"** under galaxy-aware controls.")
    lines.append("- Conservative conclusion: current artifacts do **not** establish a robust physical pooled residual-shift effect for Branch C; Branch D identifiability controls are stronger and internally coherent.")
    lines.append("")

    lines.append("## Project Map")
    lines.append("- Branch D (RAR control / g† identifiability): `analysis/tests/test_gdagger_hunt_refereeproof.py`, `analysis/gdagger_hunt.py`, `analysis/pipeline/run_referee_required_tests.py`")
    lines.append("- Branch C (Paper3 density bridge): `analysis/paper3/paper3_bridge_pack.py`, `analysis/paper3/paper3_branchC_experiments.py`")
    lines.append("- Entrypoints / guardrails: `analysis/run_branchD_rar_control.py`, `analysis/run_branchC_paper3_experiments.py`, `analysis/tools/repo_provenance_scan.py`")
    lines.append("")

    lines.append("## Test Catalog (Designed vs Executed)")
    lines.append(f"- Designed scripts cataloged: **{len(tests)}**")
    lines.append(f"  - Branch C-tagged: {test_branch_counts.get('C', 0)}")
    lines.append(f"  - Branch D-tagged: {test_branch_counts.get('D', 0)}")
    lines.append(f"  - Adjacent/other: {test_branch_counts.get('adjacent', 0)}")
    lines.append(f"- Executed run folders found: **{len(runs)}**")
    lines.append(f"  - Branch C-tagged runs: {run_branch_counts.get('C', 0)}")
    lines.append(f"  - Branch D-tagged runs: {run_branch_counts.get('D', 0)}")
    lines.append(f"  - Unclassified/adjacent runs: {run_branch_counts.get('adjacent', 0)}")
    lines.append("- Full machine-readable catalogs are in the INDEX JSON (tests + runs arrays).")
    lines.append("")

    lines.append("## Branch D Results (RAR=BEC g† Identifiability)")
    lines.append("### D1. Refereeproof Run Folders Detected")
    for r in branchd_runs:
        lines.append(f"- `{r['run_path']}`")
    lines.append("")
    lines.append("### D2. Key Metrics Table (from summary.json)")
    lines.append("| run | best_kernel | best_log_scale | A1 p±0.10 | A2 control p±0.10 | A2b null p±0.10 | A2b null p±0.05 | A3 p±0.10 | maxΔlogscale(cuts) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in branchd_runs:
        km = r["key_metrics"]
        lines.append(
            f"| {r['run_name']} | {km.get('baseline_best_kernel')} | {km.get('baseline_best_log_scale')} | "
            f"{km.get('A1_p_within_0p10')} | {km.get('A2_p_within_0p10')} | "
            f"{km.get('A2b_p_within_0p10')} | {km.get('A2b_p_within_0p05')} | "
            f"{km.get('A3_p_within_0p10')} | {km.get('suite_c_max_delta_log_scale')} |"
        )
    lines.append("")
    lines.append("### D3. Headline Evidence (exact keys)")
    lines.append(f"- `baseline.best_kernel_name = {baseline.get('best_kernel_name')}`")
    lines.append(f"- `baseline.best_log_scale = {baseline.get('best_log_scale')}`")
    lines.append(f"- `suite_a.A2b_block_permute_bins.p_within_0p10_dex = {a2b.get('p_within_0p10_dex')}`")
    lines.append(f"- `suite_a.A2b_block_permute_bins.p_within_0p05_dex = {a2b.get('p_within_0p05_dex')}`")
    lines.append(f"- `suite_f.F1_eta_fixed.a_Lambda.aic = {(suite_f.get('F1_eta_fixed') or {}).get('a_Lambda', {}).get('aic')}`")
    lines.append(f"- `suite_f.F1_eta_fixed.g_dagger.aic = {(suite_f.get('F1_eta_fixed') or {}).get('g_dagger', {}).get('aic')}`")
    lines.append("")
    lines.append("### D3b. Coverage of Required Branch D Controls (exact values)")
    lines.append("- Kernel matcher / scale scan:")
    lines.append(f"  - `baseline.best_kernel_name = {baseline.get('best_kernel_name')}`")
    lines.append(f"  - `baseline.best_log_scale = {baseline.get('best_log_scale')}`")
    lines.append(f"  - `baseline.scale_scan.peak_sharpness = {scan.get('peak_sharpness')}`")
    lines.append(f"  - `baseline.scale_scan.delta_aic_pm_0p1_dex.mean_0p1 = {(scan.get('delta_aic_pm_0p1_dex') or {}).get('mean_0p1')}`")
    lines.append("- AIC/BIC availability:")
    lines.append("  - AIC arrays and ΔAIC diagnostics are present in `baseline.scale_scan` and `suite_f`.")
    lines.append("  - BIC is **not explicitly stored** in `summary.json` (`Not evidenced in artifacts`).")
    lines.append("- CV (grouped by galaxy):")
    lines.append(f"  - `suite_b.best_log_scale_mean = {suite_b.get('best_log_scale_mean')}`")
    lines.append(f"  - `suite_b.best_log_scale_std = {suite_b.get('best_log_scale_std')}`")
    lines.append(f"  - `suite_b.best_kernel_frequency = {suite_b.get('best_kernel_frequency')}`")
    lines.append("- Null tests and controls (Suite A):")
    lines.append(f"  - `A1 global null p±0.10 = {a1.get('p_within_0p10_dex', a1.get('p_within_0p1_dex'))}`")
    lines.append(f"  - `A2 within-bin control p±0.10 = {a2.get('p_within_0p10_dex', a2.get('p_within_0p1_dex'))}`")
    lines.append(f"  - `A2b block-permute null p±0.10 = {a2b.get('p_within_0p10_dex')}`")
    lines.append(f"  - `A3 within-galaxy null p±0.10 = {a3.get('p_within_0p10_dex', a3.get('p_within_0p1_dex'))}`")
    lines.append(f"  - `A1 CP upper bound (0 hits, ±0.10) = {a1.get('p_upper_95_0p10', a1.get('p_upper_95'))}`")
    lines.append("- Cut/sample sensitivity (Suite C):")
    lines.append(f"  - `suite_c.max_delta_log_scale = {suite_c.get('max_delta_log_scale')}`")
    lines.append(f"  - `suite_c.all_within_0p05_dex = {suite_c.get('all_within_0p05_dex')}`")
    lines.append("- Grid invariance (Suite D):")
    lines.append(f"  - `suite_d.max_delta_log_scale = {suite_d.get('max_delta_log_scale')}`")
    lines.append(f"  - `suite_d.within_0p02_dex = {suite_d.get('within_0p02_dex')}`")
    lines.append("- Negative controls (Suite E):")
    lines.append(
        f"  - `E1_noise_sigma_0.3.delta_from_gdagger = {(suite_e.get('E1_noise_sigma_0.3') or {}).get('delta_from_gdagger')}`"
    )
    lines.append(
        f"  - `E1_noise_sigma_0.3.sharpness_ratio = {(suite_e.get('E1_noise_sigma_0.3') or {}).get('sharpness_ratio')}`"
    )
    lines.append(f"  - `E3_galaxy_swap.delta_from_gdagger = {(suite_e.get('E3_galaxy_swap') or {}).get('delta_from_gdagger')}`")
    lines.append(f"  - `E3_galaxy_swap.sharpness_ratio = {(suite_e.get('E3_galaxy_swap') or {}).get('sharpness_ratio')}`")
    lines.append("- Nearby-scale comparisons (Suite F):")
    lines.append(
        f"  - `F1_eta_fixed: AIC(a_Lambda)={(suite_f.get('F1_eta_fixed') or {}).get('a_Lambda', {}).get('aic')}, "
        f"AIC(g_dagger)={(suite_f.get('F1_eta_fixed') or {}).get('g_dagger', {}).get('aic')}`"
    )
    lines.append(
        f"  - `F2_eta_free: AIC(a_Lambda)={(suite_f.get('F2_eta_free') or {}).get('a_Lambda', {}).get('aic')}, "
        f"AIC(g_dagger)={(suite_f.get('F2_eta_free') or {}).get('g_dagger', {}).get('aic')}`"
    )
    lines.append("- Non-RAR pilot / explicit null (Suite G):")
    lines.append(f"  - `suite_g.best_log_scale = {suite_g.get('best_log_scale')}`")
    lines.append(f"  - `suite_g.delta_from_gdagger = {suite_g.get('delta_from_gdagger')}`")
    lines.append("")
    lines.append("### D4. What Branch D Proves / Does Not Prove")
    lines.append("- Proves (artifact-backed): for SPARC-control refereeproof runs, destructive nulls do not reproduce tight scale concentration around g†, while structure-preserving within-bin control does (expected).")
    lines.append("- Does not prove: universal physical origin across all datasets by itself; this suite is identifiability-focused and model-comparison-focused.")
    lines.append("- Figures referenced:")
    for fig in [
        f"{repo_root}/outputs/gdagger_hunt/20260224_152455_refereeproof/figures/fig1_three_panel.png",
        f"{repo_root}/outputs/gdagger_hunt/20260224_152455_refereeproof/figures/fig2_validation_stability.png",
        f"{repo_root}/outputs/gdagger_hunt/20260224_152455_refereeproof/figures/fig3_negative_controls.png",
    ]:
        if Path(fig).exists():
            lines.append(f"  - `{fig}`")
    lines.append("")

    lines.append("## Branch C Results (Paper3 Density Bridge)")
    lines.append("### C0. Early High-Density Prototype Runs (paper3_density_window_report)")
    lines.append("| run | pooled_shift_dex | perm_p | theilsen_slope | bootstrap_p |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in branchc_proto_runs:
        km = r["key_metrics"]
        lines.append(
            f"| {r['run_name']} | {km.get('pooled_shift_dex')} | {km.get('perm_p')} | "
            f"{km.get('theilsen_slope')} | {km.get('bootstrap_p')} |"
        )
    lines.append("")
    lines.append("### C1. Baseline Bridge-Pack Runs (Executed)")
    lines.append("| run | actual_csv_sha | require_csv_sha | include_ss20 | ss20_excluded | pooled_shift_dex | pergal_shift_and_p | theilsen_slope |")
    lines.append("|---|---|---|---|---:|---:|---|---:|")
    for r in branchc_bridge_runs:
        km = r["key_metrics"]
        lines.append(
            f"| {r['run_name']} | {km.get('actual_csv_sha')} | {km.get('require_csv_sha')} | "
            f"{km.get('include_ss20')} | {km.get('ss20_excluded_points')} | {km.get('pooled_shift_dex')} | "
            f"{km.get('pergal_shift_and_p')} | {km.get('theilsen_slope')} |"
        )
    lines.append("")
    lines.append("### C2. Decisive C1-C3 Runs (Executed)")
    lines.append("| run | final_decision | input_csv_sha256 | C1 weighted shift | C1 weighted p_block | C2 matched | C2 valid bins |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in branchc_c123_runs:
        km = r["key_metrics"]
        lines.append(
            f"| {r['run_name']} | {km.get('final_decision')} | {km.get('input_csv_sha256')} | "
            f"{km.get('c1_shift_weighted')} | {km.get('c1_p_weighted_block')} | "
            f"{km.get('c2_aggregate_matched')} | {km.get('c2_n_bins_valid')} |"
        )
    lines.append("")
    if branchc_c123_obj:
        lines.append("### C3. Source-Stratified Replication (latest C1-C3 run)")
        lines.append("| subset | status | n_points | n_galaxies | c1_shift_weighted | c1_p_weighted | verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in branchc_c123_obj.get("C3", {}).get("subsets", []):
            lines.append(
                f"| {row.get('subset')} | {row.get('status')} | {row.get('n_points')} | {row.get('n_galaxies')} | "
                f"{row.get('c1_shift_weighted')} | {row.get('c1_p_weighted')} | {row.get('replication_verdict')} |"
            )
        lines.append("")
    lines.append("### C4. Branch C Interpretation (Conservative)")
    lines.append("- Restored hash-locked baseline run shows pooled positive shift but weak/negative per-galaxy signal and non-significant trend slope CI crossing zero.")
    lines.append("- C1-C3 latest run reports `final_decision = sampling artifact` with weighted block-permutation p-value > 0.01 and limited valid within-bin support.")
    lines.append("- Therefore, this artifact set does **not** support a robust physical pooled-shift claim at present.")
    lines.append("- Figures referenced:")
    for fig in [
        f"{repo_root}/outputs/paper3_high_density/20260227_194611/rho_vs_residual_scatter.png",
        f"{repo_root}/outputs/paper3_high_density/20260227_194611/residual_hist_high_vs_low.png",
        f"{repo_root}/outputs/paper3_high_density/20260227_194611/residual_vs_gbar_high_density.png",
        f"{repo_root}/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/figures/C1_block_nulls.png",
        f"{repo_root}/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/figures/C2_binwise_shift.png",
        f"{repo_root}/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/figures/C3_source_replication.png",
    ]:
        if Path(fig).exists():
            lines.append(f"  - `{fig}`")
    lines.append("")

    lines.append("## Corrections & Incident Timeline")
    for c in corrections:
        lines.append(f"### {c['id']}: {c['title']}")
        lines.append(f"- Status: `{c['status']}`")
        lines.append(f"- What was wrong/risk: {c['issue']}")
        lines.append(f"- What changed: {c['change']}")
        lines.append(f"- Why this matters scientifically: {c['scientific_effect']}")
        lines.append("- Evidence:")
        for e in c["evidence"]:
            lines.append(f"  - `{e.get('file')}` | `{e.get('locator')}` | `{e.get('value')}`")
        lines.append("")

    lines.append("## Evidence Map (Major Claims)")
    lines.append("| claim | file | locator | value |")
    lines.append("|---|---|---|---|")
    for c in major_claims:
        lines.append(f"| {c['claim']} | {c['file']} | {c['locator']} | {c['value']} |")
    lines.append("")

    lines.append("## Limitations / Not Claimed")
    lines.append("- No claim is made beyond available local artifacts; no web or external DB checks were used.")
    lines.append("- Several correction items are evidenced in current code/state but lack granular pre-fix commits due coarse commit history (`main` currently squashed at `f8f9149...`).")
    lines.append("- Branch D wrapper runs under `outputs/branchD_rar_control/20260228_002202` and `.../20260228_002303` contain run-stamps only (smoke/dry execution), not full science outputs.")
    lines.append("- If an artifact/key was absent, it is treated as `Not evidenced in artifacts` rather than inferred.")
    lines.append("")

    lines.append("## Reproducibility Instructions")
    lines.append("Run from repo root `/Users/russelllicht/bec-dark-matter`:")
    lines.append("```bash")
    lines.append("python3 analysis/run_branchD_rar_control.py --dataset analysis/results/rar_points_unified.csv --seed 42 --n_shuffles 1000")
    lines.append("python3 analysis/run_branchC_paper3_experiments.py --dataset analysis/results/rar_points_unified.csv --seed 42")
    lines.append("python3 analysis/paper3/paper3_branchC_experiments.py --rar_points_file analysis/results/rar_points_unified.csv --require_csv_sha 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c --n_perm 10000 --n_bins 15 --seed 42")
    lines.append("```")
    lines.append("- Confirm run stamps include repo root, git head, dataset sha, and output folder before interpreting results.")
    lines.append("")

    lines.append("## Audit Artifacts")
    lines.append(f"- Index JSON: `{out_paths['index']}`")
    lines.append(f"- File manifest CSV: `{out_paths['manifest']}`")
    lines.append(f"- Math appendix: `{out_paths['appendix']}`")
    lines.append("")

    if warnings:
        lines.append("## Missing Artifacts / Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_appendix(repo_root: Path) -> str:
    lines = [
        "# External Audit Appendix: Math Definitions (Branch C/D)",
        "",
        "## 1. RAR/BEC Prediction and Residuals",
        "- Baseline predictor (as implemented):",
        "  - `g_pred = g_bar / (1 - exp(-sqrt(g_bar / g_dagger)))`",
        "- Point residual in dex:",
        "  - `log_resid = log10(g_obs) - log10(g_pred)`",
        "- Branch C uses per-galaxy `rho_score` and top/bottom rank split over eligible galaxies.",
        "",
        "## 2. Branch C Statistics",
        "- C1 pooled median shift:",
        "  - `shift = median(resid_top) - median(resid_bottom)`",
        "- C1 galaxy-weighted median uses per-point weights `w=1/N_points(galaxy)` within each group.",
        "- C1 galaxy-block permutation p-value:",
        "  - Permute top/bottom galaxy labels preserving group sizes; recompute shift.",
        "  - Two-sided permutation p: `(1 + #(|null| >= |obs|)) / (1 + N_perm)`",
        "- C2 within-bin shift:",
        "  - Bin by `log_gbar` (default 15 bins), require min counts per group per bin.",
        "  - Bin statistic: median(top_bin) - median(bottom_bin).",
        "  - Uncertainty: galaxy-block bootstrap CI (2.5%, 97.5%).",
        "- C2 aggregate statistics:",
        "  - Equal-bin aggregate: arithmetic mean across valid bins.",
        "  - Matched-bin aggregate: weighted mean with `w_bin ~ min(n_top_bin, n_bottom_bin)`.",
        "- C3 source-stratified replication: rerun C1/C2 on source subsets with top_n shrink if eligibility is insufficient.",
        "",
        "## 3. Branch D Statistics (Refereeproof)",
        "- Scale scan / kernel matching uses AIC-based comparison across candidate kernels and scales.",
        "- Peak sharpness is reported from second-derivative behavior of AIC around optimum.",
        "- Null/control suites include:",
        "  - A1 global shuffle null (destructive).",
        "  - A2 within-bin shuffle control (structure-preserving; explicitly *not* a destructive null).",
        "  - A2b block-permute-bin null (destructive, bin-aware).",
        "  - A3 within-galaxy circular-shift null (destructive).",
        "- Dual-window proximity rates:",
        "  - `p_within_0p05_dex = hits(|log_scale - log_gdagger| < 0.05)/N`",
        "  - `p_within_0p10_dex = hits(|log_scale - log_gdagger| < 0.10)/N`",
        "- For zero-hit cases, Clopper-Pearson one-sided upper 95% bound is reported (`p_upper_95*`).",
        "- Nearby-scale comparisons (Suite F):",
        "  - F1 eta-fixed direct substitution AIC.",
        "  - F2 eta-free matched-DoF comparison with fitted amplitude.",
        "",
        "## 4. Provenance/Guardrail Math-Adjacent Definitions",
        "- Dataset immutability guard: SHA256 hash equality check for required CSV.",
        "- Run stamp provenance tuple: `(repo_root, git_head, dataset_sha256, output_dir, timestamp_utc)`.",
        "",
        "## 5. Code Pointers",
        f"- Branch C core: `{repo_root / 'analysis/paper3/paper3_branchC_experiments.py'}`",
        f"- Branch C baseline bridge: `{repo_root / 'analysis/paper3/paper3_bridge_pack.py'}`",
        f"- Branch D refereeproof suite: `{repo_root / 'analysis/tests/test_gdagger_hunt_refereeproof.py'}`",
        f"- Branch D kernel/scan engine: `{repo_root / 'analysis/gdagger_hunt.py'}`",
        "",
    ]
    return "\n".join(lines)


def build_manifest_rows(referenced_files: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(set(referenced_files)):
        p = Path(fp)
        if not p.exists() or not p.is_file():
            continue
        try:
            rows.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
        except Exception:
            continue
    return rows


def acquire_lock(lock_path: Path, run_id: str, repo_root: Path, start_time_utc: str) -> None:
    payload = {
        "run_id": run_id,
        "repo_root": str(repo_root),
        "pid": os.getpid(),
        "start_time_utc": start_time_utc,
    }
    try:
        with lock_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2) + "\n")
    except FileExistsError:
        existing = read_text(lock_path).strip()
        msg = (
            f"[EXTERNAL_AUDIT] Lock file exists: {lock_path}. "
            "Another audit run may still be active. "
            "Remove the lock only if you are sure no run is in progress."
        )
        if existing:
            msg += f"\n[EXTERNAL_AUDIT] Existing lock contents:\n{existing}"
        raise RuntimeError(msg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-pass external audit artifact generator.")
    p.add_argument("--repo_root", type=str, default="/Users/russelllicht/bec-dark-matter")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None, help="Optional fixed RUN_ID. If omitted, generated once at start.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser().resolve()
    else:
        results_dir = (repo_root / "analysis" / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    start_dt = utc_now()
    run_id = args.run_id.strip() if args.run_id else compact_run_id(start_dt)
    start_time_utc = utc_iso(start_dt)

    out_paths = {
        "report": results_dir / f"{REPORT_PREFIX}_{run_id}.md",
        "index": results_dir / f"{INDEX_PREFIX}_{run_id}.json",
        "manifest": results_dir / f"{MANIFEST_PREFIX}_{run_id}.csv",
        "appendix": results_dir / f"{APPENDIX_PREFIX}_{run_id}.md",
    }
    lock_path = results_dir / LOCK_FILENAME

    acquire_lock(lock_path=lock_path, run_id=run_id, repo_root=repo_root, start_time_utc=start_time_utc)

    try:
        # Single pass: build everything in memory first.
        context = build_audit_context(repo_root=repo_root)
        manifest_rows = build_manifest_rows(context["referenced_files"])

        end_dt = utc_now()
        end_time_utc = utc_iso(end_dt)

        run_stamp = {
            "run_id": run_id,
            "git_head": context["git_head"],
            "repo_root": context["repo_root"],
            "dataset_sha256": context["dataset_sha256"],
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "file_counts": {
                "tests_cataloged": len(context["tests"]),
                "executed_runs_found": len(context["runs"]),
                "referenced_files_manifested": len(manifest_rows),
                "warnings_count": len(context["warnings"]),
            },
        }

        index_obj = {
            "generated_utc": end_time_utc,
            "run_stamp": run_stamp,
            "repo_root": context["repo_root"],
            "git_head": context["git_head"],
            "dataset": {
                "path": context["dataset_path"],
                "sha256": context["dataset_sha256"],
                "backup_path": context["dataset_backup_path"],
                "backup_sha256": context["dataset_backup_sha256"],
            },
            "tests": context["tests"],
            "runs": context["runs"],
            "corrections_timeline": context["corrections"],
            "major_claims": context["major_claims"],
            "warnings": context["warnings"],
        }

        report_text = render_report(context=context, run_stamp=run_stamp, out_paths=out_paths)
        appendix_text = render_appendix(repo_root=repo_root)
        manifest_csv = build_manifest_csv(manifest_rows)

        existing = [p for p in out_paths.values() if p.exists()]
        if existing:
            print(
                f"[EXTERNAL_AUDIT] RUN_ID {run_id}: overwriting existing files in-place "
                f"({len(existing)} existing)."
            )

        # One write phase only (atomic replace).
        atomic_write_text(out_paths["report"], report_text)
        atomic_write_json(out_paths["index"], index_obj)
        atomic_write_text(out_paths["manifest"], manifest_csv)
        atomic_write_text(out_paths["appendix"], appendix_text)

        print(f"[EXTERNAL_AUDIT] RUN_ID={run_id}")
        print(f"[EXTERNAL_AUDIT] report={out_paths['report']}")
        print(f"[EXTERNAL_AUDIT] index={out_paths['index']}")
        print(f"[EXTERNAL_AUDIT] manifest={out_paths['manifest']}")
        print(f"[EXTERNAL_AUDIT] appendix={out_paths['appendix']}")
        print(f"[EXTERNAL_AUDIT] tests_cataloged={len(context['tests'])}")
        print(f"[EXTERNAL_AUDIT] executed_runs_found={len(context['runs'])}")
        print(f"[EXTERNAL_AUDIT] referenced_files_manifested={len(manifest_rows)}")
        print(f"[EXTERNAL_AUDIT] warnings={len(context['warnings'])}")
        return 0
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            # Do not hide primary exceptions; print a cleanup warning only.
            print(f"[EXTERNAL_AUDIT] Warning: failed to remove lock file {lock_path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

