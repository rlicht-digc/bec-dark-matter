#!/usr/bin/env python3
"""Build a deterministic State(1-4) alignment inventory for key project trees.

Outputs:
- evidence_vault/reproducibility/alignment_inventory.csv
- evidence_vault/reproducibility/alignment_inventory_summary.md
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOTS = [
    Path("analysis/pipeline"),
    Path("analysis/results"),
    Path("analysis/results_vm"),
    Path("outputs"),
]
OUT_DIR = Path("evidence_vault/reproducibility")
OUT_CSV = OUT_DIR / "alignment_inventory.csv"
OUT_SUMMARY = OUT_DIR / "alignment_inventory_summary.md"

MAP_FILES = [
    Path("ARTIFACTS_INDEX.md"),
    Path("DATASETS_INDEX.md"),
    Path("analysis/results/tests_results_osf.html"),
    Path("analysis/results/references_osf.html"),
    Path("docs/osf_tests_results.html"),
]
OSF_INDEX = Path("analysis/results/osf_tests_index_tmp.json")

CORE_SCRIPTS = {
    "test_env_scatter_definitive.py",
    "test_mc_distance_and_inversion.py",
    "test_nonparametric_inversion.py",
    "test_binning_robustness.py",
    "test_jackknife_robustness.py",
    "test_env_confound_control.py",
    "test_split_half_replication.py",
    "test_propensity_matched_env.py",
    "test_alfalfa_yang_btfr.py",
    "test_lcdm_null_inversion.py",
    "test_kurtosis_phase_transition.py",
    "test_kurtosis_disambiguation.py",
    "test_brouwer_lensing_rar.py",
    "test_lensing_profile_shape.py",
    "test_probes_inversion_replication.py",
    "test_extended_rar_inversion.py",
}

FIGURE_EXT = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
DEPRECATED_HINTS = {"deprecated", "superseded", "broken", "do_not_use"}
QUARANTINE_HINTS = {"tmp", "temp", "draft", "partial", "pilot", "scratch", "debug"}


def rel(path: Path) -> str:
    return str(path.as_posix())


def load_map_texts() -> Dict[str, str]:
    data: Dict[str, str] = {}
    for p in MAP_FILES:
        full = REPO_ROOT / p
        if full.exists():
            data[rel(p)] = full.read_text(errors="replace")
    return data


def discover_files() -> List[Path]:
    files: List[Path] = []
    for root in TARGET_ROOTS:
        full_root = REPO_ROOT / root
        if not full_root.exists():
            continue
        for p in full_root.rglob("*"):
            if p.is_file():
                files.append(p.relative_to(REPO_ROOT))
    return sorted(files, key=rel)


def detect_type(path: Path) -> str:
    if path.suffix == ".py":
        return "script"
    if path.name.startswith("summary_") and path.suffix == ".json":
        return "summary"
    if path.suffix.lower() in FIGURE_EXT:
        return "figure"
    return "table"


def read_json(path: Path) -> Any:
    full = REPO_ROOT / path
    try:
        return json.loads(full.read_text())
    except Exception:
        return None


def summary_valid(summary_obj: Any) -> Tuple[bool, bool]:
    """Return (has_required_keys, has_verdict)."""
    if not isinstance(summary_obj, dict):
        return (False, False)
    has_required = isinstance(summary_obj.get("test_name"), str) and isinstance(
        summary_obj.get("description"), str
    )
    has_verdict = isinstance(summary_obj.get("verdict"), str) and bool(
        summary_obj.get("verdict").strip()
    )
    return (has_required, has_verdict)


def load_osf_mapping() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    script_to_summary: Dict[str, str] = {}
    summary_to_scripts: Dict[str, List[str]] = defaultdict(list)

    full = REPO_ROOT / OSF_INDEX
    if not full.exists():
        return script_to_summary, summary_to_scripts

    try:
        records = json.loads(full.read_text())
    except Exception:
        return script_to_summary, summary_to_scripts

    if not isinstance(records, list):
        return script_to_summary, summary_to_scripts

    for rec in records:
        if not isinstance(rec, dict):
            continue
        script = rec.get("script")
        summary = rec.get("summary")
        if isinstance(script, str) and isinstance(summary, str) and summary:
            script_to_summary[script] = summary
            summary_to_scripts[summary].append(script)

    return script_to_summary, summary_to_scripts


def references_for_path(path: Path, map_texts: Dict[str, str]) -> str:
    rp = rel(path)
    refs = []
    for map_file, txt in map_texts.items():
        if rp in txt:
            refs.append(map_file)
    return ";".join(refs) if refs else "UNMAPPED"


def state_and_reason(
    path: Path,
    artifact_type: str,
    refs: str,
    script_to_summary: Dict[str, str],
    summary_to_scripts: Dict[str, List[str]],
) -> Tuple[int, str]:
    rp = rel(path)
    low = rp.lower()

    if any(h in low for h in DEPRECATED_HINTS):
        return 4, "Path flagged by deprecated/broken/superseded hint."

    if artifact_type == "script":
        mapped_summary = script_to_summary.get(rp, "")
        if mapped_summary:
            s_obj = read_json(Path(mapped_summary))
            has_required, has_verdict = summary_valid(s_obj)
            is_core = path.name in CORE_SCRIPTS
            if has_required and has_verdict and is_core:
                return 1, "Core script with mapped summary containing required keys + verdict."
            if has_required and has_verdict:
                return 2, "Mapped reproducible script with valid summary and verdict."
            if has_required and not has_verdict:
                return 3, "Mapped summary missing top-level verdict."
            return 3, "Mapped summary missing required keys or unreadable."
        return 3, "No mapped summary artifact; quarantine until promoted."

    if artifact_type == "summary":
        s_obj = read_json(path)
        has_required, has_verdict = summary_valid(s_obj)
        mapped_scripts = summary_to_scripts.get(rp, [])
        has_core_link = any(Path(s).name in CORE_SCRIPTS for s in mapped_scripts)
        if has_required and has_verdict and has_core_link:
            return 1, "Core-linked summary with required keys + verdict."
        if has_required and has_verdict:
            return 2, "Valid summary, supporting evidence."
        if has_required and not has_verdict:
            return 3, "Summary missing top-level verdict."
        return 4, "Summary unreadable or missing required keys test_name/description."

    if artifact_type == "figure":
        if refs != "UNMAPPED":
            return 2, "Referenced figure artifact; supporting evidence."
        if any(h in low for h in QUARANTINE_HINTS):
            return 3, "Unmapped exploratory/debug figure."
        if low.startswith("outputs/"):
            return 3, "Output figure not mapped to single source of truth."
        return 2, "Figure in analysis tree; supporting until core-cited."

    # table / generic data artifact
    if refs != "UNMAPPED":
        return 2, "Referenced data/table artifact; supporting evidence."
    if low.startswith("analysis/results/") or low.startswith("analysis/results_vm/"):
        if any(h in low for h in QUARANTINE_HINTS):
            return 3, "Local result artifact appears exploratory/partial."
        return 2, "Result artifact in analysis tree; supporting by default."
    if low.startswith("outputs/"):
        if any(h in low for h in QUARANTINE_HINTS):
            return 3, "Outputs artifact marked as temporary/partial."
        return 3, "Outputs artifact not yet promoted to canonical map files."
    return 3, "Artifact lacks mapping evidence."


def main() -> None:
    map_texts = load_map_texts()
    script_to_summary, summary_to_scripts = load_osf_mapping()
    files = discover_files()

    rows: List[Dict[str, str]] = []
    state_counts = Counter()
    type_counts = Counter()
    root_state_counts = defaultdict(Counter)

    for p in files:
        artifact_type = detect_type(p)
        refs = references_for_path(p, map_texts)
        state, reason = state_and_reason(
            p, artifact_type, refs, script_to_summary, summary_to_scripts
        )
        row = {
            "path": rel(p),
            "type": artifact_type,
            "state": str(state),
            "reason": reason,
            "references": refs,
        }
        rows.append(row)
        state_counts[state] += 1
        type_counts[artifact_type] += 1
        root = row["path"].split("/", 1)[0]
        root_state_counts[root][state] += 1

    out_dir = REPO_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with (REPO_ROOT / OUT_CSV).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "type", "state", "reason", "references"]
        )
        w.writeheader()
        w.writerows(rows)

    with (REPO_ROOT / OUT_SUMMARY).open("w", encoding="utf-8") as f:
        f.write("# Alignment Inventory Summary\n\n")
        f.write(f"- Total artifacts scanned: {len(rows)}\n")
        f.write("- Roots scanned: analysis/pipeline, analysis/results, analysis/results_vm, outputs\n\n")
        f.write("## State Counts\n\n")
        for s in [1, 2, 3, 4]:
            f.write(f"- State {s}: {state_counts[s]}\n")
        f.write("\n## Type Counts\n\n")
        for t in ["script", "summary", "figure", "table"]:
            f.write(f"- {t}: {type_counts[t]}\n")
        f.write("\n## Root x State\n\n")
        f.write("| root | state1 | state2 | state3 | state4 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for root in sorted(root_state_counts):
            c = root_state_counts[root]
            f.write(f"| {root} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |\n")
        f.write("\n## Notes\n\n")
        f.write("- State labels are preliminary and deterministic, intended for Claude review in Step 2.\n")
        f.write("- Core-candidate scripts are identified by the predeclared core list in this generator.\n")
        f.write("- `UNMAPPED` references indicate no mention in current map files.\n")

    print(rel(OUT_CSV))
    print(rel(OUT_SUMMARY))


if __name__ == "__main__":
    main()
