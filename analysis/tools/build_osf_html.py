#!/usr/bin/env python3
"""Build OSF wiki HTML artifacts from repository summaries and references."""

from __future__ import annotations

import ast
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "analysis" / "results"
PIPELINE_DIR = ROOT / "analysis" / "pipeline"
INDEX_PATH = RESULTS_DIR / "osf_tests_index_tmp.json"
TESTS_HTML_PATH = RESULTS_DIR / "tests_results_osf.html"
REFS_HTML_PATH = RESULTS_DIR / "references_osf.html"
BIB_PATH = ROOT / "paper" / "references.bib"

TABLE_STYLE = "width:100%; border-collapse:collapse; margin:10px 0;"
TH_LEFT = "border:1px solid #ddd; padding:6px; text-align:left;"
TH_RIGHT = "border:1px solid #ddd; padding:6px; text-align:right;"
TD_LEFT = "border:1px solid #ddd; padding:6px; text-align:left;"
TD_RIGHT = "border:1px solid #ddd; padding:6px; text-align:right;"
MISSING_FINDING = (
    "Finding: TODO (missing verdict/finding text \u2014 add summary['verdict'] "
    "or Finding: docstring)"
)
FALLBACK_TODO = "TODO: requires upstream docs not present locally."

CORE_SCRIPTS = [
    "analysis/pipeline/test_env_scatter_definitive.py",
    "analysis/pipeline/test_mc_distance_and_inversion.py",
    "analysis/pipeline/test_nonparametric_inversion.py",
    "analysis/pipeline/test_binning_robustness.py",
    "analysis/pipeline/test_jackknife_robustness.py",
    "analysis/pipeline/test_env_confound_control.py",
    "analysis/pipeline/test_split_half_replication.py",
    "analysis/pipeline/test_propensity_matched_env.py",
    "analysis/pipeline/test_alfalfa_yang_btfr.py",
    "analysis/pipeline/test_brouwer_lensing_rar.py",
    "analysis/pipeline/test_lensing_profile_shape.py",
    "analysis/pipeline/test_cluster_rar_tian2020.py",
    "analysis/pipeline/test_extended_rar_inversion.py",
    "analysis/pipeline/test_probes_inversion_replication.py",
    "analysis/pipeline/test_kurtosis_phase_transition.py",
    "analysis/pipeline/test_kurtosis_disambiguation.py",
    "analysis/pipeline/test_korsaga_ml_sensitivity.py",
    "analysis/pipeline/test_lcdm_null_inversion.py",
    "analysis/pipeline/literature_crossref_tests.py",
    "analysis/pipeline/test_tf_scatter_redshift.py",
    "analysis/pipeline/integrate_mhongoose.py",
]
if (PIPELINE_DIR / "validate_mhongoose_sparc.py").exists():
    CORE_SCRIPTS.append("analysis/pipeline/validate_mhongoose_sparc.py")

CORE_SUMMARY_OVERRIDE = {
    "analysis/pipeline/test_env_scatter_definitive.py": "analysis/results/summary_env_definitive.json",
    "analysis/pipeline/test_mc_distance_and_inversion.py": "analysis/results/summary_mc_distance_and_inversion.json",
    "analysis/pipeline/test_nonparametric_inversion.py": "analysis/results/summary_nonparametric_inversion.json",
    "analysis/pipeline/test_binning_robustness.py": "analysis/results/summary_binning_robustness.json",
    "analysis/pipeline/test_jackknife_robustness.py": "analysis/results/summary_jackknife_robustness.json",
    "analysis/pipeline/test_env_confound_control.py": "analysis/results/summary_env_confound_control.json",
    "analysis/pipeline/test_split_half_replication.py": "analysis/results/summary_split_half_replication.json",
    "analysis/pipeline/test_propensity_matched_env.py": "analysis/results/summary_propensity_matched_env.json",
    "analysis/pipeline/test_alfalfa_yang_btfr.py": "analysis/results/summary_alfalfa_yang_btfr.json",
    "analysis/pipeline/test_brouwer_lensing_rar.py": "analysis/results/summary_brouwer_lensing_rar.json",
    "analysis/pipeline/test_lensing_profile_shape.py": "analysis/results/summary_lensing_profile_shape.json",
    "analysis/pipeline/test_cluster_rar_tian2020.py": "analysis/results/summary_cluster_rar_tian2020.json",
    "analysis/pipeline/test_extended_rar_inversion.py": "analysis/results/summary_extended_rar_inversion.json",
    "analysis/pipeline/test_probes_inversion_replication.py": "analysis/results/summary_probes_inversion_replication.json",
    "analysis/pipeline/test_kurtosis_phase_transition.py": "analysis/results/summary_kurtosis_phase_transition.json",
    "analysis/pipeline/test_kurtosis_disambiguation.py": "analysis/results/summary_kurtosis_disambiguation.json",
    "analysis/pipeline/test_korsaga_ml_sensitivity.py": "analysis/results/summary_korsaga_ml_sensitivity.json",
    "analysis/pipeline/test_lcdm_null_inversion.py": "analysis/results/summary_lcdm_null_inversion.json",
    "analysis/pipeline/literature_crossref_tests.py": "analysis/results/summary_literature_crossref.json",
    "analysis/pipeline/test_tf_scatter_redshift.py": "analysis/results/summary_tf_scatter_redshift.json",
    "analysis/pipeline/integrate_mhongoose.py": "analysis/results/summary_mhongoose_integration.json",
    "analysis/pipeline/validate_mhongoose_sparc.py": "analysis/results/summary_validate_mhongoose_sparc.json",
}

PREFERRED_METRIC_PATTERNS = [
    r"(^|\\.)n_matched$",
    r"(^|\\.)n_gal$",
    r"(^|\\.)n_gals$",
    r"(^|\\.)n_pts$",
    r"(^|\\.)n_points$",
    r"(^|\\.)n_field$",
    r"(^|\\.)n_dense$",
    r"log_g_dagger",
    r"inversion_point",
    r"delta_from_gdagger",
    r"levene_p",
    r"p_value",
    r"_p$",
    r"delta_sigma",
    r"kappa4|kurtosis_at_gdagger",
    r"slope",
    r"intercept",
    r"scatter",
]


def sanitize_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "H$\\alpha$": "H\u03b1",
        "H\\alpha": "H\u03b1",
        "$\\Lambda$CDM": "\u039bCDM",
        "{$\\Lambda$CDM}": "\u039bCDM",
        "\\LambdaCDM": "\u039bCDM",
        "$\\sim$": "~",
        "\\&": "&",
        "{\\'\\i}": "\u00ed",
        "\\'{i}": "\u00ed",
        "\\'i": "\u00ed",
        "Benitez-Llambay": "Ben\u00edtez-Llambay",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\textit\\{([^{}]+)\\}", r"\\1", text)
    text = re.sub(r"\\textbf\\{([^{}]+)\\}", r"\\1", text)
    text = text.replace("$", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\u2026", "")
    text = text.replace("...", "")
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def html_text(value: Any) -> str:
    return html.escape(sanitize_text(value))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_index_records() -> List[Dict[str, Any]]:
    if INDEX_PATH.exists():
        try:
            records = json.loads(INDEX_PATH.read_text())
            if isinstance(records, list):
                return records
        except Exception:
            pass

    records: List[Dict[str, Any]] = []
    for idx, script_path in enumerate(sorted(PIPELINE_DIR.glob("*.py")), start=1):
        rel = str(script_path.relative_to(ROOT))
        guess = CORE_SUMMARY_OVERRIDE.get(rel, "")
        records.append(
            {
                "idx": idx,
                "script": rel,
                "datasets": [],
                "summary": guess,
                "method": "",
                "metrics": [],
                "finding": None,
                "map": "fallback-scan",
            }
        )
    return records


def get_script_doc_summary(script_rel_path: str) -> str:
    script_path = ROOT / script_rel_path
    if not script_path.exists():
        return ""
    try:
        tree = ast.parse(script_path.read_text())
        doc = ast.get_docstring(tree) or ""
    except Exception:
        return ""
    lines = [line.strip() for line in doc.splitlines()]
    lines = [line for line in lines if line and set(line) != {"="} and set(line) != {"-"}]
    if not lines:
        return ""
    joined = " ".join(lines)
    return first_sentences(joined, n=2)


def first_sentences(text: str, n: int = 2) -> str:
    cleaned = sanitize_text(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\\s+", cleaned)
    picked = [p.strip() for p in parts if p.strip()][:n]
    return " ".join(picked) if picked else cleaned


def flatten_scalars(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{prefix}.{key}" if prefix else key
            items.extend(flatten_scalars(value, key_path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            key_path = f"{prefix}[{i}]"
            items.extend(flatten_scalars(value, key_path))
    elif isinstance(obj, (int, float, bool, str)):
        items.append((prefix, obj))
    return items


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return sanitize_text(value)


def detect_binned_summary(summary: Dict[str, Any]) -> Optional[str]:
    messages: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
            p_candidates: List[Tuple[int, float, str]] = []
            d_candidates: List[Tuple[int, float, str]] = []
            for i, row in enumerate(node):
                label = (
                    str(row.get("label") or row.get("bin") or row.get("bin_label") or f"row_{i}")
                )
                for key, value in row.items():
                    if not isinstance(value, (int, float)):
                        continue
                    lk = key.lower()
                    if "levene_p" in lk:
                        p_candidates.append((i, float(value), label))
                    if "delta_sigma" in lk:
                        d_candidates.append((i, float(value), label))
            if p_candidates or d_candidates:
                parts = []
                if p_candidates:
                    min_row = min(p_candidates, key=lambda x: x[1])
                    parts.append(
                        f"min Levene p across bins = {format_value(min_row[1])}"
                    )
                if d_candidates:
                    max_row = max(d_candidates, key=lambda x: abs(x[1]))
                    parts.append(
                        "bin with strongest |delta_sigma| = "
                        f"{sanitize_text(max_row[2])} (delta_sigma={format_value(max_row[1])})"
                    )
                if parts:
                    messages.append("; ".join(parts))
            for value in node:
                walk(value)

    walk(summary)
    if not messages:
        return None
    return messages[0]


def pick_metrics(summary: Optional[Dict[str, Any]], minimum: int = 2, maximum: int = 6) -> List[Tuple[str, Any]]:
    if not summary:
        return []
    all_scalars = flatten_scalars(summary)
    numeric = [(k, v) for k, v in all_scalars if isinstance(v, (int, float)) and not isinstance(v, bool)]
    selected: List[Tuple[str, Any]] = []
    seen = set()

    for pattern in PREFERRED_METRIC_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for key, value in numeric:
            if key in seen:
                continue
            if regex.search(key):
                selected.append((key, value))
                seen.add(key)
                break
        if len(selected) >= maximum:
            break

    if len(selected) < minimum:
        for key, value in numeric:
            if key in seen:
                continue
            selected.append((key, value))
            seen.add(key)
            if len(selected) >= minimum:
                break

    binned_summary = detect_binned_summary(summary)
    if binned_summary and len(selected) < maximum:
        selected.append(("binned_summary", binned_summary))

    return selected[:maximum]


def finding_from_script(script_rel_path: str) -> Optional[str]:
    script_path = ROOT / script_rel_path
    if not script_path.exists():
        return None
    pattern = re.compile(r"Finding:\\s*(.+)", flags=re.IGNORECASE)
    for line in script_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            return sanitize_text(match.group(1))
    return None


def choose_finding(summary: Optional[Dict[str, Any]], script_rel_path: str) -> str:
    if summary:
        if "verdict" in summary and summary["verdict"] not in (None, ""):
            return sanitize_text(summary["verdict"])
        for key in ("finding", "headline", "conclusion", "summary_text"):
            if key in summary and summary[key] not in (None, ""):
                return sanitize_text(summary[key])
    from_script = finding_from_script(script_rel_path)
    if from_script:
        return from_script
    return MISSING_FINDING.replace("Finding: ", "")


def dataset_text(record: Dict[str, Any], summary: Optional[Dict[str, Any]]) -> str:
    datasets = record.get("datasets")
    if isinstance(datasets, list) and datasets:
        return ", ".join(sanitize_text(x) for x in datasets)
    if isinstance(summary, dict):
        desc = summary.get("description")
        if isinstance(desc, str) and desc.strip():
            return first_sentences(desc, n=1)
    return FALLBACK_TODO


def method_text(record: Dict[str, Any], summary: Optional[Dict[str, Any]]) -> str:
    if isinstance(summary, dict):
        desc = summary.get("description")
        if isinstance(desc, str) and desc.strip():
            return first_sentences(desc, n=2)
    doc = get_script_doc_summary(record.get("script", ""))
    if doc:
        return doc
    fallback = record.get("method", "")
    return first_sentences(fallback, n=2) if fallback else FALLBACK_TODO


def make_table(headers: Sequence[Tuple[str, str]], rows: Sequence[Sequence[str]]) -> str:
    parts = [f'<table style="{TABLE_STYLE}">', "<thead>", "<tr>"]
    for title, align in headers:
        style = TH_RIGHT if align == "right" else TH_LEFT
        parts.append(f'<th style="{style}">{html.escape(title)}</th>')
    parts.extend(["</tr>", "</thead>", "<tbody>"])
    for row in rows:
        parts.append("<tr>")
        for cell, (_, align) in zip(row, headers):
            style = TD_RIGHT if align == "right" else TD_LEFT
            parts.append(f'<td style="{style}">{cell}</td>')
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


def build_tests_results_html(records: List[Dict[str, Any]]) -> str:
    record_map = {rec.get("script"): rec for rec in records}
    all_records: List[Dict[str, Any]] = sorted(
        records, key=lambda r: int(r.get("idx", 10**9))
    )

    parts: List[str] = []
    parts.append("<h1>Tests &amp; Results</h1>")
    parts.append(
        "<p>This page compiles test artifacts generated in this repository. "
        "Findings are resolved by strict priority: verdict, finding/headline/"
        "conclusion/summary_text, script Finding: line, then explicit TODO.</p>"
    )
    parts.append("<h2>Core Tests (Detailed)</h2>")

    for i, script_rel in enumerate(CORE_SCRIPTS, start=1):
        record = dict(record_map.get(script_rel, {}))
        record.setdefault("script", script_rel)
        summary_rel = record.get("summary") or CORE_SUMMARY_OVERRIDE.get(script_rel, "")
        record["summary"] = summary_rel
        summary_obj = load_json(ROOT / summary_rel) if summary_rel else None
        finding = choose_finding(summary_obj, script_rel)
        metrics = pick_metrics(summary_obj, minimum=2, maximum=8)

        parts.append("<hr>")
        parts.append(
            f"<h3>Test {i}: {html_text(Path(script_rel).stem)}</h3>"
        )
        parts.append(f"<p>Script: <code>{html.escape(script_rel)}</code></p>")
        parts.append(f"<p>Data: {html_text(dataset_text(record, summary_obj))}</p>")
        parts.append(f"<p>Method: {html_text(method_text(record, summary_obj))}</p>")
        parts.append(
            f"<p>Summary Artifact: <code>{html.escape(summary_rel)}</code></p>"
            if summary_rel
            else "<p>Summary Artifact: <code>TODO (missing summary mapping)</code></p>"
        )

        metric_rows: List[List[str]] = []
        if metrics:
            for key, value in metrics:
                metric_rows.append([html.escape(key), html_text(format_value(value))])
        else:
            metric_rows.append(
                [
                    "artifact_status",
                    html.escape("TODO (missing or unreadable summary JSON artifact)."),
                ]
            )
        parts.append(
            make_table(
                [("Metric", "left"), ("Value", "left")],
                metric_rows,
            )
        )
        parts.append(f"<p>Finding: {html_text(finding)}</p>")

    parts.append("<h2>Appendix: Full Inventory (one row per script)</h2>")

    inventory_rows: List[List[str]] = []
    missing_rows: List[List[str]] = []

    for rec in all_records:
        script_rel = rec.get("script", "")
        summary_rel = rec.get("summary", "")
        summary_path = ROOT / summary_rel if summary_rel else None
        summary_obj = load_json(summary_path) if summary_path else None
        finding = choose_finding(summary_obj, script_rel)
        key_metrics = pick_metrics(summary_obj, minimum=1, maximum=2)
        metric_text = "; ".join(
            f"{k}={format_value(v)}" for k, v in key_metrics
        ) if key_metrics else "artifact_status=TODO (missing or unreadable summary JSON artifact)."
        result_cell = (
            f"{Path(summary_rel).name if summary_rel else 'MISSING_SUMMARY'}; "
            f"{metric_text}; Finding: {finding}"
        )
        inventory_rows.append(
            [
                html_text(rec.get("idx", "")),
                f"<code>{html.escape(script_rel)}</code>",
                f"<code>{html.escape(summary_rel)}</code>" if summary_rel else "<code></code>",
                html_text(dataset_text(rec, summary_obj)),
                html_text(result_cell),
            ]
        )

        if not summary_rel or not summary_path or not summary_path.exists():
            searched = summary_rel if summary_rel else "No summary path in osf_tests_index_tmp.json record."
            missing_rows.append(
                [f"<code>{html.escape(script_rel)}</code>", f"<code>{html.escape(searched)}</code>"]
            )

    parts.append(
        make_table(
            [
                ("#", "right"),
                ("Script", "left"),
                ("Summary JSON", "left"),
                ("Data", "left"),
                ("Result + Finding", "left"),
            ],
            inventory_rows,
        )
    )

    parts.append("<h3>Scripts Missing Summaries</h3>")
    if missing_rows:
        parts.append(
            make_table(
                [("Script", "left"), ("Searched path(s)", "left")],
                missing_rows,
            )
        )
    else:
        parts.append("<p>All indexed scripts have summary JSON artifacts present.</p>")

    html_block = "\n".join(parts).strip()
    html_block = html_block.replace("\u2026", "").replace("...", "")
    return html_block


def parse_bib_entries(text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    current: List[str] = []
    depth = 0
    in_entry = False

    for line in text.splitlines():
        stripped = line.strip()
        if not in_entry and stripped.startswith("@"):
            in_entry = True
            current = [line]
            depth = line.count("{") - line.count("}")
            continue
        if in_entry:
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                entry_text = "\n".join(current)
                in_entry = False
                current = []

                lines = entry_text.splitlines()
                if not lines:
                    continue
                head = re.match(r"@([A-Za-z]+)\s*\{\s*([^,]+)\s*,", lines[0].strip())
                if not head:
                    continue

                entry_type = head.group(1).strip()
                key = head.group(2).strip()
                fields: Dict[str, str] = {"ENTRYTYPE": entry_type, "ID": key}

                current_key = None
                buffer: List[str] = []
                for body_line in lines[1:]:
                    s = body_line.strip()
                    if not s or s == "}":
                        continue
                    # start of a new field
                    if "=" in s and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", s):
                        if current_key:
                            fields[current_key] = " ".join(buffer).strip().rstrip(",")
                        k, v = s.split("=", 1)
                        current_key = k.strip().lower()
                        buffer = [v.strip()]
                    elif current_key:
                        buffer.append(s)
                if current_key:
                    fields[current_key] = " ".join(buffer).strip().rstrip(",")

                cleaned: Dict[str, str] = {}
                for k, v in fields.items():
                    value = v.strip()
                    while (
                        (value.startswith("{") and value.endswith("}"))
                        or (value.startswith('"') and value.endswith('"'))
                    ) and len(value) >= 2:
                        value = value[1:-1].strip()
                    cleaned[k] = sanitize_text(value)
                entries.append(cleaned)

    return entries


def format_authors(author_field: str) -> str:
    if not author_field:
        return "Unknown authors"
    parts = [sanitize_text(x) for x in author_field.split(" and ")]
    parts = [x for x in parts if x]
    return "; ".join(parts) if parts else "Unknown authors"


def build_reference_item(entry: Dict[str, str]) -> str:
    authors = format_authors(entry.get("author", ""))
    year = sanitize_text(entry.get("year", "n.d."))
    title = sanitize_text(entry.get("title", "Untitled"))
    journal = sanitize_text(entry.get("journal", ""))
    volume = sanitize_text(entry.get("volume", ""))
    pages = sanitize_text(entry.get("pages", ""))
    doi = sanitize_text(entry.get("doi", ""))
    eprint = sanitize_text(entry.get("eprint", ""))

    parts = [f"{html.escape(authors)} ({html.escape(year)}). {html.escape(title)}."]
    if journal:
        journal_bits = [journal]
        if volume:
            journal_bits.append(volume)
        if pages:
            journal_bits.append(pages)
        parts.append(" " + html.escape(", ".join(journal_bits)) + ".")
    if doi:
        doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        parts.append(
            f' DOI <a href="{html.escape(doi_url)}">{html.escape(doi_url)}</a>.'
        )
    if eprint:
        arxiv_url = f"https://arxiv.org/abs/{eprint}"
        parts.append(
            f' arXiv <a href="{html.escape(arxiv_url)}">{html.escape(eprint)}</a>.'
        )
    return "<li>" + "".join(parts).strip() + "</li>"


def build_references_html() -> str:
    bib_text = BIB_PATH.read_text() if BIB_PATH.exists() else ""
    entries = parse_bib_entries(bib_text)
    list_items: List[str] = []
    duey_present = False

    for entry in entries:
        key = entry.get("ID", "")
        title = entry.get("title", "")
        doi = entry.get("doi", "")

        if key == "McGaugh2025" or "z ~ 2.5" in title:
            continue

        if doi.endswith("10.3847/1538-3881/adaf21") or doi == "10.3847/1538-3881/adaf21":
            duey_present = True

        list_items.append(build_reference_item(entry))

    if not duey_present:
        list_items.append(
            "<li>Duey, Francis; Schombert, James M.; McGaugh, Stacy S.; Lelli, "
            "Federico (2025). The Baryonic Tully\u2013Fisher Relation. II. Stellar "
            "Mass Models. AJ 169, 186. DOI "
            '<a href="https://doi.org/10.3847/1538-3881/adaf21">'
            "https://doi.org/10.3847/1538-3881/adaf21</a>. arXiv "
            '<a href="https://arxiv.org/abs/2501.10919">2501.10919</a>.</li>'
        )

    parts = [
        "<h1>References</h1>",
        "<p>This list is generated from paper/references.bib with OSF-safe formatting "
        "(no LaTeX math macros) and stable DOI/arXiv links.</p>",
        "<p>Placeholder entry McGaugh2025 (\"z ~ 2.5\") is omitted pending verified publication metadata.</p>",
        '<ol class="references">',
    ]
    parts.extend(list_items)
    parts.append("</ol>")
    html_block = "\n".join(parts).strip()
    html_block = html_block.replace("\u2026", "").replace("...", "")
    return html_block


def main() -> None:
    records = load_index_records()
    tests_html = build_tests_results_html(records)
    refs_html = build_references_html()
    TESTS_HTML_PATH.write_text(tests_html + "\n")
    REFS_HTML_PATH.write_text(refs_html + "\n")
    print(f"Wrote {TESTS_HTML_PATH}")
    print(f"Wrote {REFS_HTML_PATH}")


if __name__ == "__main__":
    main()
