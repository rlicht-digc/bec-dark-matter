#!/usr/bin/env python3
"""Scan bec-dark-matter for potential provenance leakage from bh-singularity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

EXPECTED_REPO = Path("/Users/russelllicht/bec-dark-matter").resolve()
DEFAULT_BH_REPO = Path("/Users/russelllicht/bh-singularity").resolve()

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yml",
    ".yaml",
    ".ini",
    ".cfg",
    ".sh",
    ".csv",
}
MAX_HASH_FILE_BYTES = 3 * 1024 * 1024

EXCLUDE_DIR_NAMES = {
    ".git",
    "outputs",
    "figures",
    "datasets",
    "raw_data",
    "archive",
    "mailbox",
    "arxiv_package",
    "public_osf",
    "rerun_outputs",
    "__pycache__",
    ".pytest_cache",
}

PATTERNS = [
    re.compile(r"/Users/russelllicht/bh-singularity"),
    re.compile(r"\bbh-singularity\b"),
    re.compile(r"\bbh_singularity\b"),
    re.compile(r"subprojects/bh_singularity"),
]


def utc_now_compact() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def git_head(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "UNKNOWN"


def is_text_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    # extensionless common scripts / docs
    if "." not in path.name:
        return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        base = Path(dirpath)
        for fn in filenames:
            p = base / fn
            if p.is_symlink():
                # symlink handled separately
                continue
            yield p


def find_text_references(repo_root: Path, bh_root: Path) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []

    for path in iter_files(repo_root):
        if not is_text_candidate(path):
            continue
        rel = str(path.relative_to(repo_root))
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        hit_line = None
        hit_issue_type = "path"
        for line in text.splitlines():
            if any(p.search(line) for p in PATTERNS):
                hit_line = line.strip()
                line_low = line.lower().lstrip()
                if line_low.startswith("import ") or line_low.startswith("from "):
                    hit_issue_type = "import"
                break
        if hit_line is not None:
            findings.append(
                {
                    "file_path": rel,
                    "issue_type": hit_issue_type,
                    "suspected_origin": str(bh_root),
                    "recommended_action": (
                        "replace direct BH reference with local path or explicit vendor import"
                        if hit_issue_type == "import"
                        else "review BH path reference; remove or vendor with provenance"
                    ),
                    "evidence": hit_line[:240],
                }
            )

    # symlink references
    for path in repo_root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve()
        except Exception:
            continue
        if str(target).startswith(str(bh_root)):
            findings.append(
                {
                    "file_path": str(path.relative_to(repo_root)),
                    "issue_type": "path",
                    "suspected_origin": str(target),
                    "recommended_action": "if used by RAR/Paper3, replace with bec-dark-matter-local implementation",
                    "evidence": "symlink target points into bh-singularity",
                }
            )

    return findings


def find_identical_hashes(repo_root: Path, bh_root: Path) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []

    bh_hash_index: Dict[str, List[str]] = {}
    for p in iter_files(bh_root):
        try:
            if p.stat().st_size > MAX_HASH_FILE_BYTES:
                continue
        except Exception:
            continue
        if not is_text_candidate(p):
            continue
        try:
            h = sha256_file(p)
        except Exception:
            continue
        bh_hash_index.setdefault(h, []).append(str(p.relative_to(bh_root)))

    for p in iter_files(repo_root):
        try:
            if p.stat().st_size > MAX_HASH_FILE_BYTES:
                continue
        except Exception:
            continue
        if not is_text_candidate(p):
            continue
        rel = str(p.relative_to(repo_root))
        try:
            h = sha256_file(p)
        except Exception:
            continue
        if h in bh_hash_index:
            suspected = bh_hash_index[h][0]
            findings.append(
                {
                    "file_path": rel,
                    "issue_type": "identical_hash",
                    "suspected_origin": f"{bh_root}/{suspected}",
                    "recommended_action": "strong evidence of copy: remove or vendor under vendor/bh_singularity with README provenance",
                    "evidence": f"sha256={h}",
                }
            )

    return findings


def find_git_history_references(repo_root: Path) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    try:
        cmd = [
            "git",
            "log",
            "--all",
            "--name-only",
            "--pretty=format:",
            "-Sbh-singularity",
        ]
        out = subprocess.check_output(cmd, cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL)
        files = sorted({line.strip() for line in out.splitlines() if line.strip()})
        for f in files:
            findings.append(
                {
                    "file_path": f,
                    "issue_type": "path",
                    "suspected_origin": "git_history_search:-Sbh-singularity",
                    "recommended_action": "review history touchpoint; ensure current file has no BH dependency",
                    "evidence": "mentioned in commit diff string-search",
                }
            )
    except Exception:
        pass
    return findings


def dedupe_findings(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        key = (r["file_path"], r["issue_type"], r["suspected_origin"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda x: (x["issue_type"], x["file_path"]))
    return out


def write_markdown_report(
    out_path: Path,
    repo_root: Path,
    bh_root: Path,
    findings: Sequence[Dict[str, str]],
    git_head_repo: str,
    git_head_bh: str,
    assumptions: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Provenance Scan Report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')}")
    lines.append(f"- Repo scanned: `{repo_root}`")
    lines.append(f"- Reference BH repo: `{bh_root}`")
    lines.append(f"- Repo git HEAD: `{git_head_repo}`")
    lines.append(f"- BH git HEAD: `{git_head_bh}`")
    lines.append("")
    lines.append("## Assumptions / Uncertainties")
    for a in assumptions:
        lines.append(f"- {a}")
    lines.append("")

    ident_count = sum(1 for r in findings if r["issue_type"] == "identical_hash")
    if ident_count > 0:
        lines.append(f"- Vendoring recommendation: **required** (`{ident_count}` identical-hash matches found).")
    else:
        lines.append("- Vendoring recommendation: **no vendoring required** (no identical-hash matches found).")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    lines.append("| file_path | issue_type | suspected_origin | recommended_action |")
    lines.append("|---|---|---|---|")
    if findings:
        for r in findings:
            fp = r["file_path"].replace("|", "\\|")
            it = r["issue_type"].replace("|", "\\|")
            so = r["suspected_origin"].replace("|", "\\|")
            ra = r["recommended_action"].replace("|", "\\|")
            lines.append(f"| `{fp}` | `{it}` | `{so}` | {ra} |")
    else:
        lines.append("| (none) | (none) | (none) | no action required |")

    lines.append("")
    lines.append("## Evidence Snippets")
    lines.append("")
    if findings:
        for r in findings:
            lines.append(f"- `{r['file_path']}` [{r['issue_type']}]: {r.get('evidence', '')}")
    else:
        lines.append("- none")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Provenance scanner for bh-singularity leakage into bec-dark-matter.")
    p.add_argument("--repo_root", type=str, default=str(EXPECTED_REPO))
    p.add_argument("--bh_repo", type=str, default=str(DEFAULT_BH_REPO))
    p.add_argument("--out_report", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    bh_root = Path(args.bh_repo).expanduser().resolve()

    if not repo_root.exists() or not (repo_root / "analysis").exists():
        raise RuntimeError(f"Invalid repo_root: {repo_root}")
    if not bh_root.exists():
        raise RuntimeError(f"Invalid bh_repo: {bh_root}")

    ts = utc_now_compact()
    out_report = (
        Path(args.out_report).expanduser().resolve()
        if args.out_report
        else (repo_root / "analysis" / "results" / f"provenance_scan_{ts}.md")
    )

    assumptions = [
        "Hash comparison is limited to files <=3MB and common text-like extensions to avoid scanning large datasets/artifacts.",
        "Git history search uses -Sbh-singularity best-effort; missing references do not prove clean historical provenance.",
        "Symlink targets into bh-singularity are flagged as path issues, not identical-hash copies.",
    ]

    rows: List[Dict[str, str]] = []
    rows.extend(find_text_references(repo_root=repo_root, bh_root=bh_root))
    rows.extend(find_identical_hashes(repo_root=repo_root, bh_root=bh_root))
    rows.extend(find_git_history_references(repo_root=repo_root))
    rows = dedupe_findings(rows)

    repo_head = git_head(repo_root)
    bh_head = git_head(bh_root)

    write_markdown_report(
        out_path=out_report,
        repo_root=repo_root,
        bh_root=bh_root,
        findings=rows,
        git_head_repo=repo_head,
        git_head_bh=bh_head,
        assumptions=assumptions,
    )

    print(f"[PROVENANCE] report={out_report}")
    print(f"[PROVENANCE] findings={len(rows)}")
    print(f"[PROVENANCE] identical_hash_findings={sum(1 for r in rows if r['issue_type']=='identical_hash')}")


if __name__ == "__main__":
    main()
