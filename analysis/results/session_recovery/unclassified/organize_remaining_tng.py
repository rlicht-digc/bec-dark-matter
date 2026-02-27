#!/usr/bin/env python3
"""
Organize downloaded Jupyter/TNG artifacts dropped into Remaining_TNG.

This script is intentionally conservative:
- copies files (does not delete source)
- never overwrites non-identical files
- logs every action
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil


ROOT = Path("/Users/russelllicht/bec-dark-matter")
DEFAULT_SOURCE = ROOT / "Remaining_TNG"

DEST_SESSION = ROOT / "analysis" / "results" / "session_recovery"
DEST_SESSION_UNCLASSIFIED = DEST_SESSION / "unclassified"
DEST_PIPELINE = ROOT / "analysis" / "pipeline"
DEST_ARCHIVES = ROOT / "datasets" / "raw_archives"
DEST_BIG_BASE = ROOT / "datasets" / "TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE"
DEST_BIG_BASE_META = DEST_BIG_BASE / "meta"


SESSION_RECOVERY_FILES = {
    "registry_runs.csv",
    "registry_summary.json",
    "quality_cut_reconstruction.csv",
    "selected_ids_catalog.csv",
    "missing_vs_present.md",
    "repro_commands.sh",
    "cleanup_plan.csv",
    "cleanup_executed.csv",
    "FINAL_STATUS.md",
}

BIG_BASE_FILES = {
    "rar_points.parquet",
    "rar_points.csv",
    "galaxy_scatter_dm.csv",
    "galaxy_scatter_dm_with_env.csv",
    "master_catalog.csv",
    "dataset_manifest.json",
}

ARCHIVE_SUFFIXES = (".tgz", ".tar", ".tar.gz", ".zip")
IGNORE_BASENAMES = {"README_DROP_HERE.txt", ".DS_Store"}


@dataclass
class Action:
    source_path: str
    dest_path: str
    category: str
    action: str
    size_bytes: int
    sha256_12: str


def sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def file_size(path: Path) -> int:
    return path.stat().st_size


def classify(path: Path) -> str:
    name = path.name
    lower_name = name.lower()
    lower_full = str(path).lower()

    if lower_name.endswith(ARCHIVE_SUFFIXES):
        return "archive"

    if name in SESSION_RECOVERY_FILES:
        return "session_recovery"

    if name == "compute_big_base_qc.py":
        return "pipeline"

    if (name in BIG_BASE_FILES) or ("big_base" in lower_full):
        return "big_base_dataset"

    return "unclassified"


def destination_for(path: Path, category: str, source_root: Path) -> Path:
    name = path.name

    if category == "archive":
        return DEST_ARCHIVES / name
    if category == "session_recovery":
        return DEST_SESSION / name
    if category == "pipeline":
        return DEST_PIPELINE / name
    if category == "big_base_dataset":
        if name in {"master_catalog.csv", "dataset_manifest.json"}:
            return DEST_BIG_BASE_META / name
        return DEST_BIG_BASE / name

    rel = path.relative_to(source_root)
    return DEST_SESSION_UNCLASSIFIED / rel


def with_collision_suffix(path: Path, timestamp_tag: str) -> Path:
    stem = path.stem
    suffix = "".join(path.suffixes)
    parent = path.parent
    candidate = parent / f"{stem}__incoming_{timestamp_tag}{suffix}"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = parent / f"{stem}__incoming_{timestamp_tag}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def ensure_dirs() -> None:
    for d in [
        DEST_SESSION,
        DEST_SESSION_UNCLASSIFIED,
        DEST_PIPELINE,
        DEST_ARCHIVES,
        DEST_BIG_BASE,
        DEST_BIG_BASE_META,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def gather_files(source_root: Path) -> list[Path]:
    files = [
        p
        for p in source_root.rglob("*")
        if p.is_file() and p.name not in IGNORE_BASENAMES and not p.name.startswith(".")
    ]
    return sorted(files)


def copy_one(src: Path, dst: Path, timestamp_tag: str, dry_run: bool) -> tuple[Path, str]:
    src_hash = sha256_12(src)

    if dst.exists():
        dst_hash = sha256_12(dst)
        if src_hash == dst_hash:
            return dst, "skipped_identical"
        dst = with_collision_suffix(dst, timestamp_tag)
        if dry_run:
            return dst, "would_copy_collision_suffix"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst, "copied_collision_suffix"

    if dry_run:
        return dst, "would_copy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst, "copied"


def write_logs(actions: list[Action], dry_run: bool) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "organize_remaining_tng_dryrun" if dry_run else "organize_remaining_tng"
    csv_path = DEST_SESSION / f"{prefix}_{timestamp}.csv"
    json_path = DEST_SESSION / f"{prefix}_{timestamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "dest_path",
                "category",
                "action",
                "size_bytes",
                "sha256_12",
            ],
        )
        writer.writeheader()
        for a in actions:
            writer.writerow(a.__dict__)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "n_files": len(actions),
        "counts_by_action": {},
        "counts_by_category": {},
    }
    for a in actions:
        summary["counts_by_action"][a.action] = summary["counts_by_action"].get(a.action, 0) + 1
        summary["counts_by_category"][a.category] = summary["counts_by_category"].get(a.category, 0) + 1

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize files from Remaining_TNG into repo structure.")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help=f"Source folder to scan (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not copy files.",
    )
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    ensure_dirs()
    files = gather_files(source_root)
    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    actions: list[Action] = []

    print(f"[INFO] source={source_root}")
    print(f"[INFO] files_found={len(files)}")
    for src in files:
        category = classify(src)
        dst = destination_for(src, category, source_root)
        final_dst, action = copy_one(src, dst, timestamp_tag=timestamp_tag, dry_run=args.dry_run)
        actions.append(
            Action(
                source_path=str(src),
                dest_path=str(final_dst),
                category=category,
                action=action,
                size_bytes=file_size(src),
                sha256_12=sha256_12(src),
            )
        )

    csv_log, json_log = write_logs(actions, dry_run=args.dry_run)
    print(f"[DONE] log_csv={csv_log}")
    print(f"[DONE] log_json={json_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
