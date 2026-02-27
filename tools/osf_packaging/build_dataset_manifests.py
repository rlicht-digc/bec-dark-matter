#!/usr/bin/env python3
"""Generate OSF dataset manifests and checksum files.

This script walks immediate dataset folders under the configured dataset roots
and writes:
  - manifest.csv
  - checksums.sha256

Manifest columns:
  relative_path, bytes, sha256, source_url, citation_tag, license_note

Policy notes:
  - third-party data are not relicensed here
  - `checksums.sha256` itself is not listed in manifest.csv
  - very large files can be skipped from hashing via --hash-max-bytes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


DEFAULT_HASH_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


DATASET_METADATA = {
    "raw_data/observational/alfalfa": {
        "source_url": "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/142/170",
        "citation_tag": "TODO_ALFALFA_MAIN",
        "license_note": "Third-party dataset; original terms apply (VizieR).",
    },
    "raw_data/observational/brouwer2021": {
        "source_url": "TODO_SOURCE_URL_Brouwer2021",
        "citation_tag": "Brouwer2021",
        "license_note": "Third-party dataset; original terms apply.",
    },
    "raw_data/observational/sparc": {
        "source_url": "http://astroweb.cwru.edu/SPARC/",
        "citation_tag": "Lelli2016;McGaugh2016;Lelli2017",
        "license_note": "Third-party dataset; original terms apply.",
    },
    "raw_data/observational/cluster_rar": {
        "source_url": "https://doi.org/10.3847/1538-4357/ab8e3d",
        "citation_tag": "Tian2020",
        "license_note": "Third-party dataset; original terms apply.",
    },
    "raw_data/observational/eagle_rar": {
        "source_url": "TODO_SOURCE_URL_EAGLE_RAR_TABLES",
        "citation_tag": "Schaye2015;Keller2017;Ludlow2017",
        "license_note": "Third-party dataset; original terms apply.",
    },
    "raw_data/tng/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED": {
        "source_url": "TODO_SOURCE_URL_IllustrisTNG",
        "citation_tag": "Pillepich2018",
        "license_note": "Derived dataset built from simulation products; verify upstream terms.",
    },
    "raw_data/tng/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE": {
        "source_url": "TODO_SOURCE_URL_IllustrisTNG",
        "citation_tag": "Pillepich2018",
        "license_note": "Derived dataset built from simulation products; verify upstream terms.",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_dataset_dirs(repo_root: Path, dataset_roots: list[str]) -> list[Path]:
    datasets: list[Path] = []
    for rel_root in dataset_roots:
        root = (repo_root / rel_root).resolve()
        if not root.exists():
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            datasets.append(child)
    return datasets


def collect_files(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(x for x in dataset_dir.rglob("*") if x.is_file()):
        if p.name == "checksums.sha256":
            continue
        files.append(p)
    return files


def build_manifest_rows(
    dataset_dir: Path,
    files: list[Path],
    source_url: str,
    citation_tag: str,
    license_note: str,
    hash_max_bytes: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in files:
        rel = path.relative_to(dataset_dir).as_posix()
        size = path.stat().st_size
        if hash_max_bytes > 0 and size > hash_max_bytes:
            digest = "SKIPPED_SIZE_GT_LIMIT"
        else:
            digest = sha256_file(path)
        rows.append(
            {
                "relative_path": rel,
                "bytes": str(size),
                "sha256": digest,
                "source_url": source_url,
                "citation_tag": citation_tag,
                "license_note": license_note,
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "relative_path",
        "bytes",
        "sha256",
        "source_url",
        "citation_tag",
        "license_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_checksums(
    dataset_dir: Path, files: list[Path], out_path: Path, hash_max_bytes: int
) -> None:
    entries = list(files)
    manifest_path = dataset_dir / "manifest.csv"
    if manifest_path.exists() and manifest_path not in entries:
        entries.append(manifest_path)
    entries = sorted(entries, key=lambda p: p.relative_to(dataset_dir).as_posix())

    with out_path.open("w", encoding="utf-8") as f:
        for path in entries:
            rel = path.relative_to(dataset_dir).as_posix()
            size = path.stat().st_size
            if hash_max_bytes > 0 and size > hash_max_bytes:
                f.write(
                    f"# SKIPPED {rel} bytes={size} reason=SIZE_GT_{hash_max_bytes}\n"
                )
            else:
                f.write(f"{sha256_file(path)}  {rel}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=["raw_data/observational", "raw_data/tng"],
        help="Dataset root containing immediate dataset folders. Repeatable.",
    )
    parser.add_argument(
        "--hash-max-bytes",
        type=int,
        default=DEFAULT_HASH_MAX_BYTES,
        help="Skip hashing files larger than this byte count (0 means no limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without writing files.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        print(f"ERROR: repo root not found: {repo_root}", file=sys.stderr)
        return 2

    dataset_dirs = iter_dataset_dirs(repo_root, args.dataset_root)
    if not dataset_dirs:
        print("No dataset directories found.")
        return 1

    for dataset_dir in dataset_dirs:
        rel_dataset = dataset_dir.relative_to(repo_root).as_posix()
        meta = DATASET_METADATA.get(
            rel_dataset,
            {
                "source_url": "TODO_SOURCE_URL",
                "citation_tag": "TODO_CITATION_TAG",
                "license_note": "Third-party dataset; original terms apply.",
            },
        )

        files = collect_files(dataset_dir)
        rows = build_manifest_rows(
            dataset_dir=dataset_dir,
            files=files,
            source_url=meta["source_url"],
            citation_tag=meta["citation_tag"],
            license_note=meta["license_note"],
            hash_max_bytes=args.hash_max_bytes,
        )
        manifest_path = dataset_dir / "manifest.csv"
        checksums_path = dataset_dir / "checksums.sha256"

        if args.dry_run:
            print(
                f"[DRY RUN] {rel_dataset}: "
                f"manifest_rows={len(rows)} -> {manifest_path}, {checksums_path}"
            )
            continue

        write_manifest(manifest_path, rows)
        # refresh file list to include new manifest and exclude checksums
        files_post = collect_files(dataset_dir)
        write_checksums(dataset_dir, files_post, checksums_path, args.hash_max_bytes)

        print(
            f"{rel_dataset}: "
            f"wrote manifest.csv ({len(rows)} rows), checksums.sha256"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
