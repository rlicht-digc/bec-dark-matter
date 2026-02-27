#!/usr/bin/env python3
"""
TNG dataset manifest + lineage audit utilities.

Usage examples:
  python3 tng_dataset_lineage.py build-manifest \
    --dataset-root /Users/russelllicht/TNG_RAR_LATEST_GOOD \
    --dataset-id TNG_RAR_3000x50_SOFT1p5_RUN201626 \
    --soft-kpc 1.5 \
    --intended-use "dev sample"

  python3 tng_dataset_lineage.py audit-lineage \
    --results-root /Users/russelllicht/bec-dark-matter/analysis/results \
    --manifest /Users/russelllicht/TNG_RAR_LATEST_GOOD/meta/dataset_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SID_CANDIDATES = ("SubhaloID", "subhalo_id", "galaxy", "gal_id")


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_hash(path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(path), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def pick_id_col(cols: Sequence[str]) -> Optional[str]:
    cset = set(cols)
    for c in SID_CANDIDATES:
        if c in cset:
            return c
    return None


def load_selection_spec(raw: Optional[str]) -> Dict[str, object]:
    if not raw:
        return {}
    p = Path(raw)
    if p.exists() and p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"selection_spec_raw": raw}
    try:
        return json.loads(raw)
    except Exception:
        return {"selection_spec_raw": raw}


def infer_dataset_id(n_gal: int, n_radii_mode: int, run_tag: str, soft_kpc: Optional[float]) -> str:
    soft = "SOFTunk" if soft_kpc is None else f"SOFT{str(soft_kpc).replace('.', 'p')}"
    return f"TNG_RAR_{n_gal}x{n_radii_mode}_{soft}_RUN{run_tag}"


def build_manifest(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    rar_points = Path(args.rar_points).expanduser().resolve() if args.rar_points else None
    galaxy_scatter = Path(args.galaxy_scatter).expanduser().resolve() if args.galaxy_scatter else None
    master_catalog = Path(args.master_catalog).expanduser().resolve() if args.master_catalog else None
    tng_mass_profiles = Path(args.tng_mass_profiles).expanduser().resolve() if args.tng_mass_profiles else None

    if dataset_root:
        if rar_points is None:
            p = dataset_root / "rar_points.parquet"
            rar_points = p if p.exists() else None
        if galaxy_scatter is None:
            p = dataset_root / "galaxy_scatter_dm.csv"
            galaxy_scatter = p if p.exists() else None
        if master_catalog is None:
            p = dataset_root / "meta" / "master_catalog.csv"
            master_catalog = p if p.exists() else None
        if tng_mass_profiles is None:
            p = dataset_root / "tng_mass_profiles.npz"
            tng_mass_profiles = p if p.exists() else None

    if rar_points is None or not rar_points.exists():
        raise FileNotFoundError("rar_points.parquet is required (provide --rar-points or --dataset-root).")

    df = pd.read_parquet(rar_points)
    sid_col = pick_id_col(df.columns)
    if sid_col is None:
        raise ValueError(f"Could not find ID column in {rar_points} from candidates {SID_CANDIDATES}.")

    n_points = int(len(df))
    n_gal = int(df[sid_col].nunique())
    ppg = df.groupby(sid_col).size().astype(int)
    n_radii_mode = int(ppg.mode().iloc[0]) if len(ppg) else 0
    n_radii_median = float(np.median(ppg)) if len(ppg) else np.nan
    n_radii_min = int(ppg.min()) if len(ppg) else 0
    n_radii_max = int(ppg.max()) if len(ppg) else 0

    run_tag = args.run_tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = args.dataset_id or infer_dataset_id(n_gal, n_radii_mode, run_tag, args.soft_kpc)
    declared_n_radii = int(args.n_radii_declared) if args.n_radii_declared is not None else n_radii_mode

    warnings: List[str] = []
    if n_radii_min != n_radii_max:
        warnings.append(
            f"Points per galaxy are not uniform: min={n_radii_min}, mode={n_radii_mode}, max={n_radii_max}."
        )
    expected = n_gal * declared_n_radii
    if expected != n_points:
        warnings.append(
            f"n_points mismatch against n_gal*declared_n_radii: {n_points} vs {n_gal}*{declared_n_radii}={expected}."
        )

    compat: Dict[str, object] = {}

    if master_catalog and master_catalog.exists():
        mc = pd.read_csv(master_catalog)
        mc_sid = pick_id_col(mc.columns)
        if mc_sid is None:
            warnings.append(f"master_catalog missing ID column: {master_catalog}")
        else:
            a = set(df[sid_col].astype(str))
            b = set(mc[mc_sid].astype(str))
            compat["master_catalog"] = {
                "path": str(master_catalog),
                "n_rows": int(len(mc)),
                "n_unique_ids": int(mc[mc_sid].nunique()),
                "ids_in_rar_not_in_master": int(len(a - b)),
                "ids_in_master_not_in_rar": int(len(b - a)),
            }
            if len(a - b) > 0 or len(b - a) > 0:
                warnings.append("master_catalog IDs do not perfectly match rar_points IDs.")

    if galaxy_scatter and galaxy_scatter.exists():
        gs = pd.read_csv(galaxy_scatter)
        gs_sid = pick_id_col(gs.columns)
        if gs_sid is None:
            warnings.append(f"galaxy_scatter file missing ID column: {galaxy_scatter}")
        else:
            a = set(df[sid_col].astype(str))
            b = set(gs[gs_sid].astype(str))
            compat["galaxy_scatter_dm"] = {
                "path": str(galaxy_scatter),
                "n_rows": int(len(gs)),
                "n_unique_ids": int(gs[gs_sid].nunique()),
                "ids_in_rar_not_in_scatter": int(len(a - b)),
                "ids_in_scatter_not_in_rar": int(len(b - a)),
            }
            if len(a - b) > 0 or len(b - a) > 0:
                warnings.append("galaxy_scatter IDs do not perfectly match rar_points IDs.")

    files_meta = {
        "rar_points": {
            "path": str(rar_points),
            "size_bytes": rar_points.stat().st_size,
            "md5": file_md5(rar_points),
            "mtime_utc": dt.datetime.fromtimestamp(rar_points.stat().st_mtime, tz=dt.timezone.utc).isoformat(),
        }
    }
    for key, p in [
        ("master_catalog", master_catalog),
        ("galaxy_scatter_dm", galaxy_scatter),
        ("tng_mass_profiles_npz", tng_mass_profiles),
    ]:
        if p and p.exists():
            files_meta[key] = {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "md5": file_md5(p),
                "mtime_utc": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat(),
            }

    selection = load_selection_spec(args.selection_spec)

    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "created_utc": now_utc_iso(),
        "run_tag": run_tag,
        "dataset_root": str(dataset_root) if dataset_root else str(rar_points.parent),
        "intended_use": args.intended_use or "unspecified",
        "notes": args.notes or "",
        "counts": {
            "n_galaxies": n_gal,
            "n_points": n_points,
            "n_radii_mode": n_radii_mode,
            "n_radii_median": n_radii_median,
            "n_radii_min": n_radii_min,
            "n_radii_max": n_radii_max,
            "declared_n_radii": declared_n_radii,
        },
        "radii_grid": {
            "declared_n_radii": declared_n_radii,
            "soft_kpc": args.soft_kpc,
            "r_min_factor": args.r_min_factor,
            "r_max_factor": args.r_max_factor,
        },
        "selection_criteria": selection,
        "files": files_meta,
        "compatibility_checks": compat,
        "warnings": warnings,
        "git_hash": git_hash(Path(__file__).resolve().parents[2]),
    }

    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        root_for_output = dataset_root or rar_points.parent
        out = root_for_output / "meta" / "dataset_manifest.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    print("=" * 72)
    print("DATASET MANIFEST WRITTEN")
    print("=" * 72)
    print(f"dataset_id: {dataset_id}")
    print(f"output: {out}")
    print(f"n_galaxies={n_gal}, n_points={n_points}, mode_n_radii={n_radii_mode}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("warnings: none")
    return 0


def extract_dataset_id_from_json(path: Path) -> Optional[str]:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(obj, dict):
        for key in ("dataset_id", "input_dataset_id", "source_dataset_id"):
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                return obj[key].strip()
    return None


def extract_dataset_id_from_csv(path: Path) -> Optional[str]:
    try:
        df = pd.read_csv(path, nrows=200)
    except Exception:
        return None
    for col in ("dataset_id", "input_dataset_id", "source_dataset_id"):
        if col in df.columns:
            vals = [str(v) for v in df[col].dropna().unique() if str(v).strip()]
            if len(vals) == 1:
                return vals[0]
    return None


def extract_dataset_id_from_parquet(path: Path) -> Optional[str]:
    try:
        df = pd.read_parquet(path, columns=[c for c in ("dataset_id", "input_dataset_id", "source_dataset_id") if c])
    except Exception:
        return None
    for col in ("dataset_id", "input_dataset_id", "source_dataset_id"):
        if col in df.columns:
            vals = [str(v) for v in pd.Series(df[col]).dropna().unique() if str(v).strip()]
            if len(vals) == 1:
                return vals[0]
    return None


def audit_lineage(args: argparse.Namespace) -> int:
    known_ids: Dict[str, str] = {}
    for mp in args.manifest:
        p = Path(mp).expanduser().resolve()
        if not p.exists():
            print(f"warning: manifest not found: {p}")
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:
            print(f"warning: manifest unreadable json: {p}")
            continue
        did = obj.get("dataset_id")
        if isinstance(did, str) and did.strip():
            known_ids[did.strip()] = str(p)

    if not known_ids:
        raise ValueError("No valid dataset IDs loaded from --manifest.")

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        raise FileNotFoundError(results_root)

    exts = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())
    rows = []

    for p in results_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue

        inferred = None
        source = None

        # filename convention: *__DATASETID.*
        name = p.name
        for did in known_ids:
            if f"__{did}" in name:
                inferred = did
                source = "filename"
                break

        if inferred is None:
            if p.suffix.lower() == ".json":
                inferred = extract_dataset_id_from_json(p)
                source = "json_key" if inferred else None
            elif p.suffix.lower() == ".csv":
                inferred = extract_dataset_id_from_csv(p)
                source = "csv_column" if inferred else None
            elif p.suffix.lower() == ".parquet":
                inferred = extract_dataset_id_from_parquet(p)
                source = "parquet_column" if inferred else None

        status = "ok" if inferred in known_ids else "missing_dataset_id"
        rows.append(
            {
                "path": str(p),
                "filename": p.name,
                "detected_dataset_id": inferred,
                "detection_source": source,
                "status": status,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["status", "path"]).reset_index(drop=True)
    missing = int((out_df["status"] != "ok").sum()) if len(out_df) else 0

    out_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else results_root / "dataset_lineage_audit.csv"
    out_json = Path(args.output_json).expanduser().resolve() if args.output_json else results_root / "dataset_lineage_audit.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    summary = {
        "created_utc": now_utc_iso(),
        "results_root": str(results_root),
        "known_dataset_ids": sorted(known_ids.keys()),
        "n_files_scanned": int(len(out_df)),
        "n_missing_dataset_id": missing,
        "strict_failed": bool(args.strict and missing > 0),
        "outputs": {"csv": str(out_csv), "json": str(out_json)},
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print("DATASET LINEAGE AUDIT")
    print("=" * 72)
    print(f"results_root: {results_root}")
    print(f"known_dataset_ids: {', '.join(sorted(known_ids.keys()))}")
    print(f"files_scanned: {len(out_df)}")
    print(f"missing_dataset_id: {missing}")
    print(f"csv: {out_csv}")
    print(f"json: {out_json}")

    if args.strict and missing > 0:
        print("strict mode: FAIL (files missing dataset_id)")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TNG dataset manifest + lineage audit tools")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-manifest", help="Build dataset_manifest.json for a TNG dataset")
    b.add_argument("--dataset-root", type=str, default=None, help="Root dir containing dataset files")
    b.add_argument("--rar-points", type=str, default=None, help="Path to rar_points.parquet")
    b.add_argument("--galaxy-scatter", type=str, default=None, help="Path to galaxy_scatter_dm.csv")
    b.add_argument("--master-catalog", type=str, default=None, help="Path to meta/master_catalog.csv")
    b.add_argument("--tng-mass-profiles", type=str, default=None, help="Optional path to tng_mass_profiles.npz")
    b.add_argument("--dataset-id", type=str, default=None, help="Explicit dataset ID")
    b.add_argument("--run-tag", type=str, default=None, help="Run tag suffix for inferred dataset ID")
    b.add_argument("--n-radii-declared", type=int, default=None, help="Declared number of radii per galaxy")
    b.add_argument("--soft-kpc", type=float, default=None, help="SOFT_KPC used in extraction")
    b.add_argument("--r-min-factor", type=float, default=None, help="Radial minimum multiplier")
    b.add_argument("--r-max-factor", type=float, default=None, help="Radial maximum multiplier")
    b.add_argument("--selection-spec", type=str, default=None, help="JSON string or path with selection criteria")
    b.add_argument("--intended-use", type=str, default="unspecified", help="dev sample / final sample / etc")
    b.add_argument("--notes", type=str, default="", help="Free-form notes")
    b.add_argument("--output", type=str, default=None, help="Output manifest path")
    b.set_defaults(func=build_manifest)

    a = sub.add_parser("audit-lineage", help="Audit result files for dataset_id lineage")
    a.add_argument("--results-root", type=str, required=True, help="Directory of analysis outputs to scan")
    a.add_argument("--manifest", type=str, action="append", required=True, help="dataset_manifest.json (repeatable)")
    a.add_argument("--extensions", type=str, default="csv,json,parquet", help="Comma-separated extensions to scan")
    a.add_argument("--output-csv", type=str, default=None, help="Audit CSV output path")
    a.add_argument("--output-json", type=str, default=None, help="Audit summary JSON path")
    a.add_argument("--strict", action="store_true", help="Exit non-zero if any files lack dataset_id lineage")
    a.set_defaults(func=audit_lineage)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

