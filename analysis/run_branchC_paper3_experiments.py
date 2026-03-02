#!/usr/bin/env python3
"""Single source-of-truth Branch C entrypoint (bec-dark-matter only)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_REPO = Path("/Users/russelllicht/bec-dark-matter").resolve()
BRANCHC_SCRIPT = EXPECTED_REPO / "analysis" / "paper3" / "paper3_branchC_experiments.py"
DEFAULT_DATASET = EXPECTED_REPO / "analysis" / "results" / "rar_points_unified.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
        return out
    except Exception:
        return "UNKNOWN"


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_repo_guard(repo_root: Path, dataset_path: Path) -> None:
    cwd = Path.cwd().resolve()
    if cwd != EXPECTED_REPO:
        raise RuntimeError(
            "Wrong repo execution context. "
            f"Current cwd={cwd}. Expected cwd={EXPECTED_REPO}. "
            "Run this from /Users/russelllicht/bec-dark-matter only."
        )

    required = [
        repo_root / "analysis" / "paper3" / "paper3_bridge_pack.py",
        repo_root / "analysis" / "paper3" / "paper3_branchC_experiments.py",
        dataset_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required files for Branch C:\n- " + "\n- ".join(missing))


def backup_replaced_csvs(out_dir: Path) -> List[str]:
    backed_up: List[str] = []
    if not out_dir.exists():
        return backed_up
    existing_csvs = sorted([p for p in out_dir.rglob("*.csv") if p.is_file()])
    if not existing_csvs:
        return backed_up

    backup_dir = out_dir / f"backup_replaced_csvs_{utc_stamp()}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in existing_csvs:
        rel = p.relative_to(out_dir)
        tgt = backup_dir / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, tgt)
        backed_up.append(str(tgt))
    return backed_up


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Branch C C1-C3 experiments with strict repo guards.")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--smoke", action="store_true", help="Fast smoke mode for CI/checks.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    repo_root = EXPECTED_REPO
    dataset_path = Path(args.dataset).expanduser().resolve()
    ensure_repo_guard(repo_root=repo_root, dataset_path=dataset_path)

    ts = utc_stamp()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (repo_root / "outputs" / "paper3_high_density" / f"BRANCHC_C1C2C3_{ts}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    backed_up_csvs = backup_replaced_csvs(out_dir)

    dataset_sha = sha256_file(dataset_path)
    git_sha = git_head(repo_root)

    cmd: List[str] = [
        sys.executable,
        str(BRANCHC_SCRIPT),
        "--rar_points_file",
        str(dataset_path),
        "--out_dir",
        str(out_dir),
        "--seed",
        str(args.seed),
    ]

    if args.smoke:
        cmd.extend([
            "--n_perm",
            "200",
            "--n_perm_fallback",
            "100",
            "--n_boot_bin",
            "120",
            "--n_bins",
            "10",
            "--min_bin_group_points",
            "20",
        ])

    stamp: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "git_head": git_sha,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "output_dir": str(out_dir),
        "entrypoint": str(Path(__file__).resolve()),
        "command": cmd,
        "smoke_mode": bool(args.smoke),
        "backup_replaced_csvs": backed_up_csvs,
    }

    print("=== RUN STAMP (Branch C) ===", flush=True)
    print(f"repo_root: {stamp['repo_root']}", flush=True)
    print(f"git_head: {stamp['git_head']}", flush=True)
    print(f"dataset_sha256: {stamp['dataset_sha256']}", flush=True)
    print(f"output_dir: {stamp['output_dir']}", flush=True)
    print("============================", flush=True)

    # Write pre-run stamp, then overwrite with final status afterwards.
    stamp_path = out_dir / "run_stamp_branchC.json"
    stamp["status"] = "running"
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")

    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        stamp["status"] = "ok"
    except subprocess.CalledProcessError as exc:
        stamp["status"] = f"failed:{exc.returncode}"
        stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        raise

    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"[BRANCHC ENTRYPOINT] done: {out_dir}")


if __name__ == "__main__":
    main()
