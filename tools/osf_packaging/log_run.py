#!/usr/bin/env python3
"""Create and validate append-only run logs under logs/runs/."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "run_log_schema.json"
RUN_LOG_DIR = REPO_ROOT / "logs" / "runs"


def parse_csv(raw: str) -> List[str]:
    values = [p.strip() for p in raw.split(",")]
    return [p for p in values if p]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso_utc() -> str:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def now_compact_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(text: str, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        slug = "run"
    return slug[:max_len]


def git_head(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def load_todos(path: Path | None) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        return [f"TODO source missing: {path}"]

    lines = path.read_text(errors="replace").splitlines()
    todos = [line.strip() for line in lines if "TODO" in line.upper() and line.strip()]
    if todos:
        return todos

    non_empty = [line.strip() for line in lines if line.strip()]
    if non_empty:
        summary = " | ".join(non_empty[:3])
        return [f"No explicit TODO lines. Summary: {summary}"]
    return []


def _check_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate_date_time_utc(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate_document(document: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    required = schema.get("required", [])
    for field in required:
        if field not in document:
            errors.append(f"Missing required field: {field}")

    additional_allowed = schema.get("additionalProperties", True)
    properties = schema.get("properties", {})

    if additional_allowed is False:
        for field in document:
            if field not in properties:
                errors.append(f"Unexpected field: {field}")

    for field, rule in properties.items():
        if field not in document:
            continue
        value = document[field]

        expected_type = rule.get("type")
        if expected_type and not _check_type(value, expected_type):
            errors.append(f"Field {field} expected type {expected_type}")
            continue

        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"Field {field} must be one of {rule['enum']}")

        if isinstance(value, str):
            if "minLength" in rule and len(value) < int(rule["minLength"]):
                errors.append(f"Field {field} must have length >= {rule['minLength']}")
            pattern = rule.get("pattern")
            if pattern and re.fullmatch(pattern, value) is None:
                errors.append(f"Field {field} does not match pattern {pattern}")
            if rule.get("format") == "date-time" and not _validate_date_time_utc(value):
                errors.append(f"Field {field} must be an ISO-8601 UTC timestamp")

        if isinstance(value, list) and "items" in rule:
            item_rule = rule["items"]
            item_type = item_rule.get("type")
            item_pattern = item_rule.get("pattern")
            item_min_len = item_rule.get("minLength")
            for i, item in enumerate(value):
                if item_type and not _check_type(item, item_type):
                    errors.append(f"Field {field}[{i}] expected type {item_type}")
                    continue
                if isinstance(item, str):
                    if item_min_len is not None and len(item) < int(item_min_len):
                        errors.append(
                            f"Field {field}[{i}] must have length >= {item_min_len}"
                        )
                    if item_pattern and re.fullmatch(item_pattern, item) is None:
                        errors.append(
                            f"Field {field}[{i}] does not match pattern {item_pattern}"
                        )

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create append-only run log JSON and validate against schema."
    )
    parser.add_argument("--tool", required=True, choices=["codex", "claude", "human"])
    parser.add_argument("--purpose", required=True, help="Short purpose text.")
    parser.add_argument(
        "--prompt-file",
        help="Optional path to prompt text. If omitted, purpose text is hashed.",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated input paths scanned during the run.",
    )
    parser.add_argument(
        "--outputs",
        required=True,
        help="Comma-separated output paths changed by the run.",
    )
    parser.add_argument(
        "--todos-file",
        help="Optional file to scan for TODO lines or fallback summary text.",
    )
    parser.add_argument(
        "--correction-of",
        help="Optional prior run_id that this new log corrects.",
    )
    parser.add_argument(
        "--supersedes",
        help="Optional prior run_id that this log supersedes.",
    )
    parser.add_argument(
        "--git-commit",
        help="Explicit commit SHA to record. If omitted, current HEAD is used.",
    )
    parser.add_argument("--notes", help="Optional freeform notes.")
    parser.add_argument(
        "--warnings",
        help="Optional comma-separated warning strings.",
    )
    parser.add_argument(
        "--task-id",
        help="Optional mailbox task ID (e.g. T001). Links run log to a mailbox task.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    prompt_path = Path(args.prompt_file).resolve() if args.prompt_file else None
    if prompt_path is not None:
        if not prompt_path.exists():
            print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
            return 2
        prompt_sha = sha256_bytes(prompt_path.read_bytes())
    else:
        prompt_sha = sha256_bytes(args.purpose.encode("utf-8"))

    ts_compact = now_compact_utc()
    run_id = f"{ts_compact}_{args.tool}_{slugify(args.purpose)}_{uuid.uuid4().hex[:8]}"
    log_doc: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": now_iso_utc(),
        "tool": args.tool,
        "purpose": args.purpose.strip(),
        "prompt_sha256": prompt_sha,
        "inputs_scanned": parse_csv(args.inputs),
        "outputs_changed": parse_csv(args.outputs),
        "git_commit": args.git_commit.strip() if args.git_commit else git_head(REPO_ROOT),
        "todos_remaining": load_todos(Path(args.todos_file).resolve() if args.todos_file else None),
    }

    if args.correction_of:
        log_doc["correction_of"] = args.correction_of.strip()
    if args.supersedes:
        log_doc["supersedes"] = args.supersedes.strip()
    if args.notes:
        log_doc["notes"] = args.notes.strip()
    if args.warnings:
        log_doc["warnings"] = parse_csv(args.warnings)
    if args.task_id:
        log_doc["task_id"] = args.task_id.strip()

    schema = json.loads(RUN_SCHEMA_PATH.read_text())
    validation_errors = validate_document(log_doc, schema)
    if validation_errors:
        print("Run log validation failed:", file=sys.stderr)
        for err in validation_errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_LOG_DIR / f"{run_id}.json"
    try:
        with out_path.open("x", encoding="utf-8") as handle:
            json.dump(log_doc, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        # Extremely unlikely collision; do not overwrite.
        run_id_alt = f"{run_id}_{uuid.uuid4().hex[:4]}"
        log_doc["run_id"] = run_id_alt
        out_path = RUN_LOG_DIR / f"{run_id_alt}.json"
        with out_path.open("x", encoding="utf-8") as handle:
            json.dump(log_doc, handle, indent=2)
            handle.write("\n")

    print(str(out_path.relative_to(REPO_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
