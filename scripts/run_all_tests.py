#!/usr/bin/env python3
"""
Single-entry harness for full BEC test suite:
  - RAR/TNG referee tests 1-5
  - BH viability modes scan

Usage:
  python scripts/run_all_tests.py --config config/run_all.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import scipy
import yaml


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(payload), indent=2))


def git_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_default_config(repo_root: Path) -> Dict[str, Any]:
    return {
        "seed": 42,
        "project_root": str(repo_root),
        "output_base": "outputs",
        "tests": {"rar": True, "bh": True},
        "rar": {
            "n_shuffles": 1000,
            "strict_missing_inputs": True,
            "enabled_tests": {
                "test1_phase_peak_null": True,
                "test2_mass_matched_phase": True,
                "test3_xi_organizing": True,
                "test4_alpha_star_convergence": True,
                "test5_dataset_lineage": True,
            },
            "tng_sources": {
                "test2_mass_matched_phase": "clean_48133_points",
                "test3_xi_organizing": "clean_3000_points",
                "test4_alpha_3k": "clean_3000_points",
                "test4_alpha_48k": "clean_48133_points",
            },
        },
        "bh": {
            "enabled": True,
            "use_kappa_scan": True,
            "kappa_min": -8.0,
            "kappa_max": 8.0,
            "nk": 200,
            "log_rho_min": -12.0,
            "log_rho_max": 120.0,
            "nr": 200,
            "mass_min_msun": 3.0,
            "mass_max_msun": 1e9,
            "nm": 220,
            "u_min": 0.0,
            "u_max": 8.0,
            "nu": 25,
            "mode1_radius_policy": "max_rs_rc",
            "strict_checks": True,
            "spot_check_samples": 8,
        },
        "regression_lock": {
            "enabled": False,
            "expected_summary_path": "",
            "tolerance_abs": 1e-6,
            "fail_on_mismatch": True,
            "keypaths": [
                "rar_tests.test1_phase_peak_null.metrics.real.mu_peak",
                "rar_tests.test1_phase_peak_null.metrics.real.daic",
                "rar_tests.test2_mass_matched_phase.metrics.ks_pvalue_mass",
                "rar_tests.test3_xi_organizing.metrics.sparc.concentration_pvalue",
                "bh_modes.mode2.global_stats.frac_DEC_any.median",
                "bh_modes.mode2.global_stats.median_required_u_for_DEC.median",
            ],
        },
    }


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_finite_df(df: pd.DataFrame, cols: List[str], label: str) -> None:
    for c in cols:
        assert_true(c in df.columns, f"{label}: missing required column '{c}'")
        arr = df[c].to_numpy(dtype=float)
        assert_true(np.all(np.isfinite(arr)), f"{label}: non-finite values in column '{c}'")


def _p_in_unit_interval(p: Any) -> bool:
    if p is None:
        return False
    try:
        x = float(p)
    except Exception:
        return False
    return np.isfinite(x) and (0.0 <= x <= 1.0)


def _move_figures_to_subdir(run_dir: Path, fig_dir: Path) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []
    for p in sorted(run_dir.glob("fig_*.png")):
        dst = fig_dir / p.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(p), str(dst))
        moved.append(str(dst))
    return moved


def _relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def _get_nested(obj: Dict[str, Any], keypath: str) -> Any:
    cur: Any = obj
    for k in keypath.split("."):
        if not isinstance(cur, dict) or (k not in cur):
            raise KeyError(keypath)
        cur = cur[k]
    return cur


def _run_regression_lock(cfg: Dict[str, Any], summary_all: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "status": "SKIPPED"}

    exp_path = str(cfg.get("expected_summary_path", "")).strip()
    if not exp_path:
        return {"enabled": True, "status": "SKIPPED", "reason": "expected_summary_path not set"}
    p = Path(exp_path).expanduser().resolve()
    if not p.exists():
        return {"enabled": True, "status": "SKIPPED", "reason": f"expected summary missing: {p}"}

    expected = json.loads(p.read_text())
    tol = float(cfg.get("tolerance_abs", 1e-6))
    keypaths = list(cfg.get("keypaths", []))
    failures: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []

    for kp in keypaths:
        try:
            a = _get_nested(summary_all, kp)
            b = _get_nested(expected, kp)
        except Exception:
            failures.append({"keypath": kp, "reason": "missing keypath in current or expected"})
            continue

        if a is None or b is None:
            ok = a is b
            checks.append({"keypath": kp, "current": a, "expected": b, "ok": ok})
            if not ok:
                failures.append({"keypath": kp, "current": a, "expected": b, "reason": "None mismatch"})
            continue

        try:
            fa = float(a)
            fb = float(b)
            if not np.isfinite(fa) or not np.isfinite(fb):
                raise ValueError("non-finite")
            diff = abs(fa - fb)
            ok = diff <= tol
            checks.append({"keypath": kp, "current": fa, "expected": fb, "abs_diff": diff, "ok": ok})
            if not ok:
                failures.append({"keypath": kp, "current": fa, "expected": fb, "abs_diff": diff, "tol": tol})
        except Exception:
            ok = a == b
            checks.append({"keypath": kp, "current": a, "expected": b, "ok": ok})
            if not ok:
                failures.append({"keypath": kp, "current": a, "expected": b, "reason": "non-numeric mismatch"})

    return {
        "enabled": True,
        "status": "PASS" if len(failures) == 0 else "FAIL",
        "expected_summary_path": str(p),
        "tolerance_abs": tol,
        "checks": checks,
        "failures": failures,
        "fail_on_mismatch": bool(cfg.get("fail_on_mismatch", True)),
    }


def run_rar_suite(
    repo_root: Path,
    run_dir: Path,
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    # Import locally so this script remains light if only BH is requested.
    sys.path.insert(0, str(repo_root))
    from analysis.pipeline import run_referee_required_tests as rr  # type: ignore

    failures: List[str] = []
    results: Dict[str, Any] = {}

    structure = rr.ensure_repo_structure(repo_root)
    results["structure_check"] = structure
    if not structure.get("ok", False):
        raise RuntimeError("Repository structure check failed.")

    sparc_df, sparc_meta = rr.load_sparc_points(repo_root)
    assert_finite_df(sparc_df, ["log_gbar", "log_gobs", "R_kpc", "log_res_use"], "SPARC points")

    tng_paths = rr.discover_tng_candidates(repo_root)
    strict_missing = bool(cfg.get("strict_missing_inputs", True))
    enabled = cfg.get("enabled_tests", {})
    n_shuffles = int(cfg.get("n_shuffles", 1000))
    rng = np.random.default_rng(seed)

    def fail(test_name: str, err: Exception) -> None:
        msg = f"{test_name}: {type(err).__name__}: {err}"
        failures.append(msg)
        results[test_name] = {"status": "FAIL", "reason": msg}

    # Test 1
    if bool(enabled.get("test1_phase_peak_null", True)):
        name = "test1_phase_peak_null"
        try:
            summary = rr.run_test1_phase_peak_null(sparc_df, out_dir=run_dir, rng=rng, n_shuffles=n_shuffles)
            assert_true(int(summary["n_bins_used"]) > 0, "Test1: n_bins_used must be > 0")
            assert_true(_p_in_unit_interval(summary["shuffle_A_within_galaxy"]["p_delta"]), "Test1: invalid p_delta A")
            assert_true(_p_in_unit_interval(summary["shuffle_A_within_galaxy"]["p_daic"]), "Test1: invalid p_daic A")
            assert_true(_p_in_unit_interval(summary["shuffle_B_galaxy_label"]["p_delta"]), "Test1: invalid p_delta B")
            assert_true(_p_in_unit_interval(summary["shuffle_B_galaxy_label"]["p_daic"]), "Test1: invalid p_daic B")
            results[name] = {"status": "PASS", "metrics": summary}
        except Exception as e:
            fail(name, e)

    # Test 5 (requested early)
    if bool(enabled.get("test5_dataset_lineage", True)):
        name = "test5_dataset_lineage"
        try:
            summary = rr.run_test5_dataset_lineage(repo_root, out_dir=run_dir, sparc_df=sparc_df, tng_paths=tng_paths)
            assert_true("contamination_note" in summary, "Test5: contamination_note missing")
            results[name] = {"status": "PASS", "metrics": summary}
        except Exception as e:
            fail(name, e)

    # Test 2
    if bool(enabled.get("test2_mass_matched_phase", True)):
        name = "test2_mass_matched_phase"
        try:
            key = str(cfg.get("tng_sources", {}).get("test2_mass_matched_phase", "clean_48133_points"))
            tng_path = tng_paths.get(key)
            summary = rr.run_test2_mass_matched_phase(
                repo_root,
                out_dir=run_dir,
                sparc_df=sparc_df,
                tng_points_path=tng_path,
                rng=rng,
            )
            if summary.get("status") == "BLOCKED" and strict_missing:
                raise FileNotFoundError(f"Test2 blocked: {summary.get('reason')}; expected TNG key={key}")
            if summary.get("status") == "OK":
                assert_true(int(summary["N_matched"]) > 0, "Test2: N_matched must be > 0")
                assert_true(_p_in_unit_interval(summary["ks_pvalue_mass"]), "Test2: invalid KS p-value")
            results[name] = {"status": "PASS", "metrics": summary}
        except Exception as e:
            fail(name, e)

    # Test 3
    if bool(enabled.get("test3_xi_organizing", True)):
        name = "test3_xi_organizing"
        try:
            key = str(cfg.get("tng_sources", {}).get("test3_xi_organizing", "clean_3000_points"))
            tng_path = tng_paths.get(key)
            summary = rr.run_test3_xi_organizing(
                out_dir=run_dir,
                sparc_df=sparc_df,
                tng_points_path=tng_path,
                rng=rng,
            )
            if summary.get("tng_status") == "BLOCKED" and strict_missing:
                raise FileNotFoundError(f"Test3 blocked: {summary.get('reason')}; expected TNG key={key}")
            assert_true(_p_in_unit_interval(summary["sparc"]["concentration_pvalue"]), "Test3: invalid SPARC concentration p")
            if isinstance(summary.get("tng"), dict) and summary["tng"].get("status") == "OK":
                assert_true(_p_in_unit_interval(summary["tng"]["concentration_pvalue"]), "Test3: invalid TNG concentration p")
            results[name] = {"status": "PASS", "metrics": summary}
        except Exception as e:
            fail(name, e)

    # Test 4
    if bool(enabled.get("test4_alpha_star_convergence", True)):
        name = "test4_alpha_star_convergence"
        try:
            key3 = str(cfg.get("tng_sources", {}).get("test4_alpha_3k", "clean_3000_points"))
            key48 = str(cfg.get("tng_sources", {}).get("test4_alpha_48k", "clean_48133_points"))
            summary = rr.run_test4_alpha_convergence(
                out_dir=run_dir,
                sparc_df=sparc_df,
                tng_3k_path=tng_paths.get(key3),
                tng_48k_path=tng_paths.get(key48),
                rng=rng,
            )
            if summary.get("status") == "BLOCKED":
                raise RuntimeError(f"Test4 blocked: {summary.get('reason')}")
            tng_status = summary.get("tng_status", {})
            if strict_missing:
                assert_true(tng_status.get("TNG_3k") == "OK", f"Test4: TNG_3k unavailable for key={key3}")
                assert_true(tng_status.get("TNG_48k") == "OK", f"Test4: TNG_48k unavailable for key={key48}")
            results[name] = {"status": "PASS", "metrics": summary}
        except Exception as e:
            fail(name, e)

    # Existence checks for expected artifacts.
    expected_outputs = [
        "summary_phase_peak_null.json",
        "fig_phase_null.png",
        "summary_mass_matched_phase.json",
        "fig_mass_matched_phase.png",
        "summary_xi_organizing.json",
        "fig_xi_organizing.png",
        "summary_alpha_star_convergence.json",
        "fig_alpha_star.png",
        "summary_dataset_lineage.json",
        "dataset_lineage.md",
    ]
    for name in expected_outputs:
        p = run_dir / name
        if p.exists():
            continue
        if strict_missing:
            failures.append(f"missing expected RAR artifact: {p}")

    return results, failures, {"sparc_meta": sparc_meta, "tng_paths": {k: (str(v) if v else None) for k, v in tng_paths.items()}}


def build_outputs_index(run_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "outputs_index.json":
            continue
        files.append(
            {
                "path": _relpath(p, run_dir),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    payload = {"timestamp_utc": iso_now(), "n_files": len(files), "meta": meta, "files": files}
    write_json(run_dir / "outputs_index.json", payload)
    return payload


def build_summary_md(summary_all: Dict[str, Any], run_dir: Path) -> str:
    rm = summary_all["run_meta"]
    lines: List[str] = []
    lines.append("# Consolidated Test Summary")
    lines.append("")
    lines.append("## Repro Checklist")
    lines.append("")
    lines.append(f"- git hash: `{rm.get('git_hash')}`")
    lines.append(f"- config hash: `{rm.get('config_hash')}`")
    lines.append(f"- seed: `{rm.get('seed')}`")
    lines.append(f"- platform/python: `{rm.get('platform')}` / `{rm.get('python')}`")
    lines.append(f"- command: `{rm.get('command')}`")
    lines.append("")

    lines.append("## RAR/TNG Tests")
    lines.append("")
    lines.append("| Test | Status | Key Metrics |")
    lines.append("|---|---|---|")
    for tname in [
        "test1_phase_peak_null",
        "test2_mass_matched_phase",
        "test3_xi_organizing",
        "test4_alpha_star_convergence",
        "test5_dataset_lineage",
    ]:
        tr = summary_all.get("rar_tests", {}).get(tname, {"status": "SKIPPED"})
        st = tr.get("status", "SKIPPED")
        if st != "PASS":
            lines.append(f"| {tname} | {st} | {tr.get('reason', 'n/a')} |")
            continue
        m = tr.get("metrics", {})
        if tname == "test1_phase_peak_null":
            km = f"mu_peak={m['real']['mu_peak']:.4f}, daic={m['real']['daic']:.3f}, pΔ(A/B)={m['shuffle_A_within_galaxy']['p_delta']:.4g}/{m['shuffle_B_galaxy_label']['p_delta']:.4g}"
        elif tname == "test2_mass_matched_phase":
            if m.get("status") == "OK":
                km = f"N_matched={m['N_matched']}, KS p={m['ks_pvalue_mass']:.4g}, mu(S/T)={m['sparc']['mu_peak']:.4f}/{m['tng']['mu_peak']:.4f}"
            else:
                km = f"status={m.get('status')} ({m.get('reason')})"
        elif tname == "test3_xi_organizing":
            km = f"SPARC C={m['sparc']['concentration_C']:.4f}, p={m['sparc']['concentration_pvalue']:.4g}"
            if isinstance(m.get("tng"), dict) and m["tng"].get("status") == "OK":
                km += f"; TNG C={m['tng']['concentration_C']:.4f}, p={m['tng']['concentration_pvalue']:.4g}"
            else:
                km += f"; TNG={m.get('tng_status', m.get('tng',{}).get('status','BLOCKED'))}"
        elif tname == "test4_alpha_star_convergence":
            km = f"verdict={m.get('verdict')}"
        else:
            km = "lineage audit complete"
        lines.append(f"| {tname} | {st} | {km} |")
    lines.append("")

    lines.append("## BH Viability Modes")
    lines.append("")
    bh = summary_all.get("bh_modes", {})
    if bh.get("status") == "PASS":
        m2 = bh["summary"]["mode2"]["global_stats"]
        lines.append(
            f"- Mode2 frac_WEC_any median: `{m2['frac_WEC_any']['median']}`; "
            f"frac_DEC_any median: `{m2['frac_DEC_any']['median']}`; "
            f"median required u for DEC: `{m2['median_required_u_for_DEC']['median']}`"
        )
        lines.append(f"- Branch counts: `{bh['summary']['mode2']['branch_counts']}`")
    else:
        lines.append(f"- Status: `{bh.get('status')}`")
        lines.append(f"- Reason: `{bh.get('reason')}`")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    fig_dir = run_dir / "figures"
    for p in sorted(fig_dir.glob("*.png")):
        lines.append(f"- `{_relpath(p, run_dir)}`")
    for name in ["summary_all.json", "outputs_index.json", "summary_all.md"]:
        lines.append(f"- `{name}`")
    lines.append("")

    reg = summary_all.get("regression_lock", {})
    lines.append("## Regression Lock")
    lines.append("")
    lines.append(f"- enabled: `{reg.get('enabled')}`")
    lines.append(f"- status: `{reg.get('status')}`")
    if reg.get("status") == "FAIL":
        lines.append(f"- failures: `{len(reg.get('failures', []))}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full BEC test suite harness.")
    p.add_argument("--config", type=str, default="config/run_all.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    default_cfg = get_default_config(repo_root)

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg_raw_bytes = cfg_path.read_bytes()
    cfg_loaded = yaml.safe_load(cfg_raw_bytes) or {}
    cfg = deep_merge(default_cfg, cfg_loaded)
    config_hash = sha256_bytes(yaml.safe_dump(cfg, sort_keys=True).encode("utf-8"))

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    project_root = Path(str(cfg.get("project_root", repo_root))).expanduser().resolve()
    assert_true(project_root.exists(), f"project_root does not exist: {project_root}")
    out_base = project_root / str(cfg.get("output_base", "outputs"))
    run_dir = out_base / stamp_now()
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp_utc": iso_now(),
        "seed": seed,
        "repo_root": str(project_root),
        "git_hash": git_hash(project_root),
        "config_path": str(cfg_path),
        "config_hash": config_hash,
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "config_dump": cfg,
    }

    failures: List[str] = []
    rar_results: Dict[str, Any] = {}
    rar_meta: Dict[str, Any] = {}
    bh_result: Dict[str, Any] = {"status": "SKIPPED"}

    print("=" * 80)
    print("RUN ALL TESTS HARNESS")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Output dir:   {run_dir}")
    print(f"Config:       {cfg_path}")
    print(f"Seed:         {seed}")

    # RAR suite
    if bool(cfg.get("tests", {}).get("rar", True)):
        try:
            rar_results, rar_failures, rar_meta = run_rar_suite(
                repo_root=project_root,
                run_dir=run_dir,
                cfg=cfg.get("rar", {}),
                seed=seed,
            )
            failures.extend(rar_failures)
        except Exception as e:
            msg = f"RAR suite fatal: {type(e).__name__}: {e}"
            failures.append(msg)
            rar_results = {"fatal": {"status": "FAIL", "reason": msg, "traceback": traceback.format_exc()}}
    else:
        rar_results = {"status": "SKIPPED"}

    # BH suite
    if bool(cfg.get("tests", {}).get("bh", True)) and bool(cfg.get("bh", {}).get("enabled", True)):
        try:
            sys.path.insert(0, str(project_root))
            from scripts.run_bh_viability_modes import run_bh_viability_modes  # type: ignore

            bh_run = run_bh_viability_modes(
                config=cfg.get("bh", {}),
                output_dir=run_dir,
                figures_dir=fig_dir,
                verbose=True,
            )
            bh_result = {"status": "PASS", "summary_path": bh_run["summary_path"], "summary": bh_run["summary"]}
        except Exception as e:
            msg = f"BH suite fatal: {type(e).__name__}: {e}"
            failures.append(msg)
            bh_result = {"status": "FAIL", "reason": msg, "traceback": traceback.format_exc()}
    else:
        bh_result = {"status": "SKIPPED"}

    # Move RAR figures into figures/ after all runs.
    moved_figs = _move_figures_to_subdir(run_dir, fig_dir)

    summary_all: Dict[str, Any] = {
        "run_meta": run_meta,
        "rar_meta": rar_meta,
        "rar_tests": rar_results,
        "bh_modes": bh_result,
        "moved_figures": moved_figs,
        "failures": failures,
        "status": "PASS" if len(failures) == 0 else "FAIL",
    }

    # Optional regression lock against an expected summary.
    reg = _run_regression_lock(cfg.get("regression_lock", {}), summary_all)
    summary_all["regression_lock"] = reg
    if reg.get("status") == "FAIL" and bool(reg.get("fail_on_mismatch", True)):
        failures.append("Regression lock mismatch.")
        summary_all["failures"] = failures
        summary_all["status"] = "FAIL"

    write_json(run_dir / "summary_all.json", summary_all)

    md = build_summary_md(summary_all, run_dir)
    (run_dir / "summary_all.md").write_text(md)

    outputs_index = build_outputs_index(
        run_dir,
        meta={
            "git_hash": run_meta["git_hash"],
            "python": run_meta["python"],
            "numpy": run_meta["numpy"],
            "scipy": run_meta["scipy"],
            "matplotlib": run_meta["matplotlib"],
            "config_hash": run_meta["config_hash"],
            "seed": seed,
        },
    )
    _ = outputs_index

    print("\nOutput directory:")
    print(run_dir)
    print("\nHeadline:")
    if summary_all["status"] == "PASS":
        t1 = summary_all.get("rar_tests", {}).get("test1_phase_peak_null", {})
        if t1.get("status") == "PASS":
            m = t1["metrics"]["real"]
            print(f"  Test1 mu_peak={m['mu_peak']:.4f}, daic={m['daic']:.3f}")
        if bh_result.get("status") == "PASS":
            m2 = bh_result["summary"]["mode2"]["global_stats"]
            print(
                "  BH Mode2 medians: "
                f"WEC_any={m2['frac_WEC_any']['median']}, "
                f"DEC_any={m2['frac_DEC_any']['median']}, "
                f"u_DEC={m2['median_required_u_for_DEC']['median']}"
            )
    else:
        print("  FAIL")
        for f in failures:
            print(f"  - {f}")

    # Nonzero exit on failed assertions/tests.
    if summary_all["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
