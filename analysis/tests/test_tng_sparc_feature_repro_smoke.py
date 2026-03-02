#!/usr/bin/env python3
"""Smoke test for strict TNG↔SPARC feature reproduction pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _make_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    n_gal = 10
    n_pts = 12

    sparc_rows = []
    tng_rows = []
    for gi in range(n_gal):
        gname = f"SPARC_{gi:03d}"
        sid = 10000 + gi
        base_shift = rng.normal(0.0, 0.03)
        x = np.linspace(-12.2, -9.0, n_pts) + rng.normal(0, 0.04, n_pts)
        y_mean = x + 0.18 * np.exp(-0.5 * ((x + 9.92) / 0.35) ** 2)
        y_s = y_mean + base_shift + rng.normal(0, 0.06, n_pts)
        y_t = y_mean + 0.02 + rng.normal(0, 0.07, n_pts)
        r = np.linspace(0.8, 18.0, n_pts)
        for j in range(n_pts):
            sparc_rows.append(
                {
                    "galaxy": gname,
                    "galaxy_key": gname,
                    "source": "SPARC",
                    "log_gbar": float(x[j]),
                    "log_gobs": float(y_s[j]),
                    "R_kpc": float(r[j]),
                    "sigma_log_gobs": 0.05,
                }
            )
            tng_rows.append(
                {
                    "SubhaloID": int(sid),
                    "log_gbar": float(x[j] + rng.normal(0, 0.02)),
                    "log_gobs": float(y_t[j]),
                    "r_kpc": float(r[j] * (0.95 + 0.1 * rng.random())),
                    "lowres_flag": 0,
                }
            )

    sparc_csv = tmp_path / "sparc_unified.csv"
    tng_csv = tmp_path / "tng_points.csv"
    pd.DataFrame(sparc_rows).to_csv(sparc_csv, index=False)
    pd.DataFrame(tng_rows).to_csv(tng_csv, index=False)
    return sparc_csv, tng_csv


def test_tng_sparc_feature_repro_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "analysis" / "tng" / "tng_sparc_feature_repro.py"
    assert script.exists(), f"Missing script: {script}"

    sparc_csv, tng_csv = _make_synthetic_inputs(tmp_path)
    outdir = tmp_path / "run_out"

    cmd = [
        sys.executable,
        str(script),
        "--tng-input",
        str(tng_csv),
        "--sparc-input",
        str(sparc_csv),
        "--mode",
        "smoke",
        "--seed",
        "42",
        "--n-bootstrap",
        "40",
        "--n-shuffles",
        "20",
        "--n-bins",
        "10",
        "--bin-offsets",
        "3",
        "--K",
        "8",
        "--max-pairs",
        "8",
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)

    summary_path = outdir / "summary.json"
    run_stamp = outdir / "run_stamp.json"
    matched_samples = outdir / "matched_samples.parquet"
    report = outdir / "report.md"

    assert summary_path.exists()
    assert run_stamp.exists()
    assert matched_samples.exists()
    assert report.exists()
    assert (outdir / "figures" / "inversion_curve.png").exists()
    assert (outdir / "figures" / "kurtosis_curve.png").exists()
    assert (outdir / "figures" / "matching_diagnostics.png").exists()

    summary = json.loads(summary_path.read_text())
    for key in ("test_name", "matching", "sparc", "tng", "comparison", "split_half_replication", "shuffle_null"):
        assert key in summary
    assert "features" in summary["sparc"]
    assert "features" in summary["tng"]
    assert "bootstrap" in summary["sparc"]
    assert "bootstrap" in summary["tng"]

