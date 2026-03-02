#!/usr/bin/env python3
"""Regression tests guarding against silent SPARC dataset drift."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/russelllicht/bec-dark-matter")
PIPELINE_PATH = REPO_ROOT / "analysis" / "pipeline" / "09_unified_rar_pipeline.py"
RAR_CSV_PATH = REPO_ROOT / "analysis" / "results" / "rar_points_unified.csv"


def _load_unified_pipeline_module():
    spec = importlib.util.spec_from_file_location("unified_rar_pipeline_09", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None, f"Could not load pipeline module from {PIPELINE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sparc_loader_regression():
    mod = _load_unified_pipeline_module()

    path_info = mod.resolve_sparc_paths()
    table2 = path_info["table2"]["chosen"]
    mrt = path_info["mrt"]["chosen"]
    assert table2 is not None, (
        "SPARC table2 input missing. Expected one of: "
        f"{list(path_info['table2']['exists'].keys())}"
    )
    assert mrt is not None, (
        "SPARC MRT input missing. Expected one of: "
        f"{list(path_info['mrt']['exists'].keys())}"
    )

    out = mod.load_sparc_data()
    assert len(out) >= 2, "load_sparc_data returned unexpected tuple shape."
    sparc_points = out[0]

    assert len(sparc_points) >= 2500, (
        f"SPARC regression: too few points ({len(sparc_points)} < 2500). "
        "Potential path/parser/filter drift."
    )
    galaxy_keys = {
        p.get("galaxy_key", mod.canonicalize_galaxy_name(p.get("galaxy", "")))
        for p in sparc_points
    }
    assert len(galaxy_keys) >= 120, (
        f"SPARC regression: too few unique galaxies ({len(galaxy_keys)} < 120). "
        "Potential path/parser/filter drift."
    )


def test_unified_csv_contains_sparc_at_expected_scale():
    assert RAR_CSV_PATH.exists(), f"Unified CSV not found: {RAR_CSV_PATH}"
    df = pd.read_csv(RAR_CSV_PATH)

    assert "galaxy_key" in df.columns, "Unified CSV is missing required `galaxy_key` column."
    nonempty_keys = df["galaxy_key"].astype(str).str.strip().ne("").sum()
    assert nonempty_keys > 0, "`galaxy_key` column is present but empty."

    sparc_mask = df["source"].astype(str).eq("SPARC")
    sparc_points = int(sparc_mask.sum())
    assert sparc_points >= 2500, (
        f"Unified CSV regression: SPARC points too low ({sparc_points} < 2500)."
    )

    sparc_galaxies = int(df.loc[sparc_mask, "galaxy_key"].nunique())
    assert sparc_galaxies >= 120, (
        f"Unified CSV regression: SPARC galaxy count too low ({sparc_galaxies} < 120)."
    )

