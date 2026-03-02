#!/usr/bin/env python3
"""Data adapters for strict TNG↔SPARC feature-reproduction comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analysis.utils.galaxy_naming import canonicalize_galaxy_name


_ID_CANDIDATES = (
    "SubhaloID",
    "subhaloid",
    "subhalo_id",
    "galaxy_id",
    "galaxy",
    "id",
)
_LOG_GBAR_CANDIDATES = ("log_gbar", "log_g_bar")
_GBAR_CANDIDATES = ("g_bar", "gbar")
_LOG_GOBS_CANDIDATES = ("log_gobs", "log_g_obs")
_GOBS_CANDIDATES = ("g_obs", "gobs")
_R_CANDIDATES = ("R_kpc", "r_kpc", "radius_kpc", "radius", "R")
_SIGMA_CANDIDATES = ("sigma_log_gobs", "sigma_log_g_obs", "log_gobs_err")
_WEIGHT_CANDIDATES = ("weight", "weights", "w")


def _first_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet") or suffix.endswith(".pq"):
        return pd.read_parquet(path)
    if suffix.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def _coerce_log_quantity(df: pd.DataFrame, log_names: Iterable[str], linear_names: Iterable[str], label: str) -> np.ndarray:
    log_col = _first_column(df, log_names)
    if log_col is not None:
        out = pd.to_numeric(df[log_col], errors="coerce").to_numpy(dtype=float)
        return out

    lin_col = _first_column(df, linear_names)
    if lin_col is None:
        raise ValueError(f"Could not find {label} in columns: {list(df.columns)}")

    lin = pd.to_numeric(df[lin_col], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(np.where(lin > 0.0, lin, np.nan))
    return out


def _canonical_key(values: pd.Series) -> pd.Series:
    return values.astype(str).map(canonicalize_galaxy_name)


def load_sparc_points(
    repo_root: str,
    source_filter: Optional[Iterable[str] | str] = None,
    exclude_sources_regex: Optional[str] = None,
    sparc_input_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load SPARC points from unified CSV and map to the canonical schema."""
    root = Path(repo_root).resolve()
    path = Path(sparc_input_path).resolve() if sparc_input_path else root / "analysis" / "results" / "rar_points_unified.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = _read_table(path)
    if "source" not in df.columns:
        raise ValueError(f"Expected 'source' column in {path}")

    if source_filter is None:
        source_filter = "SPARC"

    if isinstance(source_filter, str):
        df = df.loc[df["source"].astype(str) == source_filter].copy()
    else:
        wanted = {str(x) for x in source_filter}
        df = df.loc[df["source"].astype(str).isin(wanted)].copy()

    if exclude_sources_regex:
        df = df.loc[~df["source"].astype(str).str.contains(exclude_sources_regex, regex=True, na=False)].copy()

    if df.empty:
        raise ValueError("SPARC selection produced zero rows.")

    galaxy_id_col = _first_column(df, ("galaxy", "galaxy_id", "galaxy_key"))
    if galaxy_id_col is None:
        raise ValueError(f"Could not find galaxy identifier in {path}")

    galaxy_key_col = _first_column(df, ("galaxy_key",))
    if galaxy_key_col is None:
        galaxy_key = _canonical_key(df[galaxy_id_col])
    else:
        galaxy_key = _canonical_key(df[galaxy_key_col])

    log_gbar = _coerce_log_quantity(df, _LOG_GBAR_CANDIDATES, _GBAR_CANDIDATES, "g_bar")
    log_gobs = _coerce_log_quantity(df, _LOG_GOBS_CANDIDATES, _GOBS_CANDIDATES, "g_obs")

    r_col = _first_column(df, _R_CANDIDATES)
    if r_col is None:
        r_kpc = np.full(len(df), np.nan, dtype=float)
    else:
        r_kpc = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)

    sigma_col = _first_column(df, _SIGMA_CANDIDATES)
    if sigma_col is None:
        sigma = np.full(len(df), np.nan, dtype=float)
    else:
        sigma = pd.to_numeric(df[sigma_col], errors="coerce").to_numpy(dtype=float)

    weight_col = _first_column(df, _WEIGHT_CANDIDATES)
    if weight_col is None:
        weight = np.ones(len(df), dtype=float)
    else:
        weight = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "dataset": "SPARC",
            "galaxy_id": df[galaxy_id_col].astype(str).to_numpy(),
            "galaxy_key": galaxy_key.to_numpy(),
            "log_gbar": log_gbar,
            "log_gobs": log_gobs,
            "R_kpc": r_kpc,
            "sigma_log_gobs": sigma,
            "weight": weight,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["galaxy_key", "log_gbar", "log_gobs"])
    return out.reset_index(drop=True)


def load_tng_points(tng_input_path: str) -> pd.DataFrame:
    """Load pre-exported TNG point data and map to the canonical schema."""
    path = Path(tng_input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = _read_table(path)
    id_col = _first_column(df, _ID_CANDIDATES)
    if id_col is None:
        raise ValueError(f"Could not identify TNG galaxy id column in {path}")

    log_gbar = _coerce_log_quantity(df, _LOG_GBAR_CANDIDATES, _GBAR_CANDIDATES, "g_bar")
    log_gobs = _coerce_log_quantity(df, _LOG_GOBS_CANDIDATES, _GOBS_CANDIDATES, "g_obs")

    r_col = _first_column(df, _R_CANDIDATES)
    if r_col is None:
        r_kpc = np.full(len(df), np.nan, dtype=float)
    else:
        r_kpc = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)

    sigma_col = _first_column(df, _SIGMA_CANDIDATES)
    if sigma_col is None:
        sigma = np.full(len(df), np.nan, dtype=float)
    else:
        sigma = pd.to_numeric(df[sigma_col], errors="coerce").to_numpy(dtype=float)

    weight_col = _first_column(df, _WEIGHT_CANDIDATES)
    if weight_col is None:
        weight = np.ones(len(df), dtype=float)
    else:
        weight = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    galaxy_id = df[id_col].astype(str)
    out = pd.DataFrame(
        {
            "dataset": "TNG",
            "galaxy_id": galaxy_id.to_numpy(),
            "galaxy_key": _canonical_key(galaxy_id).to_numpy(),
            "log_gbar": log_gbar,
            "log_gobs": log_gobs,
            "R_kpc": r_kpc,
            "sigma_log_gobs": sigma,
            "weight": weight,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["galaxy_key", "log_gbar", "log_gobs"])
    return out.reset_index(drop=True)

