#!/usr/bin/env python3
"""Common utilities for clean-room RAR referee reruns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Physical constants
LOG_G_DAGGER = -9.921
G_DAGGER = 1.2e-10  # m/s^2
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
MSUN = 1.989e30  # kg
KPC = 3.086e19  # m


def rar_bec(log_gbar: np.ndarray, log_gd: float = LOG_G_DAGGER) -> np.ndarray:
    """RAR prediction in log10 space for baryonic acceleration input log_gbar."""
    gbar = 10.0 ** np.asarray(log_gbar, dtype=float)
    gd = 10.0 ** float(log_gd)
    x = np.sqrt(np.maximum(gbar / gd, 1e-300))
    denom = np.maximum(1.0 - np.exp(-x), 1e-300)
    return np.log10(gbar / denom)


def healing_length_kpc(M_total_Msun: np.ndarray) -> np.ndarray:
    """Healing length xi (kpc) from total mass in solar masses."""
    m = np.asarray(M_total_Msun, dtype=float)
    return np.sqrt(G_SI * m * MSUN / G_DAGGER) / KPC


def phase_bin_edges_centers(
    width: float = 0.25,
    start_edge: float = -13.5,
    stop_edge: float = -8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact bin edges/centers for phase tests."""
    edges = np.arange(start_edge, stop_edge + 1e-12, float(width))
    centers = edges[:-1] + 0.5 * float(width)
    return edges, centers


def sanitize_json(obj: Any) -> Any:
    """Convert numpy/non-finite values to JSON-safe forms (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable sanitization."""
    path.write_text(json.dumps(sanitize_json(payload), indent=2))
