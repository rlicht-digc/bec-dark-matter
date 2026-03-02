#!/usr/bin/env python3
"""
Finite-temperature phase transition scan for Paper 3.

This module compares three candidate potentials plus a no-barrier control,
using a minimal thermal-mass corrected effective potential:

    V_eff(phi, T) = V0(phi) + 0.5 * cT2 * T^2 * phi^2

It computes:
- finite-T minima structure and Tc
- first-order classification from finite-T barrier at Tc
- Tn from bounce action criterion S3/T ~ target (overshoot/undershoot shooting)
- latent heat proxy, alpha_PT, beta/H proxy, wall thickness proxy

No baryogenesis claim is made here. This is a phase-transition viability module.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import integrate, optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------
# Constants and model identifiers
# -------------------------------
MODEL_POLY6 = "poly6_two_minima"
MODEL_QUARTIC = "quartic_symmetry_breaking"
MODEL_CUBIC = "cubic_quartic_asymmetric"
MODEL_CONTROL = "control_no_barrier"

ALL_MODELS = (MODEL_POLY6, MODEL_QUARTIC, MODEL_CUBIC, MODEL_CONTROL)
EPS = 1e-12
HBAR_C_EV_M = 1.973269804e-7  # eV * m
DEFAULT_ASYMMETRY_CSV = "outputs/paper3_cosmology_asymmetry/20260228_223839/asymmetry_scan.csv"
DEFAULT_CURATED_JSON = "analysis/paper3_cosmology/curated_params.json"


@dataclass(frozen=True)
class CaseParams:
    model_id: str
    m2: float
    lambda4: float
    cT2: float
    lambda6_or_v_or_a3: float
    v_scale: float
    thermal_cubic_E: float = 0.0


@dataclass
class MinimaInfo:
    T: float
    minima_phi: np.ndarray
    minima_V: np.ndarray
    false_phi: Optional[float]
    false_V: Optional[float]
    true_phi: Optional[float]
    true_V: Optional[float]
    deltaV_false_minus_true: Optional[float]
    two_phase: bool
    barrier_exists: bool
    barrier_phi: Optional[float]
    barrier_V: Optional[float]


def git_head_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path_text: str, repo_root: Path) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else (repo_root / p)


def parse_csv_float_list(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Parsed empty float list")
    return vals


def format_float_list_short(vals: List[float], max_items: int = 5) -> str:
    if not vals:
        return ""
    clipped = vals[: max_items]
    out = ",".join(f"{v:.6g}" for v in clipped)
    if len(vals) > max_items:
        out += ",..."
    return out


def build_bounce_temperature_lists(
    Tc: float,
    T_min: float,
    T_max: float,
    strategy: str,
    bounce_fracs: List[float],
    bounce_T_min_frac: float,
    bounce_max_T_points: int,
) -> Tuple[List[float], List[float]]:
    if not np.isfinite(Tc) or Tc <= 0:
        return [], []
    max_points = max(1, int(bounce_max_T_points))
    lo_bound = max(float(T_min), float(bounce_T_min_frac) * float(Tc))
    hi_bound = min(float(T_max), float(Tc) * 0.995)
    if hi_bound <= lo_bound:
        return [], []

    grid = np.linspace(hi_bound, lo_bound, max_points).tolist()
    frac_ts = [float(Tc) * f for f in bounce_fracs]
    if strategy == "grid":
        raw = grid
    elif strategy == "fracs":
        raw = frac_ts
    else:
        raw = grid + frac_ts

    def in_bounds(t: float) -> bool:
        return np.isfinite(t) and (t > T_min) and (t < Tc) and (t >= bounce_T_min_frac * Tc) and (t <= T_max)

    raw_keep = sorted(set(float(t) for t in raw if in_bounds(float(t))), reverse=True)
    frac_keep = sorted(set(float(t) for t in frac_ts if in_bounds(float(t))), reverse=True)
    return raw_keep, frac_keep


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    low = series.astype(str).str.strip().str.lower()
    return low.isin({"1", "true", "t", "yes", "y"})


def lambda4_from_map(m_eV: float, a_s_m: float, map_name: str) -> float:
    a_s_eV_inv = a_s_m / HBAR_C_EV_M
    if map_name == "real_scalar":
        return float(32.0 * np.pi * m_eV * a_s_eV_inv)
    if map_name == "complex_scalar":
        return float(8.0 * np.pi * m_eV * a_s_eV_inv)
    raise ValueError(f"Unsupported lambda4_map: {map_name}")


def pick_ratio_column(df: pd.DataFrame) -> str:
    preferred = "log10(V0_pred/V0_obs)"
    if preferred in df.columns:
        return preferred
    for c in df.columns:
        cl = c.lower()
        if "v0_pred" in cl and "v0_obs" in cl and "log10" in cl:
            return c
    raise ValueError("Could not find log10(V0_pred/V0_obs)-like column in asymmetry CSV")


# -------------------------------
# Potential definitions
# -------------------------------
def V0(case: CaseParams, phi: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(phi)
    if case.model_id == MODEL_POLY6:
        lam6 = case.lambda6_or_v_or_a3
        return 0.5 * case.m2 * p**2 - 0.25 * case.lambda4 * p**4 + (lam6 / 6.0) * p**6
    if case.model_id == MODEL_QUARTIC:
        v = case.lambda6_or_v_or_a3
        lam = case.lambda4
        return 0.25 * lam * (p**2 - v**2) ** 2
    if case.model_id == MODEL_CUBIC:
        a3 = case.lambda6_or_v_or_a3
        return 0.5 * case.m2 * p**2 - (a3 / 3.0) * p**3 + 0.25 * case.lambda4 * p**4
    if case.model_id == MODEL_CONTROL:
        return 0.5 * case.m2 * p**2 + 0.25 * case.lambda4 * p**4
    raise ValueError(f"Unknown model_id={case.model_id}")


def dV0_dphi(case: CaseParams, phi: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(phi)
    if case.model_id == MODEL_POLY6:
        lam6 = case.lambda6_or_v_or_a3
        return case.m2 * p - case.lambda4 * p**3 + lam6 * p**5
    if case.model_id == MODEL_QUARTIC:
        v = case.lambda6_or_v_or_a3
        lam = case.lambda4
        return lam * p * (p**2 - v**2)
    if case.model_id == MODEL_CUBIC:
        a3 = case.lambda6_or_v_or_a3
        return case.m2 * p - a3 * p**2 + case.lambda4 * p**3
    if case.model_id == MODEL_CONTROL:
        return case.m2 * p + case.lambda4 * p**3
    raise ValueError(f"Unknown model_id={case.model_id}")


def d2V0_dphi2(case: CaseParams, phi: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(phi)
    if case.model_id == MODEL_POLY6:
        lam6 = case.lambda6_or_v_or_a3
        return case.m2 - 3.0 * case.lambda4 * p**2 + 5.0 * lam6 * p**4
    if case.model_id == MODEL_QUARTIC:
        v = case.lambda6_or_v_or_a3
        lam = case.lambda4
        return lam * (3.0 * p**2 - v**2)
    if case.model_id == MODEL_CUBIC:
        a3 = case.lambda6_or_v_or_a3
        return case.m2 - 2.0 * a3 * p + 3.0 * case.lambda4 * p**2
    if case.model_id == MODEL_CONTROL:
        return case.m2 + 3.0 * case.lambda4 * p**2
    raise ValueError(f"Unknown model_id={case.model_id}")


def Veff(case: CaseParams, phi: np.ndarray | float, T: float) -> np.ndarray | float:
    p = np.asarray(phi)
    return V0(case, p) + 0.5 * case.cT2 * (T**2) * p**2 - case.thermal_cubic_E * T * p**3


def dVeff_dphi(case: CaseParams, phi: np.ndarray | float, T: float) -> np.ndarray | float:
    p = np.asarray(phi)
    return dV0_dphi(case, p) + case.cT2 * (T**2) * p - 3.0 * case.thermal_cubic_E * T * p**2


def d2Veff_dphi2(case: CaseParams, phi: np.ndarray | float, T: float) -> np.ndarray | float:
    p = np.asarray(phi)
    return d2V0_dphi2(case, p) + case.cT2 * (T**2) - 6.0 * case.thermal_cubic_E * T * p


# -------------------------------
# Minima/barrier identification
# -------------------------------
def _dedupe_sorted(vals: List[Tuple[float, float]], tol: float = 1e-4) -> List[Tuple[float, float]]:
    vals = sorted(vals, key=lambda x: x[0])
    out: List[Tuple[float, float]] = []
    for phi, V in vals:
        if not out or abs(phi - out[-1][0]) > tol:
            out.append((phi, V))
        elif V < out[-1][1]:
            out[-1] = (phi, V)
    return out


def find_local_minima(case: CaseParams, T: float, phi_max: float, n_grid: int = 700) -> Tuple[np.ndarray, np.ndarray]:
    phi_grid = np.linspace(0.0, max(phi_max, 1.0), int(n_grid))
    V_grid = Veff(case, phi_grid, T)

    idxs: List[int] = []
    n = len(phi_grid)
    for i in range(n):
        left = V_grid[i - 1] if i > 0 else np.inf
        right = V_grid[i + 1] if i < n - 1 else np.inf
        if V_grid[i] <= left and V_grid[i] <= right:
            idxs.append(i)

    minima: List[Tuple[float, float]] = []
    for i in idxs:
        lo = phi_grid[max(i - 1, 0)]
        hi = phi_grid[min(i + 1, n - 1)]
        if hi <= lo + EPS:
            phi_star = float(phi_grid[i])
            minima.append((phi_star, float(Veff(case, phi_star, T))))
            continue
        try:
            res = optimize.minimize_scalar(
                lambda x: float(Veff(case, x, T)),
                bounds=(float(lo), float(hi)),
                method="bounded",
                options={"xatol": 1e-6},
            )
            phi_star = float(res.x if res.success else phi_grid[i])
        except Exception:
            phi_star = float(phi_grid[i])
        minima.append((phi_star, float(Veff(case, phi_star, T))))

    minima = _dedupe_sorted(minima)
    if not minima:
        # Fallback: global minimum from grid
        i = int(np.argmin(V_grid))
        minima = [(float(phi_grid[i]), float(V_grid[i]))]

    return np.array([x[0] for x in minima], dtype=float), np.array([x[1] for x in minima], dtype=float)


def find_barrier_between(
    case: CaseParams,
    T: float,
    phi_a: float,
    phi_b: float,
    Va: float,
    Vb: float,
    n_grid: int = 800,
) -> Tuple[bool, Optional[float], Optional[float]]:
    lo, hi = (phi_a, phi_b) if phi_a < phi_b else (phi_b, phi_a)
    if hi - lo < 1e-6:
        return False, None, None
    phi = np.linspace(lo, hi, int(n_grid))
    V = Veff(case, phi, T)
    i_max = int(np.argmax(V))
    if i_max <= 0 or i_max >= len(phi) - 1:
        return False, None, None
    Vmax = float(V[i_max])
    if Vmax > max(Va, Vb) + 1e-8:
        return True, float(phi[i_max]), Vmax
    return False, None, None


def minima_info(case: CaseParams, T: float, phi_max: float) -> MinimaInfo:
    mins_phi, mins_V = find_local_minima(case, T, phi_max=phi_max)

    false_idx = int(np.argmin(np.abs(mins_phi)))
    false_phi = float(mins_phi[false_idx])
    false_V = float(mins_V[false_idx])

    mask_true = mins_phi > (false_phi + 1e-4)
    true_phi = None
    true_V = None
    deltaV = None
    two_phase = False
    barrier_exists = False
    barrier_phi = None
    barrier_V = None

    if np.any(mask_true):
        cand_phi = mins_phi[mask_true]
        cand_V = mins_V[mask_true]
        j = int(np.argmin(cand_V))
        true_phi = float(cand_phi[j])
        true_V = float(cand_V[j])
        two_phase = bool(abs(true_phi - false_phi) > 1e-4)

    if two_phase and true_phi is not None and true_V is not None:
        deltaV = float(false_V - true_V)
        barrier_exists, barrier_phi, barrier_V = find_barrier_between(
            case=case,
            T=T,
            phi_a=false_phi,
            phi_b=true_phi,
            Va=false_V,
            Vb=true_V,
        )

    return MinimaInfo(
        T=float(T),
        minima_phi=mins_phi,
        minima_V=mins_V,
        false_phi=false_phi,
        false_V=false_V,
        true_phi=true_phi,
        true_V=true_V,
        deltaV_false_minus_true=deltaV,
        two_phase=two_phase,
        barrier_exists=barrier_exists,
        barrier_phi=barrier_phi,
        barrier_V=barrier_V,
    )


# -------------------------------
# Tc and first-order classification
# -------------------------------
def find_Tc(case: CaseParams, T_min: float, T_max: float, phi_max: float, nT: int = 120) -> Tuple[Optional[float], Optional[MinimaInfo], str]:
    Ts = np.linspace(float(T_max), float(T_min), int(nT))
    infos: List[MinimaInfo] = [minima_info(case, float(T), phi_max) for T in Ts]

    valid = []
    for info in infos:
        if info.two_phase and info.deltaV_false_minus_true is not None and np.isfinite(info.deltaV_false_minus_true):
            valid.append((info.T, float(info.deltaV_false_minus_true), info))

    if len(valid) < 2:
        return None, None, "no_two_phase_window"

    # Look for sign change in DeltaV(T)=V_false - V_true
    bracket = None
    for (T1, d1, _), (T2, d2, _) in zip(valid[:-1], valid[1:]):
        if d1 == 0.0:
            bracket = (T1, T1)
            break
        if d1 * d2 < 0:
            bracket = (T1, T2)
            break

    if bracket is None:
        return None, None, "no_degeneracy_sign_change"

    if bracket[0] == bracket[1]:
        Tc = float(bracket[0])
        infoTc = minima_info(case, Tc, phi_max)
        return Tc, infoTc, "ok"

    T_hi, T_lo = bracket

    def f(T: float) -> float:
        info = minima_info(case, float(T), phi_max)
        if not info.two_phase or info.deltaV_false_minus_true is None:
            # Return large sentinel preserving direction from nearest endpoint.
            return np.nan
        return float(info.deltaV_false_minus_true)

    d_hi = f(T_hi)
    d_lo = f(T_lo)
    if not np.isfinite(d_hi) or not np.isfinite(d_lo) or d_hi * d_lo > 0:
        # fallback linear interpolation on bracket endpoints from sampled values
        T1, d1 = T_hi, d_hi
        T2, d2 = T_lo, d_lo
        if np.isfinite(d1) and np.isfinite(d2) and abs(d2 - d1) > EPS:
            Tc = float(T1 + (0.0 - d1) * (T2 - T1) / (d2 - d1))
            infoTc = minima_info(case, Tc, phi_max)
            return Tc, infoTc, "ok_interp"
        return None, None, "Tc_root_failed"

    try:
        Tc = float(optimize.brentq(lambda t: f(t), T_lo, T_hi, xtol=1e-4, rtol=1e-4, maxiter=80))
        infoTc = minima_info(case, Tc, phi_max)
        return Tc, infoTc, "ok"
    except Exception:
        # fallback linear interpolation
        if abs(d_lo - d_hi) > EPS:
            Tc = float(T_hi + (0.0 - d_hi) * (T_lo - T_hi) / (d_lo - d_hi))
            infoTc = minima_info(case, Tc, phi_max)
            return Tc, infoTc, "ok_interp"
        return None, None, "Tc_root_failed"


# -------------------------------
# Bounce action S3/T
# -------------------------------
def _integrate_profile_scaled(
    case: CaseParams,
    T: float,
    phi0: float,
    phi_false: float,
    v_scale: float,
    m_eff: float,
    rho_max: float,
) -> Tuple[Optional[float], Optional[Dict[str, np.ndarray]], str]:
    rho0 = 1e-6
    denom = max(v_scale * m_eff * m_eff, 1e-18)
    dV0 = float(dVeff_dphi(case, phi0, T))
    varphi0 = (phi0 - phi_false) / max(v_scale, 1e-18)
    varphi_r0 = varphi0 + (dV0 / (6.0 * denom)) * (rho0**2)
    u_r0 = (dV0 / (3.0 * denom)) * rho0

    def rhs(rho: float, y: np.ndarray) -> np.ndarray:
        varphi, u = float(y[0]), float(y[1])
        phi = phi_false + v_scale * varphi
        src = float(dVeff_dphi(case, phi, T)) / denom
        return np.array([u, src - (2.0 / max(rho, 1e-9)) * u], dtype=float)

    try:
        sol = integrate.solve_ivp(
            rhs,
            (rho0, float(rho_max)),
            np.array([varphi_r0, u_r0], dtype=float),
            method="RK45",
            max_step=max(1e-3, min(float(rho_max) / 700.0, 0.5)),
            rtol=5e-6,
            atol=1e-8,
            dense_output=False,
        )
    except Exception:
        return None, None, "integration_fail"

    if sol.y.size == 0:
        return None, None, "integration_fail"

    rho = np.asarray(sol.t, dtype=float)
    varphi = np.asarray(sol.y[0], dtype=float)
    u = np.asarray(sol.y[1], dtype=float)
    if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(varphi)) and np.all(np.isfinite(u))):
        return None, None, "nonfinite_profile"

    phi = phi_false + v_scale * varphi
    dphi = v_scale * m_eff * u
    r = rho / max(m_eff, 1e-18)
    F = float(phi[-1] - phi_false)
    if not np.isfinite(F):
        return None, {"r": r, "phi": phi, "dphi": dphi}, "nonfinite_endpoint"
    return F, {"r": r, "phi": phi, "dphi": dphi}, "ok"


def bounce_action_S3(
    case: CaseParams,
    T: float,
    info: MinimaInfo,
) -> Tuple[Optional[float], Optional[Dict[str, np.ndarray]], str, Dict[str, float]]:
    diag: Dict[str, float] = {
        "bracket_found": 0.0,
        "F_lo": np.nan,
        "F_hi": np.nan,
        "phi0_root": np.nan,
        "n_bisect_iters": 0.0,
        "r_max_used": np.nan,
        "m_eff_used": np.nan,
    }

    if not info.two_phase or info.true_phi is None or info.false_phi is None:
        return None, None, "no_two_phase", diag

    phi_false = float(info.false_phi)
    phi_true = float(info.true_phi)
    if not np.isfinite(phi_false) or not np.isfinite(phi_true):
        return None, None, "bad_minima", diag
    if abs(phi_true - phi_false) <= 1e-8:
        return None, None, "degenerate_minima", diag

    v_scale = max(abs(phi_true - phi_false), 1e-30)
    m_eff2 = abs(float(d2Veff_dphi2(case, phi_false, T)))
    m_eff = max(math.sqrt(max(m_eff2, 0.0)), 1e-6)
    rho_max = 80.0
    r_max = rho_max / m_eff
    diag["m_eff_used"] = float(m_eff)
    diag["r_max_used"] = float(r_max)

    eps_phi = max(1e-8, 1e-6 * max(v_scale, 1.0))
    lo = min(phi_false, phi_true) + eps_phi
    hi = max(phi_false, phi_true) - eps_phi
    if hi <= lo:
        return None, None, "bad_phi_bracket", diag

    F_lo, _, tag_lo = _integrate_profile_scaled(case, T, lo, phi_false, v_scale, m_eff, rho_max)
    F_hi, _, tag_hi = _integrate_profile_scaled(case, T, hi, phi_false, v_scale, m_eff, rho_max)
    diag["F_lo"] = float(F_lo) if F_lo is not None and np.isfinite(F_lo) else np.nan
    diag["F_hi"] = float(F_hi) if F_hi is not None and np.isfinite(F_hi) else np.nan

    bracket_lo = lo
    bracket_hi = hi
    bracket_F_lo = F_lo
    bracket_F_hi = F_hi
    bracket_found = False

    if (
        F_lo is not None
        and F_hi is not None
        and np.isfinite(F_lo)
        and np.isfinite(F_hi)
        and (F_lo == 0.0 or F_hi == 0.0 or F_lo * F_hi < 0.0)
    ):
        bracket_found = True
    else:
        scan_phi = np.linspace(lo, hi, 25)
        scan_F: List[Optional[float]] = []
        for p0 in scan_phi:
            Fp, _, _ = _integrate_profile_scaled(case, T, float(p0), phi_false, v_scale, m_eff, rho_max)
            scan_F.append(Fp if Fp is not None and np.isfinite(Fp) else None)
        for i in range(len(scan_phi) - 1):
            a = scan_F[i]
            b = scan_F[i + 1]
            if a is None or b is None:
                continue
            if a == 0.0 or b == 0.0 or a * b < 0.0:
                bracket_lo = float(scan_phi[i])
                bracket_hi = float(scan_phi[i + 1])
                bracket_F_lo = float(a)
                bracket_F_hi = float(b)
                bracket_found = True
                break

    if not bracket_found:
        return None, None, "no_endpoint_bracket", diag

    diag["bracket_found"] = 1.0
    lo = float(bracket_lo)
    hi = float(bracket_hi)
    flo = float(bracket_F_lo)
    fhi = float(bracket_F_hi)

    if flo == 0.0:
        phi_root = lo
        n_iter = 0
    elif fhi == 0.0:
        phi_root = hi
        n_iter = 0
    else:
        phi_root = 0.5 * (lo + hi)
        n_iter = 0
        for k in range(64):
            n_iter = k + 1
            mid = 0.5 * (lo + hi)
            fmid, _, tag_mid = _integrate_profile_scaled(case, T, mid, phi_false, v_scale, m_eff, rho_max)
            if fmid is None or not np.isfinite(fmid):
                return None, None, "endpoint_eval_fail", diag
            phi_root = float(mid)
            if abs(float(fmid)) < 1e-6:
                break
            if flo * float(fmid) <= 0.0:
                hi = float(mid)
                fhi = float(fmid)
            else:
                lo = float(mid)
                flo = float(fmid)

    diag["phi0_root"] = float(phi_root)
    diag["n_bisect_iters"] = float(n_iter)

    F_root, profile, tag_root = _integrate_profile_scaled(case, T, float(phi_root), phi_false, v_scale, m_eff, rho_max)
    if profile is None or F_root is None or not np.isfinite(F_root):
        return None, None, "root_profile_fail", diag

    r = profile["r"]
    phi = profile["phi"]
    dphi = profile["dphi"]
    V_false = float(Veff(case, phi_false, T))
    integrand = 0.5 * dphi**2 + (Veff(case, phi, T) - V_false)
    if not (np.all(np.isfinite(integrand)) and np.all(np.isfinite(r))):
        return None, None, "nonfinite_action_integrand", diag
    S3 = float(4.0 * np.pi * np.trapezoid(r**2 * integrand, r))
    if not np.isfinite(S3) or S3 <= 0:
        return None, None, "nonfinite_S3", diag

    profile_out = dict(profile)
    profile_out["diag"] = diag
    return S3, profile_out, "ok", diag


def wall_thickness_from_profile(profile: Optional[Dict[str, np.ndarray]], phi_false: float, phi_true: float) -> Optional[float]:
    if profile is None:
        return None
    r = profile["r"]
    phi = profile["phi"]
    if r.size < 4:
        return None
    hi = phi_false + 0.9 * (phi_true - phi_false)
    lo = phi_false + 0.1 * (phi_true - phi_false)

    # Profile is expected to decrease with r.
    try:
        idx_hi = np.where(phi <= hi)[0]
        idx_lo = np.where(phi <= lo)[0]
        if idx_hi.size == 0 or idx_lo.size == 0:
            return None
        r_hi = float(r[idx_hi[0]])
        r_lo = float(r[idx_lo[0]])
        return abs(r_lo - r_hi)
    except Exception:
        return None


def thinwall_s3_over_t(
    case: CaseParams,
    T: float,
    info: MinimaInfo,
    phi_grid: int = 2000,
    eps: float = 1e-18,
) -> Tuple[Optional[float], str]:
    if not info.two_phase or info.false_phi is None or info.true_phi is None:
        return None, "no_two_phase"
    phi_false = float(info.false_phi)
    phi_true = float(info.true_phi)
    if not np.isfinite(phi_false) or not np.isfinite(phi_true) or phi_true <= phi_false + 1e-10:
        return None, "degenerate_minima"
    if not np.isfinite(T) or T <= 0:
        return None, "bad_T"

    V_false = float(Veff(case, phi_false, T))
    V_true = float(Veff(case, phi_true, T))
    deltaV = V_false - V_true
    if not np.isfinite(deltaV) or deltaV <= eps:
        return None, "deltaV<=0"

    nphi = max(80, int(phi_grid))
    phi = np.linspace(phi_false, phi_true, nphi)
    V_path = Veff(case, phi, T)
    under = 2.0 * np.maximum(V_path - V_false, 0.0)
    sigma = float(np.trapezoid(np.sqrt(under), phi))
    if not np.isfinite(sigma) or sigma <= eps:
        return None, "sigma<=0"

    S3 = float((16.0 * np.pi / 3.0) * (sigma**3) / max(deltaV * deltaV, eps))
    if not np.isfinite(S3) or S3 <= 0:
        return None, "nonfinite_S3_thinwall"
    s3ot = float(S3 / max(T, eps))
    if not np.isfinite(s3ot):
        return None, "nonfinite_S3_over_T_thinwall"
    return s3ot, "ok"


# -------------------------------
# Thermodynamic proxies
# -------------------------------
def deltaV_false_minus_true(case: CaseParams, T: float, phi_max: float) -> Optional[float]:
    info = minima_info(case, T, phi_max)
    if not info.two_phase or info.deltaV_false_minus_true is None:
        return None
    return float(info.deltaV_false_minus_true)


def latent_heat_proxy(case: CaseParams, T_ref: float, phi_max: float) -> Optional[float]:
    dT = max(1e-3, 0.02 * T_ref)
    d1 = deltaV_false_minus_true(case, T_ref - dT, phi_max)
    d2 = deltaV_false_minus_true(case, T_ref + dT, phi_max)
    d0 = deltaV_false_minus_true(case, T_ref, phi_max)
    if d0 is None or d1 is None or d2 is None:
        return None
    dDelta_dT = (d2 - d1) / (2.0 * dT)
    L = -T_ref * dDelta_dT + d0
    return float(L)


def rho_rad(T: float, gstar: float) -> float:
    return float((np.pi**2 / 30.0) * gstar * T**4)


# -------------------------------
# Scan generation
# -------------------------------
def deterministic_subsample(cases: Sequence[CaseParams], n_keep: int, seed: int) -> List[CaseParams]:
    if len(cases) <= n_keep:
        return list(cases)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(cases), size=n_keep, replace=False))
    return [cases[int(i)] for i in idx]


def build_case_library() -> List[CaseParams]:
    cT2_grid = np.logspace(-3, 1, 15)
    m2_grid = np.logspace(-4, 4, 13)
    lam4_grid = np.logspace(-6, 0, 13)
    v_scales = [5.0, 10.0, 20.0, 50.0]
    a3_grid = np.logspace(-6, 1, 13)

    poly6_cases: List[CaseParams] = []
    for m2 in m2_grid:
        for lam4 in lam4_grid:
            for v in v_scales:
                lam6 = (lam4 * v * v - m2) / (v**4)
                if lam6 <= 0 or not np.isfinite(lam6):
                    continue
                for cT2 in cT2_grid:
                    poly6_cases.append(
                        CaseParams(
                            model_id=MODEL_POLY6,
                            m2=float(m2),
                            lambda4=float(lam4),
                            cT2=float(cT2),
                            lambda6_or_v_or_a3=float(lam6),
                            v_scale=float(v),
                        )
                    )

    quartic_cases: List[CaseParams] = []
    for v in v_scales:
        for lam in lam4_grid:
            for cT2 in cT2_grid:
                quartic_cases.append(
                    CaseParams(
                        model_id=MODEL_QUARTIC,
                        m2=float(np.nan),
                        lambda4=float(lam),
                        cT2=float(cT2),
                        lambda6_or_v_or_a3=float(v),
                        v_scale=float(v),
                    )
                )

    cubic_cases: List[CaseParams] = []
    for m2 in m2_grid:
        for a3 in a3_grid:
            for lam4 in lam4_grid:
                for cT2 in cT2_grid:
                    cubic_cases.append(
                        CaseParams(
                            model_id=MODEL_CUBIC,
                            m2=float(m2),
                            lambda4=float(lam4),
                            cT2=float(cT2),
                            lambda6_or_v_or_a3=float(a3),
                            v_scale=20.0,
                        )
                    )

    control_cases: List[CaseParams] = []
    for m2 in m2_grid:
        for lam4 in lam4_grid:
            for cT2 in cT2_grid:
                control_cases.append(
                    CaseParams(
                        model_id=MODEL_CONTROL,
                        m2=float(m2),
                        lambda4=float(lam4),
                        cT2=float(cT2),
                        lambda6_or_v_or_a3=float(np.nan),
                        v_scale=20.0,
                    )
                )

    # Full ordered library (no truncation in this builder).
    all_cases = poly6_cases + quartic_cases + cubic_cases + control_cases
    return all_cases


def _portal_cT2_values(args: argparse.Namespace, lambda4: float) -> Tuple[List[float], List[float]]:
    if args.cT2_mode == "free_scan":
        if args.cT2_free is None:
            raise ValueError("--cT2_free is required when --cT2_mode=free_scan")
        return [float(args.cT2_free)], [float("nan")]
    if args.cT2_mode == "self_only":
        return [float(lambda4 / 12.0)], [0.0]
    gvals = parse_csv_float_list(args.g_portal_list)
    cvals = [float(lambda4 / 12.0 + (g * g) / 12.0) for g in gvals]
    return cvals, [float(g) for g in gvals]


def _thermal_cubic_values(args: argparse.Namespace, g_portal: float) -> Tuple[List[float], List[float]]:
    if args.thermal_cubic_mode == "off":
        return [0.0], [float("nan")]
    if args.thermal_cubic_mode == "free_scan":
        if args.E_free is not None:
            return [float(args.E_free)], [float("nan")]
        evals = parse_csv_float_list(args.E_list)
        return [float(x) for x in evals], [float("nan")] * len(evals)
    kvals = parse_csv_float_list(args.kE_list)
    g = float(g_portal) if np.isfinite(g_portal) else 0.0
    evals = [float(k * (g**3) / (12.0 * np.pi)) for k in kvals]
    return evals, [float(k) for k in kvals]


def _create_curated_defaults_from_asymmetry(asym_df: pd.DataFrame, ratio_col: str) -> List[Dict[str, object]]:
    work = asym_df.copy()
    if "valid_flag" in work.columns:
        work = work[as_bool_series(work["valid_flag"])]
    work = work[np.isfinite(work[ratio_col].to_numpy(dtype=float))]
    work = work[
        np.isfinite(work["m_eV"].to_numpy(dtype=float))
        & np.isfinite(work["v_eV"].to_numpy(dtype=float))
        & np.isfinite(work["lambda4"].to_numpy(dtype=float))
        & np.isfinite(work["lambda6"].to_numpy(dtype=float))
    ]
    if work.empty:
        return []
    work["abs_log_ratio"] = np.abs(work[ratio_col].to_numpy(dtype=float))
    best = work.sort_values("abs_log_ratio").iloc[0]
    neighbors = work[work["abs_log_ratio"] <= 2.0].sort_values("abs_log_ratio").head(6)
    if neighbors.empty:
        neighbors = work.sort_values("abs_log_ratio").head(6)

    def row_to_payload(row: pd.Series, note: str) -> Dict[str, object]:
        v_GeV = float(row["v_eV"]) * 1e-9
        return {
            "model": MODEL_POLY6,
            "m_eV": float(row["m_eV"]),
            "lambda4": float(row["lambda4"]),
            "lambda6": float(row["lambda6"]),
            "v_GeV": float(v_GeV),
            "cT2": float(row["lambda4"] / 12.0),
            "notes": note,
            "log10_V0_ratio": float(row[ratio_col]),
        }

    out: List[Dict[str, object]] = []
    # Always include explicit best row.
    out.append(row_to_payload(best, "best-match asymmetry row"))
    for _, row in neighbors.iterrows():
        if len(out) >= 6:
            break
        key = (float(row["m_eV"]), float(row["v_eV"]), float(row["lambda4"]), float(row["lambda6"]))
        used = {
            (float(x["m_eV"]), float(x["v_GeV"]) * 1e9, float(x["lambda4"]), float(x.get("lambda6", np.nan)))
            for x in out
        }
        if key in used:
            continue
        out.append(row_to_payload(row, "neighbor within ~2 dex of Lambda"))

    # Hard include canonical "our numbers" anchor.
    anchor = {
        "model": MODEL_POLY6,
        "m_eV": 1.0e-22,
        "lambda4": 5.094639e-50,
        "lambda6": 5.094639e-70,
        "v_GeV": 10.0,
        "cT2": 5.094639e-50 / 12.0,
        "notes": "canonical best row anchor",
    }
    if not any(abs(float(x["m_eV"]) - anchor["m_eV"]) < 1e-30 and abs(float(x["v_GeV"]) - anchor["v_GeV"]) < 1e-9 for x in out):
        out.insert(0, anchor)
    return out


def build_constrained_cases(
    args: argparse.Namespace, repo_root: Path
) -> Tuple[List[Tuple[CaseParams, Dict[str, object]]], pd.DataFrame, int, int, int, Path]:
    asym_path = resolve_path(args.asymmetry_csv, repo_root)
    if not asym_path.exists():
        raise FileNotFoundError(f"Asymmetry CSV not found: {asym_path}")
    df = pd.read_csv(asym_path)
    ratio_col = pick_ratio_column(df)

    if "valid_flag" in df.columns:
        work = df[as_bool_series(df["valid_flag"])].copy()
    else:
        work = df.copy()

    required = ["m_eV", "v_eV", "lambda6"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Asymmetry CSV missing required columns: {missing}")

    work = work[
        np.isfinite(work["m_eV"].to_numpy(dtype=float))
        & np.isfinite(work["v_eV"].to_numpy(dtype=float))
        & np.isfinite(work["lambda6"].to_numpy(dtype=float))
    ].copy()
    if work.empty:
        raise ValueError("No finite rows available in asymmetry CSV after basic filtering")

    if "lambda4" in work.columns:
        work["lambda4_used"] = work["lambda4"].to_numpy(dtype=float)
        work["lambda4_source"] = "csv"
    else:
        if "a_s_m" not in work.columns:
            raise ValueError("Asymmetry CSV has no lambda4 and no a_s_m to derive lambda4")
        work["lambda4_used"] = [
            lambda4_from_map(float(m), float(a), args.lambda4_map)
            for m, a in zip(work["m_eV"].to_numpy(dtype=float), work["a_s_m"].to_numpy(dtype=float))
        ]
        work["lambda4_source"] = f"derived_{args.lambda4_map}"

    work = work[np.isfinite(work["lambda4_used"].to_numpy(dtype=float))].copy()
    work["abs_log_ratio"] = np.abs(work[ratio_col].to_numpy(dtype=float))
    near = work[work["abs_log_ratio"] <= float(args.select_near_lambda_dex)].copy()
    if near.empty:
        near = work.sort_values("abs_log_ratio").head(max(1, int(args.max_constrained_cases))).copy()

    best_row = work.sort_values("abs_log_ratio").head(1)
    if bool(args.always_include_best) and not best_row.empty:
        best_idx = int(best_row.index[0])
        if best_idx not in set(near.index.tolist()):
            near = pd.concat([best_row, near], axis=0)

    near = near[~near.index.duplicated(keep="first")].copy()
    near = near.sort_values("abs_log_ratio")
    if int(args.max_constrained_cases) > 0 and len(near) > int(args.max_constrained_cases):
        near = near.head(int(args.max_constrained_cases)).copy()

    cased: List[Tuple[CaseParams, Dict[str, object]]] = []
    n_portal = 1
    if args.thermal_cubic_mode == "off":
        n_e = 1
    elif args.thermal_cubic_mode == "free_scan":
        n_e = 1 if args.E_free is not None else len(parse_csv_float_list(args.E_list))
    else:
        n_e = len(parse_csv_float_list(args.kE_list))
    for idx, row in near.iterrows():
        m_eV = float(row["m_eV"])
        m_GeV = m_eV * 1e-9
        m2 = m_GeV * m_GeV
        v_GeV = float(row["v_eV"]) * 1e-9
        lam4 = float(row["lambda4_used"])
        lam6 = float(row["lambda6"])
        cvals, gvals = _portal_cT2_values(args, lam4)
        n_portal = max(n_portal, len(cvals))
        for cT2, g in zip(cvals, gvals):
            evals, kvals = _thermal_cubic_values(args, g_portal=float(g))
            for e_used, k_used in zip(evals, kvals):
                case = CaseParams(
                    model_id=MODEL_POLY6,
                    m2=float(m2),
                    lambda4=float(lam4),
                    cT2=float(cT2),
                    lambda6_or_v_or_a3=float(lam6),
                    v_scale=float(v_GeV),
                    thermal_cubic_E=float(e_used),
                )
                meta = {
                    "candidate_id": int(idx),
                    "source_row": int(idx),
                    "m_eV": m_eV,
                    "m_GeV": m_GeV,
                    "m2_GeV2": m2,
                    "a_s_m": float(row["a_s_m"]) if "a_s_m" in row and np.isfinite(row["a_s_m"]) else np.nan,
                    "v_eV": float(row["v_eV"]),
                    "v_GeV": v_GeV,
                    "lambda4_input": float(row["lambda4"]) if "lambda4" in row and np.isfinite(row["lambda4"]) else np.nan,
                    "lambda4": lam4,
                    "lambda6": lam6,
                    "lambda4_source": str(row.get("lambda4_source", "csv")),
                    "lambda4_map": args.lambda4_map,
                    "log10_V0_ratio": float(row[ratio_col]) if np.isfinite(row[ratio_col]) else np.nan,
                    "valid_flag_input": bool(row["valid_flag"]) if "valid_flag" in row else True,
                    "reason_input": str(row["reason"]) if "reason" in row else "",
                    "cT2_mode": args.cT2_mode,
                    "cT2": float(cT2),
                    "g_portal": float(g),
                    "g_portal_used": float(g),
                    "thermal_cubic_mode": args.thermal_cubic_mode,
                    "E_used": float(e_used),
                    "kE_used": float(k_used) if np.isfinite(k_used) else np.nan,
                }
                cased.append((case, meta))

    used = pd.DataFrame([x[1] for x in cased])
    return cased, used, int(len(near)), int(n_portal), int(n_e), asym_path


def build_curated_cases(
    args: argparse.Namespace, repo_root: Path
) -> Tuple[List[Tuple[CaseParams, Dict[str, object]]], pd.DataFrame, int, int, int, Path]:
    curated_path = resolve_path(args.curated_json, repo_root)
    if not curated_path.exists():
        asym_path = resolve_path(args.asymmetry_csv, repo_root)
        if not asym_path.exists():
            raise FileNotFoundError(f"Curated file missing and asymmetry CSV not found: {asym_path}")
        asym_df = pd.read_csv(asym_path)
        ratio_col = pick_ratio_column(asym_df)
        defaults = _create_curated_defaults_from_asymmetry(asym_df, ratio_col=ratio_col)
        if not defaults:
            defaults = [
                {
                    "model": MODEL_POLY6,
                    "m_eV": 1.0e-22,
                    "lambda4": 5.094639e-50,
                    "lambda6": 5.094639e-70,
                    "v_GeV": 10.0,
                    "cT2": 5.094639e-50 / 12.0,
                    "notes": "fallback canonical anchor",
                }
            ]
        curated_path.parent.mkdir(parents=True, exist_ok=True)
        curated_path.write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")

    items = json.loads(curated_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("curated_json must contain a non-empty list")
    if int(args.max_constrained_cases) > 0:
        items = items[: int(args.max_constrained_cases)]

    cased: List[Tuple[CaseParams, Dict[str, object]]] = []
    n_portal = 1
    if args.thermal_cubic_mode == "off":
        n_e = 1
    elif args.thermal_cubic_mode == "free_scan":
        n_e = 1 if args.E_free is not None else len(parse_csv_float_list(args.E_list))
    else:
        n_e = len(parse_csv_float_list(args.kE_list))
    for i, item in enumerate(items):
        model_id = str(item.get("model", MODEL_POLY6))
        if model_id != MODEL_POLY6:
            continue
        m_eV = float(item["m_eV"])
        m_GeV = m_eV * 1e-9
        m2 = m_GeV * m_GeV
        v_GeV = float(item["v_GeV"])
        v_eV = v_GeV * 1e9
        lam4 = float(item["lambda4"])
        lam6 = float(item["lambda6"]) if "lambda6" in item else float((lam4 * (v_GeV**2) - m2) / max(v_GeV**4, 1e-40))
        cvals, gvals = _portal_cT2_values(args, lam4)
        n_portal = max(n_portal, len(cvals))
        for cT2, g in zip(cvals, gvals):
            evals, kvals = _thermal_cubic_values(args, g_portal=float(g))
            for e_used, k_used in zip(evals, kvals):
                case = CaseParams(
                    model_id=MODEL_POLY6,
                    m2=float(m2),
                    lambda4=float(lam4),
                    cT2=float(cT2),
                    lambda6_or_v_or_a3=float(lam6),
                    v_scale=float(v_GeV),
                    thermal_cubic_E=float(e_used),
                )
                meta = {
                    "candidate_id": int(i),
                    "source_row": int(i),
                    "m_eV": m_eV,
                    "m_GeV": m_GeV,
                    "m2_GeV2": m2,
                    "v_eV": v_eV,
                    "v_GeV": v_GeV,
                    "lambda4": lam4,
                    "lambda6": lam6,
                    "lambda4_source": "curated_json",
                    "lambda4_map": args.lambda4_map,
                    "log10_V0_ratio": float(item["log10_V0_ratio"]) if "log10_V0_ratio" in item else np.nan,
                    "valid_flag_input": True,
                    "reason_input": str(item.get("notes", "")),
                    "cT2_mode": args.cT2_mode,
                    "cT2": float(cT2),
                    "g_portal": float(g),
                    "g_portal_used": float(g),
                    "thermal_cubic_mode": args.thermal_cubic_mode,
                    "E_used": float(e_used),
                    "kE_used": float(k_used) if np.isfinite(k_used) else np.nan,
                    "notes": str(item.get("notes", "")),
                }
                cased.append((case, meta))

    used = pd.DataFrame([x[1] for x in cased])
    return cased, used, int(len(items)), int(n_portal), int(n_e), curated_path


# -------------------------------
# Case evaluation
# -------------------------------
def evaluate_case(
    case: CaseParams,
    T_min: float,
    T_max: float,
    S3_over_T_target: float,
    gstar: float,
    tc_window: Optional[Tuple[float, float]] = None,
    tn_window: Optional[Tuple[float, float]] = None,
    require_tc_in_window: bool = True,
    barrier_min: float = 0.0,
    apply_bounce_gate: bool = False,
    min_bounce_samples: int = 1,
    bounce_T_strategy: str = "hybrid",
    bounce_fracs: Optional[List[float]] = None,
    bounce_use_barrier_filter: bool = True,
    bounce_allow_no_barrier: bool = False,
    bounce_T_min_frac: float = 0.5,
    bounce_max_T_points: int = 16,
    bounce_fallback: str = "thinwall",
    thinwall_phi_grid: int = 2000,
    thinwall_eps: float = 1e-18,
) -> Dict[str, object]:
    # Dynamic phi range.
    phi_max = max(200.0, 5.0 * max(case.v_scale, abs(case.lambda6_or_v_or_a3) if np.isfinite(case.lambda6_or_v_or_a3) else 0.0, 1.0))

    out: Dict[str, object] = {
        "model_id": case.model_id,
        "valid_flag": True,
        "fail_reason": "",
        "m2": case.m2,
        "lambda4": case.lambda4,
        "lambda6_or_v_or_a3": case.lambda6_or_v_or_a3,
        "v_scale": case.v_scale,
        "cT2": case.cT2,
        "E_used": case.thermal_cubic_E,
        "Tc": np.nan,
        "Tc_found": False,
        "Tn": np.nan,
        "first_order_flag": False,
        "phi_false_Tc": np.nan,
        "phi_true_Tc": np.nan,
        "S3_over_T_at_Tn": np.nan,
        "min_S3_over_T": np.nan,
        "T_at_min_S3_over_T": np.nan,
        "crossed_target": False,
        "bounce_attempted": False,
        "bounce_success": False,
        "n_bounce_success": 0,
        "n_bounce_T_raw": 0,
        "n_bounce_T_kept": 0,
        "barrier_present_count": 0,
        "bounce_T_list_used": "",
        "bounce_fail_code_counts": "",
        "bounce_bracket_found": False,
        "bounce_F_lo": np.nan,
        "bounce_F_hi": np.nan,
        "bounce_phi0_root": np.nan,
        "bounce_n_bisect_iters": np.nan,
        "bounce_r_max_used": np.nan,
        "bounce_m_eff_used": np.nan,
        "thinwall_min_S3_over_T": np.nan,
        "thinwall_T_at_min": np.nan,
        "thinwall_crossed_target": False,
        "thinwall_attempted_count": 0,
        "thinwall_fail_reason": "",
        "barrier_height_Tc": np.nan,
        "alpha_PT": np.nan,
        "beta_over_H": np.nan,
        "L_proxy": np.nan,
        "wall_thickness_proxy": np.nan,
        "_phi_max": phi_max,
    }

    # Parameter sanity for poly6 positivity.
    if case.model_id == MODEL_POLY6 and (not np.isfinite(case.lambda6_or_v_or_a3) or case.lambda6_or_v_or_a3 <= 0):
        out["valid_flag"] = False
        out["fail_reason"] = "invalid_lambda6"
        return out

    Tc, infoTc, tc_reason = find_Tc(case, T_min=T_min, T_max=T_max, phi_max=phi_max)
    if Tc is None or infoTc is None:
        out["fail_reason"] = tc_reason
        return out

    out["Tc"] = float(Tc)
    out["Tc_found"] = True
    out["phi_false_Tc"] = float(infoTc.false_phi) if infoTc.false_phi is not None else np.nan
    out["phi_true_Tc"] = float(infoTc.true_phi) if infoTc.true_phi is not None else np.nan

    first_order = bool(infoTc.two_phase and infoTc.barrier_exists)
    if infoTc.barrier_V is not None and infoTc.false_V is not None and infoTc.true_V is not None:
        out["barrier_height_Tc"] = float(infoTc.barrier_V - max(float(infoTc.false_V), float(infoTc.true_V)))
    barrier_ok_first = bool(np.isfinite(out["barrier_height_Tc"]) and float(out["barrier_height_Tc"]) > float(barrier_min))
    first_order = bool(first_order and barrier_ok_first)
    out["first_order_flag"] = first_order
    if not first_order:
        out["fail_reason"] = "no_barrier_at_Tc"

    # Fallback wall-thickness proxy from barrier curvature at Tc if available.
    if infoTc.barrier_phi is not None:
        curv_bar = abs(float(d2Veff_dphi2(case, infoTc.barrier_phi, Tc)))
        if curv_bar > 0:
            out["wall_thickness_proxy"] = float(1.0 / math.sqrt(max(curv_bar, 1e-12)))

    Tn = np.nan
    S3T_Tn = np.nan
    best_profile = None
    infoTn = None
    crossed_target = False

    gate_ok = first_order
    if first_order and apply_bounce_gate:
        lo_tc, hi_tc = tc_window if tc_window is not None else (10.0, 50.0)
        tc_in_window = bool(np.isfinite(Tc) and (Tc >= lo_tc) and (Tc <= hi_tc))
        gate_ok = (not require_tc_in_window or tc_in_window) and barrier_ok_first
        if not gate_ok:
            out["fail_reason"] = "bounce_gate_not_satisfied"

    if first_order and gate_ok:
        if apply_bounce_gate:
            fracs = bounce_fracs if bounce_fracs is not None else [0.98, 0.95, 0.9, 0.85, 0.8, 0.7]
            raw_t, fallback_t = build_bounce_temperature_lists(
                Tc=float(Tc),
                T_min=float(T_min),
                T_max=float(T_max),
                strategy=str(bounce_T_strategy),
                bounce_fracs=fracs,
                bounce_T_min_frac=float(bounce_T_min_frac),
                bounce_max_T_points=int(bounce_max_T_points),
            )
            out["n_bounce_T_raw"] = int(len(raw_t))
            info_cache: Dict[float, MinimaInfo] = {}
            kept_t: List[float] = []
            barrier_count = 0

            for t in raw_t:
                info = minima_info(case, float(t), phi_max)
                info_cache[float(t)] = info
                has_bar = bool(info.two_phase and info.barrier_exists)
                barrier_count += int(has_bar)
                if (not bounce_use_barrier_filter) or has_bar:
                    kept_t.append(float(t))

            # Hybrid fallback if too few points survived.
            if len(kept_t) < max(1, int(min_bounce_samples)):
                fb_kept: List[float] = []
                for t in fallback_t:
                    info = info_cache.get(float(t))
                    if info is None:
                        info = minima_info(case, float(t), phi_max)
                        info_cache[float(t)] = info
                    has_bar = bool(info.two_phase and info.barrier_exists)
                    barrier_count += int(has_bar)
                    if (not bounce_use_barrier_filter) or has_bar:
                        fb_kept.append(float(t))

                if len(fb_kept) >= max(1, int(min_bounce_samples)):
                    kept_t = fb_kept
                else:
                    if bounce_use_barrier_filter and (not bounce_allow_no_barrier):
                        kept_t = []
                        out["fail_reason"] = "no_barrier_below_Tc"
                    else:
                        # Debug path: allow attempts even without barrier filter success.
                        kept_t = fallback_t if fallback_t else raw_t

            kept_t = sorted(set(kept_t), reverse=True)
            out["n_bounce_T_kept"] = int(len(kept_t))
            out["barrier_present_count"] = int(barrier_count)
            out["bounce_T_list_used"] = format_float_list_short(kept_t)

            if len(kept_t) >= max(1, int(min_bounce_samples)):
                out["bounce_attempted"] = True
                out["min_S3_over_T"] = float(np.inf)
                fail_counts: Dict[str, int] = {}
                success_records: List[Tuple[float, float, Optional[Dict[str, np.ndarray]], MinimaInfo]] = []
                thinwall_fail_counts: Dict[str, int] = {}
                thinwall_records: List[Tuple[float, float]] = []
                thinwall_attempted_count = 0

                for t in kept_t:
                    info = info_cache.get(float(t))
                    if info is None:
                        info = minima_info(case, float(t), phi_max)
                        info_cache[float(t)] = info
                    S3, prof, reason, bdiag = bounce_action_S3(case, float(t), info)
                    if bdiag:
                        out["bounce_bracket_found"] = bool(out["bounce_bracket_found"] or bool(bdiag.get("bracket_found", 0.0)))
                        out["bounce_F_lo"] = float(bdiag.get("F_lo", np.nan))
                        out["bounce_F_hi"] = float(bdiag.get("F_hi", np.nan))
                        out["bounce_phi0_root"] = float(bdiag.get("phi0_root", np.nan))
                        out["bounce_n_bisect_iters"] = float(bdiag.get("n_bisect_iters", np.nan))
                        out["bounce_r_max_used"] = float(bdiag.get("r_max_used", np.nan))
                        out["bounce_m_eff_used"] = float(bdiag.get("m_eff_used", np.nan))
                    if S3 is None or not np.isfinite(S3):
                        code = str(reason) if reason else "bounce_fail"
                        fail_counts[code] = fail_counts.get(code, 0) + 1
                        if str(bounce_fallback) == "thinwall":
                            thinwall_attempted_count += 1
                            s3tw, tw_reason = thinwall_s3_over_t(
                                case=case,
                                T=float(t),
                                info=info,
                                phi_grid=int(thinwall_phi_grid),
                                eps=float(thinwall_eps),
                            )
                            if s3tw is not None and np.isfinite(s3tw):
                                thinwall_records.append((float(t), float(s3tw)))
                            else:
                                tw_code = str(tw_reason) if tw_reason else "thinwall_fail"
                                thinwall_fail_counts[tw_code] = thinwall_fail_counts.get(tw_code, 0) + 1
                        continue
                    s3ot = float(S3 / max(float(t), 1e-9))
                    if not np.isfinite(s3ot):
                        fail_counts["nonfinite_S3_over_T"] = fail_counts.get("nonfinite_S3_over_T", 0) + 1
                        if str(bounce_fallback) == "thinwall":
                            thinwall_attempted_count += 1
                            s3tw, tw_reason = thinwall_s3_over_t(
                                case=case,
                                T=float(t),
                                info=info,
                                phi_grid=int(thinwall_phi_grid),
                                eps=float(thinwall_eps),
                            )
                            if s3tw is not None and np.isfinite(s3tw):
                                thinwall_records.append((float(t), float(s3tw)))
                            else:
                                tw_code = str(tw_reason) if tw_reason else "thinwall_fail"
                                thinwall_fail_counts[tw_code] = thinwall_fail_counts.get(tw_code, 0) + 1
                        continue
                    success_records.append((float(t), s3ot, prof, info))
                    if s3ot < float(out["min_S3_over_T"]):
                        out["min_S3_over_T"] = float(s3ot)
                        out["T_at_min_S3_over_T"] = float(t)
                        best_profile = prof
                        infoTn = info

                out["thinwall_attempted_count"] = int(thinwall_attempted_count)
                if thinwall_records:
                    t_tw = np.array([x[0] for x in thinwall_records], dtype=float)
                    s_tw = np.array([x[1] for x in thinwall_records], dtype=float)
                    j_tw = int(np.argmin(s_tw))
                    out["thinwall_min_S3_over_T"] = float(s_tw[j_tw])
                    out["thinwall_T_at_min"] = float(t_tw[j_tw])
                    out["thinwall_crossed_target"] = bool(np.any(s_tw <= float(S3_over_T_target)))
                    out["thinwall_fail_reason"] = ""
                elif thinwall_fail_counts:
                    out["thinwall_crossed_target"] = False
                    out["thinwall_fail_reason"] = ";".join(
                        f"{k}:{v}" for k, v in sorted(thinwall_fail_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    )

                out["n_bounce_success"] = int(len(success_records))
                out["bounce_success"] = bool(len(success_records) > 0)
                if fail_counts:
                    out["bounce_fail_code_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(fail_counts.items(), key=lambda kv: (-kv[1], kv[0])))

                if len(success_records) == 0:
                    out["fail_reason"] = "bounce_solver_failed_all_T"
                else:
                    crossed_target = bool(np.any(np.array([x[1] for x in success_records], dtype=float) <= float(S3_over_T_target)))
                    if len(success_records) < len(kept_t):
                        out["fail_reason"] = "bounce_solver_failed_some_T"
                    else:
                        out["fail_reason"] = ""

                    if crossed_target:
                        lo_tn, hi_tn = tn_window if tn_window is not None else (10.0, 50.0)
                        t_pick = float(out["T_at_min_S3_over_T"]) if np.isfinite(out["T_at_min_S3_over_T"]) else np.nan
                        s_pick = float(out["min_S3_over_T"]) if np.isfinite(out["min_S3_over_T"]) else np.nan
                        if np.isfinite(t_pick) and np.isfinite(s_pick) and (t_pick >= lo_tn) and (t_pick <= hi_tn):
                            Tn = float(t_pick)
                            S3T_Tn = float(s_pick)
                            if infoTn is not None:
                                S3_ref, prof_ref, _, _ = bounce_action_S3(case, float(Tn), infoTn)
                                if S3_ref is not None and np.isfinite(S3_ref):
                                    S3T_Tn = float(S3_ref / max(float(Tn), 1e-9))
                                    best_profile = prof_ref
                        else:
                            out["fail_reason"] = "crossed_target_outside_window"
                    else:
                        if not out["fail_reason"]:
                            out["fail_reason"] = "no_S3_target_hit"
            else:
                if not out["fail_reason"]:
                    out["fail_reason"] = "insufficient_bounce_samples"
        else:
            # Explore-mode legacy behavior.
            out["bounce_attempted"] = True
            T_hi = max(T_min + 1e-4, min(T_max, Tc * 0.995))
            T_lo = T_min
            if T_hi <= T_lo + 1e-4:
                out["fail_reason"] = "Tc_too_low_for_nucleation_scan"
            else:
                T_scan = np.linspace(T_hi, T_lo, 16)
                s3_records: List[Tuple[float, float, Optional[Dict[str, np.ndarray]], MinimaInfo]] = []
                for T in T_scan:
                    info = minima_info(case, float(T), phi_max)
                    if not (info.two_phase and info.barrier_exists):
                        continue
                    S3, prof, reason, _ = bounce_action_S3(case, float(T), info)
                    if S3 is None or not np.isfinite(S3):
                        continue
                    s3ot = float(S3 / max(T, 1e-9))
                    if np.isfinite(s3ot):
                        s3_records.append((float(T), s3ot, prof, info))

                if len(s3_records) > 0:
                    sval = np.array([x[1] for x in s3_records], dtype=float)
                    tval = np.array([x[0] for x in s3_records], dtype=float)
                    jmin = int(np.argmin(sval))
                    out["min_S3_over_T"] = float(sval[jmin])
                    out["T_at_min_S3_over_T"] = float(tval[jmin])
                    crossed_target = bool(np.any(sval <= float(S3_over_T_target)))

                    if len(s3_records) >= 2:
                        cross = None
                        for (Ta, Sa, pa, ia), (Tb, Sb, pb, ib) in zip(s3_records[:-1], s3_records[1:]):
                            if (Sa - S3_over_T_target) == 0.0:
                                cross = (Ta, Sa, pa, ia, Ta, Sa, pa, ia)
                                break
                            if (Sa - S3_over_T_target) * (Sb - S3_over_T_target) < 0:
                                cross = (Ta, Sa, pa, ia, Tb, Sb, pb, ib)
                                break
                        if cross is not None:
                            Ta, Sa, pa, ia, Tb, Sb, pb, ib = cross
                            if abs(Sb - Sa) > EPS:
                                Tn = float(Ta + (S3_over_T_target - Sa) * (Tb - Ta) / (Sb - Sa))
                            else:
                                Tn = float(Ta)
                            infoTn = minima_info(case, Tn, phi_max)
                            S3, prof, _, _ = bounce_action_S3(case, Tn, infoTn)
                            if S3 is not None and np.isfinite(S3):
                                S3T_Tn = float(S3 / max(Tn, 1e-9))
                                best_profile = prof
                            else:
                                if abs(Sa - S3_over_T_target) <= abs(Sb - S3_over_T_target):
                                    Tn, S3T_Tn, best_profile, infoTn = Ta, Sa, pa, ia
                                else:
                                    Tn, S3T_Tn, best_profile, infoTn = Tb, Sb, pb, ib
                        else:
                            out["fail_reason"] = "no_S3_over_T_crossing"
                    else:
                        out["fail_reason"] = "insufficient_bounce_samples"
                else:
                    out["fail_reason"] = "insufficient_bounce_samples"

    out["crossed_target"] = bool(crossed_target)

    if np.isfinite(Tn):
        out["Tn"] = float(Tn)
        out["S3_over_T_at_Tn"] = float(S3T_Tn) if np.isfinite(S3T_Tn) else np.nan
        T_ref = float(Tn)
        info_ref = infoTn if infoTn is not None else minima_info(case, T_ref, phi_max)
    else:
        T_ref = float(Tc)
        info_ref = infoTc

    # Thermodynamic proxies at T_ref.
    L = latent_heat_proxy(case, T_ref=T_ref, phi_max=phi_max)
    if L is not None and np.isfinite(L):
        out["L_proxy"] = float(L)
        out["alpha_PT"] = float(L / max(rho_rad(T_ref, gstar), 1e-20))

    # beta/H proxy at Tn.
    if np.isfinite(Tn):
        dT = max(1e-3, 0.02 * float(Tn))
        info_m = minima_info(case, max(T_min, float(Tn) - dT), phi_max)
        info_p = minima_info(case, min(T_max, float(Tn) + dT), phi_max)
        S3m, _, _, _ = bounce_action_S3(case, max(T_min, float(Tn) - dT), info_m) if info_m.two_phase and info_m.barrier_exists else (None, None, "", {})
        S3p, _, _, _ = bounce_action_S3(case, min(T_max, float(Tn) + dT), info_p) if info_p.two_phase and info_p.barrier_exists else (None, None, "", {})
        if S3m is not None and S3p is not None and np.isfinite(S3m) and np.isfinite(S3p):
            y_m = float(S3m / max(float(Tn) - dT, 1e-9))
            y_p = float(S3p / max(float(Tn) + dT, 1e-9))
            dSdT = (y_p - y_m) / (2.0 * dT)
            out["beta_over_H"] = float(float(Tn) * dSdT)

    # Wall thickness proxy from bounce profile preferred.
    if best_profile is not None and info_ref.true_phi is not None and info_ref.false_phi is not None:
        wall = wall_thickness_from_profile(best_profile, phi_false=float(info_ref.false_phi), phi_true=float(info_ref.true_phi))
        if wall is not None and np.isfinite(wall):
            out["wall_thickness_proxy"] = float(wall)

    return out


def _evaluate_case_payload(
    payload: Tuple[
        CaseParams,
        Dict[str, object],
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        bool,
        float,
        bool,
        int,
        str,
        List[float],
        bool,
        bool,
        float,
        int,
        str,
        int,
        float,
    ]
) -> Dict[str, object]:
    (
        case,
        meta,
        T_min,
        T_max,
        S3_over_T_target,
        gstar,
        tc_lo,
        tc_hi,
        tn_lo,
        tn_hi,
        require_tc_in_window,
        barrier_min,
        apply_gate,
        min_bounce_samples,
        bounce_T_strategy,
        bounce_fracs,
        bounce_use_barrier_filter,
        bounce_allow_no_barrier,
        bounce_T_min_frac,
        bounce_max_T_points,
        bounce_fallback,
        thinwall_phi_grid,
        thinwall_eps,
    ) = payload
    out = evaluate_case(
        case=case,
        T_min=T_min,
        T_max=T_max,
        S3_over_T_target=S3_over_T_target,
        gstar=gstar,
        tc_window=(tc_lo, tc_hi),
        tn_window=(tn_lo, tn_hi),
        require_tc_in_window=require_tc_in_window,
        barrier_min=barrier_min,
        apply_bounce_gate=apply_gate,
        min_bounce_samples=min_bounce_samples,
        bounce_T_strategy=bounce_T_strategy,
        bounce_fracs=bounce_fracs,
        bounce_use_barrier_filter=bounce_use_barrier_filter,
        bounce_allow_no_barrier=bounce_allow_no_barrier,
        bounce_T_min_frac=bounce_T_min_frac,
        bounce_max_T_points=bounce_max_T_points,
        bounce_fallback=bounce_fallback,
        thinwall_phi_grid=thinwall_phi_grid,
        thinwall_eps=thinwall_eps,
    )
    out.update(meta)
    return out


# -------------------------------
# Reporting and plotting
# -------------------------------
def choose_best_cases(
    df: pd.DataFrame,
    Tn_target: float,
    Tn_window: Tuple[float, float],
    best_rank_mode: str = "min_S3_over_T",
    n_best: int = 10,
) -> pd.DataFrame:
    work = df.copy()
    work["has_Tn"] = np.isfinite(work["Tn"].to_numpy(dtype=float))
    work["score"] = 1e9
    lo, hi = Tn_window

    if best_rank_mode == "min_S3_over_T":
        mask = work["first_order_flag"].astype(bool) & np.isfinite(work["min_S3_over_T"].to_numpy(dtype=float))
        if np.any(mask):
            work.loc[mask, "score"] = work.loc[mask, "min_S3_over_T"].to_numpy(dtype=float)
            # Prefer scans that crossed target.
            work.loc[mask, "score"] -= 20.0 * work.loc[mask, "crossed_target"].astype(bool).astype(float)
    elif best_rank_mode == "strongest_alpha":
        mask = work["first_order_flag"].astype(bool) & np.isfinite(work["alpha_PT"].to_numpy(dtype=float))
        if np.any(mask):
            alpha = np.maximum(work.loc[mask, "alpha_PT"].to_numpy(dtype=float), 1e-30)
            work.loc[mask, "score"] = -np.log10(alpha)
    else:
        mask = work["first_order_flag"].astype(bool) & work["has_Tn"].astype(bool)
        if np.any(mask):
            tn = work.loc[mask, "Tn"].to_numpy(dtype=float)
            work.loc[mask, "score"] = np.abs(np.log10(np.maximum(tn, 1e-9) / max(Tn_target, 1e-9)))

    # prioritize in-window points
    in_window = mask & (work["Tn"].to_numpy(dtype=float) >= lo) & (work["Tn"].to_numpy(dtype=float) <= hi)
    if np.any(in_window):
        chosen = work.loc[in_window].sort_values(["score", "model_id"]).head(n_best).copy()
    else:
        chosen = work.loc[mask].sort_values(["score", "model_id"]).head(n_best).copy()
    return chosen


def case_from_row(row: pd.Series) -> CaseParams:
    return CaseParams(
        model_id=str(row["model_id"]),
        m2=float(row["m2"]) if np.isfinite(row["m2"]) else float("nan"),
        lambda4=float(row["lambda4"]),
        cT2=float(row["cT2"]),
        lambda6_or_v_or_a3=float(row["lambda6_or_v_or_a3"]),
        v_scale=float(row["v_scale"]),
        thermal_cubic_E=float(row["E_used"]) if "E_used" in row and np.isfinite(row["E_used"]) else 0.0,
    )


def make_plots(df: pd.DataFrame, best_df: pd.DataFrame, out_png: Path, T_min: float, T_max: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axA, axB, axC, axD = axes.flat

    # Panel A: histogram Tc/Tn for first-order cases
    fo = df[df["first_order_flag"].astype(bool)].copy()
    if not fo.empty:
        tc = fo["Tc"].to_numpy(dtype=float)
        tc = tc[np.isfinite(tc)]
        if tc.size:
            axA.hist(tc, bins=24, alpha=0.6, label="Tc", color="#1f77b4")
        tn = fo["Tn"].to_numpy(dtype=float)
        tn = tn[np.isfinite(tn)]
        if tn.size:
            axA.hist(tn, bins=24, alpha=0.6, label="Tn", color="#d62728")
        axA.legend(loc="best")
    else:
        axA.text(0.5, 0.5, "No first-order cases", ha="center", transform=axA.transAxes)
    axA.set_xlabel("Temperature [GeV]")
    axA.set_ylabel("Count")
    axA.set_title("Panel A: Tc/Tn distribution (first-order candidates)")

    # Panel B: alpha vs beta/H
    viable = df[
        df["first_order_flag"].astype(bool)
        & np.isfinite(df["alpha_PT"].to_numpy(dtype=float))
        & np.isfinite(df["beta_over_H"].to_numpy(dtype=float))
    ].copy()
    if not viable.empty:
        for mid, g in viable.groupby("model_id", sort=False):
            axB.scatter(
                g["alpha_PT"].to_numpy(dtype=float),
                g["beta_over_H"].to_numpy(dtype=float),
                s=18,
                alpha=0.7,
                label=mid,
            )
        axB.set_xscale("log")
        axB.set_yscale("log")
        axB.legend(loc="best", fontsize=8)
    else:
        axB.text(0.5, 0.5, "No viable alpha/beta points", ha="center", transform=axB.transAxes)
    axB.set_xlabel("alpha_PT")
    axB.set_ylabel("beta/H")
    axB.set_title("Panel B: alpha_PT vs beta/H")

    # Panel C: Veff(phi,Tc/Tn) for best case per model
    colors = {
        MODEL_POLY6: "#1f77b4",
        MODEL_QUARTIC: "#2ca02c",
        MODEL_CUBIC: "#d62728",
        MODEL_CONTROL: "#7f7f7f",
    }
    used_any = False
    for mid in ALL_MODELS:
        g = best_df[best_df["model_id"] == mid]
        if g.empty:
            continue
        row = g.sort_values("score").iloc[0]
        case = case_from_row(row)
        phi_max = max(200.0, 5.0 * max(case.v_scale, abs(case.lambda6_or_v_or_a3) if np.isfinite(case.lambda6_or_v_or_a3) else 0.0, 1.0))
        phi = np.linspace(0.0, phi_max, 500)
        if np.isfinite(row["Tc"]):
            Tc = float(row["Tc"])
            axC.plot(phi, Veff(case, phi, Tc), color=colors.get(mid, None), lw=1.8, label=f"{mid} @ Tc")
            used_any = True
        if np.isfinite(row["Tn"]):
            Tn = float(row["Tn"])
            axC.plot(phi, Veff(case, phi, Tn), color=colors.get(mid, None), lw=1.1, ls="--", label=f"{mid} @ Tn")
            used_any = True
    if not used_any:
        axC.text(0.5, 0.5, "No Tc/Tn curves available", ha="center", transform=axC.transAxes)
    else:
        axC.legend(loc="best", fontsize=8)
    axC.set_xlabel("phi [GeV]")
    axC.set_ylabel("Veff(phi,T) [arb. GeV^4]")
    axC.set_title("Panel C: Example Veff curves for best cases")

    # Panel D: bounce profile for best cases (where available)
    used_prof = False
    for mid in (MODEL_POLY6, MODEL_CUBIC, MODEL_QUARTIC, MODEL_CONTROL):
        g = best_df[best_df["model_id"] == mid]
        if g.empty:
            continue
        row = g.sort_values("score").iloc[0]
        if not np.isfinite(row["Tn"]):
            continue
        case = case_from_row(row)
        Tn = float(row["Tn"])
        phi_max = float(row.get("_phi_max", max(200.0, 5.0 * max(case.v_scale, 1.0))))
        info = minima_info(case, Tn, phi_max)
        if not (info.two_phase and info.barrier_exists):
            continue
        S3, profile, reason, _ = bounce_action_S3(case, Tn, info)
        if profile is None:
            continue
        axD.plot(profile["r"], profile["phi"], lw=1.5, label=f"{mid}")
        used_prof = True
    if used_prof:
        axD.legend(loc="best", fontsize=8)
    else:
        axD.text(0.5, 0.5, "No bounce profile available", ha="center", transform=axD.transAxes)
    axD.set_xlabel("r [GeV^-1]")
    axD.set_ylabel("phi(r) [GeV]")
    axD.set_title("Panel D: Bounce profiles for best cases")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def summarize_model_counts(df: pd.DataFrame, tc_window: Tuple[float, float], tn_window: Tuple[float, float]) -> pd.DataFrame:
    lo_tc, hi_tc = tc_window
    lo, hi = tn_window
    rows: List[Dict[str, object]] = []
    for mid, g in df.groupby("model_id", sort=False):
        tc_found = int(np.sum(np.isfinite(g["Tc"].to_numpy(dtype=float))))
        tc_in_win = int(
            np.sum(
                np.isfinite(g["Tc"].to_numpy(dtype=float))
                & (g["Tc"].to_numpy(dtype=float) >= lo_tc)
                & (g["Tc"].to_numpy(dtype=float) <= hi_tc)
            )
        )
        in_win = int(
            np.sum(
                np.isfinite(g["Tn"].to_numpy(dtype=float))
                & (g["Tn"].to_numpy(dtype=float) >= lo)
                & (g["Tn"].to_numpy(dtype=float) <= hi)
            )
        )
        rows.append(
            {
                "model_id": mid,
                "total": int(len(g)),
                "valid": int(np.sum(g["valid_flag"].astype(bool))),
                "first_order": int(np.sum(g["first_order_flag"].astype(bool))),
                "Tc_found": tc_found,
                "Tc_in_window": tc_in_win,
                "crossed_target_anyT": int(np.sum(g["crossed_target"].astype(bool))) if "crossed_target" in g.columns else 0,
                "BounceAttempted": int(np.sum(g["bounce_attempted"].astype(bool))) if "bounce_attempted" in g.columns else 0,
                "BounceSuccess": int(np.sum(g["bounce_success"].astype(bool))) if "bounce_success" in g.columns else 0,
                "AnyFiniteMinS3T": int(
                    np.sum(np.isfinite(g["min_S3_over_T"].to_numpy(dtype=float)))
                )
                if "min_S3_over_T" in g.columns
                else 0,
                "ThinwallFinite": int(
                    np.sum(np.isfinite(g["thinwall_min_S3_over_T"].to_numpy(dtype=float)))
                )
                if "thinwall_min_S3_over_T" in g.columns
                else 0,
                "ThinwallCrossedTarget": int(np.sum(g["thinwall_crossed_target"].astype(bool)))
                if "thinwall_crossed_target" in g.columns
                else 0,
                "Tn_in_window": in_win,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def minimal_threshold_rows(df: pd.DataFrame, tc_window: Tuple[float, float], tn_window: Tuple[float, float]) -> pd.DataFrame:
    lo_tc, hi_tc = tc_window
    lo_tn, hi_tn = tn_window
    rows: List[Dict[str, object]] = []
    for mid, g in df.groupby("model_id", sort=False):
        e_col = g["E_used"].to_numpy(dtype=float) if "E_used" in g.columns else np.full(len(g), np.nan)
        tc = g["Tc"].to_numpy(dtype=float)
        tn = g["Tn"].to_numpy(dtype=float)
        fo = g["first_order_flag"].astype(bool).to_numpy()
        ct = g["crossed_target"].astype(bool).to_numpy() if "crossed_target" in g.columns else np.zeros(len(g), dtype=bool)
        mask_fo = fo & np.isfinite(tc) & (tc >= lo_tc) & (tc <= hi_tc)
        mask_tn = ct & np.isfinite(tn) & (tn >= lo_tn) & (tn <= hi_tn)

        row: Dict[str, object] = {"model_id": mid}
        if np.any(mask_fo):
            idx = int(np.nanargmin(np.where(mask_fo, e_col, np.inf)))
            r = g.iloc[idx]
            row["first_order_threshold_E"] = float(r["E_used"]) if "E_used" in g.columns and np.isfinite(r["E_used"]) else np.nan
            row["first_order_threshold_kE"] = float(r["kE_used"]) if "kE_used" in g.columns and np.isfinite(r["kE_used"]) else np.nan
            row["first_order_threshold_g"] = float(r["g_portal_used"]) if "g_portal_used" in g.columns and np.isfinite(r["g_portal_used"]) else np.nan
        else:
            row["first_order_threshold_E"] = np.nan
            row["first_order_threshold_kE"] = np.nan
            row["first_order_threshold_g"] = np.nan

        if np.any(mask_tn):
            idx = int(np.nanargmin(np.where(mask_tn, e_col, np.inf)))
            r = g.iloc[idx]
            row["tn_window_threshold_E"] = float(r["E_used"]) if "E_used" in g.columns and np.isfinite(r["E_used"]) else np.nan
            row["tn_window_threshold_kE"] = float(r["kE_used"]) if "kE_used" in g.columns and np.isfinite(r["kE_used"]) else np.nan
            row["tn_window_threshold_g"] = float(r["g_portal_used"]) if "g_portal_used" in g.columns and np.isfinite(r["g_portal_used"]) else np.nan
        else:
            row["tn_window_threshold_E"] = np.nan
            row["tn_window_threshold_kE"] = np.nan
            row["tn_window_threshold_g"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(
    out_md: Path,
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    args: argparse.Namespace,
    csv_sha: str,
    git_sha: Optional[str],
    run_meta: Dict[str, object],
) -> None:
    lines: List[str] = []
    now = dt.datetime.now(dt.UTC).isoformat()
    tc_window = parse_window(args.Tc_window)
    tn_window = parse_window(args.Tn_window)
    model_counts = summarize_model_counts(df, tc_window=tc_window, tn_window=tn_window)
    lines.append("# Finite-T Phase Transition Report")
    lines.append("")
    lines.append("## Stamp")
    lines.append(f"- timestamp_utc: {now}")
    lines.append(f"- git_head: {git_sha if git_sha else 'None'}")
    lines.append(f"- summary_csv_sha256: {csv_sha}")
    if "mode" in run_meta:
        lines.append(f"- mode: {run_meta['mode']}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- T_min: {args.T_min}")
    lines.append(f"- T_max: {args.T_max}")
    lines.append(f"- S3/T target: {args.S3_over_T_target}")
    lines.append(f"- g*: {args.gstar}")
    lines.append(f"- Tn target: {args.Tn_target}")
    lines.append(f"- Tc window: {args.Tc_window}")
    lines.append(f"- Tn window: {args.Tn_window}")
    lines.append(f"- require_Tc_in_window: {args.require_Tc_in_window}")
    lines.append(f"- best_rank_mode: {args.best_rank_mode}")
    lines.append(f"- max_cases_total: {args.max_cases_total}")
    lines.append(f"- max_constrained_cases: {args.max_constrained_cases}")
    lines.append(f"- cT2_mode: {args.cT2_mode}")
    if args.cT2_mode == "self_plus_portal":
        lines.append(f"- g_portal_list: {args.g_portal_list}")
    if args.cT2_mode == "free_scan":
        lines.append(f"- cT2_free: {args.cT2_free}")
    lines.append(f"- lambda4_map: {args.lambda4_map}")
    lines.append(f"- barrier_min: {args.barrier_min}")
    lines.append(f"- min_bounce_samples: {args.min_bounce_samples}")
    lines.append(f"- bounce_T_strategy: {args.bounce_T_strategy}")
    lines.append(f"- bounce_fracs: {args.bounce_fracs}")
    lines.append(f"- bounce_use_barrier_filter: {args.bounce_use_barrier_filter}")
    lines.append(f"- bounce_allow_no_barrier: {args.bounce_allow_no_barrier}")
    lines.append(f"- bounce_T_min_frac: {args.bounce_T_min_frac}")
    lines.append(f"- bounce_max_T_points: {args.bounce_max_T_points}")
    lines.append(f"- bounce_fallback: {args.bounce_fallback}")
    lines.append(f"- thinwall_phi_grid: {args.thinwall_phi_grid}")
    lines.append(f"- thinwall_eps: {args.thinwall_eps}")
    lines.append(f"- thermal_cubic_mode: {args.thermal_cubic_mode}")
    if args.thermal_cubic_mode == "free_scan":
        lines.append(f"- E_free: {args.E_free}")
        lines.append(f"- E_list: {args.E_list}")
    if args.thermal_cubic_mode == "portal_proxy":
        lines.append(f"- kE_list: {args.kE_list}")
    if "source_path" in run_meta and run_meta["source_path"]:
        lines.append(f"- constrained_source: {run_meta['source_path']}")
    if "n_selected" in run_meta:
        lines.append(f"- N_selected: {run_meta['n_selected']}")
    if "n_portal" in run_meta:
        lines.append(f"- N_portal: {run_meta['n_portal']}")
    if "n_E" in run_meta:
        lines.append(f"- N_E: {run_meta['n_E']}")
    if "n_total_evals" in run_meta:
        lines.append(f"- N_total_evals: {run_meta['n_total_evals']}")
    lines.append("")

    lines.append("## Model-level outcomes")
    if model_counts.empty:
        lines.append("- No model rows found")
    else:
        lines.append(dataframe_to_markdown(model_counts))
    lines.append("")

    lines.append("## min(S3/T) diagnostics")
    if "min_S3_over_T" in df.columns:
        ms = df["min_S3_over_T"].to_numpy(dtype=float)
        ms = ms[np.isfinite(ms)]
        if ms.size:
            hist, edges = np.histogram(ms, bins=12)
            lines.append("- Histogram of min_S3_over_T (counts by bin):")
            for i in range(len(hist)):
                lines.append(f"  - [{edges[i]:.3g}, {edges[i+1]:.3g}): {int(hist[i])}")
            lines.append(f"- crossed_target_anywhere: {int(np.sum(df['crossed_target'].astype(bool)))}")
        else:
            lines.append("- No finite min_S3_over_T values.")
    else:
        lines.append("- min_S3_over_T not available.")

    lines.append("")
    lines.append("## Threshold summary")
    thr = minimal_threshold_rows(df, tc_window=tc_window, tn_window=tn_window)
    if thr.empty:
        lines.append("- No threshold rows.")
    else:
        lines.append(dataframe_to_markdown(thr))

    lines.append("")
    lines.append("## Control sanity check")
    ctrl = df[df["model_id"] == MODEL_CONTROL]
    ctrl_first = int(np.sum(ctrl["first_order_flag"].astype(bool))) if not ctrl.empty else 0
    ctrl_tn = int(np.sum(np.isfinite(ctrl["Tn"].to_numpy(dtype=float)))) if not ctrl.empty else 0
    lines.append(
        f"- control model (`{MODEL_CONTROL}`): first_order_count={ctrl_first}, finite_Tn_count={ctrl_tn}. "
        "Expected behavior is crossover/second-order with no robust nucleation."
    )
    lines.append("")

    lines.append("## Best cases (targeted around Tn)")
    if best_df.empty:
        lines.append("- No first-order cases with finite Tn were found in this scan.")
    else:
        show_cols = [
            "model_id",
            "Tc",
            "Tn",
            "alpha_PT",
            "beta_over_H",
            "L_proxy",
            "wall_thickness_proxy",
            "m2",
            "lambda4",
            "lambda6_or_v_or_a3",
            "v_scale",
            "cT2",
            "score",
        ]
        lines.append("")
        lines.append(dataframe_to_markdown(best_df.loc[:, show_cols]))

    lines.append("")
    lines.append("## Caveats")
    lines.append("- This module uses a minimal thermal-mass correction only (no one-loop thermal cubic terms).")
    lines.append("- First-order classification is based on finite-T V_eff structure (degenerate minima + barrier at Tc).")
    lines.append("- Nucleation uses numerical O(3) bounce shooting and may fail for stiff parameter points.")
    if str(getattr(args, "mode", "explore")) in {"constrained", "curated"}:
        lines.append("- Constrained/curated mode uses asymmetry-derived or curated parameters (no generic m2/lambda scans).")
    lines.append("- This module does not include explicit baryon/CP-violating portals; baryogenesis claims are deferred.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_window(window_text: str) -> Tuple[float, float]:
    parts = [x.strip() for x in str(window_text).split(",")]
    if len(parts) != 2:
        raise ValueError("--Tn_window must be formatted as 'lo,hi'")
    lo = float(parts[0])
    hi = float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid --Tn_window bounds")
    return lo, hi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finite-T phase transition scan (Paper 3).")
    p.add_argument("--mode", type=str, choices=["explore", "constrained", "curated"], default="explore")
    p.add_argument("--T_min", type=float, default=0.1)
    p.add_argument("--T_max", type=float, default=200.0)
    p.add_argument("--S3_over_T_target", type=float, default=140.0)
    p.add_argument("--gstar", type=float, default=100.0)
    p.add_argument("--Tn_target", type=float, default=20.0)
    p.add_argument("--Tc_window", type=str, default="10,50")
    p.add_argument("--Tn_window", type=str, default="10,50")
    p.add_argument("--require_Tc_in_window", action="store_true", default=True)
    p.add_argument("--no_require_Tc_in_window", action="store_false", dest="require_Tc_in_window")
    p.add_argument("--best_rank_mode", type=str, choices=["Tn_closest", "min_S3_over_T", "strongest_alpha"], default="min_S3_over_T")
    p.add_argument("--max_cases_total", type=int, default=900)
    p.add_argument("--asymmetry_csv", type=str, default=DEFAULT_ASYMMETRY_CSV)
    p.add_argument("--select_near_lambda_dex", type=float, default=3.0)
    p.add_argument("--max_constrained_cases", type=int, default=500)
    p.add_argument("--always_include_best", action="store_true", default=True)
    p.add_argument("--no_always_include_best", action="store_false", dest="always_include_best")
    p.add_argument("--cT2_mode", type=str, choices=["free_scan", "self_only", "self_plus_portal"], default="self_only")
    p.add_argument("--cT2_free", type=float, default=None)
    p.add_argument("--g_portal_list", type=str, default="0")
    p.add_argument("--lambda4_map", type=str, choices=["real_scalar", "complex_scalar"], default="real_scalar")
    p.add_argument("--curated_json", type=str, default=DEFAULT_CURATED_JSON)
    p.add_argument("--barrier_min", type=float, default=0.0)
    p.add_argument("--thermal_cubic_mode", type=str, choices=["off", "free_scan", "portal_proxy"], default="off")
    p.add_argument("--E_list", type=str, default="0,1e-6,1e-5,1e-4,1e-3,1e-2,0.03,0.05,0.1,0.2,0.3,0.5")
    p.add_argument("--kE_list", type=str, default="0.1,0.3,1,3,10")
    p.add_argument("--E_free", type=float, default=None)
    p.add_argument("--min_bounce_samples", type=int, default=1)
    p.add_argument("--bounce_T_strategy", type=str, choices=["grid", "fracs", "hybrid"], default="hybrid")
    p.add_argument("--bounce_fracs", type=str, default="0.98,0.95,0.9,0.85,0.8,0.7")
    p.add_argument("--bounce_use_barrier_filter", action="store_true", default=True)
    p.add_argument("--no_bounce_use_barrier_filter", action="store_false", dest="bounce_use_barrier_filter")
    p.add_argument("--bounce_allow_no_barrier", action="store_true", default=False)
    p.add_argument("--bounce_T_min_frac", type=float, default=0.5)
    p.add_argument("--bounce_max_T_points", type=int, default=16)
    p.add_argument("--bounce_fallback", type=str, choices=["none", "thinwall"], default="thinwall")
    p.add_argument("--thinwall_phi_grid", type=int, default=2000)
    p.add_argument("--thinwall_eps", type=float, default=1e-18)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--auto_jobs", action="store_true", default=False)
    p.add_argument("--max_jobs_cap", type=int, default=0)
    p.add_argument("--reserve_cpus", type=int, default=1)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument(
        "--n_jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker count for case evaluation.",
    )
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    repo_root = Path(__file__).resolve().parents[2]
    tc_lo, tc_hi = parse_window(args.Tc_window)
    tn_lo, tn_hi = parse_window(args.Tn_window)
    bounce_fracs = parse_csv_float_list(args.bounce_fracs)
    tstamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        suffix = ""
        if args.mode == "constrained":
            suffix = "_constrained"
        elif args.mode == "curated":
            suffix = "_curated"
        base_out_dir = repo_root / "outputs" / "paper3_finiteT" / f"{tstamp}{suffix}"
    else:
        user_out = Path(args.out_dir).expanduser()
        base_out_dir = user_out if user_out.is_absolute() else (repo_root / user_out)
    if str(args.run_tag).strip():
        base_out_dir = base_out_dir / str(args.run_tag).strip()
    out_dir = base_out_dir / f"shard_{int(args.shard_id)}_of_{int(args.num_shards)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "explore":
        summary_csv = out_dir / "finiteT_scan_summary.csv"
        constrained_used_csv = None
    else:
        summary_csv = out_dir / "constrained_finiteT_results.csv"
        constrained_used_csv = out_dir / "constrained_cases_used.csv"
    best_json = out_dir / "finiteT_best_cases.json"
    plot_png = out_dir / "finiteT_plots.png"
    report_md = out_dir / "finiteT_report.md"
    run_meta_json = out_dir / "run_metadata.json"

    source_path = ""
    if args.mode == "explore":
        print("[FT] Building scan library...", flush=True)
        all_cases_generated = build_case_library()
        if int(args.max_cases_total) <= 0:
            all_cases = all_cases_generated
        else:
            all_cases = all_cases_generated[: int(args.max_cases_total)]
        cased_full = [
            (
                case,
                {
                    "source_mode": "explore",
                    "thermal_cubic_mode": "off",
                    "E_used": 0.0,
                    "kE_used": np.nan,
                    "g_portal_used": 0.0,
                },
            )
            for case in all_cases
        ]
        selected_df = pd.DataFrame()
        n_portal = 1
        n_e = 1
    elif args.mode == "constrained":
        cased_full, selected_df, _, n_portal, n_e, src = build_constrained_cases(args, repo_root)
        source_path = str(src)
    else:
        cased_full, selected_df, _, n_portal, n_e, src = build_curated_cases(args, repo_root)
        source_path = str(src)

    # Assign deterministic global case_index after full expansion.
    cased_full = [
        (
            case,
            {**meta, "case_index": int(i)},
        )
        for i, (case, meta) in enumerate(cased_full)
    ]
    n_generated_total = len(cased_full)

    # Deterministic shard split on expanded case_index.
    shard_id = int(args.shard_id)
    num_shards = int(args.num_shards)
    cased = [(case, meta) for case, meta in cased_full if (int(meta["case_index"]) % num_shards) == shard_id]
    n_selected_for_this_shard = len(cased)

    cpu_count = int(os.cpu_count() or 1)
    if bool(args.auto_jobs):
        reserve = max(0, int(args.reserve_cpus))
        n_jobs = max(1, cpu_count - reserve)
        if int(args.max_jobs_cap) > 0:
            n_jobs = min(n_jobs, int(args.max_jobs_cap))
        jobs_mode = "auto"
    else:
        n_jobs = max(1, int(args.n_jobs))
        jobs_mode = "manual"

    if args.mode == "explore":
        print(f"MODE=explore, N_generated_total={n_generated_total}, N_evaluating={n_selected_for_this_shard}", flush=True)
    else:
        print(
            f"MODE={args.mode}, N_generated_total={n_generated_total}, N_selected_for_this_shard={n_selected_for_this_shard}, "
            f"N_portal={n_portal}, N_E={n_e}",
            flush=True,
        )

    print(
        f"[FT] shard={shard_id}/{num_shards} cpu_count={cpu_count} n_jobs={n_jobs} jobs_mode={jobs_mode}",
        flush=True,
    )
    print(f"[FT] Parallel workers: {n_jobs}", flush=True)

    run_meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "mode": args.mode,
        "N_generated_total": int(n_generated_total),
        "N_selected_for_this_shard": int(n_selected_for_this_shard),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "n_jobs_chosen": int(n_jobs),
        "cpu_count_detected": int(cpu_count),
        "jobs_mode": jobs_mode,
        "source_path": source_path,
        "args_dump": json.dumps(vars(args), sort_keys=True, default=str),
        "cloud_examples": {
            "single_vm": "python3 analysis/paper3_cosmology/finiteT_phase_transition.py --mode explore --num_shards 1 --shard_id 0 --auto_jobs",
            "two_vms_shard0": "python3 analysis/paper3_cosmology/finiteT_phase_transition.py --mode explore --num_shards 2 --shard_id 0 --auto_jobs",
            "two_vms_shard1": "python3 analysis/paper3_cosmology/finiteT_phase_transition.py --mode explore --num_shards 2 --shard_id 1 --auto_jobs",
            "batch_array": "python3 analysis/paper3_cosmology/finiteT_phase_transition.py --mode explore --num_shards N --shard_id $BATCH_TASK_INDEX --auto_jobs",
        },
    }
    run_meta_json.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    rows: List[Dict[str, object]] = []
    apply_gate = bool(args.mode != "explore")
    if n_jobs == 1:
        for i, (case, meta) in enumerate(cased, start=1):
            if i % 100 == 0 or i == 1 or i == len(cased):
                print(f"[FT] Evaluating case {i}/{len(cased)}", flush=True)
            out = evaluate_case(
                case=case,
                T_min=float(args.T_min),
                T_max=float(args.T_max),
                S3_over_T_target=float(args.S3_over_T_target),
                gstar=float(args.gstar),
                tc_window=(tc_lo, tc_hi),
                tn_window=(tn_lo, tn_hi),
                require_tc_in_window=bool(args.require_Tc_in_window),
                barrier_min=float(args.barrier_min),
                apply_bounce_gate=apply_gate,
                min_bounce_samples=int(args.min_bounce_samples),
                bounce_T_strategy=str(args.bounce_T_strategy),
                bounce_fracs=bounce_fracs,
                bounce_use_barrier_filter=bool(args.bounce_use_barrier_filter),
                bounce_allow_no_barrier=bool(args.bounce_allow_no_barrier),
                bounce_T_min_frac=float(args.bounce_T_min_frac),
                bounce_max_T_points=int(args.bounce_max_T_points),
                bounce_fallback=str(args.bounce_fallback),
                thinwall_phi_grid=int(args.thinwall_phi_grid),
                thinwall_eps=float(args.thinwall_eps),
            )
            out.update(meta)
            rows.append(out)
    else:
        payloads = [
            (
                case,
                meta,
                float(args.T_min),
                float(args.T_max),
                float(args.S3_over_T_target),
                float(args.gstar),
                float(tc_lo),
                float(tc_hi),
                float(tn_lo),
                float(tn_hi),
                bool(args.require_Tc_in_window),
                float(args.barrier_min),
                bool(apply_gate),
                int(args.min_bounce_samples),
                str(args.bounce_T_strategy),
                bounce_fracs,
                bool(args.bounce_use_barrier_filter),
                bool(args.bounce_allow_no_barrier),
                float(args.bounce_T_min_frac),
                int(args.bounce_max_T_points),
                str(args.bounce_fallback),
                int(args.thinwall_phi_grid),
                float(args.thinwall_eps),
            )
            for case, meta in cased
        ]
        chunksize = max(1, len(payloads) // (n_jobs * 8))
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, out in enumerate(ex.map(_evaluate_case_payload, payloads, chunksize=chunksize), start=1):
                if i % 100 == 0 or i == 1 or i == len(payloads):
                    print(f"[FT] Evaluating case {i}/{len(payloads)}", flush=True)
                rows.append(out)

    df = pd.DataFrame.from_records(rows)
    # Keep requested columns first.
    cols = [
        "model_id",
        "valid_flag",
        "fail_reason",
        "m2",
        "lambda4",
        "lambda6_or_v_or_a3",
        "v_scale",
        "cT2",
        "E_used",
        "Tc",
        "Tc_found",
        "Tn",
        "first_order_flag",
        "phi_false_Tc",
        "phi_true_Tc",
        "S3_over_T_at_Tn",
        "min_S3_over_T",
        "T_at_min_S3_over_T",
        "crossed_target",
        "bounce_attempted",
        "bounce_success",
        "n_bounce_success",
        "n_bounce_T_raw",
        "n_bounce_T_kept",
        "barrier_present_count",
        "bounce_T_list_used",
        "bounce_fail_code_counts",
        "bounce_bracket_found",
        "bounce_F_lo",
        "bounce_F_hi",
        "bounce_phi0_root",
        "bounce_n_bisect_iters",
        "bounce_r_max_used",
        "bounce_m_eff_used",
        "thinwall_min_S3_over_T",
        "thinwall_T_at_min",
        "thinwall_crossed_target",
        "thinwall_attempted_count",
        "thinwall_fail_reason",
        "barrier_height_Tc",
        "alpha_PT",
        "beta_over_H",
        "L_proxy",
        "wall_thickness_proxy",
    ]
    extra_cols = [c for c in df.columns if c not in cols]
    df = df.loc[:, cols + extra_cols]
    df.to_csv(summary_csv, index=False)

    if constrained_used_csv is not None:
        selected_df_shard = pd.DataFrame([meta for _, meta in cased])
        if not selected_df_shard.empty:
            selected_df_shard.to_csv(constrained_used_csv, index=False)

    rank_mode = args.best_rank_mode if args.mode != "explore" else "Tn_closest"
    best_df = choose_best_cases(
        df,
        Tn_target=float(args.Tn_target),
        Tn_window=(tn_lo, tn_hi),
        best_rank_mode=str(rank_mode),
        n_best=10,
    )

    # JSON payload with best cases and per-model best entry.
    payload: Dict[str, object] = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_head": git_head_sha(repo_root),
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": sha256_file(summary_csv),
        "config": {
            "T_min": args.T_min,
            "T_max": args.T_max,
            "S3_over_T_target": args.S3_over_T_target,
            "gstar": args.gstar,
            "Tn_target": args.Tn_target,
            "Tc_window": args.Tc_window,
            "Tn_window": args.Tn_window,
            "require_Tc_in_window": args.require_Tc_in_window,
            "best_rank_mode": rank_mode,
            "max_cases_total": args.max_cases_total,
            "max_constrained_cases": args.max_constrained_cases,
            "mode": args.mode,
            "cT2_mode": args.cT2_mode,
            "g_portal_list": args.g_portal_list,
            "thermal_cubic_mode": args.thermal_cubic_mode,
            "E_list": args.E_list,
            "kE_list": args.kE_list,
            "E_free": args.E_free,
            "lambda4_map": args.lambda4_map,
            "barrier_min": args.barrier_min,
            "min_bounce_samples": args.min_bounce_samples,
            "bounce_T_strategy": args.bounce_T_strategy,
            "bounce_fracs": args.bounce_fracs,
            "bounce_use_barrier_filter": args.bounce_use_barrier_filter,
            "bounce_allow_no_barrier": args.bounce_allow_no_barrier,
            "bounce_T_min_frac": args.bounce_T_min_frac,
            "bounce_max_T_points": args.bounce_max_T_points,
            "bounce_fallback": args.bounce_fallback,
            "thinwall_phi_grid": args.thinwall_phi_grid,
            "thinwall_eps": args.thinwall_eps,
            "source_path": source_path,
            "N_selected_for_this_shard": n_selected_for_this_shard,
            "N_generated_total": n_generated_total,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "auto_jobs": args.auto_jobs,
            "reserve_cpus": args.reserve_cpus,
            "max_jobs_cap": args.max_jobs_cap,
            "run_tag": args.run_tag,
            "N_portal": n_portal,
            "N_E": n_e,
            "N_total_evals": len(cased),
        },
        "best_overall": best_df.to_dict(orient="records"),
        "best_per_model": {},
    }
    for mid in ALL_MODELS:
        gm = best_df[best_df["model_id"] == mid]
        payload["best_per_model"][mid] = gm.sort_values("score").head(1).to_dict(orient="records") if not gm.empty else []

    with best_json.open("w") as f:
        json.dump(payload, f, indent=2)

    make_plots(df, best_df, out_png=plot_png, T_min=float(args.T_min), T_max=float(args.T_max))
    build_report(
        out_md=report_md,
        df=df,
        best_df=best_df,
        args=args,
        csv_sha=payload["summary_csv_sha256"],
        git_sha=payload["git_head"],
        run_meta={
            "mode": args.mode,
            "source_path": source_path,
            "n_selected": n_selected_for_this_shard,
            "n_generated_total": n_generated_total,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "n_portal": n_portal,
            "n_E": n_e,
            "n_total_evals": len(cased),
        },
    )

    # Final console summary required by prompt.
    model_table = summarize_model_counts(df, tc_window=(tc_lo, tc_hi), tn_window=(tn_lo, tn_hi))
    print(
        "Model | Valid | First-Order | Tc in window | Crossed_Target(any T) | "
        "BounceAttempted | BounceSuccess | AnyFiniteMinS3T | ThinwallFinite | ThinwallCrossedTarget | Tn in window"
    )
    for _, row in model_table.iterrows():
        print(
            f"{row['model_id']} | {int(row['valid'])} | {int(row['first_order'])} | "
            f"{int(row['Tc_in_window'])} | {int(row['crossed_target_anyT'])} | "
            f"{int(row['BounceAttempted'])} | {int(row['BounceSuccess'])} | {int(row['AnyFiniteMinS3T'])} | "
            f"{int(row['ThinwallFinite'])} | {int(row['ThinwallCrossedTarget'])} | "
            f"{int(row['Tn_in_window'])}"
        )

    if args.mode in {"constrained", "curated"}:
        thr = minimal_threshold_rows(df, tc_window=(tc_lo, tc_hi), tn_window=(tn_lo, tn_hi))
        for _, row in thr.iterrows():
            if args.thermal_cubic_mode == "portal_proxy":
                print(
                    f"[THRESH] {row['model_id']} first_order: kE={row['first_order_threshold_kE']}, "
                    f"g={row['first_order_threshold_g']}, E={row['first_order_threshold_E']}"
                )
                print(
                    f"[THRESH] {row['model_id']} Tn_window: kE={row['tn_window_threshold_kE']}, "
                    f"g={row['tn_window_threshold_g']}, E={row['tn_window_threshold_E']}"
                )
            else:
                print(f"[THRESH] {row['model_id']} first_order: E={row['first_order_threshold_E']}")
                print(f"[THRESH] {row['model_id']} Tn_window: E={row['tn_window_threshold_E']}")

    print("[FT] Best-case per model (Tc, Tn, alpha_PT, beta/H):")
    for mid in ALL_MODELS:
        gm = best_df[best_df["model_id"] == mid]
        if gm.empty:
            print(f"  {mid}: none")
            continue
        r = gm.sort_values("score").iloc[0]
        print(
            f"  {mid}: Tc={r['Tc']}, Tn={r['Tn']}, alpha_PT={r['alpha_PT']}, beta_over_H={r['beta_over_H']}"
        )

    print(f"[FT] Output folder: {out_dir}")
    if constrained_used_csv is not None:
        print(f"[FT] Key files:\n  {constrained_used_csv}\n  {summary_csv}\n  {plot_png}\n  {report_md}")
    else:
        print(f"[FT] Key files:\n  {summary_csv}\n  {best_json}\n  {plot_png}\n  {report_md}")


if __name__ == "__main__":
    main()
