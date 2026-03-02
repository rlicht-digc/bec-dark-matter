#!/usr/bin/env python3
"""
M_dyn Triangulation: ALFALFA×Yang vs Kinematic M_dyn at Fixed M_bar
====================================================================

Triangulates the key bridge quantity "M_dyn at fixed M_bar" using
independent mass inference without TNG.

Route 1 (primary): ALFALFA×Yang group catalog halo masses vs kinematic
    M_dyn proxy at fixed M_bar.
Route 3 (sanity): Internal SPARC consistency — outermost-point M_dyn
    vs Vflat-based M_dyn (where available).

Compares the inferred offset magnitude to the previously observed
TNG-vs-SPARC offset:
    median_log10(Mdyn_TNG/Mdyn_SPARC) ≈ +0.215 dex  (from Test 6).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.style.use("default")

# ── Physical constants ────────────────────────────────────────────────
G_SI   = 6.674e-11      # m^3 kg^-1 s^-2
MSUN   = 1.989e30        # kg
KPC    = 3.086e19        # m
H_LITTLE = 0.7           # h for Yang halo masses (M_halo/(Msun/h))


# =====================================================================
# JSON helpers
# =====================================================================

def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(obj), indent=2))


# =====================================================================
# STEP 0 — Auto-detect datasets
# =====================================================================

def find_file(root: Path, candidates: List[str], label: str) -> Path:
    """Try each candidate path relative to root, return first hit."""
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Tried:\n" +
        "\n".join(f"  {root / c}" for c in candidates)
    )


def discover_data(root: Path) -> Dict[str, Path]:
    """Auto-detect all required datasets."""
    paths: Dict[str, Path] = {}

    # SPARC per-point data
    paths["rar_points"] = find_file(root, [
        "analysis/results/rar_points_unified.csv",
    ], "SPARC RAR points")

    # SPARC BTFR (Lelli 2019) — has Vflat and logMb
    paths["btfr"] = find_file(root, [
        "raw_data/observational/hi_surveys/sparc/BTFR_Lelli2019.mrt",
    ], "SPARC BTFR table")

    # SPARC halo masses (Li et al.)
    paths["sparc_halo"] = find_file(root, [
        "raw_data/observational/hi_surveys/sparc/HI_linewidths_vs_halo_mass.mrt",
    ], "SPARC halo mass table")

    # ALFALFA α.100
    paths["alfalfa"] = find_file(root, [
        "raw_data/observational/alfalfa/alfalfa_alpha100_haynes2018.tsv",
        "data/alfalfa/alfalfa_alpha100_haynes2018.tsv",
    ], "ALFALFA α.100")

    # Yang catalogs
    yang_dir_cands = [
        "raw_data/observational/yang_catalogs",
        "data/yang_catalogs",
    ]
    yang_dir = None
    for c in yang_dir_cands:
        p = root / c
        if p.is_dir() and (p / "SDSS7").exists():
            yang_dir = p
            break
    if yang_dir is None:
        raise FileNotFoundError("Yang DR7 catalog directory not found")
    paths["yang_dir"] = yang_dir

    return paths


# =====================================================================
# STEP 1 — Load & parse data
# =====================================================================

# ── ALFALFA ───────────────────────────────────────────────────────────

def _parse_sexa_ra(s: str) -> Optional[float]:
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return 15.0 * (h + m / 60.0 + sec / 3600.0)
    except ValueError:
        return None


def _parse_sexa_dec(s: str) -> Optional[float]:
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        sign = -1 if d < 0 or s.strip().startswith("-") else 1
        return sign * (abs(d) + m / 60.0 + sec / 3600.0)
    except ValueError:
        return None


def load_alfalfa(path: Path) -> List[Dict[str, Any]]:
    galaxies: List[Dict[str, Any]] = []
    header_found = False
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.strip().startswith('"') or line.strip().startswith("---"):
                continue
            parts = line.strip().split("\t")
            if not header_found:
                if parts[0].strip() == "AGC":
                    header_found = True
                continue
            if len(parts) < 10 or parts[0].strip().startswith("-"):
                continue
            try:
                agc = parts[0].strip()
                if not agc:
                    continue
                ra = _parse_sexa_ra(parts[2].strip()) if len(parts) > 2 else None
                dec = _parse_sexa_dec(parts[3].strip()) if len(parts) > 3 else None
                if ra is None or dec is None:
                    continue
                vhel = int(parts[6].strip()) if len(parts) > 6 and parts[6].strip() else 0
                w50_s = parts[7].strip() if len(parts) > 7 else ""
                dist_s = parts[14].strip() if len(parts) > 14 else ""
                logmhi_s = parts[16].strip() if len(parts) > 16 else ""
                hi_s = parts[18].strip() if len(parts) > 18 else ""
                if not w50_s or not logmhi_s or not dist_s:
                    continue
                w50 = int(w50_s)
                dist = float(dist_s)
                logmhi = float(logmhi_s)
                hi_code = int(hi_s) if hi_s else 1
                if hi_code != 1:
                    continue
                if w50 < 20 or w50 > 600 or dist < 1.0 or dist > 250 or logmhi < 6.0:
                    continue
                galaxies.append({
                    "agc": agc, "ra": ra, "dec": dec,
                    "vhel": vhel, "w50": w50, "dist": dist,
                    "logmhi": logmhi,
                })
            except (ValueError, IndexError):
                continue
    return galaxies


# ── Yang DR7 ──────────────────────────────────────────────────────────

def load_yang_dr7(yang_dir: Path) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Dict[int, Dict], Dict[int, Dict], Dict[int, int],
]:
    # Galaxy catalog
    gal_file = yang_dir / "SDSS7"
    ra_list, dec_list, z_list, gid_list = [], [], [], []
    with open(gal_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                gid_list.append(int(parts[0]))
                ra_list.append(float(parts[2]))
                dec_list.append(float(parts[3]))
                z_list.append(float(parts[4]))
            except (ValueError, IndexError):
                continue
    yang_ra = np.array(ra_list)
    yang_dec = np.array(dec_list)
    yang_z = np.array(z_list)
    yang_galid = np.array(gid_list)

    # Galaxy-to-group (modelC)
    map_file = yang_dir / "imodelC_1"
    gal_to_grp: Dict[int, Dict] = {}
    with open(map_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                gal_to_grp[int(parts[0])] = {
                    "grp_id": int(parts[2]),
                    "central": int(parts[3]) == 1,
                }
            except (ValueError, IndexError):
                continue

    # Group catalog
    grp_file = yang_dir / "modelC_group"
    groups: Dict[int, Dict] = {}
    with open(grp_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                gid = int(parts[0])
                if gid == 0:
                    continue
                logMh_L = float(parts[6])
                logMh_M = float(parts[7])
                logMh = logMh_L if logMh_L > 0 else logMh_M
                groups[gid] = {"logMh": logMh, "z": float(parts[3])}
            except (ValueError, IndexError):
                continue

    # Richness
    grp_rich: Dict[int, int] = {}
    for info in gal_to_grp.values():
        g = info["grp_id"]
        if g > 0:
            grp_rich[g] = grp_rich.get(g, 0) + 1

    return yang_ra, yang_dec, yang_z, yang_galid, gal_to_grp, groups, grp_rich


def crossmatch_alfalfa_yang(
    alfalfa: List[Dict],
    yang_ra: np.ndarray, yang_dec: np.ndarray,
    yang_z: np.ndarray, yang_galid: np.ndarray,
    gal_to_grp: Dict, groups: Dict,
    max_sep_arcsec: float = 10.0, max_dz: float = 0.005,
) -> List[Dict[str, Any]]:
    c_light = 299792.458
    max_sep_deg = max_sep_arcsec / 3600.0

    # RA index
    ra_step = 0.5
    ra_bins: Dict[int, List[int]] = {}
    for i in range(len(yang_ra)):
        b = int(yang_ra[i] / ra_step)
        ra_bins.setdefault(b, []).append(i)

    matched: List[Dict[str, Any]] = []
    for gal in alfalfa:
        ra, dec = gal["ra"], gal["dec"]
        z_alf = gal["vhel"] / c_light

        b = int(ra / ra_step)
        cands_idx: List[int] = []
        for bb in (b - 1, b, b + 1):
            cands_idx.extend(ra_bins.get(bb, []))
        if not cands_idx:
            continue
        cands = np.array(cands_idx)

        # Dec pre-filter
        dec_ok = np.abs(yang_dec[cands] - dec) < max_sep_deg * 2
        if not np.any(dec_ok):
            continue
        filt = cands[dec_ok]

        # Angular sep
        ra1, dec1 = np.radians(ra), np.radians(dec)
        ra2, dec2 = np.radians(yang_ra[filt]), np.radians(yang_dec[filt])
        cos_sep = (np.sin(dec1) * np.sin(dec2)
                   + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
        cos_sep = np.clip(cos_sep, -1, 1)
        sep = np.degrees(np.arccos(cos_sep)) * 3600

        z_ok = np.abs(yang_z[filt] - z_alf) < max_dz
        combined = (sep < max_sep_arcsec) & z_ok
        if np.any(combined):
            sub = np.where(combined)[0]
            best = filt[sub[np.argmin(sep[sub])]]
        else:
            pos_ok = sep < max_sep_arcsec
            if not np.any(pos_ok):
                continue
            best = filt[np.argmin(sep)]

        gal_id = int(yang_galid[best])
        info = gal_to_grp.get(gal_id, {})
        grp_id = info.get("grp_id", 0)
        logMh = 0.0
        if grp_id > 0 and grp_id in groups:
            logMh = groups[grp_id]["logMh"]

        if logMh <= 0:
            continue  # no usable halo mass

        matched.append({
            **gal,
            "yang_galid": gal_id,
            "logMh_yang_h": logMh,                # log10(M200 / (Msun/h))
            "logMh_yang": logMh + np.log10(H_LITTLE),  # log10(M200 / Msun)
        })
    return matched


# ── SPARC BTFR (Lelli 2019) ──────────────────────────────────────────

def load_btfr(path: Path) -> pd.DataFrame:
    """Parse BTFR_Lelli2019.mrt fixed-width format."""
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.startswith(("T", "A", "=", "-", " ", "N")):
                # header / metadata
                if line[0] in ("T", "A", "=", "-", "N"):
                    continue
                # data line starts with space then galaxy name
            # Try parsing as data
            try:
                name = line[0:12].strip()
                if not name or name.startswith(("T", "A", "=", "-", "B", "N")):
                    continue
                logMb = float(line[12:18].strip())
                e_logMb = float(line[18:24].strip())
                Vf = float(line[36:42].strip())
                e_Vf = float(line[42:48].strip())
                rows.append({
                    "galaxy": name,
                    "logMb_btfr": logMb,
                    "e_logMb": e_logMb,
                    "Vf": Vf,
                    "e_Vf": e_Vf,
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


# ── SPARC halo masses (Li et al.) ────────────────────────────────────

def load_sparc_halo(path: Path) -> pd.DataFrame:
    """Parse HI_linewidths_vs_halo_mass.mrt."""
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            try:
                name = line[0:12].strip()
                if not name or not name[0].isalpha():
                    continue
                logM_NFW = float(line[25:31].strip())
                logM_DC14 = float(line[47:53].strip())
                rows.append({
                    "galaxy": name,
                    "logM200_NFW": logM_NFW,
                    "logM200_DC14": logM_DC14,
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


# ── SPARC per-point M_dyn proxy ──────────────────────────────────────

def compute_sparc_kinematic_mdyn(rar_path: Path) -> pd.DataFrame:
    """Compute outermost-point M_dyn proxy from per-point RAR data.

    M_dyn_outer = g_obs(R_out) * R_out^2 / G
    """
    df = pd.read_csv(rar_path)
    sparc = df[df["source"] == "SPARC"].copy()
    for c in ("galaxy", "log_gbar", "log_gobs", "R_kpc"):
        if c not in sparc.columns:
            raise RuntimeError(f"Missing column {c}")
    sparc = sparc[
        np.isfinite(sparc["log_gbar"])
        & np.isfinite(sparc["log_gobs"])
        & np.isfinite(sparc["R_kpc"])
        & (sparc["R_kpc"] > 0)
    ].copy()

    rows: List[Dict[str, Any]] = []
    for gal, g in sparc.groupby("galaxy", sort=False):
        g2 = g.sort_values("R_kpc")
        n = len(g2)
        if n < 5:
            continue
        r_out = float(g2["R_kpc"].iloc[-1])
        lgb = float(g2["log_gbar"].iloc[-1])
        lgo = float(g2["log_gobs"].iloc[-1])
        M_bar = (10.0 ** lgb) * (r_out * KPC) ** 2 / G_SI / MSUN
        M_dyn = (10.0 ** lgo) * (r_out * KPC) ** 2 / G_SI / MSUN
        if not np.isfinite(M_bar) or M_bar <= 0:
            continue
        if not np.isfinite(M_dyn) or M_dyn <= 0:
            continue
        rows.append({
            "galaxy": gal,
            "n_points": n,
            "R_out_kpc": r_out,
            "logMbar_outer": float(np.log10(M_bar)),
            "logMdyn_outer": float(np.log10(M_dyn)),
        })
    return pd.DataFrame(rows)


# =====================================================================
# STEP 2 — Route 1: ALFALFA×Yang vs kinematic M_dyn at fixed M_bar
# =====================================================================

def route1_binned_comparison(
    yang_df: pd.DataFrame,
    sparc_kin: pd.DataFrame,
    bin_width: float,
    min_bin_n: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bin by logMbar and compare median logMh_yang vs median logMdyn_kin."""

    # Determine overlap range
    yang_min, yang_max = yang_df["logMbar"].min(), yang_df["logMbar"].max()
    kin_min, kin_max = sparc_kin["logMbar_outer"].min(), sparc_kin["logMbar_outer"].max()
    lo = max(yang_min, kin_min)
    hi = min(yang_max, kin_max)

    edges = np.arange(np.floor(lo / bin_width) * bin_width,
                      hi + bin_width, bin_width)

    rows: List[Dict[str, Any]] = []
    yang_vals = yang_df["logMbar"].to_numpy()
    yang_mh = yang_df["logMh_yang"].to_numpy()
    kin_vals = sparc_kin["logMbar_outer"].to_numpy()
    kin_md = sparc_kin["logMdyn_outer"].to_numpy()

    for i in range(len(edges) - 1):
        lo_e, hi_e = edges[i], edges[i + 1]
        center = 0.5 * (lo_e + hi_e)

        y_mask = (yang_vals >= lo_e) & (yang_vals < hi_e)
        k_mask = (kin_vals >= lo_e) & (kin_vals < hi_e)

        ny = int(y_mask.sum())
        nk = int(k_mask.sum())
        if ny < min_bin_n or nk < 3:
            continue

        med_yang = float(np.median(yang_mh[y_mask]))
        med_kin = float(np.median(kin_md[k_mask]))
        delta = med_yang - med_kin

        # Bootstrap CI for delta
        boots: List[float] = []
        for _ in range(n_boot):
            by = np.median(rng.choice(yang_mh[y_mask], size=ny, replace=True))
            bk = np.median(rng.choice(kin_md[k_mask], size=nk, replace=True))
            boots.append(by - bk)
        boots_arr = np.array(boots)
        ci_lo = float(np.percentile(boots_arr, 2.5))
        ci_hi = float(np.percentile(boots_arr, 97.5))

        rows.append({
            "logMbar_center": center,
            "logMbar_lo": lo_e,
            "logMbar_hi": hi_e,
            "n_yang": ny,
            "n_kin": nk,
            "median_logMh_yang": med_yang,
            "median_logMdyn_kin": med_kin,
            "delta_yang_minus_kin": delta,
            "delta_ci95_lo": ci_lo,
            "delta_ci95_hi": ci_hi,
        })

    bin_df = pd.DataFrame(rows)

    # Global delta: pool all galaxies in overlap range
    y_in = (yang_vals >= lo) & (yang_vals <= hi)
    k_in = (kin_vals >= lo) & (kin_vals <= hi)
    if y_in.sum() > 0 and k_in.sum() > 0:
        global_delta = float(np.median(yang_mh[y_in]) - np.median(kin_md[k_in]))
        boots_g: List[float] = []
        ny_g, nk_g = int(y_in.sum()), int(k_in.sum())
        for _ in range(n_boot):
            by = np.median(rng.choice(yang_mh[y_in], size=ny_g, replace=True))
            bk = np.median(rng.choice(kin_md[k_in], size=nk_g, replace=True))
            boots_g.append(by - bk)
        boots_g_arr = np.array(boots_g)
        global_ci = [float(np.percentile(boots_g_arr, 2.5)),
                     float(np.percentile(boots_g_arr, 97.5))]
    else:
        global_delta = None
        global_ci = [None, None]

    stats = {
        "overlap_logMbar_range": [float(lo), float(hi)],
        "n_bins": len(bin_df),
        "n_yang_total": int(y_in.sum()),
        "n_kin_total": int(k_in.sum()),
        "delta_global": global_delta,
        "delta_global_ci95": global_ci,
    }
    return bin_df, stats


# =====================================================================
# STEP 3 — Route 3: SPARC internal estimator consistency
# =====================================================================

def route3_sparc_internal(
    sparc_kin: pd.DataFrame,
    btfr: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare outermost-point M_dyn vs Vflat-based M_dyn."""

    # Merge on galaxy name
    merged = sparc_kin.merge(btfr, on="galaxy", how="inner")
    # Only galaxies where Vflat is measured (Vf > 0)
    merged = merged[merged["Vf"] > 0].copy()

    if len(merged) == 0:
        return pd.DataFrame(), {"status": "BLOCKED", "reason": "No Vflat matches"}

    # Vflat-based M_dyn at same R_out: M_flat = Vf^2 * R_out / G
    Vf_ms = merged["Vf"].to_numpy() * 1e3  # km/s → m/s
    R_m = merged["R_out_kpc"].to_numpy() * KPC  # kpc → m
    M_flat = Vf_ms ** 2 * R_m / G_SI / MSUN
    merged["logMdyn_flat"] = np.log10(np.maximum(M_flat, 1e-10))
    merged["log_ratio_flat_over_outer"] = (
        merged["logMdyn_flat"] - merged["logMdyn_outer"]
    )

    ratios = merged["log_ratio_flat_over_outer"].to_numpy()
    ratios = ratios[np.isfinite(ratios)]

    median_r = float(np.median(ratios))
    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios, ddof=1))

    # Bootstrap CI
    boots = np.array([
        float(np.median(rng.choice(ratios, size=len(ratios), replace=True)))
        for _ in range(n_boot)
    ])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    stats = {
        "n_galaxies": len(merged),
        "median_log_ratio": median_r,
        "mean_log_ratio": mean_r,
        "std_log_ratio": std_r,
        "ci95_median": ci,
    }
    return merged, stats


# =====================================================================
# STEP 4 — Figures
# =====================================================================

def make_figures(
    bin_df: pd.DataFrame,
    route3_df: pd.DataFrame,
    ref_delta: float,
    out_dir: Path,
) -> None:

    # ── Fig 1: Yang vs Kinematic vs logMbar ───────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=180)
    if len(bin_df) > 0:
        x = bin_df["logMbar_center"].to_numpy()
        y_yang = bin_df["median_logMh_yang"].to_numpy()
        y_kin = bin_df["median_logMdyn_kin"].to_numpy()
        ax1.errorbar(x, y_yang, fmt="s-", color="#d62728", label="Yang halo mass",
                     markersize=6, capsize=3)
        ax1.errorbar(x, y_kin, fmt="o-", color="#1f77b4", label="SPARC kinematic M_dyn",
                     markersize=6, capsize=3)
        # 1:1 line
        all_vals = np.concatenate([y_yang, y_kin])
        lims = [all_vals.min() - 0.3, all_vals.max() + 0.3]
        ax1.plot(x, x, "k:", alpha=0.3, label="M_dyn = M_bar (1:1)")
    ax1.set_xlabel("log$_{10}$(M$_{\\rm bar}$ / M$_\\odot$)  [bin center]")
    ax1.set_ylabel("log$_{10}$(M / M$_\\odot$)")
    ax1.set_title("Yang M$_{200}$ vs SPARC kinematic M$_{\\rm dyn}$ at fixed M$_{\\rm bar}$")
    ax1.legend(fontsize=9)
    ax1.set_facecolor("white")
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig_yang_vs_kinematic.png", facecolor="white")
    plt.close(fig1)

    # ── Fig 2: Offset vs mass ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=180)
    if len(bin_df) > 0:
        x = bin_df["logMbar_center"].to_numpy()
        delta = bin_df["delta_yang_minus_kin"].to_numpy()
        ci_lo = bin_df["delta_ci95_lo"].to_numpy()
        ci_hi = bin_df["delta_ci95_hi"].to_numpy()
        yerr_lo = delta - ci_lo
        yerr_hi = ci_hi - delta
        ax2.errorbar(x, delta, yerr=[yerr_lo, yerr_hi],
                     fmt="o-", color="#2ca02c", capsize=4, markersize=6,
                     label="Yang − kinematic")
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax2.axhline(ref_delta, color="red", linestyle="--", linewidth=1.5,
                    label=f"TNG−SPARC offset = +{ref_delta:.3f} dex")
        ax2.set_xlabel("log$_{10}$(M$_{\\rm bar}$ / M$_\\odot$)  [bin center]")
        ax2.set_ylabel("$\\Delta$ log$_{10}$(M$_{\\rm dyn}$)  [Yang − kinematic]")
        ax2.set_title("Dynamical mass offset vs baryon mass")
        ax2.legend(fontsize=9)
    ax2.set_facecolor("white")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_offset_vs_mass.png", facecolor="white")
    plt.close(fig2)

    # ── Fig 3: SPARC internal estimators ──────────────────────────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    if len(route3_df) > 0 and "log_ratio_flat_over_outer" in route3_df.columns:
        ratios = route3_df["log_ratio_flat_over_outer"].dropna().to_numpy()
        logMb = route3_df["logMbar_outer"].to_numpy()

        # Histogram
        ax3a.hist(ratios, bins=25, color="#9ecae1", edgecolor="black", alpha=0.8)
        med = np.median(ratios)
        ax3a.axvline(med, color="red", linewidth=1.5,
                     label=f"median = {med:.3f}")
        ax3a.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax3a.set_xlabel("log$_{10}$(M$_{\\rm dyn,Vflat}$ / M$_{\\rm dyn,outer}$)")
        ax3a.set_ylabel("count")
        ax3a.set_title("SPARC internal estimator consistency")
        ax3a.legend(fontsize=9)

        # Scatter vs mass
        ok = np.isfinite(ratios) & np.isfinite(logMb)
        ax3b.scatter(logMb[ok], ratios[ok], s=15, alpha=0.6,
                     color="#1f77b4", edgecolors="black", linewidths=0.3)
        ax3b.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax3b.axhline(med, color="red", linestyle=":", linewidth=1,
                     label=f"median = {med:.3f}")
        ax3b.set_xlabel("log$_{10}$(M$_{\\rm bar}$ / M$_\\odot$)")
        ax3b.set_ylabel("log$_{10}$(M$_{\\rm dyn,Vflat}$ / M$_{\\rm dyn,outer}$)")
        ax3b.set_title("Estimator ratio vs baryonic mass")
        ax3b.legend(fontsize=9)
    for ax in (ax3a, ax3b):
        ax.set_facecolor("white")
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_sparc_mdyn_estimators.png", facecolor="white")
    plt.close(fig3)


# =====================================================================
# STEP 5 — Summary + report
# =====================================================================

def interpret_delta(global_delta: Optional[float], ref: float) -> str:
    if global_delta is None:
        return "inconclusive (no valid overlap bins)"

    # Yang M200 vs kinematic M_dyn have a fundamental aperture mismatch
    # (virial-scale vs rotation-curve extent). A large positive delta is
    # expected and does not directly diagnose the TNG-vs-SPARC offset.
    if global_delta > ref * 3:
        return (
            f"delta_global ({global_delta:+.3f} dex) is much larger than "
            f"the TNG-SPARC reference offset (+{ref:.3f} dex). This is "
            f"expected: Yang M_200 includes the full dark matter halo out "
            f"to the virial radius (~200 kpc), while SPARC kinematic M_dyn "
            f"is measured only to the last rotation-curve point (~20-50 kpc). "
            f"The aperture gap dominates. The TNG-SPARC offset (+{ref:.3f} dex) "
            f"is a small fraction ({ref / global_delta:.1%}) of the total "
            f"Yang-kinematic gap, meaning the TNG M_dyn excess at fixed M_bar "
            f"is NOT simply 'catching up to the full halo mass' but rather "
            f"reflects a more concentrated inner mass profile. Route 3 (SPARC "
            f"internal consistency: Vflat vs outer-point) provides a cleaner "
            f"diagnostic of estimator bias."
        )

    abs_to_zero = abs(global_delta)
    abs_to_ref = abs(global_delta - ref)
    if abs_to_zero < abs_to_ref:
        return (
            f"delta_global ({global_delta:+.3f}) is closer to 0 than to "
            f"+{ref:.3f}. Yang halo masses align more closely with SPARC "
            f"kinematic M_dyn than with TNG. This suggests TNG halos are "
            f"more concentrated/massive than observed at fixed M_bar."
        )
    elif abs_to_ref < abs_to_zero:
        return (
            f"delta_global ({global_delta:+.3f}) is closer to +{ref:.3f} "
            f"than to 0. Yang halo masses are elevated relative to SPARC "
            f"kinematic M_dyn by a similar amount as TNG, suggesting the "
            f"SPARC aperture-limited kinematic proxy underestimates halo mass."
        )
    else:
        return (
            f"delta_global ({global_delta:+.3f}) is equidistant from 0 and "
            f"+{ref:.3f}; mass-dependent stratified interpretation needed."
        )


def write_report(
    out_dir: Path,
    paths_used: Dict[str, str],
    route1_stats: Dict[str, Any],
    route3_stats: Dict[str, Any],
    ref_delta: float,
    interpretation: str,
) -> None:
    lines = [
        "# M_dyn Triangulation: ALFALFA×Yang vs SPARC Kinematic",
        "",
        "## Aperture Mismatch Caveat",
        "Yang group halo masses are M_200 estimates (virial-scale, ~200× "
        "critical density) derived from abundance matching. SPARC kinematic "
        "M_dyn is computed from the outermost rotation-curve point: "
        "M_dyn = g_obs × R_out² / G. These are fundamentally different "
        "apertures. The Yang masses trace the full halo while SPARC "
        "kinematic masses are aperture-limited to the extent of the "
        "observed rotation curve (typically 10–50 kpc). The offset between "
        "them is expected to be large and positive, reflecting the mass "
        "beyond the last measured point.",
        "",
        "## What This Comparison Allows",
        "We cannot conclude absolute mass values are correct. We CAN compare "
        "the *offset pattern* (Yang − kinematic) against the TNG−SPARC "
        "offset (+0.215 dex from Test 6). If the Yang−SPARC offset at "
        "fixed M_bar matches the TNG−SPARC offset, this suggests SPARC's "
        "aperture-limited kinematic proxy systematically underestimates "
        "the total dynamical mass. If instead the Yang−SPARC offset is "
        "much larger than the TNG−SPARC offset, then TNG's excess is "
        "only a fraction of the full halo mass and may reflect genuine "
        "over-concentration of TNG halos.",
        "",
        "## Route 1: ALFALFA×Yang vs Kinematic at Fixed M_bar",
    ]

    d = route1_stats.get("delta_global")
    ci = route1_stats.get("delta_global_ci95", [None, None])
    lines += [
        f"- N_yang in overlap: {route1_stats.get('n_yang_total', 'N/A')}",
        f"- N_kin in overlap: {route1_stats.get('n_kin_total', 'N/A')}",
        f"- Overlap range: {route1_stats.get('overlap_logMbar_range', 'N/A')}",
        f"- delta_global (Yang − kinematic): {d:+.3f}" if d is not None else "- delta_global: N/A",
        f"- 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "- 95% CI: N/A",
        f"- TNG−SPARC reference offset: +{ref_delta:.3f} dex",
        "",
    ]

    lines += [
        "## Route 3: SPARC Internal Estimator Consistency",
        f"- N galaxies with Vflat: {route3_stats.get('n_galaxies', 'N/A')}",
        f"- Median log10(Mdyn_Vflat / Mdyn_outer): "
        f"{route3_stats.get('median_log_ratio', 'N/A'):.4f}" if route3_stats.get("median_log_ratio") is not None else "",
        f"- Std: {route3_stats.get('std_log_ratio', 'N/A'):.4f}" if route3_stats.get("std_log_ratio") is not None else "",
        f"- 95% CI on median: {route3_stats.get('ci95_median', 'N/A')}",
        "",
        "If median ~ 0 and scatter is small, the outer-point M_dyn estimator "
        "is internally consistent and the TNG−SPARC offset is not an "
        "estimator artifact.",
        "",
        "## Interpretation",
        interpretation,
        "",
        "## Data Sources",
    ]
    for k, v in paths_used.items():
        lines.append(f"- {k}: `{v}`")

    (out_dir / "report_mdyn_triangulation.md").write_text("\n".join(lines) + "\n")


# =====================================================================
# Main
# =====================================================================

def run(
    root: Path,
    out_dir: Path,
    seed: int,
    bin_width: float,
    min_bin_n: int,
    n_boot: int,
    ref_delta: float,
) -> Dict[str, Any]:
    print("=" * 72)
    print("M_dyn TRIANGULATION: ALFALFA×Yang vs SPARC Kinematic")
    print("=" * 72)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # ── Discover data ─────────────────────────────────────────────────
    print("\n[1/5] Discovering datasets...")
    dp = discover_data(root)
    for k, v in dp.items():
        print(f"  {k}: {v}")

    # ── Load SPARC kinematic M_dyn ────────────────────────────────────
    print("\n[2/5] Loading SPARC kinematic M_dyn...")
    sparc_kin = compute_sparc_kinematic_mdyn(dp["rar_points"])
    print(f"  {len(sparc_kin)} SPARC galaxies with kinematic M_dyn")

    # ── Load BTFR for Vflat ───────────────────────────────────────────
    btfr = load_btfr(dp["btfr"])
    print(f"  {len(btfr)} SPARC galaxies in BTFR table, "
          f"{(btfr['Vf'] > 0).sum()} with measured Vflat")

    # ── Load & crossmatch ALFALFA × Yang ──────────────────────────────
    print("\n[3/5] Loading ALFALFA×Yang crossmatch...")
    alfalfa = load_alfalfa(dp["alfalfa"])
    print(f"  {len(alfalfa)} ALFALFA galaxies after quality cuts")

    yang_ra, yang_dec, yang_z, yang_galid, g2g, groups, rich = \
        load_yang_dr7(dp["yang_dir"])
    print(f"  {len(yang_galid)} Yang galaxies, {len(groups)} groups")

    matched = crossmatch_alfalfa_yang(
        alfalfa, yang_ra, yang_dec, yang_z, yang_galid, g2g, groups)
    print(f"  {len(matched)} ALFALFA×Yang matches with halo mass")

    # Build Yang dataframe with logMbar proxy = logMHI + 0.3
    # (M_bar ≈ 1.4 × M_HI for HI-dominated galaxies)
    yang_df = pd.DataFrame(matched)
    yang_df["logMbar"] = yang_df["logmhi"] + np.log10(1.4)

    # ── Route 1: binned comparison ────────────────────────────────────
    print("\n[4/5] Route 1: Binned comparison...")
    bin_df, r1_stats = route1_binned_comparison(
        yang_df, sparc_kin, bin_width, min_bin_n, n_boot, rng)
    print(f"  {r1_stats['n_bins']} bins, "
          f"N_yang={r1_stats['n_yang_total']}, "
          f"N_kin={r1_stats['n_kin_total']}")
    if r1_stats["delta_global"] is not None:
        print(f"  delta_global = {r1_stats['delta_global']:+.3f} "
              f"(CI95: [{r1_stats['delta_global_ci95'][0]:.3f}, "
              f"{r1_stats['delta_global_ci95'][1]:.3f}])")
    bin_df.to_csv(out_dir / "alfalfa_yang_massbins.csv", index=False)

    # ── Route 3: SPARC internal ───────────────────────────────────────
    print("\n[5/5] Route 3: SPARC internal consistency...")
    r3_df, r3_stats = route3_sparc_internal(sparc_kin, btfr, n_boot, rng)
    if r3_stats.get("n_galaxies", 0) > 0:
        print(f"  {r3_stats['n_galaxies']} galaxies with Vflat")
        print(f"  Median log10(Mdyn_flat/Mdyn_outer) = "
              f"{r3_stats['median_log_ratio']:.4f} "
              f"(std={r3_stats['std_log_ratio']:.4f})")
        r3_df.to_csv(out_dir / "sparc_internal_mdyn_check.csv", index=False)
    else:
        print("  BLOCKED: no Vflat matches")
        pd.DataFrame().to_csv(out_dir / "sparc_internal_mdyn_check.csv", index=False)

    # ── Interpretation ────────────────────────────────────────────────
    interp = interpret_delta(r1_stats.get("delta_global"), ref_delta)

    # ── Figures ───────────────────────────────────────────────────────
    make_figures(bin_df, r3_df, ref_delta, out_dir)

    # ── Summary JSON ──────────────────────────────────────────────────
    dt = time.time() - t0
    paths_used = {k: str(v) for k, v in dp.items()}

    summary: Dict[str, Any] = {
        "test": "mdyn_triangulation_alfalfa_yang",
        "status": "OK",
        "seed": seed,
        "bin_width_dex": bin_width,
        "min_bin_n": min_bin_n,
        "n_boot": n_boot,
        "reference_delta_tng_sparc": ref_delta,
        "route1": {
            **r1_stats,
            "interpretation": interp,
        },
        "route3": r3_stats,
        "paths_used": paths_used,
        "elapsed_seconds": round(dt, 1),
    }
    write_json(out_dir / "summary_mdyn_triangulation.json", summary)

    # ── Report ────────────────────────────────────────────────────────
    write_report(out_dir, paths_used, r1_stats, r3_stats, ref_delta, interp)

    # ── Run metadata ──────────────────────────────────────────────────
    meta = {
        "script": "test_mdyn_triangulation_alfalfa_yang.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": round(dt, 1),
    }
    write_json(out_dir / "run_metadata.json", meta)

    # ── Final output ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  N_yang (overlap): {r1_stats.get('n_yang_total', 'N/A')}")
    print(f"  N_sparc_kin (overlap): {r1_stats.get('n_kin_total', 'N/A')}")
    d = r1_stats.get("delta_global")
    ci = r1_stats.get("delta_global_ci95", [None, None])
    if d is not None:
        print(f"  delta_global: {d:+.3f}  CI95: [{ci[0]:.3f}, {ci[1]:.3f}]")
        if d > ref_delta * 3:
            reason = (
                f"aperture mismatch dominates: Yang M200 >> kinematic M_dyn "
                f"({d:+.3f} >> +{ref_delta:.3f}). TNG-SPARC offset is only "
                f"{ref_delta / d:.1%} of the full halo gap — TNG excess "
                f"reflects inner concentration, not total halo mass"
            )
        else:
            abs_to_zero = abs(d)
            abs_to_ref = abs(d - ref_delta)
            if abs_to_zero < abs_to_ref:
                reason = (f"closer to 0 ({abs_to_zero:.3f}) than to "
                          f"+{ref_delta:.3f} ({abs_to_ref:.3f}): "
                          f"TNG appears over-massive at fixed M_bar")
            else:
                reason = (f"closer to +{ref_delta:.3f} ({abs_to_ref:.3f}) "
                          f"than to 0 ({abs_to_zero:.3f}): SPARC kinematic "
                          f"proxy underestimates halo mass")
        print(f"  → {reason}")
    else:
        print("  delta_global: N/A (no overlap)")
    print(f"  Output folder: {out_dir}")
    print(f"  Elapsed: {dt:.1f}s")

    return summary


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="M_dyn Triangulation: ALFALFA×Yang vs SPARC Kinematic")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-width-dex", type=float, default=0.3)
    parser.add_argument("--min-bin-n", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--reference-delta-tng-sparc", type=float, default=0.215)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    root = Path(args.project_root).resolve() if args.project_root else default_root

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "outputs" / "mdyn_triangulation" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        root=root,
        out_dir=out_dir,
        seed=args.seed,
        bin_width=args.bin_width_dex,
        min_bin_n=args.min_bin_n,
        n_boot=args.n_boot,
        ref_delta=args.reference_delta_tng_sparc,
    )


if __name__ == "__main__":
    main()
