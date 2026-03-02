#!/usr/bin/env python3
"""Matching utilities for SPARC-matched TNG analog construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchConfig:
    caliper_gbar_med: float = 0.35
    caliper_gbar_iqr: float = 0.30
    caliper_r_med: float = 6.0
    min_pair_points: int = 5


def summarize_galaxies(points: pd.DataFrame) -> pd.DataFrame:
    """Compute per-galaxy summary features used for SPARC↔TNG matching."""
    grouped = points.groupby("galaxy_key", sort=True)
    rows = []
    for key, g in grouped:
        log_gbar = g["log_gbar"].to_numpy(dtype=float)
        log_gobs = g["log_gobs"].to_numpy(dtype=float)
        r_kpc = g["R_kpc"].to_numpy(dtype=float) if "R_kpc" in g.columns else np.full(len(g), np.nan, dtype=float)
        rows.append(
            {
                "galaxy_key": str(key),
                "galaxy_id": str(g["galaxy_id"].iloc[0]),
                "dataset": str(g["dataset"].iloc[0]),
                "n_points": int(len(g)),
                "log_gbar_med": float(np.nanmedian(log_gbar)),
                "log_gbar_iqr": float(np.nanpercentile(log_gbar, 75.0) - np.nanpercentile(log_gbar, 25.0)),
                "log_gobs_med": float(np.nanmedian(log_gobs)),
                "R_med": float(np.nanmedian(r_kpc)),
            }
        )
    return pd.DataFrame(rows).sort_values("galaxy_key").reset_index(drop=True)


def _robust_scale(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 1.0
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    scale = float(q75 - q25)
    return scale if scale > 1e-8 else 1.0


def nearest_neighbor_match(
    sparc_summary: pd.DataFrame,
    tng_summary: pd.DataFrame,
    config: MatchConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbor SPARC→TNG matching with explicit calipers and no replacement."""
    tng_available = tng_summary.copy().reset_index(drop=True)
    used_tng: set[str] = set()
    rows: List[Dict[str, object]] = []
    unmatched_rows: List[Dict[str, object]] = []

    scale_gbar_med = _robust_scale(sparc_summary["log_gbar_med"].to_numpy(dtype=float))
    scale_gbar_iqr = _robust_scale(sparc_summary["log_gbar_iqr"].to_numpy(dtype=float))
    scale_gobs_med = _robust_scale(sparc_summary["log_gobs_med"].to_numpy(dtype=float))
    scale_r_med = _robust_scale(sparc_summary["R_med"].to_numpy(dtype=float))

    for _, srow in sparc_summary.sort_values("galaxy_key").iterrows():
        s_key = str(srow["galaxy_key"])
        candidates = tng_available.loc[~tng_available["galaxy_key"].isin(used_tng)].copy()
        if candidates.empty:
            unmatched_rows.append({"galaxy_key_sparc": s_key, "reason": "no_tng_candidates_remaining"})
            continue

        dgbar = np.abs(candidates["log_gbar_med"].to_numpy(dtype=float) - float(srow["log_gbar_med"]))
        diqr = np.abs(candidates["log_gbar_iqr"].to_numpy(dtype=float) - float(srow["log_gbar_iqr"]))
        dr = np.abs(candidates["R_med"].to_numpy(dtype=float) - float(srow["R_med"]))
        dgobs = np.abs(candidates["log_gobs_med"].to_numpy(dtype=float) - float(srow["log_gobs_med"]))

        pass_caliper = (dgbar <= config.caliper_gbar_med) & (diqr <= config.caliper_gbar_iqr)
        finite_r = np.isfinite(dr) & np.isfinite(float(srow["R_med"]))
        if np.any(finite_r):
            pass_caliper = pass_caliper & np.where(finite_r, dr <= config.caliper_r_med, True)

        if not np.any(pass_caliper):
            unmatched_rows.append({"galaxy_key_sparc": s_key, "reason": "no_candidates_within_calipers"})
            continue

        cands = candidates.loc[pass_caliper].copy()
        dgbar_c = np.abs(cands["log_gbar_med"].to_numpy(dtype=float) - float(srow["log_gbar_med"]))
        diqr_c = np.abs(cands["log_gbar_iqr"].to_numpy(dtype=float) - float(srow["log_gbar_iqr"]))
        dr_c = np.abs(cands["R_med"].to_numpy(dtype=float) - float(srow["R_med"]))
        dgobs_c = np.abs(cands["log_gobs_med"].to_numpy(dtype=float) - float(srow["log_gobs_med"]))
        dist = np.sqrt(
            (dgbar_c / scale_gbar_med) ** 2
            + (diqr_c / scale_gbar_iqr) ** 2
            + (dgobs_c / scale_gobs_med) ** 2
            + np.where(np.isfinite(dr_c), (dr_c / scale_r_med) ** 2, 0.0)
        )

        best_idx = int(np.argmin(dist))
        best = cands.iloc[best_idx]
        t_key = str(best["galaxy_key"])
        used_tng.add(t_key)

        rows.append(
            {
                "galaxy_key_sparc": s_key,
                "galaxy_id_sparc": str(srow["galaxy_id"]),
                "galaxy_key_tng": t_key,
                "galaxy_id_tng": str(best["galaxy_id"]),
                "distance": float(dist[best_idx]),
                "delta_log_gbar_med": float(np.abs(float(srow["log_gbar_med"]) - float(best["log_gbar_med"]))),
                "delta_log_gbar_iqr": float(np.abs(float(srow["log_gbar_iqr"]) - float(best["log_gbar_iqr"]))),
                "delta_log_gobs_med": float(np.abs(float(srow["log_gobs_med"]) - float(best["log_gobs_med"]))),
                "delta_R_med": float(np.abs(float(srow["R_med"]) - float(best["R_med"]))),
                "n_points_sparc": int(srow["n_points"]),
                "n_points_tng": int(best["n_points"]),
            }
        )

    return (
        pd.DataFrame(rows).sort_values(["galaxy_key_sparc"]).reset_index(drop=True),
        pd.DataFrame(unmatched_rows).reset_index(drop=True),
    )


def build_matched_points(
    sparc_points: pd.DataFrame,
    tng_points: pd.DataFrame,
    match_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter point tables to matched galaxy sets only."""
    s_keys = set(match_table["galaxy_key_sparc"].astype(str).tolist())
    t_keys = set(match_table["galaxy_key_tng"].astype(str).tolist())
    s = sparc_points.loc[sparc_points["galaxy_key"].astype(str).isin(s_keys)].copy()
    t = tng_points.loc[tng_points["galaxy_key"].astype(str).isin(t_keys)].copy()
    return s.reset_index(drop=True), t.reset_index(drop=True)


def sample_k_points_per_galaxy(points: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Sample exactly K points per galaxy (drop galaxies with <K points)."""
    rng = np.random.default_rng(seed)
    frames = []
    for gkey, grp in points.groupby("galaxy_key", sort=True):
        if len(grp) < k:
            continue
        idx = rng.choice(grp.index.to_numpy(), size=k, replace=False)
        sample = grp.loc[np.sort(idx)].copy()
        sample["k_used"] = int(k)
        frames.append(sample)
    if not frames:
        return points.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def sample_pairwise_k_points(
    sparc_points: pd.DataFrame,
    tng_points: pd.DataFrame,
    match_table: pd.DataFrame,
    k: int,
    seed: int,
    min_pair_points: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise K sampling per matched pair, using K_eff=min(K, n_s, n_t)."""
    rng = np.random.default_rng(seed)
    s_frames: List[pd.DataFrame] = []
    t_frames: List[pd.DataFrame] = []
    pair_rows: List[Dict[str, object]] = []

    sparc_grouped = {k_: g.copy() for k_, g in sparc_points.groupby("galaxy_key", sort=False)}
    tng_grouped = {k_: g.copy() for k_, g in tng_points.groupby("galaxy_key", sort=False)}

    for pair_id, row in match_table.sort_values("galaxy_key_sparc").reset_index(drop=True).iterrows():
        s_key = str(row["galaxy_key_sparc"])
        t_key = str(row["galaxy_key_tng"])
        s_grp = sparc_grouped.get(s_key)
        t_grp = tng_grouped.get(t_key)
        if s_grp is None or t_grp is None:
            continue

        k_eff = int(min(k, len(s_grp), len(t_grp)))
        if k_eff < int(min_pair_points):
            continue

        s_idx = rng.choice(s_grp.index.to_numpy(), size=k_eff, replace=False)
        t_idx = rng.choice(t_grp.index.to_numpy(), size=k_eff, replace=False)
        s_take = s_grp.loc[np.sort(s_idx)].copy()
        t_take = t_grp.loc[np.sort(t_idx)].copy()
        s_take["pair_id"] = int(pair_id)
        t_take["pair_id"] = int(pair_id)
        s_take["paired_with"] = t_key
        t_take["paired_with"] = s_key
        s_take["k_used"] = int(k_eff)
        t_take["k_used"] = int(k_eff)

        s_frames.append(s_take)
        t_frames.append(t_take)
        pair_rows.append(
            {
                "pair_id": int(pair_id),
                "galaxy_key_sparc": s_key,
                "galaxy_key_tng": t_key,
                "k_used": int(k_eff),
                "n_points_sparc_raw": int(len(s_grp)),
                "n_points_tng_raw": int(len(t_grp)),
                "distance": float(row["distance"]),
            }
        )

    if not s_frames:
        empty = sparc_points.iloc[0:0].copy()
        return empty, empty.copy(), pd.DataFrame(pair_rows)

    s_out = pd.concat(s_frames, ignore_index=True)
    t_out = pd.concat(t_frames, ignore_index=True)
    pair_df = pd.DataFrame(pair_rows).sort_values("pair_id").reset_index(drop=True)
    return s_out, t_out, pair_df


def balance_table(
    sparc_summary: pd.DataFrame,
    tng_summary: pd.DataFrame,
    match_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compute basic pre/post matching balance diagnostics."""
    metrics = ["log_gbar_med", "log_gbar_iqr", "log_gobs_med", "R_med", "n_points"]
    out_rows: List[Dict[str, object]] = []
    if match_table.empty:
        return pd.DataFrame(out_rows)

    matched_s_keys = set(match_table["galaxy_key_sparc"].astype(str))
    matched_t_keys = set(match_table["galaxy_key_tng"].astype(str))
    s_match = sparc_summary[sparc_summary["galaxy_key"].astype(str).isin(matched_s_keys)]
    t_match = tng_summary[tng_summary["galaxy_key"].astype(str).isin(matched_t_keys)]

    for metric in metrics:
        s_all = sparc_summary[metric].to_numpy(dtype=float)
        t_all = tng_summary[metric].to_numpy(dtype=float)
        s_m = s_match[metric].to_numpy(dtype=float)
        t_m = t_match[metric].to_numpy(dtype=float)
        pooled_all = np.sqrt(0.5 * (np.nanvar(s_all) + np.nanvar(t_all)))
        pooled_m = np.sqrt(0.5 * (np.nanvar(s_m) + np.nanvar(t_m)))
        out_rows.append(
            {
                "metric": metric,
                "sparc_mean_all": float(np.nanmean(s_all)),
                "tng_mean_all": float(np.nanmean(t_all)),
                "smd_all": float((np.nanmean(s_all) - np.nanmean(t_all)) / pooled_all) if pooled_all > 1e-12 else 0.0,
                "sparc_mean_matched": float(np.nanmean(s_m)),
                "tng_mean_matched": float(np.nanmean(t_m)),
                "smd_matched": float((np.nanmean(s_m) - np.nanmean(t_m)) / pooled_m) if pooled_m > 1e-12 else 0.0,
            }
        )
    return pd.DataFrame(out_rows)

