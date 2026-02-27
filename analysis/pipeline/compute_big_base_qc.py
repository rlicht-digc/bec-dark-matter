#!/usr/bin/env python3
"""
Compute quality-cut statistics for the big_base dataset.
Resumable: checks for existing output and skips if present.
Memory-safe: processes in batches.

Usage: python3 compute_big_base_qc.py
Output: /home/tnguser/rar_profiles/20260223_061026_big_base/meta/galaxy_quality_counts.csv
"""
import os, sys, time, json
import pandas as pd
import numpy as np
from pathlib import Path

HOME = Path("/home/tnguser")
RUN_DIR = HOME / "rar_profiles" / "20260223_061026_big_base"
PARQUET = RUN_DIR / "rar_points.parquet"
OUT_QC = RUN_DIR / "meta" / "galaxy_quality_counts.csv"

N_BINS = 10
NCUT = 3

print(f"[CONFIG] PARQUET={PARQUET}")
print(f"[CONFIG] OUT_QC={OUT_QC}")
print(f"[CONFIG] N_BINS={N_BINS}, NCUT={NCUT}")
print()

# Resume check
if OUT_QC.exists():
    existing = pd.read_csv(OUT_QC)
    print(f"[RESUME] Output already exists: {len(existing)} rows. Delete to recompute.")
    sys.exit(0)

# Load parquet
t0 = time.time()
print("[STEP 1] Loading parquet...")
df = pd.read_parquet(PARQUET)
print(f"  Loaded: {len(df)} rows, {df.SubhaloID.nunique()} galaxies in {time.time()-t0:.1f}s")

# Compute gbar bins
print("[STEP 2] Computing gbar bins...")
finite_mask = np.isfinite(df.log_gbar) & np.isfinite(df.log_gobs)
df_finite = df[finite_mask].copy()
gbar_min, gbar_max = df_finite.log_gbar.min(), df_finite.log_gbar.max()
bin_edges = np.linspace(gbar_min, gbar_max, N_BINS + 1)
df_finite["gbar_bin"] = pd.cut(df_finite.log_gbar, bins=bin_edges, labels=False, include_lowest=True)
print(f"  gbar range: [{gbar_min:.2f}, {gbar_max:.2f}], {N_BINS} bins")
print(f"  Finite rows: {len(df_finite)} of {len(df)}")

# Count per galaxy per bin
print("[STEP 3] Counting per galaxy per bin...")
bin_counts = df_finite.groupby(["SubhaloID", "gbar_bin"]).size().reset_index(name="n_in_bin")
print(f"  Total binned entries: {len(bin_counts)}")

# Apply ncut
bin_ok = bin_counts[bin_counts.n_in_bin >= NCUT]
print(f"  After ncut>={NCUT}: {len(bin_ok)} entries")

# Count bins_ok per galaxy
bins_per_gal = bin_ok.groupby("SubhaloID").size().reset_index(name="bins_ok")
print(f"  Galaxies with any bins_ok: {len(bins_per_gal)}")

# Merge with all galaxy IDs
all_ids = pd.DataFrame({"SubhaloID": df.SubhaloID.unique()})
result = all_ids.merge(bins_per_gal, on="SubhaloID", how="left")
result["bins_ok"] = result.bins_ok.fillna(0).astype(int)

# Also count n_dm_pts per galaxy (from lowres_flag or scatter)
scatter_path = RUN_DIR / "galaxy_scatter_dm.csv"
if scatter_path.exists():
    scatter = pd.read_csv(scatter_path)
    if "n_dm_pts" in scatter.columns:
        result = result.merge(scatter[["SubhaloID", "n_dm_pts"]], on="SubhaloID", how="left")
    elif "sigma_robust" in scatter.columns:
        result = result.merge(scatter[["SubhaloID", "sigma_robust"]], on="SubhaloID", how="left")
        result["n_dm_pts"] = (~result.sigma_robust.isna()).astype(int) * 50
        result.drop(columns=["sigma_robust"], inplace=True)
    else:
        result["n_dm_pts"] = 50
else:
    result["n_dm_pts"] = 50

result["n_dm_pts"] = result.n_dm_pts.fillna(0).astype(int)

# Write
OUT_QC.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT_QC, index=False)
print(f"\n[DONE] Written: {OUT_QC} ({len(result)} rows)")
print(f"  bins_ok distribution: {result.bins_ok.describe()}")
print(f"  Time: {time.time()-t0:.1f}s")

del df, df_finite, bin_counts, bin_ok, result