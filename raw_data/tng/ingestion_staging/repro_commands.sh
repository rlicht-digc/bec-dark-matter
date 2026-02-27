#!/bin/bash
# Reproducible commands for TNG RAR extraction
# Generated: 2026-02-23T12:36:38.794172

# === DEV CLEAN (3000 x 50) ===
# Already complete. Canonical file:
# /home/tnguser/rar_profiles/20260222_201626/rar_points_CLEAN.parquet

# === BIG BASE (48133 x 50) ===
# Already complete. Canonical file:
# /home/tnguser/rar_profiles/20260223_061026_big_base/rar_points.parquet
# Ran via: nohup python3 big_extraction_29122.py > big_extraction.log 2>&1 &
# Completed in 5891.5s, 48133 ok, 0 failed

# === TO REPRODUCE FROM SCRATCH ===
# 1) Verify TNG100-1 data:
ls -la /home/tnguser/sims.TNG/TNG100-1/output/

# 2) Run extraction:
# nohup python3 /home/tnguser/big_extraction_29122.py > /home/tnguser/big_extraction.log 2>&1 &

# 3) Verify output:
python3 -c "
import pandas as pd
df = pd.read_parquet('/home/tnguser/rar_profiles/20260223_061026_big_base/rar_points.parquet')
print(f'Rows: {len(df)}, Galaxies: {df.SubhaloID.nunique()}, Pts/gal: {df.groupby("SubhaloID").size().min()}-{df.groupby("SubhaloID").size().max()}')
"

# === QUALITY-CUT RECIPE ===
# Base selection (TNG100-1, snap=99):
#   SubhaloFlag == 1
#   Mstar >= 1e8 Msun
#   n_star >= 100
#   n_dm >= 100
#   rhalf_star > 0
#   => 48,133 galaxies

# DEV subset quality cuts (bins8_dm10):
#   bins_ok >= 8 (out of 10 gbar bins, galaxy has enough points)
#   n_dm_pts >= 10 (minimum DM particle count threshold)
#   => 2,334 galaxies from 3,000 dev set

# Parameters:
#   N_RADII = 50
#   SOFT_KPC = 1.5
#   DM_PARTICLE_MASS = 8.85e6 Msun
#   BATCH_SIZE = 500