#!/usr/bin/env bash
# stage_for_osf.sh — Assemble all 73 OSF upload files into public_osf/staging/
# Run from repo root: bash tools/osf_packaging/stage_for_osf.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGING="$REPO_ROOT/public_osf/staging"
RESULTS="$REPO_ROOT/analysis/results"
RAW_OBS="$REPO_ROOT/raw_data/observational"

DATASETS=(
  alfalfa brouwer2021 by_system cf4 cluster_rar eagle_rar environment
  hi_surveys literature little_things littlethings mhongoose misc phangs
  probes s4g sofue sparc tempel things vizier_catalogs voids wallaby
  yang_catalogs
)

echo "=== OSF Staging ==="
echo "Repo root: $REPO_ROOT"

# Clean and recreate staging tree
rm -rf "$STAGING"
mkdir -p "$STAGING"/{wiki,core,supporting,metadata}
for ds in "${DATASETS[@]}"; do
  mkdir -p "$STAGING/datasets/$ds"
done

# --- Wiki (2 files) ---
cp "$RESULTS/tests_results_osf.html"  "$STAGING/wiki/"
cp "$RESULTS/references_osf.html"     "$STAGING/wiki/"

# --- Core summaries — State 1 (16 files) ---
CORE_JSONS=(
  summary_env_definitive
  summary_env_confound_control
  summary_propensity_matched_env
  summary_mc_distance_and_inversion
  summary_nonparametric_inversion
  summary_binning_robustness
  summary_jackknife_robustness
  summary_lcdm_null_inversion
  summary_split_half_replication
  summary_kurtosis_phase_transition
  summary_kurtosis_disambiguation
  summary_alfalfa_yang_btfr
  summary_brouwer_lensing_rar
  summary_lensing_profile_shape
  summary_probes_inversion_replication
  summary_extended_rar_inversion
)
for name in "${CORE_JSONS[@]}"; do
  cp "$RESULTS/${name}.json" "$STAGING/core/"
done

# --- Supporting summaries — State 2 (5 files) ---
SUPPORTING_JSONS=(
  summary_cluster_scale_rar
  summary_cluster_sigma_scaling
  summary_kurtosis_mhongoose
  summary_soliton_nfw_composite
  summary_tf_scatter_redshift
)
for name in "${SUPPORTING_JSONS[@]}"; do
  cp "$RESULTS/${name}.json" "$STAGING/supporting/"
done

# --- Metadata (2 files) ---
cp "$REPO_ROOT/DATASETS_INDEX.md" "$STAGING/metadata/"
cp "$REPO_ROOT/CITATION.cff"      "$STAGING/metadata/"

# --- Dataset TERMS_OF_USE + CITATION (24 × 2 = 48 files) ---
for ds in "${DATASETS[@]}"; do
  cp "$RAW_OBS/$ds/TERMS_OF_USE.md" "$STAGING/datasets/$ds/"
  cp "$RAW_OBS/$ds/CITATION.md"     "$STAGING/datasets/$ds/"
done

# --- Summary ---
echo ""
echo "Staged files by directory:"
for dir in wiki core supporting metadata; do
  count=$(find "$STAGING/$dir" -type f | wc -l | tr -d ' ')
  printf "  %-14s %s files\n" "$dir/" "$count"
done
ds_count=$(find "$STAGING/datasets" -type f | wc -l | tr -d ' ')
printf "  %-14s %s files  (%s datasets)\n" "datasets/" "$ds_count" "${#DATASETS[@]}"

total=$(find "$STAGING" -type f | wc -l | tr -d ' ')
echo ""
echo "Total: $total files (expected: 73)"
if [ "$total" -eq 73 ]; then
  echo "PASS"
else
  echo "FAIL — count mismatch!" >&2
  exit 1
fi
