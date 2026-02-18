#!/bin/bash
# Download S4G Pipeline 5 stellar mass maps for PHANGS overlap galaxies
# 45 galaxies, ~3 MB each, ~135 MB total

DEST="/Users/russelllicht/Desktop/SPARC_RAR_Project/cf4_pipeline/data/s4g_p5"
mkdir -p "$DEST"

BASE="https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies"

GALAXIES=(
IC1954 IC5273 NGC0628 NGC0685 NGC1087 NGC1097 NGC1300
NGC1365 NGC1385 NGC1511 NGC1546 NGC1559 NGC1566 NGC1672
NGC1792 NGC1809 NGC2903 NGC3351 NGC3507 NGC3511 NGC3521
NGC3596 NGC3627 NGC4207 NGC4254 NGC4293 NGC4298 NGC4303
NGC4321 NGC4496A NGC4535 NGC4536 NGC4540 NGC4548 NGC4569
NGC4579 NGC4654 NGC4781 NGC4826 NGC4941 NGC4951 NGC5042
NGC5068 NGC5248 NGC7456
)

echo "Downloading S4G P5 stellar mass maps for ${#GALAXIES[@]} galaxies..."
echo "Destination: $DEST"

SUCCESS=0
FAIL=0

for GAL in "${GALAXIES[@]}"; do
    OUTFILE="$DEST/${GAL}.stellar.fits"
    if [ -f "$OUTFILE" ]; then
        echo "  SKIP $GAL (already exists)"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi
    URL="$BASE/$GAL/P5/${GAL}.stellar.fits"
    echo -n "  $GAL ... "
    if curl -s -f -L -o "$OUTFILE" "$URL" 2>/dev/null; then
        SIZE=$(ls -lh "$OUTFILE" | awk '{print $5}')
        echo "OK ($SIZE)"
        SUCCESS=$((SUCCESS + 1))
    else
        rm -f "$OUTFILE"
        echo "FAILED"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Done: $SUCCESS succeeded, $FAIL failed out of ${#GALAXIES[@]}"
echo "Total size:"
du -sh "$DEST"
