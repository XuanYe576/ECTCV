#!/bin/bash
# Train ECTCV ablation for a single dataset, with optional skip
# Usage: ./train_single_dataset.sh <dataset_dir> [skip_base_count] [python_path]
#
# skip_base_count: number of base (no GP) experiments already done (0-13)
#   GP-A and GP-B always run all 13.
#
# Examples:
#   ./train_single_dataset.sh dataset/NUDT-SIRST 0        # all 39
#   ./train_single_dataset.sh dataset/SIRST-UAVB 1        # skip L1 base (38)
#   ./train_single_dataset.sh dataset/IRSTD-1k 2           # skip L1,L1-ONLY base (37)
#   ./train_single_dataset.sh dataset/NUAA-SIRST 2         # skip L1,L1-ONLY base (37)

DATASET="${1:?Usage: $0 <dataset_dir> [skip_base] [python]}"
SKIP_BASE="${2:-0}"
PYTHON="${3:-python}"

EPOCHS=400
BATCH=32
BASE=256
CROP=256
WARM=40

LOSSES=("L1" "L1-ONLY" "L2" "L2-ONLY" "L3" "L3-ONLY" "L3D" "L4" "L4-ONLY" "IRSOIOU" "IRSOIOU-LLOSS" "LLOSS-ONLY" "SOFTIOU")

cd "$(dirname "$0")"

echo "=== Dataset: $DATASET | Skip first $SKIP_BASE base experiments ==="
echo "=== Start: $(date) ==="

echo "--- Base MSHNet (no GP) ---"
IDX=0
for LOSS in "${LOSSES[@]}"; do
    IDX=$((IDX + 1))
    if [ "$IDX" -le "$SKIP_BASE" ]; then
        echo "[$(date)] SKIP #$IDX $LOSS (base) — already done"
        continue
    fi
    echo "[$(date)] #$IDX $LOSS (base)"
    $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
        --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM --amp
done

echo "--- MSHNet + GP Pipeline A ---"
IDX=13
for LOSS in "${LOSSES[@]}"; do
    IDX=$((IDX + 1))
    echo "[$(date)] #$IDX $LOSS + GP-A"
    $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
        --use-gaussian-pinwheel --gp-pipeline A \
        --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM --amp
done

echo "--- MSHNet + GP Pipeline B ---"
IDX=26
for LOSS in "${LOSSES[@]}"; do
    IDX=$((IDX + 1))
    echo "[$(date)] #$IDX $LOSS + GP-B"
    $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
        --use-gaussian-pinwheel --gp-pipeline B \
        --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM --amp
done

echo "=== Done: $(date) ==="
