#!/bin/bash
# ECTCV: Train L1, L2, L3, L4, SoftIoU, LLoss (base + GP Pipeline A/B)
# Usage: ./train_ectcv.sh <dataset_dir> [python_path]
# Example: ./train_ectcv.sh /path/to/NUDT-SIRST

PYTHON="${2:-python}"
DATASET="${1:-dataset/IRSTD-1k}"
# IRSTD-1k included in repo; use as default
EPOCHS=400
BATCH=32
BASE=256
CROP=256
WARM=40

LOSSES=("L1" "L1-ONLY" "L2" "L2-ONLY" "L3" "L3-ONLY" "L3D" "L4" "L4-ONLY" "IRSOIOU" "IRSOIOU-LLOSS" "LLOSS-ONLY" "SOFTIOU")

cd "$(dirname "$0")"

echo "=== Base MSHNet (no GP) ==="
for LOSS in "${LOSSES[@]}"; do
    echo "[$(date)] loss=$LOSS (base)"
    $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
        --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM
done

echo "=== MSHNet + GP Pipeline A ==="
for LOSS in "${LOSSES[@]}"; do
    echo "[$(date)] loss=$LOSS + GP-PipelineA"
    $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
        --use-gaussian-pinwheel --gp-pipeline A \
        --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM
done

echo "=== MSHNet + GP Pipeline B ==="
for LOSS in "${LOSSES[@]}"; do
    echo "[$(date)] loss=$LOSS + GP-PipelineB"
    $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
        --use-gaussian-pinwheel --gp-pipeline B \
        --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM
done

echo "Done."
