#!/bin/bash
# Serial chain: run remaining experiments for IRSTD → UAVB → NUDT
# (NUAA is already running in the nuaa tmux session)
# Skips already-completed experiments per dataset.

PYTHON="${1:-python}"
EPOCHS=400
BATCH=32
BASE=256
CROP=256
WARM=40

cd "$(dirname "$0")"

LOSSES=("L1" "L1-ONLY" "L2" "L2-ONLY" "L3" "L3-ONLY" "L3D" "L4" "L4-ONLY" "IRSOIOU" "IRSOIOU-LLOSS" "LLOSS-ONLY" "SOFTIOU")

run_experiments() {
    local DATASET="$1"
    shift
    local SKIP_BASE="$1"
    shift

    local GP_CONFIGS=("base" "A" "B")

    for GP in "${GP_CONFIGS[@]}"; do
        local IDX=0
        for LOSS in "${LOSSES[@]}"; do
            IDX=$((IDX + 1))

            if [ "$GP" = "base" ] && [ "$IDX" -le "$SKIP_BASE" ]; then
                echo "[$(date)] SKIP $LOSS ($GP) — already done"
                continue
            fi

            if [ "$GP" = "base" ]; then
                echo "[$(date)] $DATASET | loss=$LOSS (base)"
                $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
                    --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM --amp
            else
                echo "[$(date)] $DATASET | loss=$LOSS + GP-Pipeline$GP"
                $PYTHON main.py --dataset-dir "$DATASET" --loss-type "$LOSS" --mode train \
                    --use-gaussian-pinwheel --gp-pipeline "$GP" \
                    --epochs $EPOCHS --batch-size $BATCH --base-size $BASE --crop-size $CROP --warm-epoch $WARM --amp
            fi
        done
    done
}

echo "============================================"
echo "=== IRSTD-1k: 35 remaining (skip L1,L1-ONLY base) ==="
echo "============================================"
run_experiments "dataset/IRSTD-1k" 2

echo "============================================"
echo "=== SIRST-UAVB: 38 remaining (skip L1 base) ==="
echo "============================================"
run_experiments "dataset/SIRST-UAVB" 1

echo "============================================"
echo "=== NUDT-SIRST: all 39 ==="
echo "============================================"
run_experiments "dataset/NUDT-SIRST" 0

echo "[$(date)] ALL SERIAL EXPERIMENTS DONE."
