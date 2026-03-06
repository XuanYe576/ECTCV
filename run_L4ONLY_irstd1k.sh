#!/bin/bash
# Run L4-ONLY on IRSTD-1k (L1/L2/L3-ONLY already in weight/).
set -euo pipefail
DATASET="${1:-/home/ubuntu/ECTCV/dataset/IRSTD-1k}"
cd /home/ubuntu/ECTCV
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mshnet
python main.py --dataset-dir "$DATASET" --loss-type L4-ONLY --mode train \
  --epochs 400 --batch-size 32 --base-size 256 --crop-size 256 --warm-epoch 40 --amp
echo "[$(date)] DONE: L4-ONLY"
