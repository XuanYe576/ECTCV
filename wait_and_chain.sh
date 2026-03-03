#!/bin/bash
# Wait for NUAA to finish all experiments, then chain IRSTD → UAVB → NUDT serially
cd "$(dirname "$0")"

echo "[$(date)] Waiting for NUAA experiments to complete..."
while pgrep -f "main.py.*NUAA-SIRST" > /dev/null 2>&1; do
    sleep 120
done
echo "[$(date)] NUAA done! Starting serial chain: IRSTD → UAVB → NUDT"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate mshnet 2>/dev/null

./train_serial_remaining.sh
