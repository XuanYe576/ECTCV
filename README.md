# ECTCV: MSHNet with L1/L2/L3/L4/SoftIoU/LLoss and GP Variations

MSHNet for infrared small target detection, with multiple loss functions and Gaussian-Pinwheel (GP) spatial attention variants.

## Losses
- **L1** / L1-ONLY: L1-based IoU
- **L2** / L2-ONLY: L2-based IoU (SLS)
- **L3** / L3-ONLY: L3-based IoU (Mobius)
- **L4** / L4-ONLY: L4-based IoU (Man Fung)
- **SOFTIOU**: Standard Soft IoU
- **LLOSS_ONLY**: Location/shape loss only

## GP Variations
- **Base**: Standard 7×7 spatial attention
- **GP Pipeline A**: Gaussian + pinwheel, argmax over orientations
- **GP Pipeline B**: Line-energy + k-gather

## Usage

```bash
# Train (single run)
python main.py --dataset-dir /path/to/dataset --loss-type L1 --use-gaussian-pinwheel --gp-pipeline A

# Batch train all L1/L2/L3/L4/Soft/LLoss + GP
./train_ectcv.sh /path/to/dataset
```

## Requirements
- PyTorch
- scikit-image
- PIL, tqdm
