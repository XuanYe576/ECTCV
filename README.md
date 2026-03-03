# ECTCV: MSHNet with L1/L2/L3/L4/SoftIoU/LLoss and GP Variations

MSHNet for infrared small target detection, with multiple loss functions and Gaussian-Pinwheel (GP) spatial attention variants.

## Datasets

| Dataset | Link |
|---------|------|
| **NUDT-SIRST** | [TIB LDM](https://service.tib.eu/ldmservice/dataset/nudt-sirst) / [GitHub](https://github.com/YimianDai/sirst) |
| **IRSTD-1k** | [ISNet GitHub](https://github.com/RuiZhang97/ISNet) |
| **SIRST-UAVB** | [IEEE DataPort](https://ieee-dataport.org/documents/sirst-uavb-single-frame-infrared-small-target-dataset-uav-and-birds) |
| **NUAA-SIRST** | [SIRST GitHub](https://github.com/YimianDai/sirst) |

## Losses
- **L1** / L1-ONLY: L1-based IoU
- **L2** / L2-ONLY: L2-based IoU (SLS)
- **L3** / L3-ONLY: L3-based IoU (Mobius)
- **L3D**: L3 + D_loss (IR-SOIoU)
- **L4** / L4-ONLY: L4-based IoU (Man Fung)
- **IRSOIOU** / **IRSOIOU-LLOSS**: IR-SOIoU (Region Energy-Based)
- **LLOSS-ONLY**: Location/shape loss only
- **SOFTIOU**: Standard Soft IoU

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
