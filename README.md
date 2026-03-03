# ECTCV: MSHNet with L1/L2/L3/L4/SoftIoU/LLoss and GP Variations

MSHNet for infrared small target detection, with multiple loss functions and Gaussian-Pinwheel (GP) spatial attention variants.

## Datasets

**IRSTD-1k** is included in this repo (`dataset/IRSTD-1k/`). Clone and run directly.

| Dataset | Link |
|---------|------|
| **IRSTD-1k** | Included in repo / [ISNet GitHub](https://github.com/RuiZhang97/ISNet) |
| **NUDT-SIRST** | [TIB LDM](https://service.tib.eu/ldmservice/dataset/nudt-sirst) / [GitHub](https://github.com/YimianDai/sirst) |
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
# Train with included IRSTD-1k (default)
python main.py --loss-type L1 --use-gaussian-pinwheel --gp-pipeline A

# Or specify dataset path
python main.py --dataset-dir dataset/IRSTD-1k --loss-type L1

# Batch train all losses + GP
./train_ectcv.sh dataset/IRSTD-1k
```

## Requirements
- PyTorch
- scikit-image
- PIL, tqdm
