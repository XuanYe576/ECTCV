# ECTCV: MSHNet with L1/L2/L3/L4/SoftIoU/LLoss and GP Variations

MSHNet for infrared small target detection, with multiple loss functions and Gaussian-Pinwheel (GP) spatial attention variants.

Based on the original [MSHNet (CVPR 2024)](https://github.com/Lliu666/MSHNet) — *Infrared Small Target Detection with Scale and Location Sensitivity*.

## Datasets

| Dataset | Images | Download | Notes |
|---------|--------|----------|-------|
| **IRSTD-1k** | 1,001 | Included in repo (`dataset/IRSTD-1k/`) \| [ISNet GitHub](https://github.com/RuiZhang97/ISNet) | Ready to use |
| **NUDT-SIRST** | 1,327 | [BasicIRSTD Google Drive](https://drive.google.com/file/d/1LscYoPnqtE32qxv5v_dB4iOF4dW3bxL2/view?usp=sharing) \| [TIB LDM](https://service.tib.eu/ldmservice/dataset/nudt-sirst) | Download → `dataset/NUDT-SIRST/` |
| **NUAA-SIRST** | 427 | [BasicIRSTD Google Drive](https://drive.google.com/file/d/1LscYoPnqtE32qxv5v_dB4iOF4dW3bxL2/view?usp=sharing) \| [SIRST GitHub](https://github.com/YimianDai/sirst) | Download → `dataset/NUAA-SIRST/` |
| **SIRST-UAVB** | 3,000 | [Google Drive](https://drive.google.com/file/d/1hANdynk5C3fUQ1z2CqLRhAqUAfEsaWq8) \| [IEEE DataPort](https://ieee-dataport.org/documents/sirst-uavb-single-frame-infrared-small-target-dataset-uav-and-birds) | JPEG masks → run `prepare_datasets.py` |

> **Note:** SIRST-UAVB masks are originally JPEG with compression artifacts; run `prepare_datasets.py` to binarize (threshold 128) and convert to PNG. The script also deduplicates images shared across datasets.

Place each dataset under `dataset/` with the following structure:

```
dataset/<DATASET_NAME>/
├── images/        # RGB .png
├── masks/         # Binary .png (0/255)
├── trainval.txt   # Training split (one filename per line, no extension)
└── test.txt       # Test split
```

## Ablation: Loss Functions

13 loss variants for systematic comparison of area-penalty and location-penalty strategies:

| Loss | Alpha Formula | LLoss | Description |
|------|--------------|-------|-------------|
| **L1** | (min + \|Δ\|/2) / (max + \|Δ\|/2) | Yes | L1-norm area penalty |
| **L1-ONLY** | same | No | L1 without location loss |
| **L2 (SLS)** | (min + (Δ/2)²) / (max + (Δ/2)²) | Yes | Original MSHNet loss |
| **L2-ONLY** | same | No | SLS without location loss |
| **L3 (Mobius)** | (3·min + max) / (3·max + min) | Yes | Mobius transform penalty |
| **L3-ONLY** | same | No | Mobius without location loss |
| **L3D** | same as L3 | D_loss | L3 + IR-SOIoU center distance |
| **L4 (Man Fung)** | min / (max + 2·var) | Yes | Variance-based penalty |
| **L4-ONLY** | same | No | L4 without location loss |
| **IRSOIOU** | 1 − IoU^γ + α·D_loss | D_loss | Region energy-based dynamic loss |
| **IRSOIOU-LLOSS** | 1 − IoU^γ | LLoss | IR-SOIoU with LLoss instead of D_loss |
| **LLOSS-ONLY** | — | LLoss only | Pure location/shape loss (no IoU) |
| **SOFTIOU** | — | No | Standard Soft IoU baseline |

## Ablation: GP Spatial Attention

3 spatial attention configurations in each ResBlock:

| Config | Spatial Attention | Description |
|--------|------------------|-------------|
| **Base** | Conv2d(2→1, 7×7) | Standard CBAM-style (original MSHNet) |
| **GP Pipeline A** | Gaussian × Pinwheel + rotated argmax | Multi-orientation direction-sensitive |
| **GP Pipeline B** | Gaussian × Pinwheel + line-energy k-gather | Alternative aggregation strategy |

## Full Ablation Matrix

**13 losses × 3 GP configs × 4 datasets = 156 experiments**

```bash
# Run full ablation on one dataset
./train_ectcv.sh dataset/IRSTD-1k

# Run on all 4 datasets in parallel (requires ~32GB GPU memory)
tmux new -s nudt  && ./train_ectcv.sh dataset/NUDT-SIRST
tmux new -s nuaa  && ./train_ectcv.sh dataset/NUAA-SIRST
tmux new -s uavb  && ./train_ectcv.sh dataset/SIRST-UAVB
tmux new -s irstd && ./train_ectcv.sh dataset/IRSTD-1k
```

## Comparison Methods (Opap)

All comparison baselines are cloned under `~/Opap/`:

| Method | Venue | GitHub | Local Path |
|--------|-------|--------|------------|
| [MSHNet](https://github.com/Lliu666/MSHNet) | CVPR 2024 | <https://github.com/Lliu666/MSHNet> | Ours (base) |
| [DNA-Net](https://github.com/YeRen123455/Infrared-Small-Target-Detection) | TIP 2022 | <https://github.com/YeRen123455/Infrared-Small-Target-Detection> | `Opap/DNANet/` |
| [UIU-Net](https://github.com/danfenghong/IEEE_TIP_UIU-Net) | TIP 2023 | <https://github.com/danfenghong/IEEE_TIP_UIU-Net> | `Opap/UIUNet/` |
| [SCTransNet](https://github.com/xdFai/SCTransNet) | TGRS 2024 | <https://github.com/xdFai/SCTransNet> | `Opap/SCTransNet/` |
| [MTU-Net](https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection) | TGRS 2023 | <https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection> | `Opap/MTUNet/` |
| [ACM (Tianfang)](https://github.com/Tianfang-Zhang/acm-pytorch) | WACV 2021 | <https://github.com/Tianfang-Zhang/acm-pytorch> | `Opap/ACM-pytorch/` |
| [ACM (YimianDai)](https://github.com/YimianDai/open-acm) | WACV 2021 | <https://github.com/YimianDai/open-acm> | `Opap/ACM/` |

## Usage

```bash
# Single experiment
python main.py --dataset-dir dataset/IRSTD-1k --loss-type L1 --mode train \
    --epochs 400 --batch-size 32 --warm-epoch 40

# With GP Pipeline A
python main.py --dataset-dir dataset/IRSTD-1k --loss-type L1 \
    --use-gaussian-pinwheel --gp-pipeline A

# Full ablation (all 39 experiments on one dataset)
./train_ectcv.sh dataset/IRSTD-1k

# Run comparison baselines after ECTCV finishes
./Opap/run_all_baselines.sh
```

## Requirements

```
torch>=2.0
torchvision
scikit-image
Pillow
tqdm
```

## Project Structure

```
ECTCV/
├── main.py                  # Training entry point
├── train_ectcv.sh           # Batch ablation script
├── prepare_datasets.py      # Dataset preparation & deduplication
├── model/
│   ├── MSHNet.py            # Network (U-Net + CBAM + optional GP attention)
│   ├── loss.py              # All loss functions (L1–L4, SLS, IRSOIOU, etc.)
│   └── gpconv_7x7.py        # GP Pipeline B implementation
├── utils/
│   ├── data.py              # Dataset loader
│   └── metric.py            # IoU, PD, FA, ROC metrics
└── dataset/
    ├── IRSTD-1k/            # Included
    ├── NUDT-SIRST/          # Download separately
    ├── NUAA-SIRST/          # Download separately
    └── SIRST-UAVB/          # Download separately
```
