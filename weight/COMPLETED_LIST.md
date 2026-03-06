# Completed runs (epoch=399)

Runs under `weight/` that finished 400 epochs (epoch index 399).  
L*-ONLY on IRSTD-1k: L1, L2, L3 already in this list when run; L4-ONLY can be added via `run_L4ONLY_irstd1k.sh`.

| Run folder (example) | Loss   | Dataset   | Epoch |
|---------------------|--------|-----------|-------|
| MSHNet-L1-...       | L1     | IRSTD-1k  | 399   |
| MSHNet-L2-...       | L2     | IRSTD-1k  | 399   |
| MSHNet-L3-...       | L3     | IRSTD-1k  | 399   |
| (others)            | ...    | ...       | 399   |

To regenerate this list and the inference comparison table, run:
- `python run_all_val.py [--dataset-dir dataset/IRSTD-1k] [--amp]`
