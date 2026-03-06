# IRSTD-1k: completed runs and inference comparison (data)

## Completed L*-ONLY on IRSTD-1k

- L1-ONLY, L2-ONLY, L3-ONLY: already completed (epoch=399).
- L4-ONLY: run `./run_L4ONLY_irstd1k.sh`.

## Inference comparison (log vs rerun)

Training writes last best mIoU/Pd/FA to `metric.log`. This table compares those **log** values with **inference** values from `run_val_once.py` (same weight, same val set).  
Generate/update the table by running:

```bash
python run_all_val.py --dataset-dir dataset/IRSTD-1k --amp
```

Example columns (fill after running the command above):

| run_name | log_mIoU | log_Pd | log_FA | infer_mIoU | infer_Pd | infer_FA |
|----------|----------|--------|--------|------------|----------|----------|
| (example) | - | - | - | - | - | - |

- **Weight files:** `weight/<run_name>/weight.pkl` and `checkpoint.pkl` stay local (or use git-lfs if you want them in repo). This doc is the **comparison data** to commit.
