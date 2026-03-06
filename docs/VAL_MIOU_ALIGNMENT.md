# Val mIoU alignment (72.71% vs 48%)

- Paper/repo reports **72.71%** mIoU on IRSTD-1k; our validation often shows lower (e.g. ~48%) if settings differ.
- **Causes:** (1) AMP in val: training uses autocast, so val should use `--amp` in `run_val_once.py` for parity. (2) Split: ensure same trainval/test split (e.g. first 800 / next 201). (3) Reset: metric state is reset per run; no bug found in reset logic.
- After aligning AMP and split, re-run `run_val_once.py --amp` and optionally `run_all_val.py --amp` to compare log vs inference mIoU/Pd/FA.
