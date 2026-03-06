#!/usr/bin/env python
"""Run inference on all completed weight dirs (epoch=399) and compare with metric.log.
Usage: python run_all_val.py [--dataset-dir dataset/IRSTD-1k]
"""
import argparse
import os
import re
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default='dataset/IRSTD-1k')
    p.add_argument('--weight-dir', default='weight')
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()
    weight_base = args.weight_dir
    if not os.path.isdir(weight_base):
        print(f'No dir: {weight_base}')
        return
    import torch
    runs = []
    for name in sorted(os.listdir(weight_base)):
        d = os.path.join(weight_base, name)
        if not os.path.isdir(d):
            continue
        ckpt = os.path.join(d, 'weight.pkl')
        if not os.path.exists(ckpt):
            continue
        cp = os.path.join(d, 'checkpoint.pkl')
        if not os.path.exists(cp):
            continue
        try:
            c = torch.load(cp, map_location='cpu', weights_only=False)
            if c.get('epoch') != 399:
                continue
        except Exception:
            continue
        log_path = os.path.join(d, 'metric.log')
        log_iou = log_pd = log_fa = None
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            for line in reversed(lines):
                m = re.search(r'IoU\s+([\d.]+).*PD\s+([\d.]+).*FA\s+([\d.]+)', line)
                if m:
                    log_iou, log_pd, log_fa = float(m.group(1)), float(m.group(2)), float(m.group(3))
                    break
        runs.append((name, ckpt, log_iou, log_pd, log_fa))
    print(f'Found {len(runs)} completed runs. Running inference on {args.dataset_dir} ...\n')
    print('run_name\tlog_mIoU\tlog_Pd\tlog_FA\tinfer_mIoU\tinfer_Pd\tinfer_FA')
    print('-' * 100)
    for name, wpath, log_iou, log_pd, log_fa in runs:
        cmd = [sys.executable, 'run_val_once.py', '--dataset-dir', args.dataset_dir, '--weight-path', wpath, '--base-size', '256']
        if args.amp:
            cmd.append('--amp')
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.abspath(__file__)))
            text = out.stdout or out.stderr or ''
        except subprocess.TimeoutExpired:
            text = 'timeout'
        except Exception as e:
            text = str(e)
        infer_iou = infer_pd = infer_fa = None
        for line in text.splitlines():
            if 'mIoU:' in line:
                infer_iou = line.split('mIoU:')[-1].strip().split()[0]
            if 'Pd (thresh=0.5):' in line:
                infer_pd = line.split(':')[-1].strip().split()[0]
            if 'FA (thresh=0.5' in line:
                infer_fa = line.split(':')[-1].strip().split()[0]
        log_iou_s = f'{log_iou:.4f}' if log_iou is not None else '-'
        log_pd_s = f'{log_pd:.4f}' if log_pd is not None else '-'
        log_fa_s = f'{log_fa:.2f}' if log_fa is not None else '-'
        infer_iou_s = infer_iou if infer_iou else '-'
        infer_pd_s = infer_pd if infer_pd else '-'
        infer_fa_s = infer_fa if infer_fa else '-'
        print(f'{name}\t{log_iou_s}\t{log_pd_s}\t{log_fa_s}\t{infer_iou_s}\t{infer_pd_s}\t{infer_fa_s}')

if __name__ == '__main__':
    main()
