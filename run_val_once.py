#!/usr/bin/env python
"""Run validation once with a saved weight; report mIoU and Pd/FA (bin 5 = thresh 0.5).
Usage: python run_val_once.py --weight-path weight/MSHNet-L2-.../weight.pkl [--dataset-dir dataset/IRSTD-1k] [--amp]
"""
import argparse
import torch
import torch.utils.data as Data
from utils.data import IRSTD_Dataset
from utils.metric import mIoU, PD_FA
from model.MSHNet import MSHNet
import os.path as osp

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default='dataset/IRSTD-1k')
    p.add_argument('--weight-path', required=True)
    p.add_argument('--base-size', type=int, default=256)
    p.add_argument('--amp', action='store_true', help='Use autocast to match training val')
    args = p.parse_args()
    args.crop_size = args.base_size
    valset = IRSTD_Dataset(args, mode='val')
    val_loader = Data.DataLoader(valset, 1, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSHNet(3).to(device)
    state = torch.load(args.weight_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()
    mIoU_metric = mIoU(1)
    PD_FA_metric = PD_FA(1, 10, args.base_size)
    tag = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        for data, mask in val_loader:
            data, mask = data.to(device), mask.to(device)
            if args.amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _, pred = model(data, tag)
            else:
                _, pred = model(data, tag)
            pred = pred.float()
            pred_prob = torch.sigmoid(pred)
            mIoU_metric.update(pred, mask)
            PD_FA_metric.update((255.0 * pred_prob).float(), mask)
    _, mean_iou = mIoU_metric.get()
    FA, PD = PD_FA_metric.get(len(val_loader))
    pd_bin = min(5, len(PD) - 1)
    print('mIoU:', float(mean_iou))
    print('Pd (thresh=0.5):', float(PD[pd_bin]))
    print('FA (thresh=0.5, x1e6):', float(FA[pd_bin] * 1e6))
    print('Val samples:', len(val_loader))

if __name__ == '__main__':
    main()
