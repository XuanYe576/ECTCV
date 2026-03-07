"""
可视化最佳模型的预测结果，找出漏检 / 误检最严重的样本。

输出目录结构：
  vis_output/
    all/          所有样本的对比图（原图 | GT | 预测 | 叠加）
    missed/       漏检严重（IoU < threshold_miss）
    false_alarm/  误检严重（有预测但 GT 为空，或 FA 极高）
    summary.csv   每个样本的 IoU / PD / FA 排序表

用法：
  python visualize_errors.py \
    --weight weight/MSHNet-unet-edge-sma-L1-2026-02-27-05-03-00/weight.pkl \
    --out-dir vis_output \
    --threshold-miss 0.3
"""

import os
import os.path as osp
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ── 模型 ──────────────────────────────────────────────────────────
from model.MSHNet import MSHNet


# ── 数据集（轻量版，不做随机 crop，保留原图尺寸） ────────────────────
class TestDataset(Data.Dataset):
    def __init__(self, dataset_dir, base_size=256):
        txt = osp.join(dataset_dir, 'test.txt')
        imgs_dir = osp.join(dataset_dir, 'images')
        mask_dir = osp.join(dataset_dir, 'masks')
        with open(txt) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.imgs_dir = imgs_dir
        self.mask_dir = mask_dir
        self.base_size = base_size
        self.transform = transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(osp.join(self.imgs_dir, name + '.png')).convert('RGB')
        mask = Image.open(osp.join(self.mask_dir, name + '.png')).convert('L')
        img_orig = np.array(img.resize((self.base_size, self.base_size)))  # HWC uint8
        return self.transform(img), self.mask_transform(mask), img_orig, name


# ── 单样本 IoU ────────────────────────────────────────────────────
def compute_iou(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum().item()
    union = (pred_bin | gt_bin).sum().item()
    return inter / (union + 1e-8)


def compute_pd_fa(pred_bin, gt_bin, img_size):
    """简化版 PD / FA（像素级）"""
    tp = (pred_bin & gt_bin).sum().item()
    fn = (~pred_bin & gt_bin).sum().item()
    fp = (pred_bin & ~gt_bin).sum().item()
    pd = tp / (tp + fn + 1e-8)
    fa = fp / (img_size * img_size + 1e-8) * 1e6   # 每百万像素虚警
    return pd, fa


# ── 可视化单张：原图 | GT | 预测概率图 | 叠加（绿=TP 红=FP 蓝=FN） ─
def make_comparison(img_pil, gt_np, pred_prob_np, name, iou, pd, fa, base_size):
    """返回拼接好的 PIL 图像（宽=4*base_size, 高=base_size+40）"""
    W = base_size

    def to_rgb(arr_01):
        a = (arr_01 * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(a).convert('RGB')

    pred_bin = (pred_prob_np > 0.5)
    gt_bin   = (gt_np > 0.5)

    # 叠加图：绿=TP, 红=FP, 蓝=FN
    overlay = np.zeros((W, W, 3), dtype=np.uint8)
    overlay[gt_bin & pred_bin]  = [0,   200, 0  ]   # TP 绿
    overlay[pred_bin & ~gt_bin] = [220, 0,   0  ]   # FP 红
    overlay[gt_bin & ~pred_bin] = [0,   0,   220]   # FN 蓝
    # 底图（暗化原图）+ 叠加
    base_np = np.array(img_pil.resize((W, W))).astype(np.float32) * 0.5
    mask_any = (overlay.sum(-1) > 0)
    base_np[mask_any] = base_np[mask_any] * 0.3 + overlay[mask_any].astype(float) * 0.7
    overlay_pil = Image.fromarray(base_np.clip(0, 255).astype(np.uint8))

    panels = [
        img_pil.resize((W, W)),
        to_rgb(gt_np),
        to_rgb(pred_prob_np),
        overlay_pil,
    ]
    titles = ['Image', 'GT', 'Pred(sigmoid)', f'IoU={iou:.3f} PD={pd:.2f} FA={fa:.1f}']

    # 拼宽图 + 标题栏
    banner_h = 40
    canvas = Image.new('RGB', (W * 4, W + banner_h), (30, 30, 30))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    for j, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel, (j * W, banner_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((j * W + 6, 10), title, fill=(220, 220, 50), font=font)

    # 样本名放左上
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 0), name, fill=(180, 180, 180), font=font)
    return canvas


# ── 主流程 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='weight/MSHNet-unet-edge-sma-L1-2026-02-27-05-03-00/weight.pkl')
    parser.add_argument('--dataset-dir', default='dataset/IRSTD-1k')
    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--out-dir', default='vis_output')
    parser.add_argument('--threshold-miss', type=float, default=0.3,
                        help='IoU 低于此值归入 missed 目录')
    parser.add_argument('--save-all', action='store_true',
                        help='保存所有样本的对比图（默认只保存 missed + false_alarm）')
    # 模型参数（与训练时一致）
    parser.add_argument('--edge-branch', action='store_true', default=True)
    parser.add_argument('--sma', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(osp.join(args.out_dir, 'missed'), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, 'false_alarm'), exist_ok=True)
    if args.save_all:
        os.makedirs(osp.join(args.out_dir, 'all'), exist_ok=True)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSHNet(3)  # base MSHNet (no GP); weight was trained with this
    w = torch.load(args.weight, map_location=device, weights_only=True)
    state = w.get('state_dict', w.get('net', w))
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f'Model loaded, device: {device}')

    # data
    dataset = TestDataset(args.dataset_dir, args.base_size)
    loader = Data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f'Test set: {len(dataset)} images')

    records = []

    with torch.no_grad():
        for imgs, masks, img_origs, names in tqdm(loader):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            name  = names[0]

            _, pred = model(imgs, True)   # [1,1,H,W] sigmoid 输出（warm_flag=True）

            pred_np = pred[0, 0].cpu().numpy()        # [H,W] float
            gt_np   = masks[0, 0].cpu().numpy()       # [H,W] float

            pred_bin = torch.from_numpy(pred_np > 0.5)
            gt_bin   = torch.from_numpy(gt_np   > 0.5)

            iou = compute_iou(pred_bin, gt_bin)
            pd, fa = compute_pd_fa(pred_bin, gt_bin, args.base_size)
            gt_has_target = gt_bin.any().item()

            records.append({'name': name, 'iou': iou, 'pd': pd, 'fa': fa,
                             'gt_has_target': gt_has_target})

            img_pil = Image.fromarray(img_origs[0].numpy())

            canvas = make_comparison(img_pil, gt_np, pred_np, name,
                                     iou, pd, fa, args.base_size)

            if args.save_all:
                canvas.save(osp.join(args.out_dir, 'all', f'{name}_iou{iou:.3f}.png'))

            # 漏检：GT 有目标但 IoU 很低
            if gt_has_target and iou < args.threshold_miss:
                canvas.save(osp.join(args.out_dir, 'missed', f'{name}_iou{iou:.3f}.png'))

            # 误检：GT 无目标但有预测，或 FA 极高（>100）
            if (not gt_has_target and pred_bin.any().item()) or fa > 100:
                canvas.save(osp.join(args.out_dir, 'false_alarm', f'{name}_FA{fa:.0f}.png'))

    # 保存 CSV
    records.sort(key=lambda x: x['iou'])
    csv_path = osp.join(args.out_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'iou', 'pd', 'fa', 'gt_has_target'])
        writer.writeheader()
        writer.writerows(records)

    # 打印统计
    ious = [r['iou'] for r in records]
    mean_iou = np.mean(ious)
    n_missed = sum(1 for r in records if r['gt_has_target'] and r['iou'] < args.threshold_miss)
    n_fa = sum(1 for r in records if (not r['gt_has_target'] and r['pd'] == 0) or r['fa'] > 100)
    print(f'\n===== Summary =====')
    print(f'Mean mIoU : {mean_iou:.4f}')
    print(f'Missed    (IoU<{args.threshold_miss}): {n_missed} images')
    print(f'FalseAlarm(GT empty / FA>100)  : {n_fa} images')
    print(f'Results saved to {args.out_dir}/')
    print(f'  missed/       <- worst missed detections')
    print(f'  false_alarm/  <- worst false alarms')
    print(f'  summary.csv   <- all samples sorted by IoU (lowest first)')


if __name__ == '__main__':
    main()
