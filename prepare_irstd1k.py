#!/usr/bin/env python3
"""Re-generate IRSTD-1k trainval.txt and test.txt from existing images/masks (sorted, 800/201)."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default='dataset/IRSTD-1k')
    p.add_argument('--train', type=int, default=800)
    p.add_argument('--test', type=int, default=201)
    args = p.parse_args()
    base = Path(args.dataset_dir)
    imgs_dir = base / 'images' if (base / 'images').exists() else base / 'IRSTD1k_Img'
    masks_dir = base / 'masks' if (base / 'masks').exists() else base / 'IRSTD1k_Label'
    if not imgs_dir.exists() or not masks_dir.exists():
        raise SystemExit(f'Need images and masks under {args.dataset_dir}')
    names = []
    for f in imgs_dir.iterdir():
        if f.is_file() and f.suffix.lower() == '.png':
            name = f.stem
            if (masks_dir / f'{name}.png').exists():
                names.append(name)
    names = sorted(names)
    need = args.train + args.test
    train_names = names[:args.train] if len(names) >= need else names
    test_names = names[args.train:args.train + args.test] if len(names) >= need else []
    (base / 'trainval.txt').write_text('\n'.join(train_names) + '\n')
    (base / 'test.txt').write_text('\n'.join(test_names) + '\n')
    print(f'IRSTD-1k: trainval {len(train_names)}, test {len(test_names)}')

if __name__ == '__main__':
    main()
