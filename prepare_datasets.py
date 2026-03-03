#!/usr/bin/env python3
"""Prepare NUDT-SIRST, NUAA-SIRST, SIRST-UAVB datasets for ECTCV training.
- Separate NUDT/NUAA from BasicIRSTD mixed archive
- Convert SIRST-UAVB from JPG to PNG
- Create trainval.txt / test.txt for each dataset
- Deduplicate across all datasets (including existing IRSTD-1k)
"""

import os
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from collections import defaultdict

BASE = Path("/home/ubuntu/ECTCV/dataset")
BASICIRSTD = BASE / "downloads" / "basicirstd"
UAVB_SRC = BASE / "downloads" / "sirst_uavb" / "SIRST-UAVB"

NUDT_DIR = BASE / "NUDT-SIRST"
NUAA_DIR = BASE / "NUAA-SIRST"
UAVB_DIR = BASE / "SIRST-UAVB"


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_nudt():
    print("=== Setting up NUDT-SIRST ===")
    for sub in ["images", "masks"]:
        (NUDT_DIR / sub).mkdir(parents=True, exist_ok=True)

    train_names = (BASICIRSTD / "img_idx" / "train_NUDT-SIRST.txt").read_text().strip().splitlines()
    test_names = (BASICIRSTD / "img_idx" / "test_NUDT-SIRST.txt").read_text().strip().splitlines()

    all_names = train_names + test_names
    for name in all_names:
        for sub in ["images", "masks"]:
            src = BASICIRSTD / sub / f"{name}.png"
            dst = NUDT_DIR / sub / f"{name}.png"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    (NUDT_DIR / "trainval.txt").write_text("\n".join(train_names) + "\n")
    (NUDT_DIR / "test.txt").write_text("\n".join(test_names) + "\n")
    print(f"  Train: {len(train_names)}, Test: {len(test_names)}, Total: {len(all_names)}")


def setup_nuaa():
    print("=== Setting up NUAA-SIRST ===")
    for sub in ["images", "masks"]:
        (NUAA_DIR / sub).mkdir(parents=True, exist_ok=True)

    train_names = (BASICIRSTD / "img_idx" / "train_NUAA-SIRST.txt").read_text().strip().splitlines()
    test_names = (BASICIRSTD / "img_idx" / "test_NUAA-SIRST.txt").read_text().strip().splitlines()

    all_names = train_names + test_names
    for name in all_names:
        for sub in ["images", "masks"]:
            src = BASICIRSTD / sub / f"{name}.png"
            dst = NUAA_DIR / sub / f"{name}.png"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    (NUAA_DIR / "trainval.txt").write_text("\n".join(train_names) + "\n")
    (NUAA_DIR / "test.txt").write_text("\n".join(test_names) + "\n")
    print(f"  Train: {len(train_names)}, Test: {len(test_names)}, Total: {len(all_names)}")


def setup_uavb():
    print("=== Setting up SIRST-UAVB ===")
    for sub in ["images", "masks"]:
        (UAVB_DIR / sub).mkdir(parents=True, exist_ok=True)

    train_ids = UAVB_SRC.joinpath("train.txt").read_text().strip().splitlines()
    test_ids = UAVB_SRC.joinpath("val.txt").read_text().strip().splitlines()

    all_ids = train_ids + test_ids
    converted = 0
    for fid in all_ids:
        fid = fid.strip()
        if not fid:
            continue
        src_img = UAVB_SRC / "images" / "all" / f"{fid}.jpg"
        src_mask = UAVB_SRC / "masks" / f"{fid}.jpg"
        dst_img = UAVB_DIR / "images" / f"{fid}.png"
        dst_mask = UAVB_DIR / "masks" / f"{fid}.png"

        if src_img.exists() and not dst_img.exists():
            Image.open(src_img).convert("RGB").save(dst_img)
            converted += 1
        if src_mask.exists() and not dst_mask.exists():
            Image.open(src_mask).convert("L").save(dst_mask)

    (UAVB_DIR / "trainval.txt").write_text("\n".join(i.strip() for i in train_ids if i.strip()) + "\n")
    (UAVB_DIR / "test.txt").write_text("\n".join(i.strip() for i in test_ids if i.strip()) + "\n")
    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}, Converted: {converted}")


def deduplicate():
    """Check for duplicate images across all 4 datasets by MD5."""
    print("\n=== Checking for duplicates across datasets ===")
    datasets = {
        "IRSTD-1k": BASE / "IRSTD-1k",
        "NUDT-SIRST": NUDT_DIR,
        "NUAA-SIRST": NUAA_DIR,
        "SIRST-UAVB": UAVB_DIR,
    }

    hash_to_files = defaultdict(list)
    for ds_name, ds_path in datasets.items():
        img_dir = ds_path / "images"
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.iterdir()):
            if f.is_file():
                h = md5_file(f)
                hash_to_files[h].append((ds_name, f.name, f))

    dups_found = 0
    removed = 0
    for h, files in hash_to_files.items():
        if len(files) > 1:
            dups_found += 1
            keep = files[0]
            print(f"  Duplicate group (keeping {keep[0]}/{keep[1]}):")
            for ds_name, fname, fpath in files[1:]:
                print(f"    Removing {ds_name}/{fname}")
                fpath.unlink(missing_ok=True)
                mask_path = fpath.parent.parent / "masks" / fname
                mask_path.unlink(missing_ok=True)
                txt_name = fpath.stem
                for txt_file in ["trainval.txt", "test.txt"]:
                    txt_path = fpath.parent.parent / txt_file
                    if txt_path.exists():
                        lines = txt_path.read_text().strip().splitlines()
                        new_lines = [l for l in lines if l.strip() != txt_name]
                        if len(new_lines) != len(lines):
                            txt_path.write_text("\n".join(new_lines) + "\n")
                removed += 1

    if dups_found == 0:
        print("  No duplicates found across datasets.")
    else:
        print(f"  Found {dups_found} duplicate groups, removed {removed} files.")

    for ds_name, ds_path in datasets.items():
        img_dir = ds_path / "images"
        if img_dir.exists():
            count = len(list(img_dir.iterdir()))
            print(f"  {ds_name}: {count} images")


if __name__ == "__main__":
    setup_nudt()
    setup_nuaa()
    setup_uavb()
    deduplicate()
    print("\nDone! Datasets ready.")
