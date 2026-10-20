#!/usr/bin/env python3
"""Download the Seraphim drone YOLO dataset from Hugging Face for 81-class training.

This script runs on a GPU training machine/Google Colab with enough disk space,
not on the iOS builder. The full dataset is ~9.14 GB compressed plus extracted
images and labels. Labels are SINGLE class 0 ('drone'), not COCO+drone IDs.

Requires:
  pip install 'huggingface_hub>=0.28' pyyaml

Usage (full train+test; ~9 GB compressed):
  python tools/download_hf_seraphim.py --output /content/seraphim

Small smoke-test (ONLY first train batch + test, NOT full training):
  python tools/download_hf_seraphim.py --output /content/seraphim --train-batches 1

Source: https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset
Claimed license: CC BY 4.0. Verify underlying sources' license and give credit.
"""
from __future__ import annotations

import argparse
import os
import stat
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO = "lgrzybowski/seraphim-drone-detection-dataset"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def safe_unpack(path: Path, directory: Path):
    root = directory.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            dest = (directory / member.filename).resolve()
            if not dest.is_relative_to(root):
                raise ValueError(f"Archive contains unsafe path: {member.filename}")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError("Archive contains a symbolic link; refusing to unpack.")
        archive.extractall(directory)


def normalized_split_images(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSIONS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-batches", type=int, choices=(1, 2, 3, 4), default=4)
    args = p.parse_args()
    dest = args.output.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # HF has four giant train/image batch archives, one train/labels
    # archive, and one each for test images/labels. Avoid downloading the
    # 9.14GB repo when testing only one batch.
    patterns = ["train/labels/*.zip", "test/images/*.zip", "test/labels/*.zip", "LICENSE", "README.md"]
    patterns += [f"train/images/batch_{i:03d}.zip" for i in range(1, args.train_batches + 1)]
    print("Downloading", REPO, "; train batches:", args.train_batches)
    print("A complete download needs several GB of free disk space.")
    snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=patterns, local_dir=str(dest))

    for split in ("train", "test"):
        for kind in ("images", "labels"):
            folder = dest / split / kind
            archives = sorted(folder.glob("*.zip"))
            if not archives:
                raise FileNotFoundError(f"Missing expected image/label archives: {folder}")
            for archive in archives:
                print("Extracting", archive.relative_to(dest))
                safe_unpack(archive, folder)
                # Archive retained to permit retry/debugging; manually remove
                # only after verifying extraction and backing up the data.

    # Standard Seraphim layout: train/images, train/labels and test/images,
    # test/labels. Validate data rather than silently train an empty dataset.
    for split in ("train", "test"):
        images = dest / split / "images"
        labels = dest / split / "labels"
        count = normalized_split_images(images)
        label_files = list(labels.rglob("*.txt"))
        if count == 0 or not label_files:
            raise ValueError(f"No extracted image/label pairs in {split}; check ZIP directory layout.")
        # Check the first few labels to ensure the promised one-class map.
        for label in label_files[:50]:
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields and (len(fields) != 5 or fields[0] not in {"0", "0.0"}):
                    raise ValueError(f"Unexpected YOLO label in {label}: {line}")
        print(f"{split}: {count} images, {len(label_files)} label files")

    meta = {"path": str(dest), "train": "train/images", "val": "test/images", "names": {0: "drone"}, "nc": 1}
    (dest / "data.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print("Data extracted with 'test' as YOLO validation split.")
    print("IMPORTANT: this uses Seraphim's test split for validation. Reserve a NEW")
    print("independent field dataset for final test; no leakage-proof guarantee.")
    print("Use this root with tools/prepare_yolo26_81_data.py --drone-valid-split test.")
    print("This script does NOT train any neural network.")


if __name__ == "__main__":
    main()
