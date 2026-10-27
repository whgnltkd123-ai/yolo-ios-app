#!/usr/bin/env python3
"""Merge official COCO-80 YOLO detection data with licensed bird/drone data.

Outputs a single *81-class* YOLO detection dataset. COCO class IDs 0..79 do
not change; 80 = drone. Source images are linked by default (never modified).
Requires a user-provided COCO YOLO-format dataset and bird/drone dataset:
the script does NOT claim to fetch data or to train a model.

Example:
 python tools/prepare_yolo26_81_data.py \
   --coco-root /content/datasets/coco \
   --drone-root /content/datasets/bird_drone \
   --drone-yaml /content/datasets/bird_drone/data.yaml \
   --output /content/yolo26_81
"""

from __future__ import annotations

import argparse
import collections
import shutil
from pathlib import Path

import yaml

COCO_SPLITS = {"train": "train2017", "val": "val2017"}
DRONE_SPLITS = {"train": "train", "val": "valid"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Aliases are checked explicitly. Unknown classes must not silently become background.
DRONE_LABEL_MAP = {"drone": 80, "uav": 80, "bird": 14, "airplane": 4, "aeroplane": 4, "kite": 33}


def indexed_names(values):
    if isinstance(values, dict):
        try:
            keys = sorted(int(i) for i in values)
            assert keys == list(range(len(keys)))
            return [str(values.get(i, values.get(str(i)))) for i in keys]
        except (ValueError, AssertionError) as error:
            raise ValueError("Class ids must be zero-based and contiguous.") from error
    if isinstance(values, list):
        return [str(name) for name in values]
    raise ValueError("Dataset class names must be an indexed list or mapping.")


def load_coco_names():
    import ultralytics

    cfg = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco.yaml"
    names = indexed_names(yaml.safe_load(cfg.read_text(encoding="utf-8"))["names"])
    if len(names) != 80 or names[14].lower() != "bird" or names[4].lower() != "airplane":
        raise ValueError("Unrecognized upstream COCO class list; cannot safely remap.")
    return names


def read_boxes(path: Path, mapping: dict[int, int], counters: collections.Counter):
    if not path.is_file():
        return ""
    lines = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5 and not (len(fields) >= 7 and len(fields) % 2 == 1):
            raise ValueError(f"Expected a YOLO box or polygon: {path}:{line_number}")
        class_value = float(fields[0])
        if not class_value.is_integer():
            raise ValueError(f"Non-integral class id in {path}:{line_number}")
        source_class = int(class_value)
        if source_class not in mapping:
            raise ValueError(f"Unknown source class {source_class} in {path}")
        numbers = [float(value) for value in fields[1:]]
        if any(not (0 <= value <= 1) for value in numbers):
            raise ValueError(f"Invalid normalized box in {path}:{line_number}")
        if len(fields) != 5:
            # Some Seraphim labels are YOLO segmentation polygons. Preserve each
            # annotated object as its enclosing detection box; never drop it.
            xs, ys = numbers[::2], numbers[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            numbers = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            counters["polygons_converted"] += 1
        if numbers[2] <= 0 or numbers[3] <= 0:
            raise ValueError(f"Degenerate box in {path}:{line_number}")
        destination_class = mapping[source_class]
        lines.append(str(destination_class) + " " + " ".join(format(value, ".12g") for value in numbers))
        counters[destination_class] += 1
    return ("\n".join(lines) + "\n") if lines else ""


def assemble_split(
    source_images: Path,
    source_labels: Path,
    dest_images: Path,
    dest_labels: Path,
    prefix: str,
    mapping: dict[int, int],
    copy_images: bool,
    counters: collections.Counter,
):
    if not source_images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {source_images}")
    if not source_labels.is_dir():
        raise FileNotFoundError(f"YOLO label directory not found: {source_labels}")
    paths = sorted(p for p in source_images.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {source_images}")
    for index, image in enumerate(paths):
        relative = image.relative_to(source_images)
        # Prefix and ordinal prevent collisions between COCO and drone datasets.
        identifier = f"{prefix}_{index:07d}"
        out_image = dest_images / (identifier + image.suffix.lower())
        out_label = dest_labels / (identifier + ".txt")
        if copy_images:
            shutil.copy2(image, out_image)
        else:
            out_image.symlink_to(image.resolve())
        boxes = read_boxes((source_labels / relative).with_suffix(".txt"), mapping, counters)
        out_label.write_text(boxes, encoding="utf-8")
    return len(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True, help="COCO images/train2017 + labels/train2017")
    parser.add_argument("--drone-root", type=Path, required=True, help="Roboflow train/images + valid/images")
    parser.add_argument("--drone-yaml", type=Path, required=True, help="Source drone dataset data.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--drone-valid-split",
        default="valid",
        choices=("valid", "val", "test"),
        help="Seraphim uses test; typical Roboflow datasets use valid",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of symlink for portable output")
    args = parser.parse_args()

    coco_names = load_coco_names()
    drone_yaml = yaml.safe_load(args.drone_yaml.read_text(encoding="utf-8"))
    drone_names = indexed_names(drone_yaml["names"])
    normalized = [name.strip().lower() for name in drone_names]
    if not any(name in {"drone", "uav"} for name in normalized):
        raise ValueError(f"Source dataset lacks a drone class: {drone_names}")
    if any(name not in DRONE_LABEL_MAP for name in normalized):
        raise ValueError(f"Unsupported source labels {drone_names}. Relabel them explicitly.")
    drone_mapping = {i: DRONE_LABEL_MAP[name] for i, name in enumerate(normalized)}
    if len(set(drone_mapping.values())) != len(drone_mapping):
        raise ValueError("Duplicate semantic source classes must be deduplicated first.")

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty to prevent stale training labels: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        images = output / "images" / split
        labels = output / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        count = collections.Counter()
        coco_split = COCO_SPLITS[split]
        drone_split = "train" if split == "train" else args.drone_valid_split
        coco_total = assemble_split(
            args.coco_root / "images" / coco_split,
            args.coco_root / "labels" / coco_split,
            images,
            labels,
            "coco",
            {i: i for i in range(80)},
            args.copy_images,
            count,
        )
        drone_total = assemble_split(
            args.drone_root / drone_split / "images",
            args.drone_root / drone_split / "labels",
            images,
            labels,
            "drone",
            drone_mapping,
            args.copy_images,
            count,
        )
        missing = [coco_names[i] for i in range(80) if count[i] == 0]
        if missing or count[80] == 0:
            raise ValueError(
                f"{split} has no annotations for classes: {missing}; drone count={count[80]}. "
                "Use COCO train2017/val2017 (not COCO8) and a labeled drone split."
            )
        summary[split] = {
            "coco_images": coco_total,
            "drone_images": drone_total,
            "drone_boxes": count[80],
            "polygons_converted": count["polygons_converted"],
        }
    merged = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "nc": 81,
        "names": [*coco_names, "drone"],
    }
    (output / "data.yaml").write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("COCO-80 + drone index 80 merged:", output / "data.yaml")
    print("Source data MUST have approved reuse terms. Never split adjacent frames across train/val.")
    print("Counts:", summary)
    if args.drone_valid_split == "test":
        print("WARNING: Seraphim test became validation; reserve DIFFERENT independent footage for final testing.")
    print("WARNING: This only prepares data. It has NOT trained or validated any weights.")


if __name__ == "__main__":
    main()
