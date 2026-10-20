#!/usr/bin/env python3
"""Fine-tune the *native YOLO26n* detection architecture on bird+drone images.

This uses the same YOLO26 family that ships with the iOS app, NOT the slower
third-party YOLOv8s bird/drone checkpoint.  This script does not download a
dataset and never asserts that a model has been trained if the data is missing.

Examples:
  python tools/train_yolo26_drone.py --data /content/dataset/data.yaml --device 0
  python tools/train_yolo26_drone.py --data /content/dataset/data.yaml --epochs 60 --imgsz 512

Run on a GPU machine (Google Colab supported); Core ML export is a SEPARATE
Codemagic step on macOS using tools/import_trained_yolo26_drone.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def check_dataset(path: Path) -> tuple[list[str], Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"No dataset YAML at {path}. Download a labeled dataset first.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Dataset YAML must be a mapping.")
    raw = data.get("names", [])
    if isinstance(raw, dict):
        if sorted(int(k) for k in raw) != list(range(len(raw))):
            raise SystemExit("Dataset class indexes must be contiguous from zero.")
        names = [str(raw[i] if i in raw else raw[str(i)]) for i in range(len(raw))]
    elif isinstance(raw, list):
        names = [str(x) for x in raw]
    else:
        raise SystemExit("Dataset names must be a list or ID-to-label mapping.")
    labels = [n.strip().lower() for n in names]
    if len(labels) != 2 or set(labels) != {"bird", "drone"}:
        raise SystemExit(f"Expected two training labels, bird and drone; got {names}.")
    for key in ("train", "val"):
        if not data.get(key):
            raise SystemExit(f"Dataset lacks a separate {key} split.")
    if not (data.get("test") or data.get("val")):
        raise SystemExit("A held-out split is required.")
    print("WARNING: Split by complete video, not adjacent frames. Avoid dataset leakage.")
    return labels, path


def main() -> None:
    p = argparse.ArgumentParser(description="Train a 2-class YOLO26n drone/bird detector")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=512, choices=(416, 512, 640))
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="0=first CUDA GPU; mps for Apple Silicon")
    p.add_argument("--project", type=Path, default=Path("runs/yolo26-drone"))
    args = p.parse_args()
    names, data_path = check_dataset(args.data)
    if args.epochs < 1 or args.batch < 1:
        raise SystemExit("epochs and batch must be positive.")

    model = YOLO("yolo26n.pt")  # Native fast architecture; NOT yolov8s.pt
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        project=str(args.project),
        name="nano-bird-drone",
        exist_ok=True,
        seed=42,
        plots=True,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise SystemExit("Training failed to create best.pt")
    trained = YOLO(str(best))
    learned = {str(v).strip().lower() for v in trained.names.values()}
    if learned != {"bird", "drone"} or len(trained.names) != 2:
        raise SystemExit(f"Incorrect trained class names: {trained.names}")

    results = trained.val(
        data=str(data_path),
        split="test" if yaml.safe_load(data_path.read_text()).get("test") else "val",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )
    print("Held-out detection mAP50:", getattr(results.box, "map50", "unknown"))
    manifest = {
        "base_architecture": "yolo26n.pt",
        "task": "detect",
        "trained_classes": names,
        "checkpoint": str(best.resolve()),
        "image_size": args.imgsz,
        "tested_split": "test" if yaml.safe_load(data_path.read_text()).get("test") else "val",
        "limitations": "Not validated on real iPhone drone/empty-sky footage. No flutter-motion recognition.",
    }
    manifest_path = best.with_name("training-summary.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Training complete:", best)
    print("Training details:", manifest_path)
    print("Upload best.pt to a private or user-controlled HTTPS location, then use")
    print("Codemagic YOLO26 Drone Alert build with its URL and SHA256.")
    print("Caution: this two-class specialization does not automatically retain the")
    print("original COCO 80-class detector's recognition capabilities.")


if __name__ == "__main__":
    main()
