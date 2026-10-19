#!/usr/bin/env python3
"""Train a two-class bird/drone YOLOv8n model and export for the iOS app.

Usage (on a GPU-enabled Linux/macOS computer or Colab):
  pip install "ultralytics>=8.3,<9" "coremltools>=9,<11" "numpy<=2.3.5" pyyaml
  python tools/train_bird_drone.py --data /path/to/roboflow-download/data.yaml

The dataset must have separate BIRD and DRONE classes. Don't treat every
unlabeled distant dot as a positive drone. Use held-out real field videos.
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def validate_two_classes(data_yaml: Path):
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open(encoding="utf-8") as fp:
        metadata = yaml.safe_load(fp)
    raw_names = metadata.get("names", [])
    names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
    normalized = {str(name).strip().lower() for name in names}
    if not {"bird", "drone"}.issubset(normalized):
        raise SystemExit(f"Required classes 'bird' and 'drone'; got: {names}")
    if len(names) != 2:
        raise SystemExit(f"Expected exactly 2 classes for current prototype, got: {names}")
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="Downloaded Roboflow YOLO dataset data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="0=first GPU; cpu for a slow smoke test")
    p.add_argument("--skip-coreml", action="store_true", help="Train .pt only")
    args = p.parse_args()
    metadata = validate_two_classes(args.data)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=15,
        project="runs/bird_drone",
        name="yolov8n",
        exist_ok=True,
        seed=42,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise SystemExit("Training did not produce a best.pt checkpoint.")
    trained = YOLO(str(best))
    validation = trained.val(data=str(args.data.resolve()), split="val", imgsz=args.imgsz, device=args.device)
    print("Validation mAP@50:", getattr(validation.box, "map50", None))
    print("Class labels:", trained.names)
    print("NOTE: Validate drone-vs-bird false alarms on independent camera footage before field use.")

    if args.skip_coreml:
        print("Trained model:", best)
        return
    exported_path = Path(trained.export(format="coreml", imgsz=args.imgsz))
    if exported_path.suffix != ".mlpackage" or not exported_path.is_dir():
        raise SystemExit(f"Expected exported .mlpackage directory, received {exported_path}")
    destination = Path("output/yolo8n-birddrone.mlpackage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(exported_path, destination)
    print("Model ready to bundle in YOLOiOSApp/Models/Detect:", destination)


if __name__ == "__main__":
    main()
