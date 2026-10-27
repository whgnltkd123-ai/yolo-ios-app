#!/usr/bin/env python3
"""Fine-tune native YOLO26n on ONE JOINT 81-class dataset (COCO 80 + drone).

This script requires real labeled COCO 80 + drone images. It does not synthesize
drone labels, and it will not claim to preserve COCO performance without testing.

Run on a GPU computer / Colab, not a Codemagic iOS packaging build worker:
  python tools/train_yolo26_81.py --data /content/yolo26_81/data.yaml --device 0

Prerequisite: pip install 'ultralytics>=8.4.142,<9' pyyaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_names(raw):
    if isinstance(raw, list):
        return [str(name) for name in raw]
    if isinstance(raw, dict):
        return [str(raw[i] if i in raw else raw[str(i)]) for i in range(len(raw))]
    raise ValueError("Invalid names field")


def validate_joint_dataset(dataset: Path, standard_classes: list[str]) -> dict:
    if not dataset.is_file():
        raise SystemExit(f"81-class dataset is missing: {dataset}")
    meta = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise SystemExit("Invalid YOLO dataset YAML")
    names = parse_names(meta.get("names"))
    expected = [*standard_classes, "drone"]
    if names != expected or meta.get("nc", 81) != 81:
        raise SystemExit("Classes must be the exact 80 COCO ids in order + drone id 80.")
    for key in ("train", "val"):
        if not meta.get(key):
            raise SystemExit(f"Missing {key} split.")
    print("Validated exact 81-class label order. Do not relabel bird, airplane, or kite as drone.")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640, choices=(416, 512, 640))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/yolo26-81")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from its last.pt")
    parser.add_argument(
        "--coco-val-data", type=Path, help="Optional ORIGINAL COCO-80 val data.yaml for retention check"
    )
    parser.add_argument("--max-coco-map-drop", type=float, default=0.03)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch <= 0 or not 0 <= args.max_coco_map_drop <= 1:
        raise SystemExit("Epochs and batch must be positive; COCO drop must be between 0 and 1")
    model = YOLO("yolo26n.pt")
    reference_names = [str(model.names[i]) for i in range(80)]
    validate_joint_dataset(args.data.expanduser().resolve(), reference_names)

    # The Ultralytics implementation maps matching COCO classification-head
    # rows by class name on 80->81 changes. Joint training is mandatory:
    # merely adding class name 'drone' does not teach drone geometry.
    if args.resume:
        model = YOLO(str(args.resume.expanduser().resolve()))
    model.train(
        data=str(args.data.expanduser().resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        project=args.project,
        name="coco80-plus-drone",
        exist_ok=True,
        seed=42,
        cls_remap=True,
        resume=bool(args.resume),
        plots=True,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise SystemExit("No fine-tuned checkpoint was generated.")
    trained = YOLO(str(best))
    names = [str(trained.names[i]) for i in range(81)]
    if names != [*reference_names, "drone"]:
        raise SystemExit("Fine-tuned model lost the required 81-class mapping.")

    # Same merged val split tests class-80 drone AND remaining 80 classes.
    scores = trained.val(
        data=str(args.data.expanduser().resolve()),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        plots=True,
    )
    class_maps = getattr(scores.box, "maps", [])
    drone_map5095 = float(class_maps[80]) if len(class_maps) == 81 else None
    if drone_map5095 is None or drone_map5095 <= 0:
        raise SystemExit("Drone validation failed or yielded no usable drone detections.")
    print("Held-out 81-class mAP50:", float(scores.box.map50))
    print("Drone mAP50-95 (validation):", drone_map5095)

    baseline_map, trained_map = None, None
    retention = {"status": "NOT VERIFIED"}
    if args.coco_val_data:
        from ultralytics.data.utils import check_det_dataset

        coco = check_det_dataset(str(args.coco_val_data), autodownload=False)
        if parse_names(coco["names"]) != reference_names:
            raise SystemExit("COCO retention dataset must use the original 80-class order.")
        # Use precisely the same resolved validation images and settings for both models.
        evaluation = {"path": str(coco["path"]), "train": coco["train"], "val": coco["val"]}
        original_yaml = best.with_name("coco80-eval.yaml")
        joint_yaml = best.with_name("coco81-eval.yaml")
        original_yaml.write_text(yaml.safe_dump({**evaluation, "names": reference_names}), encoding="utf-8")
        joint_yaml.write_text(yaml.safe_dump({**evaluation, "names": names}), encoding="utf-8")
        reference = YOLO("yolo26n.pt")
        original_eval = reference.val(data=str(original_yaml), device=args.device, imgsz=args.imgsz, split="val")
        joint_eval = trained.val(data=str(joint_yaml), device=args.device, imgsz=args.imgsz, split="val")
        if set(original_eval.box.ap_class_index) != set(range(80)) or set(joint_eval.box.ap_class_index) != set(
            range(80)
        ):
            raise SystemExit("Retention evaluation requires ground truth for all 80 COCO classes.")
        baseline_map = float(sum(original_eval.box.maps[:80]) / 80)
        trained_map = float(sum(joint_eval.box.maps[:80]) / 80)
        retention = {
            "status": "PASS" if baseline_map - trained_map <= args.max_coco_map_drop else "FAIL",
            "baseline_map5095": baseline_map,
            "trained_map5095": trained_map,
            "maximum_allowed_drop": args.max_coco_map_drop,
            "per_class_drop": {
                name: float(original_eval.box.maps[i] - joint_eval.box.maps[i])
                for i, name in enumerate(reference_names)
            },
            "scope": "Same COCO validation images only; not a field-performance guarantee.",
        }
        print("Original YOLO26n COCO-80 mAP50-95:", baseline_map)
        print("Fine-tuned COCO-80 mAP50-95 on the same images:", trained_map)

    digest = hashlib.sha256(best.read_bytes()).hexdigest()
    manifest = {
        "model": "yolo26n.pt",
        "classes": names,
        "drone_class_id": 80,
        "data": str(args.data.resolve()),
        "image_size": args.imgsz,
        "epochs_requested": args.epochs,
        "validation_map50_81": float(scores.box.map50),
        "validation_drone_map5095": drone_map5095,
        "checkpoint_sha256": digest,
        "preservation_of_original_80": retention,
        "real_iphone_false_alarms_and_fps": "NOT TESTED",
    }
    summary = best.with_name("yolo26-81-validation.json")
    summary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TRAINED WEIGHTS:", best)
    print("SHA256:", digest)
    print("VALIDATION REPORT:", summary)
    if retention["status"] == "FAIL":
        raise SystemExit("COCO retention threshold failed. Keep the checkpoint for diagnosis; do not promote it.")
    print("Manual sign-off required before using this model for any safety-sensitive alerting.")
    print("Upload only the checkpoint you have rights to use to an accessible HTTPS URL for Codemagic.")


if __name__ == "__main__":
    main()
