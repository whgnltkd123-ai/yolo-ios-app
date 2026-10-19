#!/usr/bin/env python3
"""Evaluate bird/drone confusions in independent field videos, per sampled frame.

This is a *screening* tool, not detector mAP, because video-level labels do
not provide frame-by-frame ground-truth bounding boxes. Split recording
sessions from training data and avoid mixed/ambiguous videos here.

Example:
  python tools/evaluate_bird_drone_videos.py --model best.pt \
     --bird bird_only.mp4 --drone drone_only.mp4 --empty empty_sky.mp4

Outputs a JSON summary of thresholded frame-level detection and false alarm
rates. Supply real, independently-recorded footage before making claims.
"""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def inspect_video(model, path: Path, stride: int, conf: float):
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to read video: {path}")
    frames = 0
    reviewed = 0
    counts = {"bird": 0, "drone": 0, "both": 0, "neither": 0}
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames += 1
            if (frames - 1) % stride:
                continue
            result = model.predict(image, verbose=False, conf=conf, imgsz=640)[0]
            seen = set()
            for box in result.boxes:
                label = str(result.names[int(box.cls.item())]).strip().lower()
                if label in ("bird", "drone") and float(box.conf.item()) >= conf:
                    seen.add(label)
            reviewed += 1
            key = "both" if len(seen) == 2 else next(iter(seen)) if seen else "neither"
            counts[key] += 1
    finally:
        cap.release()
    if reviewed == 0:
        raise RuntimeError(f"No frames were decoded from {path}")
    return {
        "file": str(path),
        "source_frames": frames,
        "sampled_frames": reviewed,
        "detections": counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--bird", type=Path, help="Recording known to contain birds but no drones")
    p.add_argument("--drone", type=Path, help="Recording known to contain drones but no birds")
    p.add_argument("--empty", type=Path, help="Recording known to contain neither birds nor drones")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--confidence", type=float, default=0.60)
    p.add_argument("--output", type=Path, default=Path("bird_drone_video_evaluation.json"))
    args = p.parse_args()
    if not (1 <= args.stride <= 120):
        p.error("--stride must be 1..120")
    if not (0.05 <= args.confidence <= 1.0):
        p.error("--confidence must be between 0.05 and 1.0")
    if not any([args.bird, args.drone, args.empty]):
        p.error("Provide at least one labelled bird, drone, or empty video")

    model = YOLO(str(args.model))
    labels = {str(x).lower() for x in model.names.values()}
    if not {"bird", "drone"}.issubset(labels):
        p.error(f"Model must distinguish bird and drone. Found {model.names}")

    report = {
        "note": "Per-sampled-frame screening, not bounding-box mAP nor deployment-ready accuracy",
        "model": str(args.model),
        "confidence": args.confidence,
        "stride": args.stride,
        "videos": {},
    }
    for category in ("bird", "drone", "empty"):
        path = getattr(args, category)
        if path is not None:
            stats = inspect_video(model, path, args.stride, args.confidence)
            c = stats["detections"]
            total = stats["sampled_frames"]
            if category == "drone":
                stats["fraction_with_drone_prediction"] = (c["drone"] + c["both"]) / total
            else:
                stats["false_drone_prediction_fraction"] = (c["drone"] + c["both"]) / total
            report["videos"][category] = stats

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
