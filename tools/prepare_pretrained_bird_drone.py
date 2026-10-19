#!/usr/bin/env python3
"""Prepare an optional experimental YOLO bird/drone/aircraft model for iOS.

The source is a third-party pretrained YOLOv8 checkpoint, *not* a model we
trained. Only run this on a disposable build worker without credentials:
PyTorch .pt checkpoints use pickle and must be treated as untrusted code.

This script downloads a fixed Git blob, verifies its hash, inspects labels,
converts to Core ML without embedded NMS, and bundles a model to be tested.
No field accuracy or reliability is claimed.
"""
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from ultralytics import YOLO

REPO_COMMIT = "e4d852e2d7b2257568cb8e7c190a8691753270b1"
CHECKPOINT_GIT_BLOB_SHA1 = "2d3d549196cde9446bcfce6c4e59d7e02b5fe104"
CHECKPOINT_BYTES = 52009440
SOURCE = (
    "https://raw.githubusercontent.com/HarshaaMg/Flying_Object_Detection/"
    f"{REPO_COMMIT}/weights/refined_3class/best.pt"
)
TARGET = Path("YOLOiOSApp/Models/Detect/yolo8m-birddrone.mlpackage")


def verify_git_blob(path: Path):
    actual_size = path.stat().st_size
    if actual_size != CHECKPOINT_BYTES:
        raise RuntimeError(f"Expected {CHECKPOINT_BYTES} bytes; received {actual_size}")
    digest = hashlib.sha1()
    digest.update(f"blob {actual_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_GIT_BLOB_SHA1:
        raise RuntimeError("Git blob hash mismatch: refusing unverified checkpoint")


def main():
    checkpoint = Path(".bird_drone_model_cache/pretrained-best.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        print("Downloading pinned experimental bird/drone weights")
        with urllib.request.urlopen(SOURCE, timeout=120) as resp:
            with checkpoint.open("wb") as output:
                shutil.copyfileobj(resp, output)
    verify_git_blob(checkpoint)

    print("WARNING: The downloaded checkpoint is third-party code. Run only on an isolated worker.")
    model = YOLO(str(checkpoint))
    classes = [str(x).strip().lower() for x in model.names.values()]
    print("Checkpoint class names:", classes)
    if not {"bird", "drone"}.issubset(classes) or len(classes) != 3:
        raise RuntimeError("Expected exactly three classes, including bird and drone")

    export_path = Path(str(model.export(format="coreml", imgsz=640, nms=False, device="cpu")))
    if export_path.suffix != ".mlpackage" or not export_path.is_dir():
        raise RuntimeError(f"Core ML export did not produce a package: {export_path}")

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(export_path, TARGET)
    print(f"Bundled converted Core ML model: {TARGET}")
    print("Not independently validated: do not use as a reliable drone warning system.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Bird/drone model preparation failed: {exc}", file=sys.stderr)
        raise
