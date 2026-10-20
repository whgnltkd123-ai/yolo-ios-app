#!/usr/bin/env python3
"""Prepare an optional experimental YOLO bird/drone/aircraft model for iOS.

The source is a third-party pretrained YOLOv8 checkpoint, *not* a model we
trained. Its source repository does not provide a license; treat this as a
private feasibility test only, not a distributable product.
Only run this on a disposable build worker without credentials:
PyTorch .pt checkpoints use pickle and must be treated as untrusted code.

This script downloads a fixed Git blob, verifies its hash, inspects labels,
converts to Core ML without embedded NMS, and bundles a model to be tested.
No field accuracy or reliability is claimed.
"""
import hashlib
import os
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
from ultralytics import YOLO

REPO_COMMIT = "9a71f9e4ab9223bfa4d2aaf2ef63ba3781c229ce"
CHECKPOINT_GIT_BLOB_SHA1 = "44f69ea57b0c6e8e2d5627bf2b51f8752d6c47d1"
CHECKPOINT_BYTES = 22518115
SOURCE = (
    "https://raw.githubusercontent.com/kaal-bhairav69/Drones_vs_Birds_detection/"
    f"{REPO_COMMIT}/drone_vs_birds_detection/best%20%283%29.pt"
)
TARGET = Path("YOLOiOSApp/Models/Detect/yolo8s-birddrone.mlpackage")


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
    checkpoint = Path(".bird_drone_model_cache/bird-drone-2025-best.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        print("Downloading candidate bird/drone weights (unlicensed; testing only)")
        # Python.org's macOS Python may lack a configured system CA bundle.
        # Verify HTTPS with an explicitly trusted Mozilla CA bundle instead
        # of disabling certificate verification.
        tls_context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(SOURCE, timeout=120, context=tls_context) as resp:
                with checkpoint.open("wb") as output:
                    shutil.copyfileobj(resp, output)
        except Exception:
            # Remove partial files so a retry cannot read a truncated checkpoint.
            checkpoint.unlink(missing_ok=True)
            raise
    verify_git_blob(checkpoint)

    print("WARNING: The downloaded checkpoint is third-party code. Run only on an isolated worker.")
    model = YOLO(str(checkpoint))
    classes = [str(x).strip().lower() for x in model.names.values()]
    print("Checkpoint class names:", classes)
    if set(classes) != {"bird", "drone"} or len(classes) != 2:
        raise RuntimeError(f"Expected exactly bird/drone model labels; received {classes}")

    # Prefer responsiveness for this prototype. 416px has ~58% fewer input pixels than 640px.
    # The trade-off is potentially worse recognition of distant/small drones.
    # To compare high-detail behavior, set BIRD_DRONE_IMGSZ=640 on the build worker.
    image_size = int(os.environ.get("BIRD_DRONE_IMGSZ", "416"))
    if image_size not in (416, 512, 640):
        raise RuntimeError("BIRD_DRONE_IMGSZ must be 416, 512 or 640")
    print(f"Core ML export input: {image_size}x{image_size} (speed/detail trade-off)")
    export_path = Path(str(model.export(format="coreml", imgsz=image_size, nms=False, device="cpu")))
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
