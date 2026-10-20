#!/usr/bin/env python3
"""Import the user's OWN trained YOLO26n drone/bird/airplane weights into the iOS app.

Mandatory environment variables:
  YOLO26_DRONE_CHECKPOINT_PATH=checkpoints/aerial-three/best.pt
  (or YOLO26_DRONE_CHECKPOINT_URL=https://.../best.pt)
  YOLO26_DRONE_CHECKPOINT_SHA256=<64 hex characters>

This intentionally fails without trained weights. Downloading stock yolo26n.pt
would NOT add drone recognition. Third-party PyTorch files can execute arbitrary
code during deserialization: use only your own trusted checkpoint on an isolated
Codemagic build host without credentials/secrets.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
from ultralytics import YOLO

DEST = Path("YOLOiOSApp/Models/Detect/yolo26n-aerial3.mlpackage")


def download_verified_weights() -> Path:
    url = os.environ.get("YOLO26_DRONE_CHECKPOINT_URL", "").strip()
    local = os.environ.get("YOLO26_DRONE_CHECKPOINT_PATH", "").strip()
    sha = os.environ.get("YOLO26_DRONE_CHECKPOINT_SHA256", "").lower().strip()
    if not (url or local) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(
            "No fine-tuned YOLO26 checkpoint provided. Set "
            "YOLO26_DRONE_CHECKPOINT_PATH (or URL) and YOLO26_DRONE_CHECKPOINT_SHA256 "
            "in Codemagic. Stock YOLO26 has no drone class."
        )
    if local:
        if url:
            raise SystemExit("Specify one checkpoint source, not both PATH and URL.")
        source = Path(local)
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise SystemExit("Local checkpoint SHA256 mismatch.")
        return source
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
        raise SystemExit("Checkpoint URL must be public or authenticated externally via a safe HTTPS download link.")
    dest = Path(".verified_yolo26_weights/yolo26-drone.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "YOLO26-drone-test-builder/1"})
    context = ssl.create_default_context(cafile=certifi.where())
    hasher = hashlib.sha256()
    max_bytes = 300 * 1024 * 1024
    transferred = 0
    try:
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            if urllib.parse.urlparse(response.url).scheme != "https":
                raise RuntimeError("Refusing HTTP redirection.")
            with dest.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    transferred += len(chunk)
                    if transferred > max_bytes:
                        raise RuntimeError("Checkpoint exceeds 300 MB limit.")
                    hasher.update(chunk)
                    stream.write(chunk)
        if hasher.hexdigest() != sha:
            raise RuntimeError("Checkpoint SHA256 did not match the expected digest.")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    print(f"Verified {transferred} checkpoint bytes with SHA256.")
    return dest


def main():
    weights = download_verified_weights()
    # IMPORTANT: loading .pt files runs pickle-related parsing; ONLY YOUR OWN weights.
    model = YOLO(str(weights))
    if model.task != "detect":
        raise SystemExit(f"Expected a detection checkpoint, got: {model.task}")
    labels = [str(model.names[i]).strip().lower() for i in sorted(model.names)]
    if labels != ["drone", "bird", "airplane"]:
        raise SystemExit(f"Expected drone, bird, airplane in that order, received {labels}")
    # If available, check the training arguments for the YOLO26 pretrained model.
    ckpt = getattr(model, "ckpt", None) or {}
    args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    starting_model = str(args.get("model", ""))
    if starting_model and not (
        "yolo26" in starting_model.lower()
        or starting_model.endswith("yolo26-drone.pt")
        or starting_model.endswith("nano-bird-drone.pt")
    ):
        print("WARNING: checkpoint training arguments do not prove a YOLO26n lineage:", starting_model)
    imgsz = int(os.environ.get("YOLO26_DRONE_IMGSZ", "640"))
    if imgsz not in (416, 512, 640):
        raise SystemExit("YOLO26_DRONE_IMGSZ must be 416, 512, or 640.")
    print("Exporting trained detection checkpoint as end-to-end Core ML at", imgsz)
    export_path = Path(str(model.export(format="coreml", imgsz=imgsz, nms=False, device="cpu")))
    if not export_path.is_dir() or export_path.suffix != ".mlpackage":
        raise SystemExit(f"Core ML export failed: {export_path}")
    if not (export_path / "Manifest.json").is_file():
        raise SystemExit("Exported Core ML package lacks Manifest.json")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(export_path, DEST)
    print("Bundled fine-tuned YOLO26 model:", DEST)
    print("Original YOLO26 iOS SDK/camera unchanged; real-world FPS and accuracy untested.")


if __name__ == "__main__":
    main()
