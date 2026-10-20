#!/usr/bin/env python3
"""Verify and convert a genuinely trained COCO-80+drone YOLO26n checkpoint.

Codemagic-only packaging step. It fails closed without the user's verified
trained .pt. PyTorch checkpoint deserialization may execute code: NEVER import
untrusted third-party weights in a build with signing secrets.
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
import yaml
from ultralytics import YOLO
import ultralytics

OUT = Path("YOLOiOSApp/Models/Detect/yolo26n-drone81.mlpackage")


def download():
    url = os.environ.get("YOLO26_81_CHECKPOINT_URL", "").strip()
    sha = os.environ.get("YOLO26_81_CHECKPOINT_SHA256", "").strip().lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise RuntimeError("Set YOLO26_81_CHECKPOINT_SHA256 (64 hex) from training result.")
    target = urllib.parse.urlsplit(url)
    if target.scheme != "https" or not target.netloc or target.username or target.password:
        raise RuntimeError("Set YOLO26_81_CHECKPOINT_URL to a direct HTTPS URL of your trained best.pt.")
    cache = Path(".verified_yolo26_81_weights")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "trained-best.pt"
    digest = hashlib.sha256()
    received = 0
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={"User-Agent": "YOLO26-81-Builder/1"})
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            if urllib.parse.urlsplit(response.url).scheme != "https":
                raise RuntimeError("HTTPS redirect protection")
            with destination.open("wb") as output:
                for part in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(part)
                    if received > 300 * 1024 * 1024:
                        raise RuntimeError("Refusing checkpoint above 300MB")
                    digest.update(part)
                    output.write(part)
        if digest.hexdigest() != sha:
            raise RuntimeError("SHA256 mismatch: refusing checkpoint")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    print("Verified trained checkpoint SHA256:", digest.hexdigest())
    return destination


def names_list(mapping):
    return [str(mapping[i] if i in mapping else mapping[str(i)]) for i in range(len(mapping))]


def main():
    weights = download()
    model = YOLO(str(weights))
    if model.task != "detect":
        raise RuntimeError("Checkpoint must be a YOLO object detector.")
    coco_yaml = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco.yaml"
    coco_names = names_list(yaml.safe_load(coco_yaml.read_text(encoding="utf-8"))["names"])
    expected = coco_names + ["drone"]
    names = names_list(model.names)
    if len(coco_names) != 80 or names != expected:
        raise RuntimeError("Not a true COCO-80 + drone (id 80) checkpoint; refusing mislabelled model.")

    # Best-effort check of model family/size; exact architecture cannot be
    # authenticated from a file name alone.
    architecture = getattr(getattr(model, "model", None), "yaml", {}) or {}
    scale = architecture.get("scale") if isinstance(architecture, dict) else None
    if scale is not None and scale != "n":
        raise RuntimeError(f"Expected nano architecture; checkpoint claims scale={scale}")
    args = model.ckpt.get("train_args", {}) if isinstance(getattr(model, "ckpt", None), dict) else {}
    starting = str(args.get("model", "")).lower()
    if starting and "yolo26n" not in starting and "best.pt" not in starting and "last.pt" not in starting:
        raise RuntimeError(f"Checkpoint training metadata does not look like YOLO26n: {starting}")

    size = int(os.getenv("YOLO26_81_IMGSZ", "640"))
    if size not in (416, 512, 640):
        raise RuntimeError("YOLO26_81_IMGSZ must be 416, 512 or 640.")
    # Match official iOS YOLO26 NMS-free 8-bit CoreML export for similar speed.
    print(f"Exporting YOLO26n COCO-80+drone as Core ML INT8 {size}x{size}, nms=False.")
    result = Path(str(model.export(format="coreml", imgsz=size, nms=False, quantize=8, device="cpu")))
    if result.suffix != ".mlpackage" or not (result / "Manifest.json").is_file():
        raise RuntimeError(f"Core ML export did not produce an mlpackage: {result}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(result, OUT)
    print(f"Bundled {len(names)} classes in {OUT}. Field detection performance remains UNTESTED.")


if __name__ == "__main__":
    main()
