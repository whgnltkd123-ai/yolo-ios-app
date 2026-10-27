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
import json
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


def verify_coreml_export(weights: Path, export_path: Path, imgsz: int) -> dict:
    """Check conversion on fixed public inputs, independently of accuracy acceptance."""
    import coremltools as ct
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics.data.augment import LetterBox

    torch.set_num_threads(4)
    reference = YOLO(str(weights)).model.cpu().float().eval()
    reference.end2end = True  # Match the exporter's explicit nms=False branch.
    converted = ct.models.MLModel(str(export_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    source_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    converted.user_defined_metadata["checkpoint_sha256"] = source_sha
    spec = converted.get_spec()
    if len(spec.description.output) != 1:
        raise RuntimeError("Expected one end-to-end detection tensor")
    output_name = spec.description.output[0].name
    fixtures = sorted(Path("tools/coreml-check-inputs").glob("*.jpg"))
    if len(fixtures) != 3:
        raise RuntimeError("Missing fixed public conversion inputs")
    checks = []
    for fixture in fixtures:
        bgr = cv2.imread(str(fixture))
        rgb = LetterBox((imgsz, imgsz), auto=False)(image=bgr)[..., ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255
        with torch.no_grad():
            raw = reference(tensor)
        expected = (raw[0] if isinstance(raw, tuple) else raw).detach().numpy()[0]
        actual = np.asarray(converted.predict({"image": Image.fromarray(rgb)})[output_name])[0]
        if expected.shape != actual.shape or expected.shape != (300, 6):
            raise RuntimeError(f"Unexpected detection shape: {expected.shape}, {actual.shape}")
        if not np.isfinite(actual).all():
            raise RuntimeError("Non-finite Core ML output")
        # End-to-end TopK can reorder near-tied rows; compare scores by rank and
        # match confident boxes in both directions instead of assuming row IDs.
        np.testing.assert_allclose(np.sort(expected[:, 4]), np.sort(actual[:, 4]), atol=0.01, rtol=0.05)
        max_box_delta = 0.0
        max_score_delta = 0.0
        for left, right in ((expected, actual), (actual, expected)):
            used = set()
            for row in left[left[:, 4] >= 0.30]:
                candidates = [
                    (float(np.max(np.abs(row[:4] - r[:4]))), i)
                    for i, r in enumerate(right)
                    if i not in used and round(float(row[5])) == round(float(r[5]))
                ]
                delta, index = min(candidates, default=(float("inf"), -1))
                if index < 0 or delta > 4.0 or abs(float(row[4] - right[index, 4])) > 0.02:
                    raise RuntimeError(f"Core ML conversion mismatch: {fixture.name}")
                used.add(index)
                max_box_delta = max(max_box_delta, delta)
                max_score_delta = max(max_score_delta, abs(float(row[4] - right[index, 4])))
        checks.append(
            {
                "image": fixture.name,
                "image_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "pytorch_above_030": int((expected[:, 4] >= 0.30).sum()),
                "coreml_above_030": int((actual[:, 4] >= 0.30).sum()),
                "max_box_delta_pixels": max_box_delta,
                "max_score_delta": max_score_delta,
            }
        )
    report = {
        "checkpoint_sha256": source_sha,
        "classes": ["drone", "bird", "airplane"],
        "input_size": imgsz,
        "conversion_smoke_passed": True,
        "checks": checks,
        "limitations": "Three public frames only; not an accuracy gate or iPhone camera/device validation.",
    }
    Path("build").mkdir(exist_ok=True)
    Path("build/coreml-conversion-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    converted.save(str(DEST))
    return report


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
        "yolo26" in starting_model.lower() or starting_model.endswith(("yolo26-drone.pt", "nano-bird-drone.pt"))
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
    verify_coreml_export(weights, export_path, imgsz)
    print("Bundled fine-tuned YOLO26 model:", DEST)
    print("Original YOLO26 iOS SDK/camera unchanged; real-world FPS and accuracy untested.")


if __name__ == "__main__":
    main()
