"""Export the preserved c7 one2one detector; never train or change its decision thresholds."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parents[1]
SHA = "c7df126a039c1edde8d6b3e0c78be99f43e6770913eb1ff201976e2cf2505d26"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == SHA
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("score_export", ROOT / "tools/import_trained_yolo26_drone.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    detector = YOLO(str(args.checkpoint)).model.cpu().float().eval()
    assert detector.names == {0: "drone", 1: "bird", 2: "airplane"}, detector.names
    detector.end2end = True
    exporter.retain_aerial_class_scores(detector)
    head = detector.model[-1]
    head.export, head.format, head.dynamic = True, "onnx", False
    assets = ROOT / "android/app/src/main/assets"
    assets.mkdir(parents=True, exist_ok=True)
    dest = assets / "aerial.onnx"
    sample = torch.zeros(1, 3, 640, 640)
    with torch.inference_mode():
        detector(sample)
        torch.onnx.export(detector, sample, str(dest), input_names=["images"], output_names=["detections"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(dest)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(dest), options, providers=["CPUExecutionProvider"])
    checks = []
    fixtures = sorted((ROOT / "tools/coreml-check-inputs").glob("*.jpg"))
    assert len(fixtures) == 3
    for fixture in fixtures:
        rgb = LetterBox((640, 640), auto=False)(image=cv2.imread(str(fixture)))[..., ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255
        with torch.inference_mode():
            expected = detector(tensor).numpy()[0]
        actual = session.run(None, {"images": tensor.numpy()})[0][0]
        assert expected.shape == actual.shape == (300, 9)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(np.sort(expected[:, 4]), np.sort(actual[:, 4]), atol=0.0001, rtol=0.001)
        max_delta = 0.0
        for left, right in ((expected, actual), (actual, expected)):
            used = set()
            for row in left[left[:, 4] >= 0.10]:
                candidates = [(float(np.max(np.abs(row[:4]-r[:4]))), i) for i, r in enumerate(right)
                              if i not in used and round(float(row[5])) == round(float(r[5]))]
                delta, idx = min(candidates, default=(float("inf"), -1))
                assert delta < 0.1, (fixture, delta)
                np.testing.assert_allclose(row[4:], right[idx, 4:], atol=0.0001, rtol=0.001)
                used.add(idx)
                max_delta = max(max_delta, delta)
        checks.append({"image": fixture.name, "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                       "pytorch_above_030": int((expected[:, 4] > .30).sum()),
                       "onnx_above_030": int((actual[:, 4] > .30).sum()), "max_box_delta_px": max_delta})
    metadata = {"checkpoint_sha256": SHA, "onnx_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "classes": ["drone", "bird", "airplane"], "input": [1, 3, 640, 640],
                "output": [1, 300, 9], "normalization": "RGB CHW float32 / 255; letterbox fill 114",
                "output_fields": ["x1", "y1", "x2", "y2", "score", "class", "drone", "bird", "airplane"],
                "conversion_passed": True, "checks": checks,
                "release_clearance": "PENDING_DATA_RIGHTS_REVIEW",
                "limits": "Conversion smoke check, not Android camera/latency/accuracy certification."}
    (assets / "model-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
