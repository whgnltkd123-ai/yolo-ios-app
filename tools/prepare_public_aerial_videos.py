"""Prepare CC0 visible aerial videos as supplemental training frames.

Requires numpy, scipy, mcos-decoder==0.1.0, opencv-python-headless and PyYAML.
All public clips go into training; existing validation is retained separately.
This does not establish independence from Seraphim or learn temporal motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat
from mcos_decoder.stream import MatStream


def load_video_table(path):
    raw = loadmat(str(path))["__function_workspace__"].tobytes()
    candidates = []

    def visit(element):
        children = element.children
        if element.type_id == 14 and len(children) >= 5:
            flags = children[0].as_array()
            # mxSTRUCT_CLASS; struct field-name length and packed names follow its header.
            if flags is not None and len(flags) and int(flags[0]) & 255 == 2:
                lengths = children[3].as_array()
                if lengths is not None and len(lengths) == 1 and lengths[0] > 0:
                    size = int(lengths[0])
                    packed = children[4].payload
                    names = [packed[i:i+size].split(b"\0")[0].decode("ascii")
                             for i in range(0, len(packed), size)]
                    if names == ["Time", "AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]:
                        dims = children[1].as_array()
                        rows = int(np.prod(dims))
                        cells = children[5:]
                        if len(cells) != rows * len(names):
                            raise ValueError("Frame table dimensions do not match cell count")
                        result = []
                        for i in range(rows):
                            record = {}
                            for j, name in enumerate(names[1:], 1):
                                cell = cells[i * len(names) + j]
                                # Some labeled regions carry a Position + LabelUIDs struct.
                                if cell.children[0].as_array()[0] & 255 == 2:
                                    field_width = int(cell.children[3].as_array()[0])
                                    field_bytes = cell.children[4].payload
                                    fields = [field_bytes[k:k+field_width].split(b"\0")[0]
                                              for k in range(0, len(field_bytes), field_width)]
                                    if fields != [b"Position", b"LabelUIDs"] or len(cell.children) != 7:
                                        raise ValueError("Unexpected annotated region struct")
                                    cell = cell.children[5]
                                values = cell.children[-1].as_array()
                                shape = tuple(int(x) for x in cell.children[1].as_array())
                                if values is None or cell.children[-1].type_id != 9:
                                    raise ValueError("Unexpected bounding-box cell")
                                if values.size == 0:
                                    record[name.lower()] = []
                                elif len(shape) == 2 and shape[1] == 4:
                                    record[name.lower()] = values.reshape(shape, order="F").tolist()
                                else:
                                    raise ValueError("Unexpected bounding-box shape")
                            result.append(record)
                        candidates.append(result)
        for child in children:
            visit(child)
    for element in MatStream(raw).walk():
        visit(element)
    if len(candidates) != 1:
        raise ValueError(f"Expected one named frame table, found {len(candidates)}")
    return candidates[0]


REVISION = "442d0c6750708cc221636f7a1fd31cbd8cb0aeb8"
REPOSITORY = "DroneDetectionThesis/Drone-detection-dataset"


def download_source(root):
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
    manifest = root / "source.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached["repo"] != REPOSITORY or cached["revision"] != REVISION:
            raise ValueError("Cached source revision differs")
        tree = {"tree": cached["files"]}
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            tree = json.load(response)
    if tree.get("truncated"):
        raise ValueError("Incomplete source tree")
    files = [
        item for item in tree["tree"]
        if item["path"].startswith("Data/Video_V/V_")
        and Path(item["path"]).name.startswith(("V_DRONE_", "V_BIRD_", "V_AIRPLANE_"))
        and item["path"].endswith((".mp4", "_LABELS.mat"))
    ]
    if len(files) != 448:
        raise ValueError(f"Expected 448 pinned source files, found {len(files)}")
    for i, item in enumerate(files):
        target = root / Path(item["path"]).name
        if not target.exists():
            with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{item['path']}",
                timeout=120,
            ) as response:
                content = response.read()
            target.write_bytes(content)
        content = target.read_bytes()
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest != item["sha"]:
            raise ValueError(f"Source hash mismatch: {target}")
        if i % 40 == 0:
            print(f"Verified {i + 1}/{len(files)} source files", flush=True)
    (root / "source.json").write_text(json.dumps(
        {"repo": REPOSITORY, "revision": REVISION, "license": "CC0-1.0", "files": files},
        indent=2,
    ))


def main():
    import cv2
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory to avoid mixing old labels")
    download_source(args.source)
    base = yaml.safe_load(args.base_data.read_text())
    names = base["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != ["drone", "bird", "airplane"]:
        raise ValueError(f"Unexpected base model class order: {names}")
    base_root = Path(base["path"])
    if not base_root.is_absolute():
        base_root = (args.base_data.parent / base_root).resolve()
    train_dir, val_dir = base_root / base["train"], base_root / base["val"]
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise ValueError("Base train and validation must be existing image directories")
    images, labels = args.output / "images", args.output / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    stats = []
    for video in sorted(args.source.glob("V_*.mp4")):
        records = load_video_table(video.with_name(video.stem + "_LABELS.mat"))
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {video}")
        stride = max(1, round(fps / 2))  # Reduce almost-identical neighboring frames.
        count = saved = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if count >= len(records):
                    raise ValueError(f"Video longer than annotations: {video}")
                record = records[count]
                h, w = frame.shape[:2]
                rows = []
                for class_id, name in enumerate(names):
                    for x, y, bw, bh in record[name]:
                        if not np.isfinite([x, y, bw, bh]).all() or bw <= 0 or bh <= 0:
                            raise ValueError(f"Invalid box: {video}, frame {count}")
                        # MATLAB intrinsic pixel centers start at 1; shift to 0-based.
                        x, y = x - 1, y - 1
                        if x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1:
                            raise ValueError(f"Out-of-frame box: {video}, frame {count}")
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(w, x + bw), min(h, y + bh)
                        if x2 <= x1 or y2 <= y1:
                            raise ValueError(f"Empty clipped box: {video}")
                        rows.append(f"{class_id} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} "
                                    f"{(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
                # Never turn annotated helicopters into unlabelled background.
                if count % stride == 0 and not record["helicopter"]:
                    stem = f"{video.stem}_{count:06d}"
                    if not cv2.imwrite(str(images / f"{stem}.jpg"), frame):
                        raise OSError(f"Could not save {stem}")
                    (labels / f"{stem}.txt").write_text("\n".join(rows))
                    saved += 1
                count += 1
        finally:
            cap.release()
        if count != len(records):
            raise ValueError(f"Frame count mismatch: {video}, {count}/{len(records)}")
        stats.append({"video": video.name, "frames": count, "sampled_frames": saved})
    # Replay original images without altering their annotation files.
    originals = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = np.random.default_rng(42)
    replay = rng.choice(len(originals), size=min(5000, len(originals)), replace=False)
    training = sorted(images.glob("*.jpg")) + [originals[i] for i in replay]
    manifest = args.output / "train.txt"
    # Keep replay symlink paths: their sibling labels contain the three-class remap.
    manifest.write_text("\n".join(str(p.absolute()) for p in training) + "\n")
    (args.output / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.output.resolve()), "train": str(manifest.resolve()),
        "val": str(val_dir.resolve()), "names": dict(enumerate(names)),
    }))
    (args.output / "preparation.json").write_text(json.dumps({
        "revision": REVISION, "videos": stats, "replay_images": len(replay),
        "training_images": len(training),
        "limitations": "All public clips are training-only. Existing validation is retained, "
        "but source overlap with Seraphim is unverified. Not an independent field test. "
        "Frames train appearance, not wing-beat dynamics.",
    }, indent=2))
    print(f"Prepared {len(training)} training images; {len(stats)} public clips", flush=True)


if __name__ == "__main__":
    main()
