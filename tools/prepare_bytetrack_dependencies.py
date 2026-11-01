"""Pinned upstream headers only; no Python/OpenCV runtime ships in the app."""
import io
import json
import pathlib
import tarfile
import urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "Sources/OfficialByteTrack/dependencies"
SOURCES = [
    ("eigen", "eigenteam/eigen-git-mirror", "cf794d3b741a6278df169e58461f8529f43bce5d", "Eigen/", ["COPYING.MPL2", "COPYING.BSD"]),
    ("opencv", "opencv/opencv", "49486f61fb25722cbcf586b7f4320921d46fb38e", "modules/core/include/", ["LICENSE"]),
]
def main():
    import hashlib
    records=[]
    for name, repo, commit, prefix, licenses in SOURCES:
        marker=DEST/name/"prepared.json"
        if marker.exists() and json.loads(marker.read_text())["commit"] == commit:
            continue
        url=f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data=response.read()
        count=0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                relative=member.name.partition("/")[2]
                if not member.isfile() or not (relative.startswith(prefix) or relative in licenses):
                    continue
                target_relative=relative if name=="eigen" else relative.removeprefix(prefix)
                target=DEST/name/target_relative
                if not target.resolve().is_relative_to((DEST/name).resolve()):
                    raise ValueError("unsafe archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                count+=1
        record=dict(repo=repo,commit=commit,archive_sha256=hashlib.sha256(data).hexdigest(),files=count)
        marker.write_text(json.dumps(record,indent=2))
        records.append(record)
    print(json.dumps(records))
if __name__ == "__main__":
    main()
