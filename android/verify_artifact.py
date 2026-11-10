"""Verify packaged model, zip CRC and native ELF page alignment; not a device test."""
import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("artifact",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    prefix="base/" if args.artifact.suffix==".aab" else ""
    with zipfile.ZipFile(args.artifact) as package:
        assert package.testzip() is None
        metadata=json.loads(package.read(prefix+"assets/model-metadata.json"))
        assert metadata["conversion_passed"]
        assert metadata["checkpoint_sha256"]=="c7df126a039c1edde8d6b3e0c78be99f43e6770913eb1ff201976e2cf2505d26"
        assert hashlib.sha256(package.read(prefix+"assets/aerial.onnx")).hexdigest()==metadata["onnx_sha256"]
        for name in ("privacy.txt","usage.txt","notices.txt","licenses/AGPL-3.0.txt","licenses/ByteTrack-MIT.txt",
                     "licenses/ONNX-Runtime-MIT.txt","licenses/ONNX-Runtime-ThirdPartyNotices.txt",
                     "licenses/AndroidX-Apache-2.0.txt"):
            assert len(package.read(prefix+"assets/"+name))>20, name
        libraries=[]
        for name in package.namelist():
            if not name.endswith(".so"): continue
            data=package.read(name)
            assert data[:6]==b"\x7fELF\x02\x01", name
            phoff=struct.unpack_from("<Q",data,32)[0]
            phsize,phcount=struct.unpack_from("<HH",data,54)
            alignment=[]
            for i in range(phcount):
                header=struct.unpack_from("<IIQQQQQQ",data,phoff+i*phsize)
                if header[0]==1:
                    align=header[7]; alignment.append(align)
                    assert align>=16384 and (header[2]-header[3])%16384==0, (name,header)
            libraries.append({"name":name,"load_segment_alignment":alignment})
        assert libraries
        report={"artifact":args.artifact.name,"bytes":args.artifact.stat().st_size,
                "sha256":hashlib.sha256(args.artifact.read_bytes()).hexdigest(),"zip_crc_passed":True,
                "checkpoint_sha256":metadata["checkpoint_sha256"],"model_sha256":metadata["onnx_sha256"],
                "native_16kb_elf_passed":True,"libraries":libraries,
                "release_clearance":"PENDING_DATA_RIGHTS_AND_DEVICE_VALIDATION",
                "signature":"CHECK_SEPARATELY; release bundle has no publisher upload key"}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__=="__main__": main()
