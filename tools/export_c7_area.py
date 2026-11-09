"""Export the two trained heads of c7 for on-demand area review only."""
from pathlib import Path
import hashlib
import json
import subprocess
from types import MethodType

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.export.coreml import _coreml_gather
from import_trained_yolo26_drone import retain_aerial_class_scores

C7 = "c7df126a039c1edde8d6b3e0c78be99f43e6770913eb1ff201976e2cf2505d26"


class AreaOutputs(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == C7
        self.network = YOLO(str(weights)).model.float().eval()
        self.network.end2end = True
        retain_aerial_class_scores(self.network)

    def forward(self, image):
        primary, raw = self.network(image)
        head = self.network.model[-1]
        many = raw["one2many"]
        boxes = head._get_decode_boxes(many).transpose(1, 2)
        scores = many["scores"].sigmoid().transpose(1, 2)
        drone, index = scores[:, :, 0].topk(300, dim=1)
        auxiliary = torch.cat((head._gather(boxes, index), drone.unsqueeze(2),
                               torch.zeros_like(drone).unsqueeze(2), head._gather(scores, index)), dim=2)
        return primary, auxiliary


def compare(expected, actual):
    assert expected.shape == actual.shape == (1, 300, 9)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(expected[0,:,4], actual[0,:,4], atol=.015, rtol=.05)
    for a,b in ((expected[0],actual[0]),(actual[0],expected[0])):
        used=set()
        for row in a[a[:,4]>=.30]:
            distance,i=min(((float(np.abs(row[:4]-other[:4]).max()),i) for i,other in enumerate(b)
                            if i not in used and int(other[5])==int(row[5])),default=(float("inf"),-1))
            assert i>=0 and distance<=4,(i,distance)
            np.testing.assert_allclose(row[6:],b[i,6:],atol=.025,rtol=.05)
            used.add(i)


def main():
    import coremltools as ct
    import cv2
    from PIL import Image
    from ultralytics.data.augment import LetterBox
    torch.set_num_threads(4)
    net=AreaOutputs(Path("checkpoints/aerial-three/best.pt")).eval()
    # Ultralytics' Core ML exporter uses this gather to retain integer anchor indices through MIL.
    head=net.network.model[-1]
    head._gather=MethodType(_coreml_gather,head)
    dummy=torch.zeros(1,3,640,640)
    with torch.inference_mode():
        net(dummy)
        traced=torch.jit.trace(net,dummy,check_trace=False)
    converted=ct.convert(traced,convert_to="mlprogram",minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
        inputs=[ct.ImageType(name="image",shape=dummy.shape,scale=1/255,color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="primary"),ct.TensorType(name="auxiliary")])
    converted.user_defined_metadata["checkpoint_sha256"]=C7
    converted.user_defined_metadata["area_schema"]="c7-area-v1"
    build=Path("build/area-export");build.mkdir(parents=True,exist_ok=True)
    package=build/"C7Area.mlpackage";converted.save(str(package))
    compiled=Path("YOLOiOSApp/Models/Area");compiled.mkdir(parents=True,exist_ok=True)
    subprocess.run(["xcrun","coremlcompiler","compile",str(package),str(compiled)],check=True)
    core=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    fixtures=sorted(Path("tools/coreml-check-inputs").glob("*.jpg"));assert len(fixtures)==3
    count=0
    for p in fixtures:
        image=cv2.imread(str(p));h,w=image.shape[:2]
        for view in (image,image[:round(.6*h),:round(.6*w)]):
            rgb=LetterBox((640,640),auto=False)(image=view)[...,::-1].copy()
            with torch.inference_mode():
                expected=net(torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255)
            actual=core.predict({"image":Image.fromarray(rgb)})
            for py,key in zip(expected,("primary","auxiliary")):compare(py.numpy(),actual[key])
            count+=1
    (build/"parity.json").write_text(json.dumps(dict(checkpoint=C7,inputs=count,heads=2)))
    print("C7_AREA_TWO_HEAD_COREML_PARITY_OK",count,flush=True)


if __name__=="__main__":
    main()
