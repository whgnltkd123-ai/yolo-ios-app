# YOLO26n / 81종 (COCO 80 + 드론) 학습·빌드 준비

> 이 저장소는 학습·검증 및 IPA 빌드 도구를 제공합니다. **완성된 81종 모델이나 빌드 성공을 의미하지 않습니다.** 실제 체크포인트와 검증 보고서, Codemagic 빌드 결과를 각각 확인하세요.

## Codex에서 모은 드론 사진 재활용

- Codex/이전 드론 도감 작업에 포함된 사진과 이미지 URL은 **자료 수집 후보**로 사용 가능합니다.
- 사진 URL, 카드/설명, 복제 가능한 실제 이미지 파일, YOLO 정답 라벨은 서로 다릅니다. 인터넷 링크나 HTML 카드가 있다고 해서 자동으로 YOLO가 학습되는 것이 아닙니다.
- **실제 이미지 파일을 모으고, 촬영·저작권·학습 이용 조건을 확인한 뒤, 각 사진의 드론 위치에 바운딩박스 라벨을 달아야 합니다.**
- 공개 전시·군 사진, 뉴스 사진은 기사에 보여도 학습 및 재배포 허락이 자동으로 주어지지 않습니다.
- 도감으로 모은 기체 사진은 가까운 선명한 사진에 편향될 수 있습니다. 원거리 점상 드론, 역광, 산림 배경, FPV, 쿼드, 헥사, 고정익, 비행기, 새, 구름·전선·곤충 등 현장 사진도 별도로 모아야 합니다.
- 연결된 GitHub의 `skywatch-web` 저장소에는 학습 이미지가 확인되지 않았습니다. 별도 Codex 프로젝트 URL 또는 이미지/라벨 ZIP이 있어야 실제 데이터를 검토할 수 있습니다.

## Hugging Face에서 바로 확보 가능한 드론 정답 데이터

**실제 검색으로 확인된 출처:** [Seraphim Drone Detection Dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset) (작성자 설명에 따르면 83,483장, YOLO 드론 1클래스 라벨, 640×640 이미지, 학습 75,134/시험 8,349, 약 9.14GB 압축, CC BY 4.0). 23개 출처를 모아 중복 제거했다고 하지만 이미지에는 실제 촬영 외 합성/광고 이미지가 섞이고 라벨 품질은 별도 수동 검증을 거치지 않았습니다. 출처별 사용 조건·저작자 표시 의무를 함께 확인하세요.

이 데이터셋은 `drone`에 대한 실제 박스가 들어 있으므로 81종 학습 후보로 적합합니다. 그러나 **COCO 80종 데이터가 포함된 건 아닙니다.** COCO 2017 80종의 YOLO 정답 라벨도 별도로 준비해야 합니다.

**개발 경험 없이 진행하기 위한 [Colab 81종 학습 노트북](https://colab.research.google.com/github/whgnltkd123-ai/yolo-ios-app/blob/main/colab/YOLO26_81_Seraphim_Train.ipynb)도 추가했습니다.** 버튼으로 실행되지만 대용량 데이터 다운로드/훈련은 사용자의 GPU 런타임이 필요하며 무료 Colab 제한으로 중단될 수 있습니다. 실제 학습·검증을 실행하기 전에는 `best.pt`가 없습니다.

### Colab 또는 별도 GPU 학습 서버에서 드론 데이터 준비

```bash
pip install 'huggingface_hub>=0.28' pyyaml
python tools/download_hf_seraphim.py --output /data/seraphim
```

- Hugging Face 원본은 `train/images/*.zip` 등 이미지·라벨이 압축 배치로 올라와 있습니다. 다운로드 스크립트가 순서대로 풀고 `data.yaml`(드론 클래스 0번)을 작성합니다.
- 전체 압축 데이터만 약 9GB이므로 압축 해제용 추가 저장공간과 네트워크가 필요합니다. `--train-batches 1`은 일부만 다운로드하는 시범 확인용이며 최종 모델 성능을 증명하지 않습니다.
- Seraphim은 `train`/`test`만 구분합니다. 스크립트는 `test`를 우선 **학습 도중 검증(val)** 에 활용합니다. 별도 독립 현장 테스트 영상은 반드시 남겨둬야 합니다.

### 원본 80종과 함께 81종 데이터로 통합

```bash
python tools/prepare_yolo26_81_data.py \
  --coco-root /data/coco \
  --drone-root /data/seraphim \
  --drone-yaml /data/seraphim/data.yaml \
  --drone-valid-split test \
  --output /data/yolo26_81
```

이어 기존 `tools/train_yolo26_81.py`로 별도 GPU 환경에서 학습, 학습된 `best.pt`의 SHA256 기록, Codemagic 81종 워크플로로 Core ML/IPA 변환을 합니다.

### 대체 Hugging Face 데이터

[pathikg/drone-detection-dataset](https://huggingface.co/datasets/pathikg/drone-detection-dataset)에도 이미지와 바운딩박스가 있으며 카드 표시는 MIT 라이선스입니다. 다만 이 자료는 Parquet 객체 구조이므로 위 ZIP 추출 스크립트를 그대로 적용할 수 없습니다. 데이터 필드와 사용권을 별도 검증해 변환해야 합니다.

---
## 데이터셋 생성

1. 본인이 이용·학습할 권한이 있는 COCO 2017 공식 YOLO 형식 이미지/라벨을 `images/train2017`, `labels/train2017`, `images/val2017`, `labels/val2017`로 준비합니다.
2. 라이선스가 확인된 새/드론 이미지·YOLO 박스 라벨 데이터셋을 `train/images`, `train/labels`, `valid/images`, `valid/labels`, `data.yaml`로 준비합니다.
3. 두 데이터셋의 영상은 촬영 세션 단위로 train/val을 분리합니다. 동일 비행의 연속 프레임이 양쪽에 들어가면 검증이 왜곡됩니다.
4. 아래 명령으로 클래스 번호를 통합합니다. 0~79는 공식 COCO, 80은 `drone`입니다.

```bash
python tools/prepare_yolo26_81_data.py --coco-root /data/coco --drone-root /data/drone --drone-yaml /data/drone/data.yaml --output /data/yolo26_81
```

원본 COCO 80종+새/드론 영상의 **모든 정답 객체 박스**가 필요합니다. 기존 80종을 버린 2클래스 모델은 이 단계에서 사용하지 않습니다.

## GPU 학습 (별도 GPU / Google Colab)

Colab 노트북은 COCO의 사각형 라벨(`coco2017labels.zip`)을 받습니다. 기본
`coco.yaml` 자동 다운로드의 다각형 라벨은 현재 통합 스크립트와 맞지 않습니다.
COCO 다운로드 직전에 최소 45GiB의 여유 공간을 확인합니다.

학습 전에 Google Drive 연결을 직접 승인하세요. 노트북의 학습 결과는
`MyDrive/YOLO26-81-training/coco80-plus-drone`에 저장되며, 완료된 매 epoch의
`weights/last.pt`로 다음 세션에서 이어 학습합니다. 세션이 끊기면 `/content`의
데이터는 사라질 수 있으므로 데이터 준비 셀을 다시 실행해야 합니다.
마지막 저장 이후 진행 중이던 epoch는 다시 계산됩니다. 무료 GPU 이용 시간은 보장되지 않습니다.

```bash
pip install 'ultralytics>=8.4.142,<9' pyyaml
python tools/train_yolo26_81.py --data /data/yolo26_81/data.yaml --epochs 60 --imgsz 640 --batch 8 --device 0
```

- YOLO26n 공식 가중치로 출발하고 `cls_remap=True`로 겹치는 COCO 클래스 분류 가중치를 재활용합니다.
- 학습 결과 `runs/yolo26-81/coco80-plus-drone/weights/best.pt` 및 `yolo26-81-validation.json` 확인이 필요합니다.
- 별도 COCO 일반객체 영상·드론 현장 영상·새-only·빈 하늘 영상에서 개별 클래스 성능, 오탐, 미탐 및 iPhone FPS를 비교해야 합니다.
- 80종이 보존되는지는 **클래스 이름과 실제 성능이 모두 유지될 때만** 주장할 수 있습니다. 81개 이름이 생겼다고 인식 품질이 유지되는 건 아닙니다.
- 노트북은 `--coco-val-data`를 전달해 원본과 학습 모델을 동일한 COCO 검증 이미지로 비교합니다. 80종 평균 mAP50-95가 기본 0.03(3%포인트)보다 더 떨어지면 검증 단계가 실패하고 클래스별 감소량을 보고서에 남깁니다. 이 기준 통과는 모든 클래스의 성능 보존이나 현장 성능을 보장하지 않습니다.

## Codemagic에서 iOS IPA 만들기 (학습 가중치 준비 뒤에만)

1. 프로젝트 https://codemagic.io/app/6ad64a00a899de35292dfe3f 를 엽니다.
2. 본인이 학습한 `best.pt`를 이용 권한이 있는 HTTPS 저장소에서 내려받을 수 있게 준비합니다.
3. Codemagic 환경 변수 `YOLO26_81_CHECKPOINT_URL`, `YOLO26_81_CHECKPOINT_SHA256`를 설정합니다.
4. `YOLO26 81 Classes - COCO80 + Drone (trained model required, unsigned)` 워크플로를 실행합니다.
5. 성공 시 `YOLO26-Drone-81-unsigned.ipa`가 Artifacts에 생성됩니다. Windows Sideloadly에 파일을 넣어 아이폰에 설치합니다.

**주의:** 체크포인트가 없거나 81종/해시 검증에 실패하면 빌드가 의도적으로 중단됩니다. 이 빌드는 GPU 학습 자체를 해주지 않습니다. 모델의 드론 식별 성능은 테스트가 필요합니다.

## 경보 및 표시

- 기존 YOLO26n 카메라·추론 엔진 유지
- 기본 화면에서는 새·비행기·연·드론 등 공중 객체만 박스 표시; 버튼으로 전체 80+1 클래스 표시 전환
- 새·드론 탐지 후보의 연속 관측 결과에 따라 미확인 물체(두 음), 드론 의심(세 음), 드론 확신(사이렌) 3단계 경보. 모델 추정일 뿐 실물 확정은 아닙니다.
- 기본 COCO80 가중치에서는 `drone` 클래스가 없어 드론 경보가 작동하지 않습니다.

## 사용·라이선스

기존 Ultralytics 코드 AGPL-3.0과 데이터셋·사진·학습된 모델의 이용 조건을 별도로 준수하세요. 학습/빌드 성공은 실전 안전 성능을 증명하지 않습니다.
