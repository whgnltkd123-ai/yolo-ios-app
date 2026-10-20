# YOLO26 원본 엔진 + 새·드론 학습 + 3단계 경보

> **중요(2026-10-20):** 앱 코드와 학습·변환 경로는 준비했지만 사용자 데이터로 학습한 `best.pt`는 아직 없으며 새 YOLO26 드론 앱의 Codemagic 빌드 및 iPhone 실측 검증은 **미완료**입니다. 기본 YOLO26n에 드론 클래스가 자동으로 생기지는 않습니다.

## 공중 물체만 박스 표시 (2026-10-20)

- **YOLO26n 원본 80개 COCO 탐지 클래스는 그대로 유지합니다.** 추론은 모든 클래스에 대해 수행하고, Detect 화면에서는 기본적으로 `bird`, `airplane`, `kite`, 추후 학습된 `drone` 등 공중 객체 클래스만 박스로 표시합니다.
- 카메라 좌상단 **✈️ 공중 객체만 표시 / 📦 모든 객체 표시** 버튼으로 즉시 전환합니다.
- 화면에 박스를 그리는 부담과 시각적 혼잡은 줄지만 **YOLO26 모델의 FLOPs나 80종 추론량은 줄지 않으므로 큰 FPS 개선을 보장할 수 없습니다.**
- 새·비행기·연을 식별한다고 해서 미확인 점상 물체 또는 드론을 검출하는 것은 아닙니다. YOLO26 원본에 `drone` 클래스가 없으므로, **새로 학습한 드론 모델 없이는 3단계 드론 경보를 울리지 않습니다.**
- `YOLO iOS Build Check (unsigned)` 워크플로가 만든 `YOLOiOSApp-unsigned.ipa`는 **원본 YOLO26 80종 + 공중 객체만 표시하는 UI 시험 버전**입니다. 드론 식별 추가 버전과 혼동하지 마세요.

## 변경된 앱 구조

- iOS 카메라, YOLOView, AVCaptureVideoPreviewLayer, Vision/Core ML, 기존 YOLO26 디코더는 **그대로** 사용합니다.
- 카메라 시작 시 `YOLO26n` nano 모델을 선택합니다. `yolo26n-birddrone.mlpackage`가 있으면 그것을 우선 선택하고, 없다면 원본 `yolo26n.mlpackage`로 실행합니다.
- 별도 무거운 YOLOv8s bird/drone 체크포인트는 새 워크플로에서 사용하지 않습니다. 예전 워크플로 `Bird vs Drone - Core ML Test`는 과거 프로토타입이므로 더 이상 권장하지 않습니다.
- 원본 COCO YOLO26n은 기본적으로 `drone` 클래스가 없습니다. YOLO26n의 아키텍처를 유지하더라도 **가중치를 새·드론용으로 실제 추가학습한 경우에만** 드론 경보가 작동합니다.
- 이번에 만든 **2클래스 전문화 모델**은 원래 COCO 80종 객체 검출을 그대로 유지하지 않습니다. 80종+드론 동시 검출을 원하면 별도 통합 데이터셋과 장기 재학습/증분학습 실험이 필요합니다.

## 경보 단계 (실험적 기준: 실제 탐지 신뢰도 아님)

| 단계 | 조건(대략) | 경보 |
| --- | --- | --- |
| 미확인 물체 | 2클래스 모델이 드론 후보로 탐지하되, 신뢰도/연속 프레임 조건 부족 | 낮은 2음 경고, 조건 충족 시 발생 |
| 드론 의심 | 동일 위치 약 4프레임 이상, 평균 모델 점수 ≥0.60 | 3음 경고 |
| 드론 확신(모델 추정) | 동일 위치 약 7프레임, 평균 ≥0.83, 충분한 영상상 크기 | 변화하는 높낮이의 사이렌+진동 |

- 이 점수는 확률 보정(calibration)된 확률이 아닙니다.
- 기존 YOLO26 단일 프레임의 `bird` 탐지는 새 날개짓 *패턴*을 분석하는 기능이 아닙니다. 영상 시간 분석은 아직 별도 미구현입니다.
- 미확인은 **아무 점이나 잡는 모드가 아닙니다**. 모델이 저신뢰 드론 후보로 출력한 것 중 위치 일관성이 있는 대상을 표시합니다.
- 모델 추론 FPS와 카메라 미리보기 FPS는 별개입니다. 실제 수치는 iPhone 현장 비교가 필요합니다.

## 쉬운 학습 흐름: Windows PC + Colab 무료/유료 GPU

1. 합법적 사용 조건이 확인된 새/드론 라벨의 YOLO 이미지 데이터셋을 ZIP으로 준비합니다. 예시 후보: https://universe.roboflow.com/workspace-ajksm/drone-detection-with-birds . 공개 데이터의 라이선스와 이미지 출처, train/valid/test 분할을 직접 확인하세요.
2. ZIP을 Google Drive의 **내 드라이브**에 올립니다.
3. **Colab 노트북**: https://colab.research.google.com/github/whgnltkd123-ai/yolo-ios-app/blob/main/colab/YOLO26_Bird_Drone_Train.ipynb 를 엽니다.
4. Colab GPU 런타임을 선택하고 노트북 셀을 순서대로 실행합니다. ZIP 경로를 입력한 뒤 YOLO26n을 재학습합니다. 학습에는 데이터양에 따라 수십 분~수 시간이 걸릴 수 있고 무료 GPU 사용량 제한이 있습니다.
5. 학습에 성공하면 Google Drive에 `YOLO26-Bird-Drone-best.pt`가 생성되고 SHA256 해시가 표시됩니다.
6. 자신의 GitHub 저장소 **Releases → Draft a new release**에서 태그(예: `yolo26-drone-v1`)를 만들고 학습한 .pt를 첨부해 공개 배포하거나 신뢰할 수 있는 HTTPS 파일 저장소를 사용하세요. **배포 권한이 없는 데이터 또는 가중치는 공개하지 마세요.** Codemagic 작업자가 다운받을 수 있는 URL이 필요합니다.

## Codemagic에서 새 YOLO26 iPhone IPA 생성

1. https://codemagic.io/app/6ad64a00a899de35292dfe3f 를 엽니다.
2. GitHub main에 이번 변경이 반영된 다음, **Start new build**를 누릅니다.
3. 워크플로 **`YOLO26 Drone Alert - 3 Levels (trained model required, unsigned)`** 선택.
4. Codemagic 환경변수 그룹 또는 워크플로 입력에 아래 2가지를 설정합니다.
   - `YOLO26_DRONE_CHECKPOINT_URL`: 직접 학습한 `best.pt`의 HTTPS 다운로드 URL
   - `YOLO26_DRONE_CHECKPOINT_SHA256`: Colab에서 출력된 SHA256 해시 64자리
   - 선택 `YOLO26_DRONE_IMGSZ`: `416`, `512`(기본값), `640`
5. 빌드 시작. 가중치가 없거나 클래스가 `bird/drone`이 아니라면 **의도적으로 빌드를 중지**해 일반 YOLO26을 드론 탐지 모델로 잘못 배포하는 일을 막습니다.
6. Artifacts의 `YOLO26-Drone-Alert-unsigned.ipa`를 Windows로 내려받아 Sideloadly로 다시 설치합니다.
7. 동일 장소·거리·아이폰에서 원본 YOLO26n과 학습 버전의 AI FPS, 새 오탐, 드론 미탐 및 3단계 경보를 검증합니다.

## 다음 정확도 개선 작업

- 쿼드/헥사/FPV/고정익, 고도·거리·역광·숲·도심별 정답 라벨 및 실패 예제 수집
- 빈 하늘, 나무·구름·새·벌레·전선·비행기 negative images 포함
- 데이터셋 분리는 프레임이 아니라 **비행 영상 단위**로 실시
- 새 날개짓은 여러 프레임의 실루엣 변화와 주기를 분석하는 별도 시간적 모델로 추후 실험
- 성능 지표: 드론 종류별 재현율, 새를 드론으로 오탐한 비율, 빈 하늘 시간당 경보 횟수, FPS 및 경보 지연

## 라이선스·안전

원본 Ultralytics 앱은 AGPL-3.0이며 학습 데이터/가중치의 사용 조건은 별도 확인이 필요합니다. 탐지·경보는 실험적 보조 수단이지 안전 또는 운용 보증이 아닙니다.
