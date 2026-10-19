> **2026-10-20 시험 파이프라인 업데이트**
>
> - 현재 앱은 새/드론 동시 클래스가 있어야만 경보를 허용하고, 별도 시험 워크플로 `bird-drone-prototype`를 추가했습니다.
> - Codemagic에서 `Bird vs Drone - Core ML Test (unsigned)` 워크플로를 **수동 실행**하면 고정된 공개 YOLOv8 3클래스 가중치를 다운로드하여 Git blob 해시로 검증한 뒤 Core ML로 변환하고 앱에 넣도록 시도합니다. 결과물 이름은 `YOLO-Bird-Drone-unsigned.ipa`입니다.
> - 이는 **이미 학습된 모델을 가져오는 시험용 변환**이며, 자체 데이터셋으로 새로 학습한 것이 아닙니다. 변환 또는 빌드 성공도 아직 확인되지 않았습니다. Python/패키지 설치, 외부 파일 다운로드 문제가 나면 빌드 로그 확인이 필요합니다.
> - 모델 출처: https://github.com/HarshaaMg/Flying_Object_Detection 의 `weights/refined_3class/best.pt`. 약 52MB. 출처 README에는 MIT라 쓰여 있지만 실제 LICENSE는 AGPL-3.0이라 라이선스 표기가 불일치합니다. 신뢰성/재배포 전에 확인하세요.
> - **보안:** 미확인 .pt 체크포인트는 역직렬화할 때 코드를 실행할 위험이 있어 서명 키 등 비밀이 없는 격리된 시험 환경에서만 변환해야 합니다.
> - 영상을 실제로 확보한 경우 `tools/evaluate_bird_drone_videos.py`에 새-only/드론-only/빈 하늘 영상을 넣으면 선택 프레임마다 새를 드론으로 잘못 분류하는 비율을 계산할 수 있습니다. 이 비율은 일반적인 mAP가 아닙니다.

# 새·드론 구분 모델: 학습/검증 계획

## 지금 무엇이 구현되었나?

- 앱의 YOLO 카메라와 기본 추론 기능은 유지합니다.
- 드론만 학습한 모델로는 새와의 구분을 보장할 수 없습니다. 따라서 현재 구현은 모델의 클래스 목록에 `bird`와 `drone`(또는 지정된 드론 클래스)이 모두 있어야 **음향 경보를 허용**합니다.
- `bird`는 초록색, 드론 후보는 빨강(신뢰도 >= 60%) 또는 노랑(그 미만)으로 표시합니다.
- 소리/진동 경보는 충분한 확신의 드론 후보가 대략 같은 위치에서 3프레임 연속 확인될 때만 울리고, 10초 쿨다운을 적용합니다.
- 실시간 프레임 추론의 결과를 처리할 뿐 **아직 새·드론 학습 모델이 앱에 포함된 것은 아닙니다.**
- `미확인`은 실제 추론기가 감지한 대상을 확정하지 못한 상태이며, 모든 점·구름·나무를 자동으로 미확인이라고 간주하지 않습니다.

## 추천 데이터셋 (두 클래스 모두 포함)

- Roboflow Universe, Drone-Detection with Birds:
  https://universe.roboflow.com/workspace-ajksm/drone-detection-with-birds
  총 약 11,987개 이미지가 등록되어 있고, `bird`, `drone` 2개 클래스로 소개됩니다. 다운로드/사용 조건은 해당 페이지를 확인하세요. 공개 라이선스 표시: CC BY 4.0.
- 비교용: Bird-Drone Dataset:
  https://universe.roboflow.com/dustin-king-a4hhv/bird-drone-dataset
- 별도의 실측/촬영 데이터셋 참조: https://pmc.ncbi.nlm.nih.gov/articles/PMC10440445/

## 실제 학습 (Colab GPU 또는 별도 GPU 머신)

1. Roboflow에서 공개 데이터셋을 YOLOv8 형식으로 다운로드합니다. `data.yaml` 파일과 `train`, `valid`, `test` 디렉터리가 있어야 합니다.
2. 데이터셋의 `names`에 반드시 정확하게 `bird`와 `drone`이 모두 존재하는지 확인합니다.
3. GPU가 있는 Python 환경에서:

   ```bash
   pip install "ultralytics>=8.3,<9" "coremltools>=9,<11" "numpy<=2.3.5" pyyaml
   python tools/train_bird_drone.py --data "/path/to/data.yaml" --epochs 50 --imgsz 640
   ```

4. 출력 `output/yolo8n-birddrone.mlpackage`를 확인합니다. Core ML 내보내기는 Windows에 직접 지원되지 않지만, Colab Linux 또는 Codemagic macOS 환경에서 사용할 수 있습니다.
5. 검증되지 않은 가중치를 즉시 정식 앱에 배포하지 않습니다. 별도 보류 테스트 동영상으로 새를 드론으로 분류하는 빈도(오탐), 실제 드론을 놓치는 빈도(미탐), 이미지당 탐지 시간(FPS)을 기록합니다.
6. 모델이 검증되면 `YOLOiOSApp/Models/Detect/yolo8n-birddrone.mlpackage`로 번들링하고 다시 Codemagic iOS 빌드를 수행합니다. 이 작업은 **아직 완료되지 않았습니다.**

### 중요한 데이터 분리

동일 비행 영상의 연속 프레임이 학습셋과 테스트셋에 동시에 들어가면 성능이 부풀려집니다. 영상 단위로 train/val/test를 나눠야 합니다. 특히 멀리 점처럼 보이는 새와 드론, 구름·나무·가로등·비행기를 대상으로 별도 실패 사례를 확보해야 합니다.

### 라이선스

원본 YOLO 프로젝트의 AGPL-3.0을 준수하고, 외부 데이터셋의 CC BY 4.0 저작자 표시 요건을 검토합니다. 공개된 mAP 수치는 사용자의 아이폰 및 실제 거리에서의 성능을 보증하지 않습니다.
