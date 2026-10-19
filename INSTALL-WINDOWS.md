# Windows + 무료 Apple ID: YOLO Drone Alert 아이폰 설치

이 저장소는 [Ultralytics YOLO iOS](https://github.com/ultralytics/yolo-ios-app)를 기반으로 한 개인 테스트용 Fork입니다. 원본의 AGPL-3.0 라이선스 조건을 준수해야 합니다.

## 현재 구현

- 기존 실시간 카메라, YOLO 추론, 박스, 민감도 조절은 변경하지 않았습니다.
- 카메라 화면에 `드론 경보 ON/OFF` 버튼과 상태 표시를 추가했습니다.
- Detect 모델이 **drone, uav, quadcopter, multicopter, fpv drone** 등 전용 객체 클래스를 제공할 때만 경보합니다.
- YOLO 신뢰도가 **60% 이상**이며 현재 Confidence 슬라이더 설정 이상인 결과가 **3프레임 연속** 나오면 소리/진동을 호출합니다. 반복 경보 최소 간격은 10초입니다.
- **주의:** 기본 범용 YOLO/COCO 모델에는 보통 드론 전용 클래스가 없습니다. 이를 선택한 경우 화면에 "드론 전용 클래스가 없는 모델"이 표시되고 드론 경보가 울리지 않습니다. 실제 드론 인식에는 별도 드론 학습 모델을 추가해야 합니다.
- 분류 결과만으로 드론 여부를 확정하지 마십시오. 실사용 전 오탐/미탐 검증이 필요합니다.

## 빌드: Codemagic

1. Codemagic에서 이 저장소의 `ios-build-check` 워크플로를 실행합니다.
2. 상태가 성공이면 Artifacts에서 **YOLOiOSApp-unsigned.ipa** 파일을 다운로드합니다.
3. 이 IPA는 **서명 전 파일**입니다. 아이폰에서 바로 실행할 수 없습니다.

## Windows에서 개인 아이폰에 설치

1. [Sideloadly 공식 사이트](https://sideloadly.io/)에서 Windows 버전을 설치합니다.
2. Sideloadly 안내에 따라 Apple 공식 홈페이지의 **데스크톱용 iTunes 및 iCloud**를 설치합니다. Microsoft Store판은 문제가 될 수 있습니다.
3. USB 케이블로 본인 아이폰을 Windows PC에 연결하고 기기에서 '이 컴퓨터를 신뢰'를 선택합니다.
4. Sideloadly에 Codemagic에서 받은 `YOLOiOSApp-unsigned.ipa`를 끌어 놓습니다.
5. 본인의 무료 Apple ID로 앱에 개발자 서명을 적용한 다음 설치를 시작합니다. **Apple ID 비밀번호는 GitHub, Codemagic, 채팅창에 올리지 마세요.** 타사 서명 도구 이용 시 계정 보안에 주의하십시오.
6. 설치된 앱이 열리지 않으면 iPhone **설정 > 일반 > VPN 및 기기 관리**에서 해당 개발자 앱을 신뢰하고, 필요 시 **설정 > 개인정보 보호 및 보안 > 개발자 모드**를 켜고 재시동합니다.
7. 카메라 사용 권한을 허용합니다.

무료 Apple ID로 설치한 앱은 일반적으로 7일 뒤 개발용 서명이 만료되므로 **주기적으로 재서명/갱신**해야 합니다. Apple 정책과 iOS 버전에 따른 추가 제한이나 오류가 생길 수 있습니다.

## 참고 문서

- [Apple 무료 개인 팀 제한](https://developer.apple.com/help/account/basics/about-your-developer-account)
- [Codemagic iOS 서명 요건](https://docs.codemagic.io/yaml-code-signing/signing-ios/)
- [Sideloadly 다운로드/FAQ](https://sideloadly.io/)
- [Apple 개발자 모드](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device)

## 다음 개발 단계

드론 클래스가 포함된, 실제 검증 가능한 Core ML 객체인식 모델을 확보하고 `YOLOiOSApp/Models/Detect` 또는 앱의 원격 모델 목록에 등록합니다. 현재 원본 엔진으로 드론을 자동 학습하거나 인터넷 사진을 자체 학습하는 기능은 들어 있지 않습니다.
