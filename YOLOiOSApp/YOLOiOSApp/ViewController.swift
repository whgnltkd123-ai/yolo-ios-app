// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and provides the main user interface for model selection and
//  visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController is the primary interface for users to interact with YOLO models. Users can select different
//  models and tasks (detection, segmentation, semantic segmentation, depth, classification, pose, OBB) and visualize
//  results in real-time. The controller manages loading of local and remote models, updates UI state during loading and
//  inference, and handles capturing and sharing detection results. It also tracks model download progress and adapts
//  the layout to different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import UltralyticsYOLO

let appTasks: [(name: String, shortName: String, folder: String, yoloTask: YOLOTask)] = [
  ("Detect", "Det", "Models/Detect", .detect),
  ("Segment", "Seg", "Models/Segment", .segment),
  ("Semantic", "Sem", "Models/Semantic", .semantic),
  ("Depth", "Depth", "Models/Depth", .depth),
  ("Classify", "Cls", "Models/Classify", .classify),
  ("Pose", "Pose", "Models/Pose", .pose),
  ("OBB", "OBB", "Models/OBB", .obb),
]

/// Locally generated unknown/drone alert sounds. No bundled audio assets or network needed.
private final class DroneAlertSoundPlayer {
  private var currentPlayer: AVAudioPlayer?
  private var cachedTones: [Int: Data] = [:]
  private var currentStage: Int?
  private var stopWhenResultsExpire: DispatchWorkItem?

  deinit {
    stopWhenResultsExpire?.cancel()
    currentPlayer?.stop()
  }

  func stop() {
    stopWhenResultsExpire?.cancel()
    stopWhenResultsExpire = nil
    currentPlayer?.stop()
    currentPlayer = nil
    currentStage = nil
  }

  func play(stage: Int) {
    guard stage == 1 || stage == 3 else { return }
    // A loop must expire if camera results stop arriving.
    stopWhenResultsExpire?.cancel()
    let expiry = DispatchWorkItem { [weak self] in self?.stop() }
    stopWhenResultsExpire = expiry
    DispatchQueue.main.asyncAfter(deadline: .now() + 2, execute: expiry)
    if currentStage == stage && currentPlayer?.isPlaying == true { return }
    let wav = cachedTones[stage] ?? Self.makeTone(stage: stage)
    cachedTones[stage] = wav
    do {
      // The user explicitly enabled alerts: play through the speaker even in silent mode.
      try AVAudioSession.sharedInstance().setCategory(
        .playback, mode: .default, options: [.mixWithOthers])
      try AVAudioSession.sharedInstance().setActive(true)
      currentPlayer?.stop()
      currentPlayer = try AVAudioPlayer(data: wav)
      currentPlayer?.volume = 1.0
      currentPlayer?.numberOfLoops = -1
      currentStage = stage
      currentPlayer?.prepareToPlay()
      currentPlayer?.play()
    } catch {
      stop()
      print("Drone alert audio unavailable: \(error)")
    }
  }

  private static func makeTone(stage: Int) -> Data {
    let sampleRate = 22_050
    let duration = stage == 1 ? 0.48 : 1.6
    let count = Int(Double(sampleRate) * duration)
    // Keep a one-second silent gap between repeated alert patterns.
    let silenceCount = sampleRate
    let byteCount = (count + silenceCount) * 2
    var wav = Data(capacity: 44 + byteCount)
    func ascii(_ string: String) { wav.append(contentsOf: string.utf8) }
    func le16(_ value: UInt16) {
      wav.append(UInt8(truncatingIfNeeded: value))
      wav.append(UInt8(truncatingIfNeeded: value >> 8))
    }
    func le32(_ value: UInt32) {
      wav.append(UInt8(truncatingIfNeeded: value))
      wav.append(UInt8(truncatingIfNeeded: value >> 8))
      wav.append(UInt8(truncatingIfNeeded: value >> 16))
      wav.append(UInt8(truncatingIfNeeded: value >> 24))
    }
    ascii("RIFF")
    le32(UInt32(36 + byteCount))
    ascii("WAVEfmt ")
    le32(16)
    le16(1)  // mono PCM
    le16(1)
    le32(UInt32(sampleRate))
    le32(UInt32(sampleRate * 2))
    le16(2)
    le16(16)
    ascii("data")
    le32(UInt32(byteCount))

    var phase = 0.0
    for index in 0..<count {
      let t = Double(index) / Double(sampleRate)
      let frequency: Double
      let gate: Double
      switch stage {
      case 1:  // Unknown: two short pulses
        frequency = 630
        gate = (t < 0.13 || (t > 0.25 && t < 0.38)) ? 0.55 : 0
      default:  // High-confidence model indication: alternating siren
        frequency = 950 + 390 * sin(2 * .pi * 2.5 * t)
        gate = 0.80
      }
      phase += 2 * .pi * frequency / Double(sampleRate)
      let fade = min(1.0, min(t * 50, (duration - t) * 50))
      let sample = Int16(max(-1, min(1, sin(phase) * gate * max(0, fade))) * 32767)
      le16(UInt16(bitPattern: sample))
    }
    wav.append(Data(count: silenceCount * 2))
    return wav
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  // MARK: - External Display Support (Optional)
  // External display features remain dormant until a display is connected. See ExternalDisplay/ for details.

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet weak var View0: UIView!
  @IBOutlet weak var segmentedControl: UISegmentedControl!
  @IBOutlet weak var modelSegmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  /// Latest "x FPS - y ms" line and pre/inference/post breakdown, rendered together in `labelFPS`.
  private var lastPerformanceLabelDraw: CFTimeInterval = 0
  private var fpsText = ""
  private var breakdownText = ""
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var logoImage: UIImageView!

  let selection = UISelectionFeedbackGenerator()

  // Tracks the currently loading model for external display notification.
  var currentLoadingEntry: ModelEntry?

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()
  private let cancelLoadingButton = UIButton(type: .system)

  private var loadingOverlayView: UIView?

  // MARK: - Stable bird/drone warning levels (experimental model estimates, not identity proof)
  private enum DroneAlertStage: Int, Hashable {
    case none = 0
    case high = 3
  }

  private let aerialBoxesButton = UIButton(type: .system)
  private let droneAlertButton = UIButton(type: .system)
  private let droneAlertStatusLabel = UILabel()
  private let droneAlertAudio = DroneAlertSoundPlayer()
  private var droneAlertEnabled = true
  private var previousAlertStage: DroneAlertStage = .none
  private let monitoringButton = UIButton(type: .system)
  private let monitoringStatusLabel = UILabel()
  private let evidenceStore = AerialEvidenceStore()
  private var evidenceEnabled = UserDefaults.standard.bool(forKey: "aerial.evidence")
  private var broadcastEnabled = false

  private func installDroneAlertControls() {
    let panel = UIStackView(arrangedSubviews: [
      aerialBoxesButton, droneAlertButton,
      monitoringButton, droneAlertStatusLabel, monitoringStatusLabel,
    ])
    panel.axis = .vertical
    panel.alignment = .leading
    panel.spacing = 5
    panel.translatesAutoresizingMaskIntoConstraints = false

    aerialBoxesButton.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    aerialBoxesButton.setTitleColor(.white, for: .normal)
    aerialBoxesButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
    aerialBoxesButton.layer.cornerRadius = 8
    aerialBoxesButton.contentEdgeInsets = UIEdgeInsets(top: 7, left: 10, bottom: 7, right: 10)
    aerialBoxesButton.addTarget(self, action: #selector(toggleAerialBoxes), for: .touchUpInside)
    updateAerialBoxesButton()

    droneAlertButton.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    droneAlertButton.setTitleColor(.white, for: .normal)
    droneAlertButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
    droneAlertButton.layer.cornerRadius = 8
    droneAlertButton.contentEdgeInsets = UIEdgeInsets(top: 7, left: 10, bottom: 7, right: 10)
    droneAlertButton.addTarget(self, action: #selector(toggleDroneAlert), for: .touchUpInside)
    updateDroneAlertButton()

    droneAlertStatusLabel.text = "드론·새·비행기 모델을 확인하는 중"
    droneAlertStatusLabel.textColor = .white
    droneAlertStatusLabel.font = .systemFont(ofSize: 12, weight: .medium)
    droneAlertStatusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    droneAlertStatusLabel.layer.cornerRadius = 6
    droneAlertStatusLabel.clipsToBounds = true
    droneAlertStatusLabel.numberOfLines = 2
    monitoringButton.setTitle("채증 · 가입자 알림", for: .normal)
    monitoringButton.setTitleColor(.white, for: .normal)
    monitoringButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
    monitoringButton.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    monitoringButton.layer.cornerRadius = 8
    monitoringButton.contentEdgeInsets = UIEdgeInsets(top: 7, left: 10, bottom: 7, right: 10)
    monitoringButton.addTarget(self, action: #selector(showMonitoringSettings), for: .touchUpInside)
    monitoringStatusLabel.textColor = .white
    monitoringStatusLabel.font = .systemFont(ofSize: 11)
    monitoringStatusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    monitoringStatusLabel.numberOfLines = 2
    RemoteDroneAlerts.shared.statusChanged = { [weak self] text in
      self?.monitoringStatusLabel.text = text
    }
    NotificationCenter.default.addObserver(
      self, selector: #selector(clearAerialAlerts),
      name: UIApplication.willResignActiveNotification, object: nil)
    applyMonitoringSettings()

    yoloView.addSubview(panel)
    NSLayoutConstraint.activate([
      panel.leadingAnchor.constraint(equalTo: yoloView.leadingAnchor, constant: 12),
      panel.topAnchor.constraint(equalTo: yoloView.safeAreaLayoutGuide.topAnchor, constant: 12),
      panel.widthAnchor.constraint(lessThanOrEqualTo: yoloView.widthAnchor, multiplier: 0.84),
    ])
  }

  private func updateAerialBoxesButton() {
    let aerialOnly = yoloView.showsOnlyAerialDetections
    aerialBoxesButton.setTitle(
      aerialOnly ? "공중 객체만 표시" : "모든 객체 표시", for: .normal)
    aerialBoxesButton.accessibilityLabel =
      aerialOnly
      ? "공중 객체만 표시 중. 전체 객체 표시하기"
      : "모든 객체 표시 중. 공중 객체만 표시하기"
  }

  @objc private func toggleAerialBoxes() {
    yoloView.showsOnlyAerialDetections.toggle()
    updateAerialBoxesButton()
  }

  private func updateDroneAlertButton() {
    droneAlertButton.setTitle(droneAlertEnabled ? "경보 켜짐" : "경보 꺼짐", for: .normal)
    droneAlertButton.accessibilityLabel = droneAlertEnabled ? "경보 끄기" : "경보 켜기"
  }

  private func resetDroneCandidate() {
    droneAlertAudio.stop()
    previousAlertStage = .none
  }

  @objc private func toggleDroneAlert() {
    droneAlertEnabled.toggle()
    applyMonitoringSettings()
    if !droneAlertEnabled { droneAlertAudio.stop() }
    resetDroneCandidate()
    updateDroneAlertButton()
    droneAlertStatusLabel.text = droneAlertEnabled ? "드론·새·비행기 모델 확인 중" : "경보 꺼짐"
  }

  private func applyMonitoringSettings() {
    // Only opt-in evidence capture requires a full-resolution frame copy.
    yoloView.capturesOriginalImage = evidenceEnabled
    UserDefaults.standard.set(evidenceEnabled, forKey: "aerial.evidence")
    monitoringStatusLabel.text = "자동 채증 \(evidenceEnabled ? "ON" : "OFF") · 전송 \(broadcastEnabled ? "ON" : "OFF")"
  }

  @objc private func showMonitoringSettings() {
    let menu = UIAlertController(
      title: "채증 · 가입자 알림",
      message:
        "자동 채증은 앱 내 갤러리에만 보관합니다(최대 500MB). 사진 앱에는 선택한 자료만 저장할 수 있습니다. 가입자에게는 사진·위치 없이 드론 식별 알림만 전송합니다.\n\(RemoteDroneAlerts.shared.status)",
      preferredStyle: .actionSheet)
    menu.addAction(
      UIAlertAction(title: evidenceEnabled ? "자동 채증 끄기" : "자동 채증 켜기", style: .default) {
        [weak self] _ in
        guard let self else { return }
        evidenceEnabled.toggle()
        applyMonitoringSettings()
      })
    menu.addAction(
      UIAlertAction(title: "채증 갤러리", style: .default) { [weak self] _ in
        guard let self else { return }
        let gallery = UINavigationController(
          rootViewController: EvidenceGalleryController(store: evidenceStore))
        gallery.modalPresentationStyle = .fullScreen
        present(gallery, animated: true)
      })
    if RemoteDroneAlerts.shared.connected {
      let broadcast = UIAlertAction(
        title: broadcastEnabled ? "가입자에게 전송 끄기" : "가입자에게 전송 켜기", style: .default
      ) { [weak self] _ in
        guard let self else { return }
        broadcastEnabled.toggle()
        applyMonitoringSettings()
      }
      broadcast.isEnabled = RemoteDroneAlerts.shared.canPublish
      menu.addAction(broadcast)
      menu.addAction(
        UIAlertAction(title: "알림 구독 해제", style: .destructive) { [weak self] _ in
          Task { @MainActor in
            do {
              try await RemoteDroneAlerts.shared.leave()
              self?.broadcastEnabled = false
              self?.applyMonitoringSettings()
            } catch { self?.showMonitoringMessage(error.localizedDescription) }
          }
        })
    } else {
      menu.addAction(
        UIAlertAction(title: "초대코드로 알림 가입", style: .default) { [weak self] _ in
          self?.joinRemoteAlerts()
        })
    }
    menu.addAction(UIAlertAction(title: "닫기", style: .cancel))
    menu.popoverPresentationController?.sourceView = monitoringButton
    menu.popoverPresentationController?.sourceRect = monitoringButton.bounds
    present(menu, animated: true)
  }

  private func joinRemoteAlerts() {
    let form = UIAlertController(
      title: "가입자 알림 연결",
      message: "운영자가 제공한 서버 주소와 초대코드를 입력하세요. 연결 시 기기 알림 주소가 서버에 등록됩니다.", preferredStyle: .alert)
    form.addTextField {
      $0.placeholder = "https://알림서버주소"
      $0.keyboardType = .URL
      $0.autocapitalizationType = .none
    }
    form.addTextField {
      $0.placeholder = "초대코드"
      $0.isSecureTextEntry = true
      $0.autocapitalizationType = .none
    }
    form.addAction(UIAlertAction(title: "취소", style: .cancel))
    form.addAction(
      UIAlertAction(title: "가입", style: .default) { [weak self, weak form] _ in
        guard let fields = form?.textFields else { return }
        let server = fields[0].text ?? ""
        let code = fields[1].text ?? ""
        Task { @MainActor in
          do { try await RemoteDroneAlerts.shared.join(serverText: server, invite: code) } catch {
            self?.showMonitoringMessage(error.localizedDescription)
          }
        }
      })
    present(form, animated: true)
  }

  private func showMonitoringMessage(_ text: String) {
    let alert = UIAlertController(title: "채증 · 알림", message: text, preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "확인", style: .default))
    present(alert, animated: true)
  }

  @objc private func clearAerialAlerts() {
    resetDroneCandidate()
  }

  private func processAerialFeatures(result: YOLOResult) {
    guard currentTask == "Detect", UIApplication.shared.applicationState == .active,
      presentedViewController == nil, viewIfLoaded?.window != nil else { return }
    if evidenceEnabled {
      evidenceStore.record(result: result, model: currentModelName) { [weak self] outcome in
        switch outcome {
        case .success: self?.monitoringStatusLabel.text = "채증 완료 · 앱 내 갤러리에 보관"
        case .failure(let error):
          self?.evidenceEnabled = false
          self?.applyMonitoringSettings()
          self?.monitoringStatusLabel.text = error.localizedDescription
        }
      }
    }
    if broadcastEnabled, let drone = result.boxes.first(where: { $0.aerialDisplayLabel == "드론" }) {
      RemoteDroneAlerts.shared.publish(score: drone.conf)
    }
  }

  private func isBirdClass(_ value: String) -> Bool {
    ["bird", "birds"].contains(value.lowercased())
  }

  private func isDroneClass(_ value: String) -> Bool {
    ["drone", "uav", "drones"].contains(value.lowercased())
  }

  private func soundForStableStage(_ stage: DroneAlertStage) {
    guard stage != .none else { return }
    droneAlertAudio.play(stage: stage.rawValue)
    guard stage != previousAlertStage else { return }
    switch stage {
    case .high:
      UINotificationFeedbackGenerator().notificationOccurred(.error)
      AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    case .none: break
    }
  }

  private func processDroneAlert(result: YOLOResult) {
    guard droneAlertEnabled else { return }
    guard UIApplication.shared.applicationState == .active, presentedViewController == nil,
      viewIfLoaded?.window != nil
    else {
      resetDroneCandidate()
      return
    }
    guard currentTask == "Detect" else {
      resetDroneCandidate()
      droneAlertStatusLabel.text = "탐지(Detect) 모델을 선택하세요"
      return
    }

    // A generic COCO model cannot issue drone alerts.
    guard result.names.contains(where: isBirdClass), result.names.contains(where: isDroneClass)
    else {
      resetDroneCandidate()
      droneAlertStatusLabel.text = "새·드론 구분 모델을 선택하세요"
      return
    }
    let high = result.boxes.first { $0.aerialDisplayLabel == "드론" }
    let stage: DroneAlertStage = high != nil ? .high : .none
    if stage == .none { droneAlertAudio.stop() } else { soundForStableStage(stage) }
    previousAlertStage = stage
    droneAlertStatusLabel.text = high.map { "드론 \(Int($0.conf * 100))점" }
      ?? result.boxes.first(where: { $0.aerialDisplayLabel != nil })?.aerialDisplayLabel
      ?? "감시 중 · 현재 탐지 없음"
  }

  // MARK: - Constants
  private struct Constants {
    static let defaultTask = YOLOTask.detect
    static let progressViewWidth: CGFloat = 200
  }

  // MARK: - Loading State Management
  private func setLoadingState(_ loading: Bool, showOverlay: Bool = false, canCancel: Bool = false)
  {
    loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
    if showOverlay && loading { updateLoadingOverlay(true) }
    if !loading { updateLoadingOverlay(false) }
    cancelLoadingButton.isHidden = !(loading && canCancel)
  }

  private func updateLoadingOverlay(_ show: Bool) {
    if show && loadingOverlayView == nil {
      let overlay = UIView(frame: view.bounds)
      overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
      overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
      view.addSubview(overlay)
      loadingOverlayView = overlay
      view.bringSubviewToFront(activityIndicator)
      view.bringSubviewToFront(downloadProgressView)
      view.bringSubviewToFront(downloadProgressLabel)
      view.bringSubviewToFront(cancelLoadingButton)
    } else if !show {
      loadingOverlayView?.removeFromSuperview()
      loadingOverlayView = nil
    }
  }

  private var modelsForTask: [String: [String]] = [:]

  var currentModels: [ModelEntry] = []
  private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] =
    [:]

  var currentTask: String = ""
  var currentModelName: String = ""

  private var isLoadingModel = false
  private var currentLoadingTask: String?

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    clearAerialAlerts()
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    // Optional external display setup — dormant unless a monitor/TV is connected. See ExternalDisplay/ for details.
    setupExternalDisplayNotifications()
    if hasExternalScreen() {
      yoloView.isHidden = true
    }

    // Populate the task segmented control and discover bundled models per task.
    segmentedControl.removeAllSegments()
    appTasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.shortName, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }
    setupTaskSegmentedControl()

    setupModelSegmentedControl()

    let defaultTaskIndex = appTasks.firstIndex(where: { $0.yoloTask == Constants.defaultTask }) ?? 0
    segmentedControl.selectedSegmentIndex = defaultTaskIndex
    currentTask = appTasks[defaultTaskIndex].name

    // Always load models initially; external display handling will stop the camera if needed.
    reloadModelEntriesAndLoadFirst(for: currentTask)

    // Returning from Settings with the Core AI setting changed: relist models in the new format and load the current
    // task's first model in it. A change made while a model loads is picked up on the next return to the foreground.
    NotificationCenter.default.addObserver(
      forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main
    ) { [weak self] _ in
      guard let self = self, !self.isLoadingModel,
        preferredModelExtension != remoteModelExtension
      else { return }
      remoteModelExtension = preferredModelExtension
      appTasks.forEach { self.modelsForTask[$0.name] = self.getModelFiles(in: $0.folder) }
      self.reloadModelEntriesAndLoadFirst(for: self.currentTask)
    }

    // Wire up gestures and delegates.
    logoImage.isHidden = true
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    installDroneAlertControls()
    [yoloView.labelName, yoloView.labelFPS, yoloView.labelBreakdown].forEach { $0?.isHidden = true }
    [labelName, labelFPS, labelVersion].forEach { $0?.isHidden = true }

    // Observe slider changes to forward thresholds to an external display.
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)

    // Style labels and stamp the app version.
    [labelName, labelFPS, labelVersion].forEach {
      $0?.textColor = .white
      $0?.overrideUserInterfaceStyle = .dark
    }
    labelFPS.numberOfLines = 2
    // Clear the storyboard placeholder text and reserve the final two-line height immediately,
    // so the first inference result doesn't shift the layout below.
    labelName.text = ""
    fpsText = " "
    breakdownText = " "
    renderPerformanceLabel()
    activityIndicator.style = .large
    activityIndicator.color = .white
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    {
      labelVersion.text = "v\(version) (\(build))"
    }

    // Install the download progress views.
    [downloadProgressView, downloadProgressLabel, cancelLoadingButton].forEach {
      $0.isHidden = true
      $0.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview($0)
    }
    downloadProgressLabel.textAlignment = .center
    downloadProgressLabel.textColor = .white
    downloadProgressLabel.font = .systemFont(ofSize: 15, weight: .medium)
    downloadProgressView.progressTintColor = .white
    downloadProgressView.trackTintColor = UIColor.white.withAlphaComponent(0.25)
    var cancelButtonConfiguration = UIButton.Configuration.tinted()
    cancelButtonConfiguration.title = "Cancel"
    cancelButtonConfiguration.baseForegroundColor = .white
    cancelButtonConfiguration.baseBackgroundColor = .white
    cancelButtonConfiguration.contentInsets = NSDirectionalEdgeInsets(
      top: 8, leading: 16, bottom: 8, trailing: 16)
    cancelLoadingButton.configuration = cancelButtonConfiguration
    cancelLoadingButton.addTarget(self, action: #selector(cancelLoadingModel), for: .touchUpInside)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: Constants.progressViewWidth),
      downloadProgressView.heightAnchor.constraint(equalToConstant: 3),
      downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      downloadProgressLabel.topAnchor.constraint(
        equalTo: downloadProgressView.bottomAnchor, constant: 8),
      cancelLoadingButton.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      cancelLoadingButton.topAnchor.constraint(
        equalTo: downloadProgressLabel.bottomAnchor, constant: 10),
    ])

    ModelDownloadManager.shared.progressHandler = { [weak self] progress in
      guard let self = self else { return }
      DispatchQueue.main.async {
        self.downloadProgressView.progress = Float(progress)
        self.downloadProgressLabel.isHidden = false
        let percentage = Int(progress * 100)
        self.downloadProgressLabel.text = "Downloading \(percentage)%"
      }
    }
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    view.overrideUserInterfaceStyle = .dark
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(
        at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
      )
    else { return [] }

    let modelFiles =
      fileURLs
      .filter {
        // With the Core AI setting on, the app lists Core AI models only: bundled `.aimodel` and remote `.aimodel.zip`.
        (remoteModelExtension == "aimodel" ? ["aimodel"] : ["mlmodel", "mlpackage"])
          .contains($0.pathExtension)
      }
      .map { $0.lastPathComponent }

    return folderName == "Models/Detect" ? reorderDetectionModels(modelFiles) : modelFiles.sorted()
  }

  private func reorderDetectionModels(_ fileNames: [String]) -> [String] {
    let order: [Character: Int] = ["n": 0, "s": 1, "m": 2, "l": 3, "x": 4]
    let (official, custom) = fileNames.reduce(into: ([String](), [String]())) { result, name in
      let base = (name as NSString).deletingPathExtension.lowercased()
      base.hasPrefix("yolo") && order[base.last ?? "z"] != nil
        ? result.0.append(name) : result.1.append(name)
    }
    return custom.sorted()
      + official.sorted {
        order[($0 as NSString).deletingPathExtension.lowercased().last ?? "z"] ?? 99 < order[
          ($1 as NSString).deletingPathExtension.lowercased().last ?? "z"] ?? 99
      }
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    showModelEntries(for: taskName)

    // Keep the original model-loading pipeline; use the user-selected small architecture.
    let preferredSize: ModelSelectionManager.ModelSize = taskName == "Detect" ? .s : .n
    if let model = standardModels[preferredSize] {
      modelSegmentedControl.selectedSegmentIndex =
        ModelSelectionManager.ModelSize.allCases.firstIndex(of: preferredSize) ?? 0
      let entry = ModelEntry(
        displayName: (model.name as NSString).deletingPathExtension,
        identifier: model.name,
        isLocalBundle: model.isLocal,
        remoteURL: model.url
      )
      loadModel(entry: entry, forTask: taskName)
    }
  }

  private func showModelEntries(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)
    let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
    standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)

    let yoloTask = appTasks.first(where: { $0.name == taskName })?.yoloTask ?? .detect
    ModelSelectionManager.setupSegmentedControl(
      modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)
  }

  private func restoreCurrentModelSelection() {
    guard !currentTask.isEmpty else { return }

    if let taskIndex = appTasks.firstIndex(where: { $0.name == currentTask }) {
      segmentedControl.selectedSegmentIndex = taskIndex
    }

    showModelEntries(for: currentTask)

    if let modelIndex = ModelSelectionManager.ModelSize.allCases.firstIndex(where: { size in
      guard let model = standardModels[size] else { return false }
      return processString((model.name as NSString).deletingPathExtension) == currentModelName
    }) {
      modelSegmentedControl.selectedSegmentIndex = modelIndex
    }

    let yoloTask = appTasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect
    ModelSelectionManager.updateSegmentAppearance(
      modelSegmentedControl,
      standardModels: standardModels,
      currentTask: yoloTask
    )

    if !currentModelName.isEmpty {
      labelName.text = currentModelName
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    let localEntries = localFileNames.map { fileName -> ModelEntry in
      ModelEntry(
        displayName: (fileName as NSString).deletingPathExtension,
        identifier: fileName,
        isLocalBundle: true,
        remoteURL: nil
      )
    }

    let localModelNames = Set(localEntries.map { $0.displayName.lowercased() })

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.compactMap { (modelName, url) -> ModelEntry? in
      guard !localModelNames.contains(modelName.lowercased()) else { return nil }
      return ModelEntry(
        displayName: modelName,
        identifier: modelName,
        isLocalBundle: false,
        remoteURL: url
      )
    }

    return localEntries + remoteEntries
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      print("Model is already loading. Please wait.")
      return
    }
    isLoadingModel = true

    let needsDownload =
      !entry.isLocalBundle && !ModelCacheManager.shared.isModelDownloaded(key: entry.identifier)

    // Skip the local YOLOView reset when an external display owns the camera.
    let hasExternalDisplay = hasExternalScreen()
    if !hasExternalDisplay && !needsDownload {
      yoloView.resetLayers()
      yoloView.setInferenceFlag(ok: false)
    }

    resetDownloadProgress()
    setLoadingState(true, showOverlay: true, canCancel: needsDownload)
    currentLoadingEntry = entry
    currentLoadingTask = task

    let yoloTask = appTasks.first(where: { $0.name == task })?.yoloTask ?? .detect

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }

        guard let folderURL = appTasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          DispatchQueue.main.async { [weak self] in
            self?.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        DispatchQueue.main.async { [weak self] in
          guard let self = self else { return }
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"

          // External display path: notify it instead of loading into the local YOLOView.
          if self.hasExternalScreen() {
            self.finishLoadingModel(success: true, modelName: entry.displayName)
          } else {
            self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) {
              [weak self] result in
              DispatchQueue.main.async {
                switch result {
                case .success():
                  self?.finishLoadingModel(success: true, modelName: entry.displayName)
                case .failure(let error):
                  print(error)
                  self?.finishLoadingModel(success: false, modelName: entry.displayName)
                }
              }
            }
          }
        }
      }
    } else {
      let key = entry.identifier  // e.g. "yolo26n", "yolo26m-seg"

      if ModelCacheManager.shared.isModelDownloaded(key: key) {
        loadCachedModelAndSetToYOLOView(
          key: key, yoloTask: yoloTask, displayName: entry.displayName)
      } else {
        guard let remoteURL = entry.remoteURL else {
          self.finishLoadingModel(success: false, modelName: entry.displayName)
          return
        }

        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false

        // Show the initial download message with a properly formatted model name.
        self.downloadProgressLabel.text = "Downloading \(processString(entry.displayName))"

        let localZipFileName = remoteURL.lastPathComponent  // e.g. "yolo26n.mlpackage.zip"

        ModelCacheManager.shared.loadModel(
          from: localZipFileName,
          remoteURL: remoteURL,
          key: key
        ) { [weak self] success, loadedKey in
          guard let self = self else { return }
          if !success {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
            return
          }
          self.loadCachedModelAndSetToYOLOView(
            key: loadedKey,
            yoloTask: yoloTask,
            displayName: entry.displayName)
        }
      }
    }
  }

  private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String)
  {
    let localModelURL = ModelCacheManager.shared.modelURL(for: key)

    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"

      // External display path: notify it instead of loading into the local YOLOView.
      if self.hasExternalScreen() {
        self.finishLoadingModel(success: true, modelName: displayName)
      } else {
        self.yoloView.resetLayers()
        self.yoloView.setInferenceFlag(ok: false)
        self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) {
          [weak self] result in
          DispatchQueue.main.async {
            switch result {
            case .success():
              self?.finishLoadingModel(success: true, modelName: displayName)
            case .failure(let error):
              print(error)
              self?.finishLoadingModel(success: false, modelName: displayName)
            }
          }
        }
      }
    }
  }

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    downloadProgressLabel.text = ""
    [downloadProgressView, downloadProgressLabel, cancelLoadingButton].forEach {
      $0.isHidden = true
    }
  }

  @objc private func cancelLoadingModel() {
    guard isLoadingModel, let entry = currentLoadingEntry else { return }

    if !entry.isLocalBundle {
      ModelDownloadManager.shared.cancelDownload(key: entry.identifier)
    }

    setLoadingState(false)
    isLoadingModel = false
    currentLoadingEntry = nil
    currentLoadingTask = nil
    resetDownloadProgress()
    restoreCurrentModelSelection()

    if !hasExternalScreen(), !currentModelName.isEmpty {
      yoloView.setInferenceFlag(ok: true)
    }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.setLoadingState(false)
      self.isLoadingModel = false
      self.resetDownloadProgress()
      let loadedTask = self.currentLoadingTask ?? self.currentTask
      let hasExternalDisplay = self.hasExternalScreen()
      let keptCurrentModel =
        !success && !self.currentModelName.isEmpty
        && (self.currentLoadingEntry.map {
          !$0.isLocalBundle && !ModelCacheManager.shared.isModelDownloaded(key: $0.identifier)
        } == true)

      if success {
        self.yoloView.capturesOriginalImage = self.droneAlertEnabled && loadedTask == "Detect"
        let previousTask = self.currentTask
        self.currentTask = loadedTask
        if let taskIndex = appTasks.firstIndex(where: { $0.name == loadedTask }) {
          self.segmentedControl.selectedSegmentIndex = taskIndex
        }
        if previousTask != loadedTask {
          NotificationCenter.default.post(
            name: .taskDidChange,
            object: nil,
            userInfo: ["task": loadedTask]
          )
        }
        let yoloTask = appTasks.first(where: { $0.name == loadedTask })?.yoloTask ?? .detect
        self.currentModelName = processString(modelName)

        ModelSelectionManager.setupSegmentedControl(
          self.modelSegmentedControl,
          standardModels: self.standardModels,
          currentTask: yoloTask,
          preserveSelection: true
        )

        ModelSelectionManager.updateSegmentAppearance(
          self.modelSegmentedControl,
          standardModels: self.standardModels,
          currentTask: yoloTask
        )
      } else {
        self.restoreCurrentModelSelection()
      }

      if success && hasExternalDisplay {
        let yoloTask = appTasks.first(where: { $0.name == loadedTask })?.yoloTask ?? .detect
        var fullModelPath = ""

        if let entry = self.currentLoadingEntry {
          if entry.isLocalBundle {
            if let folderURL = appTasks.first(where: { $0.name == loadedTask })?.folder,
              let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
            {
              fullModelPath = folderPathURL.appendingPathComponent(entry.identifier).path
            }
          } else {
            let localModelURL = ModelCacheManager.shared.modelURL(for: entry.identifier)
            fullModelPath =
              FileManager.default.fileExists(atPath: localModelURL.path)
              ? localModelURL.path : ""
          }
        }

        if !fullModelPath.isEmpty {
          ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
          self.checkAndNotifyExternalDisplayIfReady()
        }
      }

      // Only toggle the local YOLOView inference flag when no external display is active.
      if !hasExternalDisplay {
        self.yoloView.setInferenceFlag(ok: success || keptCurrentModel)
      }

      if success {
        self.labelName.text = processString(modelName)
        self.resetDroneCandidate()
        if modelName.lowercased().contains("birddrone")
          || modelName.lowercased().contains("drone81")
          || modelName.lowercased().contains("aerial3")
        {
          // Retain single-frame suspicion candidates; never infer identity from motion.
          self.yoloView.setConfidenceThreshold(0.30)
          self.yoloView.setNumItemsThreshold(30)
        }
      }
      self.currentLoadingEntry = nil
      self.currentLoadingTask = nil
    }
  }

  @IBAction func vibrate(_ sender: Any) { selection.selectionChanged() }

  @IBAction func indexChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()
    guard appTasks.indices.contains(sender.selectedSegmentIndex) else { return }

    let newTask = appTasks[sender.selectedSegmentIndex].name

    if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
      let alert = UIAlertController(
        title: "\(newTask) Models not found",
        message: "Please add or define models for \(newTask).", preferredStyle: .alert)
      alert.addAction(
        UIAlertAction(title: "OK", style: .cancel) { _ in alert.dismiss(animated: true) })
      present(alert, animated: true)
      sender.selectedSegmentIndex = appTasks.firstIndex { $0.name == currentTask } ?? 0
      return
    }

    reloadModelEntriesAndLoadFirst(for: newTask)
  }

  private func setupModelSegmentedControl() {
    modelSegmentedControl.isHidden = false
    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.apportionsSegmentWidthsByContent = true
    modelSegmentedControl.addTarget(
      self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)

    modelSegmentedControl.translatesAutoresizingMaskIntoConstraints = false
    // Fill width up to a portrait-phone cap and stay centered (storyboard centerX), so landscape doesn't
    // stretch the row edge to edge - same behavior as the Flutter app's model row.
    let fillWidth = modelSegmentedControl.widthAnchor.constraint(equalToConstant: 430)
    fillWidth.priority = .defaultHigh
    NSLayoutConstraint.activate([
      fillWidth,
      modelSegmentedControl.widthAnchor.constraint(
        lessThanOrEqualTo: view.safeAreaLayoutGuide.widthAnchor, constant: -40),
    ])
  }

  private func setupTaskSegmentedControl() {
    segmentedControl.overrideUserInterfaceStyle = .dark
    segmentedControl.apportionsSegmentWidthsByContent = false
    segmentedControl.selectedSegmentTintColor = UIColor.white.withAlphaComponent(0.18)
    segmentedControl.setTitleTextAttributes(
      [
        .font: UIFont.systemFont(ofSize: 12, weight: .semibold),
        .foregroundColor: UIColor.white,
      ], for: .selected)
    segmentedControl.setTitleTextAttributes(
      [
        .font: UIFont.systemFont(ofSize: 12, weight: .medium),
        .foregroundColor: UIColor.white.withAlphaComponent(0.72),
      ], for: .normal)
  }

  @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    if sender.selectedSegmentIndex < ModelSelectionManager.ModelSize.allCases.count {
      let size = ModelSelectionManager.ModelSize.allCases[sender.selectedSegmentIndex]
      if let model = standardModels[size] {
        let entry = ModelEntry(
          displayName: (model.name as NSString).deletingPathExtension,
          identifier: model.name,
          isLocalBundle: model.isLocal,
          remoteURL: model.url
        )
        loadModel(entry: entry, forTask: currentTask)
      }
    }
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    loadingOverlayView?.frame = view.bounds
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] image in
      guard let self = self, let image = image else { return print("error capturing photo") }
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        let vc = UIActivityViewController(activityItems: [image], applicationActivities: nil)
        vc.popoverPresentationController?.sourceView = self.View0
        self.present(vc, animated: true)
      }
    }
  }

  @objc func sliderValueChanged(_ sender: UISlider) {
    // Forward threshold values to the external display (no-op if none is connected).
    let conf = Double(round(100 * yoloView.sliderConf.value)) / 100
    let iou = Double(round(100 * yoloView.sliderIoU.value)) / 100
    let maxItems = Int(yoloView.sliderNumItems.value)

    NotificationCenter.default.post(
      name: .thresholdDidChange,
      object: nil,
      userInfo: [
        "conf": conf,
        "iou": iou,
        "maxItems": maxItems,
      ]
    )
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }

}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { [weak self] in
      self?.fpsText = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
      self?.renderPerformanceLabel()
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    let breakdownTotal = result.preMs + result.inferenceMs + result.postMs
    let text =
      breakdownTotal > 0
      ? String(
        format: "%.1f pre · %.1f inference · %.1f post", result.preMs, result.inferenceMs,
        result.postMs)
      : ""
    DispatchQueue.main.async { [weak self] in
      self?.breakdownText = text
      self?.renderPerformanceLabel()
      self?.processAerialFeatures(result: result)
      self?.processDroneAlert(result: result)
    }
  }

  /// Renders the FPS line with the smaller stage-breakdown line beneath it in the single `labelFPS`,
  /// so the breakdown inherits the label's exact layout and show/hide behavior (e.g. camera switching).
  private func renderPerformanceLabel() {
    let now = CACurrentMediaTime()
    guard lastPerformanceLabelDraw == 0 || now - lastPerformanceLabelDraw >= 0.35 else { return }
    lastPerformanceLabelDraw = now
    let rendered = NSMutableAttributedString(
      string: fpsText,
      attributes: [
        .font: UIFont.preferredFont(forTextStyle: .body), .foregroundColor: UIColor.white,
      ])
    if !breakdownText.isEmpty {
      rendered.append(
        NSAttributedString(
          string: "\n" + breakdownText,
          attributes: [
            .font: UIFont.preferredFont(forTextStyle: .caption1),
            .foregroundColor: UIColor.white.withAlphaComponent(0.7),
          ]))
    }
    labelFPS.attributedText = rendered
  }

}
