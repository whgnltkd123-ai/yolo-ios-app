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

  // MARK: - Bird-versus-drone alert (YOLO inference and camera are unchanged)
  private let droneAlertButton = UIButton(type: .system)
  private let droneAlertStatusLabel = UILabel()
  private var droneAlertEnabled = true
  private var consecutiveDroneFrames = 0
  private var previousDroneBox: CGRect?
  private var lastDroneAlertTime: Date?
  private let droneAlertCooldown: TimeInterval = 10
  private let requiredConsecutiveDroneFrames = 3

  private func installDroneAlertControls() {
    let panel = UIStackView(arrangedSubviews: [droneAlertButton, droneAlertStatusLabel])
    panel.axis = .vertical
    panel.alignment = .leading
    panel.spacing = 5
    panel.translatesAutoresizingMaskIntoConstraints = false

    droneAlertButton.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    droneAlertButton.setTitleColor(.white, for: .normal)
    droneAlertButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
    droneAlertButton.layer.cornerRadius = 8
    droneAlertButton.contentEdgeInsets = UIEdgeInsets(top: 7, left: 10, bottom: 7, right: 10)
    droneAlertButton.addTarget(self, action: #selector(toggleDroneAlert), for: .touchUpInside)
    updateDroneAlertButton()

    droneAlertStatusLabel.text = "새·드론 분류 모델 확인 중"
    droneAlertStatusLabel.textColor = .white
    droneAlertStatusLabel.font = .systemFont(ofSize: 12, weight: .medium)
    droneAlertStatusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.76)
    droneAlertStatusLabel.layer.cornerRadius = 6
    droneAlertStatusLabel.clipsToBounds = true
    droneAlertStatusLabel.numberOfLines = 2

    yoloView.addSubview(panel)
    NSLayoutConstraint.activate([
      panel.leadingAnchor.constraint(equalTo: yoloView.leadingAnchor, constant: 12),
      panel.topAnchor.constraint(equalTo: yoloView.safeAreaLayoutGuide.topAnchor, constant: 12),
      panel.widthAnchor.constraint(lessThanOrEqualTo: yoloView.widthAnchor, multiplier: 0.84),
    ])
  }

  private func updateDroneAlertButton() {
    droneAlertButton.setTitle(droneAlertEnabled ? "🚨 드론 경보 ON" : "🔕 드론 경보 OFF", for: .normal)
    droneAlertButton.accessibilityLabel = droneAlertEnabled ? "드론 경보 끄기" : "드론 경보 켜기"
  }

  private func resetDroneCandidate() {
    consecutiveDroneFrames = 0
    previousDroneBox = nil
  }

  @objc private func toggleDroneAlert() {
    droneAlertEnabled.toggle()
    resetDroneCandidate()
    updateDroneAlertButton()
    droneAlertStatusLabel.text = droneAlertEnabled ? "새·드론 분류 모델 확인 중" : "경보 꺼짐"
  }

  private func isBirdClass(_ value: String) -> Bool {
    ["bird", "birds"].contains(value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
  }

  /// A drone-only model cannot be trusted to distinguish a bird from a drone.
  private func isDroneClass(_ value: String) -> Bool {
    let label = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    let droneClasses: Set<String> = [
      "drone", "drones", "uav", "uas", "quadcopter", "multicopter",
      "fpv drone", "fpv-drone", "unmanned aerial vehicle", "fixed-wing", "fixed wing",
    ]
    return droneClasses.contains(label) || label.hasPrefix("drone ") || label.hasSuffix(" drone")
  }

  private func overlaps(_ a: CGRect, _ b: CGRect) -> Bool {
    let area = max(a.width * a.height + b.width * b.height - a.intersection(b).width * a.intersection(b).height, 0)
    guard area > 0 else { return false }
    let intersection = a.intersection(b)
    guard !intersection.isNull else { return false }
    return intersection.width * intersection.height / area >= 0.30
  }

  private func matchesPreviousDrone(_ rect: CGRect) -> Bool {
    guard let previous = previousDroneBox else { return false }
    let dx = rect.midX - previous.midX
    let dy = rect.midY - previous.midY
    return (dx * dx + dy * dy) <= (0.18 * 0.18) || overlaps(rect, previous)
  }

  private func processDroneAlert(result: YOLOResult) {
    guard droneAlertEnabled else { return }
    guard currentTask == "Detect" else {
      resetDroneCandidate()
      droneAlertStatusLabel.text = "Detect 모델을 선택하세요"
      return
    }

    // Do not arm audible alerts with drone-only or general-purpose models.
    guard result.names.contains(where: isBirdClass),
          result.names.contains(where: isDroneClass) else {
      resetDroneCandidate()
      droneAlertStatusLabel.text = "새·드론 동시 학습 모델이 필요함"
      return
    }

    let threshold = max(Float(yoloView.sliderConf.value), 0.60)
    let birds = result.boxes.filter { isBirdClass($0.cls) && $0.conf >= 0.45 }
    let candidates = result.boxes
      .filter { isDroneClass($0.cls) && $0.conf >= threshold }
      .filter { candidate in
        !birds.contains(where: { bird in
          bird.conf >= candidate.conf * 0.80 && overlaps(bird.xywhn, candidate.xywhn)
        })
      }
    let best = candidates.max(by: { $0.conf < $1.conf })

    guard let candidate = best else {
      resetDroneCandidate()
      if let bird = birds.max(by: { $0.conf < $1.conf }) {
        droneAlertStatusLabel.text = String(format: "🐦 새 탐지 %.0f%% · 경보 없음", bird.conf * 100)
      } else if result.boxes.contains(where: { isDroneClass($0.cls) || isBirdClass($0.cls) }) {
        droneAlertStatusLabel.text = "🟡 미확인 · 신뢰도 부족"
      } else {
        droneAlertStatusLabel.text = "감시 중 · 탐지 대상 없음"
      }
      return
    }

    consecutiveDroneFrames = matchesPreviousDrone(candidate.xywhn)
      ? min(consecutiveDroneFrames + 1, requiredConsecutiveDroneFrames) : 1
    previousDroneBox = candidate.xywhn

    guard consecutiveDroneFrames >= requiredConsecutiveDroneFrames else {
      droneAlertStatusLabel.text = "🟡 드론 후보 검증 중 (\(consecutiveDroneFrames)/\(requiredConsecutiveDroneFrames))"
      return
    }

    droneAlertStatusLabel.text = String(format: "🚨 드론 후보 %.0f%%", candidate.conf * 100)
    let now = Date()
    guard lastDroneAlertTime.map({ now.timeIntervalSince($0) >= droneAlertCooldown }) ?? true
    else { return }
    lastDroneAlertTime = now

    UINotificationFeedbackGenerator().notificationOccurred(.warning)
    AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    AudioServicesPlayAlertSound(SystemSoundID(1005))
  }

  // MARK: - Constants
  private struct Constants {
    static let defaultTask = YOLOTask.detect
    static let logoURL = "https://www.ultralytics.com"
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
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    installDroneAlertControls()
    [yoloView.labelName, yoloView.labelFPS, yoloView.labelBreakdown].forEach { $0?.isHidden = true }

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

    if let firstSize = ModelSelectionManager.ModelSize.allCases.first,
      let model = standardModels[firstSize]
    {
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

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: Constants.logoURL) {
      UIApplication.shared.open(link)
    }
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
    alignLogoWithThresholdSliders()
  }

  private func alignLogoWithThresholdSliders() {
    guard logoImage != nil, yoloView != nil else { return }
    let sliderMidY = (yoloView.sliderConf.frame.midY + yoloView.sliderIoU.frame.midY) / 2
    guard sliderMidY.isFinite, sliderMidY > 0 else { return }

    logoImage.center = CGPoint(x: logoImage.center.x, y: sliderMidY + 6)
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
      self?.processDroneAlert(result: result)
    }
  }

  /// Renders the FPS line with the smaller stage-breakdown line beneath it in the single `labelFPS`,
  /// so the breakdown inherits the label's exact layout and show/hide behavior (e.g. camera switching).
  private func renderPerformanceLabel() {
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
