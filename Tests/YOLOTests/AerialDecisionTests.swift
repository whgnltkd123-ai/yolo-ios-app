import CoreImage
import CoreMedia
import XCTest
@testable import UltralyticsYOLO

final class AerialDecisionTests: XCTestCase {
  private func detection(_ label: String, _ score: Float, _ scores: [String: Float]?) -> Box {
    Box(index: 0, cls: label, conf: score, xywh: .zero, xywhn: .zero, classScores: scores)
  }

  private final class Results: ResultsListener {
    var result: YOLOResult?
    func on(result: YOLOResult) { self.result = result }
  }

  func testAutomaticCameraZoomOnPublicFrame() async throws {
    let folder = try XCTUnwrap(Bundle.module.url(forResource: "UnknownZoom", withExtension: nil))
    let model = folder.appendingPathComponent("yolo26s-aerial3.mlmodelc")
    guard FileManager.default.fileExists(atPath: model.path) else {
      throw XCTSkip("A compiled c7 test model is supplied by the trained-app build.")
    }
    let base: BasePredictor = try await withCheckedThrowingContinuation { continuation in
      BasePredictor.create(for: .detect, modelURL: model, isRealTime: true, useGpu: false) { result in
        continuation.resume(with: result)
      }
    }
    let detector = try XCTUnwrap(base as? ObjectDetector)
    XCTAssertTrue(detector.isRecallCheckpoint)
    detector.setConfidenceThreshold(confidence: 0.30)
    for file in ["unknown-square.png", "unknown.jpg"] {
      let image = try XCTUnwrap(CIImage(contentsOf: folder.appendingPathComponent(file)))
      var pixel: CVPixelBuffer?
      let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
      XCTAssertEqual(CVPixelBufferCreate(nil, Int(image.extent.width), Int(image.extent.height),
        kCVPixelFormatType_32BGRA, attrs, &pixel), kCVReturnSuccess)
      let buffer = try XCTUnwrap(pixel)
      CIContext().render(image, to: buffer)
      var info: CMVideoFormatDescription?
      XCTAssertEqual(CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
        imageBuffer: buffer, formatDescriptionOut: &info), noErr)
      var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
      var frame: CMSampleBuffer?
      XCTAssertEqual(CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault,
        imageBuffer: buffer, formatDescription: try XCTUnwrap(info), sampleTiming: &timing,
        sampleBufferOut: &frame), noErr)
      let sample = try XCTUnwrap(frame)
      let sink = Results()
      func run(_ enabled: Bool) throws -> [Box] {
        sink.result = nil
        detector.droneRecheckEnabled = enabled
        detector.predict(sampleBuffer: sample, onResultsListener: sink, onInferenceTime: nil)
        return try XCTUnwrap(sink.result).boxes
      }
      let original = try run(false), zoom = try run(true)
      XCTAssertEqual(original.map(\.xywh), zoom.map(\.xywh))
      if file == "unknown-square.png" {
        XCTAssertTrue(zip(original,zoom).contains { a,b in
          a.aerialDisplayLabel == "미확인 물체" && b.aerialDisplayLabel == "비행기" && b.isCropRecheck
        })
      }
      for (a,b) in zip(original,zoom) where a.aerialDisplayLabel != "미확인 물체" {
        XCTAssertEqual(a.aerialDisplayLabel,b.aerialDisplayLabel)
        XCTAssertEqual(a.conf,b.conf)
      }
      print("C7_CAMERA_UNKNOWN_ZOOM",file,original.compactMap(\.aerialDisplayLabel),zoom.compactMap(\.aerialDisplayLabel))
    }
  }

  func testZoomDoesNotPromoteAmbiguousOrNeighboringDetections() {
    let rect = CGRect(x: 100,y: 100,width: 18,height: 12)
    let crop = CGRect(x: 60,y: 40,width: 160,height: 160)
    let localRect = rect.offsetBy(dx: -crop.minX, dy: -crop.minY)
    func box(_ cls: String, _ score: Float, _ scores: [Float], _ bounds: CGRect) -> Box {
      Box(index: cls == "drone" ? 0 : 1, cls: cls, conf: score, xywh: bounds,
        xywhn: .zero,classScores: ["drone":scores[0],"bird":scores[1],"airplane":scores[2]])
    }
    let seed = box("drone",0.55,[0.55,0.46,0],rect)
    let drone = box("drone",0.82,[0.82,0.05,0],localRect)
    let bird = box("bird",0.84,[0.08,0.84,0],localRect)
    func label(_ full: [Box], _ local: [Box]) -> String? {
      ObjectDetector.zoomDecision(at: 0,in: full,local: local,crop: crop).aerialDisplayLabel
    }
    XCTAssertEqual(label([seed],[bird]),"새")
    XCTAssertEqual(label([seed],[drone]),"드론 의심")
    XCTAssertEqual(label([seed],[bird,drone]),"미확인 물체")
    XCTAssertEqual(label([seed],[box("airplane",0.7,[0.51,0.04,0.7],localRect)]),"미확인 물체")
    XCTAssertEqual(label([seed,seed],[drone]),"미확인 물체")
    XCTAssertEqual(label([seed],[]),"미확인 물체")
    XCTAssertEqual(label([seed],[box("drone",0.52,[0.52,0.04,0],localRect)]),"미확인 물체")
    XCTAssertEqual(label([box("bird",0.55,[0.46,0.55,0],rect)],[drone]),"미확인 물체")
  }


  func testRecheckedDroneIsSuspectedEvenAtHighScore() {
    var box = detection("drone", 0.88, ["drone": 0.88, "bird": 0.01, "airplane": 0.02])
    box.isCropRecheck = true
    XCTAssertEqual(box.aerialDisplayLabel, "드론 의심")
    let result = YOLOResult(orig_shape: CGSize(width: 640, height: 512), boxes: [box], speed: 0, names: ["drone", "bird", "airplane"])
    XCTAssertNil(result.confirmedAerialDrone)
    XCTAssertEqual(result.aerialWarningLevel, 2)
  }

  func testRecheckCropClampsAtRightAndTopInPortrait() {
    let crop = ObjectDetector.recheckRect(around: CGRect(x: 350, y: 5, width: 8, height: 12),
      in: CGSize(width: 361, height: 641))
    XCTAssertEqual(crop, CGRect(x: 180, y: 0, width: 181, height: 321))
  }

  func testAmbiguousAndMissingScoresRemainUnknown() {
    XCTAssertEqual(detection("drone", 0.8, nil).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.6, ["drone": 0.6, "bird": 0.4]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.29, ["drone": 0.29, "bird": 0]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.8, ["drone": 0.8, "bird": 0.1, "airplane": 0.75]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.7, ["drone": 0.7, "bird": 0.4]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("bird", 0.8, ["drone": 0.3, "bird": 0.8]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.8, ["drone": 0.8, "bird": 0.1, "airplane": 0.3]).aerialDisplayLabel, "미확인 물체")
  }

  func testSymmetricSuspicionAndConfirmation() {
    XCTAssertEqual(detection("drone", 0.3, ["drone": 0.3, "bird": 0.05]).aerialDisplayLabel, "드론 의심")
    XCTAssertEqual(detection("bird", 0.3, ["drone": 0.05, "bird": 0.3]).aerialDisplayLabel, "새 의심")
    XCTAssertEqual(detection("drone", 0.6, ["drone": 0.6, "bird": 0.1]).aerialDisplayLabel, "드론")
    XCTAssertEqual(detection("bird", 0.6, ["drone": 0.1, "bird": 0.6]).aerialDisplayLabel, "새")
  }
}
