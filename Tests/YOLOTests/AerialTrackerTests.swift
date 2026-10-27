import CoreGraphics
import CoreML
import XCTest

@testable import UltralyticsYOLO

final class AerialTrackerTests: XCTestCase {
  private func box(_ label: String = "drone", _ score: Float = 0.8, _ x: CGFloat = 0.4) -> Box {
    let rect = CGRect(x: x, y: 0.4, width: 0.04, height: 0.04)
    var result = Box(index: 0, cls: label, conf: score, xywh: rect, xywhn: rect)
    result.classScores = ["drone": label == "drone" ? score : 0.1,
                          "bird": label == "bird" ? score : 0.1, "airplane": 0.1]
    return result
  }

  func testStableScoreGapAndClassChange() {
    var tracker = AerialTracker()
    var tracks: [AerialTrack] = []
    for i in 0..<7 { tracks = tracker.update(boxes: [box()], time: Double(i) * 0.05) }
    let id = tracks[0].id
    XCTAssertEqual(tracks[0].classification, "드론")
    tracks = tracker.update(boxes: [], time: 0.35)
    XCTAssertFalse(tracks[0].observed)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    tracks = tracker.update(boxes: [box()], time: 0.4)
    XCTAssertEqual(tracks[0].id, id)
    XCTAssertEqual(tracks[0].samples, 1)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    for i in 0..<3 { tracks = tracker.update(boxes: [box("bird")], time: 0.45 + Double(i) * 0.05) }
    XCTAssertEqual(tracks[0].classification, "새")
    tracks = tracker.update(boxes: [box()], time: 0.6)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
  }

  func testAmbiguousCrossingDoesNotInheritConfirmedIdentity() {
    var tracker = AerialTracker()
    var tracks: [AerialTrack] = []
    for i in 0..<7 {
      tracks = tracker.update(
        boxes: [box("drone", 0.8, 0.38), box("drone", 0.8, 0.44)],
        time: Double(i) * 0.05)
    }
    let ids = Set(tracks.map(\.id))
    tracks = tracker.update(boxes: [box("drone", 0.8, 0.41)], time: 0.35)
    let observed = tracks.filter(\.observed)
    XCTAssertEqual(observed.count, 1)
    XCTAssertFalse(ids.contains(observed[0].id))
    XCTAssertEqual(observed[0].classification, "미확인 물체")
  }

  func testLowScoreAndResetCannotClaimDrone() {
    var tracker = AerialTracker()
    var tracks: [AerialTrack] = []
    for i in 0..<10 { tracks = tracker.update(boxes: [box("drone", 0.59)], time: Double(i) * 0.05) }
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    let id = tracks[0].id
    tracker.reset()
    tracks = tracker.update(boxes: [box()], time: 2)
    XCTAssertNotEqual(tracks[0].id, id)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
  }

  func testHighFrameRateCannotShortenConfirmationTime() {
    var tracker = AerialTracker()
    var tracks: [AerialTrack] = []
    for i in 0..<7 { tracks = tracker.update(boxes: [box()], time: Double(i) / 120) }
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    for i in 7...36 { tracks = tracker.update(boxes: [box()], time: Double(i) / 120) }
    XCTAssertEqual(tracks[0].classification, "드론")
  }

  func testWeakObservationKeepsIDButClearsConfirmation() {
    var tracker = AerialTracker()
    var tracks: [AerialTrack] = []
    for i in 0..<7 { tracks = tracker.update(boxes: [box()], time: Double(i) * 0.05) }
    let id = tracks[0].id
    tracks = tracker.update(boxes: [box("drone", 0.35)], time: 0.35)
    XCTAssertEqual(tracks[0].id, id)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    tracks = tracker.update(boxes: [box()], time: 0.40)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    tracker.suppressIdentification(for: id)
    tracks = tracker.update(boxes: [box()], time: 0.45)
    XCTAssertEqual(tracks[0].id, id)
    XCTAssertEqual(tracks[0].samples, 1)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
  }

  func testStrongObservationTakesPriorityOverCloserWeakBox() {
    var tracker = AerialTracker()
    let first = tracker.update(boxes: [box()], time: 0)[0]
    let tracks = tracker.update(boxes: [box("drone", 0.35), box("drone", 0.8, 0.41)], time: 0.05)
    let continued = tracks.first { $0.id == first.id }!
    XCTAssertEqual(continued.score, 0.8)
    XCTAssertEqual(continued.box.minX, 0.41)
    XCTAssertEqual(continued.classification, "미확인 물체")
  }

  func testTwentyPointMarginAndSixtyPointConfirmationForBothClasses() {
    for label in ["drone", "bird"] {
      for other in [Float(0.4), Float(0.399)] {
        var tracker = AerialTracker()
        var detection = box(label, 0.60)
        detection.classScores?[label == "drone" ? "bird" : "drone"] = other
        var tracks: [AerialTrack] = []
        for i in 0..<8 { tracks = tracker.update(boxes: [detection], time: Double(i) * 0.05) }
        XCTAssertEqual(tracks[0].classification,
          other == 0.4 ? "미확인 물체" : label == "drone" ? "드론" : "새")
        if other == 0.4 { XCTAssertNil(tracks[0].suspicion) }
      }
    }
  }

  func testSuspicionMissingScoresAndBothLowRemainUnconfirmed() {
    for label in ["drone", "bird"] {
      var tracker = AerialTracker()
      var tracks: [AerialTrack] = []
      for i in 0..<8 { tracks = tracker.update(boxes: [box(label, 0.40)], time: Double(i) * 0.05) }
      XCTAssertEqual(tracks[0].classification, "미확인 물체")
      XCTAssertEqual(tracks[0].suspicion, label == "drone" ? "드론 의심" : "새 의심")
      var missing = box(label, 0.95); missing.classScores = nil
      for i in 8..<16 { tracks = tracker.update(boxes: [missing], time: Double(i) * 0.05) }
      XCTAssertEqual(tracks[0].classification, "미확인 물체")
      XCTAssertNil(tracks[0].suspicion)
      tracks = tracker.update(boxes: [box(label, 0.29)], time: 0.8)
      XCTAssertTrue(tracks.allSatisfy { $0.classification == "미확인 물체" })
    }
  }

  func testSameAnchorClassVectorSurvivesEndToEndDecode() throws {
    let detector = ObjectDetector()
    detector.labels = ["drone", "bird", "airplane"]
    detector.hasAerialClassScores = true
    detector.modelInputSize = (640, 640)
    detector.inputSize = CGSize(width: 640, height: 640)
    let prediction = try MLMultiArray(shape: [1, 300, 9], dataType: .float32)
    for i in 0..<prediction.count { prediction[i] = 0 }
    for (i, value) in [100.0, 100, 140, 140, 0.65, 0, 0.65, 0.5, 0.01].enumerated() {
      prediction[i] = NSNumber(value: value)
    }
    let boxes = detector.processRawResults(prediction)
    XCTAssertEqual(boxes.count, 1)
    XCTAssertEqual(boxes[0].classScores?["bird"], 0.5)
    var tracker = AerialTracker()
    var tracks: [AerialTrack] = []
    for i in 0..<8 { tracks = tracker.update(boxes: boxes, time: Double(i) * 0.05) }
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    XCTAssertNil(tracks[0].suspicion)
  }
}
