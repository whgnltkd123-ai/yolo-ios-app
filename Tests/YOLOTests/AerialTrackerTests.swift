import CoreGraphics
import XCTest
@testable import UltralyticsYOLO

final class AerialTrackerTests: XCTestCase {
  private func box(_ label: String = "drone", _ score: Float = 0.8, _ x: CGFloat = 0.4) -> Box {
    let rect = CGRect(x: x, y: 0.4, width: 0.04, height: 0.04)
    return Box(index: 0, cls: label, conf: score, xywh: rect, xywhn: rect)
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
      tracks = tracker.update(boxes: [box("drone", 0.8, 0.38), box("drone", 0.8, 0.44)],
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
    for i in 0..<10 { tracks = tracker.update(boxes: [box("drone", 0.69)], time: Double(i) * 0.05) }
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
    let id = tracks[0].id
    tracker.reset()
    tracks = tracker.update(boxes: [box()], time: 2)
    XCTAssertNotEqual(tracks[0].id, id)
    XCTAssertEqual(tracks[0].classification, "미확인 물체")
  }
}

