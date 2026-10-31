import XCTest
@testable import UltralyticsYOLO

final class AerialDecisionTests: XCTestCase {
  private func detection(_ label: String, _ score: Float, _ scores: [String: Float]?) -> Box {
    Box(index: 0, cls: label, conf: score, xywh: .zero, xywhn: .zero, classScores: scores)
  }

  func testAmbiguousAndMissingScoresRemainUnknown() {
    XCTAssertEqual(detection("drone", 0.8, nil).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.6, ["drone": 0.6, "bird": 0.4]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.29, ["drone": 0.29, "bird": 0]).aerialDisplayLabel, "미확인 물체")
    XCTAssertEqual(detection("drone", 0.8, ["drone": 0.8, "bird": 0.1, "airplane": 0.75]).aerialDisplayLabel, "미확인 물체")
  }

  func testSymmetricSuspicionAndConfirmation() {
    XCTAssertEqual(detection("drone", 0.3, ["drone": 0.3, "bird": 0.05]).aerialDisplayLabel, "드론 의심")
    XCTAssertEqual(detection("bird", 0.3, ["drone": 0.05, "bird": 0.3]).aerialDisplayLabel, "새 의심")
    XCTAssertEqual(detection("drone", 0.6, ["drone": 0.6, "bird": 0.1]).aerialDisplayLabel, "드론")
    XCTAssertEqual(detection("bird", 0.6, ["drone": 0.1, "bird": 0.6]).aerialDisplayLabel, "새")
  }
}
