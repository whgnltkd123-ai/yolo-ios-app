import UIKit
import XCTest
@testable import UltralyticsYOLO

final class AerialDecisionTests: XCTestCase {
  private func detection(_ label: String, _ score: Float, _ scores: [String: Float]?) -> Box {
    Box(index: 0, cls: label, conf: score, xywh: .zero, xywhn: .zero, classScores: scores)
  }

  func testAreaReviewOnPublicPhoto() throws {
    let folder = try XCTUnwrap(Bundle.module.url(forResource: "AreaReview", withExtension: nil))
    let model = folder.appendingPathComponent("C7Area.mlmodelc")
    guard FileManager.default.fileExists(atPath: model.path) else {
      throw XCTSkip("Run the c7 area export to supply the compiled test resource.")
    }
    let image = try XCTUnwrap(UIImage(contentsOfFile: folder.appendingPathComponent("frame.jpg").path))
    let review = try AerialAreaSearch.inspect(image, modelURL: model)
    XCTAssertTrue(review.original.contains { $0.aerialDisplayLabel == "드론" && $0.xywh.contains(CGPoint(x: 558, y: 190)) })
    // Bottom-left tile's new proposal must map to the original photo, not the tile origin.
    XCTAssertTrue(review.candidates.contains { $0.xywh.contains(CGPoint(x: 265, y: 385)) })
    XCTAssertTrue(review.verified.allSatisfy { $0.aerialDisplayLabel == "드론 의심" })
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
