import XCTest
@testable import UltralyticsYOLO

final class MotionTrackingTests: XCTestCase {
  private func box(_ x:CGFloat, score:Float = 0.8, bird:Float = 0.02) -> Box {
    let rect = CGRect(x:x,y:0.3,width:0.08,height:0.06)
    return Box(index:0,cls:"drone",conf:score,xywh:rect,xywhn:rect,
      classScores:["drone":score,"bird":bird,"airplane":0.01])
  }
  func testIDAndLowScoreRescueDoNotCreateNewDetections() {
    let tracker = AerialMotionTracker()
    let a = tracker.update(boxes:[box(0.2)],time:1)
    let b = tracker.update(boxes:[box(0.205)],time:1.05)
    XCTAssertEqual(a.first?.id,b.first?.id)
    let c = tracker.update(boxes:[box(0.21,score:0.2)],time:1.10)
    XCTAssertEqual(b.first?.id,c.first?.id)
    XCTAssertEqual(c.first?.label,"미확인 물체")
    XCTAssertTrue(c.first?.observed == true)
    let predicted = tracker.update(boxes:[],time:1.15)
    XCTAssertFalse(predicted.first?.observed ?? true)
    XCTAssertNil(predicted.first?.detectionIndex)
    XCTAssertEqual(predicted.first?.label,"예측 유지")
    XCTAssertTrue(tracker.update(boxes:[],time:2).isEmpty)
    XCTAssertTrue(AerialMotionTracker().update(boxes:[box(0.2,score:0.2)],time:1).isEmpty)
  }
  func testCompensationTransitionPreservesIDAndDisruptionRevokesConfirmation() {
    let tracker = AerialMotionTracker()
    let a = tracker.update(boxes:[box(0.2)],time:1)
    _ = tracker.update(boxes:[box(0.2)],time:1.05)
    let stable = tracker.update(boxes:[box(0.2)],time:1.10)
    XCTAssertEqual(stable.first?.label,"드론")
    let moved = tracker.update(boxes:[box(0.23)],time:1.15,translation:CGPoint(x:0.03,y:0))
    XCTAssertEqual(moved.first?.id,a.first?.id)
    let changed = tracker.update(boxes:[box(0.23)],time:1.20,disrupted:true)
    XCTAssertEqual(changed.first?.id,a.first?.id)
    XCTAssertEqual(changed.first?.label,"미확인 물체")
  }
  func testCurrentAmbiguityCannotBeOverruledByEMA() {
    let tracker = AerialMotionTracker()
    for i in 0..<6 { _ = tracker.update(boxes:[box(0.2)],time:1+Double(i)*0.05) }
    let mixed = tracker.update(boxes:[box(0.2,score:0.7,bird:0.5)],time:1.3)
    XCTAssertEqual(mixed.first?.label,"미확인 물체")
  }
  func testUnobservedTrackCannotTriggerDroneAlert() {
    var result = YOLOResult(orig_shape:.zero,boxes:[box(0.2)],speed:0,names:["drone","bird","airplane"])
    result.tracking = AerialTrackingFrame(objects:[TrackedAerialObject(id:1,rect:.zero,
      observed:false,label:"드론",score:0.9,detectionIndex:nil)],cameraState:"test",
      processingMS:0,latencyMS:0,thermalState:0)
    XCTAssertNil(result.confirmedAerialDrone)
  }
  func testSparseFlowTranslationAndTexturelessFailure() {
    let flow = CameraMotionCompensation()
    var old = [UInt8](repeating:0,count:96*64)
    for i in old.indices {
      let row: Int = (i/96)*31
      let texture: Int = (i*i)%113
      old[i] = UInt8((i*73+row+texture)%256)
    }
    var shifted = [UInt8](repeating:0,count:old.count)
    for y in 0..<64 { for x in 2..<96 { shifted[y*96+x] = old[y*96+x-2] } }
    let vectors = flow.flow(from:old,to:shifted,excluding:[],deadline:.infinity)
    XCTAssertGreaterThan(vectors.count,8)
    XCTAssertTrue(vectors.allSatisfy { $0.x == 2 && $0.y == 0 })
    let blank = [UInt8](repeating:100,count:96*64)
    XCTAssertTrue(flow.flow(from:blank,to:blank,excluding:[],deadline:.infinity).isEmpty)
  }
}
