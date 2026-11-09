import CoreImage
import CoreMedia
import XCTest
@testable import UltralyticsYOLO

final class AerialDecisionTests: XCTestCase {
  private func detection(_ label: String, _ score: Float, _ scores: [String: Float]?) -> Box {
    Box(index: 0, cls: label, conf: score, xywh: .zero, xywhn: .zero, classScores: scores)
  }

  private final class ObservationSink: ResultsListener {
    var boxes: [Box] = []
    func on(result: YOLOResult) { boxes = result.boxes }
  }

  func testFirstSightingRequiresAnotherMeasurementAndNeverConfirmsDrone() {
    var sighting = FirstSighting()
    let size = CGSize(width: 640,height: 512)
    func box(_ cls: String, _ confidence: Float, _ x: CGFloat) -> Box {
      Box(index: cls == "drone" ? 0 : 1,cls: cls,conf: confidence,
        xywh: CGRect(x: x,y: 22,width: 12,height: 10),xywhn: .zero,
        classScores: ["drone":cls == "drone" ? confidence : 0.02,
                      "bird":cls == "bird" ? confidence : 0.02,"airplane":0.02])
    }
    let weak = box("drone",0.25,10), other = box("bird",0.24,80)
    XCTAssertNil(sighting.observation(in: [weak],size: size,time: 0,zoom: 1))
    XCTAssertEqual(sighting.observation(in: [weak,other],size: size,time: 0.05,zoom: 1),0)
    var cue = weak
    cue.isFirstSighting = true
    XCTAssertEqual(cue.aerialDisplayLabel,"미확인 물체")
    let result = YOLOResult(orig_shape:size,boxes:[cue],speed:0,names:["drone","bird","airplane"])
    XCTAssertEqual(result.aerialWarningLevel,1)
    XCTAssertNil(result.confirmedAerialDrone)
    XCTAssertNil(sighting.observation(in: [box("bird",0.85,10),weak],size:size,time:0.10,zoom:1))
    XCTAssertNil(sighting.observation(in: [weak],size:size,time:0.15,zoom:2))
    XCTAssertNil(sighting.observation(in: [],size:size,time:0.20,zoom:2))
    XCTAssertNil(sighting.observation(in: [weak],size:size,time:0.25,zoom:2))
    XCTAssertNil(sighting.observation(in: [weak],size:size,time:1.50,zoom:2))
  }

  func testFirstSightingOnTwoObservedPublicFrames() async throws {
    let folder = try XCTUnwrap(Bundle.module.url(forResource: "FirstSighting", withExtension:nil))
    let model = folder.appendingPathComponent("yolo26s-aerial3.mlmodelc")
    guard FileManager.default.fileExists(atPath:model.path) else {
      throw XCTSkip("The trained-app build supplies the compiled c7 model.")
    }
    let base: BasePredictor = try await withCheckedThrowingContinuation { continuation in
      BasePredictor.create(for:.detect,modelURL:model,isRealTime:true,useGpu:false) {
        continuation.resume(with:$0)
      }
    }
    let detector = try XCTUnwrap(base as? ObjectDetector)
    XCTAssertTrue(detector.isFirstSightingCheckpoint)
    detector.setConfidenceThreshold(confidence:0.30)
    detector.collectsFirstSightings = true
    var state = FirstSighting()
    let sink = ObservationSink()
    let context = CIContext()
    for number in 0...1 {
      let image = try XCTUnwrap(CIImage(contentsOf:folder.appendingPathComponent("frame-\(number).png")))
      var pixel: CVPixelBuffer?
      let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
      XCTAssertEqual(CVPixelBufferCreate(nil,Int(image.extent.width),Int(image.extent.height),
        kCVPixelFormatType_32BGRA,attrs,&pixel),kCVReturnSuccess)
      let buffer = try XCTUnwrap(pixel)
      context.render(image,to:buffer)
      var format: CMVideoFormatDescription?
      XCTAssertEqual(CMVideoFormatDescriptionCreateForImageBuffer(allocator:kCFAllocatorDefault,
        imageBuffer:buffer,formatDescriptionOut:&format),noErr)
      var timing = CMSampleTimingInfo(duration:.invalid,presentationTimeStamp:CMTime(value:Int64(number),timescale:25),decodeTimeStamp:.invalid)
      var frame: CMSampleBuffer?
      XCTAssertEqual(CMSampleBufferCreateReadyWithImageBuffer(allocator:kCFAllocatorDefault,imageBuffer:buffer,
        formatDescription:try XCTUnwrap(format),sampleTiming:&timing,sampleBufferOut:&frame),noErr)
      detector.predict(sampleBuffer:try XCTUnwrap(frame),onResultsListener:sink,onInferenceTime:nil)
      let index = state.observation(in:sink.boxes,size:image.extent.size,time:Double(number)/25,zoom:1)
      if number == 0 {
        XCTAssertNil(index)
      } else {
        let cue = sink.boxes[try XCTUnwrap(index)]
        XCTAssertEqual(cue.aerialDisplayLabel,"미확인 물체")
        XCTAssertLessThan(cue.conf,0.30)
        print("FIRST_SIGHTING_OBSERVED",cue.conf,cue.xywh)
      }
    }
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
