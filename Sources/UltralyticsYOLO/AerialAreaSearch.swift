import CoreML
import UIKit
import Vision

/// A single still-image inspection. Results never enter live tracking, evidence or alarm callbacks.
public struct AerialAreaReview {
  public let size: CGSize
  public let original: [Box]
  public let candidates: [Box]
  public let verified: [Box]
}

public enum AerialAreaSearch {
  private static let checkpoint = "c7df126a039c1edde8d6b3e0c78be99f43e6770913eb1ff201976e2cf2505d26"
  private enum Failure: LocalizedError {
    case model, photo, output
    var errorDescription: String? {
      switch self {
      case .model: return "정밀 탐색 모델을 확인할 수 없습니다."
      case .photo: return "촬영한 사진을 읽을 수 없습니다."
      case .output: return "탐색 모델의 출력 형식이 올바르지 않습니다."
      }
    }
  }

  public static func inspect(_ photo: UIImage, modelURL: URL) throws -> AerialAreaReview {
    guard let cg = photo.cgImage else { throw Failure.photo }
    let oriented = CIImage(cgImage: cg).oriented(CGImagePropertyOrientation(photo.imageOrientation))
    let image = oriented.transformed(by: CGAffineTransform(translationX: -oriented.extent.minX, y: -oriented.extent.minY))
    let size = image.extent.size
    let configuration = MLModelConfiguration()
    if #available(iOS 16, *) { configuration.computeUnits = .cpuAndNeuralEngine }
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
    guard metadata?["checkpoint_sha256"] == checkpoint, metadata?["area_schema"] == "c7-area-v1"
    else { throw Failure.model }
    let request = VNCoreMLRequest(model: try VNCoreMLModel(for: model))
    request.imageCropAndScaleOption = .scaleFit

    func analyze(_ rect: CGRect, floor: Float = 0.30) throws -> (primary: [Box], auxiliary: [Box]) {
      try Task.checkCancellation()
      let ciRect = CGRect(x: rect.minX, y: size.height - rect.maxY, width: rect.width, height: rect.height)
      let local = image.cropped(to: ciRect).transformed(by: CGAffineTransform(translationX: -ciRect.minX, y: -ciRect.minY))
      try VNImageRequestHandler(ciImage: local, orientation: .up, options: [:]).perform([request])
      let outputs = request.results as? [VNCoreMLFeatureValueObservation] ?? []
      func decode(_ name: String, threshold: Float) throws -> [Box] {
        guard let a = outputs.first(where: { $0.featureName == name })?.featureValue.multiArrayValue,
          a.shape.map({ $0.intValue }) == [1, 300, 9] else { throw Failure.output }
        let gain = min(640 / rect.width, 640 / rect.height)
        let padX = ((640 - (rect.width * gain).rounded()) / 2 - 0.1).rounded()
        let padY = ((640 - (rect.height * gain).rounded()) / 2 - 0.1).rounded()
        let names = ["drone", "bird", "airplane"]
        var boxes = [Box]()
        for i in 0..<300 {
          func value(_ j: Int) -> Float { a[[0, NSNumber(value: i), NSNumber(value: j)]].floatValue }
          let score = value(4)
          guard score.isFinite, score >= threshold else { continue }
          let index = value(5)
          guard index.isFinite, (0...2).contains(index), index.rounded() == index else { continue }
          let cls = Int(index)
          func x(_ j: Int) -> CGFloat { rect.minX + min(rect.width, max(0, (CGFloat(value(j)) - padX) / gain)) }
          func y(_ j: Int) -> CGFloat { rect.minY + min(rect.height, max(0, (CGFloat(value(j)) - padY) / gain)) }
          let l = x(0), t = y(1), r = x(2), b = y(3)
          guard r > l, b > t else { continue }
          let bounds = CGRect(x: l, y: t, width: r-l, height: b-t)
          let normalized = CGRect(x: l/size.width, y: t/size.height, width: bounds.width/size.width, height: bounds.height/size.height)
          boxes.append(Box(index: cls, cls: names[cls], conf: score, xywh: bounds, xywhn: normalized,
            classScores: Dictionary(uniqueKeysWithValues: names.enumerated().map { ($0.element, value(6+$0.offset)) })))
        }
        return boxes
      }
      return (try decode("primary", threshold: floor), try decode("auxiliary", threshold: 0.30).filter(isDrone))
    }

    let full = try analyze(CGRect(origin: .zero, size: size), floor: 0.05)
    var original = full.primary.filter { $0.conf >= 0.30 }
    if let seed = full.primary.first(where: { $0.cls == "drone" && (0.10..<0.30).contains($0.conf) && separate($0, original) }) {
      let crop = ObjectDetector.recheckRect(around: seed.xywh, in: size)
      if var extra = try analyze(crop).primary.first(where: { isDrone($0) && overlap($0,seed) >= 0.30 && separate($0,original) }) {
        extra.isCropRecheck = true
        original.append(extra)
      }
    }
    var proposals = full.auxiliary.filter { a in full.primary.contains { $0.cls == "drone" && overlap(a,$0) >= 0.30 } }
    let w = (size.width * 0.6).rounded(.toNearestOrEven), h = (size.height * 0.6).rounded(.toNearestOrEven)
    for y in [CGFloat.zero, size.height-h] {
      for x in [CGFloat.zero, size.width-w] {
        let local = try analyze(CGRect(x: x, y: y, width: w, height: h))
        proposals += local.primary.filter { b in isDrone(b) && local.auxiliary.contains { overlap(b,$0) >= 0.30 } }
      }
    }
    var candidates = [Box]()
    for box in proposals.sorted(by: { $0.conf > $1.conf }) where separate(box,original) && separate(box,candidates) {
      candidates.append(box)
    }
    var verified = [Box]()
    for box in candidates.prefix(2) {
      let cw = ceil(size.width/3), ch = ceil(size.height/3)
      let crop = CGRect(x: min(size.width-cw, max(0,(box.xywh.midX-cw/2).rounded(.toNearestOrEven))),
        y: min(size.height-ch, max(0,(box.xywh.midY-ch/2).rounded(.toNearestOrEven))), width: cw, height: ch)
      if try analyze(crop).primary.contains(where: { isDrone($0) && overlap($0,box) >= 0.5 }) {
        var suspect = box
        suspect.isCropRecheck = true
        verified.append(suspect)
      }
    }
    return AerialAreaReview(size: size, original: original, candidates: candidates, verified: verified)
  }

  private static func isDrone(_ b: Box) -> Bool {
    b.cls == "drone" && (b.aerialDisplayLabel == "드론" || b.aerialDisplayLabel == "드론 의심")
  }

  private static func overlap(_ a: Box, _ b: Box) -> CGFloat {
    ObjectDetector.overlap(a.xywh, b.xywh)
  }

  private static func separate(_ box: Box, _ existing: [Box]) -> Bool {
    !existing.contains { overlap(box,$0) >= 0.25 }
  }
}
