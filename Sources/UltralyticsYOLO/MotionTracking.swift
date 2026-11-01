import CoreGraphics
import Foundation
import OfficialByteTrack

public enum AerialRecognitionMode: Int, Sendable {
  case original = 0, tracking = 1, compensatedTracking = 2
}
public struct TrackedAerialObject: Sendable {
  public let id: Int
  public let rect: CGRect
  public let observed: Bool
  public let label: String
  public let score: Float
  public let detectionIndex: Int?
}
public struct AerialTrackingFrame: Sendable {
  public let objects: [TrackedAerialObject]
  public let cameraState: String
  public let processingMS: Double
  public let latencyMS: Double
  public let thermalState: Int
}

/// Coordinate and decision adapter. The official C++ engine owns IDs and association.
final class AerialMotionTracker {
  private struct Evidence {
    var rect: CGRect
    var lastSeen: Double
    var scores: [Float]?
    var hits: Int
    var decision: String?
    var since: Double
  }
  private var engine = obt_create()
  private var evidence: [Int: Evidence] = [:]
  private var offset = CGPoint.zero
  private var previousTime: Double?
  var foreground: [CGRect] { evidence.values.map(\.rect) }
  deinit { obt_destroy(engine) }
  func reset() {
    obt_destroy(engine); engine = obt_create()
    evidence.removeAll(); offset = .zero; previousTime = nil
  }
  func update(boxes: [Box], time: Double, translation: CGPoint = .zero,
              disrupted: Bool = false) -> [TrackedAerialObject] {
    guard time.isFinite else { return [] }
    let dt = previousTime.map { time - $0 } ?? 0.05
    if dt < 0 || dt > 1 { reset() }
    previousTime = time
    if translation.x.isFinite && translation.y.isFinite {
      offset.x += translation.x; offset.y += translation.y
    }
    // Stabilize coordinates outside the engine without changing upstream Kalman math.
    let valid = boxes.indices.filter {
      let b = boxes[$0], r = b.xywhn
      return b.conf.isFinite && (0.10...1).contains(b.conf)
        && [r.minX,r.minY,r.width,r.height].allSatisfy(\.isFinite)
        && r.width > 0 && r.height > 0 && r.width <= 1 && r.height <= 1
        && ["drone","bird","airplane"].contains(b.cls.lowercased())
    }.prefix(50)
    let input = valid.map { i -> OBTDetection in
      let b = boxes[i], r = b.xywhn
      return OBTDetection(x:Float((r.minX-offset.x)*640),
        y:Float((r.minY-offset.y)*640), width:Float(r.width*640),
        height:Float(r.height*640),score:b.conf,index:Int32(i))
    }
    var native = [OBTTrack](repeating: OBTTrack(), count:50)
    let count = input.withUnsafeBufferPointer { source in
      native.withUnsafeMutableBufferPointer { destination in
        obt_update(engine,source.baseAddress,Int32(source.count),destination.baseAddress,50)
      }
    }
    guard count >= 0 else { return [] }
    if disrupted {
      for id in Array(evidence.keys) {
        evidence[id]?.scores = nil; evidence[id]?.hits = 0
        evidence[id]?.decision = nil; evidence[id]?.since = time
      }
    }
    evidence = evidence.filter { time-$0.value.lastSeen <= 1 }
    var observed = Set<Int>()
    var output: [TrackedAerialObject] = []
    for t in native.prefix(Int(count)) {
      let index = Int(t.index), id = Int(t.id)
      guard boxes.indices.contains(index) else { continue }
      let b = boxes[index]
      let rect = CGRect(x:CGFloat(t.x)/640+offset.x,y:CGFloat(t.y)/640+offset.y,
                        width:CGFloat(t.width)/640,height:CGFloat(t.height)/640)
      let decision = Self.decisionClass(b)
      var e = evidence[id] ?? Evidence(rect:rect,lastSeen:time,scores:nil,
                                       hits:0,decision:nil,since:time)
      if time-e.lastSeen > 0.35 || decision == nil || decision != e.decision {
        e.scores = nil; e.hits = 0; e.since = time
      }
      e.rect=rect; e.lastSeen=time; e.decision=decision
      if b.conf >= 0.30 && !disrupted {
        e.hits += 1
        if let scores=Self.scores(b) {
          let alpha=Float(1-exp(-max(0.01,min(dt,0.35))/0.20))
          e.scores=e.scores.map { old in
            zip(old,scores).map { $0.0+alpha*($0.1-$0.0) }
          } ?? scores
        } else { e.scores=nil }
      }
      var label="미확인 물체"
      if !disrupted, b.conf >= 0.30, e.hits >= 3, let scores=e.scores,
         let current=b.aerialDisplayLabel {
        let winner=scores[0] >= scores[1] ? 0 : 1
        let proxy=Box(index:winner,cls:winner == 0 ? "drone":"bird",
          conf:scores[winner],xywh:.zero,xywhn:.zero,
          classScores:["drone":scores[0],"bird":scores[1],"airplane":scores[2]])
        let accumulated=proxy.aerialDisplayLabel ?? label
        if accumulated == current {
          let confirmed=accumulated == "드론" || accumulated == "새"
          if !confirmed || time-e.since >= 0.25 { label=accumulated }
        }
      }
      evidence[id]=e; observed.insert(id)
      output.append(TrackedAerialObject(id:id,rect:rect,observed:true,label:label,
                                        score:b.conf,detectionIndex:index))
    }
    // Last observation held briefly: not a new detection, prediction, or alert.
    for (id,e) in evidence where !observed.contains(id) && !disrupted && time-e.lastSeen <= 0.25 {
      output.append(TrackedAerialObject(id:id,rect:e.rect,observed:false,
        label:"관측 끊김",score:0,detectionIndex:nil))
    }
    return output.sorted { $0.id < $1.id }
  }
  private static func decisionClass(_ box: Box) -> String? {
    guard let label=box.aerialDisplayLabel else { return nil }
    if label == "드론" || label == "드론 의심" { return "drone" }
    if label == "새" || label == "새 의심" { return "bird" }
    return nil
  }
  private static func scores(_ b: Box) -> [Float]? {
    guard let s=b.classScores,let d=s["drone"],let v=s["bird"],let p=s["airplane"],
      [d,v,p].allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
    return [d,v,p]
  }
}
