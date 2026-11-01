import CoreGraphics
import Foundation

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

/// Independent constant-velocity Kalman axes, normalized image coordinates.
/// Swift motion state; association uses Ultralytics ByteTrack's separate
/// high/low detection stages and high-only births. No ReID model.
/// This adapter is not a numerically identical port of the Python tracker.
struct MotionAxis {
  var position: Double
  var velocity = 0.0
  var p00 = 0.001, p01 = 0.0, p11 = 0.01

  mutating func predict(_ dt: Double) {
    position += velocity * dt
    p00 += 2 * dt * p01 + dt * dt * p11 + 0.0001 * dt
    p01 += dt * p11
    p11 += 0.001 * dt
  }

  mutating func correct(_ value: Double) {
    let variance = p00 + 0.0001
    let k0 = p00 / variance, k1 = p01 / variance
    let innovation = value - position
    let old00 = p00, old01 = p01
    position += k0 * innovation
    velocity += k1 * innovation
    p00 = max(0.0000001, p00 - k0 * old00)
    p01 -= k0 * old01
    p11 = max(0.0000001, p11 - k1 * old01)
  }
}

final class AerialMotionTracker {
  struct Track {
    let id: Int
    var axes: [MotionAxis]
    var lastSeen: Double
    var scores: [Float]?
    var hits = 1
    var observations = 1
    var missedFrames = 0
    var decisionClass: String?
    var decisionSince: Double
    var observedIndex: Int?
    var score: Float
    var rect: CGRect {
      CGRect(x: axes[0].position - max(0.001, axes[2].position) / 2,
             y: axes[1].position - max(0.001, axes[3].position) / 2,
             width: max(0.001, axes[2].position), height: max(0.001, axes[3].position))
    }
  }
  private var tracks: [Track] = []
  private var nextID = 1
  private var previousTime: Double?
  var foreground: [CGRect] { tracks.map(\.rect) }

  func reset() { tracks.removeAll(); previousTime = nil } // IDs never recycled in this session.

  func update(boxes: [Box], time: Double, translation: CGPoint = .zero,
              disrupted: Bool = false) -> [TrackedAerialObject] {
    let elapsed = previousTime.map { time - $0 } ?? 0
    previousTime = time
    if elapsed < 0 { tracks.removeAll() }
    let dt = min(1.0, max(0, elapsed))
    tracks.removeAll { $0.missedFrames > 30 || time - $0.lastSeen > 3.0 }
    for i in tracks.indices {
      tracks[i].missedFrames += 1
      tracks[i].observedIndex = nil
      for j in 0..<4 { tracks[i].axes[j].predict(dt) }
      tracks[i].axes[0].position += Double(translation.x)
      tracks[i].axes[1].position += Double(translation.y)
      if disrupted {
        tracks[i].scores = nil; tracks[i].hits = 0
        tracks[i].decisionClass = nil; tracks[i].decisionSince = time
        for j in 0..<4 { tracks[i].axes[j].velocity = 0 }
      }
    }
    let valid = boxes.indices.filter {
      let b = boxes[$0], r = b.xywhn
      return b.conf.isFinite && b.conf >= 0.10 && b.conf <= 1
        && [r.minX, r.minY, r.width, r.height].allSatisfy(\.isFinite)
        && r.width > 0 && r.height > 0 && r.width <= 1 && r.height <= 1
        && ["drone", "bird", "airplane"].contains(b.cls.lowercased())
    }.prefix(50)
    var remaining = Set(valid), unused = Set(tracks.indices)
    var ambiguous = Set<Int>()
    // ByteTrack: weak detections recover existing identities, never create them.
    for high in [true, false] {
      let candidates = remaining.filter { (boxes[$0].conf >= 0.25) == high }
      var edges: [(Double, Int, Int)] = []
      for t in unused.sorted() {
        for b in candidates.sorted() {
          let a = tracks[t].rect, r = boxes[b].xywhn
          let intersection = a.intersection(r)
          let ia = intersection.isNull ? 0 : intersection.width * intersection.height
          let iou = ia / max(0.000001, a.width * a.height + r.width * r.height - ia)
          let dx = Double(a.midX-r.midX), dy = Double(a.midY-r.midY)
          let mahalanobis = dx*dx/(tracks[t].axes[0].p00+0.0001)
            + dy*dy/(tracks[t].axes[1].p00+0.0001)
          // A few pixels of size jitter must not split a tiny object's identity.
          // The floor applies to association only, never to the displayed/detector box.
          let aw = max(a.width, 0.00625), ah = max(a.height, 0.00625)
          let rw = max(r.width, 0.00625), rh = max(r.height, 0.00625)
          let scale = [rw/aw, aw/rw, rh/ah, ah/rh].max() ?? 10
          // Tiny boxes can stop overlapping after only a few pixels of motion.
          let extent = max(max(a.width,a.height),max(r.width,r.height))
          let radius = min(disrupted ? 0.02 : 0.04, max(0.0125, Double(extent)*1.5))
          let distance = sqrt(dx*dx + dy*dy)
          let nearby = distance <= radius
          guard scale < 2.5, mahalanobis < 9.21,
            iou > (disrupted ? 0.55 : 0.10) || nearby else { continue }
          edges.append((0.5*Double(1-iou) + 0.5*distance/radius + 0.05*mahalanobis, t, b))
        }
      }
      // Reject near-ties on either side instead of guessing identities in crossings.
      edges.sort { $0.0 == $1.0 ? ($0.1, $0.2) < ($1.1, $1.2) : $0.0 < $1.0 }
      var blockedTracks = Set<Int>()
      // Global minimum-cost assignment, rather than greedily consuming the
      // nearest edge and forcing the remaining object into a new identity.
      let assigned = Self.assign(edges, tracks: unused.sorted(), detections: candidates.sorted())
      for e in assigned {
        guard unused.contains(e.1), remaining.contains(e.2), !blockedTracks.contains(e.1),
          !ambiguous.contains(e.2) else { continue }
        let rivals = edges.filter {
          ($0.1 == e.1 || $0.2 == e.2) && !($0.1 == e.1 && $0.2 == e.2)
            && unused.contains($0.1) && remaining.contains($0.2)
        }
        if rivals.contains(where: { abs($0.0-e.0) < 0.08 }) {
          ambiguous.insert(e.2); blockedTracks.insert(e.1)
          for rival in rivals where abs(rival.0-e.0)<0.08 {
            ambiguous.insert(rival.2); blockedTracks.insert(rival.1)
          }
          continue
        }
        let box = boxes[e.2], r = box.xywhn
        let values = [r.midX,r.midY,r.width,r.height].map(Double.init)
        for j in 0..<4 { tracks[e.1].axes[j].correct(values[j]) }
        let continuous = time-tracks[e.1].lastSeen <= 0.35
        let decision = high ? Self.decisionClass(box) : nil
        if !continuous || decision == nil || decision != tracks[e.1].decisionClass {
          tracks[e.1].scores = nil; tracks[e.1].hits = 0
          tracks[e.1].decisionSince = time
        }
        tracks[e.1].decisionClass = decision
        tracks[e.1].observations = continuous ? tracks[e.1].observations+1 : 1
        tracks[e.1].lastSeen = time
        tracks[e.1].missedFrames = 0
        tracks[e.1].observedIndex = e.2
        tracks[e.1].score = box.conf
        if high {
          tracks[e.1].hits += 1
          if let scores = Self.scores(box) {
            let alpha = Float(1-exp(-max(dt,0.01)/0.20))
            tracks[e.1].scores = tracks[e.1].scores.map { old in
              zip(old,scores).map { pair in pair.0 + alpha*(pair.1-pair.0) }
            } ?? scores
          } else { tracks[e.1].scores = nil }
        }
        unused.remove(e.1); remaining.remove(e.2)
      }
    }
    for b in remaining.sorted() where !ambiguous.contains(b) && boxes[b].conf >= 0.25 {
      guard tracks.count < 50 else { break }
      let box = boxes[b], r = box.xywhn
      // Do not manufacture a second ID for a near-duplicate anchor/ambiguous crossing.
      if tracks.contains(where: { $0.rect.intersects(r) }) { continue }
      tracks.append(Track(id: nextID,
        axes: [r.midX,r.midY,r.width,r.height].map { MotionAxis(position: Double($0)) },
        lastSeen: time, scores: Self.scores(box), decisionClass: Self.decisionClass(box),
        decisionSince: time, observedIndex: b, score: box.conf))
      nextID += 1
    }
    return tracks.compactMap { t in
      let observed = t.observedIndex != nil
      guard observed || (!disrupted && time-t.lastSeen <= 0.25) else { return nil }
      var label = "미확인 물체"
      // Official COCO has no drone score. Do not feed it to the aerial3 margin
      // policy, which intentionally rejects a missing competitor score.
      if observed, let index = t.observedIndex, boxes[index].classScores?["drone"] == nil {
        let b = boxes[index]
        if b.conf >= 0.30, ["bird", "airplane"].contains(b.cls.lowercased()) {
          label = b.cls.lowercased() == "bird" ? "새" : "비행기"
        }
      }
      if observed, t.score >= 0.30, t.hits >= 3, let scores = t.scores,
         let index = t.observedIndex, let current = boxes[index].aerialDisplayLabel {
        let winner = scores[0] >= scores[1] ? 0 : 1
        let proxy = Box(index: winner, cls: winner == 0 ? "drone" : "bird",
          conf: scores[winner], xywh: .zero, xywhn: .zero,
          classScores: ["drone":scores[0],"bird":scores[1],"airplane":scores[2]])
        let accumulated = proxy.aerialDisplayLabel ?? label
        // EMA cannot overrule contrary or ambiguous CURRENT evidence.
        if accumulated == current {
          // Repeated frames alone are not enough at high FPS: require elapsed evidence.
          let confirmed = accumulated == "드론" || accumulated == "새"
          if !confirmed || time-t.decisionSince >= 0.25 { label = accumulated }
        }
      }
      return TrackedAerialObject(id: t.id, rect: t.rect, observed: observed,
        label: observed ? label : "예측 유지", score: observed ? t.score : 0,
        detectionIndex: t.observedIndex)
    }
  }

  private static func decisionClass(_ box: Box) -> String? {
    guard let label = box.aerialDisplayLabel else { return nil }
    if label == "드론" || label == "드론 의심" { return "drone" }
    if label == "새" || label == "새 의심" { return "bird" }
    return nil
  }

  /// Rectangular Hungarian assignment with one unmatched column per track.
  /// Uses only geometrically gated edges; unmatched is cheaper than an invalid edge.
  private static func assign(_ edges: [(Double, Int, Int)], tracks: [Int],
                             detections: [Int]) -> [(Double, Int, Int)] {
    guard !tracks.isEmpty, !detections.isEmpty else { return [] }
    let n = tracks.count, m = detections.count + n
    let rows = Dictionary(uniqueKeysWithValues: tracks.enumerated().map { ($0.element, $0.offset) })
    let cols = Dictionary(uniqueKeysWithValues: detections.enumerated().map { ($0.element, $0.offset) })
    var costs = Array(repeating: Array(repeating: 2.0, count: m), count: n)
    for i in 0..<n { for j in 0..<detections.count { costs[i][j] = 1_000_000 } }
    for e in edges { if let i = rows[e.1], let j = cols[e.2] { costs[i][j] = e.0 } }
    var u = Array(repeating: 0.0, count: n+1), v = Array(repeating: 0.0, count: m+1)
    var p = Array(repeating: 0, count: m+1), way = Array(repeating: 0, count: m+1)
    for i in 1...n {
      p[0] = i
      var j0 = 0
      var minimum = Array(repeating: Double.infinity, count: m+1)
      var used = Array(repeating: false, count: m+1)
      repeat {
        used[j0] = true
        let i0 = p[j0]
        var delta = Double.infinity, j1 = 0
        for j in 1...m where !used[j] {
          let cost = costs[i0-1][j-1] - u[i0] - v[j]
          if cost < minimum[j] { minimum[j] = cost; way[j] = j0 }
          if minimum[j] < delta { delta = minimum[j]; j1 = j }
        }
        for j in 0...m {
          if used[j] { u[p[j]] += delta; v[j] -= delta }
          else { minimum[j] -= delta }
        }
        j0 = j1
      } while p[j0] != 0
      repeat {
        let j1 = way[j0]; p[j0] = p[j1]; j0 = j1
      } while j0 != 0
    }
    return (1...detections.count).compactMap { j in
      guard p[j] > 0, costs[p[j]-1][j-1] < 2.0 else { return nil }
      return (costs[p[j]-1][j-1], tracks[p[j]-1], detections[j-1])
    }
  }

  private static func scores(_ b: Box) -> [Float]? {
    guard let s = b.classScores, let d = s["drone"], let v = s["bird"], let p = s["airplane"],
      [d,v,p].allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
    return [d,v,p]
  }
}
