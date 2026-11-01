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
/// No ReID; conservative two-stage association inspired by BoT-SORT/ByteTrack.
/// This is not a numerically identical port of either upstream implementation.
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
    var lastStrong: Double
    var scores: [Float]?
    var hits = 1
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
    if elapsed < 0 || elapsed > 0.75 { tracks.removeAll() }
    let dt = min(0.25, max(0, elapsed))
    tracks.removeAll { time - $0.lastSeen > 0.45 || time - $0.lastStrong > 0.75 }
    for i in tracks.indices {
      tracks[i].observedIndex = nil
      for j in 0..<4 { tracks[i].axes[j].predict(dt) }
      tracks[i].axes[0].position += Double(translation.x)
      tracks[i].axes[1].position += Double(translation.y)
      if disrupted {
        tracks[i].scores = nil; tracks[i].hits = 0
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
    // High-score association first. Low-score detections rescue only existing IDs.
    for high in [true, false] {
      let candidates = remaining.filter { (boxes[$0].conf >= 0.30) == high }
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
          let scale = [r.width/a.width, a.width/r.width, r.height/a.height, a.height/r.height].max() ?? 10
          guard scale < 2.5, mahalanobis < 9.21, iou > (disrupted ? 0.55 : 0.10) else { continue }
          edges.append((Double(1-iou) + 0.05*mahalanobis, t, b))
        }
      }
      // Reject near-ties on either side instead of guessing identities in crossings.
      edges.sort { $0.0 == $1.0 ? ($0.1, $0.2) < ($1.1, $1.2) : $0.0 < $1.0 }
      var blockedTracks = Set<Int>()
      for e in edges {
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
        tracks[e.1].lastSeen = time
        tracks[e.1].observedIndex = e.2
        tracks[e.1].score = box.conf
        if high {
          tracks[e.1].lastStrong = time
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
    for b in remaining.sorted() where boxes[b].conf >= 0.30 && !ambiguous.contains(b) {
      guard tracks.count < 50 else { break }
      let box = boxes[b], r = box.xywhn
      // Do not manufacture a second ID for a near-duplicate anchor/ambiguous crossing.
      if tracks.contains(where: { $0.rect.intersects(r) }) { continue }
      tracks.append(Track(id: nextID,
        axes: [r.midX,r.midY,r.width,r.height].map { MotionAxis(position: Double($0)) },
        lastSeen: time, lastStrong: time, scores: Self.scores(box), observedIndex: b, score: box.conf))
      nextID += 1
    }
    return tracks.compactMap { t in
      let observed = t.observedIndex != nil
      guard observed || (!disrupted && time-t.lastSeen <= 0.25) else { return nil }
      var label = "미확인 물체"
      if observed, t.score >= 0.30, t.hits >= 3, let scores = t.scores,
         let index = t.observedIndex, let current = boxes[index].aerialDisplayLabel {
        let winner = scores[0] >= scores[1] ? 0 : 1
        let proxy = Box(index: winner, cls: winner == 0 ? "drone" : "bird",
          conf: scores[winner], xywh: .zero, xywhn: .zero,
          classScores: ["drone":scores[0],"bird":scores[1],"airplane":scores[2]])
        let accumulated = proxy.aerialDisplayLabel ?? label
        // EMA cannot overrule contrary or ambiguous CURRENT evidence.
        if accumulated == current { label = accumulated }
      }
      return TrackedAerialObject(id: t.id, rect: t.rect, observed: observed,
        label: observed ? label : "예측 유지", score: observed ? t.score : 0,
        detectionIndex: t.observedIndex)
    }
  }

  private static func scores(_ b: Box) -> [Float]? {
    guard let s = b.classScores, let d = s["drone"], let v = s["bird"], let p = s["airplane"],
      [d,v,p].allSatisfy({ $0.isFinite && (0...1).contains($0) }) else { return nil }
    return [d,v,p]
  }
}
