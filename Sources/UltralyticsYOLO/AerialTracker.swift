import CoreGraphics
import Foundation

/// Short-lived screen-space tracks. IDs are not identities and never increase model scores.
public struct AerialTrack {
  public let id: Int
  public var box: CGRect
  public var label: String
  public var score: Float
  public var observed: Bool
  public var trail: [CGPoint]
  public var samples: Int
  public var identificationSuppressed = false
  public var classScores: [String: Float]?
  fileprivate var lastSeen: Double
  fileprivate var velocity: CGVector
  fileprivate var recent: [(String, Float, Double, Float?)]

  private func hasEvidence(threshold: Float, count: Int, duration: Double) -> Bool {
    guard observed, !identificationSuppressed, let last = recent.last else { return false }
    let consistent = recent.reversed().prefix {
      $0.0 == last.0 && $0.1 >= threshold
        && (last.0 == "airplane" || ($0.3 ?? -1) > 0.20 + 0.000001)
    }
    guard consistent.count >= count, let first = consistent.last else { return false }
    return last.2 - first.2 + 0.000001 >= duration
  }

  public var classification: String {
    guard
      hasEvidence(
        threshold: 0.60, count: label == "drone" ? 7 : 3,
        duration: label == "drone" ? 0.30 : 0.10)
    else { return "미확인 물체" }
    return label == "drone" ? "드론" : label == "bird" ? "새" : "비행기"
  }

  /// Suspicion is still an unidentified object and must not trigger confirmed-drone broadcast.
  public var suspicion: String? {
    guard classification == "미확인 물체", label != "airplane",
      hasEvidence(threshold: 0.40, count: 3, duration: 0.10)
    else { return nil }
    return label == "drone" ? "드론 의심" : "새 의심"
  }

  public var displayLabel: String {
    suspicion.map { "미확인 물체 · \($0)" } ?? classification
  }

}

/// Causal motion association with one-to-one matches and ambiguity rejection.
/// A gap keeps the ID briefly, but clears identification evidence and never emits an alert.
public struct AerialTracker {
  private var tracks: [AerialTrack] = []
  private var nextID = 1
  private var lastTime: Double?
  public init() {}

  public mutating func reset() {
    tracks.removeAll()
    lastTime = nil
  }

  /// A contradictory observation cancels accumulated evidence, not the screen-space ID.
  public mutating func suppressIdentification(for id: Int) {
    guard let i = tracks.firstIndex(where: { $0.id == id }) else { return }
    tracks[i].recent.removeAll()
    tracks[i].samples = 0
  }

  public static func category(_ label: String) -> String? {
    let value = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if ["bird", "birds"].contains(value) { return "bird" }
    if ["airplane", "aeroplane", "plane"].contains(value) { return "airplane" }
    if [
      "drone", "drones", "uav", "uas", "quadcopter", "hexacopter", "multicopter",
      "fpv drone", "fpv-drone", "fixed wing", "fixed-wing", "unmanned aerial vehicle",
    ].contains(value)
      || value.hasPrefix("drone ") || value.hasSuffix(" drone")
    {
      return "drone"
    }
    return nil
  }

  private static func margin(for box: Box) -> Float? {
    guard let scores = box.classScores, let drone = scores["drone"], let bird = scores["bird"],
      drone.isFinite, bird.isFinite, (0...1).contains(drone), (0...1).contains(bird)
    else { return nil }
    let label = category(box.cls)
    let selected = label == "drone" ? drone : bird
    guard label == "drone" || label == "bird", abs(selected - box.conf) < 0.02 else { return nil }
    return label == "drone" ? drone - bird : bird - drone
  }

  public mutating func update(boxes: [Box], time: Double) -> [AerialTrack] {
    guard time.isFinite else {
      reset()
      return []
    }
    if let lastTime, time <= lastTime || time - lastTime > 0.5 { reset() }
    lastTime = time
    tracks.removeAll { time - $0.lastSeen > 0.5 }
    let detections = Array(
      boxes.filter {
        Self.category($0.cls) != nil && $0.conf.isFinite && (0.30...1).contains($0.conf)
          && $0.xywhn.width > 0 && $0.xywhn.height > 0
          && [$0.xywhn.minX, $0.xywhn.minY, $0.xywhn.width, $0.xywhn.height].allSatisfy(\.isFinite)
      }.prefix(50))
    var costs = [[Double]](
      repeating: [Double](repeating: .infinity, count: detections.count), count: tracks.count)
    for i in tracks.indices {
      let track = tracks[i]
      let dt = min(0.5, time - track.lastSeen)
      for j in detections.indices {
        let box = detections[j].xywhn
        let scale = (box.width * box.height) / (track.box.width * track.box.height)
        guard (0.4...2.5).contains(scale) else { continue }
        let dx = box.midX - track.box.midX - track.velocity.dx * dt
        let dy = box.midY - track.box.midY - track.velocity.dy * dt
        let length = max(max(box.width, track.box.width), max(box.height, track.box.height))
        let gate = max(0.015, min(0.09, length * 2))
        let distance = hypot(dx, dy) / gate
        if distance <= 1 { costs[i][j] = Double(distance) + abs(log(Double(scale))) * 0.15 }
      }
    }
    var matchedDetections = Set<Int>()
    var matchedTracks = Set<Int>()
    var ambiguousTracks = Set<Int>()
    // ByteTrack-style staging: reliable observations first, then weaker observations for continuity.
    // This is native geometric association, not the Python ByteTrack implementation or ReID.
    for strong in [true, false] {
      for i in tracks.indices where !matchedTracks.contains(i) && !ambiguousTracks.contains(i) {
        let candidates = detections.indices.filter {
          !matchedDetections.contains($0) && (detections[$0].conf >= 0.60) == strong
            && costs[i][$0].isFinite
        }.sorted { costs[i][$0] < costs[i][$1] }
        guard !candidates.isEmpty else { continue }
        guard let j = candidates.first,
          candidates.count < 2 || costs[i][candidates[1]] - costs[i][j] > 0.2
        else {
          ambiguousTracks.insert(i)
          continue
        }
        let rivals = tracks.indices.filter {
          $0 != i && !matchedTracks.contains($0) && costs[$0][j].isFinite
        }
        guard rivals.allSatisfy({ costs[$0][j] - costs[i][j] > 0.2 }) else {
          ambiguousTracks.insert(i)
          continue
        }
        let detection = detections[j]
        let dt = max(time - tracks[i].lastSeen, 0.001)
        let measured = CGVector(
          dx: (detection.xywhn.midX - tracks[i].box.midX) / dt,
          dy: (detection.xywhn.midY - tracks[i].box.midY) / dt)
        tracks[i].velocity = CGVector(
          dx: measured.dx * 0.6 + tracks[i].velocity.dx * 0.4,
          dy: measured.dy * 0.6 + tracks[i].velocity.dy * 0.4)
        tracks[i].box = detection.xywhn
        tracks[i].label = Self.category(detection.cls)!
        tracks[i].score = detection.conf
        tracks[i].classScores = detection.classScores
        tracks[i].observed = true
        tracks[i].lastSeen = time
        tracks[i].samples += 1
        tracks[i].recent.append(
          (tracks[i].label, detection.conf, time, Self.margin(for: detection)))
        tracks[i].recent = Array(tracks[i].recent.filter { time - $0.2 <= 1 }.suffix(64))
        tracks[i].trail.append(CGPoint(x: detection.xywhn.midX, y: detection.xywhn.midY))
        tracks[i].trail = Array(tracks[i].trail.suffix(24))
        matchedDetections.insert(j)
        matchedTracks.insert(i)
      }
    }
    for i in tracks.indices where !matchedTracks.contains(i) {
      tracks[i].observed = false
      tracks[i].recent.removeAll()
      tracks[i].samples = 0
    }
    for j in detections.indices where !matchedDetections.contains(j) {
      let box = detections[j]
      tracks.append(
        AerialTrack(
          id: nextID, box: box.xywhn, label: Self.category(box.cls)!,
          score: box.conf, observed: true, trail: [CGPoint(x: box.xywhn.midX, y: box.xywhn.midY)],
          samples: 1, classScores: box.classScores, lastSeen: time, velocity: .zero,
          recent: [(Self.category(box.cls)!, box.conf, time, Self.margin(for: box))]))
      nextID += 1
    }
    // Prefer live observations; bound history when many overlapping detections are ambiguous.
    tracks = Array(tracks.sorted { $0.lastSeen > $1.lastSeen }.prefix(100))
    return tracks.sorted { $0.id < $1.id }
  }
}
