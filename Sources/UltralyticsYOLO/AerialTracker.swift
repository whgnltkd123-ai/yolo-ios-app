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
  fileprivate var lastSeen: Double
  fileprivate var velocity: CGVector
  fileprivate var recent: [(String, Float)]

  public var classification: String {
    guard observed, !identificationSuppressed, let last = recent.last else { return "미확인 물체" }
    let needed = last.0 == "drone" ? 7 : 3
    let threshold: Float = last.0 == "drone" ? 0.70 : 0.60
    guard recent.count >= needed,
      recent.suffix(needed).allSatisfy({ $0.0 == last.0 && $0.1 >= threshold })
    else { return "미확인 물체" }
    if last.0 == "drone" {
      return min(box.width, box.height) >= 0.02 ? "드론" : "미확인 물체"
    }
    return last.0 == "bird" ? "새" : "비행기"
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

  public static func category(_ label: String) -> String? {
    let value = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if ["bird", "birds"].contains(value) { return "bird" }
    if ["airplane", "aeroplane", "plane"].contains(value) { return "airplane" }
    if ["drone", "drones", "uav", "uas", "quadcopter", "hexacopter", "multicopter",
      "fpv drone", "fpv-drone", "fixed wing", "fixed-wing", "unmanned aerial vehicle"].contains(value)
      || value.hasPrefix("drone ") || value.hasSuffix(" drone") { return "drone" }
    return nil
  }

  public mutating func update(boxes: [Box], time: Double) -> [AerialTrack] {
    guard time.isFinite else { reset(); return [] }
    if let lastTime, time <= lastTime || time - lastTime > 0.5 { reset() }
    lastTime = time
    tracks.removeAll { time - $0.lastSeen > 0.5 }
    let detections = Array(boxes.filter {
      Self.category($0.cls) != nil && $0.conf.isFinite && $0.conf >= 0.30
        && $0.xywhn.width > 0 && $0.xywhn.height > 0
        && [$0.xywhn.minX, $0.xywhn.minY, $0.xywhn.width, $0.xywhn.height].allSatisfy(\.isFinite)
    }.prefix(50))
    var costs = [[Double]](repeating: [Double](repeating: .infinity, count: detections.count), count: tracks.count)
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
    for i in tracks.indices {
      let candidates = detections.indices.filter { costs[i][$0].isFinite }.sorted { costs[i][$0] < costs[i][$1] }
      guard let j = candidates.first,
        candidates.count < 2 || costs[i][candidates[1]] - costs[i][j] > 0.2,
        !matchedDetections.contains(j)
      else { tracks[i].observed = false; tracks[i].recent.removeAll(); tracks[i].samples = 0; continue }
      let rivals = tracks.indices.filter { $0 != i && costs[$0][j].isFinite }
      guard rivals.allSatisfy({ costs[$0][j] - costs[i][j] > 0.2 }) else {
        tracks[i].observed = false; tracks[i].recent.removeAll(); tracks[i].samples = 0; continue
      }
      let detection = detections[j]
      let dt = max(time - tracks[i].lastSeen, 0.001)
      let measured = CGVector(dx: (detection.xywhn.midX - tracks[i].box.midX) / dt,
                              dy: (detection.xywhn.midY - tracks[i].box.midY) / dt)
      tracks[i].velocity = CGVector(dx: measured.dx * 0.6 + tracks[i].velocity.dx * 0.4,
                                    dy: measured.dy * 0.6 + tracks[i].velocity.dy * 0.4)
      tracks[i].box = detection.xywhn
      tracks[i].label = Self.category(detection.cls)!
      tracks[i].score = detection.conf
      tracks[i].observed = true
      tracks[i].lastSeen = time
      tracks[i].samples += 1
      tracks[i].recent.append((tracks[i].label, detection.conf))
      tracks[i].recent = Array(tracks[i].recent.suffix(8))
      tracks[i].trail.append(CGPoint(x: detection.xywhn.midX, y: detection.xywhn.midY))
      tracks[i].trail = Array(tracks[i].trail.suffix(24))
      matchedDetections.insert(j)
    }
    for j in detections.indices where !matchedDetections.contains(j) {
      let box = detections[j]
      tracks.append(AerialTrack(id: nextID, box: box.xywhn, label: Self.category(box.cls)!,
        score: box.conf, observed: true, trail: [CGPoint(x: box.xywhn.midX, y: box.xywhn.midY)],
        samples: 1, lastSeen: time, velocity: .zero, recent: [(Self.category(box.cls)!, box.conf)]))
      nextID += 1
    }
    // Prefer live observations; bound history when many overlapping detections are ambiguous.
    tracks = Array(tracks.sorted { $0.lastSeen > $1.lastSeen }.prefix(100))
    return tracks.sorted { $0.id < $1.id }
  }
}
