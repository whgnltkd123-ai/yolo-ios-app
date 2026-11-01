import CoreMotion
import CoreVideo
import Foundation
import QuartzCore

/// Bounded sparse block-flow translation. No dense flow, neural model, or GPU work.
/// Camera buffers already have inference orientation; flow and boxes share normalized coordinates.
final class CameraMotionCompensation {
  struct Estimate {
    var translation = CGPoint.zero
    var state = "GMC OFF"
    var disrupted = false
  }
  private let motion = CMMotionManager()
  private var previous: [UInt8]?
  private var previousTime = 0.0
  private var previousZoom = 1.0
  private var previousSize = CGSize.zero
  private var movingUntil = 0.0
  private let width = 96, height = 64

  func start() {
    guard motion.isGyroAvailable, !motion.isGyroActive else { return }
    motion.gyroUpdateInterval = 1.0/30
    motion.startGyroUpdates()
  }
  func stop() { motion.stopGyroUpdates(); previous = nil }
  deinit { motion.stopGyroUpdates() }

  func estimate(_ buffer: CVPixelBuffer, time: Double, zoom: Double,
                foreground: [CGRect]) -> Estimate {
    if ProcessInfo.processInfo.thermalState == .serious || ProcessInfo.processInfo.thermalState == .critical {
      previous = nil
      return Estimate(state:"열 제한 · GMC 생략")
    }
    let size = CGSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
    guard let now = thumbnail(buffer) else { previous = nil; return Estimate(state:"GMC unavailable") }
    let old = previous, dt = time-previousTime
    let geometryChanged = previousSize != .zero && (previousSize != size || abs(zoom/previousZoom-1)>0.025)
    previous = now; previousTime = time; previousZoom = zoom; previousSize = size
    let rate = motion.gyroData?.rotationRate
    let gyro = rate.map { sqrt($0.x*$0.x + $0.y*$0.y + $0.z*$0.z) } ?? 0
    guard !geometryChanged, gyro < 1.5 else {
      return Estimate(state:"줌/급이동 재평가",disrupted:true)
    }
    guard let old, dt > 0, dt < 0.35 else {
      return Estimate(state:"영상 기준 준비",disrupted:dt >= 0.35 && dt < 10)
    }
    var difference = 0.0, count = 0
    for y in stride(from:4,to:height-4,by:4) {
      for x in stride(from:4,to:width-4,by:4) {
        let point = CGPoint(x:Double(x)/Double(width),y:Double(y)/Double(height))
        guard !foreground.contains(where: { $0.insetBy(dx:-0.03,dy:-0.03).contains(point) }) else { continue }
        difference += abs(Double(now[y*width+x])-Double(old[y*width+x])); count += 1
      }
    }
    if gyro > 0.06 || (count > 0 && difference/Double(count) > 7) { movingUntil = time+0.25 }
    guard time < movingUntil else { return Estimate(state:"고정 · 보정 생략") }
    let start = CACurrentMediaTime()
    let vectors = flow(from: old, to: now, excluding:foreground, deadline:start+0.008)
    guard vectors.count >= 8 else { return Estimate(state:"이동 · 보정 실패/기본 추적") }
    let dx = vectors.map(\.x).sorted()[vectors.count/2]
    let dy = vectors.map(\.y).sorted()[vectors.count/2]
    let inliers = vectors.filter { abs($0.x-dx)<=1 && abs($0.y-dy)<=1 }
    guard Double(inliers.count)/Double(vectors.count) >= 0.65, abs(dx)<4, abs(dy)<4 else {
      return Estimate(state:"이동 · 보정 불확실/기본 추적")
    }
    return Estimate(translation:CGPoint(x:dx/Double(width),y:dy/Double(height)),state:"이동 · GMC 적용")
  }

  /// 6 KB copy, not a full-resolution UIImage or retained camera frame.
  private func thumbnail(_ buffer: CVPixelBuffer) -> [UInt8]? {
    guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return nil }
    CVPixelBufferLockBaseAddress(buffer,.readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer,.readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let bytes = base.assumingMemoryBound(to:UInt8.self)
    let row = CVPixelBufferGetBytesPerRow(buffer), w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
    var image = [UInt8](repeating:0,count:width*height)
    for y in 0..<height {
      for x in 0..<width {
        let offset = (y*h/height)*row + (x*w/width)*4
        image[y*width+x] = UInt8((Int(bytes[offset])*29+Int(bytes[offset+1])*150+Int(bytes[offset+2])*77)>>8)
      }
    }
    return image
  }

  // Sparse optical flow by local patch correspondence with texture/uniqueness rejection.
  // Translation-only deliberately fails closed on rotation/parallax rather than applying a bad warp.
  func flow(from old:[UInt8], to now:[UInt8], excluding:[CGRect], deadline:Double) -> [CGPoint] {
    guard old.count == width*height, now.count == old.count else { return [] }
    var vectors: [CGPoint] = []
    for y in stride(from:8,to:height-8,by:8) {
      for x in stride(from:8,to:width-8,by:8) {
        if CACurrentMediaTime()>deadline { return [] } // bounded work; never use partial biased fit
        let point = CGPoint(x:Double(x)/Double(width),y:Double(y)/Double(height))
        if excluding.contains(where: { $0.insetBy(dx:-0.04,dy:-0.04).contains(point) }) { continue }
        var minimum = 255, maximum = 0
        for j in -2...2 { for i in -2...2 {
          let value = Int(old[(y+j)*width+x+i]); minimum = min(minimum,value); maximum = max(maximum,value)
        } }
        guard maximum-minimum > 24 else { continue }
        var best = Int.max, second = Int.max, bx = 0, by = 0
        for dy in -4...4 { for dx in -4...4 {
          var error = 0
          for j in -2...2 { for i in -2...2 {
            error += abs(Int(old[(y+j)*width+x+i])-Int(now[(y+j+dy)*width+x+i+dx]))
          } }
          if error < best { second=best; best=error; bx=dx; by=dy }
          else { second=min(second,error) }
        } }
        if best < 350 && Double(best+1)/Double(second+1)<0.8 {
          vectors.append(CGPoint(x:bx,y:by))
        }
      }
    }
    return vectors
  }
}
