import CryptoKit
import Foundation
import UIKit
import UltralyticsYOLO

final class AerialEvidenceStore {
  private let queue = DispatchQueue(label: "aerial.evidence", qos: .utility)
  private var busy = false
  private var lastSave: [String: Double] = [:]
  private var lastAnySave = -Double.infinity
  private let folder: URL = FileManager.default.urls(
    for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("AerialEvidence", isDirectory: true)

  struct Item {
    let id: String
    let date: Date
    let summary: String
    let folder: URL
    var image: URL { folder.appendingPathComponent("annotated.jpg") }
    var original: URL { folder.appendingPathComponent("original.jpg") }
    var metadata: URL { folder.appendingPathComponent("metadata.json") }
  }

  func list(completion: @escaping (Result<[Item], Error>) -> Void) {
    queue.async { [self] in
      let result = Result { () throws -> [Item] in
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(
          at: folder,
          includingPropertiesForKeys: nil, options: .skipsHiddenFiles
        ).compactMap { directory in
          guard UUID(uuidString: directory.lastPathComponent) != nil,
            let data = try? Data(contentsOf: directory.appendingPathComponent("metadata.json")),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let timestamp = json["capturedAt"] as? String,
            let date = ISO8601DateFormatter().date(from: timestamp)
          else { return nil }
          let objects = json["objects"] as? [[String: Any]] ?? []
          let labels = Set(objects.compactMap { $0["displayClass"] as? String }).sorted()
          return Item(
            id: directory.lastPathComponent, date: date,
            summary: labels.joined(separator: " · "), folder: directory)
        }.sorted { $0.date > $1.date }
      }
      DispatchQueue.main.async { completion(result) }
    }
  }

  func delete(_ item: Item, completion: @escaping (Result<Void, Error>) -> Void) {
    queue.async { [self] in
      let result = Result {
        guard UUID(uuidString: item.id) != nil,
          item.folder.standardizedFileURL
            == folder.appendingPathComponent(item.id, isDirectory: true).standardizedFileURL
        else { throw CocoaError(.fileWriteInvalidFileName) }
        try FileManager.default.removeItem(at: item.folder)
      }
      DispatchQueue.main.async { completion(result) }
    }
  }

  /// Call on main. A source frame and metadata from the same inference are kept together.
  func record(
    result: YOLOResult, model: String,
    completion: @escaping (Result<URL, Error>) -> Void
  ) {
    let now = CACurrentMediaTime()
    let eligible = result.boxes.filter { $0.aerialDisplayLabel != nil && $0.conf >= 0.30 }
    let keys = eligible.compactMap(\.aerialDisplayLabel)
    guard !busy, now - lastAnySave >= 5, let image = result.originalImage,
      !eligible.isEmpty, keys.contains(where: { now - (lastSave[$0] ?? -.infinity) >= 30 })
    else { return }
    busy = true
    let id = UUID().uuidString
    let timestamp = ISO8601DateFormatter().string(from: Date())
    let objects: [[String: Any]] = eligible.map {
      [
        "displayClass": $0.aerialDisplayLabel ?? "미확인 물체",
        "modelClass": $0.cls,
        "classScores": $0.classScores ?? [:],
        "score": $0.conf,
        "normalizedBox": [$0.xywhn.minX, $0.xywhn.minY, $0.xywhn.width, $0.xywhn.height],
      ]
    }
    queue.async { [self] in
      let outcome: Result<URL, Error>
      do {
        let manager = FileManager.default
        try manager.createDirectory(
          at: folder, withIntermediateDirectories: true,
          attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication])
        var excluded = folder
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try excluded.setResourceValues(values)
        let files = manager.enumerator(at: folder, includingPropertiesForKeys: [.fileSizeKey])
        let used = (files?.allObjects as? [URL] ?? []).reduce(0) { total, url in
          total + ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        let free =
          (try manager.attributesOfFileSystem(forPath: folder.path)[.systemFreeSize] as? NSNumber)?
          .int64Value ?? 0
        guard used < 500_000_000, free > 100_000_000 else {
          throw NSError(
            domain: "Evidence", code: 1,
            userInfo: [
              NSLocalizedDescriptionKey:
                "채증 저장 공간이 부족해 자동 채증을 껐습니다. 앱 내 채증 갤러리를 정리한 뒤 다시 켜 주세요."
            ])
        }
        guard let raw = image.jpegData(compressionQuality: 0.9) else {
          throw NSError(
            domain: "Evidence", code: 2, userInfo: [NSLocalizedDescriptionKey: "채증 이미지를 만들 수 없습니다."]
          )
        }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let annotated = UIGraphicsImageRenderer(size: image.size, format: format).image {
          renderer in
          image.draw(at: .zero)
          for detection in eligible {
            let box = CGRect(
              x: detection.xywhn.minX * image.size.width, y: detection.xywhn.minY * image.size.height,
              width: detection.xywhn.width * image.size.width,
              height: detection.xywhn.height * image.size.height)
            renderer.cgContext.setStrokeColor(
              UIColor.systemYellow.cgColor)
            renderer.cgContext.setLineWidth(3)
            renderer.cgContext.stroke(box)
            let text = "\(detection.aerialDisplayLabel ?? detection.cls) \(Int(detection.conf * 100))점"
            (text as NSString).draw(
              at: CGPoint(x: box.minX, y: max(0, box.minY - 26)),
              withAttributes: [
                .font: UIFont.systemFont(ofSize: 22), .foregroundColor: UIColor.white,
                .backgroundColor: UIColor.black,
              ])
          }
        }
        guard let marked = annotated.jpegData(compressionQuality: 0.9) else {
          throw NSError(
            domain: "Evidence", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "표시 이미지를 저장할 수 없습니다."])
        }
        guard used + raw.count + marked.count + 65_536 <= 500_000_000,
          free > Int64(raw.count + marked.count) + 100_000_000
        else {
          throw NSError(
            domain: "Evidence", code: 1,
            userInfo: [
              NSLocalizedDescriptionKey:
                "채증 갤러리 공간이 부족합니다. 필요한 자료를 내보내고 정리한 뒤 자동 채증을 다시 켜 주세요."
            ])
        }
        let temporary = folder.appendingPathComponent(".pending-" + id, isDirectory: true)
        let destination = folder.appendingPathComponent(id, isDirectory: true)
        try manager.createDirectory(at: temporary, withIntermediateDirectories: false)
        do {
          try raw.write(to: temporary.appendingPathComponent("original.jpg"), options: .atomic)
          try marked.write(to: temporary.appendingPathComponent("annotated.jpg"), options: .atomic)
          let metadata: [String: Any] = [
            "eventID": id, "capturedAt": timestamp, "model": model,
            "objects": objects,
            "originalSHA256": SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined(),
            "note":
              "Model estimates, not verified identity. No location collected. Original is the inference frame JPEG.",
          ]
          try JSONSerialization.data(
            withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]
          )
          .write(to: temporary.appendingPathComponent("metadata.json"), options: .atomic)
          try manager.moveItem(at: temporary, to: destination)
          outcome = .success(destination)
        } catch {
          try? manager.removeItem(at: temporary)
          throw error
        }
      } catch { outcome = .failure(error) }
      DispatchQueue.main.async { [self] in
        busy = false
        if case .success = outcome {
          lastAnySave = now
          for key in keys { lastSave[key] = now }
          lastSave = lastSave.filter { now - $0.value < 300 }
        }
        completion(outcome)
      }
    }
  }
}
