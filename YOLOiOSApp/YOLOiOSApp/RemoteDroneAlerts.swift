import Foundation
import Security
import UIKit
import UserNotifications

/// Opt-in, authenticated group alerts. Photos and location never leave the phone.
@MainActor
final class RemoteDroneAlerts: NSObject, UNUserNotificationCenterDelegate {
  static let shared = RemoteDroneAlerts()
  private struct Membership: Codable {
    var server: URL
    var token: String
    var canPublish: Bool
  }
  private var membership: Membership?
  private var deviceToken: String?
  private var lastPublished = -Double.infinity
  private var lastPublishedTrack: [Int: Double] = [:]
  private var publishing = false
  var statusChanged: ((String) -> Void)?
  private(set) var status = "가입자 알림 미연결" {
    didSet { statusChanged?(status) }
  }
  var connected: Bool { membership != nil }
  var canPublish: Bool { membership?.canPublish == true }
  private let service = "aerial.notification.membership"

  override private init() {
    super.init()
    var query = keychainQuery
    query[kSecReturnData as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var result: CFTypeRef?
    if SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
      let data = result as? Data
    {
      membership = try? JSONDecoder().decode(Membership.self, from: data)
    }
    if membership != nil { status = "알림 연결 확인 필요" }
  }

  private var keychainQuery: [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
      kSecAttrAccount as String: "membership",
    ]
  }

  func restoreRegistration() {
    UNUserNotificationCenter.current().delegate = self
    guard connected else { return }
    UNUserNotificationCenter.current().getNotificationSettings { settings in
      Task { @MainActor in
        if settings.authorizationStatus == .authorized
          || settings.authorizationStatus == .provisional
        {
          UIApplication.shared.registerForRemoteNotifications()
        } else {
          self.status = "알림 권한이 꺼져 있습니다"
        }
      }
    }
  }

  func join(serverText: String, invite: String) async throws {
    guard !connected else { throw message("기존 알림 구독을 먼저 해제해 주세요.") }
    guard let server = URL(string: serverText.trimmingCharacters(in: .whitespacesAndNewlines)),
      server.scheme == "https", server.host != nil, server.user == nil, server.password == nil,
      server.query == nil, server.fragment == nil, server.path.isEmpty || server.path == "/"
    else { throw message("유효한 HTTPS 알림 서버 주소를 입력하세요.") }
    let granted = try await UNUserNotificationCenter.current().requestAuthorization(options: [
      .alert, .sound, .badge,
    ])
    guard granted else { throw message("아이폰 설정에서 알림을 허용해 주세요.") }
    let result = try await request(
      server: server, path: "enroll", token: nil,
      method: "POST", body: ["invite": invite.trimmingCharacters(in: .whitespacesAndNewlines)])
    guard let token = result["token"] as? String, let publish = result["canPublish"] as? Bool else {
      throw message("서버의 가입 응답을 확인할 수 없습니다.")
    }
    let value = Membership(server: server, token: token, canPublish: publish)
    var query = keychainQuery
    query[kSecValueData as String] = try JSONEncoder().encode(value)
    query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
    let saved = SecItemAdd(query as CFDictionary, nil)
    guard saved == errSecSuccess else {
      _ = try? await request(
        server: server, path: "membership", token: token, method: "DELETE", body: nil)
      throw message("알림 연결 정보를 안전하게 저장하지 못했습니다.")
    }
    membership = value
    status = "가입 완료 · 푸시 기기 등록 중"
    UNUserNotificationCenter.current().delegate = self
    UIApplication.shared.registerForRemoteNotifications()
  }

  func registered(token: Data) {
    deviceToken = token.map { String(format: "%02x", $0) }.joined()
    guard let membership, let deviceToken else { return }
    Task {
      do {
        _ = try await request(
          server: membership.server, path: "device", token: membership.token,
          method: "PUT", body: ["deviceToken": deviceToken])
        status = "가입자 알림 연결됨"
      } catch { status = "푸시 등록 실패: \(error.localizedDescription)" }
    }
  }

  func registrationFailed(_ error: Error) {
    status = "푸시 등록 실패 · 푸시 권한이 포함된 앱 서명을 확인하세요"
  }

  func leave() async throws {
    guard let membership else { return }
    do {
      _ = try await request(
        server: membership.server, path: "membership", token: membership.token,
        method: "DELETE", body: nil)
    } catch let error as NSError where error.domain == "DroneAlertsHTTP" && error.code == 401 {
      // Expired or already revoked server membership can be removed locally.
    }
    let removed = SecItemDelete(keychainQuery as CFDictionary)
    guard removed == errSecSuccess || removed == errSecItemNotFound else {
      throw message("구독은 해제됐지만 저장된 연결 정보를 삭제하지 못했습니다.")
    }
    self.membership = nil
    deviceToken = nil
    UIApplication.shared.unregisterForRemoteNotifications()
    status = "가입자 알림 해제됨"
  }

  func publish(trackID: Int, score: Float) {
    let now = CACurrentMediaTime()
    guard let membership, membership.canPublish, !publishing,
      score >= 0.60, now - lastPublished >= 10,
      now - (lastPublishedTrack[trackID] ?? -.infinity) >= 60
    else { return }
    publishing = true
    // Bound attempts even offline; this is a live alert, never replay a stale warning later.
    lastPublished = now
    lastPublishedTrack[trackID] = now
    lastPublishedTrack = lastPublishedTrack.filter { now - $0.value < 600 }
    let id = UUID().uuidString
    Task {
      defer { publishing = false }
      do {
        let response = try await request(
          server: membership.server, path: "events", token: membership.token,
          method: "POST",
          body: ["eventID": id, "score": score, "observedAt": Date().timeIntervalSince1970])
        let count = response["accepted"] as? Int ?? 0
        let failed = response["failed"] as? Int ?? 0
        status = "푸시 접수 \(count)대 / 실패 \(failed)대 · 실제 수신 별도 확인"
      } catch { status = "가입자 알림 발송 실패: \(error.localizedDescription)" }
    }
  }

  private func request(
    server: URL, path: String, token: String?, method: String,
    body: [String: Any]?
  ) async throws -> [String: Any] {
    var request = URLRequest(url: server.appendingPathComponent("v1").appendingPathComponent(path))
    request.httpMethod = method
    request.timeoutInterval = 15
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let token { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
    if let body { request.httpBody = try JSONSerialization.data(withJSONObject: body) }
    let (data, response) = try await URLSession.shared.data(
      for: request, delegate: NoAlertRedirect.shared)
    guard let response = response as? HTTPURLResponse else { throw message("서버 응답을 확인할 수 없습니다.") }
    guard (200...299).contains(response.statusCode) else {
      throw NSError(
        domain: "DroneAlertsHTTP", code: response.statusCode,
        userInfo: [NSLocalizedDescriptionKey: "서버 연결 또는 초대코드·발송 권한을 확인해 주세요."])
    }
    return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
  }

  private func message(_ text: String) -> NSError {
    NSError(domain: "DroneAlerts", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
  }

  nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter,
    willPresent notification: UNNotification,
    withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
  ) {
    completionHandler([.banner, .list, .sound])
  }
}

private final class NoAlertRedirect: NSObject, URLSessionTaskDelegate {
  static let shared = NoAlertRedirect()
  func urlSession(
    _ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    // Never forward an invite or bearer credential to a redirected host.
    completionHandler(nil)
  }
}
