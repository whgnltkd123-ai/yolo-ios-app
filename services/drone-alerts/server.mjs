import { createServer } from "node:http";
import { connect } from "node:http2";
import { createHash, createPrivateKey, randomBytes, randomUUID, sign } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { readFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const digest = (value) => createHash("sha256").update(value).digest("hex");
const secret = () => randomBytes(32).toString("base64url");
const failure = (status, message) => Object.assign(new Error(message), { status });

export function openStore(path) {
  if (path !== ":memory:") mkdirSync(dirname(resolve(path)), { recursive: true, mode: 0o700 });
  const db = new DatabaseSync(path);
  db.exec("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;");
  db.exec("CREATE TABLE IF NOT EXISTS invites(hash TEXT PRIMARY KEY, channel TEXT NOT NULL, role TEXT NOT NULL, expires INTEGER NOT NULL);" +
    "CREATE TABLE IF NOT EXISTS members(id TEXT PRIMARY KEY, hash TEXT UNIQUE NOT NULL, channel TEXT NOT NULL, role TEXT NOT NULL, device TEXT, expires INTEGER NOT NULL);" +
    "CREATE TABLE IF NOT EXISTS events(id TEXT NOT NULL, sender TEXT NOT NULL, created INTEGER NOT NULL, accepted INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'sending', PRIMARY KEY(sender,id));");
  return db;
}

export function invite(db, channel, role, now = Date.now()) {
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(channel) || !["publisher", "subscriber"].includes(role))
    throw new Error("Use a channel name and publisher or subscriber role.");
  const value = secret();
  db.prepare("INSERT INTO invites VALUES(?,?,?,?)").run(digest(value), channel, role, now + 86400000);
  return value;
}

async function readBody(req) {
  if (!String(req.headers["content-type"] ?? "").startsWith("application/json"))
    throw failure(415, "application/json required");
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 8192) throw failure(413, "Request too large");
    chunks.push(chunk);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error();
    return value;
  } catch { throw failure(400, "Invalid JSON"); }
}

// APNs acceptance is not proof that a subscriber saw or felt a notification.
export function apnsSender(env) {
  const { APNS_KEY_FILE, APNS_KEY_ID, APNS_TEAM_ID, APNS_TOPIC, APNS_ENVIRONMENT } = env;
  if (!APNS_KEY_FILE || !APNS_KEY_ID || !APNS_TEAM_ID || !APNS_TOPIC ||
      !["sandbox", "production"].includes(APNS_ENVIRONMENT))
    throw new Error("Configure APNS_KEY_FILE, APNS_KEY_ID, APNS_TEAM_ID, APNS_TOPIC and APNS_ENVIRONMENT.");
  const key = createPrivateKey(readFileSync(APNS_KEY_FILE));
  if (key.asymmetricKeyType !== "ec" || key.asymmetricKeyDetails?.namedCurve !== "prime256v1")
    throw new Error("An APNs ES256 P-256 signing key is required.");
  let jwt, jwtTime = 0;
  const url = APNS_ENVIRONMENT === "sandbox" ? "https://api.sandbox.push.apple.com" : "https://api.push.apple.com";
  return async (device, event) => {
    const now = Math.floor(Date.now() / 1000);
    if (!jwt || now - jwtTime > 1200) {
      const header = Buffer.from(JSON.stringify({ alg: "ES256", kid: APNS_KEY_ID })).toString("base64url");
      const payload = Buffer.from(JSON.stringify({ iss: APNS_TEAM_ID, iat: now })).toString("base64url");
      const value = header + "." + payload;
      jwt = value + "." + sign("sha256", Buffer.from(value), { key, dsaEncoding: "ieee-p1363" }).toString("base64url");
      jwtTime = now;
    }
    return new Promise((resolveResult) => {
      const session = connect(url);
      let done = false, status = 0, body = "";
      const finish = (result) => {
        if (done) return;
        done = true; clearTimeout(timer); session.destroy(); resolveResult(result);
      };
      const timer = setTimeout(() => finish({ accepted: false, reason: "Timeout" }), 8000);
      session.on("error", () => finish({ accepted: false, reason: "ConnectionError" }));
      const request = session.request({
        ":method": "POST", ":path": "/3/device/" + device, authorization: "bearer " + jwt,
        "apns-topic": APNS_TOPIC, "apns-push-type": "alert", "apns-priority": "10",
        "apns-expiration": String(now + 30),
        "apns-collapse-id": "drone-" + digest(event.channel).slice(0, 24),
        "content-type": "application/json"
      });
      request.setEncoding("utf8");
      request.on("response", (headers) => { status = Number(headers[":status"]); });
      request.on("data", (data) => { if (body.length < 4096) body += data; });
      request.on("error", () => finish({ accepted: false, reason: "StreamError" }));
      request.on("end", () => {
        let reason;
        try { reason = JSON.parse(body).reason; } catch { /* APNs success has no JSON body. */ }
        finish({ accepted: status === 200, invalid: status === 410 || reason === "BadDeviceToken", reason });
      });
      request.end(JSON.stringify({
        aps: { alert: { title: "드론 식별 알림",
          body: "관측 기기에서 드론으로 판단했습니다 (" + Math.round(event.score * 100) + "점). 주변 상황을 확인하세요." },
          sound: "default", "thread-id": "drone-alerts" },
        eventID: event.id
      }));
    });
  };
}

export function alertServer({ db, sendPush, now = Date.now }) {
  const attempts = new Map();
  let activeFanouts = 0;
  const authenticate = (req) => {
    const header = req.headers.authorization ?? "";
    if (!/^Bearer [A-Za-z0-9_-]{43}$/.test(header)) throw failure(401, "Authentication required");
    const member = db.prepare("SELECT * FROM members WHERE hash=? AND expires>?")
      .get(digest(header.slice(7)), now());
    if (!member) throw failure(401, "Authentication required");
    return member;
  };
  const server = createServer(async (req, res) => {
    const reply = (status, data) => {
      res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      res.end(JSON.stringify(data));
    };
    try {
      const route = req.method + " " + req.url;
      if (route === "GET /health") return reply(200, { status: "ready" });
      if (route === "POST /v1/enroll") {
        const time = now();
        for (const [ip, value] of attempts) if (value.until < time) attempts.delete(ip);
        const ip = req.socket.remoteAddress;
        if (!attempts.has(ip) && attempts.size >= 1000) throw failure(429, "Try later");
        const limit = attempts.get(ip) ?? { count: 0, until: time + 60000 };
        attempts.set(ip, limit);
        if (++limit.count > 10) throw failure(429, "Try later");
        const body = await readBody(req);
        if (typeof body.invite !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(body.invite))
          throw failure(403, "Invalid invite");
        const invitation = db.prepare("SELECT * FROM invites WHERE hash=? AND expires>?")
          .get(digest(body.invite), time);
        if (!invitation) throw failure(403, "Invalid invite");
        if (db.prepare("SELECT COUNT(*) AS count FROM members WHERE channel=? AND expires>?")
          .get(invitation.channel, time).count >= 100) throw failure(409, "Channel full");
        const token = secret();
        db.exec("BEGIN IMMEDIATE");
        try {
          db.prepare("DELETE FROM invites WHERE hash=?").run(invitation.hash);
          db.prepare("INSERT INTO members(id,hash,channel,role,expires) VALUES(?,?,?,?,?)")
            .run(randomUUID(), digest(token), invitation.channel, invitation.role, time + 90 * 86400000);
          db.exec("COMMIT");
        } catch (error) { db.exec("ROLLBACK"); throw error; }
        return reply(201, { token, canPublish: invitation.role === "publisher" });
      }
      const member = authenticate(req);
      if (route === "PUT /v1/device") {
        const { deviceToken } = await readBody(req);
        if (typeof deviceToken !== "string" || !/^[0-9a-f]{64,200}$/.test(deviceToken))
          throw failure(400, "Invalid device token");
        // A phone belongs to one active membership. Explicit enrollment switches the account.
        db.prepare("UPDATE members SET device=NULL WHERE device=?").run(deviceToken);
        db.prepare("UPDATE members SET device=? WHERE id=?").run(deviceToken, member.id);
        return reply(200, { registered: true });
      }
      if (route === "DELETE /v1/membership") {
        db.prepare("DELETE FROM members WHERE id=?").run(member.id);
        return reply(200, { removed: true });
      }
      if (route !== "POST /v1/events") throw failure(404, "Not found");
      if (member.role !== "publisher") throw failure(403, "Publisher role required");
      const { eventID, score, observedAt } = await readBody(req);
      authenticate(req); // Revocation may have occurred while the request body was arriving.
      if (typeof eventID !== "string" || !/^[0-9a-f-]{36}$/i.test(eventID) ||
          !Number.isFinite(score) || score < 0.60 || score > 1 ||
          !Number.isFinite(observedAt) || Math.abs(now() / 1000 - observedAt) > 60)
        throw failure(400, "Invalid or stale detection event");
      const existing = db.prepare("SELECT accepted,failed,state FROM events WHERE sender=? AND id=?")
        .get(member.id, eventID);
      if (existing) return reply(200, { ...existing, duplicate: true });
      if (activeFanouts >= 4) throw failure(503, "Delivery queue busy");
      if (db.prepare("SELECT 1 FROM events WHERE sender=? AND created>? LIMIT 1")
        .get(member.id, now() - 10000)) throw failure(429, "Alert rate limit");
      db.prepare("DELETE FROM events WHERE created<?").run(now() - 7 * 86400000);
      db.prepare("DELETE FROM invites WHERE expires<?").run(now());
      db.prepare("DELETE FROM members WHERE expires<?").run(now());
      db.prepare("INSERT INTO events(id,sender,created) VALUES(?,?,?)").run(eventID, member.id, now());
      const recipients = db.prepare("SELECT id,device FROM members WHERE channel=? AND id<>? AND device IS NOT NULL AND expires>?")
        .all(member.channel, member.id, now());
      activeFanouts++;
      let accepted = 0, failed = 0, cursor = 0;
      try {
        // APNs deliveries have a 30-second deadline. Never replay stale detections.
        const deadline = now() + 30000;
        await Promise.all(Array.from({ length: Math.min(4, recipients.length) }, async () => {
          while (cursor < recipients.length) {
            const recipient = recipients[cursor++];
            if (!db.prepare("SELECT 1 FROM members WHERE id=? AND device=?")
              .get(recipient.id, recipient.device)) continue;
            if (now() >= deadline) { failed++; continue; }
            let result;
            try { result = await sendPush(recipient.device, { id: eventID, score, channel: member.channel }); }
            catch { result = { accepted: false }; }
            if (result.accepted) accepted++; else failed++;
            if (result.invalid) db.prepare("UPDATE members SET device=NULL WHERE id=? AND device=?")
              .run(recipient.id, recipient.device);
          }
        }));
        db.prepare("UPDATE events SET accepted=?,failed=?,state='complete' WHERE sender=? AND id=?")
          .run(accepted, failed, member.id, eventID);
      } finally { activeFanouts--; }
      return reply(200, { accepted, failed, state: "complete" });
    } catch (error) {
      reply(error.status ?? 500, { error: error.status ? error.message : "Internal service error" });
    }
  });
  server.requestTimeout = 15000;
  server.headersTimeout = 10000;
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const db = openStore(process.env.DATABASE_PATH ?? "./data/alerts.sqlite");
  if (process.argv[2] === "invite") {
    console.log(invite(db, process.argv[3], process.argv[4]));
    db.close();
  } else {
    const server = alertServer({ db, sendPush: apnsSender(process.env) });
    server.listen(Number(process.env.PORT ?? 8080), "127.0.0.1", () => console.log("Alert service ready on loopback"));
    const shutdown = () => server.close(() => { db.close(); process.exit(0); });
    process.on("SIGTERM", shutdown); process.on("SIGINT", shutdown);
  }
}
