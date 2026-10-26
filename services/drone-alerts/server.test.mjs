import test from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { alertServer, openStore, invite, apnsSender } from "./server.mjs";

async function fixture(t, push) {
  const db = openStore(":memory:");
  let clock = Date.now();
  const sent = [];
  const server = alertServer({ db, now: () => clock, sendPush: push ?? (async (token) => {
    sent.push(token); return { accepted: true };
  }) });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(async () => { await new Promise((resolve) => server.close(resolve)); db.close(); });
  const url = "http://127.0.0.1:" + server.address().port;
  const request = async (path, method, token, body) => {
    const result = await fetch(url + "/v1/" + path, { method,
      headers: { "Content-Type": "application/json", ...(token ? { Authorization: "Bearer " + token } : {}) },
      body: body === undefined ? undefined : JSON.stringify(body) });
    return { status: result.status, data: await result.json() };
  };
  const enroll = async (channel, role) => {
    const code = invite(db, channel, role, clock);
    const result = await request("enroll", "POST", null, { invite: code });
    assert.equal(result.status, 201);
    return { code, ...result.data };
  };
  const event = () => ({ eventID: randomUUID(), score: 0.75, observedAt: clock / 1000 });
  return { db, sent, request, enroll, event, advance: (ms) => { clock += ms; } };
}

test("single-use invites, hashed credentials, role and channel boundaries", async (t) => {
  const f = await fixture(t);
  const pub = await f.enroll("team-a", "publisher");
  const sub = await f.enroll("team-a", "subscriber");
  const other = await f.enroll("team-b", "subscriber");
  assert.equal((await f.request("enroll", "POST", null, { invite: pub.code })).status, 403);
  assert.equal(f.db.prepare("SELECT 1 FROM members WHERE hash=?").get(pub.token), undefined);
  assert.equal((await f.request("events", "POST", null, f.event())).status, 401);
  assert.equal((await f.request("events", "POST", sub.token, f.event())).status, 403);
  await f.request("device", "PUT", pub.token, { deviceToken: "a".repeat(64) });
  await f.request("device", "PUT", sub.token, { deviceToken: "b".repeat(64) });
  await f.request("device", "PUT", other.token, { deviceToken: "c".repeat(64) });
  const event = f.event();
  assert.equal((await f.request("events", "POST", pub.token, event)).data.accepted, 1);
  assert.deepEqual(f.sent, ["b".repeat(64)]);
  assert.equal((await f.request("events", "POST", pub.token, event)).data.duplicate, true);
  assert.equal(f.sent.length, 1);
  assert.equal((await f.request("events", "POST", pub.token, f.event())).status, 429);
  await f.request("membership", "DELETE", sub.token);
  f.advance(11000);
  assert.equal((await f.request("events", "POST", pub.token, f.event())).data.accepted, 0);
  assert.equal((await f.request("device", "PUT", sub.token, { deviceToken: "b".repeat(64) })).status, 401);
});

test("reject stale, low-score, malformed and oversized events without a push", async (t) => {
  const f = await fixture(t), pub = await f.enroll("team", "publisher");
  for (const event of [
    { ...f.event(), score: 0.69 }, { ...f.event(), score: 2 },
    { ...f.event(), observedAt: 0 }, { ...f.event(), eventID: "invalid" }
  ]) assert.equal((await f.request("events", "POST", pub.token, event)).status, 400);
  assert.equal((await f.request("events", "POST", pub.token, { extra: "x".repeat(10000) })).status, 413);
  assert.deepEqual(f.sent, []);
});

test("APNs rejection counts failure, invalid token is removed, no automatic retry", async (t) => {
  let calls = 0;
  const f = await fixture(t, async () => { calls++; return { accepted: false, invalid: true }; });
  const pub = await f.enroll("team", "publisher"), sub = await f.enroll("team", "subscriber");
  await f.request("device", "PUT", sub.token, { deviceToken: "d".repeat(64) });
  const event = f.event();
  const result = await f.request("events", "POST", pub.token, event);
  assert.equal(result.data.accepted, 0); assert.equal(result.data.failed, 1);
  await f.request("events", "POST", pub.token, event);
  f.advance(11000);
  await f.request("events", "POST", pub.token, f.event());
  assert.equal(calls, 1);
});

test("expired invites and memberships cannot enroll or publish", async (t) => {
  const f = await fixture(t), pub = await f.enroll("team", "publisher");
  const code = invite(f.db, "team", "subscriber", Date.now() - 2 * 86400000);
  assert.equal((await f.request("enroll", "POST", null, { invite: code })).status, 403);
  f.advance(91 * 86400000);
  assert.equal((await f.request("events", "POST", pub.token, f.event())).status, 401);
});

test("missing APNs credentials fail startup rather than pretend delivery works", () => {
  assert.throws(() => apnsSender({}), /Configure APNS/);
});
