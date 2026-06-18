# Test Cases — luke-agents (dev/qa)

These cover the abuse-hardening fixes merged to `develop` (auto-deployed to **dev/qa**).
Run them against your **dev/qa** agents URL.

## Before you start (setup once)
- Your dev/qa agents URL — call it `AGENTS` (e.g. `https://luke-agents-dev.onrender.com`).
- A terminal for `curl`.

---

## ✅ Item 1 — You can't dodge the rate limit by faking a user id (PR #42)
**What changed:** the limit used to count per `user_id` (which anyone can change). Now it counts per **IP address**, so changing `user_id` no longer gives you a fresh budget.

### Test 1a — the limit still lets normal use through
1. Send one normal request:
   ```
   curl -i -X POST "$AGENTS/chat" -H "Content-Type: application/json" \
     -d '{"message":"add an email field","user_id":"me"}'
   ```
- ✅ **PASS:** `200` with a reply.

### Test 1b — changing user_id does NOT reset the budget
> This proves the bypass is closed. (Optional, a bit advanced.)
1. Run many requests quickly, each with a **different** `user_id`. Easy loop:
   ```
   for i in $(seq 1 250); do
     curl -s -o /dev/null -w "%{http_code}\n" -X POST "$AGENTS/chat" \
       -H "Content-Type: application/json" -d "{\"message\":\"hi\",\"user_id\":\"user$i\"}"
   done | sort | uniq -c
   ```
2. Look at the counts of `200` vs `429`.
- ✅ **PASS:** after ~200 you start getting `429` (Too Many Requests) **even though every user_id is different**.
- ❌ **FAIL:** all `200`, never `429` → tell the dev (bypass still open).

---

## ✅ Item 2 — Giant inputs are rejected (PR #42)
**What changed:** very long messages / huge schemas are refused so they can't run up cost or memory.

### Test 2a — a too-long message is refused
1. ```
   curl -i -X POST "$AGENTS/chat" -H "Content-Type: application/json" \
     -d "{\"message\":\"$(python3 -c 'print("x"*20000)')\"}"
   ```
- ✅ **PASS:** `422 Unprocessable Entity` (too long).
- ❌ **FAIL:** `200` → tell the dev.

### Test 2b — a normal message is fine
1. Send a short normal message (like Test 1a).
- ✅ **PASS:** `200`.

---

## ✅ Item 3 — Feedback can't be spammed (PR #42)
**What changed:** the `/feedback` endpoint had **no** limit and accepted any rating number. Now it's rate-limited and the rating must be -1, 0, or 1.

### Test 3a — a silly rating is refused
1. ```
   curl -i -X POST "$AGENTS/feedback" -H "Content-Type: application/json" \
     -d '{"turn_id":"abc","rating":9999}'
   ```
- ✅ **PASS:** `422` (rating out of range).
- ❌ **FAIL:** `200`/ok → tell the dev.

---

## ✅ Item 4 — Optional API key lock (PR #42, OFF by default)
**What changed:** there's a new optional lock. It's **off** unless someone sets `AGENTS_API_KEY` on the service.

### Test 4a — default (no key set): app still works
1. Use the consumer app's AI form builder normally in dev/qa.
- ✅ **PASS:** it works (the lock is off by default; nothing breaks).

### Test 4b — (only if you turn the lock on) requests need the key
> Optional. To fully close "anyone can call the paid AI" you'd set `AGENTS_API_KEY` on the service AND make the app send it server-side. After setting it:
1. ```
   curl -i -X POST "$AGENTS/chat" -H "Content-Type: application/json" -d '{"message":"hi"}'
   ```
- ✅ **PASS:** `401` (key required) when no `X-Agents-Key` header is sent.

---

## Notes
- Deployed via PR #42 (merged to `develop`).
- Full "no anonymous access to paid AI" needs Item 4 turned on **and** the app routed through the gateway so the key isn't exposed in the browser — that's a follow-up.
