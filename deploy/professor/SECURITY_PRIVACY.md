# Security and privacy notes

MetaboGuard professor dashboard, prototype hosting. **Not suitable for
identifiable or clinical patient data.**

## Access control

- The professor enters a plaintext access key. The server hashes it with
  SHA-256 and compares the hex digest against `METABOGUARD_ACCESS_KEY_SHA256`
  using `hmac.compare_digest`, a constant-time comparison over two fixed-length
  64-character digests.
- The plaintext key is never stored, never logged, never written to disk, never
  echoed in a response and never placed in a URL. The browser holds it only in a
  controlled input, which is cleared as soon as the request completes.
- No key or hash is hardcoded anywhere in the source tree. `.env.example`
  contains placeholders only. If the variables are absent the server fails
  closed: login returns `503` and every protected route returns `401`.
- Failures are generic. An unknown key, an empty key and a malformed key all
  return the same `401 Invalid access key.` A rate-limited caller receives
  `429 Too many attempts. Try again later.` with a `Retry-After` header.
- Login rate limiting: 5 failures per client fingerprint within 5 minutes
  triggers a 15-minute lockout, which also blocks the correct key. The
  fingerprint is a truncated SHA-256 of the client address; the raw address is
  never logged or persisted.

## Sessions

- A successful login issues a signed token: base64url payload plus
  HMAC-SHA256 over that payload using `METABOGUARD_SESSION_SECRET`. There is no
  server-side session store, so no session state is persisted anywhere.
- Lifetime is 8 hours. Expiry is enforced from the signed `exp` claim, so an
  expired or tampered token cannot be replayed.
- The token is delivered as a cookie named `__Host-metaboguard-session` with
  `HttpOnly`, `Secure`, `SameSite=Strict`, `Path=/` and no `Domain` attribute.
  The `__Host-` prefix is mandatory: the published-site proxy strips any request
  cookie without it.
- The login response also returns the same signed token so the dashboard can
  hold it in React memory and send it as `Authorization: Bearer` in embedded
  preview contexts where the browser blocks cookie storage. It is never written
  to `localStorage`, `sessionStorage`, `IndexedDB` or a URL, and it is discarded
  on sign-out. **If you prefer cookie-only transport, remove `session_token`
  from the login response in `server/app.py` and the bearer fallback in
  `client/src/lib/api.js`; the trade-off is that the thread-preview iframe will
  not be able to authenticate.**
- Sign-out clears the cookie and the in-memory token, and subsequent protected
  calls return `401`.

## Route protection

- Unauthenticated: `/api/v1/health`, `/api/v1/status`, `/api/v1/session`,
  `/api/v1/auth/login`, `/api/v1/auth/logout`.
- Every model, simulation, dataset, evidence, reliability and clustering route depends on
  `require_session`, enforced server side. Bypassing the frontend route guard
  achieves nothing; the API refuses the request. Unknown `/api/...` paths return
  `404` rather than the SPA shell.
- Responses carry `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `X-Frame-Options: SAMEORIGIN`,
  `Cache-Control: no-store` and a restrictive `Permissions-Policy`.
- CORS is limited to `https://<subdomain>.pplx.app` with credentials; no
  wildcard origin is used.

## Uploaded data

- CSV only, 15 MB maximum, 20,000 rows maximum, 5,000 rows scored per request,
  with a wall-clock budget on analysis.
- An explicit checkbox confirming the file is de-identified research data is
  required by the API, not only by the UI.
- Direct-identifier columns are refused outright before any value is reported
  back: personal names, contact details, postal address, postcode/ZIP, NHS
  number, social security number, medical record or hospital number, exact date
  of birth, passport/licence/insurance numbers, device identifiers and
  next-of-kin fields. Object columns are also sampled for identifier-shaped
  values (email, SSN, NHS number, UK postcode, exact dates). Age and anonymous
  row identifiers are allowed.
- Outcome, label-derived and post-diagnosis TCGA columns (`Cancer`,
  `PancreaticCancer`, `tcga_*`, and the rest of the authoritative denylist) are
  reported as `prohibited` and excluded from every model input.
- Bytes are held in memory only. No temporary file is created, nothing is
  written to disk, and every exit path runs `shred_buffer`, which overwrites the
  upload buffer before releasing it. The parsed frame is dropped in the same
  `finally` block.
- Nothing is uploaded to a third party, and no request body is logged: uvicorn
  runs with `--no-access-log` and there is no telemetry or analytics of any kind.
- Results exports are rendered in memory from data the client already holds.
  There is no server-side retention and no history endpoint.

## Future-risk simulation data

- The browser generates deterministic synthetic histories from a numeric seed;
  the panel does not accept a patient record or file upload.
- `/api/v1/simulation/score` requires `simulation_mode=true`, at least two
  distinct visit times and a strict measurement-key allowlist. Names, record
  numbers, dates, free-text visit labels and arbitrary archetype text are
  refused before scoring.
- The request is scored in memory and discarded. No simulation history or result
  is persisted, logged or sent to a third party.
- `/api/v1/future-risk/score` is the clinical route and always returns `409`.
  The deployed artifact contains only selected models from the synthetic
  Synthea-derived run; the authoritative joblib/PyTorch artifact is never copied
  into the deployment.

## Static bundle

- The built bundle contains no key, no hash, no session secret and no private
  data. It ships the dashboard code, the `__PORT_5000__` deploy sentinel, and
  static copy only. Treat the bundle as public.
- Bundled JSON assets were sanitised so no authoring-machine filesystem path
  appears in any asset or API response; `server/reports.py` re-applies that
  sanitisation at load time.
- The frontend renders only the login screen until the server confirms a
  session.

## Residual risks and recommendations

1. A single shared access key gives no per-person accountability. Rotate the key
   after the review session by changing `METABOGUARD_ACCESS_KEY_SHA256`.
2. Rotating `METABOGUARD_SESSION_SECRET` invalidates all live sessions; there is
   no other revocation mechanism, because no session store is kept.
3. Rate limiting is per process and in memory, so it resets if the host restarts
   and is not shared across replicas.
4. Prototype hosting has no audit logging, no encryption at rest for uploads
   (there are none) and no formal access review. Do not process personal data
   under UK GDPR on this deployment.
