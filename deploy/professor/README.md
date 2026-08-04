# MetaboGuard - Professor Research Review Dashboard

An access-controlled, non-diagnostic review deployment of the MetaboGuard
self-supervised metabolic deviation model (KERI department). It serves a FastAPI
backend and a React single-page dashboard from one process on port 5000, and it
runs on free Perplexity published-site hosting with no paid services, no runtime
LLM calls and no external connectors.

**This is prototype hosting. It is not a clinical system and must never receive
identifiable or clinical patient data.**

---

## What it does

| Section                    | Contents                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Overview**               | Non-diagnostic discovery posture, model/version/capability, architecture, status cards, current data limitations.               |
| **Patient Probe**          | The existing probe form plus `Generate synthetic patient`. Explicit scoring only; deviation score, percentile, feature contributions, evidence boundaries. |
| **Dataset Analysis**       | CSV-only drag/drop or picker, de-identification checkbox, identifier and leakage screening, schema mapping, missingness, range violations, feature tiers, rows accepted/rejected, then in-memory scoring and a downloadable results CSV. |
| **Reliability & Clusters** | The pipeline's fail-closed reliability report, feature tiers, and the `no_stable_clusters` abstention with the survey-cycle explanation. |
| **Evidence & Methods**     | Source-linked biomarker catalogue with evidence grades, multi-marker rationale, PRoBE and TRIPOD+AI references, supported vs prohibited claims. |

No output is a diagnosis, a disease probability, a future-risk horizon or a
cancer-type claim. The future-risk head stays fail-closed and clustering
abstains, exactly as in the research pipeline.

---

## Install, build, run

```bash
# 0. In the KERI repository only: materialise generated files
#    (server/core/, client/src/lib/synthetic_patient*.js and assets/)
#    from the authoritative model artifact, research run and evidence catalogue.
#    The sandbox deployment copy already contains them.
.venv/bin/python deploy/professor/prepare_assets.py

# 1. Python runtime (no PyTorch, no XGBoost, no ChromaDB)
pip install -r requirements-deploy.txt

# 2. Frontend build -> client/dist
cd client && npm ci && npm run build && cd ..

# 3. Production start (FastAPI + built client on port 5000)
./start.sh
```

Required environment variables (no defaults, no fallbacks):

| Variable                         | Meaning                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| `METABOGUARD_ACCESS_KEY_SHA256`  | SHA-256 hex digest of the plaintext access key given to the professor.   |
| `METABOGUARD_SESSION_SECRET`     | Random secret (>= 32 bytes of entropy) used to sign session cookies.     |
| `PORT`                           | Optional. Defaults to `5000`, the port the published sandbox proxies.    |

Generate them without ever storing the key:

```bash
printf '%s' 'YOUR-ACCESS-KEY' | shasum -a 256          # -> METABOGUARD_ACCESS_KEY_SHA256
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # -> METABOGUARD_SESSION_SECRET
```

If either variable is missing, `/api/v1/status` reports
`auth_configured: false`, login returns `503`, and every protected route stays
closed. See `.env.example` for placeholders only.

## Deployment (Perplexity hosting)

```
deploy_website(project_path=".../client/dist", site_name="MetaboGuard Research Review")
publish_website(
  project_path=".../metaboguard-professor-dashboard",
  dist_path=".../metaboguard-professor-dashboard/client/dist",
  install_command="pip install -r requirements-deploy.txt",
  run_command="./start.sh",
  port=5000,
  credentials={"METABOGUARD_ACCESS_KEY_SHA256": "...", "METABOGUARD_SESSION_SECRET": "..."},
)
```

`client/src/lib/api.js` contains the `__PORT_5000__` sentinel that the deploy
step rewrites to the backend proxy path. Locally the sentinel is left in place
and the client falls back to same-origin relative requests, so development and
deployment both work from the same build.

## Tests

```bash
pip install -r requirements-test.txt && python3 -m pytest -q      # 98 API/unit tests
cd client && npm test                                              # 46 frontend tests
```

## Repository layout

```
main.py                     uvicorn entrypoint (also importable as server.app:app)
start.sh                    production start command, port 5000
requirements-deploy.txt     runtime deps (no PyTorch)
server/config.py            paths, limits, env variable names
server/auth.py              SHA-256 key check, signed __Host- session cookie, rate limiting
server/dataset.py           CSV intake: identifier/leakage screening, tiers, ephemeral parsing
server/model.py             NumPy inference wrapper, aggregates, in-memory results CSV
server/reports.py           reliability, clustering abstention, evidence payloads
server/core/                byte-for-byte copies of the authoritative KERI research modules
prepare_assets.py           vendors research modules and builds assets/ from the repo
assets/ssl_artifact/        deployed NumPy artifact (weights, preprocessor, metadata)
assets/reports/             reliability, integrity and clustering reports from the research run
assets/evidence/            biomarker evidence catalogue
client/                     Vite + React dashboard (built to client/dist)
fixtures/                   safe, identifier and leakage CSV fixtures for QA
tests/                      pytest suite
```

## Limitations

- Cross-sectional training data: no time horizon, no incidence, no future risk.
- Survey weights are not applied, so results describe the analytic sample only.
- No external validation, calibration study or PRoBE-compliant specimen design.
- Clustering is not run on uploaded data in this deployment; the validated
  pipeline's stability and negative-control runs exceed the hosting budget.
- Uploaded datasets are capped at 15 MB, 20,000 rows and 5,000 scored rows.
- The reference score distribution is stored as an 8,001-point quantile grid of
  the full 63,041-row training reference (percentile resolution 0.0125 pp).
  Weights, preprocessor and score definition are unchanged, and probe scores
  match the authoritative artifact to six decimal places.
- Nothing is persisted: no database, no upload history, no server-side results.

See `SECURITY_PRIVACY.md` for the access-control and data-handling details.
