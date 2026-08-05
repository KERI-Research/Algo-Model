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
| **Patient Probe**          | Explicit single-record scoring with a plain-language deviation band, NHANES reference percentile, submitted diabetes context, human-readable standout measurements, four cancer/diabetes research-pathway capability cards, data readiness and evidence boundaries. |
| **Dataset Analysis**       | CSV-only drag/drop or picker, de-identification checkbox, identifier and leakage screening, schema mapping, missingness, range violations, feature tiers, rows accepted/rejected, then in-memory scoring and a downloadable results CSV. |
| **Reliability & Clusters** | The pipeline's fail-closed reliability report, feature tiers, and the `no_stable_clusters` abstention with the survey-cycle explanation. |
| **Evidence & Methods**     | Source-linked biomarker catalogue with evidence grades, multi-marker rationale, PRoBE and TRIPOD+AI references, supported vs prohibited claims. |
| **How the AI works**       | Explainer reached from Overview and Evidence & Methods (and a quiet sidebar link, not a numbered section): eight-step CSS flow diagram, plain-language reading of the outputs, current vs future longitudinal model comparison, the role of TCGA, and a three-state capability table. |

The **How the AI works** page states the governing limitation verbatim: "The
current model is not trained on longitudinal data. It cannot estimate a
patient's probability of developing cancer over time, predict which cancer they
will develop, or provide a diagnosis." Capability states are `Available now`,
`Research only` and `Unavailable until longitudinal validation`; the third is a
statement about missing data, never a release schedule.

No output is a diagnosis, a disease probability, a future-risk horizon or a
cancer-type claim. The future-risk head stays fail-closed and clustering
abstains, exactly as in the research pipeline.

The Patient Probe gives pancreatic cancer and general cancers equal research
emphasis. Its four pathway cards cover diabetes-related factors and cancer,
anthropometry, weight-change and reported exposure measurements in cancer
research, temporal direction between cancer and diabetes-related changes, and
the same non-glycaemic measurements in diabetes research. Every card reports
`Probability unavailable` because the current NHANES data are cross-sectional.
Measurements listed beneath a card are supplied values with existing
reconstruction diagnostics, grouped by relevance to that research question;
they are not causal attribution, disease classification or evidence that one
condition will develop from another. Reported diabetes and onset age remain
display context and are not sent to the deployed model.

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

## Deployment A: Perplexity hosting

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

## Deployment B: Vercel (Hobby / free, production)

Vercel serves the Vite build as static files and runs FastAPI as one Python
serverless function. Nothing else changes: same routes, same auth, same
in-memory scoring, no persistence.

**Project settings (Vercel CLI or dashboard)**

| Setting                | Value                                              |
| ---------------------- | -------------------------------------------------- |
| CLI project root       | the directory containing `vercel.json` (`deploy/professor` in the KERI repo, or the sandbox project root) |
| Framework preset       | Other (`"framework": null`)                        |
| Install command        | `npm --prefix client ci`                           |
| Build command          | `npm --prefix client ci && npm --prefix client run build:vercel` |
| Output directory       | `client/dist`                                      |
| Serverless entrypoint  | `api/index.py` (exports the FastAPI `app`)         |
| Python requirements    | root `requirements.txt` (trimmed: FastAPI, pandas, NumPy, python-multipart) |
| Node / Python versions | Node 20+, Python 3.12 (Vercel default)             |

`vercel.json` supplies all of the above, so `vercel` / `vercel --prod` needs no
extra flags. Routing:

* `/api/v1/:path*` -> `/api/index` (the path is preserved, so FastAPI's
  `/api/v1/...` routes match unchanged);
* every other path -> the static build, with `index.html` as the SPA fallback
  (the filesystem is checked first, so hashed assets are served directly).

**Environment variables** - set both as Vercel *environment variables* for
Production (and Preview if you use it). Never commit them, never put them in
`vercel.json`:

```
METABOGUARD_ACCESS_KEY_SHA256   # sha256 hex digest of the access key
METABOGUARD_SESSION_SECRET      # long random secret
```

`vercel env add METABOGUARD_ACCESS_KEY_SHA256 production` (paste the digest) is
the CLI route. No frontend variable is needed: `build:vercel` sets
`VITE_DEPLOY_TARGET=vercel` itself, which makes the client call same-origin
`/api/v1` and drops the pplx proxy sentinel from the bundle entirely.

**Generated but tracked files.** `assets/`, `server/core/` and
`client/src/lib/synthetic_patient*.js` are produced by `prepare_assets.py` from
the authoritative repository, and they are **committed**. A Git-triggered Vercel
build only sees committed files: without them Vite cannot resolve
`../lib/synthetic_patient.js` and the Python function would start with no model
artifact. Both deploy modes therefore work:

* **Git integration** - push and Vercel builds from the commit. Nothing extra to do.
* **CLI** - `vercel --prod` from this directory; `.vercelignore` is used instead
  of `.gitignore` and excludes none of those paths.

Re-run `prepare_assets.py` whenever the model artifact or the vendored modules
change; any drift from the authoritative sources then appears as a normal diff
(about 700 KB of tracked generated content in total).

**Local serverless-equivalent check** (blocks scikit-learn/SciPy/joblib to
emulate the trimmed function runtime, then exercises routing, login, probe,
upload, export and rate limiting):

```bash
npm --prefix client run build:vercel
METABOGUARD_ACCESS_KEY_SHA256=$(printf '%s' 'your-key' | sha256sum | cut -d' ' -f1) \
METABOGUARD_SESSION_SECRET=local-check-secret \
python3 scripts/vercel_local_check.py your-key
```

**Function footprint**: ~155 MB unzipped (pandas 74 MB, NumPy 70 MB, FastAPI
stack ~12 MB, app code and assets under 1 MB), inside the 250 MB Hobby limit.
scikit-learn and SciPy are deliberately absent; see "Runtime profiles" below.

### Runtime profiles

| Profile  | Where                          | Installed                                        | Preprocessor                                     |
| -------- | ------------------------------ | ------------------------------------------------ | ------------------------------------------------ |
| Full     | development, CI, pplx sandbox  | `requirements-deploy.txt` (adds scikit-learn, joblib, uvicorn) | fitted `preprocessor.joblib` via the vendored module |
| Trimmed  | Vercel function                | `requirements.txt`                               | `preprocessor_params.json` replayed with NumPy    |

Both profiles produce byte-identical scores. `tests/test_inference_parity.py`
compares them row by row (890 rows including all-missing, unseen categorical
levels and string-typed values), and `tests/test_research_constants.py` compares
the exported constants and `dataset_capabilities` against the authoritative
vendored modules. `/api/v1/model` reports which path is live via
`preprocessor_path`.

### Vercel limitations to expect

* **Cold starts.** Importing pandas/NumPy and reading the artifact takes roughly
  1-3 s on a cold invocation. Dataset analysis of a few thousand rows then runs
  in well under a second.
* **60 s function ceiling** on Hobby. The 5,000-scored-row cap and the internal
  compute budget keep requests far below it, but a 15 MB / 20,000-row upload is
  parsed twice (screen, then analyse) because nothing is persisted.
* **4.5 MB request-body limit** on Vercel functions. This is stricter than the
  app's own 15 MB rule, so in practice CSV uploads over ~4.5 MB will be rejected
  by the platform before FastAPI sees them.
* **Rate limiting is per-instance.** Each serverless instance keeps its own
  in-memory counter, so a distributed attacker gets more attempts than the
  5-per-5-minutes rule suggests. Sessions are unaffected (signed cookies, no
  server state).
* **No writable persistence** (by design). Uploads are memory-only and no
  results are stored.
* Hobby projects are for non-commercial use, and the deployment is still
  prototype hosting: no identifiable or clinical patient data.

## Tests

```bash
pip install -r requirements-test.txt && python3 -m pytest -q      # 98 API/unit tests
cd client && npm test                                              # 46 frontend tests
```

## Repository layout

```
main.py                     uvicorn entrypoint for pplx hosting (server.app:app)
api/index.py                Vercel serverless entrypoint (exports the FastAPI app)
vercel.json                 Vercel build, routing and function configuration
.vercelignore               keeps tests, fixtures and pplx-only files out of Vercel
requirements.txt            trimmed Vercel function requirements (no scikit-learn)
scripts/vercel_local_check.py  serverless-equivalent local verification
start.sh                    production start command, port 5000
requirements-deploy.txt     runtime deps (no PyTorch)
server/config.py            paths, limits, env variable names
server/auth.py              SHA-256 key check, signed __Host- session cookie, rate limiting
server/dataset.py           CSV intake: identifier/leakage screening, tiers, ephemeral parsing
server/model.py             NumPy inference wrapper, aggregates, in-memory results CSV
server/inference.py         NumPy-only preprocessor + scoring (no scikit-learn needed)
server/research_constants.py  exported constants + dataset_capabilities fallback
server/reports.py           reliability, clustering abstention, evidence payloads
server/core/                byte-for-byte copies of the authoritative KERI research modules
prepare_assets.py           refreshes the tracked vendored modules and assets/ from the repo
assets/ssl_artifact/        deployed NumPy artifact (weights, preprocessor, metadata)
assets/reports/             reliability, integrity and clustering reports from the research run
assets/evidence/            biomarker evidence catalogue
client/                     Vite + React dashboard (built to client/dist)
client/src/components/HowItWorks.jsx  the "How the AI works" explainer page
fixtures/                   safe, identifier and leakage CSV fixtures for QA
tests/                      pytest suite
```

## Limitations

* Cross-sectional training data: no time horizon, no incidence, no future risk.
* Survey weights are not applied, so results describe the analytic sample only.
* No external validation, calibration study or PRoBE-compliant specimen design.
* Clustering is not run on uploaded data in this deployment; the validated
  pipeline's stability and negative-control runs exceed the hosting budget.
* Uploaded datasets are capped at 15 MB, 20,000 rows and 5,000 scored rows.
* The reference score distribution is stored as an 8,001-point quantile grid of
  the full 63,041-row training reference (percentile resolution 0.0125 pp).
  Weights, preprocessor and score definition are unchanged, and probe scores
  match the authoritative artifact to six decimal places.
* Nothing is persisted: no database, no upload history, no server-side results.

See `SECURITY_PRIVACY.md` for the access-control and data-handling details.
