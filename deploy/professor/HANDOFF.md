# Project handoff - MetaboGuard professor dashboard

For whoever picks this up next (agent or human). Read `README.md` for install and
run, `SECURITY_PRIVACY.md` for the access-control and data-handling contract.

## Where things live

| Path                                              | Purpose                                                                    |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `/Volumes/Personal-Projects/KERI/deploy/professor` | Authoritative source, inside the KERI repository. Uncommitted.              |
| `/home/user/workspace/metaboguard-professor-dashboard` | Deployable sandbox copy (identical source plus generated assets and build). |
| `server/app.py`                                   | All routes. Public: health, status, session, login, logout. Everything else needs a session. |
| `server/auth.py`                                  | SHA-256 key check, signed `__Host-` cookie, bearer fallback, rate limiter.  |
| `server/dataset.py`                               | CSV intake: identifier screening, denylist, tiers, ephemeral parsing, row rule. |
| `server/model.py`                                 | NumPy inference wrapper, aggregates, results CSV, field meanings, boundaries. |
| `server/future_risk.py`                           | Simulation-only NumPy replay, input allowlist, horizon abstentions and clinical refusal text. |
| `server/future_risk_export.py`                    | Offline export/parity harness; never imported on the request path.            |
| `server/reports.py`                               | Reliability, clustering abstention, evidence payloads, path sanitisation.    |
| `server/core/`                                    | Byte-for-byte copies of `api/{self_supervised,data_integrity,data_reliability,evidence_catalogue}.py`. Never edit here; edit `api/` and re-run `prepare_assets.py`. |
| `client/src/components/`                          | One file per dashboard section, plus `common.jsx` primitives and `Login.jsx`. |
| `client/src/components/HowItWorks.jsx`            | "How the AI works" explainer. Content lives in the exported `PIPELINE_STEPS`, `READING_OUTPUTS`, `COMPARISON_ROWS` and `CAPABILITIES` arrays, so edits are data edits and the tests read the same arrays. |
| `client/src/lib/synthetic_patient.js`             | Vendored unchanged from `frontend/src/interface/` and **tracked**. Do not edit here; edit the source and re-run `prepare_assets.py`. |
| `client/src/lib/synthetic_history.js`             | Vendored deterministic longitudinal generator used by the simulation panel.   |
| `client/src/lib/synthetic_prevention.js`          | New extension that fills the remaining prevention-allowlist inputs.         |
| `prepare_assets.py`                               | Regenerates `assets/`, `server/core/`, the vendored client modules, `preprocessor_params.json` and `research_constants.json`. |
| `api/index.py`, `vercel.json`, `.vercelignore`, `requirements.txt` | Vercel Hobby deployment: static build plus one Python function. |
| `server/inference.py`                             | NumPy-only preprocessor replay and scoring; falls back to the scikit-learn path when the exported constants are absent. |
| `server/research_constants.py`                    | Exported research constants and a verbatim `dataset_capabilities` copy for the trimmed runtime. |
| `scripts/vercel_local_check.py`                   | Serverless-equivalent local check of the Vercel routing and auth paths. |
| `fixtures/`                                       | `safe_deidentified_cohort.csv` (240 rows), `identifier_cohort_REJECT.csv`, `leakage_columns_cohort.csv`. |

## Design and content decisions

1. **Restrained clinical instrument, not a product.** Cool ink neutrals, one deep
   teal accent (`--accent: #0e5c63`), semantic colour reserved for reliability
   tiers and abstention states. Satoshi for text, JetBrains Mono for every number
   (`tabular-nums` everywhere). No stock imagery; the only graphic is the custom
   inline SVG mark in `common.jsx` (a metabolic trace that steps out of range and
   returns), reused as the favicon.
2. **Heading sizes stay small** (`--text-lg` maximum) because this is a data-dense
   dashboard, not a marketing page. Dark mode follows `prefers-color-scheme`.
3. **Navigation is state-based**, not routed: six sections in a sidebar that
   becomes a two-column button grid below 900px. No URL routing means no
   hash-routing pitfalls inside the preview iframe. "How the AI works" is an
   auxiliary section (`AUX_SECTIONS` in `App.jsx`), reached from prominent cards
   on Overview and Evidence & Methods plus a dashed sidebar helper link, so the
  numbered navigation stays at six items. Only one element ever carries
   `aria-current="page"`; the parent section is marked with
   `data-parent-of-active` instead.
4. **Terminology is fixed by the research contract.** "Metabolic deviation score",
   "reference percentile", "feature contributions", "abstained". Never "risk",
   "probability", "prediction", "cluster of cancer".
5. **Every claim surface carries its explanation class** (data observation, model
   association, published evidence) exactly as the API reports it.
6. **Two-step dataset flow.** `POST /dataset/inspect` screens; `POST /dataset/analyse`
   requires `analysis_confirmed=true` and re-sends the file. Nothing is cached
   server side between the steps, which is why the browser re-uploads.
7. **Clustering is deliberately unavailable for uploads.** The report surface shows
   the pipeline's `no_stable_clusters` abstention and the survey-cycle reason; no
   cluster is ever computed for user data.
8. **Two runtime profiles, one behaviour.** The pplx sandbox keeps scikit-learn
   and unpickles the fitted preprocessor; the Vercel function installs only
   FastAPI, pandas, NumPy and python-multipart and replays exported preprocessor
   constants with NumPy. Parity is enforced by tests, and `/api/v1/model`
   reports `preprocessor_path`. Never "optimise" one path without re-running
   `tests/test_inference_parity.py`.
9. **API base is resolved once** in `client/src/lib/api.js`:
   `VITE_DEPLOY_TARGET=vercel` -> same-origin `/api/v1`; `VITE_API_BASE` ->
   explicit override; otherwise the pplx `__PORT_5000__` sentinel. `npm run
   build` is the pplx build, `npm run build:vercel` the Vercel build.
10. **Bearer fallback exists only for cookie-blocked hosts** (the thread preview
   iframe). Cookies are the primary transport; remove the fallback if the preview
   path is not needed (see `SECURITY_PRIVACY.md`).
11. **Future-risk simulation is a separate capability.** It accepts only
  generated multi-visit histories and replays selected models from the
  Synthea-derived simulation artifact. The Vercel request path uses NumPy and
  JSON constants only; the authoritative joblib/PyTorch file is not deployed.
  `/api/v1/future-risk/score` remains a permanent `409` refusal.

## Conventions for incremental edits

- Add a route: define it in `server/app.py` with `_: dict = Depends(auth.require_session)`
  unless it is genuinely public, then add an entry to `PROTECTED_GET_ROUTES` in
  `tests/test_api_protection.py`.
- Add a probe field: extend `PREVENTION_FIELD_SCHEMA` and the relevant
  `FIELD_GROUPS` entry in `PatientProbe.jsx`, and generate it in
  `synthetic_prevention.js`. The API only accepts allowlisted features, so a new
  field must exist in `PREVENTION_FEATURES` upstream first.
- Add an identifier pattern: `DIRECT_IDENTIFIER_PATTERNS` (name-based) or
  `VALUE_IDENTIFIER_PATTERNS` (value-shaped) in `server/dataset.py`, then add a
  case to the parametrised test in `tests/test_dataset.py`.
- Interactive elements carry `data-testid` (`button-*`, `input-*`, `nav-*`,
  `text-*`, `stat-*`, `empty-*`) - Playwright QA depends on them.
- Styling goes through the tokens in `client/src/styles.css`; no inline colours.
- The explainer's flow diagram is pure CSS (`.flow`, `.flow-marker`, connector
  via `::before`) - no imagery. Wide informational tables use `.stack-table`,
  which turns rows into labelled blocks below 760px so a status badge can never
  scroll out of view on a phone.
- Safety wording is test-enforced: `client/src/components/howitworks.test.jsx`
  checks the longitudinal sentence verbatim, that risk-claim phrases appear only
  inside a negation, that every risk-over-time capability is marked
  `Unavailable until longitudinal validation`, and that no scheduling language
  ("coming soon", "planned for") appears anywhere.
- After any frontend change: `cd client && npm test && npm run build`, then
  re-deploy. The FastAPI process serves `client/dist` from disk, so a rebuild is
  visible on reload without restarting the server.
- Never commit or push from this workspace; the repository is left dirty on
  purpose.
- If the artifact or the vendored modules change, re-run `prepare_assets.py` and
  then `pytest -q` - the parity and constants tests are the guard against silent
  drift between the two runtime profiles.
- Secrets are environment-only in both deployments. Nothing secret belongs in
  `vercel.json`, `.env.example`, the client bundle or the repository.

## Vercel notes

- Deploy from the directory holding `vercel.json`. `vercel.json` carries the
  install/build commands, the output directory and the function config, so the
  CLI needs no flags.
- `client/src/lib/` was invisible to Git because the repository root
  `.gitignore` has a broad `lib/` rule (line 594). `deploy/professor/.gitignore`
  now re-includes that directory; keep the negation if you touch that file.
- Generated paths (`assets/`, `server/core/`, and synthetic generators in `client/src/lib/`)
  are required at build/run time and are therefore **tracked**, so Git-triggered
  builds work. Never re-add them to `.gitignore`: doing so reproduces the Vite
  "cannot resolve ../lib/synthetic_patient.js" build failure and leaves the
  function without a model artifact.
- `PatientProbe.jsx` imports the whole synthetic surface from
  `../lib/synthetic_prevention.js`, which re-exports `REQUIRED_PROBE_FIELDS` and
  `SYNTHETIC_FIELD_SCHEMA` from the vendored module. Keep that single import path.
- `scripts/vercel_local_check.py` is the fast regression gate for the Vercel
  path configuration: build with `build:vercel`, then run it.

## Known constraints

- The synthetic simulator is software verification only. Its held-out event
  counts are underpowered, its calibration is in-simulation only and none of its
  estimates may be presented as patient risk.
