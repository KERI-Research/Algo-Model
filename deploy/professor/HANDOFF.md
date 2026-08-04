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
| `server/reports.py`                               | Reliability, clustering abstention, evidence payloads, path sanitisation.    |
| `server/core/`                                    | Byte-for-byte copies of `api/{self_supervised,data_integrity,data_reliability,evidence_catalogue}.py`. Never edit here; edit `api/` and re-run `prepare_assets.py`. |
| `client/src/components/`                          | One file per dashboard section, plus `common.jsx` primitives and `Login.jsx`. |
| `client/src/lib/synthetic_patient.js`             | Vendored unchanged from `frontend/src/interface/`. Do not edit.              |
| `client/src/lib/synthetic_prevention.js`          | New extension that fills the remaining prevention-allowlist inputs.         |
| `prepare_assets.py`                               | Regenerates `assets/`, `server/core/` and the vendored client modules from the repo. |
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
3. **Navigation is state-based**, not routed: five sections in a sidebar that
   becomes a two-column button grid below 900px. No URL routing means no
   hash-routing pitfalls inside the preview iframe.
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
8. **Bearer fallback exists only for cookie-blocked hosts** (the thread preview
   iframe). Cookies are the primary transport; remove the fallback if the preview
   path is not needed (see `SECURITY_PRIVACY.md`).

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
- After any frontend change: `cd client && npm test && npm run build`, then
  re-deploy. The FastAPI process serves `client/dist` from disk, so a rebuild is
  visible on reload without restarting the server.
- Never commit or push from this workspace; the repository is left dirty on
  purpose.

## Known constraints

- The agent sandbox cannot write into `/Volumes/Personal-Projects/KERI`, so
  `prepare_assets.py`, `npm ci`, `npm run build` and `pytest` must be run there by
  a human shell. All of them were run successfully in the sandbox copy against
  byte-identical sources.
- Two leftover marker files could not be deleted from the repository by the agent:
  `deploy/professor/.keep`, `deploy/professor/assets/ssl_artifact/.keep` (both now
  carry an explanatory line) and an empty `.pc_write_test.txt` at the repository
  root, which can be deleted safely.
