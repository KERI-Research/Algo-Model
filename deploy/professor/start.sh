#!/usr/bin/env bash
# Production start command for the professor deployment.
# Serves the FastAPI API and the built client on port 5000.
set -euo pipefail
cd "$(dirname "$0")"

# A deployment-only .env file may be mounted beside this script when the
# hosting platform cannot inject arbitrary app configuration variables.
# The file is excluded from version control and never served by the client.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec python3 -m uvicorn server.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-5000}" \
  --log-level warning \
  --no-access-log \
  --proxy-headers \
  --forwarded-allow-ips '*'
