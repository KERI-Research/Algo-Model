"""Vercel Python serverless entrypoint.

Vercel maps ``api/index.py`` to a single Python function and serves the ASGI
application exported as ``app``. ``vercel.json`` rewrites ``/api/v1/*`` here, so
the FastAPI routes keep their existing paths and the browser talks to a
same-origin ``/api/v1`` - no proxy sentinel involved.

The repository root is added to ``sys.path`` because the function's working
directory is the deployment root, and ``server/`` plus ``assets/`` are shipped
with the function via ``includeFiles`` in ``vercel.json``.

Runtime profile: this function installs only FastAPI, pandas, NumPy and
python-multipart (see the root ``requirements.txt``). Scoring uses the exported
preprocessor constants through ``server.inference``; scikit-learn, SciPy, joblib
and PyTorch are absent by design and never imported.

Secrets stay environment-only: ``METABOGUARD_ACCESS_KEY_SHA256`` and
``METABOGUARD_SESSION_SECRET`` are read from the Vercel environment at request
time. Nothing is written to disk and no state survives an invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app import app  # noqa: E402

# Vercel's Python runtime looks for `app` (ASGI) or `handler`.
handler = app

__all__ = ["app", "handler"]
