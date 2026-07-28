"""Score patient-style JSON records with a trained MetaboGuard SSL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from self_supervised import score_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    records = payload if isinstance(payload, list) else [payload]
    result = score_records(pd.DataFrame(records), args.artifact)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
