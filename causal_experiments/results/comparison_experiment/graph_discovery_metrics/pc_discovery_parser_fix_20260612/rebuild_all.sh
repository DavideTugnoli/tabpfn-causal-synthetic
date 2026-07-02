#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"

python3 \
  "$HERE/../scripts/compute_discovery_metrics.py" \
  --repo-root "$REPO" \
  --output-dir "$HERE/outputs"

python3 "$HERE/build_report.py"
