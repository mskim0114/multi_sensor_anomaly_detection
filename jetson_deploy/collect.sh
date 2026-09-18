#!/usr/bin/env bash
# One command per action; resolve paths independently of the current directory.
set -euo pipefail
COLLECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$COLLECT_DIR/run_python.sh" "$COLLECT_DIR/collect.py" "$@"
