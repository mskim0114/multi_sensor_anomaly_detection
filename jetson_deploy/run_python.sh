#!/usr/bin/env bash
# JETSON-RUNTIME python launcher.
#
# Runs a script inside $HOME/venvs/factory_runtime with PYTHONNOUSERSITE=1 so
# that packages in $HOME/.local are never used as runtime dependencies.
# See docs/ENVIRONMENT_POLICY.md.
#
# Usage:
#   ./jetson_deploy/run_python.sh jetson_deploy/scripts/07_read_sps30.py --i2c-port /dev/i2c-1
#   ./jetson_deploy/run_python.sh jetson_deploy/check_environment.py
#   ./jetson_deploy/run_python.sh -c 'import onnxruntime; print(onnxruntime.__version__)'
#
# Override the venv location with FACTORY_JETSON_VENV.

set -euo pipefail

VENV="${FACTORY_JETSON_VENV:-$HOME/venvs/factory_runtime}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: JETSON-RUNTIME venv not found at: $VENV" >&2
    echo "" >&2
    echo "Create it first:" >&2
    echo "    ./jetson_deploy/setup_jetson_env.sh" >&2
    echo "" >&2
    echo "Or point FACTORY_JETSON_VENV at an existing venv." >&2
    exit 1
fi

# The whole point of this launcher. Do not remove.
export PYTHONNOUSERSITE=1

exec "$PY" "$@"
