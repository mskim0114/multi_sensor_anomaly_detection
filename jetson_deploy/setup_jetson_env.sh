#!/usr/bin/env bash
# Create and verify the JETSON-RUNTIME Python environment.
#
#   ./jetson_deploy/setup_jetson_env.sh
#
# What it does NOT do, by policy (docs/ENVIRONMENT_POLICY.md):
#   no sudo pip, no apt install, no JetPack/CUDA/TensorRT changes,
#   no device-tree or pinmux changes, no reboot.
# Missing system packages are reported with the command to run, then it stops.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FACTORY_JETSON_VENV:-$HOME/venvs/factory_runtime}"
REQ="$ROOT/jetson_deploy/requirements-jetson.txt"
CON="$ROOT/jetson_deploy/constraints-jetson.txt"

step()  { printf '\n=== %s\n' "$1"; }
ok()    { printf '  OK    %s\n' "$1"; }
warn()  { printf '  WARN  %s\n' "$1"; }
fail()  { printf '  FAIL  %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- 0. preflight
step "0. Preflight"

[ "$(uname -s)" = "Linux" ] || fail "Linux only (found $(uname -s))"
ok "kernel: $(uname -sr) $(uname -m)"

if [ -r /etc/nv_tegra_release ]; then
    ok "L4T: $(head -1 /etc/nv_tegra_release | cut -c1-60)"
else
    warn "/etc/nv_tegra_release not found - this may not be a Jetson."
    warn "JetPack packages (tensorrt, cv2, Jetson.GPIO) will be missing."
fi

PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PYVER" = "3.10" ] || fail "python3 must be 3.10.x (found $PYVER). Do not change the system python."
ok "python3: $(python3 --version 2>&1)"

python3 -c 'import venv' 2>/dev/null \
    || fail "python3 venv module missing. Install it yourself, then re-run:
            sudo apt install python3-venv"
ok "venv module available"

command -v i2cdetect >/dev/null 2>&1 \
    || warn "i2cdetect not found. For I2C diagnostics: sudo apt install i2c-tools"

for grp in i2c gpio video; do
    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx "$grp"; then
        ok "group: $grp"
    else
        warn "user is not in group '$grp'. Device access will need sudo."
        warn "  fix: sudo usermod -aG i2c,gpio,video \"\$USER\"   (re-login required)"
    fi
done

[ -r "$REQ" ] || fail "missing $REQ"
[ -r "$CON" ] || fail "missing $CON"
ok "requirements + constraints found"

# ------------------------------------------------------- 1. JetPack visibility
step "1. JetPack / APT python packages (must be visible with PYTHONNOUSERSITE=1)"
env PYTHONNOUSERSITE=1 python3 - <<'PY'
import importlib, sys
required = ["tensorrt", "cv2"]
optional = ["Jetson.GPIO", "yaml", "PIL"]
missing = []
for name in required + optional:
    try:
        m = importlib.import_module(name)
        print(f"  OK    {name:14s} {str(getattr(m, '__version__', 'unknown')):10s} {getattr(m, '__file__', '')}")
    except BaseException as exc:
        tag = "FAIL " if name in required else "WARN "
        print(f"  {tag} {name:14s} {type(exc).__name__}: {str(exc)[:60]}")
        if name in required:
            missing.append(name)
if missing:
    print("", file=sys.stderr)
    print(f"Required JetPack packages missing: {', '.join(missing)}", file=sys.stderr)
    print("These come from JetPack/APT and must NOT be installed from PyPI:", file=sys.stderr)
    print("    sudo apt install tensorrt python3-opencv python3-jetson-gpio", file=sys.stderr)
    raise SystemExit(1)
PY

# ------------------------------------------------------------------ 2. the venv
step "2. venv at $VENV"
if [ -d "$VENV" ]; then
    ok "already exists - reusing (not deleted, not recreated)"
else
    python3 -m venv --system-site-packages "$VENV"
    ok "created with --system-site-packages"
fi
[ -x "$VENV/bin/python" ] || fail "$VENV/bin/python is not executable"

if ! grep -q 'include-system-site-packages *= *true' "$VENV/pyvenv.cfg" 2>/dev/null; then
    fail "$VENV was NOT created with --system-site-packages.
            JetPack tensorrt/cv2 would be invisible. Remove it and re-run."
fi
ok "include-system-site-packages = true"

# Make PYTHONNOUSERSITE survive a plain `source .../activate` too.
if ! grep -q 'PYTHONNOUSERSITE' "$VENV/bin/activate"; then
    printf '\n# Added by setup_jetson_env.sh - see docs/ENVIRONMENT_POLICY.md\nexport PYTHONNOUSERSITE=1\n' \
        >> "$VENV/bin/activate"
    ok "appended 'export PYTHONNOUSERSITE=1' to bin/activate"
else
    ok "bin/activate already exports PYTHONNOUSERSITE"
fi

# ------------------------------------------- 2.5 ORT supply-chain gate
step "2.5 ONNX Runtime supply-chain gate"
# onnxruntime-gpu has no aarch64 distribution on PyPI, so it comes from the
# jetson-ai-lab index. It is pinned as a hashed direct URL rather than via a
# global --extra-index-url, which would make that third-party index a candidate
# for every other package too. pip verifies the digest before installing; this
# gate makes the expected value auditable and fails loudly if it ever drifts.
EXPECTED_ORT_SHA256="d980b934b9a29c1a9d6f39751edd7662b69fadd75556a10ff363773a58ce0950"

if grep -qE '^[[:space:]]*--(extra-)?index-url' "$REQ"; then
    fail "$REQ contains an active --index-url/--extra-index-url directive.
            A global index directive is not allowed - onnxruntime-gpu must stay
            scoped to a single hashed direct URL."
fi
ok "no global index directive in requirements"

ORT_LINE="$(grep -E '^onnxruntime-gpu @ https://.*#sha256=' "$REQ" || true)"
[ -n "$ORT_LINE" ] || fail "$REQ has no hashed direct-URL pin for onnxruntime-gpu."
FOUND_ORT_SHA256="${ORT_LINE##*#sha256=}"
if [ "$FOUND_ORT_SHA256" != "$EXPECTED_ORT_SHA256" ]; then
    fail "onnxruntime-gpu digest mismatch.
            requirements : $FOUND_ORT_SHA256
            expected     : $EXPECTED_ORT_SHA256
            Re-verify the artifact before changing either value."
fi
ok "onnxruntime-gpu pinned to sha256 ${EXPECTED_ORT_SHA256:0:16}... (pip enforces it)"

# --------------------------------------------------------------- 3. pip install
step "3. Install project packages (PYTHONNOUSERSITE=1)"
echo "  Without PYTHONNOUSERSITE=1 pip would treat \$HOME/.local packages as"
echo "  'Requirement already satisfied' and leave the venv empty."
env PYTHONNOUSERSITE=1 "$VENV/bin/python" -m pip install -r "$REQ" -c "$CON"
ok "pip install completed"

# ------------------------------------------------------- 4. provenance gate
step "4. Provenance gate (no runtime dependency may come from \$HOME/.local)"
env PYTHONNOUSERSITE=1 "$VENV/bin/python" - <<'PY'
import importlib, os, sys

modules = ["numpy", "onnxruntime", "cv2", "tensorrt", "serial",
           "sensirion_i2c_sps30", "sensirion_i2c_scd30",
           "sensirion_i2c_driver", "sensirion_driver_adapters"]
user_site = os.path.join(os.path.expanduser("~"), ".local")
offenders, missing = [], []

for name in modules:
    try:
        mod = importlib.import_module(name)
    except BaseException as exc:
        missing.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name:28s} {type(exc).__name__}")
        continue
    path = os.path.realpath(getattr(mod, "__file__", "") or "")
    ver = str(getattr(mod, "__version__", "unknown"))
    if path.startswith(os.path.realpath(user_site)):
        offenders.append((name, path))
        print(f"  FAIL  {name:28s} {ver:12s} USER-SITE {path}")
    else:
        print(f"  OK    {name:28s} {ver:12s} {path}")

if offenders or missing:
    print("", file=sys.stderr)
    for name, path in offenders:
        print(f"user-site dependency: {name} -> {path}", file=sys.stderr)
    for name, err in missing:
        print(f"not importable: {name} ({err})", file=sys.stderr)
    raise SystemExit(1)
PY
ok "no user-site dependencies"

# ----------------------------------------------------------- 5. smoke test
step "5. Smoke test"
env PYTHONNOUSERSITE=1 "$VENV/bin/python" - <<'PY'
import sys
import numpy, cv2, onnxruntime as ort

print(f"  numpy       {numpy.__version__}")
print(f"  cv2         {cv2.__version__}")
print(f"  onnxruntime {ort.__version__}")

if not numpy.__version__.startswith("1."):
    print("", file=sys.stderr)
    print(f"numpy {numpy.__version__} is 2.x. JetPack OpenCV / APT pandas / APT PIL are",
          file=sys.stderr)
    print("compiled against the numpy 1.x ABI and will fail to import.", file=sys.stderr)
    raise SystemExit(1)

# numpy <-> cv2 interop actually exercises the C ABI, unlike a bare import.
cv2.resize(numpy.zeros((4, 4), dtype=numpy.float32), (2, 2))
print("  numpy <-> cv2 interop OK (no ABI conflict)")

providers = ort.get_available_providers()
print(f"  providers   {providers}")
for want in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
    if want not in providers:
        print(f"  WARN  {want} not available - GPU inference will fall back to CPU")
PY
ok "smoke test passed"

# -------------------------------------------------------- 6. next steps
step "6. Done"
cat <<NEXT

  Environment ready: $VENV

  Run python through the launcher (it sets PYTHONNOUSERSITE=1 for you):

      ./jetson_deploy/run_python.sh jetson_deploy/check_environment.py

  Recommended verification, in order:

      ./jetson_deploy/run_python.sh jetson_deploy/scripts/01_check_environment.py
      ./jetson_deploy/run_python.sh jetson_deploy/scripts/03_verify_accuracy.py --small

  The second one reproduces a stored metric. Compare its macro-F1 against
  jetson_deploy/results/jetson_accuracy_small.json before trusting the build.

NEXT
