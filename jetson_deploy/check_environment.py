#!/usr/bin/env python3
"""JETSON-RUNTIME environment checker. READ-ONLY.

Reports which interpreter is running, where every runtime dependency comes from,
which sensor devices exist, and the git state of the working tree.

This is an *environment* checker, not a sensor test and not an inference test: it
never opens a device, never sends a command to any sensor, and never runs a model.
It installs and modifies nothing.

Consequence to keep in mind: a provider listed as "available" here has NOT been
shown to execute inference. Use jetson_deploy/scripts/03_verify_accuracy.py
--small --provider cuda, or a provider diagnostic, for actual inference
validation. See docs/JETSON_ENVIRONMENT.md section 9.

Run it through the launcher so PYTHONNOUSERSITE=1 is set:

    ./jetson_deploy/run_python.sh jetson_deploy/check_environment.py

Exit codes:
    0  all required checks passed
    1  a required check failed

See docs/ENVIRONMENT_POLICY.md and docs/JETSON_ENVIRONMENT.md.
"""

from __future__ import annotations

import glob
import importlib
import os
from pathlib import Path
import platform
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
USER_SITE_ROOT = os.path.realpath(os.path.join(os.path.expanduser("~"), ".local"))

# (module, required, note)
MODULES = [
    ("numpy",                     True,  "pinned 1.26.4 for the JetPack OpenCV numpy 1.x ABI"),
    ("onnxruntime",               True,  "jetson-ai-lab wheel, not on PyPI for aarch64"),
    ("tensorrt",                  True,  "JetPack/APT only"),
    ("cv2",                       False, "JetPack/APT only"),
    ("Jetson.GPIO",               False, "APT python3-jetson-gpio"),
    ("serial",                    False, "pyserial"),
    ("sensirion_i2c_sps30",       True,  "SPS30 particulate matter"),
    ("sensirion_i2c_scd30",       True,  "SCD30 CO2/temp/humidity"),
    ("sensirion_i2c_driver",      True,  "Sensirion base driver"),
    ("sensirion_driver_adapters", True,  "Sensirion I2C adapter"),
]

# Devices this project's confirmed wiring depends on.
EXPECTED_I2C = {
    "/dev/i2c-7": "MAIN  400 kHz  pin 3/5    ADS1115 0x48, SGP30 0x58",
    "/dev/i2c-1": "SLOW  100 kHz  pin 27/28  SPS30 0x69, SCD30 0x61",
}

failures: list[str] = []
warnings: list[str] = []


def header(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def origin_label(path: str) -> str:
    real = os.path.realpath(path or "")
    if not real:
        return "?"
    if real.startswith(USER_SITE_ROOT):
        return "USER-SITE"
    if real.startswith(os.path.realpath(sys.prefix)) and sys.prefix != sys.base_prefix:
        return "VENV"
    if real.startswith("/usr/lib/python3"):
        return "APT/JetPack"
    if real.startswith("/usr/local/lib/python3"):
        return "SYSTEM-LOCAL"
    return "OTHER"


def check_interpreter() -> None:
    header("1. Interpreter")
    print(f"  executable        : {sys.executable}")
    print(f"  version           : {platform.python_version()}")
    print(f"  prefix            : {sys.prefix}")
    print(f"  base_prefix       : {sys.base_prefix}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"  running in venv   : {in_venv}")
    if not in_venv:
        warnings.append("not running inside a venv (expected $HOME/venvs/factory_runtime)")

    nousersite = os.environ.get("PYTHONNOUSERSITE")
    print(f"  PYTHONNOUSERSITE  : {nousersite if nousersite else '(unset)'}")
    print(f"  ENABLE_USER_SITE  : {site.ENABLE_USER_SITE}")
    if site.ENABLE_USER_SITE:
        failures.append(
            "user-site is ENABLED. Run through ./jetson_deploy/run_python.sh, "
            "or set PYTHONNOUSERSITE=1."
        )
        print("  ^^ FAIL: $HOME/.local packages can shadow the venv and JetPack.")
    else:
        print("  ^^ OK: $HOME/.local is excluded from sys.path")


def check_platform() -> None:
    header("2. Platform")
    print(f"  machine           : {platform.machine()}")
    print(f"  kernel            : {platform.release()}")
    l4t = Path("/etc/nv_tegra_release")
    if l4t.is_file():
        print(f"  L4T               : {l4t.read_text().splitlines()[0][:64]}")
    else:
        print("  L4T               : (not found - not a Jetson?)")
        warnings.append("/etc/nv_tegra_release not found")
    boot = Path("/etc/nv_boot_control.conf")
    if boot.is_file():
        for line in boot.read_text().splitlines():
            if line.startswith("TNSPEC"):
                print(f"  board             : {line.split(None, 1)[-1]}")
                break
    osr = Path("/etc/os-release")
    if osr.is_file():
        for line in osr.read_text().splitlines():
            if line.startswith("PRETTY_NAME"):
                print(f"  os                : {line.split('=', 1)[1].strip().strip(chr(34))}")
                break


def check_modules() -> None:
    header("3. Runtime dependencies (origin must never be USER-SITE)")
    print(f"  {'module':28s} {'origin':13s} {'version':12s} note")
    print("  " + "-" * 70)
    for name, required, note in MODULES:
        try:
            mod = importlib.import_module(name)
        except BaseException as exc:
            tag = "MISSING" if required else "absent "
            print(f"  {name:28s} {tag:13s} {'-':12s} {type(exc).__name__}")
            if required:
                failures.append(f"required module not importable: {name} ({type(exc).__name__})")
            else:
                warnings.append(f"optional module not importable: {name}")
            continue

        path = getattr(mod, "__file__", "") or ""
        label = origin_label(path)
        version = str(getattr(mod, "__version__", "unknown"))
        print(f"  {name:28s} {label:13s} {version:12s} {note}")
        if label == "USER-SITE":
            failures.append(f"{name} is loaded from $HOME/.local: {path}")

    # numpy major version gates the JetPack OpenCV ABI.
    try:
        import numpy
        if not numpy.__version__.startswith("1."):
            failures.append(
                f"numpy {numpy.__version__} is 2.x - JetPack OpenCV / APT pandas / APT PIL "
                "are built against the numpy 1.x ABI and will fail to import"
            )
    except BaseException:
        pass


def check_providers() -> None:
    header("4. ONNX Runtime execution providers (AVAILABILITY ONLY)")
    print("  This checker reports which providers the onnxruntime build EXPOSES.")
    print("  It does NOT run a model, so it cannot tell you whether a provider")
    print("  actually executes inference. Those are different things:")
    print("    available          - present in the onnxruntime build")
    print("    configured         - accepted when creating an InferenceSession")
    print("    actually working   - sess.run() returns correct output")
    print("  For real inference validation run:")
    print("    jetson_deploy/scripts/03_verify_accuracy.py --small --provider cuda")
    print("")
    try:
        import onnxruntime as ort
    except BaseException as exc:
        print(f"  onnxruntime import            FAIL  {type(exc).__name__}: {exc}")
        failures.append(f"onnxruntime not importable: {type(exc).__name__}")
        return
    print(f"  onnxruntime import            PASS  ({ort.__version__})")
    providers = ort.get_available_providers()
    print(f"  raw get_available_providers() {providers}")
    print("")
    for want, label in (
        ("CPUExecutionProvider", "CPU provider available     "),
        ("CUDAExecutionProvider", "CUDA provider available    "),
        ("TensorrtExecutionProvider", "TensorRT provider available"),
    ):
        if want in providers:
            print(f"  {label}   PASS")
        else:
            print(f"  {label}   WARN (not in build)")
            warnings.append(f"{want} not available")
    print("")
    print("  CPU actual inference          NOT TESTED BY THIS CHECKER")
    print("  CUDA actual inference         NOT TESTED BY THIS CHECKER  (verified 2026-08-31)")
    print("  TensorRT actual inference     NOT TESTED BY THIS CHECKER")
    if "TensorrtExecutionProvider" in providers:
        print("")
        print("  KNOWN ISSUE 2026-08-31: TensorrtExecutionProvider is available and a")
        print("  session can be created, but the first sess.run() segfaults (SIGSEGV,")
        print("  exit 139) inside libnvinfer.so.10 on this board. trtexec runs the same")
        print("  ONNX end to end, so the fault is in the onnxruntime TensorRT EP")
        print("  integration, not in TensorRT or the model. Use --provider cuda.")
        print("  Details: docs/JETSON_ENVIRONMENT.md section 9.")


def check_devices() -> None:
    header("5. Devices (existence and permissions only - nothing is opened)")

    print("  I2C:")
    present = sorted(glob.glob("/dev/i2c-*"))
    for path, note in EXPECTED_I2C.items():
        if path in present:
            print(f"    OK    {path:14s} {note}")
            if not os.access(path, os.R_OK | os.W_OK):
                warnings.append(f"{path} exists but is not read/write for this user")
                print(f"          WARN no rw access - is the user in the 'i2c' group?")
        else:
            print(f"    FAIL  {path:14s} missing  ({note})")
            failures.append(f"expected I2C device missing: {path}")
    others = [p for p in present if p not in EXPECTED_I2C]
    if others:
        print(f"    other adapters present: {' '.join(others)}")

    print("  SPI:")
    spidevs = sorted(glob.glob("/dev/spidev*"))
    if spidevs:
        for path in spidevs:
            print(f"    OK    {path}")
        print("    note: node existence does NOT prove routing to header pins 19/21/23/24.")
        print("          BME680 stays PENDING until that mapping is verified.")
    else:
        print("    none - SPI pinmux not enabled (BME680 unavailable)")
        warnings.append("no /dev/spidev* - SPI not enabled")

    print("  USB thermal camera:")
    videos = sorted(glob.glob("/dev/video*"))
    byid = sorted(glob.glob("/dev/v4l/by-id/*PureThermal*"))
    if byid:
        for path in byid:
            print(f"    OK    {os.path.basename(path)}")
    elif videos:
        print(f"    WARN  video devices present but no PureThermal by-id link: {' '.join(videos)}")
        warnings.append("no PureThermal by-id link")
    else:
        print("    none")
        warnings.append("no /dev/video* - thermal camera not connected")


def check_git() -> None:
    header("6. Git state (for experiment reproducibility metadata)")

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  git unavailable: {type(exc).__name__}: {exc}")
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    sha = git("rev-parse", "HEAD")
    if sha is None:
        print("  not a git repository, or git failed")
        warnings.append("git state unavailable")
        return

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain")
    dirty = bool(status)

    print(f"  repository        : {ROOT}")
    print(f"  git_commit_sha    : {sha}")
    print(f"  branch            : {branch}")
    print(f"  git_dirty         : {dirty}")
    if dirty:
        lines = status.splitlines() if status else []
        for line in lines[:10]:
            print(f"                      {line}")
        if len(lines) > 10:
            print(f"                      ... and {len(lines) - 10} more")
        print("  note: results produced from a dirty tree must not be cited as final numbers.")


def main() -> int:
    print("JETSON-RUNTIME environment check (read-only)")
    check_interpreter()
    check_platform()
    check_modules()
    check_providers()
    check_devices()
    check_git()

    header("Summary")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for item in failures:
            print(f"    - {item}")
    else:
        print("  FAILURES: none")
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for item in warnings:
            print(f"    - {item}")
    else:
        print("  WARNINGS: none")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
