#!/usr/bin/env python3
"""Start, stop, or inspect the real sensor collector without a terminal session."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from sensors.runtime import (AcquisitionLock, atomic_json, owned_pidfd,
                             process_identity, runtime_dir, terminate_owned, utc_now)

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sensor_runner", ROOT / "scripts/11_collect_sensors.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def session_status(session):
    if not session:
        return {"status": "stopped", "alive": False}
    state = read_json(Path(session["run_dir"]) / "control.json")
    fd = owned_pidfd(session["identity"])
    alive = fd is not None
    if fd is not None:
        os.close(fd)
    if state.get("token") != session["token"]:
        state = {"status": "starting"}
    result = {**state, "alive": alive, "run_dir": session["run_dir"],
              "pid": session["identity"]["pid"]}
    if not alive and result["status"] not in ("completed", "stopped", "failed"):
        result.update(status="failed", failure="process exited before finalization")
    return result


def show(state):
    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)


def continuous_args(argv):
    if not any(a == "--duration" or a.startswith("--duration=") for a in argv):
        return ["--duration", "0", *argv]
    return argv


def managed_run(run_dir: Path, token: str, argv) -> int:
    identity = process_identity(os.getpid())

    def update(status, **extra):
        atomic_json(run_dir / "control.json", {
            "status": status, "token": token, "identity": identity,
            "updated_utc": utc_now(), "run_dir": str(run_dir), **extra})

    try:
        return runner.main(argv, run_dir=run_dir, on_state=update)
    except BaseException as exc:
        update("failed", failure=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise


def start(argv) -> int:
    argv = continuous_args(argv)
    args = runner.parse_args(argv)  # Reject mistakes before spawning a process.
    control_root = runtime_dir()
    with AcquisitionLock(control_root / "control.lock"):
        session_path = control_root / "session.json"
        session = read_json(session_path)
        current = session_status(session)
        if current["alive"]:
            print("Collection is already active; existing settings retained.")
            show(current)
            return 0
        # Surface a trial/diagnostic's ownership before allocating output files.
        # The collector independently acquires the same lock before device open.
        with AcquisitionLock():
            pass
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = Path(args.out_dir).expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        token = uuid.uuid4().hex
        with (run_dir / "collector.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "_run",
                 str(run_dir), token, *argv], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                close_fds=True)
        try:
            identity = process_identity(child.pid)
            session = {"identity": identity, "token": token, "run_dir": str(run_dir)}
            atomic_json(session_path, session)
        except BaseException:
            if child.poll() is None:
                terminate_owned(process_identity(child.pid), timeout=30.0)
            raise
    # Do not hold the control lock during startup: stop must remain available.
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            result = session_status(session)
            if result["status"] in ("running", "completed", "stopped", "failed"):
                show(result)
                return 1 if result["status"] == "failed" else 0
            if child.poll() is not None:
                show(session_status(session))
                return 1
            time.sleep(0.1)
        show(session_status(session))
        print("Startup/finalization is still in progress. Use status or stop.")
        return 0
    except BaseException:
        if child.poll() is None:
            terminate_owned(session["identity"], timeout=30.0)
        raise


def stop(argv) -> int:
    parser = argparse.ArgumentParser(prog="collect.sh stop")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="maximum seconds for graceful flush/validation; no forced kill")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    root = runtime_dir()
    with AcquisitionLock(root / "control.lock"):
        session = read_json(root / "session.json")
        state = session_status(session)
        if not state["alive"]:
            show(state)
            return 1 if state["status"] == "failed" else 0
        # The run's first state is published only after its signal handlers
        # are installed. Before that, SIGTERM could bypass graceful cleanup.
        deadline = time.monotonic() + min(5.0, args.timeout)
        while (not read_json(Path(session["run_dir"]) / "control.json")
               and session_status(session)["alive"]):
            if time.monotonic() >= deadline:
                print("Collector has not installed its stop handler yet; retry stop.", file=sys.stderr)
                return 2
            time.sleep(0.05)
        if not terminate_owned(session["identity"], timeout=args.timeout):
            show(session_status(session))
            print("Still stopping: timeout expired; no forced termination was used.", file=sys.stderr)
            return 2
        state = session_status(session)
        show(state)
        return 0 if state["status"] in ("stopped", "completed") else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_run":
        return managed_run(Path(argv[1]), argv[2], argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "run", "dashboard"))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        print("start/run options: --duration SECONDS (default 0), --out-dir PATH,\n"
              "  --save-ct-raw, --thermal-device PATH, --enable-sgp30\n"
              "run stays in the foreground; Ctrl+C flushes and stops.\n"
              "dashboard: local browser display and Start/Stop buttons (port 8765).")
        return 0
    args = parser.parse_args(argv[:1])
    if args.command == "dashboard":
        from dashboard import main as dashboard_main
        return dashboard_main(argv[1:])
    if args.command == "start":
        return start(argv[1:])
    if args.command == "stop":
        return stop(argv[1:])
    if args.command == "run":
        return runner.main(continuous_args(argv[1:]))
    if len(argv) != 1:
        parser.error("status takes no arguments")
    show(session_status(read_json(runtime_dir() / "session.json")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
