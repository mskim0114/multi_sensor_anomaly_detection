#!/usr/bin/env python3
"""Local sensor dashboard. The HTTP process never opens sensor devices."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import collect
from sensors.live import (MAX_PREVIEW_BYTES, clean_json, preview_dir, preview_path,
                          thermal_payload, volatile_json)
from sensors.runtime import owned_pidfd, process_identity, runtime_dir

ROOT = Path(__file__).resolve().parent


def read_object(path):
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_PREVIEW_BYTES + 1)
        if len(data) > MAX_PREVIEW_BYTES:
            raise ValueError("Preview or state file is too large")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except FileNotFoundError:
        return {}


def alive(identity):
    fd = owned_pidfd(identity or {})
    if fd is None:
        return False
    os.close(fd)
    return True


class Replay:
    """Read at most 120 recorded snapshots; never operate the collector."""
    def __init__(self, run_dir):
        self.root = Path(run_dir).expanduser().resolve()
        self.snapshots = []
        with (self.root / "snapshots.jsonl").open("rb") as stream:
            for _ in range(120):
                line = stream.readline(MAX_PREVIEW_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_PREVIEW_BYTES:
                    raise ValueError("Recorded snapshot exceeds size limit")
                self.snapshots.append(json.loads(line))
        if not self.snapshots:
            raise ValueError("No recorded snapshots to replay")
        self.start = time.monotonic()

    def preview(self):
        import numpy as np
        snapshot = self.snapshots[int(time.monotonic() - self.start) % len(self.snapshots)]
        flir = snapshot["sensors"].get("flir", {})
        frame = None
        error = None
        try:
            chunk = flir.get("thermal_chunk")
            if chunk:
                path = (self.root / chunk).resolve()
                if path.parent != self.root or not path.name.startswith("thermal_"):
                    raise ValueError("Invalid thermal chunk path")
                with np.load(path, allow_pickle=False) as arrays:
                    index = flir["thermal_index"]
                    if not isinstance(index, int) or index < 0:
                        raise ValueError("Invalid thermal index")
                    if int(arrays["sequences"][index]) != snapshot["sequence"]:
                        raise ValueError("Recorded thermal sequence mismatch")
                    frame = arrays["frames"][index]
            thermal = thermal_payload(snapshot, (snapshot["sequence"], frame))
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            thermal = None
            error = f"Recorded thermal unavailable: {exc}"
        return {"run_id": self.root.name, "run_dir": str(self.root),
                "snapshot": snapshot, "thermal": thermal, "publisher": {}}, error


class Dashboard:
    def __init__(self, *, out_dir, duration=0, save_ct_raw=False, replay=None, cache_root=None):
        self.settings = {"out_dir": str(Path(out_dir).expanduser().resolve()),
                         "duration_s": duration, "save_ct_raw": save_ct_raw, "thermal_hz": 1}
        self.cache_root = cache_root if cache_root is not None else preview_dir()
        self.replay = Replay(replay) if replay else None
        self.lock = threading.Lock()
        self.pending = {"start": False, "stop": False}
        self.last_result = None
        self.missing_run = None
        self.missing_since = None
        self.token = secrets.token_urlsafe(32)

    def controls(self):
        with self.lock:
            return {"enabled": self.replay is None,
                    "start_pending": self.pending["start"], "stop_pending": self.pending["stop"],
                    "last_result": self.last_result}

    def state(self):
        result = {"mode": "replay" if self.replay else "live", "settings": self.settings,
                  "controls": self.controls(), "preview": None, "preview_age_ms": None,
                  "preview_current": False, "preview_error": None,
                  "session": {"status": "stopped", "alive": False, "managed": False,
                              "run_dir": None, "failure": None, "snapshot_count": None}}
        if self.replay:
            result["preview"], result["preview_error"] = self.replay.preview()
            result["session"]["run_dir"] = str(self.replay.root)
            return result
        try:
            session = read_object(runtime_dir() / "session.json")
            state = collect.session_status(session)
            state["managed"] = bool(session)
            identity = session.get("identity")
            active = read_object(self.cache_root / "active.json")
            if active and (not session or (not state["alive"] and alive(active.get("identity")))):
                identity = active.get("identity")
                is_alive = alive(identity)
                state = {"status": "running" if is_alive else "stopped", "alive": is_alive,
                         "managed": False, "run_dir": active["run_dir"]}
            result["session"].update({key: value for key, value in state.items() if key != "token"})
            if state.get("run_dir"):
                preview = read_object(preview_path(Path(state["run_dir"]), self.cache_root))
                if preview:
                    if (preview.get("run_dir") != state["run_dir"] or
                            preview.get("identity") != identity):
                        raise ValueError("Preview does not match the collection session")
                    result["preview"] = preview
                    stamp = preview["snapshot"].get("snapshot_created_monotonic_ns")
                    same_boot = (identity or {}).get("boot_id") == process_identity(os.getpid())["boot_id"]
                    if isinstance(stamp, int) and same_boot:
                        age = (time.monotonic_ns() - stamp) / 1e6
                        result["preview_age_ms"] = max(0, age)
                        result["preview_current"] = bool(state["alive"] and 0 <= age <= 3000)
                    result["session"]["snapshot_count"] = preview["snapshot"]["sequence"] + 1
                    result["preview_error"] = preview.get("publisher", {}).get("last_error")
            with self.lock:
                missing = state["alive"] and state.get("status") == "running" and not result["preview"]
                if missing:
                    if self.missing_run != state.get("run_dir"):
                        self.missing_run, self.missing_since = state.get("run_dir"), time.monotonic()
                    if time.monotonic() - self.missing_since > 3:
                        result["preview_error"] = "수집 중이지만 실시간 미리보기가 도착하지 않습니다. collector.log를 확인하세요."
                else:
                    self.missing_run = self.missing_since = None
            if state["alive"] and result["preview"] and not result["preview_current"]:
                result["preview_error"] = result["preview_error"] or "실시간 미리보기 갱신이 3초 이상 지연되었습니다."
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["preview_error"] = f"{type(exc).__name__}: {exc}"
            result["controls"]["enabled"] = False
        return result

    def command(self, action):
        args = [sys.executable, "-u", str(ROOT / "collect.py"), action]
        if action == "start":
            args += ["--out-dir", self.settings["out_dir"], "--duration", str(self.settings["duration_s"])]
            if self.settings["save_ct_raw"]:
                args += ["--save-ct-raw"]
        return args

    def submit(self, action):
        if action not in ("start", "stop"):
            raise ValueError("Unknown action")
        if self.replay:
            raise RuntimeError("Controls are disabled in replay mode")
        with self.lock:
            if self.pending[action] or (action == "start" and self.pending["stop"]):
                raise RuntimeError("Collection control is already in progress")
            self.pending[action] = True
            self.last_result = None
            threading.Thread(target=self._control, args=(action,), daemon=True,
                             name=f"dashboard-{action}").start()

    def _control(self, action):
        try:
            # An immediate Stop must also cover a Start which has not yet
            # published its managed session. This waits for lifecycle state,
            # never retries a sensor operation.
            if action == "stop":
                deadline = time.monotonic() + 8
                while self.controls()["start_pending"]:
                    current = self.state()["session"]
                    if current["alive"] and current["managed"]:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Start is still preparing; use Stop again when the session appears")
                    time.sleep(0.05)
            # No shell and no browser-supplied paths/options. Existing lifecycle
            # code owns hardware locking, identity checks and graceful flush.
            # An unlinked regular log FD survives this dashboard's exit.
            # A pipe would give the helper BrokenPipeError and could trigger
            # its startup cleanup, terminating the otherwise independent run.
            with tempfile.TemporaryFile(dir=self.cache_root) as log:
                completed = subprocess.run(self.command(action), stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True)
                log.seek(max(0, log.tell() - 12000))
                message = log.read().decode("utf-8", errors="replace").strip()
            outcome = {"action": action, "message": message,
                       "error": None if completed.returncode == 0 else
                       f"Collection {action} exited with code {completed.returncode}"}
        except Exception as exc:
            outcome = {"action": action, "message": str(exc), "error": type(exc).__name__}
        with self.lock:
            self.pending[action] = False
            self.last_result = outcome


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, app):
        self.app = app
        super().__init__(("127.0.0.1", port), Handler)
        self.hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "SensorDashboard/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_):
        pass  # Never log authentication material or sensor values.

    def respond(self, status, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(clean_json(data), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self, *, api=False, mutation=False):
        host = self.headers.get("Host", "")
        if host not in self.server.hosts:
            self.respond(403, {"error": "Invalid Host"})
            return False
        origin = self.headers.get("Origin")
        if ((mutation and origin != f"http://{host}") or
                (origin is not None and origin != f"http://{host}")):
            self.respond(403, {"error": "Invalid Origin"})
            return False
        if api and not secrets.compare_digest(self.headers.get("X-Dashboard-Token", ""), self.server.app.token):
            self.respond(401, {"error": "Use the dashboard URL printed in the Jetson terminal"})
            return False
        return True

    def do_GET(self):
        if not self.authorized(api=self.path.startswith("/api/")):
            return
        if self.path == "/api/state":
            self.respond(200, self.server.app.state())
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path not in assets:
            self.respond(404, {"error": "Not found"})
            return
        name, kind = assets[self.path]
        self.respond(200, (ROOT / "dashboard_web" / name).read_bytes(), kind)

    def do_POST(self):
        if not self.authorized(api=True, mutation=True):
            return
        if self.path != "/api/control":
            self.respond(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 256 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid request size")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise ValueError("JSON is required")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) != {"action"} or body["action"] not in ("start", "stop"):
                raise ValueError("Expected a start or stop action")
            self.server.app.submit(body["action"])
        except (ValueError, UnicodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.respond(409, {"error": str(exc)})
        else:
            self.respond(202, {"accepted": body["action"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out-dir", default=str(ROOT / "results/sensor_collection"))
    parser.add_argument("--duration", type=float, default=0, help="0: collect until Stop")
    parser.add_argument("--save-ct-raw", action="store_true", help="also save optional CT ADC raw samples")
    parser.add_argument("--replay", type=Path, help="view up to 120 recorded snapshots; controls disabled")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error("Invalid port or duration")
    app = Dashboard(out_dir=args.out_dir, duration=args.duration,
                    save_ct_raw=args.save_ct_raw, replay=args.replay)
    with Server(args.port, app) as server:
        url = f"http://127.0.0.1:{server.server_port}/#token={app.token}"
        connection = app.cache_root / f"dashboard-{server.server_port}.json"
        identity = process_identity(os.getpid())
        volatile_json(connection, {"url": url, "identity": identity, "mode": "replay" if app.replay else "live"})
        print(f"Jetson sensor dashboard: {url}\n"
              "Open this URL in the Jetson browser. Start/Stop controls collection.\n"
              "Closing this dashboard does not stop collection; use collect.sh stop.", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            if read_object(connection).get("identity") == identity:
                connection.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
