"""The uplink agent: scan → hash → resumable upload → verified commit, plus heartbeats.

Config (JSON, default ~/.config/factory_safety/uplink.json)::

    {"device_id": "jetson-keti-01",
     "server_urls": ["https://10.252.219.59:8443", "https://<public-ip>:<forwarded-port>"],
     "token_file": "~/.config/factory_safety/uplink.token",
     "ca_file": "~/.config/factory_safety/hub-ca.pem",
     "roots": {"dataset": "~/projects/factory_safety/dataset",
               "runs": "~/projects/factory_safety/jetson_deploy/results/sensor_collection"},
     "include_test_units": true, "settle_s": 120, "scan_interval_s": 60,
     "heartbeat_interval_s": 10, "thermal_every_s": 30, "chunk_bytes": 4194304,
     "max_upload_bytes_per_s": 0, "allow_remote_control": false}

``allow_remote_control`` (default false) lets an operator on the hub start/stop the free-run
collector (``collect.py start|stop`` only; trials are never started remotely), reset the SPS30 and
switch the ZED side recorder on/off (``zed_on``/``zed_off`` only flip ``enabled`` in
~/.config/factory_safety/zed.json). It is a device-side opt-in: a hub cannot turn it on.

Nothing local is ever deleted. A unit whose files change after it was committed is reported
as ``local_changed_after_commit`` and never re-uploaded over the hub copy.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from . import AGENT_VERSION, PROTOCOL
from .client import HubClient, HubRejected, HubUnavailable
from .ledger import Ledger
from .units import HARDWARE_LOCK, Unit, discover, list_files, manifest_digest, process_alive, sha256_file
from zedcap import control as zed_control   # sibling package, standard library only

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("~/.config/factory_safety/uplink.json").expanduser()
STATE_DIR = Path("~/.cache/factory_safety/uplink").expanduser()
DEFAULTS = {"include_test_units": True, "settle_s": 120, "scan_interval_s": 60,
            "heartbeat_interval_s": 10, "thermal_every_s": 30, "chunk_bytes": 4 * 1024 * 1024,
            "max_upload_bytes_per_s": 0, "ca_file": None, "allow_remote_control": False,
            "hardware_lock": str(HARDWARE_LOCK), "zed_config": str(zed_control.CONFIG_PATH),
            "roots": {"dataset": str(REPO / "dataset"),
                      "runs": str(REPO / "jetson_deploy" / "results" / "sensor_collection")}}

log = logging.getLogger("uplink")


def load_config(path: Path) -> dict:
    config = dict(DEFAULTS)
    config.update(json.loads(Path(path).expanduser().read_text()))
    for key in ("device_id", "server_urls", "token_file"):
        if not config.get(key):
            raise ValueError(f"config needs {key}")
    token_path = Path(config["token_file"]).expanduser()
    mode = token_path.stat().st_mode
    if mode & 0o077:
        raise PermissionError(f"{token_path} must not be readable by others (chmod 600)")
    config["token"] = token_path.read_text().strip()
    if config.get("ca_file"):
        config["ca_file"] = str(Path(config["ca_file"]).expanduser())
    config["roots"] = {k: Path(v).expanduser() for k, v in config["roots"].items()}
    return config


class Throttle:
    def __init__(self, bytes_per_s: float):
        self.rate, self.start, self.sent = bytes_per_s, time.monotonic(), 0

    def consume(self, n: int) -> None:
        if self.rate <= 0:
            return
        self.sent += n
        ahead = self.sent / self.rate - (time.monotonic() - self.start)
        if ahead > 0:
            time.sleep(ahead)


class Agent:
    def __init__(self, config: dict, *, state_dir: Path = STATE_DIR, client: HubClient | None = None):
        self.config = config
        self.ledger = Ledger(Path(state_dir) / "ledger.sqlite")
        self.client = client or HubClient(config["server_urls"], config["token"], config["device_id"],
                                          ca_file=config.get("ca_file"))
        self.stop = threading.Event()
        self.last_scan: dict = {}
        self.connected = None
        self.current = None
        self._last_thermal = 0.0
        self._last_zed_preview = None
        self._command_lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._git = _git_state()

    # ----------------------------------------------------------- manifests
    def manifest(self, unit: Unit) -> list[dict]:
        files = []
        for rel, info in list_files(unit.path):
            sha = self.ledger.cached_hash(unit.key, rel, info.st_size, info.st_mtime_ns)
            if sha is None:
                sha = sha256_file(unit.path / rel)
                after = (unit.path / rel).stat()
                if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
                    raise RuntimeError(f"{rel} changed while hashing")
                self.ledger.store_hash(unit.key, rel, info.st_size, info.st_mtime_ns, sha)
            files.append({"path": rel, "size": info.st_size, "sha256": sha,
                          "mtime_ns": info.st_mtime_ns})
        return files

    # -------------------------------------------------------------- upload
    def upload(self, unit: Unit) -> str:
        """Upload one unit; returns its resulting ledger state."""
        record = self.ledger.unit(unit.key)
        files = self.manifest(unit)
        digest = manifest_digest(files, unit.key)
        total = sum(f["size"] for f in files)
        if record and record["state"] in ("committed", "local_changed_after_commit"):
            if record["manifest_sha256"] == digest:
                return "committed"
            self.ledger.set_state(unit.key, "local_changed_after_commit",
                                  error=f"local manifest {digest[:12]} differs from committed "
                                        f"{record['manifest_sha256'][:12]}; not re-uploaded")
            log.error("local raw unit %s changed after commit; NOT re-uploading", unit.key)
            return "local_changed_after_commit"
        self.ledger.set_state(unit.key, "uploading", manifest_sha256=digest, files=len(files), total=total,
                              attempt=True)
        self.current = {"unit": unit.key, "bytes": total, "sent": 0}
        meta = {"status": unit.status, "agent_version": AGENT_VERSION, "git": self._git,
                "local_path": str(unit.path)}
        wire = [{k: f[k] for k in ("path", "size", "sha256")} for f in files]
        begin = self.client.request("POST", "/api/v1/device/units/begin",
                                    json_body={"unit": unit.key, "files": wire, "meta": meta})
        if begin["state"] != "committed":
            if begin.get("manifest_sha256") != digest:
                raise RuntimeError("hub computed a different manifest digest")
            throttle = Throttle(float(self.config["max_upload_bytes_per_s"]))
            for f in files:
                self._send_file(unit, f, int(begin["received"].get(f["path"], 0)), throttle)
            for f in files:   # the unit must still be exactly what was hashed
                info = (unit.path / f["path"]).stat()
                if (info.st_size, info.st_mtime_ns) != (f["size"], f["mtime_ns"]):
                    raise RuntimeError(f"{f['path']} changed during upload")
            receipt = self.client.request("POST", "/api/v1/device/units/commit",
                                          json_body={"unit": unit.key, "manifest_sha256": digest})
        else:
            receipt = begin["receipt"]
        if receipt.get("manifest_sha256") != digest or receipt.get("files") != len(files) \
                or receipt.get("bytes") != total:
            raise RuntimeError("hub receipt does not match the local manifest")
        self.ledger.set_state(unit.key, "committed", receipt=receipt)
        self.current = None
        log.info("committed %s (%d files, %d bytes)", unit.key, len(files), total)
        return "committed"

    def _send_file(self, unit: Unit, entry: dict, offset: int, throttle: Throttle) -> None:
        size, chunk = entry["size"], int(self.config["chunk_bytes"])
        if offset >= size and size > 0:
            return
        with open(unit.path / entry["path"], "rb") as stream:
            while True:
                stream.seek(offset)
                data = stream.read(min(chunk, size - offset))
                query = {"unit": unit.key, "path": entry["path"], "offset": offset}
                try:
                    reply = self.client.request("PUT", "/api/v1/device/units/file", query=query, body=data)
                except HubRejected as error:
                    if error.status == 409 and "received" in error.body:
                        offset = int(error.body["received"])   # resume where the hub actually is
                        continue
                    if error.status == 422:
                        self.ledger.forget_hashes(unit.key)     # re-hash next time; never overwrite
                    raise
                throttle.consume(len(data))
                offset = int(reply["received"])
                if self.current:
                    self.current["sent"] += len(data)
                if reply.get("complete") or offset >= size:
                    return

    def scan_once(self) -> dict:
        units, skipped = discover(self.config["roots"], include_test=self.config["include_test_units"],
                                  settle_s=float(self.config["settle_s"]),
                                  lock_path=Path(self.config["hardware_lock"]).expanduser())
        result = {"final": len(units), "skipped": len(skipped), "committed": 0, "errors": 0}
        for unit in sorted(units, key=lambda u: u.newest_mtime):
            if self.stop.is_set():
                break
            record = self.ledger.unit(unit.key)
            if record and record["state"] == "committed" and self._unchanged(unit, record):
                continue
            try:
                state = self.upload(unit)
                result["committed"] += state == "committed"
                self.connected = True
            except HubUnavailable as error:
                self.connected = False
                self.ledger.set_state(unit.key, "pending", error=f"hub unreachable: {error}")
                log.warning("hub unreachable, will retry: %s", error)
                break
            except (HubRejected, RuntimeError, OSError, ValueError) as error:
                result["errors"] += 1
                self.ledger.set_state(unit.key, "error", error=str(error))
                log.error("upload of %s failed: %s", unit.key, error)
            finally:
                self.current = None
        result["skipped_detail"] = skipped[:50]
        self.last_scan = dict(result, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return result

    def _unchanged(self, unit: Unit, record: dict) -> bool:
        """Cheap check: every file still matches its cached (size, mtime) hash entry."""
        try:
            files = list_files(unit.path)
        except (OSError, ValueError):
            return False
        return len(files) == record["files"] and all(
            self.ledger.cached_hash(unit.key, rel, info.st_size, info.st_mtime_ns) for rel, info in files)

    # ----------------------------------------------------------- heartbeat
    def heartbeat_body(self) -> dict:
        body = {"protocol": PROTOCOL, "agent_version": AGENT_VERSION, "device_id": self.config["device_id"],
                "interval_s": self.config["heartbeat_interval_s"], "sent_utc": _utc(), "sent_epoch": time.time(),
                "host": socket.gethostname(), "uptime_s": _uptime(), "load": os.getloadavg(),
                "memory": _meminfo(), "temperatures_c": _thermal_zones(), "local_ip": self._local_ip(),
                "disks": {k: _disk(v) for k, v in self.config["roots"].items()},
                "collector": _collector_state(), "uploads": self.ledger.summary(),
                "current_upload": self.current, "last_scan": {k: v for k, v in self.last_scan.items()
                                                              if k != "skipped_detail"},
                "git": self._git, "remote_control": bool(self.config.get("allow_remote_control")),
                "zed": self.zed_state()}
        preview = self.zed_preview()
        if preview is not None:
            body["zed_preview"] = preview
        live = _latest_preview()
        if live is not None:
            due = time.time() - self._last_thermal >= float(self.config["thermal_every_s"])
            if due and live.get("thermal"):
                self._last_thermal = time.time()
            else:
                live["thermal"] = None
            body["live"] = live
        return body

    def _zed_config_path(self) -> Path:
        return Path(self.config.get("zed_config") or DEFAULTS["zed_config"]).expanduser()

    def zed_state(self) -> dict:
        """Switch position and the recorder's own status (without its thumbnail)."""
        config = zed_control.load_config(self._zed_config_path())
        status = zed_control.read_status() or {}
        keep = ("state", "reason", "error", "run_id", "session", "frames", "missed_ticks", "last_tick",
                "bytes_written", "grab_fps", "camera")
        return {"enabled": config["enabled"], "config_error": config.get("config_error"),
                "changed_utc": config.get("changed_utc"), "changed_by": config.get("changed_by"),
                "recorder_running": bool(status), **{k: status.get(k) for k in keep if k in status}}

    def zed_preview(self) -> dict | None:
        """A new ZED thumbnail at most every thermal_every_s (it is ~15 kB of base64)."""
        status = zed_control.read_status() or {}
        preview = status.get("preview")
        if not isinstance(preview, dict) or status.get("state") != "recording":
            return None
        key = (preview.get("run_id"), preview.get("tick"))
        if key == self._last_zed_preview:
            return None
        self._last_zed_preview = key
        return {k: preview.get(k) for k in ("run_id", "tick", "utc", "jpeg_b64")}

    def _local_ip(self) -> str | None:
        host = urlsplit(self.client.active_url).hostname
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((host, 9))   # no packet is sent for UDP connect
                return probe.getsockname()[0]
        except OSError:
            return None

    # -------------------------------------------------------- remote control
    def handle_commands(self, commands) -> None:
        """Run hub commands one at a time in a worker thread; results are reported later."""
        for command in commands if isinstance(commands, list) else []:
            if not isinstance(command, dict):
                continue
            command_id, action = command.get("id"), command.get("action")
            if not isinstance(command_id, str) or not self.ledger.command_seen(command_id, str(action)):
                continue                                   # malformed, or already handled
            log.info("hub command %s %s requested by %s", command_id, action, command.get("requested_by"))
            if not self.config.get("allow_remote_control"):
                result = {"ok": False, "error": "remote control is disabled on this device (allow_remote_control)"}
            elif action not in ("start", "stop", "reset_sps30", "zed_on", "zed_off"):
                result = {"ok": False, "error": f"unsupported action {action!r}"}
            elif not self._command_lock.acquire(blocking=False):
                result = {"ok": False, "error": "another command is still running on the device"}
            else:
                threading.Thread(target=self._execute, args=(command_id, action, command.get("requested_by")),
                                 name="command", daemon=True).start()
                continue
            self.ledger.command_done(command_id, result)

    def _execute(self, command_id: str, action: str, requested_by=None) -> None:
        try:
            if action in ("zed_on", "zed_off"):
                result = self.zed_command(action, requested_by)
            else:
                result = self.collector_command(action, command_id)
        except Exception as error:   # report, never crash the agent
            result = {"ok": False, "error": repr(error)[:300]}
        finally:
            self._command_lock.release()
        log.info("hub command %s %s -> %s", command_id, action, result)
        self.ledger.command_done(command_id, result)
        self.report_command_results()

    def collector_command(self, action: str, command_id: str = "manual") -> dict:
        """start: a transient user unit, so the collector runs at normal priority instead of
        inheriting this agent's Nice=10 / idle I/O; stop: collect.py's own graceful stop."""
        launcher = str(REPO / "jetson_deploy" / "run_python.sh")
        collect = str(REPO / "jetson_deploy" / "collect.py")
        if action == "reset_sps30":
            return self.reset_sps30(launcher)
        if action == "start":
            suffix = command_id if re.fullmatch(r"[0-9a-f]{24}", command_id or "") else f"{time.time_ns():x}"
            argv = ["systemd-run", "--user", "--collect", "--quiet", f"--unit=factory-collect-{suffix}",
                    "-p", "Type=forking", "-p", "TimeoutStartSec=60", f"--working-directory={REPO}"]
            if os.environ.get("FACTORY_JETSON_VENV"):
                argv.append(f"--setenv=FACTORY_JETSON_VENV={os.environ['FACTORY_JETSON_VENV']}")
            argv += [launcher, collect, "start"]
            timeout = 90
        else:
            argv = [launcher, collect, "stop", "--timeout", "90"]
            timeout = 120
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env=dict(os.environ, PYTHONNOUSERSITE="1"))
        state = _collector_state() or {}
        deadline = time.monotonic() + 40
        while state.get("status") == "starting" and time.monotonic() < deadline:   # collect.py gives up at 20 s
            time.sleep(1)
            state = _collector_state() or {}
        expected = ("running",) if action == "start" else ("stopped", "completed")
        ok = done.returncode == 0 and state.get("status") in expected
        return {"ok": bool(ok), "returncode": done.returncode, "collector": state,
                "stderr_tail": done.stderr[-300:] if done.returncode else ""}

    def zed_command(self, action: str, requested_by=None) -> dict:
        """Flip the ZED switch. The recorder picks it up within about a second; the collector is
        not touched, so this is allowed during a run (a new ZED session starts or the current one ends)."""
        config = zed_control.set_enabled(action == "zed_on", by=f"hub:{str(requested_by or '?')[:60]}",
                                         path=self._zed_config_path())
        ok = not config.get("config_error") and config["enabled"] == (action == "zed_on")
        return {"ok": ok, "zed": {"enabled": config["enabled"], "config_error": config.get("config_error"),
                                  "recorder_running": zed_control.read_status() is not None}}

    def reset_sps30(self, launcher: str) -> dict:
        """Operator-requested SPS30 soft reset, only while no acquisition runs (the tool also
        takes the hardware lock itself and refuses if anything holds the bus)."""
        state = _collector_state() or {}
        if state.get("alive"):
            return {"ok": False, "error": "collector is running; stop it before resetting the SPS30",
                    "collector": state}
        tool = str(REPO / "jetson_deploy" / "scripts" / "13_reset_sps30.py")
        done = subprocess.run([launcher, tool], capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, PYTHONNOUSERSITE="1"))
        try:
            report = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            report = {"stdout_tail": done.stdout[-300:]}
        steps = report.get("steps") or {}
        summary = {name: ("ok" if step.get("ok") else step.get("error", "failed"))
                   for name, step in steps.items() if isinstance(step, dict)}
        return {"ok": done.returncode == 0, "returncode": done.returncode, "sps30": {
            "ok": report.get("ok"), "steps": summary, "measurement": report.get("measurement"),
            "hint": report.get("hint"), "error": report.get("error"), "log": report.get("log")},
            "stderr_tail": done.stderr[-300:] if done.returncode not in (0, 1) else ""}

    def report_command_results(self) -> None:
        if not self._report_lock.acquire(blocking=False):
            return                                      # the other thread is already reporting
        try:
            self._report_pending()
        finally:
            self._report_lock.release()

    def _report_pending(self) -> None:
        for command_id, result in self.ledger.unreported_commands():
            try:
                self.client.request("POST", "/api/v1/device/command_result",
                                    json_body={"id": command_id, "ok": bool(result.get("ok")), "result": result})
            except HubRejected as error:
                if error.status in (400, 404):             # the hub will never accept it; stop retrying
                    self.ledger.command_reported(command_id)
                continue
            except HubUnavailable:
                return
            self.ledger.command_reported(command_id)

    def heartbeat_loop(self) -> None:
        while not self.stop.is_set():
            try:
                reply = self.client.request("POST", "/api/v1/device/heartbeat", json_body=self.heartbeat_body())
                self.handle_commands(reply.get("commands"))
                self.report_command_results()
                if self.connected is not True:
                    log.info("hub reachable at %s", self.client.active_url)
                self.connected = True
            except HubUnavailable as error:
                if self.connected is not False:
                    log.warning("heartbeat: hub unreachable: %s", error)
                self.connected = False
            except Exception as error:  # a heartbeat problem must not stop uploads
                log.error("heartbeat failed: %r", error)
            self.stop.wait(float(self.config["heartbeat_interval_s"]))

    def run(self) -> None:
        threading.Thread(target=self.heartbeat_loop, name="heartbeat", daemon=True).start()
        backoff = 0.0
        while not self.stop.is_set():
            try:
                result = self.scan_once()
                backoff = 0.0 if self.connected else min(600.0, max(15.0, backoff * 2))
                if result["committed"] or result["errors"]:
                    log.info("scan: %s", {k: v for k, v in result.items() if k != "skipped_detail"})
            except Exception as error:
                log.exception("scan failed: %r", error)
                backoff = min(600.0, max(15.0, backoff * 2))
            self.stop.wait(max(float(self.config["scan_interval_s"]), backoff))


# --------------------------------------------------------------- host facts
def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uptime() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError):
        return None


def _meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                out[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _thermal_zones() -> dict:
    out = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            out[(zone / "type").read_text().strip()] = int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            continue
    return out


def _disk(path: Path) -> dict | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return {"total": usage.total, "free": usage.free}


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _collector_state() -> dict | None:
    """Collector state as collect.py reports it, without its control token or process identity."""
    path = Path("~/.cache/factory_safety/acquisition/session.json").expanduser()
    try:
        session = json.loads(path.read_text())
        control = json.loads((Path(session["run_dir"]) / "control.json").read_text())
    except (OSError, ValueError, KeyError, TypeError):
        control = {}
        try:
            session = json.loads(path.read_text())
        except (OSError, ValueError):
            return {"status": "stopped", "alive": False}
    identity = session.get("identity") or {}
    alive = _process_alive(identity)
    status = control.get("status") if control.get("token") == session.get("token") else "starting"
    if not alive and status not in ("completed", "stopped", "failed"):
        status = "failed"   # same rule as collect.py: exited before finalization
    keep = ("updated_utc", "started_utc", "completed_utc", "error", "failure")
    return {"status": status, "alive": alive, "run_dir": session.get("run_dir"),
            **{k: control[k] for k in keep if k in control}}


_process_alive = process_alive   # one implementation, shared with units.py


def _latest_preview(max_age_s: float = 5.0) -> dict | None:
    """The collector's volatile latest-only preview (not raw data), if fresh."""
    roots = [Path("/run/user") / str(os.getuid()) / "factory-safety-preview",
             Path("/dev/shm") / f"factory-safety-preview-{os.getuid()}"]
    candidates = [p for root in roots if root.is_dir() for p in root.glob("run-*.json")]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    age = time.time() - newest.stat().st_mtime
    if age > max_age_s:
        return None
    try:
        value = json.loads(newest.read_text())
    except (OSError, ValueError):
        return None
    snapshot = value.get("snapshot") or {}
    sensors = {}
    for name, record in (snapshot.get("sensors") or {}).items():
        record = record or {}
        values = {k: v for k, v in (record.get("values") or {}).items()
                  if not isinstance(v, (dict, list)) and "monotonic" not in k}
        sensors[name] = {"status": record.get("status"), "fresh": record.get("fresh"),
                         "age_ms": record.get("age_ms"), "values": values}
    return {"run_id": value.get("run_id"), "age_s": round(age, 1),
            "snapshot": {"sequence": snapshot.get("sequence"), "timestamp_utc": snapshot.get("timestamp_utc"),
                         "quality": snapshot.get("quality"), "sensors": sensors},
            "thermal": value.get("thermal")}


# --------------------------------------------------------------------- CLI
def _setup_logging(state_dir: Path, verbose: bool) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(state_dir / "uplink.log", maxBytes=5_000_000, backupCount=5)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    log.handlers[:] = [handler, stream]
    log.setLevel(logging.DEBUG if verbose else logging.INFO)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="uplink_agent.py",
                                     description="Ship completed raw units to the edge hub.")
    parser.add_argument("command", choices=["run", "once", "check", "status", "pending"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.state_dir, args.verbose)
    config = load_config(args.config)
    agent = Agent(config, state_dir=args.state_dir)
    if args.command == "check":
        reply = agent.client.request("GET", "/api/v1/device/whoami")
        print(json.dumps({"hub": agent.client.active_url, **reply}))
        return 0 if reply.get("device") == config["device_id"] else 1
    if args.command == "status":
        print(json.dumps(agent.ledger.summary(), indent=1))
        return 0
    if args.command == "pending":
        units, skipped = discover(config["roots"], include_test=config["include_test_units"],
                                  settle_s=float(config["settle_s"]),
                                  lock_path=Path(config["hardware_lock"]).expanduser())
        rows = [{"unit": u.key, "status": u.status, "ledger": (agent.ledger.unit(u.key) or {}).get("state")}
                for u in units]
        print(json.dumps({"final": rows, "skipped": skipped}, indent=1))
        return 0
    if args.command == "once":
        result = agent.scan_once()
        print(json.dumps({k: v for k, v in result.items() if k != "skipped_detail"}))
        return 0 if result["errors"] == 0 and agent.connected is not False else 1
    signal.signal(signal.SIGTERM, lambda *_: agent.stop.set())
    log.info("uplink agent %s for %s → %s", AGENT_VERSION, config["device_id"], config["server_urls"])
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
