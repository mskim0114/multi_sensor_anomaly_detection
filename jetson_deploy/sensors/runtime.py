"""Linux process ownership and durable metadata for sensor acquisition."""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import tempfile


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def runtime_dir() -> Path:
    # Deliberately independent of checkout, output directory and working dir.
    path = Path.home() / ".cache" / "factory_safety" / "acquisition"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class AcquisitionBusyError(RuntimeError):
    pass


class AcquisitionLock:
    """Advisory lock shared by every Python hardware entrypoint for this user.

    Never unlink a lock file: another process may already hold the same inode.
    Non-cooperating external programs (e.g. gst-launch) must be stopped separately.
    """

    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else runtime_dir() / "hardware.lock"
        self._stream = None

    def acquire(self):
        if self._stream is not None:
            return self
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            owner = stream.read(1024).strip()
            stream.close()
            raise AcquisitionBusyError(f"Acquisition already active; lock: {self.path}; owner: {owner}") from None
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        stream.seek(0)
        stream.truncate()
        json.dump(process_identity(os.getpid()), stream)
        stream.flush()
        return self

    def release(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()


def process_identity(pid: int) -> dict:
    # comm (inside parentheses) may itself contain spaces and parentheses.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def owned_pidfd(identity: dict) -> int | None:
    """Pin a process first, then verify its birth identity before any signal."""
    fd = None
    try:
        fd = os.pidfd_open(int(identity["pid"]))
        if process_identity(int(identity["pid"])) != identity:
            os.close(fd)
            return None
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        if poller.poll(0):
            os.close(fd)
            return None
        return fd
    except (OSError, ValueError, KeyError, IndexError):
        if fd is not None:
            os.close(fd)
        return None


def terminate_owned(identity: dict, timeout: float) -> bool:
    """Request graceful shutdown. Never escalate to KILL or signal a reused PID."""
    fd = owned_pidfd(identity)
    if fd is None:
        return True
    try:
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            return True
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        return bool(poller.poll(int(timeout * 1000)))
    finally:
        os.close(fd)
