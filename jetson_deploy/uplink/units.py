"""Find raw units that are finished and safe to ship, and describe them by SHA-256 manifest.

A unit is final only when its producer says so and nothing in it has changed for
``settle_s`` seconds:
* trial (``dataset/<group>/<trial>/experiment.json``): status completed / failed / aborted /
  preflight_failed (all are uploaded; ``status`` tells them apart);
* free run (``<runs>/<run>/metadata.json``): status completed / stopped / failed;
* and, when the ZED side recorder wrote into it (``<unit>/zed/zed.json``), that recorder has
  marked it complete or is provably gone.
Files are read only. Symlinks and special files make a unit unshippable (reported, skipped).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
FINAL_TRIAL = {"completed", "failed", "aborted", "preflight_failed"}
FINAL_RUN = {"completed", "stopped", "failed"}
# A producer that died (reboot, power loss, kill) never writes a final status. Such a unit is
# shipped as "interrupted" only when its owner is provably gone and nothing has changed for
# settle_s; files are never touched. Statuses a live producer could still be in:
UNFINISHED = {None, "created", "ready", "starting", "running", "stopping"}
INTERRUPTED = "interrupted"
HARDWARE_LOCK = Path("~/.cache/factory_safety/acquisition/hardware.lock").expanduser()


@dataclass(frozen=True)
class Unit:
    key: str          # "dataset/<group>/<trial>" or "runs/<run>"
    path: Path
    status: str
    newest_mtime: float


def _status(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def process_alive(identity) -> bool:
    """pid alive and the same process (start ticks and boot id), read from /proc only."""
    try:
        pid = int(identity["pid"])
        stat = Path(f"/proc/{pid}/stat").read_text()
        ticks = stat.rsplit(")", 1)[1].split()[19]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, KeyError, ValueError, IndexError, TypeError):
        return False
    return ticks == str(identity.get("start_ticks")) and boot == identity.get("boot_id")


def _json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def run_owner_gone(run_dir: Path) -> bool | None:
    """Free run: its control.json names the collector process. None when it cannot be proven
    (no control.json / no identity, e.g. runs from before the control file existed)."""
    control = _json(run_dir / "control.json")
    identity = (control or {}).get("identity")
    if not isinstance(identity, dict):
        return None
    return not process_alive(identity)


def acquisition_owner_alive(lock_path: Path = HARDWARE_LOCK) -> bool:
    """Trials record no process identity. Every acquisition holds the hardware lock and writes
    its identity into the lock file; flock is released when the holder dies, so a dead
    recorded owner means no acquisition (and no trial) is running. Read-only: never locks."""
    owner = _json(lock_path) if lock_path.is_file() else None
    return bool(owner) and process_alive(owner)


def zed_still_writing(unit_dir: Path) -> bool:
    """True while the ZED recorder named in zed/zed.json is alive and has not marked it complete."""
    meta = _json(unit_dir / "zed" / "zed.json")
    if meta is None or meta.get("status") == "complete":
        return False
    recorder = meta.get("recorder")
    return isinstance(recorder, dict) and process_alive(recorder)


def list_files(unit_dir: Path) -> list[tuple[str, os.stat_result]]:
    """All regular files under the unit, relative POSIX paths. Raises on anything unusual."""
    out = []
    for dirpath, dirnames, filenames in os.walk(unit_dir, followlinks=False):
        for name in dirnames:
            if (Path(dirpath) / name).is_symlink():
                raise ValueError(f"symlink directory {name}")
        for name in filenames:
            full = Path(dirpath) / name
            info = full.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"not a regular file: {full.relative_to(unit_dir)}")
            rel = full.relative_to(unit_dir).as_posix()
            if not all(NAME.match(p) for p in rel.split("/")) or rel.count("/") > 3:
                raise ValueError(f"unshippable file name {rel!r}")
            out.append((rel, info))
    return sorted(out)


def discover(roots: dict[str, Path], *, include_test: bool = True, settle_s: float = 60.0,
             now: float | None = None, lock_path: Path = HARDWARE_LOCK) -> tuple[list[Unit], list[dict]]:
    """Return (final units, skipped units with reasons). Never raises for one bad unit."""
    now = time.time() if now is None else now
    acquisition_alive = None   # looked up once, only if needed
    candidates: list[tuple[str, Path, Path, set]] = []
    dataset = roots.get("dataset")
    if dataset and Path(dataset).is_dir():
        for group in sorted(Path(dataset).iterdir()):
            if group.is_symlink() or not group.is_dir() or not NAME.match(group.name):
                continue
            if group.name.startswith("_") and not include_test:
                continue
            for trial in sorted(group.iterdir()):
                if trial.is_symlink():
                    continue      # a link could point anywhere; only real unit directories ship
                if trial.is_dir() and NAME.match(trial.name) and (trial / "experiment.json").is_file():
                    candidates.append((f"dataset/{group.name}/{trial.name}", trial,
                                       trial / "experiment.json", FINAL_TRIAL))
    runs = roots.get("runs")
    if runs and Path(runs).is_dir():
        for run in sorted(Path(runs).iterdir()):
            if not run.is_symlink() and run.is_dir() and NAME.match(run.name) and (run / "metadata.json").is_file():
                candidates.append((f"runs/{run.name}", run, run / "metadata.json", FINAL_RUN))
    units, skipped = [], []
    for key, path, status_file, final in candidates:
        status = _status(status_file)
        if status not in final:
            if status not in UNFINISHED:
                skipped.append({"unit": key, "reason": f"status {status!r} is not final"})
                continue
            if key.startswith("runs/"):
                gone = run_owner_gone(path)
                if gone is None:
                    skipped.append({"unit": key, "reason": f"status {status!r} is not final and no "
                                    "control.json identity proves the collector is gone"})
                    continue
            else:
                if acquisition_alive is None:
                    acquisition_alive = acquisition_owner_alive(lock_path)
                gone = not acquisition_alive
            if not gone:
                skipped.append({"unit": key, "reason": f"status {status!r}: producer may still be running"})
                continue
            status = INTERRUPTED
        if zed_still_writing(path):
            skipped.append({"unit": key, "reason": "ZED recorder is still writing zed/"})
            continue
        try:
            files = list_files(path)
        except (OSError, ValueError) as error:
            skipped.append({"unit": key, "reason": str(error)})
            continue
        newest = max(info.st_mtime for _, info in files)
        if now - newest < settle_s:
            skipped.append({"unit": key, "reason": "settling (recently modified)"})
            continue
        units.append(Unit(key, path, status, newest))
    return units, skipped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(files: list[dict], unit_key: str) -> str:
    """Same text as the hub's ``sha256sum`` manifest (paths relative to raw/<device>)."""
    text = "".join(f"{f['sha256']}  {unit_key}/{f['path']}\n" for f in sorted(files, key=lambda f: f["path"]))
    return hashlib.sha256(text.encode()).hexdigest()
