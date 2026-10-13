"""Nightly copy of the hub data to a second disk, verified file by file afterwards.

Layout under ``dest``::

    current/            mirror of data_root (raw, manifests, receipts, annotations, config, tls)
    current/hub.sqlite  consistent snapshot (SQLite online backup API), not a file copy of a live DB
    replaced/<UTC>/     anything the mirror had to delete or overwrite (rsync --backup), never purged here
    status.json         result of the last run (also written to <data_root>/backup_status.json for the web)

``staging/`` (unfinished uploads) is not copied. After copying, every committed unit in the copy
is re-hashed against its own manifest and the manifest against its receipt, so a run only
reports ok when the second disk holds an intact, checkable copy of every receipt.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

STATUS_NAME = "backup_status.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def snapshot_sqlite(source: Path, target: Path) -> None:
    """Consistent copy of a live WAL database (readers and the hub keep running)."""
    temp = target.with_name(target.name + ".tmp")
    if temp.exists():
        temp.unlink()
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(temp)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    os.replace(temp, target)


def verify_copy(root: Path) -> dict:
    """Re-hash every unit of a mirrored data root against its manifest and receipt."""
    units, problems, files, size = 0, [], 0, 0
    for receipt_path in sorted((root / "receipts").rglob("*.json")):
        try:
            receipt = json.loads(receipt_path.read_text())
            device, unit = receipt["device"], receipt["unit"]
        except (OSError, ValueError, KeyError, TypeError):
            problems.append(f"unreadable receipt {receipt_path.relative_to(root)}")
            continue
        units += 1
        manifest = root / "manifests" / device / (unit + ".sha256")
        try:
            text = manifest.read_text()
        except OSError:
            problems.append(f"{device}/{unit}: manifest missing")
            continue
        if hashlib.sha256(text.encode()).hexdigest() != receipt.get("manifest_sha256"):
            problems.append(f"{device}/{unit}: manifest does not match receipt")
        for line in text.splitlines():
            sha, rel = line.split("  ", 1)
            path = root / "raw" / device / rel
            if not path.is_file() or _sha256(path) != sha:
                problems.append(f"{device}/{rel}")
            else:
                files += 1
                size += path.stat().st_size
    return {"units": units, "files": files, "bytes": size, "problems": problems[:100],
            "problem_count": len(problems)}


def _existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def run_backup(data_root: Path, dest: Path, *, rsync: str = "rsync", allow_same_filesystem: bool = False) -> dict:
    data_root, dest = Path(data_root).resolve(), Path(dest).resolve()
    started = time.monotonic()
    status = {"started_utc": utc_now(), "source": str(data_root), "dest": str(dest), "ok": False}
    try:
        if dest == data_root or data_root in dest.parents or dest in data_root.parents:
            raise ValueError("backup destination must be outside the data root")
        if not allow_same_filesystem and os.stat(_existing(dest)).st_dev == os.stat(data_root).st_dev:
            raise ValueError("backup destination is on the same filesystem as the data root")
        current = dest / "current"
        current.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        command = [rsync, "-a", "--delete", "--backup", f"--backup-dir={dest / 'replaced' / stamp}",
                   "--chmod=Du+rwx",                          # read-only unit dirs must stay writable in the copy
                   "--exclude=/staging/", "--exclude=/hub.sqlite*", f"--exclude=/{STATUS_NAME}*",
                   f"{data_root}/", f"{current}/"]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"rsync exit {done.returncode}: {done.stderr.strip()[-400:]}")
        if (data_root / "hub.sqlite").is_file():
            snapshot_sqlite(data_root / "hub.sqlite", current / "hub.sqlite")
        check = verify_copy(current)
        status.update(verify=check, ok=check["problem_count"] == 0,
                      replaced_files=sum(1 for p in (dest / "replaced" / stamp).rglob("*") if p.is_file())
                      if (dest / "replaced" / stamp).exists() else 0,
                      dest_free_bytes=shutil.disk_usage(dest).free)
        if not status["ok"]:
            status["error"] = f"{check['problem_count']} problem(s) in the copy"
    except Exception as error:   # recorded and shown on the web; the next run tries again
        status["error"] = f"{type(error).__name__}: {error}"[:500]
    status.update(finished_utc=utc_now(), seconds=round(time.monotonic() - started, 1))
    for path in (dest / "status.json", data_root / STATUS_NAME):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, status)
        except OSError:
            pass
    return status
