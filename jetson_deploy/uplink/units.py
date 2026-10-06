"""Find raw units that are finished and safe to ship, and describe them by SHA-256 manifest.

A unit is final only when its producer says so and nothing in it has changed for
``settle_s`` seconds:
* trial (``dataset/<group>/<trial>/experiment.json``): status completed / failed / aborted /
  preflight_failed (all are uploaded; ``status`` tells them apart);
* free run (``<runs>/<run>/metadata.json``): status completed / stopped / failed.
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
             now: float | None = None) -> tuple[list[Unit], list[dict]]:
    """Return (final units, skipped units with reasons). Never raises for one bad unit."""
    now = time.time() if now is None else now
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
            skipped.append({"unit": key, "reason": f"status {status!r} is not final"})
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
