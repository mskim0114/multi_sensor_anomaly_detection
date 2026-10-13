"""On/off switch, live status and "which run is active" for the ZED recorder. Standard library only.

The switch is a small JSON file the recorder re-reads every loop, so turning the camera on or off
(locally with ``14_zed_recorder.py on|off`` or from the hub through the uplink agent) takes
effect within about a second, also in the middle of a run. Turning it off ends the current ZED
session cleanly; turning it on again starts a new session in the same run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import time

from uplink.units import FINAL_RUN, FINAL_TRIAL, process_alive

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("~/.config/factory_safety/zed.json").expanduser()
STATUS_NAME = "zed-status.json"
ACTIVE_NAME = "active.json"          # written by the collector's live preview publisher
DEFAULT_CONFIG = {
    "enabled": False,
    "resolution": "HD720",           # camera capture; the saved left image keeps this size
    "fps": 15,                       # grab rate; one frame per collector tick is saved
    "depth_mode": "PERFORMANCE",
    "depth_size": [640, 360],        # saved depth map size (width, height)
    "jpeg_quality": 85,
    "segment_frames": 60,            # frames per segment file pair (one minute at 1 Hz)
    "preview_every_s": 10,           # thumbnail for the hub's live page
    "roots": [str(REPO / "dataset"), str(REPO / "jetson_deploy" / "results" / "sensor_collection")],
}
RESOLUTIONS = ("HD720", "HD1080", "HD2K", "VGA")
DEPTH_MODES = ("NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL_LIGHT", "NEURAL")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Defaults overlaid with the file. A missing file means disabled; a broken file too (and
    says why), so a typo can never start recording by surprise."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return config
    except (OSError, ValueError) as error:
        return dict(config, enabled=False, config_error=f"{type(error).__name__}: {error}"[:200])
    if not isinstance(raw, dict):
        return dict(config, enabled=False, config_error="config is not a JSON object")
    config.update({k: v for k, v in raw.items() if k in DEFAULT_CONFIG or k in ("changed_utc", "changed_by")})
    problems = []
    if not isinstance(config["enabled"], bool):
        problems.append("enabled must be true or false")
    if config["resolution"] not in RESOLUTIONS:
        problems.append(f"resolution must be one of {RESOLUTIONS}")
    if config["depth_mode"] not in DEPTH_MODES:
        problems.append(f"depth_mode must be one of {DEPTH_MODES}")
    for key, low, high in (("fps", 1, 100), ("jpeg_quality", 30, 100), ("segment_frames", 1, 600),
                           ("preview_every_s", 1, 3600)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            problems.append(f"{key} must be an integer in {low}..{high}")
    size = config["depth_size"]
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and 16 <= v <= 4096 for v in size)):
        problems.append("depth_size must be [width, height]")
    if not (isinstance(config["roots"], list) and all(isinstance(r, str) for r in config["roots"])):
        problems.append("roots must be a list of paths")
    if problems:
        return dict(json.loads(json.dumps(DEFAULT_CONFIG)), enabled=False, config_error="; ".join(problems))
    return config


def set_enabled(enabled: bool, by: str = "local", path: Path = CONFIG_PATH) -> dict:
    """Flip only ``enabled`` (other keys in the file are kept) and return the effective config."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw = raw if isinstance(raw, dict) else {}
    except FileNotFoundError:
        raw = {}
    raw.update(enabled=bool(enabled), changed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               changed_by=str(by)[:80])
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(raw, indent=1, ensure_ascii=False) + "\n")
    return load_config(path)


def _atomic_text(path: Path, text: str) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def runtime_root(create: bool = False) -> Path:
    """The collector's volatile preview directory (same rule as sensors.live.preview_dir)."""
    uid = os.getuid()
    user_runtime = Path("/run/user") / str(uid)
    if user_runtime.is_dir() and not user_runtime.is_symlink() and user_runtime.stat().st_uid == uid:
        root = user_runtime / "factory-safety-preview"
    else:
        root = Path("/dev/shm") / f"factory-safety-preview-{uid}"
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
            raise PermissionError(f"preview directory must be private and user-owned: {root}")
    return root


def write_status(status: dict, root: Path | None = None) -> None:
    root = root if root is not None else runtime_root(create=True)
    _atomic_text(root / STATUS_NAME, json.dumps(dict(status, updated_epoch=time.time()), ensure_ascii=False))


def read_status(root: Path | None = None, max_age_s: float = 10.0) -> dict | None:
    """The recorder's last status, or None when it is not running (missing or stale file)."""
    path = (root if root is not None else runtime_root()) / STATUS_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or time.time() - float(value.get("updated_epoch") or 0) > max_age_s:
        return None
    return value


def process_identity(pid: int) -> dict:
    """pid + start ticks + boot id: what uplink.units.process_alive checks."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _inside(path: Path, roots) -> bool:
    for root in roots:
        try:
            path.relative_to(Path(root).expanduser().resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def active_run(roots, root: Path | None = None) -> tuple[dict | None, str]:
    """(run, reason). A run is active when the collector named in active.json is alive, its run
    directory is inside one of ``roots`` and it has not written a final status yet."""
    active = _json((root if root is not None else runtime_root()) / ACTIVE_NAME)
    if not active or not isinstance(active.get("run_dir"), str):
        return None, "no collection has run since boot"
    run_dir = Path(active["run_dir"])
    if not process_alive(active.get("identity") or {}):
        return None, "no collection running"
    try:
        run_dir = run_dir.resolve(strict=True)
    except OSError:
        return None, "run directory is missing"
    if not run_dir.is_dir() or run_dir.is_symlink():
        return None, "run directory is not a directory"
    if not _inside(run_dir, roots):
        return None, f"run is outside the configured roots ({run_dir.name})"
    experiment = _json(run_dir / "experiment.json")
    if experiment is not None:
        if experiment.get("status") in FINAL_TRIAL:
            return None, "trial already finished"
    else:
        control = _json(run_dir / "control.json") or {}
        if control.get("status") in FINAL_RUN:
            return None, "run already finished"
    return {"run_dir": run_dir, "run_id": run_dir.name, "identity": active["identity"],
            "kind": "trial" if experiment is not None else "run"}, "collecting"
