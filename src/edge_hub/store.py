"""Storage for the edge hub: resumable staging, verified commit, read-only raw tree, index.

Layout under ``data_root``::

    config/devices.json          device id -> SHA-256 of its upload token (never the token)
    config/viewers.json          viewer user -> PBKDF2 hash
    staging/<device>/<key>/      unit being received (state.json + data/<files>.part)
    raw/<device>/<unit>/         committed units, byte-identical to the edge, read-only
    manifests/<device>/<unit>.sha256   ``sha256sum -c`` manifest, paths relative to raw/<device>
    receipts/<device>/<unit>.json
    hub.sqlite                   index of units, devices and recent telemetry

A unit is ``dataset/<group>/<trial>`` (trial runner) or ``runs/<run>`` (free-run collector).
``raw/<device>/dataset/<group>`` is directly usable as ``build_windows.py --scan-dir``.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import zipfile

NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
DEVICE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
UNIT_ROOTS = {"dataset": 3, "runs": 2}           # first component -> number of components
MAX_OPEN_UNITS = 64                              # unfinished uploads per device
MIN_FREE_BYTES = 20 * 1024 ** 3                  # keep this much free on the data disk
MAX_FRAMES_PER_CHUNK = 1000
MAX_FILES = 20000
MAX_FILE_BYTES = 64 * 1024 ** 3
SHA = re.compile(r"^[0-9a-f]{64}$")
TELEMETRY_KEEP_S = 48 * 3600


class HubError(Exception):
    """An error with an HTTP status, raised for client mistakes and conflicts."""

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def check_device(device: str) -> str:
    if not isinstance(device, str) or not DEVICE.match(device):
        raise HubError(400, "invalid device id")
    return device


def check_unit(unit: str) -> tuple[str, ...]:
    parts = tuple(unit.split("/")) if isinstance(unit, str) else ()
    if not parts or parts[0] not in UNIT_ROOTS or len(parts) != UNIT_ROOTS[parts[0]]:
        raise HubError(400, "unit must be dataset/<group>/<trial> or runs/<run>")
    if not all(NAME.match(p) for p in parts[1:]):
        raise HubError(400, "invalid unit name")
    return parts


def check_relpath(path: str) -> tuple[str, ...]:
    parts = tuple(path.split("/")) if isinstance(path, str) else ()
    if not 1 <= len(parts) <= 4 or not all(NAME.match(p) for p in parts):
        raise HubError(400, f"invalid file path {path!r}")
    if any(part.endswith(".part") for part in parts):
        raise HubError(400, "path components may not end in .part")
    return parts


def manifest_text(files: list[dict], unit: str) -> str:
    """``sha256sum`` format, paths relative to raw/<device>, sorted, LF line ends."""
    return "".join(f"{f['sha256']}  {unit}/{f['path']}\n" for f in sorted(files, key=lambda f: f["path"]))


def manifest_digest(files: list[dict], unit: str) -> str:
    return hashlib.sha256(manifest_text(files, unit).encode()).hexdigest()


def check_manifest(files) -> list[dict]:
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise HubError(400, "manifest must list 1..20000 files")
    seen, clean = set(), []
    for item in files:
        if not isinstance(item, dict):
            raise HubError(400, "manifest entries must be objects")
        path, size, sha = item.get("path"), item.get("size"), item.get("sha256")
        check_relpath(path)
        if path in seen:
            raise HubError(400, f"duplicate path {path}")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_FILE_BYTES:
            raise HubError(400, f"invalid size for {path}")
        if not isinstance(sha, str) or not SHA.match(sha):
            raise HubError(400, f"invalid sha256 for {path}")
        seen.add(path)
        clean.append({"path": path, "size": size, "sha256": sha})
    # a file and a directory of the same name cannot both exist
    dirs = {"/".join(p.split("/")[:i]) for p in seen for i in range(1, p.count("/") + 1)}
    if dirs & seen:
        raise HubError(400, "a path is both a file and a directory")
    return sorted(clean, key=lambda f: f["path"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True).encode()


MAX_DOC_BYTES = 4 * 1024 * 1024


def _read_json(path: Path, limit: int = MAX_DOC_BYTES):
    """A JSON object from a file, or None (missing, too large, not JSON, not an object)."""
    try:
        if path.stat().st_size > limit:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


class Store:
    def __init__(self, data_root: Path):
        self.root = Path(data_root).resolve()
        for sub in ("config", "staging", "raw", "manifests", "receipts"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        os.chmod(self.root / "config", 0o700)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._db_lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "hub.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS units(
                device TEXT, unit TEXT, kind TEXT, files INTEGER, bytes INTEGER,
                manifest_sha256 TEXT, committed_utc TEXT, summary TEXT,
                PRIMARY KEY(device, unit));
            CREATE TABLE IF NOT EXISTS devices(
                device TEXT PRIMARY KEY, last_seen_utc TEXT, last_seen_epoch REAL,
                remote_addr TEXT, status TEXT);
            CREATE TABLE IF NOT EXISTS telemetry(
                device TEXT, epoch REAL, values_json TEXT);
            CREATE INDEX IF NOT EXISTS telemetry_device_epoch ON telemetry(device, epoch);
            CREATE INDEX IF NOT EXISTS telemetry_epoch ON telemetry(epoch);
        """)
        self.db.commit()
        self.live: dict[str, dict] = {}
        self._last_telemetry: dict[str, float] = {}

    # ---------------------------------------------------------------- helpers
    def _lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _staging(self, device: str, unit: str) -> Path:
        key = hashlib.sha256(unit.encode()).hexdigest()[:24]
        return self.root / "staging" / device / key

    def raw_dir(self, device: str, unit: str) -> Path:
        check_device(device)
        return self.root / "raw" / device / Path(*check_unit(unit))

    def _receipt_path(self, device: str, unit: str) -> Path:
        return self.root / "receipts" / device / (unit + ".json")

    def _committed(self, device: str, unit: str):
        receipt = _read_json(self._receipt_path(device, unit))
        return receipt if self.raw_dir(device, unit).is_dir() and receipt else None

    # ------------------------------------------------------------- upload API
    def begin(self, device: str, unit: str, files, meta=None) -> dict:
        check_device(device)
        check_unit(unit)
        files = check_manifest(files)
        digest = manifest_digest(files, unit)
        with self._lock(f"{device}:{unit}"):
            done = self._committed(device, unit)
            if done:
                if done["manifest_sha256"] != digest:
                    raise HubError(409, "unit already committed with a different manifest",
                                   committed_manifest_sha256=done["manifest_sha256"])
                return {"state": "committed", "receipt": done}
            if self.raw_dir(device, unit).exists():
                raise HubError(409, "raw directory exists without a receipt; manual review needed")
            stage = self._staging(device, unit)
            state = _read_json(stage / "state.json")
            if state and state.get("manifest_sha256") != digest:
                # the edge re-hashed and changed its mind: the old partial bytes are not reusable
                shutil.rmtree(stage)
                state = None
            if not state:
                open_units = [d for d in (self.root / "staging" / device).glob("*") if d.is_dir()] \
                    if (self.root / "staging" / device).is_dir() else []
                if len(open_units) >= MAX_OPEN_UNITS:
                    raise HubError(429, "too many unfinished uploads for this device")
                need = sum(f["size"] for f in files)
                if need > shutil.disk_usage(self.root).free - MIN_FREE_BYTES:
                    raise HubError(507, "not enough free space on the hub for this unit")
                state = {"device": device, "unit": unit, "manifest_sha256": digest, "files": files,
                         "meta": meta if isinstance(meta, dict) else {}, "created_utc": utc_now()}
                (stage / "data").mkdir(parents=True, exist_ok=True)
                _atomic_write(stage / "state.json", _json_bytes(state))
            return {"state": "open", "manifest_sha256": digest, "received": self._received(stage, files)}

    def _received(self, stage: Path, files: list[dict]) -> dict:
        out = {}
        for f in files:
            final = stage / "data" / f["path"]
            part = final.with_name(final.name + ".part")
            if final.is_file():
                out[f["path"]] = f["size"]
            elif part.is_file():
                have = part.stat().st_size
                if have >= f["size"]:
                    # interrupted between the last write and verify/rename: finish (or drop) it now
                    have = 0
                    if part.stat().st_size == f["size"] and sha256_file(part) == f["sha256"]:
                        os.replace(part, final)
                        have = f["size"]
                    else:
                        part.unlink()
                out[f["path"]] = have
            else:
                out[f["path"]] = 0
        return out

    def put(self, device: str, unit: str, path: str, offset: int, stream, length: int) -> dict:
        check_device(device)
        check_unit(unit)
        check_relpath(path)
        with self._lock(f"{device}:{unit}"):
            stage = self._staging(device, unit)
            state = _read_json(stage / "state.json")
            if not state:
                raise HubError(409, "no open upload for this unit; call begin first")
            entry = next((f for f in state["files"] if f["path"] == path), None)
            if entry is None:
                raise HubError(400, "path is not in the manifest")
            final = stage / "data" / path
            part = final.with_name(final.name + ".part")
            if final.is_file():
                return {"path": path, "received": entry["size"], "complete": True}
            have = part.stat().st_size if part.is_file() else 0
            if offset != have:
                raise HubError(409, "offset does not match received bytes", received=have)
            if have + length > entry["size"]:
                raise HubError(400, "chunk would exceed the declared size", received=have)
            part.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "ab") as out:
                left = length
                while left:
                    block = stream.read(min(left, 1 << 20))
                    if not block:
                        break
                    out.write(block)
                    left -= len(block)
                out.flush()
                os.fsync(out.fileno())
            have = part.stat().st_size
            if have - offset != length:
                # truncate back so a broken connection never leaves half a chunk behind
                with open(part, "r+b") as out:
                    out.truncate(offset)
                raise HubError(400, "request body shorter than Content-Length", received=offset)
            if have == entry["size"]:
                actual = sha256_file(part)
                if actual != entry["sha256"]:
                    part.unlink()
                    raise HubError(422, "sha256 mismatch; file discarded", path=path, received=0,
                                   expected=entry["sha256"], actual=actual)
                os.replace(part, final)
                return {"path": path, "received": have, "complete": True}
            return {"path": path, "received": have, "complete": False}

    def commit(self, device: str, unit: str, manifest_sha256: str) -> dict:
        check_device(device)
        check_unit(unit)
        with self._lock(f"{device}:{unit}"):
            done = self._committed(device, unit)
            if done:
                if done["manifest_sha256"] != manifest_sha256:
                    raise HubError(409, "unit already committed with a different manifest")
                return done
            stage = self._staging(device, unit)
            state = _read_json(stage / "state.json")
            if not state or state["manifest_sha256"] != manifest_sha256:
                raise HubError(409, "no open upload with this manifest")
            files = state["files"]
            missing = [f["path"] for f in files if not (stage / "data" / f["path"]).is_file()]
            if missing:
                raise HubError(409, "files are not complete", missing=missing[:20])
            data = stage / "data"
            extra = sorted(str(p.relative_to(data)) for p in data.rglob("*") if p.is_file()
                           and str(p.relative_to(data)) not in {f["path"] for f in files})
            if extra:
                raise HubError(409, "staging holds files outside the manifest", extra=extra[:20])
            for f in files:   # size again, sha was checked when each file completed
                if (data / f["path"]).stat().st_size != f["size"]:
                    raise HubError(409, f"size changed in staging for {f['path']}")
            target = self.raw_dir(device, unit)
            if target.exists():
                raise HubError(409, "raw directory exists without a receipt; manual review needed")
            target.parent.mkdir(parents=True, exist_ok=True)
            for p in sorted(data.rglob("*"), reverse=True):
                os.chmod(p, 0o444 if p.is_file() else 0o555)
            os.rename(data, target)
            os.chmod(target, 0o555)
            _fsync_dir(target.parent)
            total = sum(f["size"] for f in files)
            text = manifest_text(files, unit)
            receipt = {"device": device, "unit": unit, "files": len(files), "bytes": total,
                       "manifest_sha256": manifest_sha256, "committed_utc": utc_now(),
                       "raw_path": str(target.relative_to(self.root)),
                       "meta": state.get("meta", {})}
            assert hashlib.sha256(text.encode()).hexdigest() == manifest_sha256
            _atomic_write(self.root / "manifests" / device / (unit + ".sha256"), text.encode(), 0o444)
            _atomic_write(self._receipt_path(device, unit), _json_bytes(receipt), 0o444)
            shutil.rmtree(stage)
            self._index(device, unit, receipt)
            return receipt

    def _index(self, device: str, unit: str, receipt: dict) -> None:
        try:
            summary = unit_summary(self.raw_dir(device, unit))
        except Exception as error:  # a malformed document must not hide a verified unit
            summary = {"summary_error": repr(error)[:200]}
        with self._db_lock:
            self.db.execute("INSERT OR REPLACE INTO units VALUES(?,?,?,?,?,?,?,?)",
                            (device, unit, unit.split("/")[0], receipt["files"], receipt["bytes"],
                             receipt["manifest_sha256"], receipt["committed_utc"], json.dumps(summary)))
            self.db.commit()

    def reindex(self) -> int:
        count = 0
        for receipt_path in (self.root / "receipts").rglob("*.json"):
            receipt = _read_json(receipt_path)
            if receipt and self.raw_dir(receipt["device"], receipt["unit"]).is_dir():
                self._index(receipt["device"], receipt["unit"], receipt)
                count += 1
        return count

    def verify(self, device: str | None = None) -> list[dict]:
        """Re-hash every committed unit against its manifest (an integrity audit, read-only)."""
        results = []
        for receipt_path in sorted((self.root / "receipts").rglob("*.json")):
            receipt = _read_json(receipt_path)
            if not receipt or (device and receipt["device"] != device):
                continue
            base = self.root / "raw" / receipt["device"]
            manifest = self.root / "manifests" / receipt["device"] / (receipt["unit"] + ".sha256")
            bad = []
            try:
                text = manifest.read_text()
            except OSError:
                results.append({"device": receipt["device"], "unit": receipt["unit"], "ok": False,
                                "problems": ["manifest file missing"]})
                continue
            if hashlib.sha256(text.encode()).hexdigest() != receipt["manifest_sha256"]:
                bad.append("manifest file does not match receipt")
            for line in text.splitlines():
                sha, rel = line.split("  ", 1)
                path = base / rel
                if not path.is_file() or sha256_file(path) != sha:
                    bad.append(rel)
            listed = {line.split("  ", 1)[1] for line in text.splitlines()}
            on_disk = {str(p.relative_to(base)) for p in (base / receipt["unit"]).rglob("*") if p.is_file()}
            bad += sorted(f"unlisted: {p}" for p in on_disk - listed)
            results.append({"device": receipt["device"], "unit": receipt["unit"], "ok": not bad, "problems": bad})
        return results

    # -------------------------------------------------------------- telemetry
    def heartbeat(self, device: str, remote_addr: str, body: dict) -> None:
        check_device(device)
        now = time.time()
        live = _clean_live(body.get("live"))
        status = _clean_status(body)
        status["received_utc"] = utc_now()
        record = self.live.setdefault(device, {})
        record["status"] = status
        previous = record.get("live") or {}
        key = lambda value: (value.get("run_id"), (value.get("snapshot") or {}).get("sequence"))  # noqa: E731
        values = {}
        if live is not None:
            if live.get("thermal") is None and previous.get("thermal") and previous.get("run_id") == live.get("run_id"):
                live = dict(live, thermal=previous["thermal"], thermal_reused=True)
            # one telemetry point per collector snapshot, and at most one per 2 s per device
            if key(live) != key(previous) and now - self._last_telemetry.get(device, 0) >= 2:
                values = _scalar_values(live)
                self._last_telemetry[device] = now
            record["live"] = dict(live, stale=False, received_utc=status["received_utc"])
        elif previous:
            record["live"] = dict(previous, stale=True)   # collector stopped: keep, but say so
        with self._db_lock:
            self.db.execute("INSERT OR REPLACE INTO devices VALUES(?,?,?,?,?)",
                            (device, status["received_utc"], now, remote_addr, json.dumps(status)))
            if values:
                self.db.execute("INSERT INTO telemetry VALUES(?,?,?)", (device, now, json.dumps(values)))
            self.db.execute("DELETE FROM telemetry WHERE epoch < ?", (now - TELEMETRY_KEEP_S,))
            self.db.commit()

    # ------------------------------------------------------------- viewer API
    def overview(self) -> dict:
        now = time.time()
        with self._db_lock:
            devices = self.db.execute("SELECT device,last_seen_utc,last_seen_epoch,remote_addr,status "
                                      "FROM devices ORDER BY device").fetchall()
            counts = dict(((d, k), (n, b)) for d, k, n, b in self.db.execute(
                "SELECT device, kind, COUNT(*), SUM(bytes) FROM units GROUP BY device, kind"))
        registered = sorted(self.devices().keys())
        out = []
        for device in sorted(set(registered) | {d[0] for d in devices}):
            row = next((d for d in devices if d[0] == device), None)
            try:
                status = json.loads(row[4]) if row else {}
                interval = float(status.get("interval_s", 10))
            except (TypeError, ValueError):
                status, interval = {"error": "unreadable status"}, 10.0
            age = now - row[2] if row else None
            live = self.live.get(device, {}).get("live") or {}
            out.append({"device": device, "registered": device in registered,
                        "last_seen_utc": row[1] if row else None,
                        "age_s": round(age, 1) if age is not None else None,
                        "online": age is not None and age < 3 * interval + 5,
                        "remote_addr": row[3] if row else None, "status": status,
                        "latest_values": _scalar_values(live),
                        "latest_timestamp_utc": (live.get("snapshot") or {}).get("timestamp_utc"),
                        "units": {k: {"count": counts.get((device, k), (0, 0))[0],
                                      "bytes": counts.get((device, k), (0, 0))[1] or 0}
                                  for k in UNIT_ROOTS}})
        return {"server_utc": utc_now(), "devices": out}

    def live_view(self, device: str) -> dict:
        check_device(device)
        return self.live.get(device, {})

    def telemetry(self, device: str, since_s: float = 3600) -> dict:
        check_device(device)
        with self._db_lock:
            rows = self.db.execute("SELECT epoch, values_json FROM telemetry WHERE device=? AND epoch>=? "
                                   "ORDER BY epoch DESC LIMIT 20000", (device, time.time() - since_s)).fetchall()[::-1]
        stride = max(1, len(rows) // 1500)
        return {"device": device, "points": [{"t": e, **json.loads(v)} for e, v in rows[::stride]]}

    def units(self, device: str | None = None) -> list[dict]:
        query = "SELECT device,unit,kind,files,bytes,manifest_sha256,committed_utc,summary FROM units"
        args = ()
        if device:
            query += " WHERE device=?"
            args = (check_device(device),)
        with self._db_lock:
            rows = self.db.execute(query + " ORDER BY committed_utc DESC", args).fetchall()
        return [{"device": r[0], "unit": r[1], "kind": r[2], "files": r[3], "bytes": r[4],
                 "manifest_sha256": r[5], "committed_utc": r[6], "summary": json.loads(r[7])} for r in rows]

    def unit_detail(self, device: str, unit: str) -> dict:
        base = self.raw_dir(device, unit)
        receipt = self._committed(device, unit)
        if not receipt:
            raise HubError(404, "unit not found")
        files = sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())
        thermal = sorted(f for f in files if re.fullmatch(r"thermal_\d+\.npz", f))
        columns = []
        if (base / "scalars.csv").is_file():
            with open(base / "scalars.csv", newline="") as stream:
                columns = next(csv.reader(stream), [])
        docs = {name: _read_json(base / name) for name in ("experiment.json", "timing_report.json")
                if (base / name).is_file()}
        return {"receipt": receipt, "summary": unit_summary(base), "files": files,
                "thermal_chunks": thermal, "scalar_columns": columns, "documents": docs}

    def unit_scalars(self, device: str, unit: str, columns: list[str], max_points: int = 2000) -> dict:
        base = self.raw_dir(device, unit)
        if not self._committed(device, unit) or not (base / "scalars.csv").is_file():
            raise HubError(404, "no scalars.csv for this unit")
        try:
            return self._scalars(base, columns, max_points)
        except csv.Error as error:
            raise HubError(422, f"unreadable scalars.csv: {error}")

    @staticmethod
    def _scalars(base: Path, columns: list[str], max_points: int) -> dict:
        with open(base / "scalars.csv", newline="") as stream:
            header = next(csv.reader(stream), [])
            total = sum(1 for _ in stream)
        wanted = [c for c in columns if c in header][:8]
        stride = max(1, total // max_points)
        picked = []
        with open(base / "scalars.csv", newline="") as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if i % stride == 0:
                    picked.append(row)

        def num(value):
            try:
                x = float(value)
                return x if x == x and abs(x) != float("inf") else None
            except (TypeError, ValueError):
                return None
        return {"rows": total, "stride": stride,
                "sequence": [num(r.get("sequence")) for r in picked],
                "timestamp_utc": [r.get("timestamp_utc") for r in picked[:1] + picked[-1:]],
                "series": {c: [num(r.get(c)) for r in picked] for c in wanted}}

    def unit_thermal(self, device: str, unit: str, chunk: str, index: int) -> dict:
        base = self.raw_dir(device, unit)
        if not self._committed(device, unit) or not re.fullmatch(r"thermal_\d+\.npz", chunk or ""):
            raise HubError(404, "no such thermal chunk")
        path = base / chunk
        if not path.is_file():
            raise HubError(404, "no such thermal chunk")
        size = 120 * 160 * 2
        try:
            with zipfile.ZipFile(path) as archive:
                shape, frame = _npz_slice(archive, "frames", "<u2", (120, 160), index)
                seq_shape, seq_bytes = _npz_slice(archive, "sequences", "<i8", (), index)
        except (zipfile.BadZipFile, KeyError, ValueError, OSError) as error:
            raise HubError(422, f"unreadable thermal chunk: {error}")
        if seq_shape[0] != shape[0] or len(frame) != size:
            raise HubError(422, "inconsistent thermal chunk")
        seq = int.from_bytes(seq_bytes, "little", signed=True)
        import base64
        return {"chunk": chunk, "index": index, "frames_in_chunk": shape[0], "sequence": seq,
                "shape": [120, 160], "dtype": "<u2", "conversion": "raw / 100 - 273.15",
                "data": base64.b64encode(frame).decode()}

    # ---------------------------------------------------------------- devices
    def devices(self) -> dict:
        return _read_json(self.root / "config" / "devices.json") or {}

    def add_device(self, device: str, token_sha256: str, note: str = "") -> None:
        check_device(device)
        devices = self.devices()
        devices[device] = {"token_sha256": token_sha256, "created_utc": utc_now(), "note": note}
        _atomic_write(self.root / "config" / "devices.json", _json_bytes(devices), 0o600)

    def revoke_device(self, device: str) -> bool:
        devices = self.devices()
        if device not in devices:
            return False
        devices[device]["revoked"] = True
        devices[device]["revoked_utc"] = utc_now()
        _atomic_write(self.root / "config" / "devices.json", _json_bytes(devices), 0o600)
        return True

    def device_for_token(self, token: str) -> str | None:
        digest = hashlib.sha256(token.encode()).hexdigest()
        import hmac
        for device, entry in self.devices().items():
            if hmac.compare_digest(entry.get("token_sha256", ""), digest) and not entry.get("revoked"):
                return device
        return None


def _scalar_values(live: dict | None) -> dict:
    """Flatten the latest snapshot's sensor values into name -> number (for trends)."""
    snapshot = (live or {}).get("snapshot") or {}
    out = {}
    for sensor, record in (snapshot.get("sensors") or {}).items():
        values = (record or {}).get("values") or {}
        if isinstance(values, dict):
            for key, value in values.items():
                if (isinstance(value, (int, float)) and not isinstance(value, bool) and value == value
                        and "monotonic" not in key and not key.startswith("vdiff_")):
                    out[f"{sensor}.{key}"] = value
    return out


def _npz_slice(archive: zipfile.ZipFile, name: str, descr: str, tail: tuple, index: int):
    """Row ``index`` of a C-order little-endian .npy member, without numpy and without
    decompressing more than the header plus that row (bounded against zip bombs)."""
    import ast
    info = archive.getinfo(name + ".npy")
    item = 2 if descr == "<u2" else 8
    row = item * (1 if not tail else tail[0] * tail[1])
    if info.file_size > 4096 + row * MAX_FRAMES_PER_CHUNK:
        raise ValueError(f"{name} is larger than any real chunk")
    with archive.open(info) as stream:
        head = stream.read(12)
        if head[:6] != b"\x93NUMPY" or head[6] not in (1, 2):
            raise ValueError("not an npy array")
        header_len = int.from_bytes(head[8:10], "little") if head[6] == 1 else int.from_bytes(head[8:12], "little")
        if header_len > 4096:
            raise ValueError("npy header too long")
        start = 10 if head[6] == 1 else 12
        header = ast.literal_eval((head[start:] + stream.read(header_len - (12 - start))).decode("latin1"))
        if not isinstance(header, dict) or header.get("fortran_order") or header.get("descr") != descr:
            raise ValueError(f"unsupported array header {header!r}"[:200])
        shape = tuple(header.get("shape", ()))
        if len(shape) != 1 + len(tail) or tuple(shape[1:]) != tail or not all(isinstance(x, int) for x in shape):
            raise ValueError(f"unexpected shape {shape}")
        if not 0 <= index < shape[0]:
            raise HubError(404, "frame index out of range")
        remaining = row * index
        while remaining:                       # skip rows; zip members only stream forward
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("truncated array")
            remaining -= len(block)
        data = stream.read(row)
    return shape, data


def unit_summary(base: Path) -> dict:
    """Small descriptive fields for the index. Copies values; never interprets them as labels."""
    experiment = _read_json(base / "experiment.json") or {}
    metadata = _read_json(base / "metadata.json") or {}
    timing = _read_json(base / "timing_report.json") or {}
    master = timing.get("master") if isinstance(timing.get("master"), dict) else {}
    quality = timing.get("quality") if isinstance(timing.get("quality"), dict) else {}
    first = lambda *values: next((v for v in values if v is not None), None)  # noqa: E731
    out = {
        "status": first(experiment.get("status"), metadata.get("status"), timing.get("status")),
        "trial_id": first(experiment.get("trial_id"), metadata.get("run_id")),
        "scenario_id": first(experiment.get("scenario_id"), metadata.get("scenario_id")),
        "severity_level": first(experiment.get("severity_level"), metadata.get("severity_level")),
        "test_mode": experiment.get("test_mode"),
        "protocol_compliant": experiment.get("protocol_compliant"),
        "operator_note": experiment.get("operator_note"),
        "started_utc": first(experiment.get("started_at_utc"), metadata.get("started_utc")),
        "completed_utc": first(experiment.get("completed_at_utc"), metadata.get("completed_utc")),
        "abort_reason": experiment.get("abort_reason"),
        "snapshots": master.get("snapshot_count"),
        "missed_ticks": master.get("missed_ticks"),
        "windows_valid": quality.get("windows_valid"),
        "windows_evaluated": quality.get("windows_evaluated"),
        "host": first(experiment.get("host"), metadata.get("host")),
        "git_commit": first(experiment.get("git_commit"), metadata.get("git_commit")),
    }
    return {k: v for k, v in out.items() if v is not None}


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and abs(value) != float("inf") else None


def _short(value, limit=300):
    return value[:limit] if isinstance(value, str) else None


def _clean_tree(value, depth=0, budget=None):
    """JSON-safe copy limited in depth, keys and string length (status is shown, never trusted)."""
    budget = budget if budget is not None else [4000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 5:
        return None
    if isinstance(value, dict):
        return {str(k)[:80]: _clean_tree(v, depth + 1, budget) for k, v in list(value.items())[:100]}
    if isinstance(value, list):
        return [_clean_tree(v, depth + 1, budget) for v in value[:100]]
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, bool) or value is None:
        return value
    return _number(value)


def _clean_status(body: dict) -> dict:
    status = _clean_tree({k: v for k, v in body.items() if k != "live"})
    interval = _number(status.get("interval_s"))
    status["interval_s"] = interval if interval and 1 <= interval <= 3600 else 10
    return status


def _clean_live(live) -> dict | None:
    if not isinstance(live, dict):
        return None
    snapshot = live.get("snapshot") if isinstance(live.get("snapshot"), dict) else {}
    sensors = {}
    raw_sensors = snapshot.get("sensors") if isinstance(snapshot.get("sensors"), dict) else {}
    for name, record in list(raw_sensors.items())[:32]:
        if not isinstance(record, dict) or not isinstance(name, str):
            continue
        values = record.get("values") if isinstance(record.get("values"), dict) else {}
        sensors[name[:40]] = {
            "status": _short(record.get("status"), 40), "fresh": record.get("fresh") if isinstance(record.get("fresh"), bool) else None,
            "age_ms": _number(record.get("age_ms")),
            "values": {str(k)[:60]: _number(v) for k, v in list(values.items())[:40] if _number(v) is not None}}
    thermal = live.get("thermal") if isinstance(live.get("thermal"), dict) else None
    if thermal is not None:
        data = thermal.get("data")
        thermal = ({"data": data, "snapshot_sequence": _number(thermal.get("snapshot_sequence"))}
                   if isinstance(data, str) and len(data) == 51200 else None)   # 120*160*2 bytes, base64
    return {"run_id": _short(live.get("run_id"), 120), "age_s": _number(live.get("age_s")),
            "snapshot": {"sequence": _number(snapshot.get("sequence")),
                         "timestamp_utc": _short(snapshot.get("timestamp_utc"), 40),
                         "quality": _clean_tree(snapshot.get("quality")), "sensors": sensors},
            "thermal": thermal}
