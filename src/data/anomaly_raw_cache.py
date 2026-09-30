"""EXP-004 raw Normal-window cache: the canonical reader's arrays, stored exactly as read.

P1b re-injects synthetic events into fit/dev Normal windows every epoch. Doing that from
raw means decoding and re-hashing 90 files per window each time; caching the synthetic
windows instead would cost 44.8 GiB and change the raw -> inject -> normalize contract.
This cache keeps the contract: it stores what the canonical reader returns for each
fit/dev pure-Normal window, so ``inject_blind`` and ``normalized_pair`` are re-run on the
same arrays and their results are byte-identical to the canonical path.

Rules this module enforces (review update (23) C and (25) B):

* fit/dev pure-Normal windows only. Calibration, held-out, unassigned, hard-invalid or
  non-Normal windows are refused before the reader is called or any file is created.
* No dtype conversion. The caller declares the dtype of each array; a reader output that
  differs is a failure, never a silent promotion or demotion. Shape and C-contiguity are
  checked the same way. Byte order must be native.
* Raw bytes are read only through the injected reader. The canonical reader
  (``fit_anomaly_controls.make_reader``) verifies each raw file against the raw manifest
  while the cache is built, so the "raw content changed" check runs once, at build time.
* The source manifest digest, per-window raw file digests and the base/role/window order
  digests are recorded, so a consumer can tell which windows, in which order, from which
  raw content, the cache holds.
* Provenance and window identities are stored as they read back from JSON; anything
  that would not survive the round trip is refused before the reader is called.
* The output directory must be new and outside the raw data root. A failure leaves
  ``failure.json`` next to the partial files; nothing is deleted or retried.
* After flushing, every array file is streamed back from disk. Its payload must equal the
  bytes the reader returned, and the whole file must equal an independently built npy
  header followed by those bytes.

BIN sampling and dev weighting are still proposals and are deliberately absent here.
This is a NumPy leaf: it imports neither torch nor the ``src.data`` package.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _leaf(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loaded by path so the torch-importing src.data package is never imported.
sa = _leaf("_exp004_raw_cache_sa", "src/data/synthetic_anomaly.py")

SCHEMA = "exp004-raw-normal-window-cache-v1"
ALLOWED_ROLES = ("fit", "dev")
WINDOW_TICKS = 30
ARRAYS = ("sensor", "thermal", "labels")
WINDOW_SHAPES = {"sensor": (WINDOW_TICKS, len(sa.CHANNELS)),
                 "thermal": (WINDOW_TICKS, *sa.THERMAL_SHAPE),
                 "labels": (WINDOW_TICKS,)}
# Same layout the canonical reader opens; kept here only to look files up in the ledger.
RAW_FILES = (("csv", "원천데이터"), ("bin", "원천데이터"), ("json", "라벨링데이터"))
MANIFEST = "manifest.json"
FAILURE = "failure.json"
_BLOCK = 1 << 22


def _file_name(name):
    return f"{name}.npy"


def declared_layout(dtypes):
    """Validate the caller's dtype declaration; returns {array: np.dtype}.

    Every array must be declared explicitly. A non-native byte order is refused because
    the canonical reader never produces one and the stored bytes would not round-trip.
    """
    if not isinstance(dtypes, dict) or set(dtypes) != set(ARRAYS):
        raise ValueError(f"declare exactly the dtypes of {ARRAYS}")
    layout = {}
    for name in ARRAYS:
        dtype = np.dtype(dtypes[name])
        if dtype.kind not in "fiu" or dtype.hasobject or dtype.fields is not None:
            raise ValueError(f"{name} must be a plain numeric dtype, got {dtype}")
        if dtype.byteorder not in ("=", "|"):
            raise ValueError(f"{name} dtype {dtype.str} is not native byte order")
        layout[name] = dtype
    if layout["labels"].kind not in "iu":
        raise ValueError("labels must be an integer dtype")
    return layout


def required_bytes(count, layout):
    """Payload bytes of the three arrays for ``count`` windows, headers excluded."""
    layout = declared_layout(layout)
    return {name: int(count * np.prod(WINDOW_SHAPES[name])) * layout[name].itemsize
            for name in ARRAYS}


def select_windows(windows, roles):
    """Eligible fit then dev pure-Normal windows, each role in manifest order.

    Matches the order ``train_anomaly_pilot`` uses (fit windows followed by dev windows),
    so a cache index and a pilot index refer to the same window.
    """
    chosen = []
    for role in ALLOWED_ROLES:
        part = [w for w in windows
                if roles.get(w["device"]) == role and w["hard_valid"] is True
                and w["pure_Normal"] is True]
        if not part:
            raise ValueError(f"no eligible pure-Normal window in role {role}")
        expected = sorted(d for d, r in roles.items() if r == role)
        if sorted({w["device"] for w in part}) != expected:
            raise ValueError(f"role {role} is missing eligible equipment")
        chosen.extend(part)
    return chosen


def check_windows(windows, roles):
    """Refuse anything but fit/dev pure-Normal 30-tick windows. Returns the role list."""
    if not windows:
        raise ValueError("no windows to cache")
    seen, window_roles = set(), []
    for window in windows:
        identifier = window.get("base_window_id")
        role = roles.get(window.get("device"))
        if role not in ALLOWED_ROLES:
            raise ValueError(f"cache holds {ALLOWED_ROLES} only; refused role {role!r} "
                             f"for window {identifier!r}")
        if window.get("hard_valid") is not True or window.get("pure_Normal") is not True:
            raise ValueError(f"window {identifier!r} is not hard-valid pure Normal")
        ids = window.get("raw_base_ids")
        if not isinstance(ids, list) or len(ids) != WINDOW_TICKS:
            raise ValueError(f"window {identifier!r} does not list {WINDOW_TICKS} raw ticks")
        if identifier in seen:
            raise ValueError(f"duplicate window {identifier!r}")
        seen.add(identifier)
        window_roles.append(role)
    return window_roles


def raw_sources(window, data_root, ledger):
    """[[path, sha256], ...] of every raw file behind a window, taken from the ledger."""
    root = Path(data_root).resolve()
    entries = []
    for identifier in window["raw_base_ids"]:
        split, _, name = identifier.partition("/")
        if split != "Training" or not sa._NAME.fullmatch(name):
            raise ValueError(f"invalid raw base ID {identifier!r}")
        for suffix, directory in RAW_FILES:
            path = str(root / split / directory / f"{name}.{suffix}")
            if path not in ledger:
                raise ValueError(f"raw file is absent from the source manifest: {path}")
            entries.append([path, ledger[path]])
    return entries


def window_identity(window, role):
    return {"base_window_id": window["base_window_id"], "device": window["device"],
            "role": role, "session_id": window.get("session_id"),
            "start_index": window.get("start_index"),
            "raw_base_ids": list(window["raw_base_ids"])}


def _npy_header(dtype, shape):
    """The npy header ``open_memmap`` writes, rebuilt independently of the written file."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False,
                 "shape": tuple(shape)})
    return buffer.getvalue()


def _read_header(path):
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported npy version {version} in {path}")
        return shape, fortran, dtype, handle.tell()


def file_digests(path):
    """(file sha256, payload sha256, header offset) from one streaming pass."""
    _, _, _, offset = _read_header(path)
    file_digest, payload_digest = hashlib.sha256(), hashlib.sha256()
    with open(path, "rb") as handle:
        header = handle.read(offset)
        if len(header) != offset:
            raise ValueError(f"{path} ended inside its header")
        file_digest.update(header)
        for block in iter(lambda: handle.read(_BLOCK), b""):
            file_digest.update(block)
            payload_digest.update(block)
    return file_digest.hexdigest(), payload_digest.hexdigest(), offset


def check_array(name, array, dtype):
    """Reader output must already be exactly as declared; nothing is converted here."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"reader returned {type(array).__name__} for {name}, not an ndarray")
    if array.dtype != dtype or array.dtype.str != dtype.str:
        raise ValueError(f"reader {name} dtype {array.dtype.str} differs from declared "
                         f"{dtype.str}; the cache never promotes or demotes")
    if array.shape != WINDOW_SHAPES[name]:
        raise ValueError(f"reader {name} shape {array.shape} differs from "
                         f"{WINDOW_SHAPES[name]}")
    if not array.flags.c_contiguous:
        raise ValueError(f"reader {name} array is not C-contiguous")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"reader {name} array is not finite")


def plain_json(value, what):
    """``value`` exactly as it reads back from JSON, or a refusal.

    The manifest digest is taken before the manifest is written and recomputed by
    ``attach`` after it is read back, so anything whose JSON round trip differs (integer
    keys, tuples, numpy scalars, NaN) would give a cache that builds but never attaches.
    Mixed key types are refused rather than silently merged.
    """
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False))
    except (TypeError, ValueError) as exception:
        raise ValueError(f"{what} is not plain JSON: {exception}") from exception


def _write_json_atomically(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_failure(out, stage, exception, context):
    record = {"stage": stage, "exception": type(exception).__name__,
              "message": str(exception), "timestamp": datetime.now(timezone.utc).isoformat(),
              "partial_files_preserved": sorted(p.name for p in out.iterdir()
                                                if p.name != FAILURE),
              **context}
    (out / FAILURE).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str)
                               + "\n")


class RawNormalCache:
    """A verified on-disk copy of the canonical reader's fit/dev Normal windows."""

    def __init__(self, out, manifest):
        self.out = Path(out)
        self.manifest = manifest
        self.layout = {name: np.dtype(manifest["arrays"][name]["dtype"]) for name in ARRAYS}
        self.windows = manifest["windows"]
        self.count = len(self.windows)
        self._index = {w["base_window_id"]: i for i, w in enumerate(self.windows)}
        self._data = None
        self.verified = False

    # ----------------------------------------------------------------- build
    @classmethod
    def build(cls, out, windows, roles, reader, *, data_root, raw_manifest,
              source_manifest_sha256, dtypes, provenance=None, progress=None,
              free_space_margin=1.05):
        """Read every window once through ``reader`` and store its arrays unchanged.

        ``reader(window) -> (sensor, thermal, labels)`` is the only raw access. Everything
        that can be checked without reading raw data is checked before the reader is
        called and before ``out`` is created.
        """
        layout = declared_layout(dtypes)
        window_roles = check_windows(windows, roles)
        out, root = Path(out).expanduser().resolve(), Path(data_root).expanduser().resolve()
        if out.exists():
            raise ValueError(f"cache output {out} already exists; the cache never overwrites")
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("cache output must be separate from the raw data root")
        if sa.digest_json(raw_manifest) != source_manifest_sha256:
            raise ValueError("raw manifest content differs from its recorded digest")
        ledger = {item["path"]: item["sha256"] for item in raw_manifest}
        sources = [raw_sources(w, root, ledger) for w in windows]
        identities = plain_json([window_identity(w, r) for w, r in zip(windows, window_roles)],
                                "window identity")
        provenance = plain_json(provenance, "provenance")
        count = len(windows)
        payload = required_bytes(count, layout)
        out.parent.mkdir(parents=True, exist_ok=True)
        free, needed = shutil.disk_usage(out.parent).free, int(sum(payload.values()) * free_space_margin)
        if free < needed:
            raise ValueError(f"cache needs about {needed} bytes but {free} are free at {out.parent}")

        out.mkdir(exist_ok=False)
        started = time.perf_counter()
        stage, context = "open", {}
        try:
            maps = {name: np.lib.format.open_memmap(
                        out / _file_name(name), mode="w+", dtype=layout[name],
                        shape=(count, *WINDOW_SHAPES[name]))
                    for name in ARRAYS}
            source = {name: hashlib.sha256() for name in ARRAYS}
            expected_file = {name: hashlib.sha256(_npy_header(layout[name], maps[name].shape))
                             for name in ARRAYS}
            records, read_seconds = [], 0.0
            stage = "read"
            try:
                for index, window in enumerate(windows):
                    context = {"window_index": index,
                               "base_window_id": window["base_window_id"],
                               "windows_written": index}
                    read_started = time.perf_counter()
                    arrays = reader(window)
                    read_seconds += time.perf_counter() - read_started
                    if not isinstance(arrays, tuple) or len(arrays) != len(ARRAYS):
                        raise ValueError("reader must return (sensor, thermal, labels)")
                    arrays = dict(zip(ARRAYS, arrays))
                    digests = {}
                    for name in ARRAYS:
                        check_array(name, arrays[name], layout[name])
                    if not (arrays["labels"] == 0).all():
                        raise ValueError("reader labels are not 30-tick pure Normal")
                    for name in ARRAYS:
                        data = memoryview(arrays[name]).cast("B")
                        source[name].update(data)
                        expected_file[name].update(data)
                        digests[name] = hashlib.sha256(data).hexdigest()
                        maps[name][index] = arrays[name]
                    records.append({**identities[index],
                                    "raw_files_sha256": sa.digest_json(sources[index]),
                                    "array_sha256": digests})
                    if progress and (index + 1) % 100 == 0:
                        progress(index + 1, count)
                context = {"windows_written": count}
                stage = "flush"
                for name in ARRAYS:
                    maps[name].flush()
            finally:
                maps.clear()

            stage = "verify_after_flush"
            verify_started = time.perf_counter()
            arrays_record = {}
            for name in ARRAYS:
                path = out / _file_name(name)
                shape, fortran, dtype, _ = _read_header(path)
                declared_shape = (count, *WINDOW_SHAPES[name])
                if shape != declared_shape or fortran or dtype != layout[name]:
                    raise ValueError(f"{path.name} header disagrees with the declared layout")
                file_sha, payload_sha, offset = file_digests(path)
                source_sha = source[name].hexdigest()
                if payload_sha != source_sha:
                    raise ValueError(f"{path.name} payload read back from disk differs from "
                                     "the arrays the reader returned")
                if file_sha != expected_file[name].hexdigest():
                    raise ValueError(f"{path.name} differs from the expected npy header "
                                     "followed by the reader's bytes")
                size = path.stat().st_size
                if size != offset + payload[name]:
                    raise ValueError(f"{path.name} is {size} bytes, expected "
                                     f"{offset + payload[name]}")
                arrays_record[name] = {"file": path.name, "dtype": layout[name].str,
                                       "shape": list(declared_shape), "window_shape":
                                       list(WINDOW_SHAPES[name]), "c_contiguous": True,
                                       "fortran_order": False, "header_bytes": offset,
                                       "payload_bytes": payload[name], "file_bytes": size,
                                       "file_sha256": file_sha, "payload_sha256": payload_sha,
                                       "source_payload_sha256": source_sha}
            verify_seconds = time.perf_counter() - verify_started

            stage = "manifest"
            manifest = {
                "schema": SCHEMA, "protocol_id": sa.PROTOCOL, "allowed_roles": list(ALLOWED_ROLES),
                "windows_count": count,
                "role_counts": {role: window_roles.count(role) for role in ALLOWED_ROLES},
                "arrays": arrays_record,
                "dtype_policy": "declared_exact_no_promotion_no_demotion_native_byte_order",
                "source_manifest_sha256": source_manifest_sha256,
                "data_root": str(root),
                "window_sources_sha256": sa.digest_json([r["raw_files_sha256"] for r in records]),
                "base_order_sha256": sa.digest_json([r["base_window_id"] for r in records]),
                "role_order_sha256": sa.digest_json(window_roles),
                "window_order_sha256": sa.digest_json(identities),
                "windows": records,
                "raw_access": "injected_reader_only_once_per_window",
                "verified_after_flush": True,
                "verification_scope": ("header_parsed_and_payload_compared_to_reader_bytes_"
                                       "and_file_compared_to_independent_header_plus_bytes"),
                "build_wall_seconds": time.perf_counter() - started,
                "reader_seconds": read_seconds, "verify_seconds": verify_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hostname": platform.node(), "python_version": platform.python_version(),
                "numpy_version": np.__version__, "provenance": provenance,
            }
            manifest["sha256"] = sa.digest_json(manifest)
            _write_json_atomically(out / MANIFEST, manifest)
        except BaseException as exception:
            _write_failure(out, stage, exception, context)
            raise
        cache = cls(out, manifest)
        cache.verified = True
        return cache

    # ----------------------------------------------------------------- reuse
    @classmethod
    def attach(cls, out):
        """Reopen a finished cache. Every array file is re-digested against the manifest."""
        out = Path(out).expanduser().resolve()
        if (out / FAILURE).exists():
            raise ValueError(f"cache at {out} recorded a failure; it is not usable")
        if not (out / MANIFEST).exists():
            raise ValueError(f"cache at {out} has no manifest; it never finished")
        manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        claimed = manifest.pop("sha256", None)
        if claimed is None or sa.digest_json(manifest) != claimed:
            raise ValueError("cache manifest content differs from its recorded digest")
        manifest["sha256"] = claimed
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unknown cache schema")
        cache = cls(out, manifest)
        if len(cache._index) != cache.count or manifest["windows_count"] != cache.count:
            raise ValueError("cache manifest window list is inconsistent")
        if any(w["role"] not in ALLOWED_ROLES for w in cache.windows):
            raise ValueError("cache manifest lists a window outside fit/dev")
        for name in ARRAYS:
            entry = manifest["arrays"][name]
            file_sha, payload_sha, _ = file_digests(out / entry["file"])
            if file_sha != entry["file_sha256"] or payload_sha != entry["payload_sha256"]:
                raise ValueError(f"{entry['file']} content changed since the cache was built")
        cache.verified = True
        return cache

    def open(self):
        if not self.verified:
            raise ValueError("cache was never verified; build it or attach it")
        data = {}
        for name in ARRAYS:
            array = np.load(self.out / _file_name(name), mmap_mode="r", allow_pickle=False)
            if array.shape != (self.count, *WINDOW_SHAPES[name]) or array.dtype != self.layout[name]:
                raise ValueError(f"{name} file disagrees with the cache manifest")
            data[name] = array
        self._data = data
        return self

    def window(self, index):
        """(sensor, thermal, labels) of one window as plain ndarrays, dtype unchanged."""
        if self._data is None:
            raise ValueError("cache must be opened, and so verified, before use")
        return tuple(np.array(self._data[name][index], copy=True) for name in ARRAYS)

    def reader(self, verify_each_read=False):
        """A drop-in for the canonical reader, restricted to the windows this cache holds.

        A window is served only if its whole recorded identity (device, session, start index,
        raw tick IDs) matches the one presented, so
        a calibration or held-out window, or a manifest that has since changed, is refused.
        """
        def read(window):
            index = self._index.get(window.get("base_window_id"))
            if index is None:
                raise ValueError(f"window {window.get('base_window_id')!r} is not in this "
                                 "fit/dev cache")
            recorded = self.windows[index]
            try:
                presented = plain_json(window_identity(window, recorded["role"]),
                                       "window identity")
            except (KeyError, TypeError, ValueError):
                presented = None
            if (presented is None
                    or presented != {key: recorded[key] for key in presented}
                    or window.get("hard_valid") is not True
                    or window.get("pure_Normal") is not True):
                raise ValueError(f"window {recorded['base_window_id']!r} differs from the "
                                 "one cached")
            arrays = self.window(index)
            if verify_each_read:
                for name, array in zip(ARRAYS, arrays):
                    if hashlib.sha256(memoryview(array).cast("B")).hexdigest() != \
                            recorded["array_sha256"][name]:
                        raise ValueError(f"{name} of {recorded['base_window_id']!r} changed "
                                         "since the cache was built")
            return arrays
        return read
