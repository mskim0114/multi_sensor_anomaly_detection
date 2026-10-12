"""Local upload ledger (SQLite) kept outside the dataset: hash cache, unit state, receipts."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3
import threading


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS file_hash(
                unit TEXT, path TEXT, size INTEGER, mtime_ns INTEGER, sha256 TEXT,
                PRIMARY KEY(unit, path));
            CREATE TABLE IF NOT EXISTS units(
                unit TEXT PRIMARY KEY, state TEXT, manifest_sha256 TEXT, files INTEGER,
                bytes INTEGER, attempts INTEGER DEFAULT 0, last_error TEXT, receipt TEXT,
                updated_utc TEXT);
            CREATE TABLE IF NOT EXISTS commands(
                id TEXT PRIMARY KEY, action TEXT, received_utc TEXT, result TEXT, reported INTEGER DEFAULT 0);
        """)
        self.db.commit()

    def cached_hash(self, unit: str, path: str, size: int, mtime_ns: int) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT sha256 FROM file_hash WHERE unit=? AND path=? AND size=? "
                                  "AND mtime_ns=?", (unit, path, size, mtime_ns)).fetchone()
        return row[0] if row else None

    def store_hash(self, unit: str, path: str, size: int, mtime_ns: int, sha256: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO file_hash VALUES(?,?,?,?,?)",
                            (unit, path, size, mtime_ns, sha256))
            self.db.commit()

    def forget_hashes(self, unit: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM file_hash WHERE unit=?", (unit,))
            self.db.commit()

    def unit(self, unit: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT unit,state,manifest_sha256,files,bytes,attempts,last_error,"
                                  "receipt,updated_utc FROM units WHERE unit=?", (unit,)).fetchone()
        if not row:
            return None
        keys = ("unit", "state", "manifest_sha256", "files", "bytes", "attempts", "last_error",
                "receipt", "updated_utc")
        record = dict(zip(keys, row))
        record["receipt"] = json.loads(record["receipt"]) if record["receipt"] else None
        return record

    def set_state(self, unit: str, state: str, *, manifest_sha256=None, files=None, total=None,
                  error=None, receipt=None, attempt=False) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO units(unit, state, updated_utc) VALUES(?,?,?)",
                            (unit, state, utc_now()))
            self.db.execute(
                "UPDATE units SET state=?, manifest_sha256=COALESCE(?, manifest_sha256), "
                "files=COALESCE(?, files), bytes=COALESCE(?, bytes), last_error=?, "
                "receipt=COALESCE(?, receipt), attempts=attempts+?, updated_utc=? WHERE unit=?",
                (state, manifest_sha256, files, total, error,
                 json.dumps(receipt) if receipt is not None else None, 1 if attempt else 0, utc_now(), unit))
            self.db.commit()

    def command_seen(self, command_id: str, action: str) -> bool:
        """Record a command id; False if it was already recorded (never execute twice)."""
        with self.lock:
            cursor = self.db.execute("INSERT OR IGNORE INTO commands(id, action, received_utc) VALUES(?,?,?)",
                                     (command_id, action, utc_now()))
            # bounded: keep the newest 1000 reported rows (ids stay unique per hub, which never reuses them)
            self.db.execute("DELETE FROM commands WHERE reported=1 AND rowid NOT IN "
                            "(SELECT rowid FROM commands ORDER BY rowid DESC LIMIT 1000)")
            self.db.commit()
            return cursor.rowcount == 1

    def command_done(self, command_id: str, result: dict) -> None:
        with self.lock:
            self.db.execute("UPDATE commands SET result=? WHERE id=?", (json.dumps(result), command_id))
            self.db.commit()

    def unreported_commands(self) -> list[tuple[str, dict]]:
        with self.lock:
            rows = self.db.execute("SELECT id, result FROM commands WHERE result IS NOT NULL AND reported=0").fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def command_reported(self, command_id: str) -> None:
        with self.lock:
            self.db.execute("UPDATE commands SET reported=1 WHERE id=?", (command_id,))
            self.db.commit()

    def summary(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT state, COUNT(*), COALESCE(SUM(bytes),0) FROM units "
                                   "GROUP BY state").fetchall()
            last = self.db.execute("SELECT unit, updated_utc FROM units WHERE state='committed' "
                                   "ORDER BY updated_utc DESC LIMIT 1").fetchone()
            error = self.db.execute("SELECT unit, last_error, updated_utc FROM units WHERE last_error "
                                    "IS NOT NULL ORDER BY updated_utc DESC LIMIT 1").fetchone()
        return {"by_state": {s: {"units": n, "bytes": b} for s, n, b in rows},
                "last_committed": {"unit": last[0], "utc": last[1]} if last else None,
                "last_error": {"unit": error[0], "error": error[1], "utc": error[2]} if error else None}
