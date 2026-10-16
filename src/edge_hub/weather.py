"""Where each device is installed (entered on the web) and the outdoor weather there.

Nothing here touches the device or its data. A person records the site of a device (label,
latitude, longitude, and from when it applies); the hub then keeps hourly outdoor temperature,
relative humidity and surface pressure for that place from Open-Meteo, a free global weather
API that needs no key. Because only coordinates and UTC time are needed, the same works after
the device moves (Germany testbed, docs/EDGE_HUB.md §6) and for data that reaches KETI later by
another route: enter the new site with its start time and the hours are fetched for it.

* Site history is append-only. A new entry closes the previous one at its own ``valid_from``;
  a wrong entry is corrected by adding another one, and both stay on record with who and when.
* Coordinates are kept to 2 decimals (about 1 km). The weather grid is coarser than that and
  an exact building position is not needed.
* Hours are fetched once and kept with their source. Recent hours (``RECENT_DAYS``) come from the
  forecast API's past data, older hours from the ERA5 archive, which runs about 5 days behind.
  These are model values for a grid cell, not a station measurement next to the sensor.
* Outdoor values are context for the dataset builder and the web page, never a model input or a
  label, and never used to correct a sensor.

Standard library only (the hub runs on the system python3).
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from .store import HubError, Store, _atomic_write, _json_bytes, _read_json, check_device, utc_now

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ("temperature_2m", "relative_humidity_2m", "surface_pressure")
RECENT_DAYS = 60            # inside this window use the forecast API (it keeps ~92 past days)
MAX_DAYS_PER_REQUEST = 31
MAX_BACKFILL_DAYS = 400
STATUS_NAME = "weather_status.json"
SOURCE_FORECAST = "open-meteo:forecast(best_match)"
SOURCE_ARCHIVE = "open-meteo:archive(era5)"
USER_AGENT = "factory-safety-edge-hub (KETI research; hourly site weather)"
HOUR = 3600


def _parse_utc(text) -> float:
    if not isinstance(text, str) or not text.strip():
        raise HubError(400, "valid_from_utc must be an ISO time")
    try:
        value = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HubError(400, "valid_from_utc must be an ISO time, e.g. 2026-10-15T00:00:00Z")
    if value.tzinfo is None:
        raise HubError(400, "valid_from_utc needs a time zone (use Z for UTC)")
    return value.timestamp()


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _day(epoch: float) -> dt.date:
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date()


def http_get_json(url: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(8 * 1024 * 1024))


class Weather:
    def __init__(self, store: Store, fetch=None):
        self.store = store
        self.fetch = fetch or http_get_json
        with store._db_lock:
            store.db.executescript("""
                CREATE TABLE IF NOT EXISTS device_sites(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, device TEXT, valid_from_epoch REAL,
                    valid_from_utc TEXT, label TEXT, latitude REAL, longitude REAL,
                    note TEXT, set_by TEXT, set_utc TEXT);
                CREATE INDEX IF NOT EXISTS device_sites_device ON device_sites(device, valid_from_epoch);
                CREATE TABLE IF NOT EXISTS weather_hourly(
                    latitude REAL, longitude REAL, hour_epoch INTEGER, temperature_c REAL,
                    humidity_pct REAL, pressure_hpa REAL, source TEXT, fetched_utc TEXT,
                    PRIMARY KEY(latitude, longitude, hour_epoch));
            """)
            store.db.commit()

    # ------------------------------------------------------------ site history
    def set_site(self, device: str, label, latitude, longitude, valid_from_utc, user: str,
                 note="", now: float | None = None) -> dict:
        now = time.time() if now is None else now
        check_device(device)
        if device not in self.store.devices():
            raise HubError(404, "unknown device")
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise HubError(400, "label: 1..80 characters")
        try:
            lat, lon = float(latitude), float(longitude)
        except (TypeError, ValueError):
            raise HubError(400, "latitude and longitude must be numbers")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or lat != lat or lon != lon:
            raise HubError(400, "latitude -90..90, longitude -180..180")
        start = now if valid_from_utc in (None, "") else _parse_utc(valid_from_utc)
        if start > now + 86400:
            raise HubError(400, "valid_from_utc is more than a day in the future")
        if start < now - MAX_BACKFILL_DAYS * 86400:
            raise HubError(400, f"valid_from_utc is more than {MAX_BACKFILL_DAYS} days in the past")
        if not isinstance(note, str) or len(note) > 300:
            raise HubError(400, "note: up to 300 characters")
        record = (device, start, _iso(start), label.strip(), round(lat, 2), round(lon, 2), note.strip(),
                  user, utc_now())
        with self.store._db_lock:
            cursor = self.store.db.execute(
                "INSERT INTO device_sites(device,valid_from_epoch,valid_from_utc,label,latitude,longitude,note,"
                "set_by,set_utc) VALUES(?,?,?,?,?,?,?,?,?)", record)
            self.store.db.commit()
            new_id = cursor.lastrowid
        return next(s for s in self.history(device) if s["id"] == new_id)

    def history(self, device: str) -> list[dict]:
        """All entries in force order. An entry ends where the next one (by valid_from) starts;
        for equal valid_from the later entry wins (it was a correction)."""
        check_device(device)
        with self.store._db_lock:
            rows = self.store.db.execute(
                "SELECT id,valid_from_epoch,valid_from_utc,label,latitude,longitude,note,set_by,set_utc "
                "FROM device_sites WHERE device=? ORDER BY valid_from_epoch, id", (device,)).fetchall()
        out = []
        for i, r in enumerate(rows):
            later = [x for x in rows[i + 1:] if x[1] > r[1]]
            superseded = any(x[1] == r[1] for x in rows[i + 1:])
            out.append({"id": r[0], "valid_from_epoch": r[1], "valid_from_utc": r[2], "label": r[3],
                        "latitude": r[4], "longitude": r[5], "note": r[6], "set_by": r[7], "set_utc": r[8],
                        "valid_until_epoch": later[0][1] if later else None,
                        "valid_until_utc": later[0][2] if later else None,
                        "superseded": superseded})
        return out

    def periods(self, device: str) -> list[dict]:
        return [s for s in self.history(device) if not s["superseded"]]

    def site_at(self, device: str, epoch: float) -> dict | None:
        for site in self.periods(device):
            if site["valid_from_epoch"] <= epoch and (site["valid_until_epoch"] is None
                                                      or epoch < site["valid_until_epoch"]):
                return site
        return None

    # ---------------------------------------------------------------- reading
    def _rows(self, lat: float, lon: float, start: float, end: float) -> list[dict]:
        with self.store._db_lock:
            rows = self.store.db.execute(
                "SELECT hour_epoch,temperature_c,humidity_pct,pressure_hpa,source FROM weather_hourly "
                "WHERE latitude=? AND longitude=? AND hour_epoch>=? AND hour_epoch<=? ORDER BY hour_epoch",
                (lat, lon, int(start), int(end))).fetchall()
        return [{"t": r[0], "utc": _iso(r[0]), "temperature_c": r[1], "humidity_pct": r[2],
                 "pressure_hpa": r[3], "source": r[4]} for r in rows]

    def outdoor(self, device: str, start: float, end: float) -> list[dict]:
        """Hourly rows for [start, end], each taken at the site in force at that hour."""
        out = []
        for site in self.periods(device):
            lo = max(start, site["valid_from_epoch"])
            hi = min(end, site["valid_until_epoch"] - 1 if site["valid_until_epoch"] else end)
            if lo > hi:
                continue
            out += [dict(r, site_id=site["id"]) for r in self._rows(site["latitude"], site["longitude"],
                                                                    lo // HOUR * HOUR, hi)]
        return sorted(out, key=lambda r: r["t"])

    def latest(self, device: str, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        rows = self.outdoor(device, now - 6 * HOUR, now)
        return rows[-1] if rows else None

    def for_span(self, device: str, start: float, end: float) -> dict:
        """Outdoor context for a unit: the hours touching [start, end] and their mean."""
        rows = self.outdoor(device, start // HOUR * HOUR, end)
        def mean(key):
            values = [r[key] for r in rows if isinstance(r[key], (int, float))]
            return round(sum(values) / len(values), 2) if values else None
        site = self.site_at(device, start)
        return {"site": site and {k: site[k] for k in ("id", "label", "latitude", "longitude")},
                "hours": rows, "temperature_c": mean("temperature_c"), "humidity_pct": mean("humidity_pct"),
                "pressure_hpa": mean("pressure_hpa")}

    def summary(self, device: str, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        site = self.site_at(device, now)
        return {"site": site, "outdoor": self.latest(device, now)}

    def status(self) -> dict | None:
        path = self.store.root / STATUS_NAME
        return _read_json(path) if path.is_file() else None

    # --------------------------------------------------------------- fetching
    def _missing_days(self, lat: float, lon: float, start: float, end: float) -> list[dt.date]:
        """Days in [start, end] with at least one hour not yet stored."""
        first, last = int(start // HOUR * HOUR), int(end // HOUR * HOUR)
        if last < first:
            return []
        with self.store._db_lock:
            have = {r[0] for r in self.store.db.execute(
                "SELECT hour_epoch FROM weather_hourly WHERE latitude=? AND longitude=? AND hour_epoch>=? "
                "AND hour_epoch<=?", (lat, lon, first, last))}
        return sorted({_day(h) for h in range(first, last + 1, HOUR) if h not in have})

    @staticmethod
    def _runs(days: list[dt.date]) -> list[tuple[dt.date, dt.date]]:
        runs = []
        for day in days:
            if runs and day == runs[-1][1] + dt.timedelta(days=1) and \
                    (day - runs[-1][0]).days < MAX_DAYS_PER_REQUEST:
                runs[-1] = (runs[-1][0], day)
            else:
                runs.append((day, day))
        return runs

    def _fetch_range(self, lat: float, lon: float, first: dt.date, last: dt.date, now: float) -> int:
        recent = _day(now) - dt.timedelta(days=RECENT_DAYS)
        url, source = (FORECAST_URL, SOURCE_FORECAST) if last >= recent else (ARCHIVE_URL, SOURCE_ARCHIVE)
        query = {"latitude": f"{lat:.2f}", "longitude": f"{lon:.2f}", "hourly": ",".join(HOURLY),
                 "start_date": first.isoformat(), "end_date": last.isoformat(),
                 "timezone": "GMT", "timeformat": "unixtime"}
        body = self.fetch(url + "?" + urllib.parse.urlencode(query))
        hourly = body.get("hourly") if isinstance(body, dict) else None
        if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
            raise ValueError(f"unexpected reply: {str(body)[:200]}")
        series = [hourly.get(k) or [None] * len(hourly["time"]) for k in HOURLY]
        fetched, rows = utc_now(), []
        for i, hour in enumerate(hourly["time"]):
            values = [s[i] if i < len(s) else None for s in series]
            # only completed hours, and only hours with a value (the archive lags a few days)
            if not isinstance(hour, int) or hour + HOUR > now or values[0] is None:
                continue
            rows.append((lat, lon, hour, *values, source, fetched))
        with self.store._db_lock:
            self.store.db.executemany("INSERT OR REPLACE INTO weather_hourly VALUES(?,?,?,?,?,?,?,?)", rows)
            self.store.db.commit()
        return len(rows)

    def update(self, now: float | None = None) -> dict:
        """Fetch every missing completed hour of every site period. Safe to run at any interval."""
        now = time.time() if now is None else now
        started = time.monotonic()
        status = {"started_utc": utc_now(), "ok": True, "requests": 0, "hours_stored": 0, "errors": []}
        wanted: dict[tuple[float, float], list[tuple[float, float]]] = {}
        for device in sorted(self.store.devices()):
            for site in self.periods(device):
                end = min(now - HOUR, site["valid_until_epoch"] or now)
                start = max(site["valid_from_epoch"], now - MAX_BACKFILL_DAYS * 86400)
                if start <= end:
                    wanted.setdefault((site["latitude"], site["longitude"]), []).append((start, end))
        for (lat, lon), spans in wanted.items():
            days = sorted({d for start, end in spans for d in self._missing_days(lat, lon, start, end)})
            for first, last in self._runs(days):
                status["requests"] += 1
                try:
                    status["hours_stored"] += self._fetch_range(lat, lon, first, last, now)
                except (urllib.error.URLError, OSError, ValueError, TypeError) as error:
                    status["ok"] = False
                    status["errors"].append(f"{lat},{lon} {first}..{last}: {type(error).__name__}: {error}"[:300])
        status.update(finished_utc=utc_now(), seconds=round(time.monotonic() - started, 1),
                      sites=len(wanted), errors=status["errors"][:20])
        _atomic_write(self.store.root / STATUS_NAME, _json_bytes(status))
        return status
