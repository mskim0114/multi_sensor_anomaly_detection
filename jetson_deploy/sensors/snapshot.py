"""Snapshot schema for the multi-sensor collector.

One master tick produces one snapshot. The master tick is 1.0 s because the
trained model consumes 30 ticks at 1 Hz per window; that time base is fixed and
this layer must not change it.

A snapshot carries, per sensor, an observation of the shape:

    {"status": ..., "fresh": bool, "age_ms": float|None, "values": {...}}

`fresh` and `age_ms` exist because sensors do not share one native rate. A
sensor slower than the master tick (SCD30 at 0.5 Hz) legitimately repeats its
value across consecutive snapshots. That repetition is marked `fresh=false`
with a growing `age_ms`, never by comparing values - identical readings from a
stable environment are normal and must not be mistaken for staleness.
"""

from __future__ import annotations

STATUS_OK = "ok"
STATUS_WARMING_UP = "warming_up"
STATUS_STALE = "stale"
STATUS_ERROR = "error"
STATUS_DISABLED = "disabled"

SCHEMA_VERSION = 1

# Flat per-tick scalar view. The full nested snapshot lives in snapshots.jsonl;
# this CSV is the convenient form for quick inspection and plotting.
SCALAR_FIELDS = [
    "sequence",
    "timestamp_utc",
    "tick_jitter_ms",
    # NTC - model channel
    "ntc_status",
    "ntc_temperature_c",
    # CT - model channels. Only CT1 exists physically.
    "ct1_status",
    "ct1_vrms",
    "ct1_current_a_nominal",
    "ct1_sample_count",
    "ct1_actual_sps",
    "ct1_clipping",
    "ct2_status",
    "ct3_status",
    "ct4_status",
    # SPS30 - model channels PM1.0 / PM2.5 / PM10 (PM4.0 collected, not used)
    "sps30_status",
    "sps30_fresh",
    "sps30_age_ms",
    "pm1_0_ug_m3",
    "pm2_5_ug_m3",
    "pm4_0_ug_m3",
    "pm10_ug_m3",
    # context sensors
    "sgp30_status",
    "sgp30_warmup",
    "sgp30_interval_ms",
    "eco2_ppm",
    "tvoc_ppb",
    "scd30_status",
    "scd30_fresh",
    "scd30_age_ms",
    "co2_ppm",
    "scd30_temperature_c",
    "scd30_humidity_pct",
    "bme680_status",
    "bme680_temperature_c",
    "bme680_humidity_pct",
    "bme680_pressure_hpa",
    "bme680_gas_ohm",
    # thermal - model input
    "flir_status",
    "flir_age_ms",
    "flir_min_c",
    "flir_max_c",
    "flir_mean_c",
    "thermal_chunk",
    "thermal_index",
]


def observation(
    status: str,
    values: dict | None = None,
    *,
    fresh: bool | None = None,
    age_ms: float | None = None,
    error: str | None = None,
    consecutive_errors: int = 0,
    extra: dict | None = None,
) -> dict:
    """Build one sensor observation."""
    obs: dict = {"status": status, "values": values or {}}
    if fresh is not None:
        obs["fresh"] = fresh
    if age_ms is not None:
        obs["age_ms"] = round(age_ms, 3)
    if error is not None:
        obs["error"] = error
    if consecutive_errors:
        obs["consecutive_errors"] = consecutive_errors
    if extra:
        obs.update(extra)
    return obs


def _get(snapshot: dict, sensor: str, key: str, default=None):
    obs = snapshot["sensors"].get(sensor) or {}
    if key in obs:
        return obs[key]
    return (obs.get("values") or {}).get(key, default)


def scalar_row(snapshot: dict) -> dict:
    """Project a snapshot onto the flat SCALAR_FIELDS view."""
    g = _get
    return {
        "sequence": snapshot["sequence"],
        "timestamp_utc": snapshot["timestamp_utc"],
        "tick_jitter_ms": snapshot["tick_jitter_ms"],

        "ntc_status": g(snapshot, "ntc", "status"),
        "ntc_temperature_c": g(snapshot, "ntc", "temperature_c"),

        "ct1_status": g(snapshot, "ct1", "status"),
        "ct1_vrms": g(snapshot, "ct1", "vrms"),
        "ct1_current_a_nominal": g(snapshot, "ct1", "current_a_nominal"),
        "ct1_sample_count": g(snapshot, "ct1", "sample_count"),
        "ct1_actual_sps": g(snapshot, "ct1", "actual_sample_rate"),
        "ct1_clipping": g(snapshot, "ct1", "clipping"),
        "ct2_status": g(snapshot, "ct2", "status"),
        "ct3_status": g(snapshot, "ct3", "status"),
        "ct4_status": g(snapshot, "ct4", "status"),

        "sps30_status": g(snapshot, "sps30", "status"),
        "sps30_fresh": g(snapshot, "sps30", "fresh"),
        "sps30_age_ms": g(snapshot, "sps30", "age_ms"),
        "pm1_0_ug_m3": g(snapshot, "sps30", "pm1_0_ug_m3"),
        "pm2_5_ug_m3": g(snapshot, "sps30", "pm2_5_ug_m3"),
        "pm4_0_ug_m3": g(snapshot, "sps30", "pm4_0_ug_m3"),
        "pm10_ug_m3": g(snapshot, "sps30", "pm10_ug_m3"),

        "sgp30_status": g(snapshot, "sgp30", "status"),
        "sgp30_warmup": g(snapshot, "sgp30", "warmup"),
        "sgp30_interval_ms": g(snapshot, "sgp30", "interval_ms"),
        "eco2_ppm": g(snapshot, "sgp30", "eco2_ppm"),
        "tvoc_ppb": g(snapshot, "sgp30", "tvoc_ppb"),

        "scd30_status": g(snapshot, "scd30", "status"),
        "scd30_fresh": g(snapshot, "scd30", "fresh"),
        "scd30_age_ms": g(snapshot, "scd30", "age_ms"),
        "co2_ppm": g(snapshot, "scd30", "co2_ppm"),
        "scd30_temperature_c": g(snapshot, "scd30", "temperature_c"),
        "scd30_humidity_pct": g(snapshot, "scd30", "humidity_pct"),

        "bme680_status": g(snapshot, "bme680", "status"),
        "bme680_temperature_c": g(snapshot, "bme680", "temperature_c"),
        "bme680_humidity_pct": g(snapshot, "bme680", "humidity_pct"),
        "bme680_pressure_hpa": g(snapshot, "bme680", "pressure_hpa"),
        "bme680_gas_ohm": g(snapshot, "bme680", "gas_ohm"),

        "flir_status": g(snapshot, "flir", "status"),
        "flir_age_ms": g(snapshot, "flir", "age_ms"),
        "flir_min_c": g(snapshot, "flir", "min_c"),
        "flir_max_c": g(snapshot, "flir", "max_c"),
        "flir_mean_c": g(snapshot, "flir", "mean_c"),
        "thermal_chunk": g(snapshot, "flir", "thermal_chunk"),
        "thermal_index": g(snapshot, "flir", "thermal_index"),
    }
