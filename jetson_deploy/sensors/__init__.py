"""Sensor acquisition layer for the Jetson runtime.

This package is a raw data acquisition layer. It does NOT feed the inference
pipeline and does NOT define the model input vector. The trained model input
stays [NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4] as declared in
src/data/config.py; BME680, SGP30 and SCD30 are collection/context sensors.
"""

from .snapshot import (  # noqa: F401
    SCHEMA_VERSION,
    SCALAR_FIELDS,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_STALE,
    STATUS_WARMING_UP,
    observation,
    scalar_row,
)
