"""Resolve project resources independently of the working directory.

Relative paths are anchored to this checkout; absolute paths (for example,
external datasets supplied in a DataConfig YAML) remain absolute.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(relative: str | Path) -> str:
    return str((PROJECT_ROOT / Path(relative).expanduser()).resolve())
