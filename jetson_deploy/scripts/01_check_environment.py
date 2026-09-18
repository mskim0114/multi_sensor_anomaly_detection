#!/usr/bin/env python3
"""Compatibility entry point for the canonical JETSON-RUNTIME checker.

Re-exec through run_python.sh to select factory_runtime and disable user-site,
including when invoked with a system Python from an old command or guide.
"""

import os
from pathlib import Path
import sys


def main():
    deploy = Path(__file__).resolve().parents[1]
    launcher = deploy / "run_python.sh"
    checker = deploy / "check_environment.py"
    os.execv(str(launcher), [str(launcher), str(checker), *sys.argv[1:]])


if __name__ == "__main__":
    main()
