#!/usr/bin/env python3
"""ZED 2i side recorder and its on/off switch.

Usage (the recorder needs the ZED venv; on/off/status run with any Python):
    PYTHONNOUSERSITE=1 ~/venvs/zed_runtime/bin/python -I jetson_deploy/scripts/14_zed_recorder.py run
    python3 jetson_deploy/scripts/14_zed_recorder.py on|off|status

``run`` is what the factory-zed user service executes. While the switch is on and a sensor
collection (free run or trial) is running, it stores one left JPEG (plus a depth map if depth_mode is not NONE) per collector tick
under ``<run>/zed/``; otherwise the camera stays closed. See jetson_deploy/zedcap/recorder.py for
the file layout. The collector itself is not changed and does not depend on this process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zedcap import control  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "on", "off", "status"])
    parser.add_argument("--config", type=Path, default=control.CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.command in ("on", "off"):
        config = control.set_enabled(args.command == "on", by="local", path=args.config)
        print(json.dumps({"enabled": config["enabled"], "config": str(args.config),
                          "config_error": config.get("config_error")}, ensure_ascii=False))
        return 0 if not config.get("config_error") else 1
    if args.command == "status":
        config = control.load_config(args.config)
        status = control.read_status()
        if status:
            status.pop("preview", None)
        print(json.dumps({"enabled": config["enabled"], "config_error": config.get("config_error"),
                          "recorder": status or "not running"}, indent=1, ensure_ascii=False))
        return 0
    from zedcap.camera import ZedCamera
    from zedcap.recorder import encode_jpeg_pillow
    from zedcap.service import Recorder
    recorder = Recorder(camera_factory=ZedCamera, encoder_factory=encode_jpeg_pillow,
                        config_path=args.config, identity=control.process_identity(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: recorder.stop.set())
    signal.signal(signal.SIGINT, lambda *_: recorder.stop.set())
    recorder.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
