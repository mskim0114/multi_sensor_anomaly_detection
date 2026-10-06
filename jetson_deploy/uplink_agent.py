"""Entry point: ./jetson_deploy/run_python.sh jetson_deploy/uplink_agent.py run|once|check|status|pending"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uplink.agent import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
