"""ZED 2i side recorder: follows the running sensor collection and stores 1 Hz stereo frames
(left RGB JPEG + depth) next to it, aligned to the collector's ticks.

It is a separate process with its own Python (pyzed needs numpy >= 2; the collector runtime is
pinned to numpy 1.26), so it can never slow down or break the 1 Hz collector. It only writes
into ``<run>/zed/`` of a run whose collector is alive, and it never touches collector files.
``control`` is standard library only (used by the uplink agent); ``recorder`` needs numpy;
``camera`` needs pyzed and Pillow.
"""

SCHEMA = "zed-v1"
