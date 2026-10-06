"""Edge hub: receives completed raw units from field Jetsons and serves a monitoring page.

Server side of the edge → server data path (docs/EDGE_HUB.md). Standard library only, so it
runs from any Python >= 3.10 without touching the training venv.

Raw units arrive byte-identical, are verified by SHA-256 before they become visible, and are
then read-only. Nothing here converts, relabels or deletes raw data.
"""

PROTOCOL = "factory-edge-hub-v1"
