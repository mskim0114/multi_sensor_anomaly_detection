"""Uplink: ships completed raw units from this Jetson to the edge hub, and sends heartbeats.

Outbound HTTPS only, so it works from any site network that allows outbound traffic. It only
reads acquisition output; it never writes into, converts or deletes a raw unit. Upload state
lives in its own ledger outside the dataset. Standard library only (no new runtime packages).
"""

PROTOCOL = "factory-edge-hub-v1"
AGENT_VERSION = "1"
