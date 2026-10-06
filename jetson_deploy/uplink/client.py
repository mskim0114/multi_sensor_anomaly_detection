"""Minimal HTTPS client for the hub (http.client + ssl), with server failover.

``server_urls`` is tried in order and the last working one is kept, so one config can list
e.g. a site-LAN address and an overlay-VPN / public name. TLS verifies the server against a
pinned CA file (from the hub's gen-cert) or, when none is given, the system trust store.
Plain http is accepted only for localhost (tests, or a local TLS-terminating tunnel).
"""
from __future__ import annotations

import http.client
import json
import ssl
from urllib.parse import urlencode, urlsplit


class HubUnavailable(Exception):
    """No server URL could be reached (network down, DNS, TLS, timeout)."""


class HubRejected(Exception):
    """The hub answered with an error status."""

    def __init__(self, status: int, body: dict):
        super().__init__(f"HTTP {status}: {body.get('error', body)}")
        self.status, self.body = status, body


class HubClient:
    def __init__(self, server_urls: list[str], token: str, device_id: str, *, ca_file: str | None = None,
                 timeout: float = 60.0):
        if not server_urls:
            raise ValueError("at least one server URL is needed")
        for url in server_urls:
            parts = urlsplit(url)
            if parts.scheme == "http" and parts.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError(f"plain http is only allowed to localhost: {url}")
            if parts.scheme not in ("http", "https"):
                raise ValueError(f"unsupported URL {url}")
        self.urls, self.token, self.device_id, self.timeout = list(server_urls), token, device_id, timeout
        self.context = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
        self.context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.active = 0

    def _connection(self, url: str):
        parts = urlsplit(url)
        if parts.scheme == "https":
            return http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=self.timeout,
                                               context=self.context)
        return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=self.timeout)

    def request(self, method: str, path: str, *, query: dict | None = None, json_body=None,
                body: bytes | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.token}", "X-Device-Id": self.device_id,
                   "User-Agent": "factory-uplink/1"}
        if json_body is not None:
            body = json.dumps(json_body, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        elif body is not None:
            headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(len(body or b""))
        target = path + ("?" + urlencode(query) if query else "")
        errors = []
        order = [self.active] + [i for i in range(len(self.urls)) if i != self.active]
        for index in order:
            base = self.urls[index]
            prefix = urlsplit(base).path.rstrip("/")
            connection = self._connection(base)
            try:
                connection.request(method, prefix + target, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
            except (OSError, http.client.HTTPException, ssl.SSLError) as error:
                errors.append(f"{base}: {error!r}")
                continue
            finally:
                connection.close()
            self.active = index
            try:
                value = json.loads(raw or b"{}")
            except ValueError:
                value = {"error": raw[:200].decode("utf-8", "replace")}
            if response.status >= 400:
                raise HubRejected(response.status, value if isinstance(value, dict) else {"error": value})
            return value
        raise HubUnavailable("; ".join(errors))

    @property
    def active_url(self) -> str:
        return self.urls[self.active]
