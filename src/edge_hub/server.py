"""HTTP(S) front of the edge hub: device upload API, viewer login and the monitoring page.

Two credential kinds, never mixed:
* devices send ``Authorization: Bearer <token>``; a device token can only upload and send
  heartbeats for its own device id;
* viewers log in with user/password (PBKDF2) and get an HttpOnly session cookie; viewer
  routes are read-only.
"""
from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import ssl
import threading
import time
from urllib.parse import parse_qs, urlsplit

from . import PROTOCOL
from .store import HubError, Store, check_device, utc_now

WEB = Path(__file__).with_name("web")
MAX_JSON = 4 * 1024 * 1024
MAX_LOGIN_JSON = 4096
MAX_HEARTBEAT = 256 * 1024
MAX_DISCARD = 64 * 1024
MAX_CONNECTIONS = 64
_DUMMY = None  # PBKDF2 entry used for unknown users, so timing does not reveal user names
MAX_CHUNK = 32 * 1024 * 1024
SESSION_S = 12 * 3600
PBKDF2_ITERATIONS = 310_000


def hash_password(password: str, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return {"salt": salt.hex(), "hash": digest.hex(), "iterations": PBKDF2_ITERATIONS}


def check_password(entry: dict, password: str) -> bool:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(entry["salt"]),
                                 int(entry["iterations"]))
    return hmac.compare_digest(digest.hex(), entry["hash"])


class Sessions:
    def __init__(self):
        self._items: dict[str, tuple[str, float]] = {}
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def create(self, user: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.time()
            self._items = {k: v for k, v in self._items.items() if v[1] > now}
            self._items[token] = (user, now + SESSION_S)
        return token

    def user(self, token: str | None) -> str | None:
        with self._lock:
            item = self._items.get(token or "")
            return item[0] if item and item[1] > time.time() else None

    def drop(self, token: str | None) -> None:
        with self._lock:
            self._items.pop(token or "", None)

    def reserve(self, addr: str) -> bool:
        """Count this attempt as a failure up front (atomically); False when over the limit."""
        with self._lock:
            now = time.time()
            if len(self._failures) > 10000:      # bound memory under address spraying
                self._failures = {k: v for k, v in self._failures.items() if v and v[-1] > now - 600}
            recent = [t for t in self._failures.get(addr, []) if t > now - 600]
            if len(recent) >= 10:
                self._failures[addr] = recent
                return False
            recent.append(now)
            self._failures[addr] = recent
            return True

    def succeeded(self, addr: str) -> None:
        with self._lock:
            if self._failures.get(addr):
                self._failures[addr].pop()

    def throttled(self, addr: str) -> bool:
        with self._lock:
            return len([t for t in self._failures.get(addr, []) if t > time.time() - 600]) >= 10

    def failed(self, addr: str) -> None:
        with self._lock:
            self._failures.setdefault(addr, []).append(time.time())

    def drop_users(self, valid: set) -> None:
        with self._lock:
            self._items = {k: v for k, v in self._items.items() if v[0] in valid}


class Hub:
    def __init__(self, store: Store, *, secure_cookie: bool, trust_proxy: bool = False):
        self.store, self.sessions = store, Sessions()
        self.secure_cookie, self.trust_proxy = secure_cookie, trust_proxy

    def viewers(self) -> dict:
        path = self.store.root / "config" / "viewers.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}


class _CountingReader:
    def __init__(self, stream):
        self.stream, self.count = stream, 0

    def read(self, n: int) -> bytes:
        data = self.stream.read(n)
        self.count += len(data)
        return data


class Handler(BaseHTTPRequestHandler):
    server_version = "EdgeHub/1"
    sys_version = ""
    hub: Hub  # set on the subclass made by make_server
    protocol_version = "HTTP/1.1"
    timeout = 30  # idle or stalled connections are closed

    # ------------------------------------------------------------- plumbing
    def log_message(self, fmt, *args):  # one line per request, no bodies, no tokens
        self.server.log(f"{self.client_ip()} {self.command} {urlsplit(self.path).path} {fmt % args}")

    def client_ip(self) -> str:
        # behind our own proxy, the right-most entry is the one that proxy added (not forgeable)
        if self.hub.trust_proxy and self.headers.get("X-Forwarded-For"):
            return self.headers["X-Forwarded-For"].split(",")[-1].strip()
        return self.client_address[0]

    def send_json(self, status: int, value, headers: dict | None = None) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        for key, val in (headers or {}).items():
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                         "script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

    def read_json(self, limit: int = MAX_JSON) -> dict:
        length = self._length(limit)
        self.body_consumed = True
        try:
            value = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise HubError(400, "body is not JSON")
        if not isinstance(value, dict):
            raise HubError(400, "body must be a JSON object")
        return value

    def _length(self, limit: int) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise HubError(411, "Content-Length required")
        if not 0 <= length <= limit:
            raise HubError(413, "body too large")
        return length

    def _discard_body(self) -> None:
        if self.body_consumed:
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if 0 < length <= MAX_DISCARD:
            self.rfile.read(length)
        elif length > MAX_DISCARD:
            self.close_connection = True   # do not read (or buffer) a large unauthenticated body

    def cookie(self, name: str) -> str | None:
        for part in self.headers.get("Cookie", "").split(";"):
            key, _, value = part.strip().partition("=")
            if key == name:
                return value
        return None

    # ---------------------------------------------------------------- auth
    def device(self) -> str:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HubError(401, "device token required")
        device = self.hub.store.device_for_token(auth[7:].strip())
        if device is None:
            raise HubError(401, "unknown device token")
        claimed = self.headers.get("X-Device-Id")
        if claimed and claimed != device:
            raise HubError(403, "token does not belong to this device id")
        return device

    def viewer(self) -> str:
        user = self.hub.sessions.user(self.cookie("hub_session"))
        if user is None:
            raise HubError(401, "login required")
        if user not in self.hub.viewers():            # removed viewers lose their sessions at once
            self.hub.sessions.drop_users(set(self.hub.viewers()))
            raise HubError(401, "login required")
        return user

    # -------------------------------------------------------------- dispatch
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        self.body_consumed = False
        try:
            route = ROUTES.get((method, url.path))
            if route is None and method == "GET" and not url.path.startswith("/api/"):
                return self.static(url.path)
            if route is None:
                raise HubError(404, "no such route")
            route(self, query)
        except HubError as error:
            if method in ("POST", "PUT"):
                self._discard_body()
            self.send_json(error.status, {"error": error.message, **error.extra})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:  # report, keep serving
            self.server.log(f"internal error on {url.path}: {error!r}")
            self.close_connection = True
            try:
                self.send_json(500, {"error": "internal error"})
            except OSError:
                pass

    def static(self, path: str) -> None:
        name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        if name not in {"index.html", "app.js", "style.css"}:
            raise HubError(404, "not found")
        body = (WEB / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(name)[0] + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------ device API
    def r_health(self, query):
        self.send_json(200, {"protocol": PROTOCOL, "server_utc": utc_now()})

    def r_whoami(self, query):
        self.send_json(200, {"device": self.device(), "protocol": PROTOCOL})

    def r_begin(self, query):
        device = self.device()
        body = self.read_json()
        self.send_json(200, self.hub.store.begin(device, body.get("unit"), body.get("files"), body.get("meta")))

    def r_put(self, query):
        device = self.device()
        try:
            offset = int(query.get("offset", ""))
        except ValueError:
            raise HubError(400, "offset required")
        length = self._length(MAX_CHUNK)
        body = _CountingReader(self.rfile)
        self.body_consumed = True
        try:
            reply = self.hub.store.put(device, query.get("unit"), query.get("path"), offset, body, length)
        except HubError:
            # drain what the store did not read, so the client still gets this answer
            left = length - body.count
            while left > 0:
                block = self.rfile.read(min(left, 1 << 20))
                if not block:
                    break
                left -= len(block)
            raise
        self.send_json(200, reply)

    def r_commit(self, query):
        device = self.device()
        body = self.read_json()
        receipt = self.hub.store.commit(device, body.get("unit"), body.get("manifest_sha256"))
        self.server.log(f"committed {device} {receipt['unit']} files={receipt['files']} bytes={receipt['bytes']}")
        self.send_json(200, receipt)

    def r_heartbeat(self, query):
        device = self.device()
        body = self.read_json(MAX_HEARTBEAT)
        self.hub.store.heartbeat(device, self.client_ip(), body)
        self.send_json(200, {"ok": True, "server_utc": utc_now()})

    # ------------------------------------------------------------ viewer API
    def r_login(self, query):
        global _DUMMY
        addr = self.client_ip()
        body = self.read_json(MAX_LOGIN_JSON)
        if not self.hub.sessions.reserve(addr):
            raise HubError(429, "too many failed logins; wait 10 minutes")
        user, password = body.get("user"), body.get("password")
        entry = self.hub.viewers().get(user) if isinstance(user, str) else None
        if _DUMMY is None:
            _DUMMY = hash_password(secrets.token_urlsafe(16))
        ok = check_password(entry or _DUMMY, password if isinstance(password, str) else "")
        if not entry or not ok:
            self.server.log(f"login failed from {addr}")
            raise HubError(401, "wrong user or password")
        self.hub.sessions.succeeded(addr)
        token = self.hub.sessions.create(user)
        flags = "HttpOnly; SameSite=Strict; Path=/" + ("; Secure" if self.hub.secure_cookie else "")
        self.server.log(f"login {user} from {addr}")
        self.send_json(200, {"user": user}, {"Set-Cookie": f"hub_session={token}; Max-Age={SESSION_S}; {flags}"})

    def r_logout(self, query):
        self.hub.sessions.drop(self.cookie("hub_session"))
        flags = "HttpOnly; SameSite=Strict; Path=/" + ("; Secure" if self.hub.secure_cookie else "")
        self.send_json(200, {"ok": True}, {"Set-Cookie": f"hub_session=; Max-Age=0; {flags}"})

    def r_me(self, query):
        self.send_json(200, {"user": self.viewer()})

    def r_overview(self, query):
        self.viewer()
        self.send_json(200, self.hub.store.overview())

    def r_live(self, query):
        self.viewer()
        self.send_json(200, self.hub.store.live_view(check_device(query.get("device", ""))))

    def r_telemetry(self, query):
        self.viewer()
        try:
            since = min(48 * 3600.0, max(60.0, float(query.get("since_s", 3600))))
        except ValueError:
            raise HubError(400, "since_s must be a number")
        self.send_json(200, self.hub.store.telemetry(query.get("device", ""), since))

    def r_units(self, query):
        self.viewer()
        self.send_json(200, {"units": self.hub.store.units(query.get("device") or None)})

    def r_unit(self, query):
        self.viewer()
        self.send_json(200, self.hub.store.unit_detail(query.get("device", ""), query.get("unit", "")))

    def r_unit_scalars(self, query):
        self.viewer()
        columns = [c for c in query.get("columns", "").split(",") if c]
        self.send_json(200, self.hub.store.unit_scalars(query.get("device", ""), query.get("unit", ""), columns))

    def r_unit_thermal(self, query):
        self.viewer()
        try:
            index = int(query.get("index", "0"))
        except ValueError:
            raise HubError(400, "index must be an integer")
        self.send_json(200, self.hub.store.unit_thermal(query.get("device", ""), query.get("unit", ""),
                                                        query.get("chunk", ""), index))


ROUTES = {
    ("GET", "/api/v1/health"): Handler.r_health,
    ("GET", "/api/v1/device/whoami"): Handler.r_whoami,
    ("POST", "/api/v1/device/units/begin"): Handler.r_begin,
    ("PUT", "/api/v1/device/units/file"): Handler.r_put,
    ("POST", "/api/v1/device/units/commit"): Handler.r_commit,
    ("POST", "/api/v1/device/heartbeat"): Handler.r_heartbeat,
    ("POST", "/api/v1/login"): Handler.r_login,
    ("POST", "/api/v1/logout"): Handler.r_logout,
    ("GET", "/api/v1/me"): Handler.r_me,
    ("GET", "/api/v1/overview"): Handler.r_overview,
    ("GET", "/api/v1/live"): Handler.r_live,
    ("GET", "/api/v1/telemetry"): Handler.r_telemetry,
    ("GET", "/api/v1/units"): Handler.r_units,
    ("GET", "/api/v1/unit"): Handler.r_unit,
    ("GET", "/api/v1/unit/scalars"): Handler.r_unit_scalars,
    ("GET", "/api/v1/unit/thermal"): Handler.r_unit_thermal,
}


class HubServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, handler, log):
        self.log = log
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):   # full: refuse instead of growing threads
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        import sys
        error = sys.exc_info()[1]
        self.log(f"connection from {client_address[0]} dropped: {error!r}")


def make_server(store: Store, host: str, port: int, *, tls_cert: str | None = None,
                tls_key: str | None = None, trust_proxy: bool = False, log=print) -> HubServer:
    hub = Hub(store, secure_cookie=bool(tls_cert) or trust_proxy, trust_proxy=trust_proxy)
    handler = type("BoundHandler", (Handler,), {"hub": hub})
    server = HubServer((host, port), handler, log)
    if tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(tls_cert, tls_key)
        # handshake lazily in the request thread, so one slow client never blocks accept()
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    return server
