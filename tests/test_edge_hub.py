"""Edge hub + uplink agent end to end over real TLS on localhost (standard library only)."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jetson_deploy"))

from src.edge_hub import __main__ as hub_cli  # noqa: E402
from src.edge_hub.server import hash_password, make_server  # noqa: E402
from src.edge_hub.store import HubError, Store, manifest_text  # noqa: E402
from uplink.agent import Agent  # noqa: E402
from uplink.client import HubClient, HubRejected, HubUnavailable  # noqa: E402
from uplink.units import discover  # noqa: E402

OLD = time.time() - 3600


def write(path: Path, data: bytes, mtime: float = OLD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))


def npy(array_bytes: bytes, descr: str, shape: tuple) -> bytes:
    header = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode()
    pad = 64 - (10 + len(header) + 1) % 64
    header += b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + array_bytes


def make_trial(root: Path, group: str, name: str, status: str = "completed", frames: int = 3,
               mtime: float = OLD) -> Path:
    trial = root / group / name
    write(trial / "experiment.json", json.dumps({"trial_id": name, "scenario_id": group, "status": status,
                                                 "test_mode": True}).encode(), mtime)
    write(trial / "metadata.json", json.dumps({"run_id": name}).encode(), mtime)
    rows = ["sequence,timestamp_utc,ntc_temperature_c,flir_max_c"]
    rows += [f"{i},2026-10-06T00:00:{i:02d}+00:00,{25 + i * 0.1:.2f},{40 + i}" for i in range(frames)]
    write(trial / "scalars.csv", ("\n".join(rows) + "\n").encode(), mtime)
    write(trial / "snapshots.jsonl", b"".join(json.dumps({"sequence": i}).encode() + b"\n" for i in range(frames)), mtime)
    pixels = b"".join(((29315 + i * 100 + p % 7).to_bytes(2, "little")) for i in range(frames) for p in range(120 * 160))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("frames.npy", npy(pixels, "<u2", (frames, 120, 160)))
        archive.writestr("sequences.npy", npy(b"".join(i.to_bytes(8, "little") for i in range(frames)), "<i8", (frames,)))
    write(trial / "thermal_000000.npz", buffer.getvalue(), mtime)
    write(trial / "big.bin", os.urandom(300_000), mtime)       # forces several chunks
    return trial


class HubFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="edge_hub_test_"))
        cls.data_root = cls.tmp / "hub"
        hub_cli.main(["gen-cert", "--data-root", str(cls.data_root), "--san", "IP:127.0.0.1", "--san", "DNS:localhost"])
        cls.store = Store(cls.data_root)
        cls.token_file = cls.tmp / "device.token"
        hub_cli.main(["add-device", "--data-root", str(cls.data_root), "--device-id", "jetson-test-01",
                      "--token-out", str(cls.token_file)])
        cls.token = cls.token_file.read_text().strip()
        other = cls.tmp / "other.token"
        hub_cli.main(["add-device", "--data-root", str(cls.data_root), "--device-id", "jetson-test-02",
                      "--token-out", str(other)])
        cls.other_token = other.read_text().strip()
        viewers = cls.data_root / "config" / "viewers.json"
        viewers.write_text(json.dumps({"viewer": hash_password("correct horse battery"),
                                       "boss": dict(hash_password("operator horse battery"), role="operator")}))
        tls = cls.data_root / "tls"
        cls.server = make_server(cls.store, "127.0.0.1", 0, tls_cert=str(tls / "server.pem"),
                                 tls_key=str(tls / "server.key"), log=lambda m: None)
        cls.port = cls.server.server_address[1]
        cls.url = f"https://127.0.0.1:{cls.port}"
        cls.ca = str(tls / "ca.pem")
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for path in sorted(cls.tmp.rglob("*"), reverse=True):   # raw is read-only by design
            try:
                if not path.is_symlink():
                    os.chmod(path, 0o700 if path.is_dir() else 0o600)
            except FileNotFoundError:   # e.g. a SQLite -wal file removed as its connection closes
                pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.edge = Path(tempfile.mkdtemp(dir=self.tmp, prefix="edge_"))
        self.dataset, self.runs = self.edge / "dataset", self.edge / "runs"
        self.dataset.mkdir()
        self.runs.mkdir()

    def agent(self, *, urls=None, token=None, device="jetson-test-01", chunk=65536, state=None) -> Agent:
        config = {"device_id": device, "server_urls": urls or [self.url], "token": token or self.token,
                  "ca_file": self.ca, "roots": {"dataset": self.dataset, "runs": self.runs},
                  "include_test_units": True, "settle_s": 60, "scan_interval_s": 1,
                  "heartbeat_interval_s": 10, "thermal_every_s": 0, "chunk_bytes": chunk,
                  "max_upload_bytes_per_s": 0, "hardware_lock": str(self.edge / "test-hardware.lock")}
        client = HubClient(config["server_urls"], config["token"], device, ca_file=self.ca, timeout=10)
        return Agent(config, state_dir=state or (self.edge / "state"), client=client)

    def raw(self, unit: str, device="jetson-test-01") -> Path:
        return self.data_root / "raw" / device / unit

    def https(self, method, path, body=None, headers=None):
        context = ssl.create_default_context(cafile=self.ca)
        conn = http.client.HTTPSConnection("127.0.0.1", self.port, context=context, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, dict(response.getheaders()), (json.loads(data) if data[:1] in (b"{", b"[") else data)

    def login(self, user="viewer", password="correct horse battery") -> str:
        status, headers, _ = self.https("POST", "/api/v1/login", json.dumps(
            {"user": user, "password": password}), {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        return headers["Set-Cookie"].split(";")[0]


class TransferTests(HubFixture):
    def test_unit_arrives_byte_identical_read_only_and_checkable_with_sha256sum(self):
        trial = make_trial(self.dataset, "_smoke", "normal_A")
        agent = self.agent()
        result = agent.scan_once()
        self.assertEqual((result["committed"], result["errors"]), (1, 0))
        raw = self.raw("dataset/_smoke/normal_A")
        for path in trial.rglob("*"):
            if path.is_file():
                rel = path.relative_to(trial)
                self.assertEqual((raw / rel).read_bytes(), path.read_bytes(), rel)
                self.assertEqual((raw / rel).stat().st_mode & 0o222, 0, f"{rel} must be read-only")
        self.assertEqual(sorted(p.name for p in raw.iterdir()), sorted(p.name for p in trial.iterdir()))
        manifest = self.data_root / "manifests" / "jetson-test-01" / "dataset/_smoke/normal_A.sha256"
        check = subprocess.run(["sha256sum", "-c", str(manifest)], cwd=self.data_root / "raw" / "jetson-test-01",
                               capture_output=True, text=True, env=dict(os.environ, LC_ALL="C"))
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
        record = agent.ledger.unit("dataset/_smoke/normal_A")
        self.assertEqual(record["state"], "committed")
        self.assertEqual(record["receipt"]["manifest_sha256"],
                         hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertFalse(any(self.store.root.joinpath("staging").rglob("*.part")))
        self.assertEqual(self.store.verify("jetson-test-01")[0]["ok"], True)
        # the local unit was only read
        self.assertTrue(all(p.stat().st_mtime == OLD for p in trial.rglob("*") if p.is_file()))

    def test_second_scan_uploads_nothing_and_a_new_ledger_is_answered_from_the_receipt(self):
        make_trial(self.dataset, "_smoke", "normal_B")
        self.assertEqual(self.agent().scan_once()["committed"], 1)
        again = self.agent()
        with unittest.mock.patch.object(again, "_send_file", side_effect=AssertionError("re-sent")):
            self.assertEqual(again.scan_once()["committed"], 0)
            fresh = self.agent(state=self.edge / "fresh_state")      # ledger lost: begin returns receipt
            self.assertEqual(fresh.scan_once()["committed"], 1)
        self.assertEqual(fresh.ledger.unit("dataset/_smoke/normal_B")["state"], "committed")

    def test_interrupted_upload_resumes_from_the_bytes_the_hub_has(self):
        trial = make_trial(self.dataset, "_smoke", "normal_C")
        agent = self.agent(chunk=50_000)
        files = agent.manifest(next(u for u in discover({"dataset": self.dataset})[0]))
        wire = [{k: f[k] for k in ("path", "size", "sha256")} for f in files]
        self.store.begin("jetson-test-01", "dataset/_smoke/normal_C", wire)
        half = (trial / "big.bin").read_bytes()[:123_457]
        self.store.put("jetson-test-01", "dataset/_smoke/normal_C", "big.bin", 0, io.BytesIO(half), len(half))
        sent = []
        original = agent.client.request

        def spy(method, path, **kw):
            if method == "PUT":
                sent.append((kw["query"]["path"], kw["query"]["offset"], len(kw["body"])))
            return original(method, path, **kw)
        agent.client.request = spy
        self.assertEqual(agent.scan_once()["committed"], 1)
        big = [s for s in sent if s[0] == "big.bin"]
        self.assertEqual(big[0][1], 123_457, "must resume, not restart")
        self.assertEqual(sum(s[2] for s in big), 300_000 - 123_457)
        self.assertEqual(self.raw("dataset/_smoke/normal_C").joinpath("big.bin").read_bytes(),
                         (trial / "big.bin").read_bytes())

    def test_offset_disagreement_is_corrected_by_the_hub(self):
        make_trial(self.dataset, "_smoke", "normal_D")
        agent = self.agent()
        unit = discover({"dataset": self.dataset})[0][0]
        files = agent.manifest(unit)
        self.store.begin("jetson-test-01", unit.key, [{k: f[k] for k in ("path", "size", "sha256")} for f in files])
        with self.assertRaises(HubRejected) as caught:
            agent.client.request("PUT", "/api/v1/device/units/file",
                                 query={"unit": unit.key, "path": "big.bin", "offset": 999}, body=b"x")
        self.assertEqual((caught.exception.status, caught.exception.body["received"]), (409, 0))
        self.assertEqual(agent.scan_once()["committed"], 1)

    def test_corrupted_bytes_are_rejected_and_never_committed(self):
        make_trial(self.dataset, "_smoke", "normal_E")
        agent = self.agent()
        unit = discover({"dataset": self.dataset})[0][0]
        files = agent.manifest(unit)
        info = (unit.path / "big.bin").stat()
        agent.ledger.store_hash(unit.key, "big.bin", info.st_size, info.st_mtime_ns, "0" * 64)  # stale cache
        result = agent.scan_once()
        self.assertEqual((result["committed"], result["errors"]), (0, 1))
        self.assertFalse(self.raw(unit.key).exists())
        self.assertIn("sha256 mismatch", agent.ledger.unit(unit.key)["last_error"])
        self.assertEqual(agent.scan_once()["committed"], 1, "hash cache is dropped, next scan re-hashes")

    def test_local_change_after_commit_is_reported_and_never_overwrites_the_hub_copy(self):
        trial = make_trial(self.dataset, "_smoke", "normal_F")
        agent = self.agent()
        agent.scan_once()
        hub_copy = self.raw("dataset/_smoke/normal_F").joinpath("scalars.csv").read_bytes()
        write(trial / "scalars.csv", b"tampered\n", OLD - 10)
        result = agent.scan_once()
        self.assertEqual(result["committed"], 0)
        self.assertEqual(agent.ledger.unit("dataset/_smoke/normal_F")["state"], "local_changed_after_commit")
        self.assertEqual(self.raw("dataset/_smoke/normal_F").joinpath("scalars.csv").read_bytes(), hub_copy)
        # and a direct attempt with the new manifest is refused by the hub
        with self.assertRaises(HubError) as caught:
            self.store.begin("jetson-test-01", "dataset/_smoke/normal_F",
                             [{"path": "x", "size": 1, "sha256": "a" * 64}])
        self.assertEqual(caught.exception.status, 409)

    def test_unfinished_settling_and_odd_units_are_not_shipped(self):
        make_trial(self.dataset, "normal", "trial_001", status="running")
        make_trial(self.dataset, "normal", "trial_002", mtime=time.time())       # just written
        odd = make_trial(self.dataset, "normal", "trial_003")
        os.symlink("/etc/hostname", odd / "link")
        make_trial(self.dataset, "normal", "trial_004", status="aborted")        # aborted is shipped
        run = self.runs / "20261006T000000.000000Z"
        write(run / "metadata.json", json.dumps({"status": "running"}).encode())
        lock = self.edge / "busy.lock"   # an acquisition is alive: a "running" trial may be live
        stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        lock.write_text(json.dumps({"pid": os.getpid(), "start_ticks": stat[19],
                                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}))
        units, skipped = discover({"dataset": self.dataset, "runs": self.runs}, settle_s=60, lock_path=lock)
        self.assertEqual([u.key for u in units], ["dataset/normal/trial_004"])
        reasons = {s["unit"]: s["reason"] for s in skipped}
        self.assertIn("may still be running", reasons["dataset/normal/trial_001"])
        self.assertIn("settling", reasons["dataset/normal/trial_002"])
        self.assertIn("regular file", reasons["dataset/normal/trial_003"])
        self.assertIn("no control.json identity", reasons["runs/20261006T000000.000000Z"])
        units, _ = discover({"dataset": self.dataset}, include_test=False, settle_s=60, lock_path=lock)
        self.assertNotIn("_smoke", {u.key.split("/")[1] for u in units})

    def test_free_runs_are_shipped_under_runs(self):
        run = self.runs / "20261006T010203.000000Z"
        write(run / "metadata.json", json.dumps({"status": "stopped"}).encode())
        write(run / "scalars.csv", b"sequence\n0\n")
        self.assertEqual(self.agent().scan_once()["committed"], 1)
        self.assertTrue(self.raw("runs/20261006T010203.000000Z").joinpath("scalars.csv").is_file())

    def test_failover_to_the_next_url_and_pending_when_none_answers(self):
        make_trial(self.dataset, "_smoke", "normal_G")
        dead = "https://127.0.0.1:9"   # discard port, nothing listens
        agent = self.agent(urls=[dead, self.url])
        self.assertEqual(agent.scan_once()["committed"], 1)
        self.assertEqual(agent.client.active_url, self.url)
        make_trial(self.dataset, "_smoke", "normal_H")
        offline = self.agent(urls=[dead])
        result = offline.scan_once()
        self.assertEqual(result["committed"], 0)
        self.assertIs(offline.connected, False)
        self.assertEqual(offline.ledger.unit("dataset/_smoke/normal_H")["state"], "pending")

    def test_tls_is_verified_against_the_pinned_ca(self):
        client = HubClient([self.url], self.token, "jetson-test-01", timeout=5)  # system CAs only
        with self.assertRaises(HubUnavailable) as caught:
            client.request("GET", "/api/v1/device/whoami")
        self.assertIn("CERTIFICATE_VERIFY_FAILED", str(caught.exception))
        with self.assertRaises(ValueError):
            HubClient(["http://10.0.0.1:8080"], self.token, "jetson-test-01")


class SecurityTests(HubFixture):
    def test_device_tokens_are_required_and_bound_to_their_device(self):
        status, _, body = self.https("GET", "/api/v1/device/whoami")
        self.assertEqual(status, 401)
        status, _, body = self.https("GET", "/api/v1/device/whoami", headers={"Authorization": "Bearer nope"})
        self.assertEqual(status, 401)
        status, _, body = self.https("GET", "/api/v1/device/whoami", headers={
            "Authorization": f"Bearer {self.token}", "X-Device-Id": "jetson-test-02"})
        self.assertEqual(status, 403)
        status, _, body = self.https("GET", "/api/v1/device/whoami", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual((status, body["device"]), (200, "jetson-test-01"))
        devices = json.loads((self.data_root / "config" / "devices.json").read_text())
        self.assertNotIn(self.token, json.dumps(devices), "only token hashes are stored")
        self.assertEqual((self.data_root / "config" / "devices.json").stat().st_mode & 0o077, 0)

    def test_one_device_cannot_touch_another_devices_units(self):
        make_trial(self.dataset, "_smoke", "normal_S")
        self.agent().scan_once()
        other = self.agent(token=self.other_token, device="jetson-test-02", state=self.edge / "s2")
        other.scan_once()
        # the same unit name lands in the other device's own tree, never in jetson-test-01's
        self.assertTrue(self.raw("dataset/_smoke/normal_S", "jetson-test-02").is_dir())
        self.assertEqual(len(self.store.units("jetson-test-01")), len(
            [u for u in self.store.units() if u["device"] == "jetson-test-01"]))

    def test_path_traversal_and_bad_names_are_refused(self):
        auth = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        bad_units = ["dataset/../etc", "dataset/_smoke/..", "runs/a/b", "other/x", "dataset/a/.hidden", "runs/"]
        for unit in bad_units:
            status, _, body = self.https("POST", "/api/v1/device/units/begin", json.dumps(
                {"unit": unit, "files": [{"path": "a", "size": 1, "sha256": "a" * 64}]}), auth)
            self.assertEqual(status, 400, unit)
        for path in ["../x", "/abs", "a//b", "a/../../b", "x.part", ".env", "a\\b"]:
            status, _, body = self.https("POST", "/api/v1/device/units/begin", json.dumps(
                {"unit": "runs/r1", "files": [{"path": path, "size": 1, "sha256": "a" * 64}]}), auth)
            self.assertEqual(status, 400, path)
        status, _, _ = self.https("GET", "/../../etc/passwd")
        self.assertEqual(status, 404)

    def test_chunks_cannot_exceed_the_declared_size(self):
        self.store.begin("jetson-test-01", "runs/r2", [{"path": "a", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}])
        status, _, body = self.https("PUT", "/api/v1/device/units/file?unit=runs/r2&path=a&offset=0", b"abcd",
                                     {"Authorization": f"Bearer {self.token}"})
        self.assertEqual(status, 400)
        status, _, body = self.https("PUT", "/api/v1/device/units/file?unit=runs/r2&path=a&offset=0", b"abc",
                                     {"Authorization": f"Bearer {self.token}"})
        self.assertEqual((status, body["complete"]), (200, True))

    def test_viewer_routes_need_a_login_and_device_tokens_do_not_count(self):
        for path in ["/api/v1/overview", "/api/v1/units", "/api/v1/live?device=jetson-test-01"]:
            status, _, _ = self.https("GET", path, headers={"Authorization": f"Bearer {self.token}"})
            self.assertEqual(status, 401, path)
        status, _, _ = self.https("POST", "/api/v1/login", json.dumps({"user": "viewer", "password": "wrong"}),
                                  {"Content-Type": "application/json"})
        self.assertEqual(status, 401)
        cookie = self.login()
        status, headers, body = self.https("GET", "/api/v1/overview", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        status, _, _ = self.https("POST", "/api/v1/device/heartbeat", b"{}", {"Cookie": cookie})
        self.assertEqual(status, 401, "a viewer session cannot act as a device")

    def test_repeated_failed_logins_are_throttled(self):
        hub = self.server.RequestHandlerClass.hub
        for _ in range(10):
            hub.sessions.failed("10.9.9.9")
        self.assertTrue(hub.sessions.throttled("10.9.9.9"))
        self.assertFalse(hub.sessions.throttled("127.0.0.2"))


class MonitoringTests(HubFixture):
    def test_heartbeat_live_view_overview_and_unit_views(self):
        make_trial(self.dataset, "_smoke", "normal_M", frames=4)
        agent = self.agent()
        agent.scan_once()
        frame = (29315 + 500).to_bytes(2, "little") * (120 * 160)
        import base64
        live = {"run_id": "r", "age_s": 0.5, "snapshot": {"sequence": 7, "timestamp_utc": "2026-10-06T00:00:07Z",
                "sensors": {"ntc": {"status": "ok", "values": {"temperature_c": 31.5}},
                            "flir": {"status": "ok", "values": {"max_c": 45.0}}}},
                "thermal": {"data": base64.b64encode(frame).decode(), "snapshot_sequence": 7}}
        with unittest.mock.patch("uplink.agent._latest_preview", return_value=live):
            body = agent.heartbeat_body()
        self.assertIsNotNone(body["live"]["thermal"])
        agent.client.request("POST", "/api/v1/device/heartbeat", json_body=body)
        cookie = self.login()
        _, _, overview = self.https("GET", "/api/v1/overview", headers={"Cookie": cookie})
        mine = next(d for d in overview["devices"] if d["device"] == "jetson-test-01")
        self.assertTrue(mine["online"])
        self.assertEqual(mine["latest_values"]["ntc.temperature_c"], 31.5)
        self.assertGreaterEqual(mine["units"]["dataset"]["count"], 1)
        _, _, tele = self.https("GET", "/api/v1/telemetry?device=jetson-test-01&since_s=600", headers={"Cookie": cookie})
        self.assertEqual(tele["points"][-1]["flir.max_c"], 45.0)
        _, _, livev = self.https("GET", "/api/v1/live?device=jetson-test-01", headers={"Cookie": cookie})
        self.assertEqual(livev["live"]["thermal"]["snapshot_sequence"], 7)
        unit = "dataset/_smoke/normal_M"
        _, _, detail = self.https("GET", f"/api/v1/unit?device=jetson-test-01&unit={unit}", headers={"Cookie": cookie})
        self.assertEqual(detail["summary"]["status"], "completed")
        self.assertEqual(detail["thermal_chunks"], ["thermal_000000.npz"])
        _, _, scal = self.https("GET", f"/api/v1/unit/scalars?device=jetson-test-01&unit={unit}&columns=ntc_temperature_c",
                                headers={"Cookie": cookie})
        self.assertEqual(scal["series"]["ntc_temperature_c"], [25.0, 25.1, 25.2, 25.3])
        _, _, th = self.https("GET", f"/api/v1/unit/thermal?device=jetson-test-01&unit={unit}&chunk=thermal_000000.npz&index=2",
                              headers={"Cookie": cookie})
        raw = base64.b64decode(th["data"])
        self.assertEqual((th["sequence"], th["frames_in_chunk"], len(raw)), (2, 4, 120 * 160 * 2))
        self.assertEqual(int.from_bytes(raw[:2], "little"), 29315 + 200)
        status, _, _ = self.https("GET", f"/api/v1/unit/thermal?device=jetson-test-01&unit={unit}&chunk=../x&index=0",
                                  headers={"Cookie": cookie})
        self.assertEqual(status, 404)
        status, headers, page = self.https("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Factory Edge Hub", page)

    def test_preview_reader_drops_the_collector_token_and_monotonic_fields(self):
        from uplink import agent as agent_module
        state = Path(tempfile.mkdtemp(dir=self.tmp))
        session = state / ".cache/factory_safety/acquisition/session.json"
        session.parent.mkdir(parents=True)
        run = state / "run"
        run.mkdir()
        (run / "control.json").write_text(json.dumps({"status": "running", "token": "SECRET",
                                                      "updated_utc": "u"}))
        stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        me = {"pid": os.getpid(), "start_ticks": stat[19],
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        session.write_text(json.dumps({"token": "SECRET", "identity": me, "run_dir": str(run)}))
        with unittest.mock.patch.dict(os.environ, {"HOME": str(state)}):
            collector = agent_module._collector_state()
            self.assertEqual(collector, {"status": "running", "alive": True, "run_dir": str(run), "updated_utc": "u"})
            self.assertNotIn("SECRET", json.dumps(collector))
            session.write_text(json.dumps({"token": "SECRET", "identity": dict(me, start_ticks="1"),
                                           "run_dir": str(run)}))
            self.assertEqual(agent_module._collector_state()["status"], "failed", "a reused pid is not alive")
            session.unlink()
            self.assertEqual(agent_module._collector_state(), {"status": "stopped", "alive": False})



class HardeningTests(HubFixture):
    """Regression tests for the 2026-10-06 security review (H1-H4, M1-M3, L1-L2)."""

    def test_h1_a_fully_written_part_left_by_a_crash_is_finished_at_begin(self):
        trial = make_trial(self.dataset, "_smoke", "normal_R1")
        agent = self.agent()
        unit = discover({"dataset": self.dataset})[0][0]
        files = agent.manifest(unit)
        wire = [{k: f[k] for k in ("path", "size", "sha256")} for f in files]
        state = self.store.begin("jetson-test-01", unit.key, wire)
        stage = self.store._staging("jetson-test-01", unit.key) / "data"
        (stage / "big.bin.part").write_bytes((trial / "big.bin").read_bytes())   # crash before verify/rename
        (stage / "scalars.csv.part").write_bytes(b"x" * (trial / "scalars.csv").stat().st_size)  # full but wrong
        again = self.store.begin("jetson-test-01", unit.key, wire)
        self.assertEqual(again["received"]["big.bin"], 300_000)
        self.assertEqual(again["received"]["scalars.csv"], 0, "a wrong full part is dropped, not trusted")
        self.assertEqual(agent.scan_once()["committed"], 1)
        self.assertEqual(self.raw(unit.key).joinpath("scalars.csv").read_bytes(), (trial / "scalars.csv").read_bytes())

    def test_h2_a_zip_bomb_thermal_chunk_is_refused_without_inflating_it(self):
        trial = make_trial(self.dataset, "_smoke", "normal_R2")
        buffer = io.BytesIO()
        frames = 5000   # far more than any real chunk
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("frames.npy", npy(bytes(frames * 120 * 160 * 2), "<u2", (frames, 120, 160)))
            archive.writestr("sequences.npy", npy(bytes(frames * 8), "<i8", (frames,)))
        write(trial / "thermal_000001.npz", buffer.getvalue())
        self.agent().scan_once()
        with self.assertRaises(HubError) as caught:
            self.store.unit_thermal("jetson-test-01", "dataset/_smoke/normal_R2", "thermal_000001.npz", 0)
        self.assertEqual(caught.exception.status, 422)
        frame = self.store.unit_thermal("jetson-test-01", "dataset/_smoke/normal_R2", "thermal_000000.npz", 1)
        self.assertEqual(frame["sequence"], 1)

    def test_h3_malformed_documents_do_not_break_commit_index_or_detail(self):
        trial = make_trial(self.dataset, "_smoke", "normal_R3")
        write(trial / "experiment.json", json.dumps({"status": "completed"}).encode())
        write(trial / "timing_report.json", b"[1]")
        write(trial / "metadata.json", b"\xff\xfe not json")
        self.assertEqual(self.agent().scan_once()["committed"], 1)
        listed = [u for u in self.store.units("jetson-test-01") if u["unit"] == "dataset/_smoke/normal_R3"]
        self.assertEqual(listed[0]["summary"]["status"], "completed")
        self.store.unit_detail("jetson-test-01", "dataset/_smoke/normal_R3")
        self.assertGreaterEqual(self.store.reindex(), 1)

    def test_h4_hostile_heartbeats_cannot_break_the_overview(self):
        hostile = {"interval_s": "abc", "temperatures_c": {"x": "hot"}, "collector": {"run_dir": 5},
                   "git": {"commit": 7}, "deep": [[[[[[[[1]]]]]]]],
                   "live": {"run_id": "r", "snapshot": {"sequence": 1, "sensors": {"a": [1], "b": {"values": {"v": "1", "w": 2}}}},
                            "thermal": {"data": "A" * 10}}}
        self.store.heartbeat("jetson-test-01", "1.2.3.4", hostile)
        self.store.heartbeat("jetson-test-01", "1.2.3.4", hostile)   # same run/sequence again
        overview = self.store.overview()
        mine = next(d for d in overview["devices"] if d["device"] == "jetson-test-01")
        self.assertEqual(mine["status"]["interval_s"], 10)
        self.assertEqual(mine["latest_values"], {"b.w": 2})
        self.assertIsNone(self.store.live_view("jetson-test-01")["live"]["thermal"])
        json.dumps(overview, allow_nan=False)
        auth = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        status, _, _ = self.https("POST", "/api/v1/device/heartbeat", b"{" + b" " * (300 * 1024) + b"}", auth)
        self.assertEqual(status, 413)

    def test_m1_large_unauthenticated_bodies_are_not_read(self):
        status, _, _ = self.https("POST", "/api/v1/login", b"x" * 10_000, {"Content-Type": "application/json"})
        self.assertEqual(status, 413)
        status, headers, _ = self.https("PUT", "/api/v1/device/units/file?unit=runs/x&path=a&offset=0", b"x" * 200_000)
        self.assertEqual(status, 401)

    def test_m2_login_limit_is_atomic_and_unknown_users_cost_the_same(self):
        hub = self.server.RequestHandlerClass.hub
        addr = "10.7.7.7"
        allowed = [hub.sessions.reserve(addr) for _ in range(12)]
        self.assertEqual(allowed.count(True), 10)
        hub.sessions.succeeded("10.7.7.8")      # harmless on an unknown address
        start = time.perf_counter()
        self.https("POST", "/api/v1/login", json.dumps({"user": "nobody", "password": "x"}), {"Content-Type": "application/json"})
        unknown = time.perf_counter() - start
        self.assertGreater(unknown, 0.05, "PBKDF2 runs for unknown users too")

    def test_l1_revoked_devices_and_removed_viewers_lose_access(self):
        cookie = self.login()
        viewers = self.data_root / "config" / "viewers.json"
        saved = viewers.read_text()
        viewers.write_text("{}")
        try:
            status, _, _ = self.https("GET", "/api/v1/me", headers={"Cookie": cookie})
            self.assertEqual(status, 401)
        finally:
            viewers.write_text(saved)
        token_file = self.tmp / "revoke.token"
        hub_cli.main(["add-device", "--data-root", str(self.data_root), "--device-id", "jetson-test-09",
                      "--token-out", str(token_file)])
        token = token_file.read_text().strip()
        auth = {"Authorization": f"Bearer {token}"}
        self.assertEqual(self.https("GET", "/api/v1/device/whoami", headers=auth)[0], 200)
        self.assertEqual(hub_cli.main(["revoke-device", "--data-root", str(self.data_root), "--device-id", "jetson-test-09"]), 0)
        self.assertEqual(self.https("GET", "/api/v1/device/whoami", headers=auth)[0], 401)

    def test_symlinked_unit_directories_are_not_shipped(self):
        real = make_trial(self.edge / "elsewhere", "g", "outside")
        (self.dataset / "_smoke").mkdir(exist_ok=True)
        os.symlink(real, self.dataset / "_smoke" / "linked")
        os.symlink(real.parent, self.dataset / "linkedgroup")
        units, _ = discover({"dataset": self.dataset}, settle_s=60)
        self.assertEqual([u.key for u in units], [])



class RemoteControlTests(HubFixture):
    """Operator start/stop: queued on the hub, delivered once via heartbeat, device opt-in."""

    def setUp(self):
        super().setUp()
        self.online(True)

    def online(self, remote_control):
        self.store.heartbeat("jetson-test-01", "127.0.0.1", {"remote_control": remote_control, "interval_s": 10})

    def test_commands_are_refused_for_offline_or_non_opted_in_devices(self):
        self.finish_open_commands()
        self.online(False)
        with self.assertRaises(HubError) as caught:
            self.store.request_command("jetson-test-01", "start", "boss")
        self.assertIn("disabled", caught.exception.message)
        self.online(True)
        with self.store._db_lock:
            self.store.db.execute("UPDATE devices SET last_seen_epoch=0 WHERE device='jetson-test-01'")
            self.store.db.commit()
        with self.assertRaises(HubError) as caught:
            self.store.request_command("jetson-test-01", "start", "boss")
        self.assertIn("offline", caught.exception.message)
        self.online(True)

    def test_add_viewer_rerun_keeps_the_operator_role(self):
        viewers = self.data_root / "config" / "viewers.json"
        saved = viewers.read_text()
        try:
            with unittest.mock.patch("getpass.getpass", return_value="a new long password"):
                self.assertEqual(hub_cli.main(["add-viewer", "--data-root", str(self.data_root), "--user", "boss"]), 0)
            self.assertEqual(json.loads(viewers.read_text())["boss"]["role"], "operator")
        finally:
            viewers.write_text(saved)

    def control(self, cookie, action="start", device="jetson-test-01", headers=None):
        base = {"Cookie": cookie, "Content-Type": "application/json", "X-Hub-Request": "1"}
        base.update(headers or {})
        return self.https("POST", "/api/v1/control", json.dumps({"device": device, "action": action}), base)

    def finish_open_commands(self):
        for command in self.store.take_commands("jetson-test-01"):
            self.store.command_result("jetson-test-01", command["id"], True, {})

    def test_only_operators_can_queue_and_only_from_the_same_origin(self):
        self.finish_open_commands()
        viewer = self.login()
        self.assertEqual(self.control(viewer)[0], 403)
        boss = self.login("boss", "operator horse battery")
        _, _, me = self.https("GET", "/api/v1/me", headers={"Cookie": boss})
        self.assertEqual(me["role"], "operator")
        self.assertEqual(self.control(boss, headers={"X-Hub-Request": "0"})[0], 403)
        self.assertEqual(self.control(boss, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.control(boss, action="reboot")[0], 400)
        self.assertEqual(self.control(boss, device="jetson-nope-99")[0], 404)
        status, _, record = self.control(boss, headers={"Origin": f"https://127.0.0.1:{self.port}"})
        self.assertEqual((status, record["state"], record["requested_by"]), (200, "pending", "boss"))
        self.assertEqual(self.control(boss, action="stop")[0], 409, "one open command per device")
        self.finish_open_commands()

    def test_a_command_is_delivered_once_and_its_result_recorded(self):
        self.finish_open_commands()
        record = self.store.request_command("jetson-test-01", "start", "boss")
        auth = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        beat = json.dumps({"remote_control": True}).encode()
        _, _, first = self.https("POST", "/api/v1/device/heartbeat", beat, auth)
        _, _, second = self.https("POST", "/api/v1/device/heartbeat", beat, auth)
        self.assertEqual([c["id"] for c in first["commands"]], [record["id"]])
        self.assertEqual(second["commands"], [])
        other = {"Authorization": f"Bearer {self.other_token}", "Content-Type": "application/json"}
        status, _, _ = self.https("POST", "/api/v1/device/command_result",
                                  json.dumps({"id": record["id"], "ok": True}), other)
        self.assertEqual(status, 404, "a device cannot report another device's command")
        status, _, reply = self.https("POST", "/api/v1/device/command_result",
                                      json.dumps({"id": record["id"], "ok": True, "result": {"collector": {"status": "running"}}}), auth)
        self.assertEqual(reply["state"], "done")
        _, _, again = self.https("POST", "/api/v1/device/command_result", json.dumps({"id": record["id"], "ok": False}), auth)
        self.assertTrue(again["duplicate"])
        cookie = self.login()
        _, _, listed = self.https("GET", "/api/v1/commands?device=jetson-test-01", headers={"Cookie": cookie})
        self.assertEqual(listed["commands"][0]["state"], "done")
        self.assertEqual(listed["commands"][0]["result"]["collector"]["status"], "running")

    def test_undelivered_commands_expire_and_are_never_run_later(self):
        self.finish_open_commands()
        record = self.store.request_command("jetson-test-01", "stop", "boss")
        with self.store._db_lock:
            self.store.db.execute("UPDATE commands SET expires_epoch=0 WHERE id=?", (record["id"],))
            self.store.db.commit()
        self.assertEqual(self.store.take_commands("jetson-test-01"), [])
        self.assertEqual(self.store.commands("jetson-test-01")[0]["state"], "expired")

    def test_revoked_devices_cannot_be_commanded(self):
        token_file = self.tmp / "rc.token"
        hub_cli.main(["add-device", "--data-root", str(self.data_root), "--device-id", "jetson-test-07",
                      "--token-out", str(token_file)])
        hub_cli.main(["revoke-device", "--data-root", str(self.data_root), "--device-id", "jetson-test-07"])
        with self.assertRaises(HubError) as caught:
            self.store.request_command("jetson-test-07", "start", "boss")
        self.assertEqual(caught.exception.status, 404)

    def test_device_runs_only_when_it_opted_in_and_never_twice(self):
        agent = self.agent()
        calls = []
        agent.collector_command = lambda action, command_id=None: calls.append(action) or {"ok": True, "collector": {"status": "running"}}
        agent.handle_commands([{"id": "a" * 24, "action": "start"}])
        self.assertEqual(calls, [], "allow_remote_control defaults to off")
        self.assertIn("disabled", agent.ledger.unreported_commands()[0][1]["error"])
        agent.config["allow_remote_control"] = True
        for _ in range(2):   # the same id twice (redelivery) runs once
            agent.handle_commands([{"id": "b" * 24, "action": "start"}, {"id": "c" * 24, "action": "rm -rf"},
                                   "junk", {"id": 5}])
        for _ in range(50):
            if any(cid == "b" * 24 for cid, _ in agent.ledger.unreported_commands()):
                break
            time.sleep(0.05)
        self.assertEqual(calls, ["start"])
        results = dict(agent.ledger.unreported_commands())
        self.assertIn("unsupported", results["c" * 24]["error"])

    def test_results_reach_the_hub_and_are_not_resent(self):
        self.finish_open_commands()
        record = self.store.request_command("jetson-test-01", "start", "boss")
        agent = self.agent()
        agent.config["allow_remote_control"] = True
        agent.collector_command = lambda action, command_id=None: {"ok": True, "collector": {"status": "running"}}
        reply = agent.client.request("POST", "/api/v1/device/heartbeat", json_body={"remote_control": True})
        agent.handle_commands(reply["commands"])
        for _ in range(100):
            if self.store.commands("jetson-test-01")[0]["state"] == "done" and not agent.ledger.unreported_commands():
                break
            time.sleep(0.05)
        self.assertEqual(self.store.commands("jetson-test-01")[0]["state"], "done")
        self.assertEqual(agent.ledger.unreported_commands(), [])

    def test_reset_sps30_is_queued_and_refused_on_the_device_while_collecting(self):
        self.finish_open_commands()
        record = self.store.request_command("jetson-test-01", "reset_sps30", "boss")
        self.assertEqual(record["action"], "reset_sps30")
        self.finish_open_commands()
        agent = self.agent()
        with unittest.mock.patch("uplink.agent._collector_state", return_value={"status": "running", "alive": True}), \
                unittest.mock.patch("uplink.agent.subprocess.run") as run:
            result = agent.collector_command("reset_sps30", "d" * 24)
        self.assertFalse(result["ok"])
        self.assertIn("stop it", result["error"])
        run.assert_not_called()
        done = unittest.mock.Mock(returncode=1, stdout=json.dumps({"ok": False, "hint": "power cycle needed",
                                  "steps": {"probe_before": {"ok": False, "error": "TimeoutError"}}}) + "\n", stderr="")
        with unittest.mock.patch("uplink.agent._collector_state", return_value={"status": "failed", "alive": False}), \
                unittest.mock.patch("uplink.agent.subprocess.run", return_value=done) as run:
            result = agent.collector_command("reset_sps30", "e" * 24)
        self.assertFalse(result["ok"])
        self.assertEqual(result["sps30"]["hint"], "power cycle needed")
        self.assertEqual(result["sps30"]["steps"], {"probe_before": "TimeoutError"})
        self.assertTrue(run.call_args[0][0][1].endswith("13_reset_sps30.py"))

    def test_set_role_cli(self):
        viewers = self.data_root / "config" / "viewers.json"
        saved = viewers.read_text()
        try:
            self.assertEqual(hub_cli.main(["set-role", "--data-root", str(self.data_root), "--user", "viewer",
                                           "--role", "operator"]), 0)
            self.assertEqual(json.loads(viewers.read_text())["viewer"]["role"], "operator")
            self.assertEqual(hub_cli.main(["set-role", "--data-root", str(self.data_root), "--user", "ghost",
                                           "--role", "operator"]), 1)
        finally:
            viewers.write_text(saved)



class InterruptedUnitTests(HubFixture):
    """Units whose producer died before writing a final status (reboot, power loss)."""

    def identity(self, alive):
        stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if alive:
            return {"pid": os.getpid(), "start_ticks": stat[19], "boot_id": boot}
        return {"pid": os.getpid(), "start_ticks": stat[19], "boot_id": "00000000-previous-boot"}

    def free_run(self, name, *, identity, mtime=OLD):
        run = self.runs / name
        write(run / "metadata.json", json.dumps({"run_id": name}).encode(), mtime)     # no status
        write(run / "scalars.csv", b"sequence\n0\n1\n", mtime)
        if identity is not None:
            write(run / "control.json", json.dumps({"status": "running", "token": "t", "identity": identity}).encode(), mtime)
        return run

    def test_free_run_of_a_dead_collector_ships_as_interrupted_unchanged(self):
        run = self.free_run("20261012T031001.604592Z", identity=self.identity(alive=False))
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in run.iterdir()}
        agent = self.agent()
        self.assertEqual(agent.scan_once()["committed"], 1)
        self.assertEqual({p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in run.iterdir()}, before)
        listed = next(u for u in self.store.units("jetson-test-01") if u["unit"] == "runs/20261012T031001.604592Z")
        self.assertEqual((listed["summary"]["status"], listed["summary"]["status_source"]), ("interrupted", "uplink_inferred"))

    def test_unprovable_or_live_or_recent_runs_are_not_shipped(self):
        self.free_run("20260903T041706Z", identity=None)                       # legacy, no control.json
        self.free_run("20261012T040000.000000Z", identity=self.identity(alive=True))
        self.free_run("20261012T050000.000000Z", identity=self.identity(alive=False), mtime=time.time())
        units, skipped = discover({"runs": self.runs}, settle_s=60, lock_path=self.edge / "no.lock")
        self.assertEqual(units, [])
        reasons = {s["unit"]: s["reason"] for s in skipped}
        self.assertIn("no control.json identity", reasons["runs/20260903T041706Z"])
        self.assertIn("may still be running", reasons["runs/20261012T040000.000000Z"])
        self.assertIn("settling", reasons["runs/20261012T050000.000000Z"])

    def test_trial_is_interrupted_only_when_no_acquisition_is_alive(self):
        make_trial(self.dataset, "_smoke", "combustion_20261012T070000Z", status="running")
        lock = self.edge / "hardware.lock"
        lock.write_text(json.dumps(self.identity(alive=True)))
        units, skipped = discover({"dataset": self.dataset}, settle_s=60, lock_path=lock)
        self.assertEqual(units, [])
        self.assertIn("may still be running", skipped[0]["reason"])
        lock.write_text(json.dumps(self.identity(alive=False)))
        units, _ = discover({"dataset": self.dataset}, settle_s=60, lock_path=lock)
        self.assertEqual([(u.key, u.status) for u in units], [("dataset/_smoke/combustion_20261012T070000Z", "interrupted")])
        units, _ = discover({"dataset": self.dataset}, settle_s=60, lock_path=self.edge / "never.lock")
        self.assertEqual([u.status for u in units], ["interrupted"], "no lock file: nothing has ever held the bus")

    def test_unknown_non_final_statuses_are_still_refused(self):
        make_trial(self.dataset, "_smoke", "dust_20261012T080000Z", status="mystery")
        units, skipped = discover({"dataset": self.dataset}, settle_s=60, lock_path=self.edge / "never.lock")
        self.assertEqual(units, [])
        self.assertIn("is not final", skipped[0]["reason"])



class ScenarioViewTests(HubFixture):
    def test_definitions_match_the_trial_runner_and_the_recorded_columns(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("runner_for_hub", ROOT / "jetson_deploy/scripts/12_run_trial.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        from sensors.snapshot import SCALAR_FIELDS
        definitions = json.loads((ROOT / "src/edge_hub/scenarios.json").read_text())
        ids = [s["id"] for s in definitions["scenarios"]]
        self.assertEqual(sorted(ids), sorted(runner.SCENARIOS), "every runner scenario is described, and only those")
        self.assertEqual({s["id"] for s in definitions["scenarios"] if s["status"] == "bench"},
                         {"combustion", "vapor_release", "dust"})
        for s in definitions["scenarios"]:
            with self.subTest(scenario=s["id"]):
                self.assertTrue(set(s["relevant_columns"]) <= set(SCALAR_FIELDS), s["relevant_columns"])
                if s["status"] == "bench":
                    self.assertTrue(s["expected"] and s["safety"] and s["bench_intervention"])

    def test_scenarios_api_needs_login_and_counts_received_trials(self):
        status, _, _ = self.https("GET", "/api/v1/scenarios")
        self.assertEqual(status, 401)
        make_trial(self.dataset, "dust", "dust_20261012T090000Z")
        self.agent().scan_once()
        cookie = self.login()
        status, _, spec = self.https("GET", "/api/v1/scenarios", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertEqual(spec["definitions_version"], "multisensor-v0.2")
        dust = next(s for s in spec["scenarios"] if s["id"] == "dust")
        self.assertGreaterEqual(dust["received"]["count"], 1)
        self.assertIn("dust_20261012T090000Z", [t["unit"].split("/")[-1] for t in dust["received"]["recent"]])
        combustion = next(s for s in spec["scenarios"] if s["id"] == "combustion")
        self.assertEqual(combustion["received"]["count"], 0)



class LabelingTests(HubFixture):
    """P1: live intervention events and human annotations (labels) for received units."""

    START = 1791820800.0   # 2026-10-12T03:00:00Z, device clock of tick 0

    def bench_trial(self, name="dust_20261012T030000Z", ticks=120, scenario="dust"):
        trial = make_trial(self.dataset, "_smoke", name, frames=3)
        import datetime as dt
        rows = ["sequence,timestamp_utc,pm10_ug_m3,pm2_5_ug_m3"]
        for i in range(ticks):
            stamp = dt.datetime.fromtimestamp(self.START + i, dt.timezone.utc).isoformat()
            rows.append(f"{i},{stamp},{10 + (i % 10) * 0.1 if i < 30 else 50 + i:.2f},{5 + (i % 5) * 0.1:.2f}")
        write(trial / "scalars.csv", ("\n".join(rows) + "\n").encode())
        write(trial / "experiment.json", json.dumps({
            "trial_id": name, "scenario_id": scenario, "status": "completed", "test_mode": True,
            "phase_plan": [{"phase": "baseline", "start_tick": 0, "end_tick": 29},
                           {"phase": "anomaly", "start_tick": 30, "end_tick": 89},
                           {"phase": "recovery", "start_tick": 90, "end_tick": ticks - 1}]}).encode())
        self.agent().scan_once()
        return f"dataset/_smoke/{name}"

    def post(self, path, body, cookie, headers=None):
        base = {"Cookie": cookie, "Content-Type": "application/json", "X-Hub-Request": "1"}
        base.update(headers or {})
        return self.https("POST", path, json.dumps(body), base)

    def labels(self):
        return self.server.RequestHandlerClass.hub.labels

    def test_events_are_operator_only_voidable_and_report_the_open_intervention(self):
        viewer, boss = self.login(), self.login("boss", "operator horse battery")
        body = {"device": "jetson-test-01", "kind": "intervention_start", "scenario": "combustion", "note": "30 cm"}
        self.assertEqual(self.post("/api/v1/events", body, viewer)[0], 403)
        self.assertEqual(self.post("/api/v1/events", body, boss, {"X-Hub-Request": "0"})[0], 403)
        self.assertEqual(self.post("/api/v1/events", dict(body, kind="label_it"), boss)[0], 400)
        status, _, start = self.post("/api/v1/events", body, boss)
        self.assertEqual((status, start["user"]), (200, "boss"))
        _, _, listed = self.https("GET", "/api/v1/events?device=jetson-test-01", headers={"Cookie": viewer})
        self.assertEqual(listed["open"]["id"], start["id"])
        self.assertEqual(self.post("/api/v1/events", dict(body, kind="intervention_end"), boss)[0], 200)
        _, _, listed = self.https("GET", "/api/v1/events?device=jetson-test-01", headers={"Cookie": viewer})
        self.assertIsNone(listed["open"])
        status, _, _ = self.post("/api/v1/events/void", {"id": start["id"]}, boss)
        self.assertEqual(status, 200)
        _, _, listed = self.https("GET", "/api/v1/events?device=jetson-test-01", headers={"Cookie": viewer})
        voided = next(e for e in listed["events"] if e["id"] == start["id"])
        self.assertTrue(voided["voided"], "voided, never deleted")
        self.assertEqual(self.post("/api/v1/events/void", {"id": start["id"]}, boss)[0], 404)

    def test_heartbeat_clock_offset_maps_hub_time_events_to_device_ticks(self):
        unit = self.bench_trial()
        self.store.heartbeat("jetson-test-01", "x", {"sent_epoch": time.time() - 2.5})
        offset = self.store.live["jetson-test-01"]["status"]["clock_offset_s"]
        self.assertAlmostEqual(offset, 2.5, delta=0.5)
        labels = self.labels()
        start = labels.add_event("jetson-test-01", "intervention_start", "boss", "dust")
        end = labels.add_event("jetson-test-01", "intervention_end", "boss", "dust")
        with self.store._db_lock:   # place them at device ticks 40 and 75 (hub clock = device clock + offset)
            for event, tick in ((start, 40), (end, 75)):
                self.store.db.execute("UPDATE events SET hub_epoch=? WHERE id=?", (self.START + tick + offset, event["id"]))
            self.store.db.commit()
        mapped = {e["id"]: e for e in labels.unit_events("jetson-test-01", unit)["events"]}
        self.assertEqual((mapped[start["id"]]["tick"], mapped[end["id"]]["tick"]), (40, 75))
        self.assertLess(mapped[start["id"]]["tick_error_s"], 0.01)

    def test_baseline_band_comes_from_the_baseline_phase_only(self):
        unit = self.bench_trial()
        band = self.labels().baseline("jetson-test-01", unit, ["pm10_ug_m3", "nope"])
        self.assertTrue(band["available"])
        pm = band["stats"]["pm10_ug_m3"]
        self.assertEqual(pm["n"], 30)
        self.assertAlmostEqual(pm["mean"], 10.45, places=6)       # ticks 0..29 only, the jump at 30 is excluded
        self.assertAlmostEqual(pm["high"] - pm["mean"], 3 * pm["std"], places=9)
        self.assertNotIn("nope", band["stats"])

    def test_annotations_are_validated_revisioned_bound_to_raw_and_operator_only(self):
        unit = self.bench_trial()
        viewer, boss = self.login(), self.login("boss", "operator horse battery")
        good = {"device": "jetson-test-01", "unit": unit, "base_revision": 0, "note": "first pass",
                "intervals": [{"start_tick": 0, "end_tick": 29, "label": "normal"},
                              {"start_tick": 31, "end_tick": 80, "label": "dust", "confidence": "high"},
                              {"start_tick": 81, "end_tick": 110, "label": "transition", "confidence": "medium"}],
                "onsets": {"pm": 31}}
        self.assertEqual(self.post("/api/v1/annotation", good, viewer)[0], 403)
        bad_cases = [dict(good, intervals=[{"start_tick": 0, "end_tick": 40, "label": "dust"},
                                           {"start_tick": 40, "end_tick": 50, "label": "normal"}]),     # overlap
                     dict(good, intervals=[{"start_tick": 0, "end_tick": 120, "label": "dust"}]),       # past end
                     dict(good, intervals=[{"start_tick": 0, "end_tick": 5, "label": "cat"}]),          # label
                     dict(good, intervals=[{"start_tick": 0, "end_tick": 5, "label": "dust", "confidence": "sure"}]),
                     dict(good, onsets={"smell": 3}), dict(good, base_revision="0")]
        for body in bad_cases:
            self.assertEqual(self.post("/api/v1/annotation", body, boss)[0], 400, body)
        status, _, saved = self.post("/api/v1/annotation", good, boss)
        self.assertEqual((status, saved["revision"]), (200, 1))
        self.assertEqual(self.post("/api/v1/annotation", good, boss)[0], 409, "stale base_revision")
        second = dict(good, base_revision=1, intervals=good["intervals"][:2])
        self.assertEqual(self.post("/api/v1/annotation", second, boss)[2]["revision"], 2)
        _, _, current = self.https("GET", f"/api/v1/annotation?device=jetson-test-01&unit={unit}", headers={"Cookie": viewer})
        self.assertEqual((current["revision"], len(current["current"]["intervals"])), (2, 2))
        self.assertEqual(current["current"]["guide_version"], "labeling-guide-v1")
        receipt = json.loads((self.data_root / "receipts/jetson-test-01" / (unit + ".json")).read_text())
        self.assertEqual(current["current"]["manifest_sha256"], receipt["manifest_sha256"])
        files = sorted((self.data_root / "annotations/anno-v1/jetson-test-01" / unit).glob("rev-*.json"))
        self.assertEqual(len(files), 2)
        self.assertTrue(all(p.stat().st_mode & 0o222 == 0 for p in files), "revisions are read-only")
        _, _, exported = self.https("GET", "/api/v1/annotations/export", headers={"Cookie": viewer})
        mine = next(a for a in exported["annotations"] if a["unit"] == unit)
        self.assertTrue(mine["raw_matches"])
        self.assertEqual(self.store.verify("jetson-test-01")[-1]["ok"], True, "raw untouched by labeling")


def add_zed(unit_dir: Path, ticks=(3, 4, 5, 7), mtime: float = OLD) -> list[bytes]:
    """A zed-v1 directory as the side recorder writes it: one segment, JPEGs stored back to back."""
    jpegs = [b"\xff\xd8\xff\xe0" + bytes([t]) * (50 + t) for t in ticks]
    offsets, total = [0], 0
    for j in jpegs:
        total += len(j)
        offsets.append(total)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("left_jpeg.npy", npy(b"".join(jpegs), "|u1", (total,)))
        archive.writestr("left_offsets.npy", npy(b"".join(o.to_bytes(8, "little") for o in offsets), "<i8", (len(offsets),)))
    write(unit_dir / "zed" / "seg_01_00000_left.npz", buffer.getvalue(), mtime)
    write(unit_dir / "zed" / "seg_01_00000_depth.npz", b"PK-not-read-by-the-hub", mtime)
    rows = ["session,segment,index,tick,frame_monotonic_ns,image_timestamp_ns,offset_ms"]
    rows += [f"1,0,{i},{t},{t * 10**9},{t * 10**9},-4.5" for i, t in enumerate(ticks)]
    write(unit_dir / "zed" / "frames.csv", ("\n".join(rows) + "\n").encode(), mtime)
    meta = {"schema": "zed-v1", "status": "complete", "camera": {"model": "ZED 2i", "serial_number": 1},
            "sessions": [{"session": 1, "frames": len(ticks), "missed_ticks": 1, "first_tick": ticks[0],
                          "last_tick": ticks[-1], "stop_reason": "collection_ended"}],
            "recorder": {"pid": 1, "start_ticks": "1", "boot_id": "gone"}}
    write(unit_dir / "zed" / "zed.json", json.dumps(meta).encode(), mtime)
    return jpegs


class ZedTests(HubFixture):
    """ZED side recorder on the hub: on/off commands, live thumbnail, frames of received units."""

    def test_zed_commands_are_queued_and_only_flip_the_device_switch(self):
        self.store.heartbeat("jetson-test-01", "127.0.0.1", {"remote_control": True, "interval_s": 10})
        for command in self.store.take_commands("jetson-test-01"):
            self.store.command_result("jetson-test-01", command["id"], True, {})
        record = self.store.request_command("jetson-test-01", "zed_on", "boss")
        self.assertEqual(record["action"], "zed_on")
        for command in self.store.take_commands("jetson-test-01"):
            self.store.command_result("jetson-test-01", command["id"], True, {})
        agent = self.agent()
        switch = self.edge / "zed.json"
        switch.write_text(json.dumps({"enabled": False, "jpeg_quality": 70}))
        agent.config.update(allow_remote_control=True, zed_config=str(switch))
        agent.collector_command = lambda *a, **k: self.fail("ZED commands never touch the collector")
        agent.handle_commands([{"id": "f" * 24, "action": "zed_on", "requested_by": "boss"}])
        for _ in range(100):
            if agent.ledger.unreported_commands():
                break
            time.sleep(0.02)
        result = dict(agent.ledger.unreported_commands())["f" * 24]
        saved = json.loads(switch.read_text())
        self.assertEqual((result["ok"], result["zed"]["enabled"]), (True, True))
        self.assertEqual((saved["enabled"], saved["jpeg_quality"], saved["changed_by"]), (True, 70, "hub:boss"))
        self.assertTrue(agent.zed_command("zed_off")["ok"])
        self.assertFalse(json.loads(switch.read_text())["enabled"])

    def test_heartbeat_carries_zed_status_and_a_checked_thumbnail(self):
        import base64
        agent = self.agent()
        switch = self.edge / "zed.json"
        switch.write_text(json.dumps({"enabled": True}))
        agent.config["zed_config"] = str(switch)
        thumb = base64.b64encode(b"\xff\xd8\xff\xe0" + b"x" * 400).decode()
        status = {"state": "recording", "run_id": "r1", "frames": 12, "missed_ticks": 0, "pid": 9,
                  "preview": {"run_id": "r1", "tick": 11, "utc": "2026-10-13T00:00:11Z", "jpeg_b64": thumb}}
        with unittest.mock.patch("zedcap.control.read_status", return_value=status):
            body = agent.heartbeat_body()
            again = agent.heartbeat_body()
        self.assertEqual((body["zed"]["enabled"], body["zed"]["state"], body["zed"]["frames"]), (True, "recording", 12))
        self.assertNotIn("preview", body["zed"])
        self.assertEqual(body["zed_preview"]["tick"], 11)
        self.assertNotIn("zed_preview", again, "the same thumbnail is sent once")
        agent.client.request("POST", "/api/v1/device/heartbeat", json_body=body)
        cookie = self.login()
        _, _, live = self.https("GET", "/api/v1/live?device=jetson-test-01", headers={"Cookie": cookie})
        self.assertEqual((live["zed_preview"]["tick"], live["zed_preview"]["stale"]), (11, False))
        self.assertNotIn("zed_preview", live["status"])
        self.assertEqual(live["status"]["zed"]["state"], "recording")
        for bad in ({"jpeg_b64": base64.b64encode(b"GIF89a" + b"x" * 400).decode()}, {"jpeg_b64": "A" * 200_000},
                    {"jpeg_b64": "not base64!!" * 20}, "junk"):
            self.store.heartbeat("jetson-test-02", "127.0.0.1", {"zed_preview": bad, "zed": {"state": "idle"}})
            self.assertNotIn("zed_preview", self.store.live_view("jetson-test-02"))
        self.store.heartbeat("jetson-test-01", "127.0.0.1", {"zed": {"state": "idle"}})
        self.assertTrue(self.store.live_view("jetson-test-01")["zed_preview"]["stale"])

    def test_received_unit_serves_the_saved_frame_nearest_a_tick(self):
        trial = make_trial(self.dataset, "_smoke", "dust_Z", frames=8)
        jpegs = add_zed(trial)
        agent = self.agent()
        agent.scan_once()
        unit = "dataset/_smoke/dust_Z"
        self.assertTrue((self.raw(unit) / "zed" / "frames.csv").is_file())
        cookie = self.login()
        _, _, detail = self.https("GET", f"/api/v1/unit?device=jetson-test-01&unit={unit}", headers={"Cookie": cookie})
        self.assertEqual((detail["zed"]["frames"], detail["zed"]["status"], detail["summary"]["zed_frames"]), (4, "complete", 4))
        import base64
        for tick, expect in ((4, 4), (6, 5), (0, 3), (99, 7)):
            status, _, frame = self.https("GET", f"/api/v1/unit/zed?device=jetson-test-01&unit={unit}&tick={tick}",
                                          headers={"Cookie": cookie})
            with self.subTest(tick=tick):
                self.assertEqual((status, frame["frame_tick"], frame["saved_ticks"]), (200, expect, [3, 7]))
                self.assertEqual(base64.b64decode(frame["jpeg_b64"]), jpegs[(3, 4, 5, 7).index(expect)])
        self.assertEqual(self.https("GET", f"/api/v1/unit/zed?device=jetson-test-01&unit={unit}&tick=x",
                                    headers={"Cookie": cookie})[0], 400)
        self.assertEqual(self.https("GET", f"/api/v1/unit/zed?device=jetson-test-01&unit={unit}&tick=1")[0], 401)
        make_trial(self.dataset, "_smoke", "dust_noZ", frames=3)
        agent.scan_once()
        self.assertEqual(self.https("GET", "/api/v1/unit/zed?device=jetson-test-01&unit=dataset/_smoke/dust_noZ&tick=1",
                                    headers={"Cookie": cookie})[0], 404)
        self.assertTrue(all(r["ok"] for r in self.store.verify("jetson-test-01")))

    def test_language_strings_are_served_and_cover_both_languages(self):
        status, headers, body = self.https("GET", "/i18n.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertIn(b"en:", body)
        status, _, page = self.https("GET", "/")
        self.assertIn(b'src="i18n.js"', page)
        self.assertIn(b'data-lang="en"', page)
        spec = json.loads((ROOT / "src/edge_hub/scenarios.json").read_text())
        for scenario in spec["scenarios"]:
            for field in ("factory_situation", "bench_intervention", "remark", "safety", "must_not_change"):
                if scenario.get(field):
                    self.assertIn(f"{field}_en", scenario, f"{scenario['id']}.{field} has no English text")


class LocalDeletionTests(HubFixture):
    """delete_after_upload: local copies go only after both sides re-verified them."""

    def agent(self, **kw):
        agent = super().agent(**kw)
        agent.config["keep_days"] = 0          # most tests check verification, not retention
        return agent

    def test_prune_deletes_only_verified_units_and_leaves_a_tombstone(self):
        trial = make_trial(self.dataset, "dust", "trial_001")
        agent = self.agent()
        agent.scan_once()
        listing = agent.prune(dry_run=True)
        self.assertEqual([(r["unit"], r["action"]) for r in listing], [("dataset/dust/trial_001", "would_delete")])
        self.assertTrue((trial / "experiment.json").exists(), "a dry run deletes nothing")
        report = agent.prune()
        self.assertEqual([r["action"] for r in report], ["deleted"])
        self.assertEqual([p.name for p in trial.iterdir()], ["UPLOADED_TO_HUB.json"])
        tomb = json.loads((trial / "UPLOADED_TO_HUB.json").read_text())
        receipt = json.loads((self.data_root / "receipts/jetson-test-01/dataset/dust/trial_001.json").read_text())
        self.assertEqual(tomb["manifest_sha256"], receipt["manifest_sha256"])
        self.assertEqual(agent.ledger.unit("dataset/dust/trial_001")["state"], "deleted_local")
        self.assertFalse(any(p.name.startswith(".deleting") for p in trial.parent.iterdir()))
        self.assertEqual(agent.scan_once()["final"], 0, "a tombstone is never a unit again")
        self.assertTrue(self.store.verify_unit("jetson-test-01", "dataset/dust/trial_001")["ok"], "hub copy untouched")
        import importlib.util
        spec = importlib.util.spec_from_file_location("trial_runner_alloc", ROOT / "jetson_deploy/scripts/12_run_trial.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertEqual(runner.allocate_trial_dir(self.dataset, "dust", False)[1], "trial_002",
                         "numbering continues past a deleted trial")

    def test_changed_local_files_or_a_bad_hub_copy_keep_the_local_unit(self):
        changed = make_trial(self.dataset, "dust", "trial_010")
        corrupt = make_trial(self.dataset, "dust", "trial_011")
        pending = make_trial(self.dataset, "dust", "trial_012")
        agent = self.agent()
        agent.scan_once()
        agent.ledger.set_state("dataset/dust/trial_012", "error", error="simulated failed upload")
        write(changed / "big.bin", b"edited after the upload")
        hub_file = self.raw("dataset/dust/trial_011") / "big.bin"
        os.chmod(hub_file, 0o644)
        hub_file.write_bytes(b"bit rot")
        os.chmod(hub_file, 0o444)
        report = {r["unit"]: r for r in agent.prune()}
        self.assertEqual(report["dataset/dust/trial_010"]["action"], "kept")
        self.assertIn("differ", report["dataset/dust/trial_010"]["reason"])
        self.assertEqual(report["dataset/dust/trial_011"]["action"], "kept")
        self.assertIn("failed verification", report["dataset/dust/trial_011"]["reason"])
        self.assertEqual(report["dataset/dust/trial_012"]["action"], "kept")
        for trial in (changed, corrupt, pending):
            self.assertTrue((trial / "experiment.json").exists())

    def test_verify_api_is_per_device_and_scan_prunes_only_when_enabled(self):
        make_trial(self.dataset, "dust", "trial_020")
        agent = self.agent()
        agent.scan_once()
        self.assertTrue((self.dataset / "dust/trial_020/experiment.json").exists(), "off by default")
        other = HubClient([self.url], self.other_token, "jetson-test-02", ca_file=self.ca, timeout=10)
        with self.assertRaises(HubRejected) as caught:
            other.request("POST", "/api/v1/device/units/verify", json_body={"unit": "dataset/dust/trial_020"})
        self.assertEqual(caught.exception.status, 404)
        status, _, _ = self.https("POST", "/api/v1/device/units/verify", json.dumps({"unit": "dataset/dust/trial_020"}),
                                  {"Content-Type": "application/json"})
        self.assertEqual(status, 401)
        agent.config["delete_after_upload"] = True
        self.assertEqual(agent.scan_once()["deleted"], 1)
        self.assertFalse((self.dataset / "dust/trial_020/experiment.json").exists())

    def test_retention_keeps_recent_uploads_until_the_disk_runs_low_then_oldest_first(self):
        old = make_trial(self.dataset, "dust", "trial_040")
        agent = self.agent()
        agent.scan_once()
        new = make_trial(self.dataset, "dust", "trial_041")
        agent.scan_once()
        receipt = agent.ledger.unit("dataset/dust/trial_040")["receipt"]
        agent.ledger.set_state("dataset/dust/trial_040", "committed",
                               receipt=dict(receipt, committed_utc="2026-01-01T00:00:00+00:00"))
        agent.config.update(keep_days=7, min_free_gb=50)
        plenty = unittest.mock.Mock(free=200 * 1024 ** 3)
        with unittest.mock.patch("uplink.agent.shutil.disk_usage", return_value=plenty):
            report = {r["unit"]: r for r in agent.prune()}
        self.assertEqual(report["dataset/dust/trial_040"]["action"], "deleted")
        self.assertIn("retention period over", report["dataset/dust/trial_040"]["reason"])
        self.assertEqual(report["dataset/dust/trial_041"]["action"], "kept")
        self.assertIn("days left", report["dataset/dust/trial_041"]["reason"])
        low = unittest.mock.Mock(free=10 * 1024 ** 3)
        with unittest.mock.patch("uplink.agent.shutil.disk_usage", return_value=low):
            report = agent.prune()
        self.assertEqual([(r["unit"], r["action"]) for r in report], [("dataset/dust/trial_041", "deleted")])
        self.assertIn("free space below 50 GB", report[0]["reason"])
        self.assertEqual([p.name for p in old.iterdir()], ["UPLOADED_TO_HUB.json"])
        self.assertEqual([p.name for p in new.iterdir()], ["UPLOADED_TO_HUB.json"])

    def test_an_interrupted_deletion_is_finished_with_a_tombstone(self):
        trial = make_trial(self.dataset, "dust", "trial_030")
        agent = self.agent()
        agent.scan_once()
        os.rename(trial, trial.with_name(".deleting-trial_030"))      # crash right after the rename
        self.assertEqual(agent.finish_interrupted_deletions(), 1)
        self.assertEqual([p.name for p in trial.iterdir()], ["UPLOADED_TO_HUB.json"])
        self.assertFalse(trial.with_name(".deleting-trial_030").exists())
        stray = self.dataset / "dust" / ".deleting-not_ours"
        stray.mkdir()
        agent.finish_interrupted_deletions()
        self.assertTrue(stray.exists(), "never removes a directory the ledger does not know")


class BackupTests(HubFixture):
    """Nightly copy to a second disk: mirror, consistent DB snapshot, re-hash of the copy."""

    def test_backup_copies_verifies_and_reports_to_the_web(self):
        from src.edge_hub.backup import run_backup
        make_trial(self.dataset, "dust", "trial_050")
        self.agent().scan_once()
        with self.store._db_lock:
            self.store.db.execute("CREATE TABLE IF NOT EXISTS backup_probe(x TEXT)")
            self.store.db.execute("INSERT INTO backup_probe VALUES('kept')")
            self.store.db.commit()
        dest = self.tmp / f"backup_{time.time_ns()}"
        status = run_backup(self.data_root, dest, allow_same_filesystem=True)
        self.assertTrue(status["ok"], status)
        self.assertGreaterEqual(status["verify"]["units"], 1)
        copy = dest / "current"
        self.assertTrue((copy / "raw/jetson-test-01/dataset/dust/trial_050/big.bin").is_file())
        self.assertFalse((copy / "staging").exists(), "unfinished uploads are not copied")
        import sqlite3
        snap = sqlite3.connect(copy / "hub.sqlite")
        self.assertEqual(snap.execute("SELECT x FROM backup_probe").fetchall(), [("kept",)])
        snap.close()
        cookie = self.login()
        _, _, overview = self.https("GET", "/api/v1/overview", headers={"Cookie": cookie})
        self.assertTrue(overview["backup"]["ok"])
        self.assertGreaterEqual(overview["backup"]["units"], 1)
        # a damaged copy is detected by the next run's verification only if rsync does not repair it:
        # rsync -a repairs size/mtime changes, so damage the copy without touching size and mtime
        target = copy / "raw/jetson-test-01/dataset/dust/trial_050/big.bin"
        info = target.stat()
        os.chmod(target, 0o644)
        data = bytearray(target.read_bytes()); data[0] ^= 0xFF
        target.write_bytes(bytes(data))
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
        from src.edge_hub.backup import verify_copy
        self.assertEqual(verify_copy(copy)["problem_count"], 1)

    def test_backup_refuses_the_data_disk_itself_and_reports_failures(self):
        from src.edge_hub.backup import run_backup
        status = run_backup(self.data_root, self.data_root / "inside")
        self.assertFalse(status["ok"])
        self.assertIn("outside the data root", status["error"])
        status = run_backup(self.data_root, self.tmp / "same_fs")
        self.assertFalse(status["ok"])
        self.assertIn("same filesystem", status["error"])
        status = run_backup(self.data_root, self.tmp / "no_rsync", rsync="/nonexistent/rsync", allow_same_filesystem=True)
        self.assertFalse(status["ok"])
        self.assertEqual(json.loads((self.data_root / "backup_status.json").read_text())["ok"], False)
        (self.data_root / "backup_status.json").unlink()


class WeatherTests(HubFixture):
    """Device site history entered on the web, and hourly outdoor weather fetched for it."""

    NOW = 1792077600.0          # 2026-10-15T15:20:00Z

    def setUp(self):
        super().setUp()
        with self.store._db_lock:      # tables are created by the hub's own Weather at startup
            self.store.db.execute("DELETE FROM device_sites")
            self.store.db.execute("DELETE FROM weather_hourly")
            self.store.db.commit()

    def weather(self, replies=None):
        from src.edge_hub.weather import Weather
        calls = []

        def fetch(url):
            from urllib.parse import parse_qs, urlsplit
            q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
            calls.append((url.split("?")[0], q))
            import datetime as dt
            first = dt.datetime.fromisoformat(q["start_date"] + "T00:00:00+00:00").timestamp()
            last = dt.datetime.fromisoformat(q["end_date"] + "T23:00:00+00:00").timestamp()
            hours = list(range(int(first), int(last) + 1, 3600))
            temp = [20.0 + (h // 3600) % 24 * 0.1 for h in hours]
            if replies == "archive_lag":
                temp = [v if h < first + 3 * 3600 else None for h, v in zip(hours, temp)]
            return {"hourly": {"time": hours, "temperature_2m": temp,
                               "relative_humidity_2m": [60.0] * len(hours),
                               "surface_pressure": [1010.0] * len(hours)}}
        return Weather(self.store, fetch=fetch), calls

    def test_site_history_validation_and_correction(self):
        from src.edge_hub.store import HubError
        w, _ = self.weather()
        with self.assertRaises(HubError):
            w.set_site("jetson-unknown-9", "lab", 37.4, 127.1, None, "boss", now=self.NOW)
        for lat, lon in ((91, 0), (0, 181), ("x", 0), (float("nan"), 0)):
            with self.assertRaises(HubError):
                w.set_site("jetson-test-02", "lab", lat, lon, None, "boss", now=self.NOW)
        with self.assertRaises(HubError):
            w.set_site("jetson-test-02", "", 37.4, 127.1, None, "boss", now=self.NOW)
        with self.assertRaises(HubError):
            w.set_site("jetson-test-02", "lab", 37.4, 127.1, "2026-10-15T00:00:00", "boss", now=self.NOW)  # no zone
        with self.assertRaises(HubError):
            w.set_site("jetson-test-02", "lab", 37.4, 127.1, "2026-10-20T00:00:00Z", "boss", now=self.NOW)
        a = w.set_site("jetson-test-02", "KETI lab", 37.38123, 127.11789, "2026-10-15T00:00:00Z", "boss", now=self.NOW)
        self.assertEqual((a["latitude"], a["longitude"]), (37.38, 127.12), "kept to about 1 km")
        b = w.set_site("jetson-test-02", "Germany testbed", 50.78, 6.08, "2026-10-15T12:00:00Z", "boss", now=self.NOW)
        fix = w.set_site("jetson-test-02", "KETI lab B", 37.39, 127.11, "2026-10-15T00:00:00Z", "boss", now=self.NOW)
        history = w.history("jetson-test-02")
        self.assertEqual([h["id"] for h in history], [a["id"], fix["id"], b["id"]])
        self.assertTrue(history[0]["superseded"], "same start: the later entry is a correction, both kept")
        self.assertEqual(w.site_at("jetson-test-02", self.NOW - 10 * 3600)["label"], "KETI lab B")
        self.assertEqual(w.site_at("jetson-test-02", self.NOW)["label"], "Germany testbed")
        self.assertIsNone(w.site_at("jetson-test-02", self.NOW - 2 * 86400), "nothing before the first entry")

    def test_update_fetches_missing_completed_hours_per_site_and_joins_by_time(self):
        w, calls = self.weather()
        w.set_site("jetson-test-01", "KETI lab", 37.38, 127.12, "2026-10-15T00:00:00Z", "boss", now=self.NOW)
        w.set_site("jetson-test-01", "Germany testbed", 50.78, 6.08, "2026-10-15T10:00:00Z", "boss", now=self.NOW)
        status = w.update(now=self.NOW)
        self.assertTrue(status["ok"], status)
        self.assertEqual(status["sites"], 2)
        self.assertTrue(all(url.endswith("/v1/forecast") for url, _ in calls), "recent days use the forecast API")
        self.assertEqual({(q["latitude"], q["longitude"]) for _, q in calls}, {("37.38", "127.12"), ("50.78", "6.08")})
        rows = w.outdoor("jetson-test-01", self.NOW - 86400, self.NOW)
        self.assertEqual(rows[0]["utc"], "2026-10-15T00:00:00Z")
        self.assertEqual(rows[-1]["utc"], "2026-10-15T14:00:00Z", "the running hour is not stored")
        self.assertEqual(len(rows), 15, "one row per hour, no overlap at the site change")
        sites = {r["utc"][11:13]: r["site_id"] for r in rows}
        self.assertNotEqual(sites["09"], sites["10"], "the hour of the move belongs to the new site")
        again = w.update(now=self.NOW)
        self.assertEqual(again["requests"], 0, "nothing missing, nothing fetched")
        span = w.for_span("jetson-test-01", self.NOW - 3 * 3600, self.NOW - 3600)
        self.assertEqual(span["site"]["label"], "Germany testbed")
        self.assertEqual(span["humidity_pct"], 60.0)

    def test_old_periods_use_the_archive_and_lagging_hours_are_retried(self):
        w, calls = self.weather("archive_lag")
        w.set_site("jetson-test-02", "old lab", 10.0, 20.0, "2026-06-01T00:00:00Z", "boss", now=self.NOW)
        w.set_site("jetson-test-02", "new lab", 11.0, 21.0, "2026-06-02T00:00:00Z", "boss", now=self.NOW)
        status = w.update(now=self.NOW)
        old = [q for url, q in calls if q["latitude"] == "10.00"]
        self.assertEqual(len(old), 1)
        self.assertTrue(next(url for url, q in calls if q["latitude"] == "10.00").endswith("/v1/archive"))
        june2 = 1780358400.0      # 2026-06-02T00:00:00Z
        self.assertEqual(len(w.outdoor("jetson-test-02", 0, june2 - 1)), 3, "only hours with a value")
        calls.clear()
        w.update(now=self.NOW)
        self.assertIn(("10.00", "2026-06-01"), {(q["latitude"], q["start_date"]) for _, q in calls},
                      "hours the archive did not have yet are asked for again")
        self.assertTrue(status["ok"])

    def test_fetch_errors_are_reported_not_raised(self):
        from src.edge_hub.weather import Weather
        import urllib.error

        def broken(url):
            raise urllib.error.URLError("no route to host")
        w = Weather(self.store, fetch=broken)
        w.set_site("jetson-test-01", "lab", 1.0, 2.0, "2026-10-15T13:00:00Z", "boss", now=self.NOW)
        status = w.update(now=self.NOW)
        self.assertFalse(status["ok"])
        self.assertIn("no route to host", status["errors"][0])
        self.assertFalse(json.loads((self.data_root / "weather_status.json").read_text())["ok"])

    def test_web_routes_viewer_reads_operator_sets(self):
        viewer, boss = self.login(), self.login("boss", "operator horse battery")
        base = {"Content-Type": "application/json", "X-Hub-Request": "1"}
        body = {"device": "jetson-test-02", "label": "web lab", "latitude": 37.4, "longitude": 127.1,
                "valid_from_utc": "2026-10-16T00:00:00Z", "note": "from the web"}
        status, _, reply = self.https("POST", "/api/v1/site", json.dumps(body), dict(base, Cookie=viewer))
        self.assertEqual(status, 403)
        status, _, _ = self.https("POST", "/api/v1/site", json.dumps(body), {"Cookie": boss, "Content-Type": "application/json"})
        self.assertEqual(status, 403, "state change needs the same-origin header")
        status, _, reply = self.https("POST", "/api/v1/site", json.dumps(body), dict(base, Cookie=boss))
        self.assertEqual(status, 200, reply)
        self.assertEqual((reply["set_by"], reply["label"]), ("boss", "web lab"))
        status, _, reply = self.https("GET", "/api/v1/site?device=jetson-test-02", headers={"Cookie": viewer})
        self.assertEqual(status, 200)
        self.assertEqual(reply["history"][-1]["note"], "from the web")
        _, _, overview = self.https("GET", "/api/v1/overview", headers={"Cookie": viewer})
        dev = next(d for d in overview["devices"] if d["device"] == "jetson-test-02")
        self.assertIn("site", dev)
        status, _, reply = self.https("GET", "/api/v1/outdoor?device=jetson-test-02&since_s=86400", headers={"Cookie": viewer})
        self.assertEqual((status, type(reply["hours"])), (200, list))

    def test_cli_set_site_and_weather(self):
        rc = hub_cli.main(["set-site", "--data-root", str(self.data_root), "--device-id", "jetson-test-02",
                           "--label", "cli lab", "--lat", "37.4", "--lon", "127.1", "--valid-from", "2026-10-16T00:00:00Z"])
        self.assertEqual(rc, 0)
        self.assertEqual(hub_cli.main(["set-site", "--data-root", str(self.data_root), "--device-id", "jetson-test-02",
                                       "--label", "bad", "--lat", "99", "--lon", "0"]), 2)
        with unittest.mock.patch("src.edge_hub.weather.http_get_json", side_effect=OSError("offline in test")):
            self.assertEqual(hub_cli.main(["weather", "--data-root", str(self.data_root)]), 1)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
