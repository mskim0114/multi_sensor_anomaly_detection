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
                  "max_upload_bytes_per_s": 0}
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
        units, skipped = discover({"dataset": self.dataset, "runs": self.runs}, settle_s=60)
        self.assertEqual([u.key for u in units], ["dataset/normal/trial_004"])
        reasons = {s["unit"]: s["reason"] for s in skipped}
        self.assertIn("not final", reasons["dataset/normal/trial_001"])
        self.assertIn("settling", reasons["dataset/normal/trial_002"])
        self.assertIn("regular file", reasons["dataset/normal/trial_003"])
        self.assertIn("not final", reasons["runs/20261006T000000.000000Z"])
        units, _ = discover({"dataset": self.dataset}, include_test=False, settle_s=60)
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


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
