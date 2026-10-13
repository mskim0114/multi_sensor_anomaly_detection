"""Edge hub CLI.  Run from the repository root:  python -m src.edge_hub <command> ...

  init         --data-root DIR
  add-device   --data-root DIR --device-id ID --token-out FILE   (token written 0600, never printed)
  revoke-device --data-root DIR --device-id ID                   (token stops working at once)
  add-viewer   --data-root DIR --user NAME [--role operator]     (password typed at a prompt)
  set-role     --data-root DIR --user NAME --role viewer|operator  (operator may start/stop collection)
  gen-cert     --data-root DIR --san IP:10.252.219.59 --san DNS:hub.example
  serve        --data-root DIR [--host 0.0.0.0] [--port 8443] [--tls-cert --tls-key | --plain-http]
  verify       --data-root DIR [--device ID]                     (re-hash committed units)
  reindex      --data-root DIR
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from .server import hash_password, make_server
from .store import Store, check_device


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, file=sys.stderr, flush=True)


def cmd_add_device(args) -> int:
    store = Store(args.data_root)
    check_device(args.device_id)
    out = Path(args.token_out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    token = secrets.token_urlsafe(32)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(token + "\n")
    store.add_device(args.device_id, hashlib.sha256(token.encode()).hexdigest(), args.note or "")
    print(f"device {args.device_id} registered; token written to {out} (mode 600). "
          f"Copy it to the Jetson, then delete this copy.")
    return 0


def cmd_add_viewer(args) -> int:
    store = Store(args.data_root)
    password = getpass.getpass(f"password for {args.user}: ")
    if len(password) < 12 or password != getpass.getpass("again: "):
        print("passwords differ or shorter than 12 characters", file=sys.stderr)
        return 2
    path = store.root / "config" / "viewers.json"
    viewers = json.loads(path.read_text()) if path.exists() else {}
    role = args.role or (viewers.get(args.user) or {}).get("role", "viewer")   # a password reset keeps the role
    viewers[args.user] = dict(hash_password(password), role=role)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(viewers, stream, indent=1)
    os.replace(tmp, path)
    print(f"viewer {args.user} saved")
    return 0


def cmd_set_role(args) -> int:
    path = Store(args.data_root).root / "config" / "viewers.json"
    viewers = json.loads(path.read_text()) if path.exists() else {}
    if args.user not in viewers:
        print(f"no viewer {args.user}", file=sys.stderr)
        return 1
    viewers[args.user]["role"] = args.role
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(viewers, stream, indent=1)
    os.replace(tmp, path)
    print(f"{args.user} is now {args.role}")
    return 0


def cmd_gen_cert(args) -> int:
    """A private CA (stays on the server) and a server certificate signed by it.

    Devices pin ca.pem, so a changed network name only needs a new server certificate.
    """
    tls = Path(args.data_root) / "tls"
    tls.mkdir(parents=True, exist_ok=True)
    os.chmod(tls, 0o700)
    ca_key, ca_pem = tls / "ca.key", tls / "ca.pem"
    if not ca_key.exists():
        subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
                        "-nodes", "-keyout", str(ca_key), "-out", str(ca_pem), "-days", "3650",
                        "-subj", "/CN=factory-safety edge hub CA",
                        "-addext", "basicConstraints=critical,CA:TRUE",
                        "-addext", "keyUsage=critical,keyCertSign,cRLSign"], check=True)
        os.chmod(ca_key, 0o600)
    key, csr, crt = tls / "server.key", tls / "server.csr", tls / "server.pem"
    subprocess.run(["openssl", "req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
                    "-keyout", str(key), "-out", str(csr), "-subj", "/CN=factory-safety edge hub"], check=True)
    os.chmod(key, 0o600)
    ext = tls / "server.ext"
    ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
                   "extendedKeyUsage=serverAuth\nsubjectAltName=" + ",".join(args.san) + "\n")
    subprocess.run(["openssl", "x509", "-req", "-in", str(csr), "-CA", str(ca_pem), "-CAkey", str(ca_key),
                    "-CAcreateserial", "-out", str(crt), "-days", "825", "-extfile", str(ext)], check=True)
    csr.unlink()
    print(f"CA certificate for devices: {ca_pem}\nserver certificate: {crt}  key: {key}")
    return 0


def cmd_serve(args) -> int:
    store = Store(args.data_root)
    if not args.tls_cert and not args.plain_http:
        print("give --tls-cert/--tls-key, or --plain-http when TLS ends at a proxy/tunnel "
              "in front of a 127.0.0.1 bind", file=sys.stderr)
        return 2
    if args.plain_http and args.host not in ("127.0.0.1", "::1", "localhost"):
        print("--plain-http only binds to localhost", file=sys.stderr)
        return 2
    server = make_server(store, args.host, args.port, tls_cert=args.tls_cert, tls_key=args.tls_key,
                         trust_proxy=args.trust_proxy, log=log)
    scheme = "https" if args.tls_cert else "http"
    log(f"edge hub serving {scheme}://{args.host}:{args.port} data_root={store.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.edge_hub", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "add-device", "revoke-device", "add-viewer", "set-role", "gen-cert", "serve", "verify",
                 "reindex", "backup"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", required=True, type=Path)
        if name == "revoke-device":
            p.add_argument("--device-id", required=True)
        if name == "add-device":
            p.add_argument("--device-id", required=True)
            p.add_argument("--token-out", required=True)
            p.add_argument("--note")
        if name in ("add-viewer", "set-role"):
            p.add_argument("--user", required=True)
            p.add_argument("--role", choices=("viewer", "operator"), default=None, required=name == "set-role")
        if name == "gen-cert":
            p.add_argument("--san", action="append", required=True,
                           help="subjectAltName entry, e.g. IP:10.252.219.59 or DNS:hub.example.org")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8443)
            p.add_argument("--tls-cert")
            p.add_argument("--tls-key")
            p.add_argument("--plain-http", action="store_true")
            p.add_argument("--trust-proxy", action="store_true",
                           help="take the client address from X-Forwarded-For (only behind your own proxy)")
        if name == "verify":
            p.add_argument("--device")
        if name == "backup":
            p.add_argument("--dest", required=True, type=Path,
                           help="directory on another disk, e.g. /mnt/data-ssd/keti_data/factory_safety/edge_hub_backup")
    args = parser.parse_args(argv)
    if args.command == "init":
        print(f"initialised {Store(args.data_root).root}")
        return 0
    if args.command == "add-device":
        return cmd_add_device(args)
    if args.command == "revoke-device":
        ok = Store(args.data_root).revoke_device(args.device_id)
        print("revoked" if ok else "no such device")
        return 0 if ok else 1
    if args.command == "add-viewer":
        return cmd_add_viewer(args)
    if args.command == "set-role":
        return cmd_set_role(args)
    if args.command == "gen-cert":
        return cmd_gen_cert(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "backup":
        from .backup import run_backup
        status = run_backup(args.data_root, args.dest)
        print(json.dumps(status, indent=1))
        return 0 if status["ok"] else 1
    if args.command == "reindex":
        print(f"indexed {Store(args.data_root).reindex()} units")
        return 0
    results = Store(args.data_root).verify(args.device)
    bad = [r for r in results if not r["ok"]]
    print(json.dumps({"units": len(results), "failed": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
