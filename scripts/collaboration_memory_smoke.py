"""Opt-in native OV smoke test using local model config and synthetic sources.

Starts an isolated workspace/server. It never edits the running dev workspace or
reads a real Slack message. Model and embedding calls use the supplied config.
Run: .venv/bin/python scripts/collaboration_memory_smoke.py --config ov.conf
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["initial", "daily"], default="initial")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    calls = []
    secret = os.urandom(24).hex()
    user = "01000000-0000-4000-8000-000000000123"
    operation_id = hashlib.sha256(("collaboration-smoke-" + args.mode).encode()).hexdigest()
    source_ref = "collaboration:" + hashlib.sha256(b"synthetic-thread").hexdigest()
    snapshots = {}

    class Source(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if (
                self.headers.get("Authorization") != "Bearer " + secret
                or self.headers.get("X-Spark-User-Id") != user
            ):
                self.send_error(403)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            op = {
                "slackGetIdentity": "identity",
                "slackListChannels": "channels",
                "slackListUsers": "users",
                "slackGetUser": "user",
            }.get(body["tool"])
            if body["tool"] == "check":
                response = {"success": True}
            elif body["tool"] in ("readSourceSnapshot", "readCollaborationEvidence"):
                arguments = body["arguments"]
                text = snapshots.get((arguments["sourceRef"], arguments["sourceVersion"]))
                if text is None:
                    self.send_error(404)
                    return
                offset = arguments.get("offset", 0)
                end = min(len(text), offset + arguments.get("limit", 24000))
                response = {
                    "success": True,
                    "provider": "slack",
                    "workspaceId": "T_SMOKE",
                    "sourceRef": arguments["sourceRef"],
                    "sourceVersion": arguments["sourceVersion"],
                    "text": text[offset:end],
                    "nextOffset": end if end < len(text) else None,
                    "totalChars": len(text),
                }
            elif body["tool"] == "listDailyActivity":
                response = {
                    "success": True,
                    "provider": "slack",
                    "workspaceId": "T_SMOKE",
                    "entries": [
                        {
                            "kind": "slackThread",
                            "channelId": "C_STUDY",
                            "threadTs": "1790812801.000001",
                            "items": [{"excerpt": "Navigation demo passed its first test"}],
                        }
                    ],
                    "complete": True,
                    "contentComplete": False,
                    "nextCursor": None,
                }
            elif body["tool"] not in (
                "slackGetIdentity",
                "slackListChannels",
                "slackListUsers",
                "slackGetUser",
                "slackSearchMessages",
                "slackFetchHistory",
                "slackFetchThread",
            ):
                self.send_error(400)
                return
            else:
                data = {
                    "identity": {
                        "userId": "U_SELF",
                        "teamId": "T_SMOKE",
                        "team": "Aurora Study Group",
                        "user": "Alex",
                    },
                    "channels": {
                        "channels": [
                            {
                                "id": "C_STUDY",
                                "name": "study",
                                "purpose": {"value": "Prepare for the robotics exhibition"},
                            }
                        ]
                    },
                    "users": {"members": [{"id": "U_SELF", "realName": "Alex"}]},
                    "user": {"user": {"id": "U_SELF", "realName": "Alex"}},
                }.get(
                    op,
                    {
                        "messages": [
                            {
                                "ts": "1790812801.000001",
                                "user": "U_SELF",
                                "text": "We are preparing our robotics exhibition. The navigation demo passed its first test today. I will coordinate the next test.",
                            }
                        ]
                    },
                )
                version = hashlib.sha256(json.dumps(data).encode()).hexdigest()
                snapshots[(source_ref, version)] = json.dumps(data)
                response = {
                    "success": True,
                    "data": {**data, "nextCursor": None},
                    "sources": [
                        {
                            "sourceRef": source_ref,
                            "sourceVersion": version,
                            "provider": "slack",
                            "workspaceId": "T_SMOKE",
                        }
                    ],
                }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
    threading.Thread(target=source.serve_forever, daemon=True).start()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="ov-collaboration-smoke-") as directory:
            port = free_port()
            config["server"].update(host="127.0.0.1", port=port)
            config["storage"]["workspace"] = directory + "/workspace"
            config["storage"]["agfs"] = {
                "backend": "local",
                "queuefs": {"backend": "sqlite", "db_path": directory + "/queue.db"},
            }
            config["storage"]["vectordb"]["backend"] = "local"
            config.setdefault("memory", {}).update(
                source_base_url=f"http://127.0.0.1:{source.server_port}", source_api_key=secret
            )
            config["log"] = {"level": "WARNING", "output": "stdout"}
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            config_path.chmod(0o600)
            log = (Path(directory) / "server.log").open("w+")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "openviking_cli.server_bootstrap",
                    "--config",
                    str(config_path),
                ],
                stdout=log,
                stderr=log,
            )
            headers = {
                "X-OpenViking-User": user,
                "X-OpenViking-Account": "collaboration-smoke",
                "X-API-Key": config["server"].get("root_api_key", ""),
            }
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", headers=headers, timeout=45
            ) as client:
                for _ in range(90):
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if process.poll() is not None:
                        raise RuntimeError("Isolated OV server failed to start")
                    time.sleep(1)

                def api(method, path, body=None):
                    response = client.request(method, path, json=body)
                    if not response.is_success:
                        raise RuntimeError(
                            f"{method} {path}: HTTP {response.status_code}: {response.text[:1000]}"
                        )
                    return response.json()["result"]

                project = api(
                    "POST",
                    "/api/v1/projects",
                    {
                        "name": "Aurora Study Group",
                        "createKey": "smoke",
                        "source": {"provider": "slack", "workspaceId": "T_SMOKE"},
                    },
                )
                payload = {
                    "operationId": operation_id,
                    "origin": {"kind": "automation", "runId": "smoke"},
                    "text": "Explore this connected Slack workspace, learn what this project means and update its Project memory using actual source evidence."
                    if args.mode == "initial"
                    else "Browse listDailyActivity for the fixed day, read relevant full threads and update Project memory from today's evidence.",
                    "messages": [],
                    "sourceKinds": [],
                    "targets": [{"kind": "memory", "memoryUri": project["uri"]}],
                    "collaboration": {
                        "mode": args.mode,
                        "window": {
                            "start": "2026-09-30T00:00:00Z",
                            "end": "2026-10-01T00:00:00Z",
                            "timeZone": "UTC",
                        },
                        "scopes": [
                            {
                                "provider": "slack",
                                "connectionId": "ca_smoke",
                                "workspaceId": "T_SMOKE",
                                "selfId": "U_SELF",
                            }
                        ],
                        "coverage": {},
                    },
                }
                api("POST", "/api/v1/memory-updates", payload)
                for _ in range(110):
                    receipt = api("GET", "/api/v1/memory-updates/" + operation_id)
                    if receipt["status"] not in ("accepted", "running"):
                        break
                    time.sleep(3)
                assert receipt["status"] in ("completed", "needsClarification"), {
                    **{k: receipt.get(k) for k in ("status", "errors")},
                    "tools": [call["tool"] for call in calls],
                }
                assert project["uri"] in receipt["appliedUris"], receipt["appliedUris"]
                assert any(call["tool"].startswith("slack") for call in calls)
                if args.mode == "daily":
                    assert any(call["tool"] == "listDailyActivity" for call in calls)
                assert not any(call["tool"] == "readCollaboration" for call in calls)
                assert any(call["tool"] == "check" for call in calls)
                listing = api("GET", "/api/v1/projects")
                assert any(
                    p["projectId"] == project["projectId"] and p["revision"] > 1
                    for p in listing["projects"]
                )
                before = len(calls)
                replay = api("POST", "/api/v1/memory-updates", payload)
                assert replay["status"] == receipt["status"] and len(calls) == before
                print(
                    json.dumps(
                        {
                            "status": receipt["status"],
                            "appliedFiles": len(receipt["appliedUris"]),
                            "sourceCalls": len(calls),
                            "idempotentReplay": True,
                            "isolatedWorkspace": True,
                        }
                    )
                )
            process.terminate()
            process.wait(timeout=20)
    finally:
        source.shutdown()
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=20)


if __name__ == "__main__":
    main()
