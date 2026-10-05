"""Opt-in real-model test against loopback OV, in a new isolated user namespace.

No real user's memory or connector content is read. Uses existing model config.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--base", default="http://127.0.0.1:1933")
args = parser.parse_args()
if urlparse(args.base).hostname not in ("127.0.0.1", "localhost"):
    raise ValueError("This smoke test only writes to local development OV")
config = json.loads(Path(args.config).read_text())
key = (
    config.get("server", {}).get("root_api_key")
    or Path.home().joinpath(".openviking/root_api_key").read_text().strip()
)
owner = str(uuid4())
client = httpx.Client(
    base_url=args.base,
    timeout=30,
    headers={"X-API-Key": key, "X-OpenViking-Account": "default", "X-OpenViking-User": owner},
)


def request(method, path, body=None):
    response = client.request(method, path, json=body)
    response.raise_for_status()
    return response.json()["result"]


created = request(
    "POST",
    "/api/v1/focuses",
    {
        "name": "Be healthier",
        "createKey": str(uuid4()),
        "userIntent": "Have more energy and protect my sleep. Do not turn this into a weight-loss goal.",
    },
)
assert created["status"] == "active"
identifier = created["focusId"]
assert request("GET", "/api/v1/focuses")["focuses"][0]["focusId"] == identifier
operation = hashlib.sha256((owner + "health-update").encode()).hexdigest()
original = "I care about feeling healthier. I tried walking after lunch with my colleague, and it gave me more energy for the rest of the day. I want to try that again. I don't remember which day it happened."
request(
    "POST",
    "/api/v1/memory-updates",
    {
        "operationId": operation,
        "text": "Use the original user's update to enrich their existing health Focus with the supported experience. Preserve their explicit intent and the unknown date. Do not create a person identity for an unnamed colleague.",
        "origin": {"conversationId": "focus-smoke", "turnId": "one", "toolCallId": "save"},
        "targets": [{"kind": "memory", "memoryUri": created["uri"]}],
        "messages": [
            {
                "id": "original",
                "role": "user",
                "sourceRef": "conversation:focus-smoke/message:original",
                "content": json.dumps(
                    {"role": "user", "parts": [{"type": "text", "text": original}]}
                ),
            }
        ],
        "sourceKinds": [],
    },
)
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    run = request("GET", "/api/v1/memory-updates/" + operation)
    if run["status"] in ("completed", "noChange", "failed", "needsRecovery", "needsClarification"):
        break
    time.sleep(2)
else:
    raise TimeoutError("Focus memory update did not finish")
updated = request("GET", "/api/v1/focuses/" + identifier)
assert run["status"] == "completed", run
assert updated["revision"] > created["revision"]
assert updated["userIntent"] == created["userIntent"]
assert updated["name"] == created["name"]
assert updated["content"] != created["content"]
edit = {
    "operationId": str(uuid4()),
    "expectedRevision": updated["revision"],
    "fields": {"status": "archived"},
}
archived = request("PATCH", "/api/v1/focuses/" + identifier, edit)
assert request("PATCH", "/api/v1/focuses/" + identifier, edit)["revision"] == archived["revision"]
assert request("GET", "/api/v1/focuses")["focuses"] == []
print(
    json.dumps(
        {
            "userId": owner,
            "focusId": identifier,
            "operationId": operation,
            "status": run["status"],
            "appliedUris": run.get("appliedUris"),
            "revision": archived["revision"],
            "content": updated["content"],
        },
        ensure_ascii=False,
    )
)
