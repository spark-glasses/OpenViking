"""Evidence-backed structured profile observations, delivered to PeopleService.

The outbox contains observations, not a second question store. Conflicts are
recorded by QuestionStore in OV and never overwrite a user's correction.
"""

import hashlib
import json
from uuid import UUID
from datetime import datetime, timezone
from openviking.session.memory.person_paths import person_anchor_from_uri
from openviking.session.memory.person_identity import load_person_identity, memory_root
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import NotFoundError

FIELDS = {"organization", "jobTitle", "notes"}


def receipt_uri(ctx, proposal_id):
    return memory_root(ctx) + "people/.profile-receipts/" + proposal_id + ".json"


def queue_root(ctx):
    return memory_root(ctx) + "people/.profile-proposals/"


async def collect_profile_proposals(fs, ctx, result, provider, archive_uri):
    if not archive_uri:
        return
    # Only actually supplied conversation/source text can support a new profile
    # fact. The application's projected profile is not independent evidence.
    evidence = provider.get_conversation_text()
    for call in getattr(provider, "evidence", []):
        if call.get("tool") in ("readEmail", "readTranscript", "readSource", "readContext"):
            evidence += "\n" + json.dumps(call.get("result"), ensure_ascii=False)
    for uri in set(result.written_uris + result.edited_uris):
        if person_anchor_from_uri(uri, memory_root(ctx)) is None:
            continue
        anchor = person_anchor_from_uri(uri, memory_root(ctx))
        try:
            UUID(anchor)
        except ValueError:
            continue
        identity = await load_person_identity(fs, ctx, anchor)
        if not identity or identity.get("deleted"):
            continue
        page = MemoryFileUtils.read(await fs.read_file(uri, ctx=ctx), uri=uri)
        raw = page.extra_fields.get("profileFacts", "[]")
        try:
            facts = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(facts, list) or len(facts) > 10:
                continue
            for fact in facts:
                if not isinstance(fact, dict) or fact.get("field") not in FIELDS:
                    continue
                value, quote = fact.get("value"), fact.get("evidenceQuote")
                if not isinstance(value, str) or not value.strip() or len(value) > 2000:
                    continue
                if not isinstance(quote, str) or len(quote) < 4 or quote not in evidence:
                    continue
                digest = hashlib.sha256(
                    json.dumps(
                        [archive_uri, anchor, fact], sort_keys=True, ensure_ascii=False
                    ).encode()
                ).hexdigest()
                target = queue_root(ctx) + digest + ".json"
                try:
                    await fs.read_file(receipt_uri(ctx, digest), ctx=ctx)
                    continue
                except NotFoundError:
                    pass
                try:
                    await fs.read_file(target, ctx=ctx)
                    continue
                except NotFoundError:
                    pass
                observation = {
                    "id": digest,
                    "personId": identity["personId"],
                    "anchorId": anchor,
                    "memoryUri": uri,
                    "field": fact["field"],
                    "value": value,
                    "evidenceQuote": quote,
                    "sourceRefs": [archive_uri],
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "status": "pending",
                }
                await fs.write_file(target, json.dumps(observation, ensure_ascii=False), ctx=ctx)
        except (ValueError, TypeError):
            continue


async def pending(fs, ctx):
    try:
        entries = await fs.ls(queue_root(ctx), ctx=ctx, show_all_hidden=True, node_limit=10000)
    except NotFoundError:
        return []
    result = []
    for entry in entries:
        uri = entry.get("uri", "")
        if uri.startswith(queue_root(ctx)) and uri.endswith(".json"):
            record = json.loads(await fs.read_file(uri, ctx=ctx))
            if record.get("status") == "pending":
                result.append(record)
    return sorted(result, key=lambda record: (record["createdAt"], record["id"]))[:20]
