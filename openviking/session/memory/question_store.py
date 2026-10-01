"""OV-owned, subject-scoped questions. Markdown is a view of structured records.

All supported writers share the per-owner lock. OpenViking's workspace process lock
makes this sufficient for its single-process storage deployment; this is not a
cross-process locking mechanism when that workspace protection is disabled.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import unicodedata
import weakref
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from openviking.core.namespace import user_space_fragment
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError

_LOCKS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalized_time(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def owner_lock(ctx):
    key = (ctx.account_id, ctx.user.user_id)
    lock = _LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[key] = lock
    return lock


def memory_root(ctx):
    return f"viking://user/{user_space_fragment(ctx)}/memories/"


def question_uri(ctx, subject):
    kind, identifier = subject.get("kind"), subject.get("id")
    if kind == "self":
        return memory_root(ctx) + "self/questions.md"
    if (
        kind not in ("person", "matter")
        or not isinstance(identifier, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", identifier)
    ):
        raise InvalidArgumentError("Invalid question subject")
    return (
        memory_root(ctx)
        + f"{'people' if kind == 'person' else 'matters'}/{identifier}/questions.md"
    )


def is_question_uri(uri, ctx):
    root = memory_root(ctx)
    return uri == root + "self/questions.md" or bool(
        re.fullmatch(
            re.escape(root) + r"(?:people|matters)/[A-Za-z0-9_-]{1,160}/questions\.md", uri
        )
    )


def topic_key(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 160:
        raise ValueError("Question topicKey must contain 1-160 characters")
    value = re.sub(r"[^\w]+", "_", unicodedata.normalize("NFKC", value).casefold()).strip("_")
    if not value:
        raise ValueError("Question topicKey must be nonempty")
    return value


def required_time(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("missing timezone")
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp with timezone") from exc
    return parsed.astimezone(timezone.utc).isoformat()


def proposal_delivery_metadata(entry):
    """Validate bounded metadata; providers validate its source-specific authority."""
    result = {}
    if "purpose" in entry:
        purpose = entry["purpose"]
        if not isinstance(purpose, str) or not 1 <= len(purpose.strip()) <= 160:
            raise ValueError("Question purpose must contain 1-160 characters")
        result["purpose"] = purpose.strip()
    if "scope" in entry:
        scope = entry["scope"]
        if not isinstance(scope, dict):
            raise ValueError("Question scope must be a JSON object")

        def check(value, depth=0):
            if depth > 8:
                raise ValueError("Question scope exceeds nesting limit")
            if isinstance(value, dict):
                if len(value) > 64 or any(
                    not isinstance(key, str) or not 1 <= len(key) <= 128 for key in value
                ):
                    raise ValueError("Question scope has too many or invalid keys")
                for child in value.values():
                    check(child, depth + 1)
            elif isinstance(value, list):
                if len(value) > 100:
                    raise ValueError("Question scope contains too many entries")
                for child in value:
                    check(child, depth + 1)
            elif value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError("Question scope must contain JSON values")

        check(scope)
        if len(json.dumps(scope, allow_nan=False, ensure_ascii=False).encode("utf-8")) > 20000:
            raise ValueError("Question scope exceeds 20000 bytes")
        result["scope"] = copy.deepcopy(scope)
    if "timing" in entry:
        timing = entry["timing"]
        if not isinstance(timing, dict) or set(timing) != {"kind", "occurredAt", "sourceRefs"}:
            raise ValueError("Invalid question timing fields")
        if timing["kind"] != "meetingEnded":
            raise ValueError("Invalid question timing kind")
        refs = timing["sourceRefs"]
        if (
            not isinstance(refs, list)
            or not refs
            or len(refs) > 20
            or any(
                not isinstance(ref, str)
                or not re.fullmatch(r"transcript:[0-9a-fA-F-]{36}", ref)
                or ref not in entry.get("sourceRefs", [])
                for ref in refs
            )
        ):
            raise ValueError("Question timing must reference supplied transcript evidence")
        result["timing"] = {
            "kind": "meetingEnded",
            "occurredAt": required_time(timing["occurredAt"], "timing.occurredAt"),
            "sourceRefs": sorted(set(refs)),
        }
    if "delivery" in entry:
        delivery = entry["delivery"]
        if not isinstance(delivery, dict) or set(delivery) - {"mode", "notBefore", "expiresAt"}:
            raise ValueError("Invalid question delivery fields")
        mode = delivery.get("mode")
        if mode not in ("contextual", "timeBound"):
            raise ValueError("Invalid question delivery mode")
        if mode == "contextual":
            if set(delivery) != {"mode"}:
                raise ValueError("Contextual delivery does not have a time window")
            result["delivery"] = {"mode": mode}
        else:
            start = required_time(delivery.get("notBefore"), "delivery.notBefore")
            end = required_time(delivery.get("expiresAt"), "delivery.expiresAt")
            if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
                raise ValueError("Question delivery window must end after it starts")
            result["delivery"] = {"mode": mode, "notBefore": start, "expiresAt": end}
    return result


def delivery_id(item):
    delivery = item.get("delivery", {})
    if delivery.get("mode") != "timeBound":
        return None
    return str(
        uuid5(
            NAMESPACE_URL,
            f"question-delivery:{item['questionId']}:{delivery['notBefore']}:{delivery['expiresAt']}",
        )
    )


def delivery_cycle_id(item):
    """Only an explicit user deferral starts a new presentation opportunity.

    New evidence, partial answers and re-extraction never renew this cycle. The
    application owns scheduling policy; this identifier only makes receipts and
    retries stable across process restarts and subject moves.
    """
    latest_deferral = next(
        (
            event["eventId"]
            for event in reversed(item.get("events", []))
            if event["action"] == "deferred"
        ),
        "initial",
    )
    return str(uuid5(NAMESPACE_URL, f"question-cycle:{item['questionId']}:{latest_deferral}"))


def validate_proposals(raw, allowed_refs):
    entries = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(entries, list) or len(entries) > 20:
        raise ValueError("Question entries must be a list with at most 20 proposals")
    normalized, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "questionId",
            "topicKey",
            "text",
            "sourceRefs",
            "relatedSubjectUris",
            "ownershipUncertain",
            "purpose",
            "scope",
        }:
            raise ValueError(
                "Invalid question proposal fields; lifecycle and timing are not extraction-controlled"
            )
        topic = topic_key(entry.get("topicKey"))
        if topic in seen:
            raise ValueError("Duplicate question topicKey within the subject")
        seen.add(topic)
        text, refs = entry.get("text"), entry.get("sourceRefs")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError("Question text must contain 1-1000 characters")
        if (
            not isinstance(refs, list)
            or not refs
            or len(refs) > 20
            or any(ref not in allowed_refs for ref in refs)
        ):
            raise ValueError("Question sourceRefs must reference supplied or read evidence")
        item = {"topicKey": topic, "text": text.strip(), "sourceRefs": sorted(set(refs))}
        if entry.get("questionId"):
            UUID(entry["questionId"])
            item["questionId"] = entry["questionId"]
        for name in ("relatedSubjectUris", "ownershipUncertain"):
            if name in entry:
                item[name] = entry[name]
        item.update(proposal_delivery_metadata(entry))
        normalized.append(item)
    return normalized


def render_questions(subject, records):
    lines = [f"# Questions — {subject['kind']}: {subject['id']}", ""]
    if subject.get("memoryUri"):
        lines.extend([f"Subject memory: {subject['memoryUri']}", ""])
    for item in records:
        lines.extend(
            [
                f"## {item['text']}",
                f"- ID: {item['questionId']}",
                f"- Topic: {item['topicKey']}",
                f"- State: {item['state']}",
            ]
        )
        if item.get("purpose"):
            lines.append(f"- Purpose: {item['purpose']}")
        if item.get("scope"):
            lines.append("- Scope: " + json.dumps(item["scope"], ensure_ascii=False))
        if item.get("timing"):
            lines.append("- Timing evidence: " + json.dumps(item["timing"], ensure_ascii=False))
        delivery = item.get("delivery", {})
        if delivery.get("mode") == "timeBound":
            lines.append(f"- Delivery window: {delivery['notBefore']} to {delivery['expiresAt']}")
        if item.get("notBefore"):
            lines.append(f"- Do not ask before: {item['notBefore']}")
        if item.get("ownershipUncertain"):
            lines.append("- Subject assignment needs clarification.")
        for ref in item.get("sourceRefs", []):
            lines.append(f"- Evidence: {ref}")
        for uri in item.get("relatedSubjectUris", []):
            lines.append(f"- Related subject: {uri}")
        for event in item.get("events", []):
            lines.extend(
                [
                    f"- {event['action']} at {event['at']} ({event['sourceRef']}):",
                    "  " + event["evidenceText"].replace("\n", "\n  "),
                ]
            )
        if item.get("resolution"):
            lines.append(f"- Resolution: {item['resolution']}")
        if item.get("propagation"):
            lines.append(f"- Memory propagation: {item['propagation']['status']}")
        lines.append("")
    return "\n".join(lines)


class QuestionStore:
    def __init__(self, viking_fs, ctx, vikingdb=None):
        self.fs, self.ctx, self.db = viking_fs, ctx, vikingdb

    async def _load_page(self, uri):
        if not is_question_uri(uri, self.ctx):
            raise InvalidArgumentError("Question URI is outside the current user's question pages")
        try:
            raw = await self.fs.read_file(uri, ctx=self.ctx)
        except NotFoundError:
            return None
        page = MemoryFileUtils.read(raw, uri=uri)
        if page.extra_fields.get("questionFormatVersion") != 1:
            raise InvalidArgumentError("Unsupported question page format")
        return page

    async def _pages(self):
        # No persistent global index. The native tree is bounded to these three levels.
        root = memory_root(self.ctx).rstrip("/")
        try:
            entries = await self.fs.tree(root, ctx=self.ctx, node_limit=10000, level_limit=4)
        except NotFoundError:
            return []
        pages = []
        for entry in entries:
            uri = entry.get("uri", "")
            if is_question_uri(uri, self.ctx):
                page = await self._load_page(uri)
                if page is not None:
                    pages.append(page)
        return pages

    def _public(self, page, item):
        return {
            **{key: copy.deepcopy(value) for key, value in item.items() if key != "wordingHistory"},
            "questionUri": page.uri,
            "subject": copy.deepcopy(page.extra_fields["subject"]),
            "revision": page.extra_fields.get("revision", 0),
            "deliveryCycleId": delivery_cycle_id(item),
            **({"deliveryId": delivery_id(item)} if delivery_id(item) else {}),
        }

    async def _recover_moves(self):
        directory = memory_root(self.ctx) + ".question-moves"
        try:
            entries = await self.fs.ls(
                directory, ctx=self.ctx, show_all_hidden=True, node_limit=1000
            )
        except NotFoundError:
            return
        for entry in entries:
            uri = entry.get("uri", "")
            if uri.startswith(directory + "/") and uri.endswith(".json"):
                await self._finish_move(uri, json.loads(await self.fs.read_file(uri, ctx=self.ctx)))

    async def _finish_move(self, journal_uri, move):
        """Replay a small write-ahead move record before serving any questions."""
        source = await self._load_page(move["sourceUri"])
        target = await self._load_page(move["targetUri"])
        if target is None:
            target = MemoryFile(
                uri=move["targetUri"],
                memory_type="questions",
                extra_fields={
                    "questionFormatVersion": 1,
                    "subject": move["subject"],
                    "questions": [],
                    "revision": 0,
                },
            )
        question_id = move["questionId"]
        # No supported writer can access either copy before this replay completes.
        # The journal contains the complete pre-move record, including all answers.
        record = copy.deepcopy(move["question"])
        target.extra_fields["questions"] = [
            q for q in target.extra_fields["questions"] if q["questionId"] != question_id
        ] + [record]
        await self._save(target)
        if source:
            source.extra_fields["questions"] = [
                q for q in source.extra_fields["questions"] if q["questionId"] != question_id
            ]
            await self._save(source)
        await self.refresh(target.uri)
        if source:
            await self.refresh(source.uri)
        await self.fs.rm(journal_uri, ctx=self.ctx)

    async def move_person(self, from_anchor, to_anchor, target_memory_uri):
        """Identity merges move question ownership without resetting lifecycle."""
        subject = {"kind": "person", "id": to_anchor, "memoryUri": target_memory_uri}
        target_uri = question_uri(self.ctx, subject)
        async with owner_lock(self.ctx):
            await self._recover_moves()
            source = await self._load_page(question_uri(self.ctx, {"kind": "person", "id": from_anchor}))
            if not source:
                return
            for item in list(source.extra_fields.get("questions", [])):
                target = await self._load_page(target_uri)
                record = copy.deepcopy(item)
                if target and any(q["topicKey"] == record["topicKey"] and q["questionId"] != record["questionId"] for q in target.extra_fields["questions"]):
                    record["topicKey"] = from_anchor + ":" + record["topicKey"]
                journal_uri = memory_root(self.ctx) + ".question-moves/" + record["questionId"] + ".json"
                move = {"questionId": record["questionId"], "sourceUri": source.uri,
                        "targetUri": target_uri, "subject": subject, "question": record}
                await self.fs.write_file(journal_uri, json.dumps(move, ensure_ascii=False), ctx=self.ctx)
                await self._finish_move(journal_uri, move)

    async def _find(self, question_id):
        UUID(question_id)
        await self._recover_moves()
        for page in await self._pages():
            for item in page.extra_fields.get("questions", []):
                if item["questionId"] == question_id:
                    return page, item
        raise NotFoundError(question_id, "question")

    async def _save(self, page):
        page.content = render_questions(
            page.extra_fields["subject"], page.extra_fields["questions"]
        )
        page.extra_fields["revision"] = page.extra_fields.get("revision", 0) + 1
        await self.fs.write_file(page.uri, MemoryFileUtils.write(page), ctx=self.ctx)

    async def refresh(self, uri):
        if self.db is not None:
            from openviking.session.memory.memory_updater import MemoryUpdater

            await MemoryUpdater.refresh_file_embedding(
                viking_fs=self.fs, vikingdb=self.db, uri=uri, memory_type="questions", ctx=self.ctx
            )

    async def discover(self, uri, subject, proposals, metadata=None):
        """Native extraction can merge evidence; it cannot mutate user lifecycle."""
        if question_uri(self.ctx, subject) != uri:
            raise InvalidArgumentError("Question subject and URI disagree")
        proposals = [{**proposal, **proposal_delivery_metadata(proposal)} for proposal in proposals]
        async with owner_lock(self.ctx):
            await self._recover_moves()
            page = await self._load_page(uri)
            if page is None:
                page = MemoryFile(
                    uri=uri,
                    memory_type="questions",
                    extra_fields={
                        "questionFormatVersion": 1,
                        "subject": subject,
                        "questions": [],
                        "revision": 0,
                    },
                )
            records = page.extra_fields["questions"]
            for proposal in proposals:
                question_id = proposal.get("questionId")
                item = (
                    next((q for q in records if q["questionId"] == question_id), None)
                    if question_id
                    else None
                )
                if question_id and item is None:
                    old_page, old_item = await self._find(question_id)
                    journal_uri = memory_root(self.ctx) + ".question-moves/" + question_id + ".json"
                    move = {
                        "questionId": question_id,
                        "sourceUri": old_page.uri,
                        "targetUri": uri,
                        "subject": subject,
                        "question": old_item,
                    }
                    await self.fs.write_file(
                        journal_uri, json.dumps(move, ensure_ascii=False), ctx=self.ctx
                    )
                    await self._finish_move(journal_uri, move)
                    page = await self._load_page(uri)
                    records = page.extra_fields["questions"]
                    item = next(q for q in records if q["questionId"] == question_id)
                if item is None:
                    item = next((q for q in records if q["topicKey"] == proposal["topicKey"]), None)
                if item is None:
                    item = {
                        "questionId": str(uuid4()),
                        "topicKey": proposal["topicKey"],
                        "text": proposal["text"],
                        "sourceRefs": [],
                        "state": "open",
                        "createdAt": now_iso(),
                        "updatedAt": now_iso(),
                        "events": [],
                        "answers": [],
                    }
                    records.append(item)
                # Source identity and its first timing fact survive rediscovery,
                # subject moves, retries and answers. Never refresh a meeting's time
                # to "now" merely because its input was processed late. Preserve
                # old delivery windows for legacy readers until they are migrated.
                for name in ("purpose", "scope", "timing", "delivery"):
                    if name in proposal and name not in item:
                        item[name] = copy.deepcopy(proposal[name])
                # User answers/states survive both stale snapshots and rewording.
                if item["state"] == "open":
                    if item["text"] != proposal["text"]:
                        # A queued canonical delivery may still display an earlier
                        # wording from this cycle. Keep that exact evidence in OV
                        # so its delayed receipt remains verifiable after refresh.
                        history = item.setdefault("wordingHistory", [])
                        if item["text"] not in history:
                            history.append(item["text"])
                    item["text"] = proposal["text"]
                item["sourceRefs"] = sorted(set(item["sourceRefs"]) | set(proposal["sourceRefs"]))
                item["relatedSubjectUris"] = sorted(
                    set(item.get("relatedSubjectUris", []))
                    | set(proposal.get("relatedSubjectUris", []))
                )
                item["ownershipUncertain"] = bool(
                    proposal.get("ownershipUncertain", item.get("ownershipUncertain", False))
                )
                item["updatedAt"] = now_iso()
            for key, value in (metadata or {}).items():
                if key.startswith(("email_", "meeting_")):
                    page.extra_fields[key] = value
            await self._save(page)
            return [self._public(page, item) for item in records]

    async def get(self, question_id):
        async with owner_lock(self.ctx):
            page, item = await self._find(question_id)
            return self._public(page, item)

    async def list(self):
        async with owner_lock(self.ctx):
            await self._recover_moves()
            return [
                self._public(page, q)
                for page in await self._pages()
                for q in page.extra_fields["questions"]
            ]

    async def candidates(
        self, *, limit=3, related_subject_ids=(), conversation_id=None, recent_text=""
    ):
        now = now_iso()
        records = await self.list()
        eligible = []
        terms = set(re.findall(r"\w{3,}", recent_text.casefold()))
        for item in records:
            if item.get("delivery", {}).get("mode") == "timeBound":
                continue
            if item["state"] in ("resolved", "dismissed") or (
                item.get("notBefore") and item["notBefore"] > now
            ):
                continue
            if item["state"] == "asked":
                asked = [e for e in item["events"] if e["action"] == "asked"]
                if not asked or asked[-1]["conversationId"] != conversation_id:
                    continue
            score = int(item["subject"]["id"] in related_subject_ids) * 10 + int(
                item["subject"]["kind"] == "self"
            )
            score += len(terms & set(re.findall(r"\w{3,}", item["text"].casefold())))
            eligible.append((score, item))
        eligible.sort(key=lambda pair: (-pair[0], pair[1]["createdAt"], pair[1]["questionId"]))
        return [item for _, item in eligible[:limit]]

    async def due(self, *, limit=3):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise InvalidArgumentError("Question due limit must be between 1 and 20")
        now = datetime.fromisoformat(now_iso())
        eligible = []
        for item in await self.list():
            delivery = item.get("delivery", {})
            if delivery.get("mode") != "timeBound" or item["state"] not in ("open", "deferred"):
                continue
            start = datetime.fromisoformat(delivery["notBefore"])
            end = datetime.fromisoformat(delivery["expiresAt"])
            if item.get("notBefore"):
                start = max(start, datetime.fromisoformat(item["notBefore"]))
            already_delivered = any(
                event.get("deliveryReceipt", {}).get("deliveryId") == item["deliveryId"]
                for event in item["events"]
            )
            if start <= now < end and not already_delivered:
                eligible.append(item)
        eligible.sort(key=lambda q: (q["delivery"]["expiresAt"], q["createdAt"], q["questionId"]))
        return eligible[:limit]

    def _delivery_receipt(self, item, data):
        receipt = data.get("deliveryReceipt")
        if not isinstance(receipt, dict) or set(receipt) != {
            "deliveryId",
            "channel",
            "receivedAt",
            "messageId",
        }:
            raise InvalidArgumentError(
                "Asked delivery events require a visible-device delivery receipt"
            )
        if receipt["channel"] not in ("glasses", "phone"):
            raise InvalidArgumentError("Invalid delivery receipt channel")
        if receipt["messageId"] != data["messageId"] or not receipt["messageId"]:
            raise InvalidArgumentError("Delivery receipt message does not match the question event")
        try:
            at = required_time(receipt["receivedAt"], "deliveryReceipt.receivedAt")
        except ValueError as exc:
            raise InvalidArgumentError(str(exc)) from exc
        received = datetime.fromisoformat(at)
        if received > datetime.fromisoformat(now_iso()):
            raise InvalidArgumentError("Delivery receipt cannot be in the future")
        return {**receipt, "receivedAt": at}

    async def record(self, data):
        action = data["action"]
        role = data["evidenceRole"]
        if role != ("assistant" if action == "asked" else "user"):
            raise InvalidArgumentError("Question event role does not match action")
        async with owner_lock(self.ctx):
            page, item = await self._find(data["questionId"])
            receipt = None
            if action == "asked" and (
                data.get("deliveryReceipt") is not None
                or item.get("delivery", {}).get("mode") == "timeBound"
            ):
                receipt = self._delivery_receipt(item, data)
                delivered = next(
                    (
                        e
                        for e in item["events"]
                        if e.get("deliveryReceipt", {}).get("deliveryId") == receipt["deliveryId"]
                    ),
                    None,
                )
                if delivered:
                    if delivered["messageId"] != data["messageId"]:
                        raise InvalidArgumentError(
                            "This delivery was already received as another message"
                        )
                    return {"question": self._public(page, item), "duplicate": True}
                # A retry of an already-recorded cycle is safe even after an answer
                # or another deferral. A new receipt must belong to the current
                # cycle; old cycles can never reopen a resolved/dismissed question.
                if receipt["deliveryId"] == delivery_cycle_id(item):
                    # Application policy chooses when/where to present. OV checks
                    # user intent below, but does not impose a meeting deadline.
                    pass
                elif receipt["deliveryId"] == delivery_id(item):
                    # Compatibility for callers still using the original bounded
                    # delivery ID. Do not weaken their historical window contract.
                    delivery = item["delivery"]
                    received = datetime.fromisoformat(receipt["receivedAt"])
                    start = datetime.fromisoformat(delivery["notBefore"])
                    if not start <= received < datetime.fromisoformat(delivery["expiresAt"]):
                        raise InvalidArgumentError(
                            "Delivery receipt is outside the question window"
                        )
                else:
                    raise InvalidArgumentError(
                        "Delivery receipt does not match this question cycle"
                    )
                if item.get("notBefore") and datetime.fromisoformat(
                    receipt["receivedAt"]
                ) < datetime.fromisoformat(item["notBefore"]):
                    raise InvalidArgumentError("Delivery receipt predates the question deferral")
            event_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"question-event:{item['questionId']}:{data['conversationId']}:{data['messageId']}",
                )
            )
            existing = next((e for e in item["events"] if e["eventId"] == event_id), None)
            if existing:
                if existing["action"] != action:
                    raise InvalidArgumentError(
                        "The message already has a different question action"
                    )
                return {"question": self._public(page, item), "duplicate": True}
            if action == "asked":

                def normalize(text):
                    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())

                valid_wordings = [item["text"]]
                if receipt:
                    valid_wordings.extend(item.get("wordingHistory", []))
                if not any(
                    normalize(text) in normalize(data["evidenceText"]) for text in valid_wordings
                ):
                    raise InvalidArgumentError("Asked evidence must contain the actual question")
                if item["state"] in ("asked", "resolved", "dismissed") or (
                    not receipt and item.get("notBefore") and item["notBefore"] > now_iso()
                ):
                    raise InvalidArgumentError("Question is not currently askable")
            at = (
                receipt["receivedAt"]
                if receipt
                else normalized_time(data.get("evidenceAt")) or now_iso()
            )
            source_ref = f"conversation:{data['conversationId']}/message:{data['messageId']}"
            event = {
                "eventId": event_id,
                "action": action,
                "evidenceText": data["evidenceText"],
                "conversationId": data["conversationId"],
                "messageId": data["messageId"],
                "sourceRef": source_ref,
                "at": at,
                "turnId": data.get("turnId"),
            }
            if data.get("confirmedSpeakerAssignment") is not None:
                if action != "resolved" or item.get("purpose") != "speakerIdentity":
                    raise InvalidArgumentError(
                        "Speaker confirmation requires a resolved speaker identity question"
                    )
                spec = await self._meeting_spec(item)
                mapping = data["confirmedSpeakerAssignment"]
                if not isinstance(mapping, dict) or set(mapping) - {
                    "personId",
                    "personMemoryUri",
                    "isSelf",
                }:
                    raise InvalidArgumentError("Invalid confirmed speaker assignment")
                is_self = mapping.get("isSelf") is True
                person = next(
                    (
                        p
                        for p in spec.people
                        if p.personId == mapping.get("personId")
                        or p.personMemoryUri == mapping.get("personMemoryUri")
                    ),
                    None,
                )
                uri = person.personMemoryUri if person else mapping.get("personMemoryUri")
                if is_self and (uri or mapping.get("personId")):
                    raise InvalidArgumentError("Self mapping cannot identify another person")
                if not is_self:
                    if not isinstance(uri, str) or not re.fullmatch(
                        re.escape(memory_root(self.ctx)) + r"people/[A-Za-z0-9_-]+\.md", uri
                    ):
                        raise InvalidArgumentError(
                            "Speaker confirmation requires an existing same-user person or frozen candidate"
                        )
                    if person and (
                        (mapping.get("personId") and mapping["personId"] != person.personId)
                        or (mapping.get("personMemoryUri") and mapping["personMemoryUri"] != uri)
                    ):
                        raise InvalidArgumentError("Confirmed contact and person anchor disagree")
                    if not person:
                        await self.fs.read_file(uri, ctx=self.ctx)
                scope = item["scope"]
                confirmed = {
                    "speakerRef": scope["speakerRef"],
                    "sourceRef": scope["sourceRef"],
                    "sourceVersion": scope["sourceVersion"],
                    "startMs": scope["startMs"],
                    "endMs": scope["endMs"],
                    "personMemoryUri": uri,
                    "personId": person.personId if person else None,
                    "isSelf": is_self,
                    "status": "confirmed",
                    "questionId": item["questionId"],
                    "answerSourceRef": source_ref,
                }
                event["confirmedSpeakerAssignment"] = confirmed
                item["confirmedSpeakerAssignment"] = confirmed
            if (
                action == "resolved"
                and item.get("purpose") == "speakerIdentity"
                and data.get("confirmedSpeakerAssignment") is None
            ):
                revoked = item.pop("confirmedSpeakerAssignment", None)
                if revoked:
                    event["revokedSpeakerAssignment"] = revoked
            if data.get("notBefore") is not None and action in ("deferred", "partial"):
                try:
                    event["notBefore"] = required_time(data["notBefore"], "notBefore")
                except ValueError as exc:
                    raise InvalidArgumentError(str(exc)) from exc
            if receipt:
                event["deliveryReceipt"] = receipt
            item["events"].append(event)
            item["updatedAt"] = now_iso()
            if action != "asked":
                item["answers"].append(
                    {
                        "text": data["evidenceText"],
                        "sourceRef": source_ref,
                        "conversationId": data["conversationId"],
                        "messageId": data["messageId"],
                        "at": at,
                        "action": action,
                    }
                )
            item["state"] = {
                "asked": "asked",
                "resolved": "resolved",
                "deferred": "deferred",
                "dismissed": "dismissed",
                "partial": "deferred",
            }[action]
            if action in ("deferred", "partial"):
                item["notBefore"] = (
                    event.get("notBefore")
                    or (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
                )
            else:
                item.pop("notBefore", None)
            if action == "resolved":
                item["resolution"] = data.get("resolution") or data["evidenceText"]
                item["propagation"] = {"eventId": event_id, "status": "pending"}
            await self._save(page)
            public = self._public(page, item)
        await self.refresh(page.uri)
        return {"question": public, "duplicate": False}

    async def _meeting_spec(self, item):
        from openviking.session.memory.meeting_context import MeetingContext

        scope = item.get("scope", {})
        uri = scope.get("meetingContextUri", "")
        expected_prefix = f"viking://user/{user_space_fragment(self.ctx)}/sessions/"
        if (
            not isinstance(uri, str)
            or not uri.startswith(expected_prefix)
            or not uri.endswith("/meeting_context.json")
            or any(x in uri for x in ("..", "%", "?", "#", "\\"))
        ):
            raise InvalidArgumentError("Speaker question lost its frozen meeting context")
        spec = MeetingContext.model_validate(json.loads(await self.fs.read_file(uri, ctx=self.ctx)))
        spec.validate_owner(self.ctx)
        if (
            scope.get("sourceRef") != spec.sourceRef
            or scope.get("sourceVersion") != spec.sourceVersion
            or scope.get("meetingMemoryUri") != spec.meetingMemoryUri
        ):
            raise InvalidArgumentError("Speaker question does not match its frozen recording")
        if not any(
            w.identity_start == scope.get("startMs") and w.identity_end == scope.get("endMs")
            for w in spec.mediaWindows
        ):
            raise InvalidArgumentError("Speaker question lost its meeting window")
        return spec

    async def propagate(self, question_id, sessions):
        """Persist the answer first, then submit/reconcile one native extraction session."""
        from openviking.message import TextPart

        async with owner_lock(self.ctx):
            page, item = await self._find(question_id)
            propagation = item.get("propagation")
            if not propagation or propagation["status"] == "done":
                return self._public(page, item)
            session_id = (
                "question-" + uuid5(NAMESPACE_URL, f"{question_id}:{propagation['eventId']}").hex
            )
            propagation["sessionId"] = session_id
            session = await sessions.get(session_id, self.ctx, auto_create=True)
            session.meta.question_context = {
                "subject": page.extra_fields["subject"],
                "questionId": question_id,
            }
            meeting_spec = None
            if item.get("purpose") == "speakerIdentity":
                from openviking.session.memory.meeting_context import meeting_memory_policy

                meeting_spec = await self._meeting_spec(item)
                scope = item["scope"]
                meeting_spec.confirmedAssignments = [
                    a
                    for a in meeting_spec.confirmedAssignments
                    if (a.get("speakerRef"), a.get("startMs"), a.get("endMs"))
                    != (scope["speakerRef"], scope["startMs"], scope["endMs"])
                ]
                if item.get("confirmedSpeakerAssignment"):
                    mapping = item["confirmedSpeakerAssignment"]
                    meeting_spec.confirmedAssignments = [
                        a
                        for a in meeting_spec.confirmedAssignments
                        if (a.get("speakerRef"), a.get("startMs"), a.get("endMs"))
                        != (mapping["speakerRef"], mapping["startMs"], mapping["endMs"])
                    ] + [mapping]
                session.meta.meeting_context = meeting_spec.model_dump()
                session.meta.memory_policy = meeting_memory_policy()
            await session._save_meta()
            archive_uri = propagation.get("archiveUri") or f"{session.uri}/history/archive_001"
            try:
                await self.fs.read_file(archive_uri + "/.done", ctx=self.ctx)
                propagation["status"] = "done"
            except NotFoundError:
                try:
                    failure = await self.fs.read_file(archive_uri + "/.failed.json", ctx=self.ctx)
                except NotFoundError:
                    failure = None
                task = (
                    await sessions.get_commit_task(propagation["taskId"], self.ctx)
                    if propagation.get("taskId")
                    else None
                )
                if task and task["status"] in ("pending", "running"):
                    propagation["status"] = "submitted"
                elif failure:
                    if meeting_spec is not None:
                        committed = await session.retry_meeting_archive(
                            archive_uri.rsplit("/", 1)[-1]
                        )
                        propagation.update(
                            status="submitted",
                            taskId=committed.get("task_id"),
                            archiveUri=committed.get("archive_uri"),
                        )
                    else:
                        propagation.update(status="failed", error=str(failure)[:2000])
                else:
                    try:
                        await self.fs.read_file(archive_uri + "/messages.jsonl", ctx=self.ctx)
                    except NotFoundError:
                        # No archive means no native write was started. A fixed session
                        # and owner lock allow safe retry after response loss here.
                        if not session.messages:
                            if meeting_spec is not None:
                                from openviking.message import Message

                                original_archive = item["scope"]["meetingContextUri"].rsplit(
                                    "/", 1
                                )[0]
                                raw_messages = await self.fs.read_file(
                                    original_archive + "/messages.jsonl", ctx=self.ctx
                                )
                                for line in raw_messages.splitlines():
                                    if line.strip():
                                        original = Message.from_dict(json.loads(line))
                                        session.add_message(original.role, original.parts)
                            event = next(
                                e for e in item["events"] if e["eventId"] == propagation["eventId"]
                            )
                            payload = json.dumps(
                                {
                                    "questionUri": page.uri,
                                    "subject": page.extra_fields["subject"],
                                    "question": item["text"],
                                    "userAnswer": event["evidenceText"],
                                    "sourceRef": event["sourceRef"],
                                    "meetingScope": item.get("scope"),
                                    "confirmedSpeakerAssignment": item.get(
                                        "confirmedSpeakerAssignment"
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            session.add_message(
                                "user",
                                [
                                    TextPart(
                                        "A user has explicitly clarified this memory question. Update relevant user/person/project memory using the actual answer, including a negative answer when it rejects a previous hypothesis. Do not infer new facts from refusal. The original question record already stores this answer; do not edit its question page.\n"
                                        + payload
                                    )
                                ],
                            )
                            await session._write_to_agfs_async(messages=session.messages)
                        propagation["status"] = "pending"
                        await self._save(page)
                        committed = await session.commit_async()
                        propagation.update(
                            status="submitted",
                            taskId=committed.get("task_id"),
                            archiveUri=committed.get("archive_uri"),
                        )
                    else:
                        # The archive exists but its worker is unknown after a restart.
                        # Never create a second answer session or silently mark success.
                        if meeting_spec is not None:
                            try:
                                committed = await session.retry_meeting_archive(
                                    archive_uri.rsplit("/", 1)[-1]
                                )
                                propagation.update(
                                    status="submitted",
                                    taskId=committed.get("task_id"),
                                    archiveUri=committed.get("archive_uri"),
                                )
                            except InvalidArgumentError:
                                propagation.update(status="unknown", archiveUri=archive_uri)
                        else:
                            propagation.update(status="unknown", archiveUri=archive_uri)
            await self._save(page)
            return self._public(page, item)
