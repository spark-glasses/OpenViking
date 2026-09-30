"""Observable ownership, persistence, lifecycle and concurrent-write guarantees."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier


class FS:
    def __init__(self):
        self.files = {}

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        await asyncio.sleep(0)
        self.files[uri] = content

    async def ls(self, uri, **kwargs):
        return [
            {"uri": key}
            for key in self.files
            if key.startswith(uri + "/") and "/" not in key[len(uri) + 1 :]
        ]

    async def rm(self, uri, **kwargs):
        self.files.pop(uri, None)

    async def tree(self, uri, **kwargs):
        return [{"uri": key} for key in self.files if key.startswith(uri + "/")]


@pytest.fixture
def setup():
    fs = FS()
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    return fs, ctx, QuestionStore(fs, ctx)


def proposal(**changes):
    return {
        "topicKey": "user_alias_junkuan",
        "text": "Do you also use Junkuan?",
        "sourceRefs": ["email:00000000-0000-4000-8000-000000000001"],
        **changes,
    }


async def create(store, ctx, subject=None):
    subject = subject or {"kind": "self", "id": "self"}
    return (await store.discover(question_uri(ctx, subject), subject, [proposal()]))[0]


def event(q, action, **changes):
    return {
        "questionId": q["questionId"],
        "action": action,
        "evidenceText": q["text"]
        if action == "asked"
        else "No, that greeting was for another person.",
        "conversationId": "chat-1",
        "messageId": "asked-1" if action == "asked" else "answer-1",
        "evidenceRole": "assistant" if action == "asked" else "user",
        **changes,
    }


@pytest.mark.asyncio
async def test_subject_pages_have_distinct_ids_and_single_canonical_structure(setup):
    fs, ctx, store = setup
    records = [
        await create(store, ctx, subject)
        for subject in (
            {"kind": "self", "id": "self"},
            {"kind": "person", "id": "anchor-1"},
            {"kind": "matter", "id": "matter-1"},
        )
    ]
    assert len({q["questionId"] for q in records}) == 3
    assert records[0]["questionUri"].endswith("/self/questions.md")
    for q in records:
        memory = MemoryFileUtils.read(fs.files[q["questionUri"]])
        assert memory.extra_fields["questions"][0]["text"] in memory.content
        assert "entries" not in memory.extra_fields
        assert q["state"] == "open"


@pytest.mark.asyncio
async def test_answer_persists_across_new_store_and_stale_extraction_cannot_reset_it(setup):
    fs, ctx, store = setup
    q = await create(store, ctx)
    await store.record(event(q, "asked"))
    await store.record(event(q, "resolved"))
    fresh = QuestionStore(fs, ctx)
    await fresh.discover(
        q["questionUri"],
        q["subject"],
        [
            proposal(
                text="Is Junkuan another name?",
                sourceRefs=["email:00000000-0000-4000-8000-000000000002"],
            )
        ],
    )
    actual = await fresh.get(q["questionId"])
    assert actual["state"] == "resolved"
    assert actual["answers"][0]["text"].startswith("No,")
    assert len(actual["sourceRefs"]) == 2
    assert actual["propagation"]["status"] == "pending"
    assert await fresh.candidates(conversation_id="chat-1") == []


@pytest.mark.asyncio
async def test_candidate_is_not_asked_and_duplicate_message_is_idempotent(setup):
    _, ctx, store = setup
    q = await create(store, ctx)
    await store.candidates()
    assert (await store.get(q["questionId"]))["events"] == []
    asked = event(q, "asked")
    await store.record(asked)
    assert (await store.record(asked))["duplicate"] is True
    with pytest.raises(InvalidArgumentError):
        await store.record(event(q, "asked", messageId="another-ask", conversationId="chat-2"))
    assert await store.candidates(conversation_id="chat-2") == []
    assert len(await store.candidates(conversation_id="chat-1")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,state",
    [
        ("resolved", "resolved"),
        ("dismissed", "dismissed"),
        ("partial", "deferred"),
        ("deferred", "deferred"),
    ],
)
async def test_spontaneous_answers_and_refusals_are_distinct(setup, action, state):
    _, ctx, store = setup
    q = await create(store, ctx)
    result = await store.record(event(q, action))
    assert result["question"]["state"] == state
    assert len(result["question"]["answers"]) == 1
    assert ("propagation" in result["question"]) is (action == "resolved")


@pytest.mark.asyncio
async def test_defer_normalizes_timezone_and_validates_evidence_role(setup):
    _, ctx, store = setup
    q = await create(store, ctx)
    result = await store.record(event(q, "deferred", notBefore="2027-01-01T01:00:00-08:00"))
    assert result["question"]["notBefore"] == "2027-01-01T09:00:00+00:00"
    with pytest.raises(InvalidArgumentError):
        await store.record(event(q, "resolved", messageId="x", evidenceRole="assistant"))


@pytest.mark.asyncio
async def test_concurrent_answer_and_native_discovery_preserve_both(setup):
    _, ctx, store = setup
    q = await create(store, ctx)
    await asyncio.gather(
        store.record(event(q, "resolved")),
        store.discover(
            q["questionUri"],
            q["subject"],
            [proposal(sourceRefs=["email:00000000-0000-4000-8000-000000000002"])],
        ),
    )
    actual = await store.get(q["questionId"])
    assert actual["state"] == "resolved"
    assert len(actual["answers"]) == 1
    assert len(actual["sourceRefs"]) == 2


@pytest.mark.asyncio
async def test_cross_owner_access_and_move_preserve_answer_and_identity(setup):
    fs, ctx, store = setup
    q = await create(store, ctx)
    await store.record(event(q, "resolved"))
    other_ctx = RequestContext(user=UserIdentifier("account", "bob"), role=Role.ROOT)
    with pytest.raises(NotFoundError):
        await QuestionStore(fs, other_ctx).get(q["questionId"])
    target = {"kind": "person", "id": "anchor"}
    await store.discover(question_uri(ctx, target), target, [proposal(questionId=q["questionId"])])
    actual = await store.get(q["questionId"])
    assert actual["subject"] == target
    assert actual["state"] == "resolved"
    assert len(actual["answers"]) == 1
    assert len(await store.list()) == 1


@pytest.mark.asyncio
async def test_interrupted_move_repairs_duplicate_before_serving_records(setup):
    fs, ctx, store = setup
    q = await create(store, ctx)
    await store.record(event(q, "resolved"))
    original_write = fs.write_file

    async def fail_source(uri, content, **kwargs):
        if uri == q["questionUri"]:
            raise OSError("simulated source removal failure")
        await original_write(uri, content, **kwargs)

    fs.write_file = fail_source
    target = {"kind": "person", "id": "anchor"}
    with pytest.raises(OSError):
        await store.discover(
            question_uri(ctx, target), target, [proposal(questionId=q["questionId"])]
        )
    fs.write_file = original_write
    fresh = QuestionStore(fs, ctx)
    actual = await fresh.get(q["questionId"])
    assert actual["subject"] == target
    assert actual["state"] == "resolved"
    assert len(actual["answers"]) == 1
    assert len(await fresh.list()) == 1
    assert not any(".question-moves/" in uri for uri in fs.files)


@pytest.mark.asyncio
async def test_existing_archive_is_reconciled_without_resubmitting(setup):
    fs, ctx, store = setup
    q = await create(store, ctx)
    await store.record(event(q, "resolved"))
    session = SimpleNamespace(
        uri="viking://user/alice/sessions/answer", meta=SimpleNamespace(), _save_meta=AsyncMock()
    )
    sessions = SimpleNamespace(
        get=AsyncMock(return_value=session), get_commit_task=AsyncMock(return_value=None)
    )
    fs.files[session.uri + "/history/archive_001/messages.jsonl"] = "{}"
    actual = await store.propagate(q["questionId"], sessions)
    assert actual["propagation"]["status"] == "unknown"
    fs.files[session.uri + "/history/archive_001/.done"] = "{}"
    actual = await store.propagate(q["questionId"], sessions)
    assert actual["propagation"]["status"] == "done"
    assert actual["state"] == "resolved"


@pytest.mark.asyncio
async def test_generic_content_write_cannot_overwrite_existing_question_page(setup):
    from openviking.storage.content_write import ContentWriteCoordinator

    fs, ctx, store = setup
    q = await create(store, ctx)
    fs._ensure_mutable_access = lambda *args: None
    original = fs.files[q["questionUri"]]
    with pytest.raises(InvalidArgumentError):
        await ContentWriteCoordinator(fs).write(
            uri=q["questionUri"], content="erase answer", ctx=ctx
        )
    assert fs.files[q["questionUri"]] == original


@pytest.mark.asyncio
async def test_generic_native_operation_cannot_retype_or_delete_question_page(setup):
    from openviking.session.memory.dataclass import ResolvedOperation
    from openviking.session.memory.memory_updater import MemoryUpdater

    fs, ctx, store = setup
    q = await create(store, ctx)
    updater = MemoryUpdater()
    updater._viking_fs = fs
    with pytest.raises(ValueError):
        await updater._apply_upsert(
            ResolvedOperation(
                memory_type="entities",
                uris=[q["questionUri"]],
                memory_fields={"content": "erase answer"},
            ),
            ctx,
        )
    with pytest.raises(ValueError):
        await updater._apply_delete(q["questionUri"], ctx)
    assert (await store.get(q["questionId"]))["state"] == "open"


@pytest.fixture
def clock(monkeypatch):
    from openviking.session.memory import question_store

    current = {"now": "2030-01-01T12:00:00+00:00"}
    monkeypatch.setattr(question_store, "now_iso", lambda: current["now"])
    return current


def timed_proposal(**changes):
    return proposal(
        purpose="speaker_identity",
        scope={
            "speakerRef": "transcript-1:version-1:speaker-0",
            "sourceVersion": "version-1",
            "mediaWindows": [{"startMs": 0, "endMs": 10000}],
        },
        delivery={
            "mode": "timeBound",
            "notBefore": "2030-01-01T11:00:00Z",
            "expiresAt": "2030-01-01T13:00:00Z",
        },
        **changes,
    )


async def create_timed(store, ctx, changes=None):
    entry = {**timed_proposal(), **(changes or {})}
    subject = {"kind": "matter", "id": "meeting-1"}
    records = await store.discover(question_uri(ctx, subject), subject, [entry])
    return next(q for q in records if q["topicKey"] == entry["topicKey"])


def received_event(q, **changes):
    return event(
        q,
        "asked",
        deliveryReceipt={
            "deliveryId": q["deliveryId"],
            "channel": "glasses",
            "receivedAt": "2030-01-01T11:59:00Z",
            "messageId": "asked-1",
            **changes,
        },
    )


@pytest.mark.asyncio
async def test_due_and_contextual_candidates_are_disjoint_read_only_views(setup, clock):
    _, ctx, store = setup
    contextual = await create(store, ctx)
    timed = await create_timed(store, ctx)
    due = await store.due()
    assert [q["questionId"] for q in due] == [timed["questionId"]]
    assert [q["questionId"] for q in await store.candidates()] == [contextual["questionId"]]
    assert due[0]["deliveryId"] == (await store.get(timed["questionId"]))["deliveryId"]
    assert due[0]["revision"] == timed["revision"]
    assert due[0]["state"] == "open"
    assert due[0]["events"] == []


@pytest.mark.asyncio
async def test_due_window_is_half_open_and_expiry_preserves_record(setup, clock):
    fs, ctx, store = setup
    q = await create_timed(store, ctx)
    for now, expected in (
        ("2030-01-01T10:59:59+00:00", False),
        ("2030-01-01T11:00:00+00:00", True),
        ("2030-01-01T12:59:59+00:00", True),
        ("2030-01-01T13:00:00+00:00", False),
    ):
        clock["now"] = now
        assert bool(await store.due()) is expected
    stored = await QuestionStore(fs, ctx).get(q["questionId"])
    assert stored["state"] == "open"
    assert stored["scope"]["sourceVersion"] == "version-1"
    assert stored["deliveryId"] == q["deliveryId"]


@pytest.mark.asyncio
async def test_rediscovery_never_renews_window_or_changes_original_speaker_scope(setup, clock):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    clock["now"] = "2030-01-03T12:00:00+00:00"
    proposal = timed_proposal()
    proposal["delivery"] = {
        "mode": "timeBound",
        "notBefore": "2030-01-03T11:00:00Z",
        "expiresAt": "2030-01-03T13:00:00Z",
    }
    proposal["scope"]["sourceVersion"] = "version-2"
    await store.discover(q["questionUri"], q["subject"], [proposal])
    actual = await store.get(q["questionId"])
    assert actual["delivery"] == q["delivery"]
    assert actual["scope"] == q["scope"]
    assert actual["deliveryId"] == q["deliveryId"]
    assert actual["revision"] > q["revision"]
    assert await store.due() == []


@pytest.mark.asyncio
async def test_subject_move_preserves_delivery_id_and_initial_metadata(setup, clock):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    target = {"kind": "person", "id": "confirmed-anchor"}
    await store.discover(question_uri(ctx, target), target, [proposal(questionId=q["questionId"])])
    actual = await store.get(q["questionId"])
    assert actual["deliveryId"] == q["deliveryId"]
    assert actual["scope"] == q["scope"]
    assert actual["subject"] == target


@pytest.mark.asyncio
async def test_generated_text_without_visible_device_receipt_is_not_asked(setup, clock):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    with pytest.raises(InvalidArgumentError):
        await store.record(event(q, "asked"))
    assert (await store.get(q["questionId"]))["events"] == []
    assert len(await store.due()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"deliveryId": "00000000-0000-4000-8000-000000000001"},
        {"channel": "generated"},
        {"messageId": "different-message"},
        {"receivedAt": "2030-01-01T10:59:59Z"},
        {"receivedAt": "2030-01-01T13:00:00Z"},
        {"receivedAt": "2030-01-01T12:00:01Z"},
        {"receivedAt": "2030-01-01T11:30:00"},
        {"receivedAt": "invalid"},
    ],
)
async def test_invalid_delivery_receipts_cannot_change_question_state(setup, clock, changes):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    with pytest.raises(InvalidArgumentError):
        await store.record(received_event(q, **changes))
    assert (await store.get(q["questionId"]))["state"] == "open"


@pytest.mark.asyncio
async def test_received_time_is_authoritative_and_retry_is_idempotent_after_expiry(setup, clock):
    fs, ctx, store = setup
    q = await create_timed(store, ctx)
    clock["now"] = "2030-01-01T14:00:00+00:00"
    data = received_event(q)
    data["evidenceAt"] = "2030-01-01T10:00:00Z"
    actual = (await store.record(data))["question"]
    assert actual["state"] == "asked"
    assert actual["events"][0]["at"] == "2030-01-01T11:59:00+00:00"
    assert actual["events"][0]["deliveryReceipt"]["channel"] == "glasses"
    await store.record(event(q, "partial", messageId="late-partial"))
    retry = await QuestionStore(fs, ctx).record(data)
    assert retry["duplicate"] is True
    assert retry["question"]["state"] == "deferred"
    assert len(retry["question"]["events"]) == 2
    assert await store.due() == []
    with pytest.raises(InvalidArgumentError):
        await store.record(event(q, "asked"))


@pytest.mark.asyncio
async def test_duplicate_delivery_cannot_be_reassigned_to_another_message(setup, clock):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    await store.record(received_event(q))
    data = received_event(q, messageId="second-visible-message")
    data["messageId"] = "second-visible-message"
    with pytest.raises(InvalidArgumentError):
        await store.record(data)
    assert len((await store.get(q["questionId"]))["events"]) == 1


@pytest.mark.asyncio
async def test_deferred_time_bound_respects_deferral_without_renewing_expiry(setup, clock):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    await store.record(event(q, "deferred", notBefore="2030-01-01T12:30:00Z"))
    assert await store.due() == []
    with pytest.raises(InvalidArgumentError):
        await store.record(received_event(q))
    clock["now"] = "2030-01-01T12:30:00+00:00"
    assert (await store.due())[0]["deliveryId"] == q["deliveryId"]
    await store.record(received_event(q, receivedAt="2030-01-01T12:30:00Z"))
    await store.record(event(q, "partial", messageId="partial", notBefore="2030-01-01T12:31:00Z"))
    clock["now"] = "2030-01-01T12:40:00+00:00"
    assert await store.due() == []


@pytest.mark.asyncio
async def test_late_answer_survives_expiry_rediscovery_and_store_restart(setup, clock):
    fs, ctx, store = setup
    q = await create_timed(store, ctx)
    clock["now"] = "2030-01-02T12:00:00+00:00"
    await store.record(event(q, "resolved", evidenceText="That speaker was Ethan."))
    await store.discover(q["questionUri"], q["subject"], [timed_proposal()])
    actual = await QuestionStore(fs, ctx).get(q["questionId"])
    assert actual["state"] == "resolved"
    assert actual["answers"][0]["text"] == "That speaker was Ethan."
    assert actual["propagation"]["status"] == "pending"
    assert actual["deliveryId"] == q["deliveryId"]
    assert await store.due() == []


@pytest.mark.asyncio
async def test_due_is_scoped_limited_and_earliest_expiry_first(setup, clock):
    fs, ctx, store = setup
    late = await create_timed(store, ctx)
    early = await create_timed(
        store,
        ctx,
        {
            "topicKey": "different_question",
            "delivery": {
                "mode": "timeBound",
                "notBefore": "2030-01-01T11:00:00Z",
                "expiresAt": "2030-01-01T12:30:00Z",
            },
        },
    )
    assert [q["questionId"] for q in await store.due(limit=1)] == [early["questionId"]]
    assert {q["questionId"] for q in await store.due()} == {late["questionId"], early["questionId"]}
    other_ctx = RequestContext(user=UserIdentifier("account", "bob"), role=Role.ROOT)
    assert await QuestionStore(fs, other_ctx).due() == []
    with pytest.raises(InvalidArgumentError):
        await store.due(limit=100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", ""),
        ("purpose", "x" * 161),
        ("scope", []),
        ("scope", {"tooLarge": "x" * 20000}),
        ("scope", {"tooDeep": [[[[[[[[["x"]]]]]]]]]}),
        ("scope", {"invalid": float("nan")}),
        ("scope", {"tooMany": list(range(101))}),
        ("delivery", {"mode": "unknown"}),
        ("delivery", {"mode": "contextual", "expiresAt": "2030-01-01T13:00:00Z"}),
        (
            "delivery",
            {
                "mode": "timeBound",
                "notBefore": "2030-01-01T13:00:00Z",
                "expiresAt": "2030-01-01T13:00:00Z",
            },
        ),
        ("delivery", {"mode": "timeBound", "notBefore": "2030-01-01T11:00:00"}),
    ],
)
def test_proposal_delivery_metadata_is_bounded_and_validated(field, value):
    from openviking.session.memory.question_store import proposal_delivery_metadata

    entry = {**timed_proposal(), field: value}
    with pytest.raises(ValueError):
        proposal_delivery_metadata(entry)


def test_proposals_normalize_timezones_and_copy_source_scope():
    from openviking.session.memory.question_store import proposal_delivery_metadata

    entry = timed_proposal()
    entry["delivery"]["notBefore"] = "2030-01-01T03:00:00-08:00"
    normalized = proposal_delivery_metadata(entry)
    assert normalized["delivery"]["notBefore"] == "2030-01-01T11:00:00+00:00"
    normalized["scope"]["sourceVersion"] = "modified"
    assert entry["scope"]["sourceVersion"] == "version-1"


@pytest.mark.asyncio
async def test_questions_router_exposes_due_get_revision_and_receipt_contract(
    setup, clock, monkeypatch
):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from openviking.server.auth import get_request_context
    from openviking.server.routers import questions

    _, ctx, store = setup
    q = await create_timed(store, ctx)
    app = FastAPI()
    app.include_router(questions.router)
    app.dependency_overrides[get_request_context] = lambda: ctx
    monkeypatch.setattr(questions, "store", lambda _ctx: store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/questions/due", json={"limit": 1})
        assert response.status_code == 200
        candidate = response.json()["result"]["questions"][0]
        reread = (await client.get("/api/v1/questions/" + q["questionId"])).json()["result"]
        assert candidate["deliveryId"] == reread["deliveryId"]
        assert candidate["revision"] == reread["revision"]
        response = await client.post(
            "/api/v1/questions/record", json=received_event(q, channel="phone")
        )
        assert response.status_code == 200
        assert response.json()["result"]["question"]["state"] == "asked"
        assert (await client.post("/api/v1/questions/due", json={})).json()["result"][
            "questions"
        ] == []
        assert (await client.post("/api/v1/questions/due", json={"limit": 21})).status_code == 422


@pytest.mark.asyncio
async def test_explicit_contextual_delivery_keeps_legacy_asked_contract(setup, clock):
    _, ctx, store = setup
    subject = {"kind": "self", "id": "self"}
    q = (
        await store.discover(
            question_uri(ctx, subject), subject, [proposal(delivery={"mode": "contextual"})]
        )
    )[0]
    assert [candidate["questionId"] for candidate in await store.candidates()] == [q["questionId"]]
    assert "deliveryId" not in q
    actual = (await store.record(event(q, "asked")))["question"]
    assert actual["state"] == "asked"
    assert await store.due() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["resolved", "dismissed"])
async def test_answer_or_dismissal_before_delivery_prevents_due_and_receipt(setup, clock, action):
    _, ctx, store = setup
    q = await create_timed(store, ctx)
    await store.record(event(q, action))
    assert await store.due() == []
    with pytest.raises(InvalidArgumentError):
        await store.record(received_event(q))
    assert (await store.get(q["questionId"]))["state"] == action


TRANSCRIPT_REF = "transcript:00000000-0000-4000-8000-000000000001"


def timing_proposal(**changes):
    return proposal(
        sourceRefs=[TRANSCRIPT_REF],
        timing={
            "kind": "meetingEnded",
            "occurredAt": "2030-01-01T03:00:00-08:00",
            "sourceRefs": [TRANSCRIPT_REF],
        },
        **changes,
    )


async def create_with_timing(store, ctx):
    subject = {"kind": "matter", "id": "meeting-1"}
    return (await store.discover(question_uri(ctx, subject), subject, [timing_proposal()]))[0]


def cycle_event(q, **changes):
    return event(
        q,
        "asked",
        deliveryReceipt={
            "deliveryId": q["deliveryCycleId"],
            "channel": "phone",
            "receivedAt": "2030-01-01T11:59:00Z",
            "messageId": "asked-1",
            **changes,
        },
    )


@pytest.mark.asyncio
async def test_provider_timing_is_source_backed_immutable_and_not_a_delivery_window(setup, clock):
    fs, ctx, store = setup
    q = await create_with_timing(store, ctx)
    assert q["timing"] == {
        "kind": "meetingEnded",
        "occurredAt": "2030-01-01T11:00:00+00:00",
        "sourceRefs": [TRANSCRIPT_REF],
    }
    assert "delivery" not in q and "deliveryId" not in q
    assert (await store.candidates())[0]["questionId"] == q["questionId"]
    assert await store.due() == []  # Legacy due endpoint remains a legacy view.
    changed = timing_proposal()
    changed["timing"]["occurredAt"] = "2030-01-02T11:00:00Z"
    await store.discover(q["questionUri"], q["subject"], [changed])
    reread = await QuestionStore(fs, ctx).get(q["questionId"])
    assert reread["timing"] == q["timing"]
    assert reread["deliveryCycleId"] == q["deliveryCycleId"]


@pytest.mark.parametrize(
    "timing",
    [
        {"kind": "meetingEnded", "occurredAt": "2030-01-01T11:00:00Z", "sourceRefs": []},
        {
            "kind": "meetingEnded",
            "occurredAt": "2030-01-01T11:00:00Z",
            "sourceRefs": ["transcript:00000000-0000-4000-8000-000000000002"],
        },
        {
            "kind": "meetingEnded",
            "occurredAt": "2030-01-01T11:00:00Z",
            "sourceRefs": ["email:00000000-0000-4000-8000-000000000001"],
        },
        {
            "kind": "meetingEnded",
            "occurredAt": "2030-01-01T11:00:00",
            "sourceRefs": [TRANSCRIPT_REF],
        },
        {
            "kind": "meetingEnded",
            "occurredAt": "2030-01-01T11:00:00Z",
            "sourceRefs": [TRANSCRIPT_REF],
            "expiresAt": "2030-01-01T13:00:00Z",
        },
        {"kind": "askNow", "occurredAt": "2030-01-01T11:00:00Z", "sourceRefs": [TRANSCRIPT_REF]},
    ],
)
@pytest.mark.asyncio
async def test_invalid_provider_timing_never_persists(setup, timing):
    _, ctx, store = setup
    subject = {"kind": "self", "id": "self"}
    entry = timing_proposal()
    entry["timing"] = timing
    with pytest.raises(ValueError):
        await store.discover(question_uri(ctx, subject), subject, [entry])
    assert await store.list() == []


@pytest.mark.parametrize("field", ["timing", "delivery", "state", "answers", "events"])
def test_extraction_cannot_control_presentation_or_user_lifecycle(field):
    from openviking.session.memory.question_store import validate_proposals

    entry = proposal(**{field: {}})
    with pytest.raises(ValueError):
        validate_proposals([entry], set(entry["sourceRefs"]))


@pytest.mark.asyncio
async def test_cycle_changes_only_for_explicit_deferral_and_survives_restart_move(setup, clock):
    from uuid import NAMESPACE_URL, uuid5

    fs, ctx, store = setup
    q = await create_with_timing(store, ctx)
    assert q["deliveryCycleId"] == str(
        uuid5(NAMESPACE_URL, f"question-cycle:{q['questionId']}:initial")
    )
    partial = (await store.record(event(q, "partial", notBefore="2030-01-01T12:30:00Z")))[
        "question"
    ]
    assert partial["deliveryCycleId"] == q["deliveryCycleId"]
    assert partial["events"][-1]["notBefore"] == "2030-01-01T12:30:00+00:00"
    deferred = (
        await store.record(
            event(q, "deferred", messageId="defer-1", notBefore="2030-01-01T04:30:00-08:00")
        )
    )["question"]
    last = deferred["events"][-1]
    assert last["notBefore"] == "2030-01-01T12:30:00+00:00"
    assert deferred["deliveryCycleId"] == str(
        uuid5(NAMESPACE_URL, f"question-cycle:{q['questionId']}:{last['eventId']}")
    )
    assert deferred["deliveryCycleId"] != q["deliveryCycleId"]
    duplicate = await store.record(
        event(q, "deferred", messageId="defer-1", notBefore="2030-01-01T12:30:00Z")
    )
    assert duplicate["duplicate"]
    assert duplicate["question"]["deliveryCycleId"] == deferred["deliveryCycleId"]
    target = {"kind": "person", "id": "anchor-1"}
    await store.discover(question_uri(ctx, target), target, [proposal(questionId=q["questionId"])])
    fresh = await QuestionStore(fs, ctx).get(q["questionId"])
    assert fresh["deliveryCycleId"] == deferred["deliveryCycleId"]
    assert fresh["timing"] == q["timing"]
    assert fresh["events"] == deferred["events"]


@pytest.mark.asyncio
async def test_implicit_default_deferral_is_not_recorded_as_explicit_user_time(setup, clock):
    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    deferred = (await store.record(event(q, "deferred")))["question"]
    assert "notBefore" in deferred
    assert "notBefore" not in deferred["events"][-1]
    assert deferred["deliveryCycleId"] != q["deliveryCycleId"]


@pytest.mark.asyncio
async def test_cycle_receipt_after_policy_window_and_multichannel_retry_after_answer(setup, clock):
    fs, ctx, store = setup
    q = await create_with_timing(store, ctx)
    clock["now"] = "2030-01-03T12:00:00+00:00"
    data = cycle_event(q, receivedAt="2030-01-03T11:59:00Z")
    asked = (await store.record(data))["question"]
    assert asked["state"] == "asked"
    assert asked["events"][-1]["deliveryReceipt"]["deliveryId"] == q["deliveryCycleId"]
    await store.record(event(q, "resolved"))
    data["deliveryReceipt"]["channel"] = "glasses"
    retry = await QuestionStore(fs, ctx).record(data)
    assert retry["duplicate"] and retry["question"]["state"] == "resolved"
    assert len(retry["question"]["events"]) == 2


@pytest.mark.asyncio
async def test_stale_cycle_cannot_reopen_after_deferral_but_recorded_retry_is_safe(setup, clock):
    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    await store.record(cycle_event(q))
    deferred = (await store.record(event(q, "deferred", notBefore="2030-01-01T12:30:00Z")))[
        "question"
    ]
    assert (await store.record(cycle_event(q)))["duplicate"]
    with pytest.raises(InvalidArgumentError):
        await store.record(cycle_event(deferred))
    clock["now"] = "2030-01-01T12:30:00+00:00"
    new = cycle_event(deferred, messageId="asked-2", receivedAt="2030-01-01T12:30:00Z")
    new["messageId"] = "asked-2"
    assert (await store.record(new))["question"]["state"] == "asked"
    assert len((await store.get(q["questionId"]))["events"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["resolved", "dismissed"])
async def test_unrecorded_old_cycle_receipt_cannot_reopen_terminal_question(setup, clock, action):
    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    await store.record(event(q, "deferred", notBefore="2030-01-01T11:00:00Z"))
    await store.record(event(q, action, messageId="terminal"))
    with pytest.raises(InvalidArgumentError):
        await store.record(cycle_event(q))
    assert (await store.get(q["questionId"]))["state"] == action


@pytest.mark.asyncio
async def test_timing_question_can_be_asked_contextually_without_device_receipt(setup, clock):
    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    actual = (await store.record(event(q, "asked")))["question"]
    assert actual["state"] == "asked"
    assert "deliveryReceipt" not in actual["events"][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"receivedAt": "2030-01-01T12:01:00Z"},
        {"messageId": "wrong"},
        {"channel": "generated"},
        {"deliveryId": "00000000-0000-4000-8000-000000000001"},
    ],
)
async def test_cycle_receipt_contract_validates_time_message_channel_and_cycle(
    setup, clock, changes
):
    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    with pytest.raises(InvalidArgumentError):
        await store.record(cycle_event(q, **changes))
    assert (await store.get(q["questionId"]))["state"] == "open"


@pytest.mark.asyncio
async def test_unrecorded_prior_cycle_is_rejected_after_user_deferral(setup, clock):
    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    await store.record(event(q, "deferred", notBefore="2030-01-01T11:00:00Z"))
    with pytest.raises(InvalidArgumentError):
        await store.record(cycle_event(q))
    assert (await store.get(q["questionId"]))["state"] == "deferred"


@pytest.mark.asyncio
async def test_public_route_preserves_timing_cycle_and_user_time_history(setup, clock, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from openviking.server.auth import get_request_context
    from openviking.server.routers import questions

    _, ctx, store = setup
    q = await create_with_timing(store, ctx)
    app = FastAPI()
    app.include_router(questions.router)
    app.dependency_overrides[get_request_context] = lambda: ctx
    monkeypatch.setattr(questions, "store", lambda _ctx: store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        listed = (await client.get("/api/v1/questions")).json()["result"]["questions"][0]
        assert listed["timing"] == q["timing"]
        assert listed["deliveryCycleId"] == q["deliveryCycleId"]
        response = await client.post("/api/v1/questions/record", json=cycle_event(q))
        assert response.status_code == 200
        asked = response.json()["result"]["question"]
        assert asked["events"][0]["deliveryReceipt"]["deliveryId"] == q["deliveryCycleId"]
        response = await client.post(
            "/api/v1/questions/record",
            json=event(q, "deferred", notBefore="2030-01-01T12:10:00Z"),
        )
        assert response.status_code == 200
        reread = (await client.get("/api/v1/questions/" + q["questionId"])).json()["result"]
        assert reread["deliveryCycleId"] != q["deliveryCycleId"]
        assert reread["events"][-1]["eventId"]
        assert reread["events"][-1]["notBefore"] == "2030-01-01T12:10:00+00:00"


@pytest.mark.asyncio
async def test_delayed_receipt_can_verify_original_canonical_wording_after_rediscovery(
    setup, clock
):
    fs, ctx, store = setup
    q = await create_with_timing(store, ctx)
    replacement = {**timing_proposal(), "text": "Did that speaker call you Junkuan?"}
    await store.discover(q["questionUri"], q["subject"], [replacement])
    reread = await store.get(q["questionId"])
    assert reread["text"] == replacement["text"]
    assert reread["deliveryCycleId"] == q["deliveryCycleId"]
    assert "wordingHistory" not in reread
    assert "wordingHistory" not in (await store.list())[0]
    with pytest.raises(InvalidArgumentError):
        await store.record(event(q, "asked"))  # Ordinary chat must use current wording.
    with pytest.raises(InvalidArgumentError):
        await store.record({**cycle_event(q), "evidenceText": "Unrelated invented question?"})
    actual = await QuestionStore(fs, ctx).record(cycle_event(q))
    assert actual["question"]["state"] == "asked"
    assert actual["question"]["events"][-1]["evidenceText"] == q["text"]
    assert "wordingHistory" not in actual["question"]
    page = MemoryFileUtils.read(fs.files[q["questionUri"]])
    assert page.extra_fields["questions"][0]["wordingHistory"] == [q["text"]]
