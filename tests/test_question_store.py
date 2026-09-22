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
