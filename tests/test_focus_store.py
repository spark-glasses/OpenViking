import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.focus_store import FocusStore
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from tests.test_question_store import FS, proposal


@pytest.fixture
def store():
    return FocusStore(FS(), RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT))


@pytest.mark.asyncio
async def test_user_creation_is_immediate_and_retry_preserves_identity(store):
    args = dict(
        name="Be healthier", create_key="tap-one", origin="user", user_intent="Sleep better"
    )
    first = await store.ensure(**args)
    assert first == await store.ensure(**args)
    assert first["status"] == "active"
    assert "visibility" not in first
    assert first["userIntent"] == "Sleep better"
    assert (await store.list(status="active"))["focuses"][0]["focusId"] == first["focusId"]
    with pytest.raises(AlreadyExistsError):
        await store.ensure(**(args | {"name": "Different request"}))


@pytest.mark.asyncio
async def test_discovery_needs_evidence_and_starts_unlisted(store):
    with pytest.raises(InvalidArgumentError):
        await store.ensure(name="Workspace", create_key="slack")
    focus = await store.ensure(
        name="Study",
        create_key="study",
        discovery_reason="Repeated study plans",
        source_refs=["session-message:one"],
    )
    assert focus["origin"] == "discovered" and focus["status"] == "forming"
    assert (await store.list(status="active"))["focuses"] == []
    with pytest.raises(InvalidArgumentError):
        await store.update(
            focus["focusId"], expected_revision=1, actor="model", fields={"status": "active"}
        )
    surfaced = await store.update(
        focus["focusId"],
        expected_revision=1,
        actor="model",
        fields={"summary": "Preparing for a degree", "status": "active"},
    )
    assert surfaced["status"] == "active"


@pytest.mark.asyncio
async def test_archiving_cannot_be_undone_or_recreated_by_discovery(store):
    focus = await store.ensure(
        name="Study",
        create_key="study",
        discovery_reason="Study plans",
        source_refs=["session-message:one"],
    )
    archived = await store.update(
        focus["focusId"], expected_revision=1, fields={"status": "archived"}
    )
    repeated = await store.ensure(
        name="Study",
        create_key="another-run",
        discovery_reason="More study plans",
        source_refs=["session-message:two"],
    )
    assert repeated["focusId"] == archived["focusId"]
    assert repeated["status"] == "archived"
    with pytest.raises(InvalidArgumentError):
        await store.update(
            focus["focusId"],
            expected_revision=2,
            actor="model",
            fields={"status": "active", "summary": "More information"},
        )
    restored = await store.update(
        focus["focusId"], expected_revision=2, fields={"status": "active"}
    )
    assert restored["status"] == "active"
    assert restored["focusId"] == focus["focusId"]


@pytest.mark.asyncio
async def test_user_cannot_set_internal_or_removed_states(store):
    focus = await store.ensure(name="Health", create_key="health", origin="user")
    for fields in ({"visibility": "hidden"}, {"status": "forming"}, {"status": "hidden"}):
        with pytest.raises(InvalidArgumentError):
            await store.update(focus["focusId"], expected_revision=1, fields=fields)


@pytest.mark.asyncio
async def test_status_filter_applies_before_pagination(store):
    values = [await store.ensure(name=str(i), create_key=str(i), origin="user") for i in range(4)]
    await store.update(values[0]["focusId"], expected_revision=1, fields={"status": "archived"})
    forming = await store.ensure(
        name="Study",
        create_key="forming",
        discovery_reason="Study matters",
        source_refs=["session:one"],
    )
    first = await store.list(status="active", limit=2)
    second = await store.list(status="active", after=first["nextCursor"], limit=2)
    assert {v["focusId"] for v in first["focuses"] + second["focuses"]} == {
        v["focusId"] for v in values[1:]
    }
    assert second["nextCursor"] is None
    assert [v["focusId"] for v in (await store.list(status="archived"))["focuses"]] == [
        values[0]["focusId"]
    ]
    assert [v["focusId"] for v in (await store.list(status="forming"))["focuses"]] == [
        forming["focusId"]
    ]


@pytest.mark.asyncio
async def test_user_intent_and_name_are_protected_but_narrative_can_grow(store):
    focus = await store.ensure(
        name="Health", create_key="health", origin="user", user_intent="More energy"
    )
    for fields in ({"name": "Weight loss"}, {"userIntent": "Lose weight"}, {"status": "archived"}):
        with pytest.raises(InvalidArgumentError):
            await store.update(focus["focusId"], expected_revision=1, actor="model", fields=fields)
    result = await store.update(
        focus["focusId"],
        expected_revision=1,
        actor="model",
        fields={"content": "The user is improving sleep."},
    )
    assert result["userIntent"] == "More energy"
    assert result["content"] == "The user is improving sleep."


@pytest.mark.asyncio
async def test_two_writers_cannot_silently_overwrite(store):
    focus = await store.ensure(name="Startup", create_key="startup", origin="user")
    outcomes = await asyncio.gather(
        *[
            store.update(focus["focusId"], expected_revision=1, fields={"name": n})
            for n in ("A", "B")
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(x, AlreadyExistsError) for x in outcomes) == 1


@pytest.mark.asyncio
async def test_native_journal_cleanup_reuses_extraction_lock(store):
    handle = object()
    store.lock_handle = handle
    original = store.fs.rm
    seen = []

    async def locked_remove(uri, **kwargs):
        seen.append(kwargs.get("lock_handle"))
        if kwargs.get("lock_handle") is not handle:
            raise RuntimeError("Cannot acquire another owner's directory lock")
        return await original(uri, **kwargs)

    store.fs.rm = locked_remove
    value = await store.ensure(name="Health", create_key="native-lock", origin="user")
    await store.update(
        value["focusId"], expected_revision=1, actor="model", fields={"content": "Sleep matters"}
    )
    assert seen and all(lock is handle for lock in seen)


@pytest.mark.asyncio
async def test_edit_receipt_recovers_a_lost_response(store):
    focus = await store.ensure(name="Startup", create_key="startup", origin="user")
    args = dict(
        expected_revision=1, fields={"userIntent": "Build useful glasses"}, operation_id="edit-one"
    )
    first = await store.update(focus["focusId"], **args)
    assert await store.update(focus["focusId"], **args) == first
    assert first["revision"] == 2


@pytest.mark.asyncio
async def test_interrupted_metadata_write_is_recoverable(store):
    focus = await store.ensure(name="Health", create_key="health", origin="user")
    write = store.fs.write_file

    async def fail(uri, content, **kwargs):
        if uri == focus["metadataUri"]:
            raise OSError("interrupted")
        await write(uri, content, **kwargs)

    store.fs.write_file = fail
    with pytest.raises(OSError):
        await store.update(
            focus["focusId"], expected_revision=1, fields={"userIntent": "Sleep better"}
        )
    store.fs.write_file = write
    recovered = store.view(await store.get(focus["focusId"]))
    assert recovered["userIntent"] == "Sleep better"
    assert recovered["revision"] == 2


@pytest.mark.asyncio
async def test_catalog_pagination_and_owner_isolation(store):
    values = [await store.ensure(name=str(i), create_key=str(i), origin="user") for i in range(3)]
    first = await store.list(limit=2)
    second = await store.list(after=first["nextCursor"], limit=2)
    assert {v["focusId"] for v in first["focuses"] + second["focuses"]} == {
        v["focusId"] for v in values
    }
    other = FocusStore(
        store.fs, RequestContext(user=UserIdentifier("account", "bob"), role=Role.ROOT)
    )
    assert (await other.list())["focuses"] == []
    with pytest.raises(NotFoundError):
        await other.get(values[0]["focusId"])


@pytest.mark.asyncio
async def test_questions_live_under_focus_and_retain_subject(store):
    focus = await store.ensure(name="Study", create_key="study", origin="user")
    subject = {"kind": "focus", "id": focus["focusId"], "memoryUri": focus["uri"]}
    questions = QuestionStore(store.fs, store.ctx)
    value = (await questions.discover(question_uri(store.ctx, subject), subject, [proposal()]))[0]
    assert value["questionUri"] == focus["questionsUri"]
    assert (await questions.get(value["questionId"]))["subject"] == subject


@pytest.mark.asyncio
async def test_new_user_focus_indexes_intent_before_any_model_narrative(store, monkeypatch):
    from openviking.session.memory.memory_updater import MemoryUpdater
    from openviking.storage.queuefs.embedding_msg_converter import EmbeddingMsgConverter

    indexed = []

    def convert(context):
        indexed.append(context.vectorize.text)
        return SimpleNamespace(telemetry_id=None)

    monkeypatch.setattr(EmbeddingMsgConverter, "from_context", convert)
    db = SimpleNamespace(has_queue_manager=True, enqueue_embedding_msg=AsyncMock(return_value=True))
    focus = await store.ensure(
        name="Health",
        create_key="intent-index",
        origin="user",
        user_intent="Have energy for hiking",
    )
    assert await MemoryUpdater.refresh_file_embedding(
        viking_fs=store.fs, vikingdb=db, uri=focus["uri"], memory_type="focuses", ctx=store.ctx
    )
    assert "Have energy for hiking" in indexed[0]
    assert "Health" in indexed[0]


@pytest.mark.asyncio
async def test_user_rename_updates_title_and_preserves_learned_body(store):
    focus = await store.ensure(name="Health", create_key="rename", origin="user")
    await store.update(
        focus["focusId"],
        expected_revision=1,
        actor="model",
        fields={"content": "# Health\n\nSleep helps."},
    )
    changed = await store.update(
        focus["focusId"], expected_revision=2, fields={"name": "More energy"}
    )
    assert changed["content"] == "# More energy\n\nSleep helps."
