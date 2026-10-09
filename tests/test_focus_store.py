import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.focus_store import FocusStore
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking_cli.exceptions import (
    AlreadyExistsError,
    ConflictError,
    InvalidArgumentError,
    NotFoundError,
)
from openviking_cli.session.user_id import UserIdentifier
from tests.test_question_store import FS, proposal


@pytest.fixture
def store():
    return FocusStore(FS(), RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT))


async def discovered(store, name="Study", key="study"):
    return await store.ensure(
        name=name,
        create_key=key,
        intent="They keep planning their study time",
        source_refs=["session-message:one"],
    )


def record(store, focus):
    return json.loads(store.fs.files[focus["metadataUri"]])


@pytest.mark.asyncio
async def test_user_creation_is_immediate_and_a_name_is_used_once(store):
    first = await store.ensure(name="Be healthier", origin="user", intent="Sleep better")
    assert first["status"] == "active" and first["origin"] == "user"
    assert first["intent"] == "Sleep better" and first["intentSource"] == "user"
    assert (await store.list(status="active"))["focuses"][0]["focusId"] == first["focusId"]
    with pytest.raises(AlreadyExistsError, match=first["focusId"]):
        await store.ensure(name="be healthier", origin="user")
    await store.update(first["focusId"], fields={"status": "archived"})
    with pytest.raises(AlreadyExistsError):
        await store.ensure(name="Be healthier", origin="user")
    assert len((await store.list())["focuses"]) == 1


@pytest.mark.asyncio
async def test_user_creation_takes_over_a_forming_focus_of_that_name(store):
    forming = await discovered(store)
    taken = await store.ensure(name="study", origin="user", intent="Finish my degree")
    assert taken["focusId"] == forming["focusId"]
    assert taken["status"] == "active" and taken["origin"] == "discovered"
    assert taken["intent"] == "Finish my degree" and taken["intentSource"] == "user"
    kept = await discovered(store, name="Health", key="health")
    confirmed = await store.ensure(name="Health", origin="user")
    assert confirmed["status"] == "active"
    assert confirmed["intent"] == kept["intent"] and confirmed["intentSource"] == "guessed"


@pytest.mark.asyncio
async def test_discovery_needs_evidence_and_a_guess_and_starts_unlisted(store):
    for missing in (
        {"create_key": "k", "intent": "Because"},
        {"create_key": "k", "source_refs": ["session:one"]},
        {"intent": "Because", "source_refs": ["session:one"]},
    ):
        with pytest.raises(InvalidArgumentError):
            await store.ensure(name="Workspace", **missing)
    focus = await discovered(store)
    assert focus["origin"] == "discovered" and focus["status"] == "forming"
    assert focus["intentSource"] == "guessed"
    assert focus == await discovered(store)
    assert (await store.list(status="active"))["focuses"] == []
    with pytest.raises(InvalidArgumentError):
        await store.update(focus["focusId"], actor="model", fields={"status": "active"})
    surfaced = await store.update(
        focus["focusId"],
        actor="model",
        fields={"summary": "Preparing for a degree", "status": "active"},
    )
    assert surfaced["status"] == "active"


@pytest.mark.asyncio
async def test_archiving_cannot_be_undone_or_recreated_by_discovery(store):
    focus = await discovered(store)
    archived = await store.update(focus["focusId"], fields={"status": "archived"})
    repeated = await discovered(store, key="another-run")
    assert repeated["focusId"] == archived["focusId"] and repeated["status"] == "archived"
    with pytest.raises(InvalidArgumentError):
        await store.update(
            focus["focusId"], actor="model", fields={"status": "active", "summary": "More"}
        )
    restored = await store.update(focus["focusId"], fields={"status": "active"})
    assert restored["status"] == "active" and restored["focusId"] == focus["focusId"]


@pytest.mark.asyncio
async def test_nobody_sets_the_draft_state_or_the_fields_the_system_keeps(store):
    focus = await store.ensure(name="Health", origin="user")
    for actor in ("user", "model"):
        for fields in (
            {"status": "forming"},
            {"status": "hidden"},
            {"origin": "discovered"},
            {"sourceRefs": []},
            {"intentSource": "user"},
            {"revision": 9},
        ):
            with pytest.raises(InvalidArgumentError):
                await store.update(focus["focusId"], actor=actor, fields=fields)
    assert record(store, focus)["revision"] == 1


@pytest.mark.asyncio
async def test_status_filter_applies_before_pagination(store):
    values = [await store.ensure(name=str(i), origin="user") for i in range(4)]
    await store.update(values[0]["focusId"], fields={"status": "archived"})
    forming = await discovered(store)
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
async def test_extraction_leaves_what_is_the_users_and_writes_the_rest(store):
    focus = await store.ensure(name="Health", origin="user", intent="More energy")
    for fields in ({"name": "Weight loss"}, {"intent": "Lose weight"}, {"status": "archived"}):
        with pytest.raises(InvalidArgumentError):
            await store.update(focus["focusId"], actor="model", fields=fields)
    result = await store.update(
        focus["focusId"],
        actor="model",
        fields={"content": "The user is improving sleep.", "summary": "Sleep first"},
    )
    assert result["intent"] == "More energy" and result["intentSource"] == "user"
    assert result["content"] == "The user is improving sleep."
    assert result["summary"] == "Sleep first"
    # Sending the name or the stated intent back unchanged is not a change.
    same = await store.update(
        focus["focusId"], actor="model", fields={"name": "Health", "intent": "More energy"}
    )
    assert same["name"] == "Health" and same["intentSource"] == "user"


@pytest.mark.asyncio
async def test_extraction_guesses_an_intent_only_where_the_user_stated_none(store):
    focus = await store.ensure(name="Startup", origin="user")
    assert focus["intent"] == "" and focus["intentSource"] is None
    guessed = await store.update(
        focus["focusId"], actor="model", fields={"intent": "Wants the company funded"}
    )
    assert guessed["intentSource"] == "guessed"
    improved = await store.update(
        focus["focusId"], actor="model", fields={"intent": "Wants the round closed by spring"}
    )
    assert improved["intent"] == "Wants the round closed by spring"
    stated = await store.update(focus["focusId"], fields={"intent": "Build useful glasses"})
    assert stated["intentSource"] == "user"
    with pytest.raises(InvalidArgumentError):
        await store.update(focus["focusId"], actor="model", fields={"intent": "A new guess"})
    cleared = await store.update(focus["focusId"], fields={"intent": ""})
    assert cleared["intentSource"] is None
    again = await store.update(focus["focusId"], actor="model", fields={"intent": "A new guess"})
    assert again["intentSource"] == "guessed"


@pytest.mark.asyncio
async def test_a_forming_focus_is_extractions_draft_until_the_user_touches_it(store):
    focus = await discovered(store)
    renamed = await store.update(focus["focusId"], actor="model", fields={"name": "Degree"})
    assert renamed["name"] == "Degree" and renamed["status"] == "forming"
    named = await store.update(focus["focusId"], fields={"name": "My degree"})
    assert named["status"] == "active"
    with pytest.raises(InvalidArgumentError):
        await store.update(focus["focusId"], actor="model", fields={"name": "Degree"})
    other = await discovered(store, name="Health", key="health")
    meant = await store.update(other["focusId"], fields={"intent": "Stay well"})
    assert meant["status"] == "active" and meant["intentSource"] == "user"
    third = await discovered(store, name="Garden", key="garden")
    declined = await store.update(third["focusId"], fields={"status": "archived"})
    assert declined["status"] == "archived"
    fourth = await discovered(store, name="Travel", key="travel")
    summarized = await store.update(fourth["focusId"], fields={"summary": "Trips this year"})
    assert summarized["status"] == "forming"


@pytest.mark.asyncio
async def test_two_writers_cannot_silently_overwrite(store):
    focus = await store.ensure(name="Startup", origin="user")
    outcomes = await asyncio.gather(
        *[
            store.update(focus["focusId"], expected_revision=1, fields={"name": n})
            for n in ("A", "B")
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(x, ConflictError) for x in outcomes) == 1


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
    value = await store.ensure(name="Health", origin="user")
    await store.update(value["focusId"], actor="model", fields={"content": "Sleep matters"})
    assert seen and all(lock is handle for lock in seen)


@pytest.mark.asyncio
async def test_interrupted_metadata_write_is_recoverable(store):
    focus = await store.ensure(name="Health", origin="user")
    write = store.fs.write_file

    async def fail(uri, content, **kwargs):
        if uri == focus["metadataUri"]:
            raise OSError("interrupted")
        await write(uri, content, **kwargs)

    store.fs.write_file = fail
    with pytest.raises(OSError):
        await store.update(focus["focusId"], fields={"intent": "Sleep better"})
    store.fs.write_file = write
    recovered = store.view(await store.get(focus["focusId"]))
    assert recovered["intent"] == "Sleep better"
    assert recovered["revision"] == 2


@pytest.mark.asyncio
async def test_catalog_pagination_and_owner_isolation(store):
    values = [await store.ensure(name=str(i), origin="user") for i in range(3)]
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
    assert (await other.ensure(name="0", origin="user"))["focusId"] != values[0]["focusId"]


@pytest.mark.asyncio
async def test_questions_live_under_focus_and_retain_subject(store):
    focus = await store.ensure(name="Study", origin="user")
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
    focus = await store.ensure(name="Health", origin="user", intent="Have energy for hiking")
    assert await MemoryUpdater.refresh_file_embedding(
        viking_fs=store.fs, vikingdb=db, uri=focus["uri"], memory_type="focuses", ctx=store.ctx
    )
    assert "Have energy for hiking" in indexed[0]
    assert "Health" in indexed[0]


@pytest.mark.asyncio
async def test_user_rename_updates_title_and_preserves_learned_body(store):
    focus = await store.ensure(name="Health", origin="user")
    await store.update(
        focus["focusId"], actor="model", fields={"content": "# Health\n\nSleep helps."}
    )
    changed = await store.update(focus["focusId"], fields={"name": "More energy"})
    assert changed["content"] == "# More energy\n\nSleep helps."


# ---- the two files written as files, from the user's side


@pytest.mark.asyncio
async def test_writing_the_record_of_a_focus_that_is_not_there_creates_it(store):
    written = await store.write_file(
        "startup",
        "focus.json",
        json.dumps({"name": "Startup", "intent": "Close the round"}),
        "create",
    )
    focus = (await store.list())["focuses"][0]
    assert written == focus["metadataUri"] and "/startup/" not in written
    assert focus["name"] == "Startup" and focus["status"] == "active"
    assert focus["intentSource"] == "user"
    assert store.fs.files[focus["uri"]].startswith("# Startup")
    summarized = await store.write_file(
        "x", "focus.json", json.dumps({"name": "Health", "summary": "Sleep first"}), "create"
    )
    assert json.loads(store.fs.files[summarized])["summary"] == "Sleep first"
    for content in (
        "not json",
        "[]",
        json.dumps({"intent": "No name"}),
        json.dumps({"name": 7}),
        json.dumps({"name": "Draft", "status": "forming"}),
        json.dumps({"name": "Other", "origin": "discovered"}),
    ):
        with pytest.raises(InvalidArgumentError):
            await store.write_file("y", "focus.json", content, "create")
    with pytest.raises(AlreadyExistsError):
        await store.write_file("z", "focus.json", json.dumps({"name": "startup"}), "create")
    for mode in ("replace", "append"):
        with pytest.raises(NotFoundError):
            await store.write_file("z", "focus.json", json.dumps({"name": "Late"}), mode)
    assert len((await store.list())["focuses"]) == 2


@pytest.mark.asyncio
async def test_a_written_record_changes_only_what_the_users_side_owns(store):
    focus = await store.ensure(name="Health", origin="user", intent="More energy")
    identifier = focus["focusId"]
    current = record(store, focus)

    async def write(changes):
        return await store.write_file(
            identifier, "focus.json", json.dumps(current | changes), "replace"
        )

    await write({"name": "Feel well", "summary": "Sleep first", "status": "archived"})
    after = record(store, focus)
    assert (after["name"], after["summary"], after["status"]) == (
        "Feel well",
        "Sleep first",
        "archived",
    )
    assert after["revision"] == 2 and after["intentSource"] == "user"
    # What was read before that change does not write over it.
    with pytest.raises(ConflictError):
        await write({"name": "Older copy"})
    current = after
    for changes in (
        {"focusId": "11111111-1111-4111-8111-111111111111"},
        {"origin": "discovered"},
        {"intentSource": "guessed"},
        {"sourceRefs": ["session:made-up"]},
        {"createdAt": "2020-01-01T00:00:00+00:00"},
        {"status": "forming"},
        {"status": "done"},
        {"name": ""},
        {"favourite": "blue"},
    ):
        with pytest.raises(InvalidArgumentError):
            await write(changes)
    for content, mode in (("{ not json", "replace"), ('{"name": "More"}', "append")):
        with pytest.raises(InvalidArgumentError):
            await store.write_file(identifier, "focus.json", content, mode)
    with pytest.raises(AlreadyExistsError):
        await store.write_file(identifier, "focus.json", json.dumps({"name": "Again"}), "create")
    assert record(store, focus) == after
    # The record as read, written back, changes nothing; a part of it is enough.
    await write({})
    await store.write_file(identifier, "focus.json", json.dumps({"intent": "Hike more"}), "replace")
    final = record(store, focus)
    assert final["revision"] == 3 and final["intent"] == "Hike more"
    assert final["name"] == "Feel well"


@pytest.mark.asyncio
async def test_a_written_page_replaces_or_extends_the_narrative(store):
    focus = await store.ensure(name="Health", origin="user")
    identifier = focus["focusId"]
    written = await store.write_file(identifier, "memory.md", "# Health\n\nSleep helps.", "replace")
    assert written == focus["uri"]
    await store.write_file(identifier, "memory.md", "## Mine\n- Ran twice.", "append")
    await store.write_file(identifier, "memory.md", "\n- Ran again.", "append")
    page = await store.get(identifier)
    assert page.content == "# Health\n\nSleep helps.\n\n## Mine\n- Ran twice.\n- Ran again."
    assert page.extra_fields["revision"] == 4 and page.extra_fields["name"] == "Health"
    # A page handed back with its fields attached is taken as the narrative alone.
    await store.write_file(
        identifier,
        "memory.md",
        "# Health\n\nRewritten.\n\n<!-- MEMORY_FIELDS\n"
        '{"status": "archived", "memory_type": "focuses"}\n-->',
        "replace",
    )
    page = await store.get(identifier)
    assert page.content == "# Health\n\nRewritten." and page.extra_fields["status"] == "active"
    with pytest.raises(AlreadyExistsError):
        await store.write_file(identifier, "memory.md", "again", "create")
    for folder in ("no-such-focus", "11111111-1111-4111-8111-111111111111"):
        with pytest.raises(NotFoundError):
            await store.write_file(folder, "memory.md", "text", "replace")


@pytest.mark.asyncio
async def test_a_file_write_of_either_file_reaches_the_store(store):
    from openviking.storage.content_write import ContentWriteCoordinator

    store.fs._ensure_mutable_access = lambda *args: None
    writer = ContentWriteCoordinator(store.fs)
    root = "viking://user/alice/memories/focuses/"
    created = await writer.write(
        uri=root + "garden/focus.json",
        content=json.dumps({"name": "Garden"}),
        ctx=store.ctx,
        mode="create",
    )
    focus = (await store.list())["focuses"][0]
    assert created["uri"] == focus["metadataUri"]
    await writer.write(
        uri=focus["uri"], content="# Garden\n\nTomatoes.", ctx=store.ctx, mode="replace"
    )
    assert (await store.get(focus["focusId"])).content == "# Garden\n\nTomatoes."
    before = dict(store.fs.files)
    for uri, content in (
        (focus["metadataUri"], "{ broken"),
        (focus["metadataUri"], json.dumps(record(store, focus) | {"status": "forming"})),
    ):
        with pytest.raises(InvalidArgumentError):
            await writer.write(uri=uri, content=content, ctx=store.ctx, mode="replace")
    assert store.fs.files == before
    assert (await store.list())["focuses"][0]["name"] == "Garden"
