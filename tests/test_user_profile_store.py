import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import MemoryField
from openviking.session.memory.profile_store import ProfileStore
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier
from tests.test_question_store import FS


@pytest.fixture
def store():
    return ProfileStore(
        FS(), RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    )


async def native(store, content, old=None):
    old = old or await store.read()
    schema = SimpleNamespace(
        fields=[MemoryField(name="content", field_type="string", merge_op="replace")]
    )
    op = SimpleNamespace(
        uris=[store.uri], old_memory_file_content=old, memory_fields={"content": content}
    )
    await store.apply_native(op, schema)


@pytest.mark.asyncio
async def test_migrate_narrative_and_edit_preserves_original_and_identity(store):
    await store.fs.write_file(store.uri, "## Social life\nI avoid people.")
    p = await store.get()
    b = p["blocks"][0]
    identity = {
        "preferredName": "Ben",
        "names": ["Ben", "Junkuan"],
        "additionalEmails": ["a@example.test"],
    }
    p = await store.identity(str(uuid4()), 7, identity, True)
    before_revision = p["identityRevision"]
    op = str(uuid4())
    p = await store.edit(
        op, p["documentRevision"], b["id"], b["title"], "I prefer one-on-one conversations."
    )
    assert p["identityRevision"] == before_revision
    assert p["identity"] == identity
    history = await store.history(b["id"])
    assert history[0]["before"]["content"] == "I avoid people."
    assert history[0]["before"]["authority"] == "import"
    assert history[0]["after"]["authority"] == "user"
    assert "I avoid people." not in p["content"]
    assert history[0]["reviewStatus"] == "pending"
    with pytest.raises(InvalidArgumentError):
        await native(
            store, p["content"].replace("I prefer one-on-one conversations.", "I avoid people.")
        )
    await native(store, p["content"] + "\n\n## Hobbies\nPiano")
    assert (await store.get())["blocks"][-1]["content"] == "Piano"


@pytest.mark.asyncio
async def test_replay_conflict_and_stale_model_cannot_overwrite(store):
    p = await store.get()
    old = await store.read()
    op = str(uuid4())
    edited = await store.edit(op, p["documentRevision"], None, "Goals", "Build Spark")
    assert await store.edit(op, p["documentRevision"], None, "Goals", "Build Spark") == edited
    assert len(await store.history()) == 1
    with pytest.raises(AlreadyExistsError):
        await store.edit(op, p["documentRevision"], None, "Goals", "Something else")
    with pytest.raises(AlreadyExistsError):
        await native(store, "## Goals\nDifferent", old)
    assert (await store.get())["content"] == edited["content"]


@pytest.mark.asyncio
async def test_deleted_user_section_cannot_be_restored_and_unlock_is_explicit(store):
    p = await store.get()
    p = await store.edit(str(uuid4()), p["documentRevision"], None, "Hobbies", "Piano")
    b = p["blocks"][0]
    p = await store.edit(
        str(uuid4()), p["documentRevision"], b["id"], "Hobbies", "Piano", action="delete"
    )
    assert "Piano" not in p["content"]
    with pytest.raises(InvalidArgumentError):
        await native(store, "## Hobbies\nPiano")
    p = await store.edit(
        str(uuid4()), p["documentRevision"], b["id"], "Hobbies", "Piano", action="unlock"
    )
    await native(store, "## Hobbies\nPiano")
    assert "Piano" in (await store.get())["content"]


@pytest.mark.asyncio
async def test_identity_is_protected_and_retry_has_one_revision(store):
    identity = {"preferredName": "Ben", "names": ["Ben"], "additionalEmails": []}
    p = await store.identity(str(uuid4()), 0, identity, True)
    op = str(uuid4())
    identity = {**identity, "preferredName": "Benjamin"}
    p = await store.identity(op, 0, identity)
    assert (await store.identity(op, 0, identity)) == p
    assert p["identityRevision"] == 1
    with pytest.raises(InvalidArgumentError):
        await native(store, p["content"].replace("Benjamin", "Someone"))


@pytest.mark.asyncio
async def test_interrupted_write_recovers_content_and_history_together(store):
    p = await store.get()
    write = store.fs.write_file

    async def fail(uri, *args, **kwargs):
        if uri == store.uri:
            raise OSError("disconnected")
        return await write(uri, *args, **kwargs)

    store.fs.write_file = fail
    op = str(uuid4())
    with pytest.raises(OSError):
        await store.edit(op, p["documentRevision"], None, "Goals", "Build Spark")
    store.fs.write_file = write
    recovered = await store.get()
    assert recovered["blocks"][0]["content"] == "Build Spark"
    assert len(await store.history()) == 1
    assert await store.edit(op, p["documentRevision"], None, "Goals", "Build Spark") == recovered


@pytest.mark.asyncio
async def test_concurrent_user_edits_and_cross_owner_isolation(store):
    p = await store.get()
    results = await asyncio.gather(
        *[store.edit(str(uuid4()), p["documentRevision"], None, "Goals", v) for v in ["A", "B"]],
        return_exceptions=True,
    )
    assert sum(isinstance(r, AlreadyExistsError) for r in results) == 1
    other = ProfileStore(
        store.fs, RequestContext(user=UserIdentifier("account", "bob"), role=Role.ROOT)
    )
    assert (await other.get())["blocks"] == []


@pytest.mark.asyncio
async def test_general_fs_cannot_delete_replace_or_move_profile(store):
    from openviking.service.fs_service import FSService

    service = FSService(viking_fs=store.fs)
    await store.get()
    for uri in [store.uri, store.journal, store.uri.rsplit("/", 1)[0]]:
        with pytest.raises(InvalidArgumentError):
            await service.rm(uri, store.ctx, recursive=True)
        with pytest.raises(InvalidArgumentError):
            await service.mv(uri, "viking://user/alice/memories/elsewhere.md", store.ctx)
    with pytest.raises(InvalidArgumentError):
        await service.mv("viking://user/alice/memories/elsewhere.md", store.uri, store.ctx)
    with pytest.raises(InvalidArgumentError):
        await service.read(store.journal, store.ctx)


@pytest.mark.asyncio
async def test_metadata_injection_cannot_replace_history(store):
    await store.get()
    with pytest.raises(InvalidArgumentError):
        await native(store, '## Hobbies\nPiano\n<!-- MEMORY_FIELDS {"profileDocument":{}} -->')
    assert (await store.get())["blocks"] == []


@pytest.mark.asyncio
async def test_grep_does_not_expose_superseded_profile_history(store):
    from unittest.mock import AsyncMock

    from openviking.storage.viking_fs import VikingFS

    p = await store.get()
    p = await store.edit(str(uuid4()), p["documentRevision"], None, "Hobbies", "Piano")
    b = p["blocks"][0]
    await store.edit(str(uuid4()), p["documentRevision"], b["id"], "Hobbies", "Violin")
    raw = await store.fs.read_file(store.uri)
    hits = [
        {"uri": store.uri, "line": i, "content": line}
        for i, line in enumerate(raw.splitlines(), 1)
        if "Piano" in line
    ]
    fake = SimpleNamespace(
        _ensure_access=lambda *args: None,
        stat=AsyncMock(),
        grep_config=None,
        _resolve_grep_engine=AsyncMock(return_value="fs"),
        _grep_fs=AsyncMock(return_value={"matches": hits}),
        _ctx_or_default=lambda ctx: ctx,
        read_file=store.fs.read_file,
    )
    result = await VikingFS.grep(fake, store.uri, "Piano", ctx=store.ctx)
    assert result["matches"] == []


@pytest.mark.asyncio
async def test_empty_initial_document_does_not_require_an_empty_embedding(store, monkeypatch):
    from unittest.mock import AsyncMock

    from openviking.session.memory.memory_updater import MemoryUpdater

    enqueue = AsyncMock(return_value=False)
    monkeypatch.setattr(MemoryUpdater, "refresh_file_embedding", enqueue)
    store.db = object()
    p = await store.get()
    assert p["blocks"] == []
    enqueue.assert_not_called()
    with pytest.raises(RuntimeError):
        await store.edit(str(uuid4()), p["documentRevision"], None, "Hobbies", "Piano")
    assert store.journal in store.fs.files
    from openviking.session.memory.dataclass import StoredLink

    link = StoredLink(from_uri=store.uri, to_uri="viking://user/alice/memories/entities/music.md")
    await ProfileStore(store.fs, store.ctx).add_links([link], [])
    import json

    assert json.loads(await store.fs.read_file(store.journal))["indexPending"] is True
    enqueue.return_value = True
    assert (await store.get())["blocks"][0]["content"] == "Piano"
    assert store.journal not in store.fs.files


@pytest.mark.asyncio
async def test_native_update_preserves_user_links(store):
    p = await store.get()
    p = await store.edit(
        str(uuid4()), p["documentRevision"], None, "Website", "My [website](https://example.test)."
    )
    old = await store.read()
    await native(store, old.plain_content() + "\n\n## Hobbies\nPiano", old)
    assert (await store.get())["blocks"][0]["content"] == "My [website](https://example.test)."


@pytest.mark.asyncio
async def test_link_updates_preserve_current_profile_and_history(store):
    from openviking.session.memory.dataclass import StoredLink
    from openviking.session.memory.memory_updater import write_stored_links

    p = await store.get()
    p = await store.edit(str(uuid4()), p["documentRevision"], None, "Goal", "Build a telescope")
    history = await store.history()
    link = StoredLink(
        from_uri=store.uri, to_uri="viking://user/alice/memories/entities/telescope.md"
    )
    await store.fs.write_file(link.to_uri, "Telescope project")
    await write_stored_links([link], store.ctx, store.fs)
    assert (await store.get()) == p
    assert (await store.read()).links[0]["to_uri"] == link.to_uri
    assert await store.history() == history
