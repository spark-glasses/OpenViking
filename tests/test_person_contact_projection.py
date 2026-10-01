"""Contact projection lifecycle, provenance preservation, and crash recovery."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.person_contact_store import PersonContactStore
from openviking.session.memory.person_identity import (
    CONTACT_SECTION_START,
    apply_person_identity,
    identity_directory_uri,
    identity_uri,
    load_active_people,
    load_person_identity,
    normalize_person_alias,
    person_aliases,
    person_memory_uri,
    preserve_person_identity,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import ConflictError, InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier

CONTACT = "11111111-1111-4111-8111-111111111111"
ANCHOR = "22222222-2222-4222-8222-222222222222"
OTHER = "33333333-3333-4333-8333-333333333333"


class FS:
    agfs = None

    def __init__(self):
        self.files = {}
        self.fail = None

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        if self.fail and self.fail(uri, content):
            self.fail = None
            raise OSError("interrupted write")
        self.files[uri] = content


@pytest.fixture
def env():
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    fs = FS()
    body = {
        "personId": CONTACT,
        "anchorId": ANCHOR,
        "revision": 1,
        "profile": {
            "displayName": "Sungryull Sohn",
            "givenName": "Sungryull",
            "familyName": "Sohn",
            "emails": ["sohn@example.test"],
            "organization": "Acme",
        },
    }
    return SimpleNamespace(
        fs=fs,
        ctx=ctx,
        body=body,
        store=PersonContactStore(fs, ctx),
        uri=person_memory_uri(ctx, ANCHOR),
    )


@pytest.mark.asyncio
async def test_first_sync_provisions_canonical_person_without_model(env):
    result = await env.store.sync(env.body)
    assert result["memoryUri"] == env.uri
    assert result["status"] == "synced"
    assert result["indexStatus"] == "unavailable"
    doc = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    assert doc.memory_type == "people"
    assert doc.extra_fields["anchorId"] == ANCHOR
    identity = await load_person_identity(env.fs, env.ctx, ANCHOR)
    assert identity["profile"]["emails"] == ["sohn@example.test"]
    assert identity["projectionState"] == "complete"
    assert (await load_active_people(env.fs, env.ctx))[0]["anchorId"] == ANCHOR
    assert (await env.store.sync(env.body))["status"] == "unchanged"


@pytest.mark.asyncio
async def test_contact_updates_preserve_accumulated_memory_and_other_metadata(env):
    memory = MemoryFile.from_parsed(
        uri=env.uri,
        parsed={
            "content": "He offered to review our prototype.",
            "email_source_refs": ["email:old-source"],
            "memory_type": "people",
        },
    )
    env.fs.files[env.uri] = MemoryFileUtils.write(memory)
    await env.store.sync(env.body)
    updated = {
        **env.body,
        "revision": 2,
        "profile": {**env.body["profile"], "organization": "Nova"},
    }
    await env.store.sync(updated)
    doc = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    assert "He offered to review our prototype." in doc.content
    assert doc.extra_fields["email_source_refs"] == ["email:old-source"]
    assert doc.extra_fields["contact_projection_revision"] == 2
    assert doc.content.count(CONTACT_SECTION_START) == 1
    assert (await env.store.sync(env.body))["status"] == "stale"
    assert (await load_person_identity(env.fs, env.ctx, ANCHOR))["profile"][
        "organization"
    ] == "Nova"


@pytest.mark.asyncio
async def test_same_revision_conflict_and_same_anchor_contact_id_remap(env):
    await env.store.sync(env.body)
    with pytest.raises(ConflictError):
        await env.store.sync({**env.body, "profile": {"displayName": "Different person"}})
    moved = await env.store.sync({**env.body, "revision": 2, "personId": OTHER})
    assert moved["memoryUri"] == env.uri
    assert moved["personId"] == OTHER
    with pytest.raises(ConflictError):
        await env.store.sync({**env.body, "anchorId": OTHER, "revision": 3})


@pytest.mark.asyncio
async def test_deleted_contact_is_tombstone_not_memory_deletion(env):
    await env.store.sync(env.body)
    doc = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    doc.content += "\n\nHistorical memory remains."
    env.fs.files[env.uri] = MemoryFileUtils.write(doc)
    result = await env.store.sync({**env.body, "revision": 2, "deleted": True})
    assert result["deleted"] is True
    assert "Historical memory remains." in MemoryFileUtils.read(env.fs.files[env.uri]).content
    assert await load_active_people(env.fs, env.ctx) == []
    assert (await env.store.sync(env.body))["status"] == "stale"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["markdown", "directory", "index", "receipt"])
async def test_same_revision_repairs_partial_projection_without_erasing_memory(env, failed_stage):
    env.store._index = AsyncMock(return_value="queued")
    env.store.db = SimpleNamespace(has_queue_manager=True)
    if failed_stage == "markdown":
        env.fs.fail = lambda uri, content: uri == env.uri
    elif failed_stage == "directory":
        env.fs.fail = lambda uri, content: (
            uri == identity_directory_uri(env.ctx) and bool(json.loads(content)["people"])
        )
    elif failed_stage == "receipt":
        env.fs.fail = lambda uri, content: (
            uri == identity_uri(env.ctx, ANCHOR)
            and json.loads(content)["projectionState"] == "complete"
        )
    else:
        env.store._index.side_effect = [RuntimeError("queue unavailable"), "queued"]
    with pytest.raises((OSError, RuntimeError)):
        await env.store.sync(env.body)
    pending = await load_person_identity(env.fs, env.ctx, ANCHOR)
    assert pending["projectionState"] == "pending"
    assert (await env.store.sync(env.body))["status"] == "synced"
    assert (await load_person_identity(env.fs, env.ctx, ANCHOR))["projectionState"] == "complete"
    assert (await env.store.sync(env.body))["status"] == "unchanged"


@pytest.mark.asyncio
async def test_contact_reservation_survives_crash_before_projection_record(env):
    env.fs.fail = lambda uri, content: uri == identity_uri(env.ctx, ANCHOR)
    with pytest.raises(OSError):
        await env.store.sync(env.body)
    with pytest.raises(ConflictError):
        await env.store.sync({**env.body, "anchorId": OTHER})
    assert (await env.store.sync(env.body))["status"] == "synced"


def test_deterministic_aliases_include_given_names_and_both_chinese_orders():
    aliases = person_aliases(
        {"displayName": "禧 贺", "givenName": "禧", "familyName": "贺", "nickname": "Hexi"}
    )
    assert "贺禧" in aliases
    assert "禧贺" in aliases
    assert "Hexi" in aliases
    assert "Sungryull" in person_aliases(
        {"displayName": "Sungryull Sohn", "givenName": "Sungryull", "familyName": "Sohn"}
    )
    assert normalize_person_alias("  SUNGRYULL   Sohn ") == "sungryull sohn"


@pytest.mark.asyncio
async def test_model_cannot_overwrite_registered_contact_identity(env):
    await env.store.sync(env.body)
    changed = MemoryFile.from_parsed(
        uri=env.uri,
        parsed={
            "content": "New durable conversation memory.",
            "anchorId": OTHER,
            "personId": OTHER,
        },
    )
    repaired = await preserve_person_identity(env.fs, env.ctx, env.uri, changed)
    assert repaired.extra_fields["anchorId"] == ANCHOR
    assert repaired.extra_fields["personId"] == CONTACT
    assert "New durable conversation memory." in repaired.content
    assert repaired.content.count(CONTACT_SECTION_START) == 1
    assert (
        apply_person_identity(repaired, await load_person_identity(env.fs, env.ctx, ANCHOR)).content
        == repaired.content
    )


@pytest.mark.asyncio
async def test_native_updater_preserves_managed_section(env):
    await env.store.sync(env.body)
    registry = MemoryTypeRegistry()
    updater = MemoryUpdater(registry=registry, vikingdb=None)
    updater._viking_fs = env.fs
    old = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    op = ResolvedOperation(
        memory_type="people",
        uris=[env.uri],
        old_memory_file_content=old,
        memory_fields={
            "content": {
                "blocks": [
                    {
                        "search": old.content,
                        "replace": "A new meeting established a project connection.",
                    }
                ]
            }
        },
    )
    await updater._apply_upsert(op, env.ctx)
    stored = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    assert stored.extra_fields["anchorId"] == ANCHOR
    assert stored.extra_fields["contact_projection_revision"] == 1
    assert "A new meeting established a project connection." in stored.content
    assert stored.content.count(CONTACT_SECTION_START) == 1


@pytest.mark.asyncio
async def test_owner_isolation_and_lookup_errors_fail_closed(env):
    await env.store.sync(env.body)
    other_ctx = RequestContext(user=UserIdentifier("account", "other"), role=Role.ROOT)
    assert await load_active_people(env.fs, other_ctx) == []
    assert await load_person_identity(env.fs, other_ctx, ANCHOR) is None
    raw = json.loads(env.fs.files[identity_uri(env.ctx, ANCHOR)])
    raw["userId"] = "other"
    env.fs.files[identity_uri(env.ctx, ANCHOR)] = json.dumps(raw)
    with pytest.raises(InvalidArgumentError):
        await preserve_person_identity(
            env.fs, env.ctx, env.uri, MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
        )


@pytest.mark.asyncio
async def test_upstream_unregistered_people_remain_supported(env):
    memory = MemoryFile.from_parsed(uri=env.uri, parsed={"content": "Standalone OV memory"})
    assert await preserve_person_identity(env.fs, env.ctx, env.uri, memory) is memory
    assert memory.content == "Standalone OV memory"


@pytest.mark.asyncio
async def test_projection_enqueues_embedding_directly_without_semantic_generation(env, monkeypatch):
    from openviking.session.memory import memory_updater

    queue = SimpleNamespace(
        has_queue_manager=True, enqueue_embedding_msg=AsyncMock(return_value=True)
    )
    generate = AsyncMock(
        side_effect=AssertionError("No semantic generation for contact projection")
    )
    monkeypatch.setattr(memory_updater.MemoryUpdater, "generate_overview", generate)
    store = PersonContactStore(env.fs, env.ctx, queue)
    result = await store.sync(env.body)
    assert result["indexStatus"] == "queued"
    queue.enqueue_embedding_msg.assert_awaited_once()
    message = queue.enqueue_embedding_msg.call_args.args[0]
    assert message.context_data["uri"] == env.uri
    assert message.context_data["owner_user_id"] == "alice"
    generate.assert_not_awaited()
    assert (await store.sync(env.body))["status"] == "unchanged"
    assert queue.enqueue_embedding_msg.await_count == 1


@pytest.mark.asyncio
async def test_projection_lock_uses_same_native_person_path(env, monkeypatch):
    import openviking.storage.transaction as transaction

    captured = []

    class Lock:
        def __init__(self, manager, paths, lock_mode):
            captured.append((paths, lock_mode))

        async def __aenter__(self):
            pass

        async def __aexit__(self, *args):
            pass

    env.fs.agfs = object()
    env.fs._uri_to_path = lambda uri, ctx: "account-path/" + uri
    monkeypatch.setattr(transaction, "get_lock_manager", lambda: object())
    monkeypatch.setattr(transaction, "LockContext", Lock)
    await env.store.sync(env.body)
    paths, mode = captured[0]
    assert mode == "exact"
    assert "account-path/" + env.uri in paths
    assert "account-path/" + identity_directory_uri(env.ctx) in paths
    assert "account-path/" + identity_uri(env.ctx, ANCHOR) in paths


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["start", "end"])
@pytest.mark.parametrize("legacy", [False, True])
async def test_missing_marker_is_repaired_before_native_apply_without_losing_narrative(
    env, missing, legacy
):
    from openviking.session.memory.dataclass import ResolvedOperations
    from openviking.session.memory.person_identity import CONTACT_SECTION_END
    from openviking.session.memory.session_extract_context_provider import (
        SessionExtractContextProvider,
    )

    await env.store.sync(env.body)
    old = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    if legacy:
        # Existing projections predate the new full-profile source link.
        old.content = "\n".join(
            line for line in old.content.splitlines() if '"fullProfileUri"' not in line
        )
    old.content += "\n\nEarlier invitation was not accepted."
    marker = CONTACT_SECTION_START if missing == "start" else CONTACT_SECTION_END
    proposed = old.content.replace(marker, "") + "\n\nHe now studies pottery on Saturdays."
    operation = ResolvedOperation(
        memory_type="people",
        uris=[env.uri],
        old_memory_file_content=old,
        memory_fields={"content": proposed},
    )
    provider = object.__new__(SessionExtractContextProvider)
    provider._ctx = None
    provider._read_file_contents = {env.uri: old}
    provider.validate_canonical_people_operations(
        ResolvedOperations(upsert_operations=[operation], delete_file_contents=[], errors=[])
    )
    updater = MemoryUpdater(registry=MemoryTypeRegistry(), vikingdb=None)
    updater._viking_fs = env.fs
    await updater._apply_upsert(operation, env.ctx)
    stored = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    assert "Earlier invitation was not accepted." in stored.content
    assert "He now studies pottery on Saturdays." in stored.content
    assert stored.content.count(CONTACT_SECTION_START) == 1
    assert stored.content.count(CONTACT_SECTION_END) == 1
    assert identity_uri(env.ctx, ANCHOR) in stored.content


@pytest.mark.asyncio
async def test_unknown_partial_contact_section_is_rejected_before_any_write(env):
    from openviking.session.memory.dataclass import ResolvedOperations
    from openviking.session.memory.person_identity import CONTACT_SECTION_END
    from openviking.session.memory.session_extract_context_provider import (
        SessionExtractContextProvider,
    )

    await env.store.sync(env.body)
    old = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    before = dict(env.fs.files)
    operation = ResolvedOperation(
        memory_type="people",
        uris=[env.uri],
        old_memory_file_content=old,
        memory_fields={
            "content": {
                "blocks": [
                    {
                        "search": old.content,
                        "replace": old.content.replace(CONTACT_SECTION_END, "").replace(
                            "Acme", "Invented"
                        ),
                    }
                ]
            }
        },
    )
    provider = object.__new__(SessionExtractContextProvider)
    provider._ctx = None
    provider._read_file_contents = {env.uri: old}
    with pytest.raises(ValueError):
        provider.validate_canonical_people_operations(
            ResolvedOperations(upsert_operations=[operation], delete_file_contents=[], errors=[])
        )
    assert env.fs.files == before


@pytest.mark.asyncio
async def test_large_valid_contact_keeps_full_source_with_explicit_bounded_projection(env):
    profile = {
        "displayName": "Given " * 120,
        "givenName": "G" * 255,
        "middleName": "M" * 255,
        "familyName": "F" * 255,
        "aliases": ["Alias" * 200],
        "notes": "A long imported note. " * 4000,
        "emails": [f"address-{i}@example.test" for i in range(130)] + ["x" * 2000],
        "phones": [str(i) for i in range(110)],
    }
    await env.store.sync({**env.body, "profile": profile})
    raw = await load_person_identity(env.fs, env.ctx, ANCHOR)
    assert raw["profile"]["notes"] == profile["notes"]
    assert raw["profile"]["emails"] == profile["emails"]
    assert raw["profile"]["phones"] == profile["phones"]
    assert raw["profile"]["aliases"] == profile["aliases"]
    assert "G" * 255 + "M" * 255 + "F" * 255 in raw["aliases"]
    doc = MemoryFileUtils.read(env.fs.files[env.uri], uri=env.uri)
    projection = json.loads(doc.content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert projection["fullProfileUri"] == identity_uri(env.ctx, ANCHOR)
    assert projection["projectionTruncated"]["notes"]["totalChars"] == len(profile["notes"])
    assert projection["projectionTruncated"]["emails"]["totalItems"] == 131
    assert len(projection["notes"]) == 4000
    assert len(projection["emails"]) == 32
    directory = await load_active_people(env.fs, env.ctx)
    assert directory[0]["aliases"] == raw["aliases"]
