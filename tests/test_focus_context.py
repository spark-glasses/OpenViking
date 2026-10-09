from types import SimpleNamespace

import pytest

from openviking.session.memory.focus_context import FocusContext
from openviking.session.memory.memory_write_context import MemoryWriteContext

pytest_plugins = ["tests.test_focus_store"]


def context(store, *, confirmed=True):
    write = MemoryWriteContext(store.fs, store.ctx)
    provider = SimpleNamespace(
        _viking_fs=store.fs,
        _ctx=store.ctx,
        get_memory_write_context=lambda: write,
        register_memory_read=lambda uri, page: write.read_files.update({uri: page}),
        has_unconfirmed_speakers=lambda: not confirmed,
        read_file_contents=write.read_files,
    )
    return FocusContext(provider), write


@pytest.mark.asyncio
async def test_discovery_requires_read_source_and_registers_canonical_memory(store):
    focuses, write = context(store)
    args = dict(
        name="Health",
        intent="The user is prioritizing sleep",
        evidence=[
            {"sourceRef": "conversation:one", "quote": "I want to sleep better"},
        ],
    )
    with pytest.raises(ValueError):
        await focuses.execute("ensureFocus", args)
    assert (await store.list())["focuses"] == []
    write.remember("conversation:one", "I want to sleep better", kind="userAnswer")
    with pytest.raises(ValueError):
        await focuses.execute(
            "ensureFocus", args | {"evidence": [{"sourceRef": "conversation:one", "quote": " "}]}
        )
    first = await focuses.execute("ensureFocus", args)
    repeated = await focuses.execute("ensureFocus", args)
    assert first["focusId"] == repeated["focusId"]
    assert first["uri"] in write.read_files
    assert first["status"] == "forming"
    assert first["intent"] == "The user is prioritizing sleep" and first["intentSource"] == "guessed"
    assert (await store.list(status="active"))["focuses"] == []


@pytest.mark.asyncio
async def test_unconfirmed_speaker_cannot_create_or_modify_focus(store):
    focuses, write = context(store, confirmed=False)
    write.remember("transcript:one", "I care about health")
    with pytest.raises(ValueError):
        await focuses.execute(
            "ensureFocus",
            dict(
                name="Health",
                intent="A speaker said so",
                evidence=[
                    {"sourceRef": "transcript:one", "quote": "I care about health"},
                ],
            ),
        )
    existing = await store.ensure(name="Study", origin="user")
    write.read_files[existing["uri"]] = await store.get(existing["focusId"])
    op = SimpleNamespace(
        memory_type="focuses",
        uris=[existing["uri"]],
        memory_fields={"focusId": existing["focusId"]},
    )
    with pytest.raises(ValueError):
        focuses.validate(SimpleNamespace(upsert_operations=[op]))
    assert (await store.get(existing["focusId"])).extra_fields["revision"] == 1


@pytest.mark.asyncio
async def test_validation_uses_read_revision_and_rejects_another_target(store):
    focuses, write = context(store)
    existing = await store.ensure(name="Study", origin="user")
    op = SimpleNamespace(
        memory_type="focuses",
        uris=[existing["uri"]],
        memory_fields={"focusId": existing["focusId"]},
    )
    operations = SimpleNamespace(upsert_operations=[op])
    with pytest.raises(ValueError):
        focuses.validate(operations)
    page = await store.get(existing["focusId"])
    write.read_files[page.uri] = page
    focuses.validate(operations)
    assert op.old_memory_file_content is page
    op.uris = [existing["uri"].replace("alice", "bob")]
    with pytest.raises(ValueError):
        focuses.validate(operations)
