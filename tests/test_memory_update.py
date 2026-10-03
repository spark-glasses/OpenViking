"""Contextual updates: data contracts, isolation, native routing and recovery."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openviking.message import Message, TextPart
from openviking.models.vlm.base import ToolCall
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.memory_update_context import MemoryUpdateContext
from openviking.session.memory.memory_update_context_provider import MemoryUpdateContextProvider
from openviking.session.memory.memory_update_store import MemoryUpdateStore
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig

PERSON = "viking://user/alice/memories/people/bob-anchor/memory.md"
ARCHIVE = "viking://user/alice/sessions/memory-update-op-1/history/archive_001"
REF = "conversation:chat-1/message:user-1"
EMAIL = "email:00000000-0000-4000-8000-000000000001"


class FS:
    agfs = None

    def __init__(self):
        self.files = {}

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content

    async def search(self, query, **kwargs):
        return SimpleNamespace(to_dict=lambda: {"memories": []})

    async def ls(self, uri, **kwargs):
        return []


@pytest.fixture
def env(monkeypatch):
    config = SimpleNamespace(memory=MemoryConfig(), vlm=SimpleNamespace(is_available=lambda: False))
    for module in (
        "openviking_cli.utils.config",
        "openviking.session.memory.session_extract_context_provider",
        "openviking.session.memory.memory_update_context_provider",
        "openviking.session.memory.extract_loop",
        "openviking.session.compressor_v2",
    ):
        monkeypatch.setattr(module + ".get_openviking_config", lambda: config)
    fs = FS()
    for module in ("openviking.session.compressor_v2", "openviking.session.memory.memory_updater"):
        monkeypatch.setattr(module + ".get_viking_fs", lambda: fs)
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    spec = {
        "operationId": "op-1",
        "text": "Update Bob's employer to Nova",
        "origin": {"conversationId": "chat-1", "turnId": "turn-1", "toolCallId": "tool-1"},
        "targets": [
            {
                "kind": "person",
                "personId": "bob-contact",
                "anchorId": "bob-anchor",
                "memoryUri": PERSON,
                "displayName": "Bob",
            }
        ],
        "messages": [
            {
                "id": "user-1",
                "role": "user",
                "content": "I mean Bob from Acme. He joined Nova.",
                "sourceRef": REF,
            },
            {
                "id": "tool-1",
                "role": "tool",
                "content": '{"success":true,"personId":"bob-contact","anchorId":"bob-anchor"}',
            },
        ],
        "sourceKinds": ["email", "transcript"],
    }

    def provider(**changes):
        return MemoryUpdateContextProvider(
            memory_update_context={**spec, **changes},
            archive_uri=ARCHIVE,
            messages=[Message(id="task", role="user", parts=[TextPart(spec["text"])])],
            ctx=ctx,
            viking_fs=fs,
        )

    return SimpleNamespace(config=config, fs=fs, ctx=ctx, spec=spec, provider=provider)


def test_context_owner_and_tools(env):
    context = MemoryUpdateContext.model_validate(env.spec)
    context.validate_owner(env.ctx)
    provider = env.provider()
    assert set(provider.get_tools()) == {
        "read",
        "search",
        "readContext",
        "searchSources",
        "readSource",
    }
    assert {s.memory_type for s in provider.get_memory_schemas(env.ctx)} <= {
        "people",
        "profile",
        "preferences",
        "entities",
        "events",
        "questions",
    }
    with pytest.raises(ValueError):
        env.provider(
            targets=[{**env.spec["targets"][0], "memoryUri": PERSON.replace("alice", "other")}]
        )
    assert (
        MemoryUpdateContext.model_validate({**env.spec, "messages": env.spec["messages"] * 2})
        .messages[2]
        .sourceRef
        == REF
    )


@pytest.mark.asyncio
async def test_snapshot_is_complete_pageable_and_preserves_tool_roles(env):
    messages = [{**env.spec["messages"][0], "content": "A" * 45000}, env.spec["messages"][1]]
    provider = env.provider(messages=messages)
    chunks, offset = [], 0
    while True:
        page = await provider.execute_tool(
            ToolCall(str(offset), "readContext", {"offset": offset, "limit": 10000})
        )
        chunks.append(page["text"])
        if page["nextOffset"] is None:
            break
        offset = page["nextOffset"]
    recovered = json.loads("".join(chunks))
    assert recovered[0]["content"] == "A" * 45000
    assert recovered[0]["sourceRef"] == REF
    assert recovered[1]["role"] == "tool"
    assert json.loads(recovered[1]["content"])["personId"] == "bob-contact"
    assert len(provider.evidence) == len(chunks)
    await provider.execute_tool(ToolCall("again", "readContext", {"offset": 0, "limit": 10000}))
    assert len(provider.evidence) == len(chunks)


@pytest.mark.asyncio
async def test_cannot_execute_arbitrary_tool_or_read_other_user(env):
    provider = env.provider()
    for name, args in (
        ("read", {"uri": PERSON.replace("alice", "other")}),
        ("read", {"uri": PERSON.replace("people", "../people")}),
        ("shell", {"command": "anything"}),
    ):
        with pytest.raises(ValueError):
            await provider.execute_tool(ToolCall("x", name, args))


@pytest.mark.asyncio
async def test_confirmed_person_creation_and_existing_cross_document_updates(env):
    provider = env.provider()
    await provider.prefetch()
    op = ResolvedOperation(
        memory_type="people",
        uris=[PERSON],
        memory_fields={"anchorId": "bob-anchor", "content": "Bob is at Nova"},
    )
    provider.validate_operations(
        ResolvedOperations(upsert_operations=[op], delete_file_contents=[], errors=[])
    )
    assert op.memory_fields["memory_update_source_refs"] == [REF]
    other = ResolvedOperation(
        memory_type="people",
        uris=[PERSON.replace("bob-anchor", "invented")],
        memory_fields={"content": "Unknown"},
    )
    with pytest.raises(ValueError):
        provider.validate_operations(
            ResolvedOperations(upsert_operations=[other], delete_file_contents=[], errors=[])
        )
    with pytest.raises(ValueError):
        provider.validate_operations(
            ResolvedOperations(
                upsert_operations=[
                    ResolvedOperation(memory_type="identity", uris=[PERSON], memory_fields={})
                ],
                delete_file_contents=[],
                errors=[],
            )
        )


@pytest.mark.asyncio
async def test_sources_scoped_and_original_version_preserved(env, monkeypatch):
    env.config.memory.source_base_url = "https://source.invalid"
    env.config.memory.source_api_key = "server-secret"
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "sourceRef": EMAIL,
                "contentVersion": "v1",
                "body": "Bob joined Nova",
                "nextCursor": "opaque",
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    provider = env.provider(sourceKinds=["email"])
    result = await provider.execute_tool(
        ToolCall(
            "read", "readSource", {"sourceRef": EMAIL, "sourceVersion": "v1", "cursor": "opaque"}
        )
    )
    assert result["sourceRef"] == EMAIL
    request = requests[0]
    assert request.url.path == "/internal/memory/sources/read"
    assert request.headers["X-Spark-User-Id"] == "alice"
    assert request.headers["X-Spark-Memory-Operation-Id"] == "op-1"
    assert json.loads(request.content)["sourceVersion"] == "v1"
    assert json.loads(request.content)["cursor"] == "opaque"
    assert EMAIL in provider._source_refs
    with pytest.raises(ValueError):
        await provider.execute_tool(
            ToolCall("wrong", "readSource", {"sourceRef": EMAIL.replace("email", "transcript")})
        )
    with pytest.raises(ValueError):
        await provider.execute_tool(
            ToolCall("wrong", "searchSources", {"query": "Bob", "kinds": ["transcript"]})
        )


@pytest.mark.asyncio
async def test_operation_retry_same_input_does_not_duplicate_native_run(env):
    compressor = SimpleNamespace(extract_long_term_memories=AsyncMock())

    async def extract(**kwargs):
        await kwargs["memory_update_before_apply"](
            ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
        )
        await kwargs["memory_update_after_apply"](
            {
                "writtenUris": [],
                "editedUris": [],
                "errors": [],
                "questionRefs": [],
                "sourceRefs": [REF],
                "changed": False,
            }
        )

    compressor.extract_long_term_memories.side_effect = extract
    store = MemoryUpdateStore(env.fs, env.ctx, compressor)
    store._launch = lambda operation_id: None
    first = await store.submit(env.spec)
    assert first["status"] == "accepted"
    await asyncio.gather(store.run("op-1"), store.run("op-1"))
    final = await store.submit(env.spec)
    assert final["status"] == "noChange"
    assert compressor.extract_long_term_memories.call_count == 1
    with pytest.raises(AlreadyExistsError):
        await store.submit({**env.spec, "text": "different"})
    other = MemoryUpdateStore(
        env.fs, RequestContext(user=UserIdentifier("account", "other"), role=Role.ROOT), compressor
    )
    with pytest.raises(NotFoundError):
        await other.get("op-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("after_apply", [False, True])
async def test_only_pre_apply_failures_can_retry(env, after_apply):
    async def extract(**kwargs):
        if after_apply:
            await kwargs["memory_update_before_apply"](
                ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
            )
        raise OSError("interrupted")

    store = MemoryUpdateStore(env.fs, env.ctx, SimpleNamespace(extract_long_term_memories=extract))
    store._launch = lambda operation_id: None
    await store.submit(env.spec)
    await store.run("op-1")
    result = await store.get("op-1")
    assert result["status"] == ("needsRecovery" if after_apply else "failed")
    if after_apply:
        with pytest.raises(InvalidArgumentError):
            await store.retry("op-1")
    else:
        assert (await store.retry("op-1"))["status"] == "accepted"


@pytest.mark.asyncio
async def test_native_compressor_selects_update_provider_and_callbacks(env, monkeypatch):
    captured = {}

    class Loop:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.context_provider = kwargs["context_provider"]

        async def run(self):
            assert isinstance(self.context_provider, MemoryUpdateContextProvider)
            await self.context_provider.prefetch()
            return ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[]), []

    monkeypatch.setattr(
        SessionCompressorV2, "_get_or_create_react", lambda self, **kwargs: Loop(**kwargs)
    )
    before, after = AsyncMock(), AsyncMock()
    await SessionCompressorV2(None).extract_long_term_memories(
        messages=[Message(id="task", role="user", parts=[TextPart("update")])],
        ctx=env.ctx,
        archive_uri=ARCHIVE,
        memory_update_context=env.spec,
        memory_update_before_apply=before,
        memory_update_after_apply=after,
        strict_extract_errors=True,
    )
    before.assert_awaited_once()
    after.assert_awaited_once()
    assert after.call_args.args[0]["changed"] is False
    assert "memory_update_result.json" in " ".join(env.fs.files)


def test_message_provenance_round_trips():
    message = Message(
        id="stable-id",
        role="user",
        parts=[TextPart("Bob")],
        metadata={"sourceRef": REF, "memoryOperations": [{"operationId": "op-1"}]},
    )
    loaded = Message.from_dict(message.to_dict())
    assert loaded.id == message.id
    assert loaded.metadata == message.metadata


@pytest.mark.asyncio
async def test_latest_original_and_receipt_survive_long_context(env):
    messages = [{"id": "old", "role": "user", "content": "A" * 45000}, *env.spec["messages"]]
    provider = env.provider(messages=messages)
    prefetched = await provider.prefetch()
    recent = next(
        json.loads(message["content"])["recentEvidence"]
        for message in prefetched
        if message.get("role") == "user" and "recentEvidence" in json.loads(message["content"])
    )
    assert (
        next(m for m in recent if m["id"] == "user-1")["content"]
        == env.spec["messages"][0]["content"]
    )
    assert (
        json.loads(next(m for m in recent if m["role"] == "tool")["content"])["anchorId"]
        == "bob-anchor"
    )
    page = await provider.execute_tool(ToolCall("original", "readContext", {"messageIndex": 2}))
    assert json.loads(page["text"])["role"] == "tool"


@pytest.mark.asyncio
async def test_anchor_cannot_change_even_when_uri_is_allowed(env):
    provider = env.provider()
    await provider.prefetch()
    op = ResolvedOperation(
        memory_type="people",
        uris=[PERSON],
        memory_fields={"anchorId": "different", "content": "Bob"},
    )
    with pytest.raises(ValueError):
        provider.validate_operations(
            ResolvedOperations(upsert_operations=[op], delete_file_contents=[], errors=[])
        )


@pytest.mark.asyncio
async def test_question_only_result_is_needs_clarification(env):
    async def extract(**kwargs):
        await kwargs["memory_update_before_apply"](
            ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
        )
        question = {
            "questionId": "q1",
            "uri": "viking://user/alice/memories/questions.md",
            "text": "Which Bob?",
        }
        await kwargs["memory_update_after_apply"](
            {
                "writtenUris": [question["uri"]],
                "editedUris": [],
                "nonQuestionUris": [],
                "errors": [],
                "questionRefs": [question],
                "unresolvedItems": [question],
                "sourceRefs": [REF],
                "changed": True,
            }
        )

    store = MemoryUpdateStore(env.fs, env.ctx, SimpleNamespace(extract_long_term_memories=extract))
    store._launch = lambda _: None
    await store.submit(env.spec)
    await store.run("op-1")
    outcome = await store.get("op-1")
    assert outcome["status"] == "needsClarification"
    assert outcome["unresolvedItems"][0]["text"] == "Which Bob?"


@pytest.mark.asyncio
async def test_message_receipt_prevents_duplicate_after_archive(env, monkeypatch):
    import openviking.storage.transaction as transaction
    from openviking.server.routers.sessions import AddMessageRequest, _add_identified_message

    class Lock:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            pass

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(transaction, "get_lock_manager", lambda: None)
    monkeypatch.setattr(transaction, "LockContext", Lock)
    env.fs._uri_to_path = lambda uri, ctx: uri
    session = SimpleNamespace(messages=[])

    def add(specs):
        for spec in specs:
            session.messages.append(
                Message(
                    id=spec["id"], role=spec["role"], parts=spec["parts"], metadata=spec["metadata"]
                )
            )

    session.add_messages = add
    service = SimpleNamespace(
        viking_fs=env.fs, sessions=SimpleNamespace(get=AsyncMock(return_value=session))
    )
    body = AddMessageRequest(
        id="user-1", role="user", content="Bob joined Nova", metadata={"sourceRef": REF}
    )
    first = await _add_identified_message(service, "chat-1", body, env.ctx)
    assert first["duplicate"] is False
    original = session.messages[0].to_dict()
    session.messages.clear()
    second = await _add_identified_message(service, "chat-1", body, env.ctx)
    assert second["duplicate"] is True
    assert session.messages == []
    # Crash after append, before receipt; source survives an intervening archive.
    env.fs.files = {
        key: value for key, value in env.fs.files.items() if "message_receipts" not in key
    }
    env.fs.files["viking://user/alice/sessions/chat-1/history/archive_001/messages.jsonl"] = (
        json.dumps(original)
    )
    env.fs.ls = AsyncMock(return_value=[{"name": "archive_001"}])
    assert (await _add_identified_message(service, "chat-1", body, env.ctx))["duplicate"] is True
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await _add_identified_message(
            service, "chat-1", body.model_copy(update={"content": "changed"}), env.ctx
        )
    assert exc.value.status_code == 409


def test_contextual_budget_config_is_independent_of_vlm_output(env):
    config = MemoryConfig(
        memory_update_max_input_tokens=2048,
        memory_update_max_source_chars=4000,
        memory_update_max_tool_calls=2,
    )
    restored = MemoryConfig.from_dict(config.to_dict())
    assert restored.memory_update_max_input_tokens == 2048
    env.config.memory = restored
    provider = env.provider()
    assert provider.max_input_tokens == 2048
    assert provider.max_source_chars == 4000
    assert provider.max_tool_calls == 2
    assert env.config.vlm.is_available() is False


@pytest.mark.asyncio
async def test_native_model_call_counts_actual_messages_and_tool_schema(env):
    from openviking.models.vlm.base import VLMResponse
    from openviking.session.memory.extract_loop import ExtractLoop

    env.config.memory.memory_update_max_input_tokens = 1024
    provider = env.provider()
    vlm = SimpleNamespace(
        model="test",
        get_completion_async=AsyncMock(
            return_value=VLMResponse(tool_calls=[ToolCall("next", "readContext", {})])
        ),
    )
    loop = ExtractLoop(vlm, viking_fs=env.fs, ctx=env.ctx, context_provider=provider)
    messages = [{"role": "user", "content": "A" * 1000}]
    loop._tool_schemas = [{"type": "function", "function": {"name": "readContext"}}]
    await loop._call_llm(messages)
    assert vlm.get_completion_async.await_count == 1
    loop._tool_schemas = [
        {
            "type": "function",
            "function": {"name": "readContext", "parameters": {"description": "A" * 5000}},
        }
    ]
    with pytest.raises(RuntimeError):
        await loop._call_llm(messages)
    assert vlm.get_completion_async.await_count == 1


@pytest.mark.asyncio
async def test_cached_context_repetitions_count_toward_actual_input_budget(env):
    from openviking.models.vlm.base import VLMResponse
    from openviking.session.memory.extract_loop import ExtractLoop
    from openviking.session.memory.tools import add_tool_call_pair_to_messages

    env.config.memory.memory_update_max_input_tokens = 2048
    provider = env.provider(messages=[{"id": "large", "role": "user", "content": "A" * 5000}])
    vlm = SimpleNamespace(
        model="test",
        get_completion_async=AsyncMock(
            return_value=VLMResponse(tool_calls=[ToolCall("next", "readContext", {})])
        ),
    )
    loop = ExtractLoop(vlm, viking_fs=env.fs, ctx=env.ctx, context_provider=provider)
    loop._tool_schemas = []
    args = {"offset": 0, "limit": 4500}
    first = await provider.execute_tool(ToolCall("first", "readContext", args))
    source_chars = provider._source_chars
    messages = []
    add_tool_call_pair_to_messages(messages, "first", "readContext", args, first)
    await loop._call_llm(messages)
    cached = await provider.execute_tool(ToolCall("second", "readContext", args))
    assert provider._source_chars == source_chars
    add_tool_call_pair_to_messages(messages, "second", "readContext", args, cached)
    with pytest.raises(RuntimeError):
        await loop._call_llm(messages)
    assert vlm.get_completion_async.await_count == 1


@pytest.mark.asyncio
async def test_input_budget_failure_preserves_snapshot_and_never_applies(env, monkeypatch):
    from openviking.models.vlm.base import VLMResponse
    from openviking.session.memory.extract_loop import ExtractLoop

    env.config.memory.memory_update_max_input_tokens = 1024
    vlm = SimpleNamespace(model="test", get_completion_async=AsyncMock(return_value=VLMResponse()))
    monkeypatch.setattr(
        SessionCompressorV2,
        "_get_or_create_react",
        lambda self, **kwargs: ExtractLoop(
            vlm,
            viking_fs=env.fs,
            ctx=env.ctx,
            context_provider=kwargs["context_provider"],
            isolation_handler=kwargs["isolation_handler"],
        ),
    )
    apply = AsyncMock()
    monkeypatch.setattr(
        SessionCompressorV2,
        "_get_or_create_updater",
        lambda *args, **kwargs: SimpleNamespace(apply_operations=apply),
    )
    store = MemoryUpdateStore(env.fs, env.ctx, SessionCompressorV2(None))
    store._launch = lambda _: None
    await store.submit(env.spec)
    frozen = env.fs.files[ARCHIVE + "/memory_update_context.json"]
    await store.run("op-1")
    result = await store.get("op-1")
    assert result["status"] == "failed"
    assert result["retryable"] is True
    assert result["appliedUris"] == []
    assert env.fs.files[ARCHIVE + "/memory_update_context.json"] == frozen
    vlm.get_completion_async.assert_not_awaited()
    apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_configured_source_and_tool_limits_are_enforced(env):
    env.config.memory.memory_update_max_tool_calls = 1
    provider = env.provider()
    await provider.execute_tool(ToolCall("first", "readContext", {"limit": 10}))
    with pytest.raises(RuntimeError):
        await provider.execute_tool(ToolCall("second", "readContext", {"limit": 10}))
    env.config.memory.memory_update_max_source_chars = 1000
    provider = env.provider(messages=[{"id": "long", "role": "user", "content": "A" * 2000}])
    with pytest.raises(RuntimeError):
        await provider.execute_tool(ToolCall("large", "readContext", {}))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entities", "events"])
async def test_requested_new_matter_needs_verified_absence_not_a_fabricated_target(env, kind):
    uri = f"viking://user/alice/memories/{kind}/new-project.md"
    provider = env.provider(text="Remember that I started a project called New Project", targets=[])
    operation = ResolvedOperation(
        memory_type=kind, uris=[uri], memory_fields={"content": "The user started New Project"}
    )
    operations = ResolvedOperations(
        upsert_operations=[operation], delete_file_contents=[], errors=[]
    )
    with pytest.raises(ValueError):
        provider.validate_operations(operations)
    await provider.execute_tool(ToolCall("missing", "read", {"uri": uri}))
    provider.validate_operations(operations)
    assert uri in provider._missing_uris
    assert operation.memory_fields["memory_update_operation_id"] == "op-1"
    wrong = ResolvedOperation(
        memory_type=kind, uris=[PERSON], memory_fields={"content": "Wrong folder"}
    )
    await provider.execute_tool(ToolCall("person", "read", {"uri": PERSON}))
    with pytest.raises(ValueError):
        provider.validate_operations(
            ResolvedOperations(upsert_operations=[wrong], delete_file_contents=[], errors=[])
        )


@pytest.mark.asyncio
async def test_contact_full_profile_read_is_scoped_and_preserves_line_pagination(env):
    from openviking.session.memory.person_identity import identity_uri

    uri = identity_uri(env.ctx, "11111111-1111-4111-8111-111111111111")
    env.fs.files[uri] = '{\n  "profile": {\n    "notes": "original full notes"\n  }\n}'
    provider = env.provider()
    result = await provider.execute_tool(ToolCall("profile", "read", {
        "uri": uri, "offset": 2, "limit": 1,
    }))
    assert "original full notes" in result["content"]
    assert '"profile"' not in result["content"]
    assert uri not in provider._fully_read
    for forbidden in (
        uri.replace("alice", "other"),
        uri.replace("/profile.json", "/../profile.json"),
        uri.replace("11111111-1111-4111-8111-111111111111", "directory"),
    ):
        with pytest.raises(ValueError):
            await provider.execute_tool(ToolCall("invalid", "read", {"uri": forbidden}))


def test_composed_contact_name_is_accepted_as_explicit_target(env):
    display_name = "G" * 255 + " " + "M" * 255 + " " + "F" * 255
    spec = {**env.spec, "targets": [{**env.spec["targets"][0], "displayName": display_name}]}
    context = MemoryUpdateContext.model_validate(spec)
    context.validate_owner(env.ctx)
    assert context.targets[0].displayName == display_name
