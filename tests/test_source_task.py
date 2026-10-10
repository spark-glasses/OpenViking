"""Source tasks: work on an outside source that writes no memory."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openviking.message import Message, TextPart
from openviking.models.vlm.base import ToolCall, VLMResponse
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_update_context import MemoryUpdateContext
from openviking.session.memory.memory_update_context_provider import MemoryUpdateContextProvider
from openviking.session.memory.memory_update_store import MemoryUpdateStore
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig

ARCHIVE = "viking://user/alice/sessions/memory-update-task-1/history/archive_001"
ANSWER = {
    "name": "answerAccount",
    "description": "Say who an account is.",
    "parameters": {
        "type": "object",
        "properties": {"account": {"type": "string"}, "person": {"type": "string"}},
        "required": ["account"],
        "additionalProperties": False,
    },
}


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
    config.memory.source_base_url = "https://backend.invalid"
    config.memory.source_api_key = "bridge-secret"
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
        "operationId": "task-1",
        "text": "Say who each of these accounts is.",
        "origin": {"kind": "automation", "runId": "job-1"},
        "sourceTask": {
            "source": "slack",
            "tools": [ANSWER],
            "budget": {"toolCalls": 3, "sourceChars": 5000, "seconds": 45},
        },
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


def bridge(monkeypatch, handler):
    """Answers the provider's bridge calls with `handler`, and keeps the requests."""
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(record), **kwargs),
    )
    return requests


def test_a_source_task_carries_only_its_text_tools_and_budget(env):
    task = MemoryUpdateContext.model_validate(env.spec)
    assert task.reasoning_seconds() == 45
    assert MemoryUpdateContext.model_validate(
        {k: v for k, v in env.spec.items() if k != "sourceTask"}
    ).reasoning_seconds() > 45
    for extra in (
        {"messages": [{"id": "m", "role": "user", "content": "hello"}]},
        {"sourceKinds": ["email"]},
        {
            "collaboration": {
                "mode": "initial",
                "scopes": [
                    {"provider": "slack", "connectionId": "c", "workspaceId": "T", "selfId": "U"}
                ],
            }
        },
    ):
        with pytest.raises(ValueError):
            MemoryUpdateContext.model_validate({**env.spec, **extra})
    for tools in ([ANSWER, ANSWER], [{**ANSWER, "name": "read"}]):
        with pytest.raises(ValueError):
            MemoryUpdateContext.model_validate(
                {**env.spec, "sourceTask": {**env.spec["sourceTask"], "tools": tools}}
            )


def test_the_model_gets_memory_reading_and_the_tools_the_task_brought(env):
    provider = env.provider()
    assert provider.get_tools() == ["read", "search", "searchPeople", "answerAccount"]
    assert [tool.to_schema()["function"]["name"] for tool in provider.task_tools()] == [
        "answerAccount"
    ]
    assert provider.get_memory_schemas(env.ctx) == []
    assert provider.question_writes_enabled is False


@pytest.mark.asyncio
async def test_a_task_tool_is_carried_out_by_the_bridge_of_its_source(env, monkeypatch):
    requests = bridge(
        monkeypatch,
        lambda request: httpx.Response(200, json={"success": True, "data": {"saved": True}}),
    )
    provider = env.provider()
    result = await provider.execute_tool(
        ToolCall("1", "answerAccount", {"account": "T1:U1", "person": "new"})
    )
    assert result == {"success": True, "data": {"saved": True}}
    [request] = requests
    assert str(request.url) == "https://backend.invalid/internal/memory/slack"
    assert request.headers["Authorization"] == "Bearer bridge-secret"
    assert request.headers["X-User-Id"] == "alice"
    assert request.headers["X-Memory-Operation-Id"] == "task-1"
    assert json.loads(request.content) == {
        "tool": "answerAccount",
        "arguments": {"account": "T1:U1", "person": "new"},
    }


@pytest.mark.asyncio
async def test_a_failed_call_goes_back_to_the_model_and_a_lost_operation_ends_the_task(
    env, monkeypatch
):
    answers = iter(
        [
            httpx.Response(200, json={"success": False, "error": "That account was not handed over"}),
            httpx.Response(404, json={"success": False, "error": "operationUnavailable"}),
        ]
    )
    bridge(monkeypatch, lambda request: next(answers))
    provider = env.provider()
    refused = await provider.execute_tool(ToolCall("1", "answerAccount", {"account": "T1:U9"}))
    assert refused["success"] is False
    with pytest.raises(httpx.HTTPStatusError):
        await provider.execute_tool(ToolCall("2", "answerAccount", {"account": "T1:U1"}))


@pytest.mark.asyncio
async def test_the_task_s_own_budget_bounds_its_calls(env, monkeypatch):
    bridge(monkeypatch, lambda request: httpx.Response(200, json={"success": True}))
    provider = env.provider()
    for index in range(3):
        await provider.execute_tool(ToolCall(str(index), "answerAccount", {"account": f"T1:U{index}"}))
    with pytest.raises(RuntimeError):
        await provider.execute_tool(ToolCall("over", "answerAccount", {"account": "T1:U3"}))


def test_a_source_task_may_not_write_memory(env):
    provider = env.provider()
    provider.validate_operations(
        ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
    )
    with pytest.raises(ValueError):
        provider.validate_operations(
            ResolvedOperations(
                upsert_operations=[
                    ResolvedOperation(
                        memory_type="entities",
                        uris=["viking://user/alice/memories/entities/acme.md"],
                        memory_fields={},
                    )
                ],
                delete_file_contents=[],
                errors=[],
            )
        )


@pytest.mark.asyncio
async def test_the_loop_offers_the_task_tools_and_ends_on_an_empty_answer(env, monkeypatch):
    requests = bridge(
        monkeypatch, lambda request: httpx.Response(200, json={"success": True})
    )
    vlm = SimpleNamespace(
        model="test",
        get_completion_async=AsyncMock(
            side_effect=[
                VLMResponse(tool_calls=[ToolCall("1", "answerAccount", {"account": "T1:U1"})]),
                VLMResponse(content="{}"),
            ]
        ),
    )
    provider = env.provider()
    await provider.prepare_extraction_messages()
    isolation = MemoryIsolationHandler(
        env.ctx,
        provider.get_extract_context(),
        allowed_memory_types=set(),
        allow_self=True,
        allowed_peer_ids=set(),
    )
    isolation.prepare_messages()
    loop = ExtractLoop(
        vlm,
        viking_fs=env.fs,
        ctx=env.ctx,
        context_provider=provider,
        isolation_handler=isolation,
    )
    operations, _ = await loop.run()
    offered = [
        tool["function"]["name"]
        for tool in vlm.get_completion_async.await_args_list[0].kwargs["tools"]
    ]
    assert "answerAccount" in offered
    assert not {"listFocuses", "ensureFocus"} & set(offered)
    assert len(requests) == 1
    assert operations is not None and not operations.upsert_operations


@pytest.mark.asyncio
async def test_a_source_task_takes_no_memory_locks_and_runs_within_its_own_bounds(
    env, monkeypatch
):
    env.fs.agfs = object()

    def no_locks(*args, **kwargs):
        raise AssertionError("a source task asked for memory locks")

    monkeypatch.setattr("openviking.storage.transaction.get_lock_manager", no_locks)
    monkeypatch.setattr("openviking.storage.transaction.init_lock_manager", no_locks)
    seen = {}

    class Loop:
        def __init__(self, **kwargs):
            self.context_provider = kwargs["context_provider"]
            self.max_iterations = 0

        async def run(self):
            seen["turns"] = self.max_iterations
            await self.context_provider.prefetch()
            return ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[]), []

    monkeypatch.setattr(
        SessionCompressorV2, "_get_or_create_react", lambda self, **kwargs: Loop(**kwargs)
    )
    after = AsyncMock()
    await SessionCompressorV2(None).extract_long_term_memories(
        messages=[Message(id="task", role="user", parts=[TextPart("task")])],
        ctx=env.ctx,
        archive_uri=ARCHIVE,
        memory_update_context=env.spec,
        memory_update_before_apply=AsyncMock(),
        memory_update_after_apply=after,
        strict_extract_errors=True,
    )
    # Every call the budget allows can have a turn of its own.
    assert seen["turns"] >= env.spec["sourceTask"]["budget"]["toolCalls"]
    after.assert_awaited_once()
    assert after.call_args.args[0]["changed"] is False


@pytest.mark.asyncio
async def test_an_operation_s_deadline_follows_the_time_its_task_is_given(env):
    spec = {
        **env.spec,
        "sourceTask": {**env.spec["sourceTask"], "budget": {**env.spec["sourceTask"]["budget"], "seconds": 900}},
    }
    started = {}

    class Compressor:
        async def extract_long_term_memories(self, **kwargs):
            record = json.loads(
                env.fs.files[
                    "viking://user/alice/sessions/memory-update-task-1/memory_update.json"
                ]
            )
            started.update(deadline=record["deadlineAt"], updated=record["updatedAt"])
            await kwargs["memory_update_after_apply"](
                {
                    "writtenUris": [],
                    "editedUris": [],
                    "errors": [],
                    "questionRefs": [],
                    "sourceRefs": [],
                    "changed": False,
                }
            )

    store = MemoryUpdateStore(env.fs, env.ctx, Compressor())
    await store.submit(spec)
    import asyncio

    for _ in range(50):
        if (await store.get("task-1"))["status"] == "noChange":
            break
        await asyncio.sleep(0.01)
    assert (await store.get("task-1"))["status"] == "noChange"
    assert started["deadline"] - started["updated"] > 900
