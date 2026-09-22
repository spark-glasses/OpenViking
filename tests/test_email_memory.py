"""Behavioral coverage for bounded native email extraction and durable source access."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from openviking.message import Message, TextPart
from openviking.models.vlm.base import ToolCall, VLMResponse
from openviking.server.identity import RequestContext, Role
from openviking.service.session_service import SessionService
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.email_context import EmailContext, email_memory_policy
from openviking.session.memory.email_context_provider import EmailContextProvider
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.session import Session, SessionMeta
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig

INITIAL_REF = "email:00000000-0000-4000-8000-000000000001"
HISTORY_REF = "email:00000000-0000-4000-8000-000000000002"
PERSON = "viking://user/alice/memories/people/stable-person.md"
ARCHIVE = "viking://user/alice/sessions/email-batch/history/archive_001"


class MemoryFS:
    agfs = None

    def __init__(self):
        self.files = {}
        self.search_calls = []

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content

    async def rm(self, uri, **kwargs):
        self.files.pop(uri, None)

    async def ls(self, uri, **kwargs):
        return [
            {"name": key[len(uri) + 1 :], "uri": key, "isDir": False}
            for key in self.files
            if key.startswith(uri + "/") and "/" not in key[len(uri) + 1 :]
        ]

    async def tree(self, uri, **kwargs):
        return [{"uri": key, "isDir": False} for key in self.files if key.startswith(uri + "/")]

    async def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return SimpleNamespace(to_dict=lambda: {"memories": []})


@pytest.fixture
def setup(monkeypatch):
    config = SimpleNamespace(memory=MemoryConfig(), vlm=SimpleNamespace(is_available=lambda: False))
    for module in (
        "openviking_cli.utils.config",
        "openviking.session.memory.session_extract_context_provider",
        "openviking.session.memory.email_context_provider",
        "openviking.session.memory.extract_loop",
        "openviking.session.compressor_v2",
    ):
        monkeypatch.setattr(module + ".get_openviking_config", lambda: config)
    fs = MemoryFS()
    for module in ("openviking.session.compressor_v2", "openviking.session.memory.memory_updater"):
        monkeypatch.setattr(module + ".get_viking_fs", lambda: fs)
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    spec = {
        "batchId": "batch-1",
        "contactId": "current-contact",
        "anchorId": "stable-person",
        "contactName": "Ethan",
        "emails": ["ethan@example.test"],
        "personMemoryUri": PERSON,
        "sourceRefs": [INITIAL_REF],
    }
    messages = [
        Message(
            id="input-1",
            role="user",
            parts=[
                TextPart(
                    "New email from Ethan: continue the previous Atlas proposal. " + INITIAL_REF
                )
            ],
        )
    ]

    def provider(**changes):
        return EmailContextProvider(
            email_context={**spec, **changes},
            archive_uri=ARCHIVE,
            attempt_id="attempt-1",
            messages=messages,
            ctx=ctx,
            viking_fs=fs,
        )

    return SimpleNamespace(
        config=config, fs=fs, ctx=ctx, spec=spec, provider=provider, messages=messages
    )


def operation(uri=PERSON, kind="people", text="Ethan works on Atlas."):
    return ResolvedOperation(
        memory_type=kind,
        uris=[uri],
        memory_fields={
            "content": {"blocks": [{"search": "", "replace": text}]},
        },
    )


def operations(*items):
    return ResolvedOperations(upsert_operations=list(items), delete_file_contents=[], errors=[])


def test_email_config_round_trip_and_hard_limits(setup):
    spec = EmailContext.model_validate(setup.spec)
    spec.validate_owner(setup.ctx)
    assert spec.maxToolCalls == 20
    assert spec.maxSourceChars == 200000
    legacy = EmailContext.model_validate(
        {**setup.spec, "maxToolCalls": 12, "maxSourceChars": 120000}
    )
    assert legacy.maxToolCalls == 12
    assert legacy.maxSourceChars == 120000
    meta = SessionMeta(session_id="s", email_context=spec.model_dump())
    assert SessionMeta.from_dict(meta.to_dict()).email_context == spec.model_dump()
    with pytest.raises(ValueError):
        setup.provider(maxToolCalls=25)
    with pytest.raises(ValueError):
        setup.provider(maxSourceChars=200001)
    with pytest.raises(ValueError):
        setup.provider(personMemoryUri="viking://user/bob/memories/people/stable-person.md")
    with pytest.raises(ValueError):
        setup.provider(anchorId="../another")


def test_schemas_are_email_only_and_anchor_stays_stable(setup):
    assert "people" not in MemoryTypeRegistry().list_names()
    provider = setup.provider(contactId="replacement-contact")
    assert provider._get_registry().get("people").filename_template == "stable-person.md"
    assert set(provider.get_tools()) == {"read", "search", "searchEmails", "readEmail"}
    assert {s.memory_type for s in provider.get_memory_schemas(setup.ctx)} == {
        "people",
        "entities",
        "events",
        "questions",
    }


@pytest.mark.asyncio
async def test_search_scope_is_enforced_even_for_root(setup):
    provider = setup.provider()
    await provider.execute_tool(ToolCall("1", "search", {"query": "Atlas", "limit": 999}))
    _, kwargs = setup.fs.search_calls[0]
    assert kwargs["target_uri"] == ["viking://user/alice/memories"]
    assert kwargs["limit"] == 30  # native search adds 10 before filtering
    with pytest.raises(ValueError):
        await provider.execute_tool(
            ToolCall("2", "read", {"uri": "viking://user/bob/memories/profile.md"})
        )
    with pytest.raises(ValueError):
        await provider.execute_tool(
            ToolCall("3", "read", {"uri": "viking://user/alice/memories/../resources/a"})
        )


@pytest.mark.asyncio
async def test_prefetch_missing_anchor_allows_only_that_new_document(setup):
    provider = setup.provider()
    await provider.prefetch()
    provider.validate_operations(operations(operation()))
    with pytest.raises(ValueError):
        provider.validate_operations(
            operations(operation(uri=PERSON.replace("stable-person", "someone-else")))
        )
    with pytest.raises(ValueError):
        provider.validate_operations(
            operations(
                operation(
                    uri="viking://user/alice/memories/entities/project/atlas.md", kind="entities"
                )
            )
        )
    assert PERSON in provider._missing_uris
    repeated = await provider.execute_tool(ToolCall("read-again", "read", {"uri": PERSON}))
    assert "error" in repeated  # not a synthetic success which would cause endless refetch


@pytest.mark.asyncio
async def test_existing_matter_requires_read_then_can_update(setup):
    uri = "viking://user/alice/memories/entities/project/atlas.md"
    setup.fs.files[uri] = "# Atlas\nOld proposal."
    provider = setup.provider()
    with pytest.raises(ValueError):
        provider.validate_operations(operations(operation(uri=uri, kind="entities")))
    await provider.execute_tool(ToolCall("r", "read", {"uri": uri}))
    provider.validate_operations(operations(operation(uri=uri, kind="entities")))


@pytest.mark.asyncio
async def test_dynamic_email_evidence_ranges_are_persisted(setup, monkeypatch):
    provider = setup.provider()
    read = AsyncMock(
        return_value={
            "success": True,
            "sourceRef": HISTORY_REF,
            "contentVersion": "v2",
            "body": "Atlas proposal B",
            "offset": 24000,
            "nextOffset": 48000,
        }
    )
    monkeypatch.setattr(provider, "_email_request", read)
    await provider.execute_tool(
        ToolCall("r", "readEmail", {"sourceRef": HISTORY_REF, "offset": 24000, "limit": 24000})
    )
    evidence = json.loads(setup.fs.files[ARCHIVE + "/email_evidence.json"])
    assert evidence["sourceRefs"] == [INITIAL_REF, HISTORY_REF]
    assert evidence["calls"][0]["tool"] == "readEmail"
    assert evidence["calls"][0]["arguments"]["offset"] == 24000
    assert evidence["calls"][0]["result"]["contentVersion"] == "v2"
    assert (
        setup.fs.files[ARCHIVE + "/attempts/attempt-1/email_evidence.json"]
        == setup.fs.files[ARCHIVE + "/email_evidence.json"]
    )
    await provider.prefetch()
    item = operation(text="Ethan chose B " + HISTORY_REF)
    provider.validate_operations(operations(item))
    assert item.memory_fields["email_source_refs"] == [INITIAL_REF, HISTORY_REF]
    with pytest.raises(ValueError):
        provider.validate_operations(
            operations(operation(text="email:00000000-0000-4000-8000-000000000099"))
        )


@pytest.mark.asyncio
async def test_tool_count_and_source_size_fail_closed(setup, monkeypatch):
    provider = setup.provider(maxToolCalls=1)
    await provider.execute_tool(ToolCall("1", "search", {"query": "first"}))
    with pytest.raises(RuntimeError, match="tool-call budget"):
        await provider.execute_tool(ToolCall("2", "search", {"query": "second"}))
    provider = setup.provider(maxSourceChars=1000)
    setup.fs.files[PERSON] = "x" * 2000
    with pytest.raises(RuntimeError, match="source-context budget"):
        await provider.prefetch()
    assert PERSON not in provider._fully_read
    assert PERSON not in provider.read_file_contents


@pytest.mark.asyncio
async def test_default_tool_budget_accepts_twenty_calls_then_stops(setup):
    provider = setup.provider()
    for index in range(20):
        await provider.execute_tool(ToolCall(str(index), "search", {"query": str(index)}))
    with pytest.raises(RuntimeError, match="tool-call budget"):
        await provider.execute_tool(ToolCall("overflow", "search", {"query": "overflow"}))
    assert len(setup.fs.search_calls) == 20


@pytest.mark.asyncio
async def test_initial_and_dynamic_sources_share_total_budget(setup, monkeypatch):
    provider = setup.provider(maxSourceChars=1000)
    initial_chars = len(provider.get_conversation_text())
    first = {
        "success": True,
        "sourceRef": HISTORY_REF,
        "contentVersion": "v2",
        "body": "x" * 500,
    }
    bridge = AsyncMock(return_value=first)
    monkeypatch.setattr(provider, "_email_request", bridge)
    await provider.execute_tool(ToolCall("first", "readEmail", {"sourceRef": HISTORY_REF}))
    evidence = json.loads(setup.fs.files[ARCHIVE + "/email_evidence.json"])
    assert evidence["sourceChars"] == initial_chars + len(json.dumps(first, ensure_ascii=False))
    with pytest.raises(RuntimeError, match="source-context budget"):
        await provider.execute_tool(
            ToolCall("next-page", "readEmail", {"sourceRef": HISTORY_REF, "offset": 500})
        )
    assert len(provider.evidence) == 1


@pytest.mark.asyncio
async def test_artifact_reader_is_not_exposed_or_dispatched(setup, monkeypatch):
    provider = setup.provider()
    bridge = AsyncMock()
    monkeypatch.setattr(provider, "_email_request", bridge)
    with pytest.raises(ValueError, match="Unknown email extraction tool"):
        await provider.execute_tool(ToolCall("old", "readArtifact", {"sourceRef": HISTORY_REF}))
    bridge.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_uses_trusted_identity_and_passes_ranges(setup, monkeypatch):
    setup.config.memory.email_source_base_url = "http://bridge.test"
    setup.config.memory.email_source_api_key = "test-only-secret"
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "sourceRef": HISTORY_REF,
                "contentVersion": "v2",
                "body": "Proposal B",
                "offset": 100,
                "nextOffset": 200,
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "openviking.session.memory.email_context_provider.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await setup.provider()._email_request(
        "readEmail", {"sourceRef": HISTORY_REF, "offset": 100, "limit": 999999}
    )
    assert result["nextOffset"] == 200
    assert requests[0].url.path == "/internal/memory/emails/read"
    assert requests[0].headers["X-User-Id"] == "alice"
    assert requests[0].headers["X-Memory-Batch-Id"] == "batch-1"
    assert json.loads(requests[0].content)["limit"] == 24000


@pytest.mark.asyncio
async def test_session_creation_persists_context_and_forces_safe_policy(setup, monkeypatch):
    service = SessionService()
    session = Mock(meta=SessionMeta(session_id="s"))
    session.exists = AsyncMock(return_value=False)
    session.ensure_exists = AsyncMock()
    monkeypatch.setattr(service, "session", lambda *args: session)
    await service.create(
        setup.ctx, "s", memory_policy={"self": {"enabled": False}}, email_context=setup.spec
    )
    assert session.meta.email_context["anchorId"] == "stable-person"
    assert session.meta.memory_policy == email_memory_policy()
    session.ensure_exists.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_loop_reads_history_and_applies_memory(setup, monkeypatch):
    history = {
        "success": True,
        "sourceRef": HISTORY_REF,
        "contentVersion": "v2",
        "body": "Ethan chose proposal B for Atlas.",
    }
    bridge = AsyncMock(return_value=history)
    monkeypatch.setattr(EmailContextProvider, "_email_request", bridge)
    final = json.dumps(
        {
            "people": [
                {
                    "page_id": 100,
                    "content": {
                        "blocks": [
                            {
                                "search": "",
                                "replace": "# Ethan\nEthan chose proposal B for Atlas. "
                                + HISTORY_REF,
                            }
                        ]
                    },
                }
            ],
            "entities": [],
            "events": [],
            "questions": [],
            "delete_uris": [],
        }
    )
    model = SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(
            side_effect=[
                VLMResponse(tool_calls=[ToolCall("s", "search", {"query": "Atlas proposal"})]),
                VLMResponse(tool_calls=[ToolCall("r", "readEmail", {"sourceRef": HISTORY_REF})]),
                final,
            ]
        ),
    )
    setup.config.vlm.get_vlm_instance = lambda: model
    compressor = SessionCompressorV2(vikingdb=None)
    result = await compressor.extract_long_term_memories(
        messages=setup.messages,
        ctx=setup.ctx,
        strict_extract_errors=True,
        archive_uri=ARCHIVE,
        email_context=setup.spec,
        email_attempt_id="attempt-1",
    )
    assert len(result) == 1
    person = MemoryFileUtils.read(setup.fs.files[PERSON])
    assert "proposal B" in person.content
    assert HISTORY_REF in person.extra_fields["email_source_refs"]
    outcome = json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])
    assert outcome["outcome"] == "applied"
    assert outcome["writtenUris"] == [PERSON]
    assert outcome["errors"] == []
    assert setup.fs.search_calls
    assert bridge.await_count == 2
    assert bridge.call_args.args[0] == "searchEmails"


@pytest.mark.asyncio
async def test_native_no_change_result_is_distinct_from_failure(setup):
    model = SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(
            return_value='{"people":[],"questions":[],"events":[],"entities":[],"delete_uris":[]}'
        ),
    )
    setup.config.vlm.get_vlm_instance = lambda: model
    compressor = SessionCompressorV2(vikingdb=None)
    assert (
        await compressor.extract_long_term_memories(
            messages=setup.messages,
            ctx=setup.ctx,
            strict_extract_errors=True,
            archive_uri=ARCHIVE,
            email_context=setup.spec,
        )
        == []
    )
    assert json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])["outcome"] == "no_change"


@pytest.mark.asyncio
async def test_provider_failure_is_reported_in_result_and_raised(setup):
    model = SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(side_effect=RuntimeError("provider unavailable")),
    )
    setup.config.vlm.get_vlm_instance = lambda: model
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
            messages=setup.messages,
            ctx=setup.ctx,
            strict_extract_errors=True,
            archive_uri=ARCHIVE,
            email_context=setup.spec,
        )
    outcome = json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])
    assert outcome["outcome"] == "failed"
    assert outcome["writtenUris"] == []
    assert outcome["errors"]


@pytest.mark.asyncio
async def test_retry_uses_archive_spec_and_refuses_duplicate_active_task(setup, monkeypatch):
    setup.fs.files[ARCHIVE + "/email_context.json"] = json.dumps(setup.spec)
    setup.fs.files[ARCHIVE + "/.failed.json"] = "{}"
    setup.fs.files[ARCHIVE + "/messages.jsonl"] = setup.messages[0].to_jsonl()
    session = Session(viking_fs=setup.fs, ctx=setup.ctx, session_id="email-batch")
    tracker = SimpleNamespace(create_if_no_running=AsyncMock(return_value=None))
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    with pytest.raises(InvalidArgumentError):
        await session.retry_email_archive("archive_001")
    tracker.create_if_no_running.assert_awaited_once()
    with pytest.raises(InvalidArgumentError):
        await session.retry_email_archive("../elsewhere")


@pytest.mark.asyncio
async def test_retry_creates_no_new_archive_and_keeps_frozen_anchor(setup, monkeypatch):
    import asyncio

    setup.fs.files[ARCHIVE + "/email_context.json"] = json.dumps(setup.spec)
    setup.fs.files[ARCHIVE + "/.failed.json"] = "{}"
    setup.fs.files[ARCHIVE + "/messages.jsonl"] = setup.messages[0].to_jsonl()
    session = Session(viking_fs=setup.fs, ctx=setup.ctx, session_id="email-batch")
    session.meta.email_context = {**setup.spec, "contactId": "changed-live-contact"}
    session._run_memory_extraction = AsyncMock()
    tracker = SimpleNamespace(
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="retry-1"))
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    result = await session.retry_email_archive("archive_001")
    await asyncio.sleep(0)
    assert result["archive_uri"] == ARCHIVE
    assert result["task_id"] == "retry-1"
    assert (
        session._run_memory_extraction.call_args.kwargs["email_context"]["contactId"]
        == "current-contact"
    )


@pytest.mark.asyncio
async def test_questions_have_stable_ids_and_validated_evidence(setup):
    from openviking.session.memory.question_store import QuestionStore, validate_proposals

    provider = setup.provider()
    await provider.prefetch()
    entry = {
        "topicKey": "User Alias Junkuan",
        "text": "Do you also use Junkuan?",
        "sourceRefs": [INITIAL_REF],
    }
    item = operation(uri=provider.question_uri, kind="questions")
    item.memory_fields = {
        "subjectKind": "self",
        "subjectId": "self",
        "subjectMemoryUri": "",
        "entries": json.dumps([entry]),
    }
    provider.validate_operations(operations(item))
    store = QuestionStore(setup.fs, setup.ctx)
    saved = await store.discover(
        provider.question_uri,
        {"kind": "self", "id": "self"},
        json.loads(item.memory_fields["entries"]),
    )
    updated = await store.discover(
        provider.question_uri,
        {"kind": "self", "id": "self"},
        validate_proposals([{**entry, "text": "Is Junkuan another name you use?"}], {INITIAL_REF}),
    )
    assert saved[0]["questionId"] == updated[0]["questionId"]
    assert updated[0]["text"] == "Is Junkuan another name you use?"
    with pytest.raises(ValueError):
        validate_proposals([{**entry, "sourceRefs": [HISTORY_REF]}], {INITIAL_REF})
    with pytest.raises(ValueError):
        validate_proposals([{**entry, "state": "asked"}], {INITIAL_REF})


@pytest.mark.asyncio
async def test_native_question_result_matches_saved_document(setup, monkeypatch):
    entry = {
        "topicKey": "user_alias_junkuan",
        "text": "Do you also use Junkuan?",
        "sourceRefs": [INITIAL_REF],
    }
    final = json.dumps(
        {
            "people": [],
            "entities": [],
            "events": [],
            "questions": [
                {
                    "page_id": 100,
                    "subjectKind": "self",
                    "subjectId": "self",
                    "subjectMemoryUri": "",
                    "entries": json.dumps([entry]),
                }
            ],
            "delete_uris": [],
        }
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(return_value=final)
    )
    monkeypatch.setattr(
        EmailContextProvider,
        "_email_request",
        AsyncMock(return_value={"success": True, "results": []}),
    )
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=setup.messages,
        ctx=setup.ctx,
        strict_extract_errors=True,
        archive_uri=ARCHIVE,
        email_context=setup.spec,
    )
    outcome = json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])
    assert outcome["outcome"] == "applied"
    assert len(outcome["questionRefs"]) == 1
    saved = MemoryFileUtils.read(setup.fs.files["viking://user/alice/memories/self/questions.md"])
    assert outcome["questionRefs"] == [
        {
            "questionId": saved.extra_fields["questions"][0]["questionId"],
            "uri": "viking://user/alice/memories/self/questions.md",
        }
    ]
    assert saved.extra_fields["questions"][0]["state"] == "open"
    assert saved.extra_fields["questions"][0]["events"] == []


@pytest.mark.asyncio
async def test_changed_binding_prevents_all_writes(setup, monkeypatch):
    final = json.dumps(
        {
            "people": [{"page_id": 100, "content": "Ethan joined Atlas."}],
            "entities": [],
            "events": [],
            "questions": [],
            "delete_uris": [],
        }
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(return_value=final)
    )
    monkeypatch.setattr(
        EmailContextProvider,
        "_email_request",
        AsyncMock(side_effect=RuntimeError("batch binding changed")),
    )
    with pytest.raises(RuntimeError, match="binding changed"):
        await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
            messages=setup.messages,
            ctx=setup.ctx,
            strict_extract_errors=True,
            archive_uri=ARCHIVE,
            email_context=setup.spec,
        )
    assert PERSON not in setup.fs.files
    assert json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])["outcome"] == "failed"


@pytest.mark.asyncio
async def test_email_merge_failures_are_not_reported_as_success(setup, monkeypatch):
    setup.fs.files[PERSON] = "# Ethan\nOriginal understanding."
    invalid_patch = json.dumps(
        {
            "people": [
                {
                    "page_id": 1,
                    "content": {
                        "blocks": [
                            {"search": "a nonexistent unique fragment", "replace": "overwritten"}
                        ]
                    },
                }
            ],
            "entities": [],
            "events": [],
            "questions": [],
            "delete_uris": [],
        }
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(return_value=invalid_patch)
    )
    monkeypatch.setattr(
        EmailContextProvider,
        "_email_request",
        AsyncMock(return_value={"success": True, "results": []}),
    )
    with pytest.raises(RuntimeError, match="updates failed"):
        await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
            messages=setup.messages,
            ctx=setup.ctx,
            strict_extract_errors=True,
            archive_uri=ARCHIVE,
            email_context=setup.spec,
        )
    assert setup.fs.files[PERSON] == "# Ethan\nOriginal understanding."
    result = json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])
    assert result["outcome"] == "failed"
    assert result["errors"]


@pytest.mark.asyncio
async def test_redo_restores_frozen_provider_configuration(setup, monkeypatch):
    from openviking.service.task_tracker import TaskStatus
    from openviking.storage.transaction.lock_manager import LockManager

    setup.fs._uri_to_path = lambda uri, ctx=None: uri
    setup.fs.files[ARCHIVE + "/email_context.json"] = json.dumps(setup.spec)
    tracker = SimpleNamespace(
        get=AsyncMock(
            side_effect=[
                SimpleNamespace(status=TaskStatus.FAILED),
                SimpleNamespace(status=TaskStatus.COMPLETED),
            ]
        )
    )
    session = SimpleNamespace(load=AsyncMock(), _run_memory_extraction=AsyncMock())
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: setup.fs)
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr("openviking.session.Session", lambda **kwargs: session)
    manager = SimpleNamespace(
        _async_agfs=SimpleNamespace(cat=AsyncMock(return_value=setup.messages[0].to_jsonl())),
        _enqueue_semantic=AsyncMock(),
    )
    await LockManager._redo_session_memory(
        manager,
        {
            "archive_uri": ARCHIVE,
            "session_uri": ARCHIVE.rsplit("/history", 1)[0],
            "account_id": "account",
            "user_id": "alice",
            "role": "root",
            "task_id": "t1",
            "email_context": {**setup.spec, "contactId": "stale-mutable-context"},
        },
    )
    assert (
        session._run_memory_extraction.call_args.kwargs["email_context"]["contactId"]
        == "current-contact"
    )
    assert session._run_memory_extraction.call_args.kwargs["memory_policy"] == email_memory_policy()
    manager._enqueue_semantic.assert_not_awaited()


@pytest.mark.asyncio
async def test_redo_fails_closed_if_email_context_was_lost(setup, monkeypatch):
    from openviking.storage.transaction.lock_manager import LockManager

    setup.fs._uri_to_path = lambda uri, ctx=None: uri
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: setup.fs)
    manager = SimpleNamespace(
        _async_agfs=SimpleNamespace(cat=AsyncMock(return_value=setup.messages[0].to_jsonl())),
        _enqueue_semantic=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="frozen archive context"):
        await LockManager._redo_session_memory(
            manager,
            {
                "archive_uri": ARCHIVE,
                "session_uri": ARCHIVE.rsplit("/history", 1)[0],
                "account_id": "account",
                "user_id": "alice",
                "role": "root",
                "email_context": setup.spec,
            },
        )
    manager._enqueue_semantic.assert_not_awaited()


@pytest.mark.asyncio
async def test_unread_citation_is_repaired_by_reading_source_before_write(setup, monkeypatch):
    final = json.dumps(
        {
            "people": [{"page_id": 100, "content": "Ethan selected proposal B. " + HISTORY_REF}],
            "entities": [],
            "events": [],
            "questions": [],
            "delete_uris": [],
        }
    )
    bridge = AsyncMock(
        return_value={
            "success": True,
            "sourceRef": HISTORY_REF,
            "contentVersion": "v2",
            "body": "Ethan selected B.",
        }
    )
    monkeypatch.setattr(EmailContextProvider, "_email_request", bridge)
    completion = AsyncMock(
        side_effect=[
            final,
            VLMResponse(
                tool_calls=[ToolCall("read-missing", "readEmail", {"sourceRef": HISTORY_REF})]
            ),
            final,
        ]
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model", get_completion_async=completion
    )
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=setup.messages,
        ctx=setup.ctx,
        strict_extract_errors=True,
        archive_uri=ARCHIVE,
        email_context=setup.spec,
    )
    assert completion.await_count == 3
    assert bridge.await_args_list[0].args == ("readEmail", {"sourceRef": HISTORY_REF})
    assert (
        HISTORY_REF
        in MemoryFileUtils.read(setup.fs.files[PERSON]).extra_fields["email_source_refs"]
    )
    assert json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])["outcome"] == "applied"


@pytest.mark.asyncio
async def test_repeated_invalid_citation_exhausts_repairs_without_writing(setup):
    final = json.dumps(
        {
            "people": [{"page_id": 100, "content": "Unsupported assertion " + HISTORY_REF}],
            "entities": [],
            "events": [],
            "questions": [],
            "delete_uris": [],
        }
    )
    completion = AsyncMock(return_value=final)
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model", get_completion_async=completion
    )
    with pytest.raises(ValueError, match=HISTORY_REF):
        await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
            messages=setup.messages,
            ctx=setup.ctx,
            strict_extract_errors=True,
            archive_uri=ARCHIVE,
            email_context=setup.spec,
        )
    assert completion.await_count == 3
    assert PERSON not in setup.fs.files
    assert json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])["outcome"] == "failed"


@pytest.mark.asyncio
async def test_retry_no_change_reindexes_previous_writes_and_recovers_questions(setup, monkeypatch):
    from openviking.session.memory.memory_updater import MemoryUpdater

    provider = setup.provider()
    from openviking.session.memory.question_store import QuestionStore

    entries = await QuestionStore(setup.fs, setup.ctx).discover(
        provider.question_uri,
        {"kind": "self", "id": "self"},
        [
            {
                "topicKey": "user_alias_junkuan",
                "text": "Do you also use Junkuan?",
                "sourceRefs": [HISTORY_REF],
            }
        ],
        {"email_source_refs": [HISTORY_REF]},
    )
    setup.fs.files[ARCHIVE + "/email_result.json"] = json.dumps(
        {
            "batchId": "batch-1",
            "outcome": "failed",
            "writtenUris": [provider.question_uri],
            "editedUris": [],
            "errors": ["indexing failed"],
        }
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(
            return_value='{"people":[],"questions":[],"events":[],"entities":[],"delete_uris":[]}'
        ),
    )
    monkeypatch.setattr(
        EmailContextProvider,
        "_email_request",
        AsyncMock(return_value={"success": True, "results": []}),
    )
    indexed = []

    async def vectorize(self, result, *args, **kwargs):
        indexed.extend(result.written_uris + result.edited_uris)
        return len(result.written_uris + result.edited_uris)

    monkeypatch.setattr(MemoryUpdater, "_vectorize_memories", vectorize)
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=setup.messages,
        ctx=setup.ctx,
        strict_extract_errors=True,
        archive_uri=ARCHIVE,
        email_context=setup.spec,
        email_attempt_id="retry-2",
    )
    result = json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])
    assert result["outcome"] == "applied"
    assert result["reindexedUris"] == [provider.question_uri]
    assert indexed == [provider.question_uri]
    assert result["questionRefs"] == [
        {"questionId": entries[0]["questionId"], "uri": provider.question_uri}
    ]
    assert ARCHIVE + "/attempts/retry-2/memory_diff.json" in setup.fs.files


@pytest.mark.asyncio
async def test_phase_failure_overrides_applied_result_and_keeps_provenance(setup):
    setup.fs.files[ARCHIVE + "/.done"] = "{}"
    setup.fs.files[ARCHIVE + "/email_result.json"] = json.dumps(
        {
            "batchId": "batch-1",
            "outcome": "applied",
            "writtenUris": [PERSON],
            "editedUris": [],
            "sourceRefs": [INITIAL_REF],
            "questionRefs": [{"questionId": "pending", "uri": "question-page"}],
            "errors": [],
        }
    )
    session = Session(viking_fs=setup.fs, ctx=setup.ctx, session_id="email-batch")
    await session._record_email_phase_failure(
        ARCHIVE, "attempt-1", setup.spec, "embedding processing failed"
    )
    failed = json.loads(setup.fs.files[ARCHIVE + "/email_result.json"])
    assert failed["outcome"] == "failed"
    assert failed["partial"] is True
    assert failed["writtenUris"] == [PERSON]
    assert failed["sourceRefs"] == [INITIAL_REF]
    assert failed["questionRefs"] == []
    assert failed == json.loads(setup.fs.files[ARCHIVE + "/attempts/attempt-1/email_result.json"])


@pytest.mark.asyncio
async def test_email_failure_removes_done_marker(setup):
    setup.fs.files[ARCHIVE + "/.done"] = "{}"
    session = Session(viking_fs=setup.fs, ctx=setup.ctx, session_id="email-batch")
    await session._record_email_phase_failure(ARCHIVE, "attempt-1", setup.spec, "task store failed")
    assert ARCHIVE + "/.done" not in setup.fs.files


@pytest.mark.asyncio
async def test_orphan_retry_is_delayed_and_requires_no_existing_task(setup, monkeypatch):
    import asyncio
    from datetime import datetime, timedelta, timezone

    setup.fs.files[ARCHIVE + "/email_context.json"] = json.dumps(setup.spec)
    setup.fs.files[ARCHIVE + "/messages.jsonl"] = setup.messages[0].to_jsonl()
    session = Session(viking_fs=setup.fs, ctx=setup.ctx, session_id="email-batch")
    session.meta.commit_count = 1
    session.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
    session._run_memory_extraction = AsyncMock()
    tracker = SimpleNamespace(
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="recovered"))
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    with pytest.raises(InvalidArgumentError):
        await session.retry_email_archive("archive_001")
    tracker.create_if_no_running.assert_not_awaited()
    session.meta.last_commit_at = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
    result = await session.retry_email_archive("archive_001")
    await asyncio.sleep(0)
    assert result["task_id"] == "recovered"
    assert tracker.create_if_no_running.await_args.kwargs["require_no_existing"] is True
    assert session._run_memory_extraction.await_count == 1
    tracker.create_if_no_running.return_value = None
    with pytest.raises(InvalidArgumentError):
        await session.retry_email_archive("archive_001")


@pytest.mark.asyncio
async def test_failed_task_cannot_hide_behind_stale_done_marker(setup, monkeypatch):
    import asyncio

    setup.fs.files[ARCHIVE + "/email_context.json"] = json.dumps(setup.spec)
    setup.fs.files[ARCHIVE + "/messages.jsonl"] = setup.messages[0].to_jsonl()
    setup.fs.files[ARCHIVE + "/.done"] = "{}"
    setup.fs.files[ARCHIVE + "/.failed.json"] = "{}"
    session = Session(viking_fs=setup.fs, ctx=setup.ctx, session_id="email-batch")
    session._run_memory_extraction = AsyncMock()
    tracker = SimpleNamespace(
        list_tasks=AsyncMock(return_value=[SimpleNamespace(status="failed", task_id="old")]),
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="retry")),
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    result = await session.retry_email_archive("archive_001")
    await asyncio.sleep(0)
    assert result["status"] == "accepted"
    assert result["task_id"] == "retry"
    assert session._run_memory_extraction.await_count == 1


@pytest.mark.asyncio
async def test_native_answer_propagation_updates_existing_person_anchor(setup):
    setup.fs.files[PERSON] = "# Ethan\nEthan works at Old Company."
    final = json.dumps(
        {
            "people": [
                {
                    "page_id": 1,
                    "content": {
                        "blocks": [
                            {
                                "search": "Ethan works at Old Company.",
                                "replace": "Ethan works at New Company, confirmed by the user.",
                            }
                        ]
                    },
                }
            ],
            "profile": [],
            "preferences": [],
            "entities": [],
            "events": [],
            "delete_uris": [],
        }
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(return_value=final)
    )
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=setup.messages,
        ctx=setup.ctx,
        strict_extract_errors=True,
        archive_uri=ARCHIVE,
        question_context={
            "subject": {"kind": "person", "id": "stable-person", "memoryUri": PERSON}
        },
    )
    person = MemoryFileUtils.read(setup.fs.files[PERSON])
    assert "New Company" in person.content
    assert not any("/entities/" in uri for uri in setup.fs.files)


@pytest.mark.asyncio
async def test_native_matter_question_routes_to_server_resolved_page(setup, monkeypatch):
    from openviking.session.memory.question_store import QuestionStore

    matter = "viking://user/alice/memories/entities/project/atlas.md"
    setup.fs.files[matter] = "# Atlas\nAn existing project."
    proposal = {
        "topicKey": "atlas_owner",
        "text": "Is Atlas your startup?",
        "sourceRefs": [INITIAL_REF],
    }
    final = json.dumps(
        {
            "people": [],
            "entities": [],
            "events": [],
            "questions": [
                {
                    "page_id": 100,
                    "subjectKind": "matter",
                    "subjectId": "atlas",
                    "subjectMemoryUri": matter,
                    "entries": json.dumps([proposal]),
                }
            ],
            "delete_uris": [],
        }
    )
    setup.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(
            side_effect=[
                VLMResponse(tool_calls=[ToolCall("read-matter", "read", {"uri": matter})]),
                final,
            ]
        ),
    )
    monkeypatch.setattr(
        EmailContextProvider,
        "_email_request",
        AsyncMock(return_value={"success": True, "results": []}),
    )
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=setup.messages,
        ctx=setup.ctx,
        strict_extract_errors=True,
        archive_uri=ARCHIVE,
        email_context=setup.spec,
    )
    records = await QuestionStore(setup.fs, setup.ctx).list()
    assert len(records) == 1
    assert records[0]["subject"]["kind"] == "matter"
    assert records[0]["subject"]["memoryUri"] == matter
    assert "/matters/" in records[0]["questionUri"]
