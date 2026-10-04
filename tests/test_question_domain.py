"""Common Question behavior independent of source-specific provider subclasses."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.memory_write_context import MemoryWriteContext
from openviking.session.memory.question_service import QuestionService
from openviking.session.memory.question_store import QuestionStore
from openviking_cli.exceptions import InvalidArgumentError
from tests.test_question_store import event
from tests.test_shared_questions import REF, proposal

pytest_plugins = ["tests.test_question_store", "tests.test_canonical_people_extraction"]


def domain(setup):
    fs, ctx, _ = setup
    write = MemoryWriteContext(fs, ctx)
    write.remember(REF, "I work on Spark", kind="userAnswer")
    return QuestionService(write)


@pytest.mark.asyncio
async def test_unknown_owner_uses_unassigned_without_creating_a_person(setup):
    service = domain(setup)
    item = (await service.submit({"subjectKind": "unassigned", "entries": [proposal()]}))[0]
    assert item["questionUri"].endswith("/unassigned/questions.md")
    assert not any("/people/" in path for path in setup[0].files)
    assert item["state"] == "open" and item["asking"] == "allowed"
    assert item["events"] == []


@pytest.mark.asyncio
async def test_explicit_relocation_preserves_id_and_history_but_requires_current_revision(setup):
    service = domain(setup)
    q = (await service.submit({"subjectKind": "unassigned", "entries": [proposal()]}))[0]
    q = (await service.record(event(q, "dismissed")))["question"]
    uri = "viking://user/alice/memories/people/ethan/memory.md"
    service.write.read_files[uri] = MemoryFile(uri=uri, memory_type="people", content="Ethan")
    service.write.read_files[q["questionUri"]] = await service.store._load_page(q["questionUri"])
    move = {
        "subjectKind": "person",
        "subjectMemoryUri": uri,
        "entries": [
            proposal(action="relocate", questionId=q["questionId"], expectedRevision=q["revision"])
        ],
    }
    moved = (await service.submit(move))[0]
    assert moved["questionId"] == q["questionId"]
    assert moved["asking"] == "muted" and moved["events"][:-1] == q["events"]
    assert moved["questionUri"].endswith("/people/ethan/questions.md")
    assert (await service.submit(move))[0]["revision"] == moved["revision"]
    assert len(await service.store.list()) == 1


@pytest.mark.asyncio
async def test_source_inference_cannot_move_unknown_question_to_a_person(setup):
    service = domain(setup)
    q = (await service.submit({"subjectKind": "unassigned", "entries": [proposal()]}))[0]
    uri = "viking://user/alice/memories/people/ethan/memory.md"
    service.write.read_files[uri] = MemoryFile(uri=uri, memory_type="people", content="Ethan")
    service.write.read_files[q["questionUri"]] = await service.store._load_page(q["questionUri"])
    service.write.sources[REF][0]["kind"] = "sourceEvidence"
    with pytest.raises(ValueError, match="user confirmation"):
        await service.submit(
            {
                "subjectKind": "person",
                "subjectMemoryUri": uri,
                "entries": [
                    proposal(
                        action="relocate",
                        questionId=q["questionId"],
                        expectedRevision=q["revision"],
                    )
                ],
            }
        )
    assert (await service.store.get(q["questionId"]))["subject"]["kind"] == "unassigned"


@pytest.mark.asyncio
async def test_new_semantic_provider_gets_questions_without_its_schema_opt_in(env, monkeypatch):
    from openviking.session.compressor_v2 import SessionCompressorV2
    from openviking.session.memory.session_extract_context_provider import (
        SessionExtractContextProvider,
    )

    original = SessionExtractContextProvider.get_memory_schemas
    monkeypatch.setattr(
        SessionExtractContextProvider,
        "get_memory_schemas",
        lambda self, ctx: [
            schema for schema in original(self, ctx) if schema.memory_type != "questions"
        ],
    )
    provider = env.provider("I work on Spark")
    env.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
        model="test",
        get_completion_async=AsyncMock(
            return_value=json.dumps(
                {"questions": [{"subjectKind": "unassigned", "entries": [proposal()]}]}
            )
        ),
    )
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=provider.messages, ctx=env.ctx, strict_extract_errors=True
    )
    questions = await QuestionStore(env.fs, env.ctx).list()
    assert len(questions) == 1 and questions[0]["subject"]["kind"] == "unassigned"


@pytest.mark.parametrize(
    "meta,parts",
    [
        ({"kind": "memoryQuestionCandidates"}, [{"type": "text", "text": "I work on Spark"}]),
        ({"source": "workCompletion"}, [{"type": "text", "text": "I work on Spark"}]),
        ({}, [{"type": "tool_result", "isError": True, "result": "I work on Spark"}]),
        ({}, [{"type": "tool_result", "name": "readQuestion", "result": "I work on Spark"}]),
    ],
)
def test_synthetic_context_and_failed_results_are_not_evidence(setup, meta, parts):
    write = MemoryWriteContext(setup[0], setup[1])
    write.add_messages(
        [
            SimpleNamespace(
                role="tool", sourceRef=REF, content=json.dumps({"metadata": meta, "parts": parts})
            )
        ],
        structured=True,
    )
    with pytest.raises(ValueError, match="not supplied/read"):
        write.verify({"sourceRef": REF, "quote": "I work on Spark"})


@pytest.mark.asyncio
async def test_direct_api_uses_same_service_and_preserves_source_evidence(setup, monkeypatch):
    from openviking.server.routers import questions as router

    fs, ctx, _ = setup
    monkeypatch.setattr(
        router, "get_service", lambda: SimpleNamespace(viking_fs=fs, vikingdb_manager=None)
    )
    ref = "conversation:chat/message:user-1"
    body = router.WriteRequest(
        context={
            "operationId": "operation",
            "text": "Spark role",
            "origin": {"conversationId": "chat", "turnId": "turn", "toolCallId": "call"},
            "messages": [
                {"id": "user-1", "role": "user", "sourceRef": ref, "content": "I work on Spark"}
            ],
        },
        operation={
            "subjectKind": "unassigned",
            "entries": [
                proposal(
                    sourceRefs=[ref], evidence=[{"sourceRef": ref, "quote": "I work on Spark"}]
                )
            ],
        },
    )
    response = await router.write_question(body, ctx)
    q = response.result["questions"][0]
    assert q["evidence"][0]["sourceRef"] == ref
    assert q["subject"]["kind"] == "unassigned"
    body.operation.entries[0].evidence[0].quote = "I am the CEO"
    with pytest.raises(InvalidArgumentError):
        await router.write_question(body, ctx)
    assert len(await QuestionStore(fs, ctx).list()) == 1


@pytest.mark.asyncio
async def test_business_conflict_retains_receipt_without_claiming_raw_email_was_read(setup):
    service = domain(setup)
    q = (
        await service.discover_business_event(
            {"kind": "unassigned", "id": "unassigned"},
            proposal(sourceRefs=["email:original"]),
            event_ref="observation:identity",
            event={"name": "Ethan", "uncertain": True},
        )
    )[0]
    assert q["sourceRefs"] == ["email:original", "observation:identity"]
    assert q["evidence"][0]["sourceRef"] == "observation:identity"
    assert q["state"] == "open"


@pytest.mark.asyncio
async def test_direct_api_resolves_existing_unassigned_question(setup, monkeypatch):
    from openviking.server.routers import questions as router

    service = domain(setup)
    q = (await service.submit({"subjectKind": "unassigned", "entries": [proposal()]}))[0]
    fs, ctx, _ = setup
    monkeypatch.setattr(
        router, "get_service", lambda: SimpleNamespace(viking_fs=fs, vikingdb_manager=None)
    )
    body = router.WriteRequest(
        context={
            "operationId": "answer",
            "text": "Confirm role",
            "origin": {"conversationId": "chat", "turnId": "turn", "toolCallId": "call"},
            "messages": [
                {
                    "id": "user-1",
                    "role": "user",
                    "sourceRef": "conversation:chat/message:user-1",
                    "content": "I work on Spark",
                }
            ],
        },
        operation={
            "subjectKind": "unassigned",
            "entries": [
                proposal(
                    action="resolve",
                    questionId=q["questionId"],
                    expectedRevision=q["revision"],
                    resolution="The user works on Spark",
                    sourceRefs=["conversation:chat/message:user-1"],
                    evidence=[
                        {
                            "sourceRef": "conversation:chat/message:user-1",
                            "quote": "I work on Spark",
                        }
                    ],
                )
            ],
        },
    )
    response = await router.write_question(body, ctx)
    actual = response.result["questions"][0]
    assert actual["state"] == "resolved"
    assert actual["resolutionRecord"]["kind"] == "userAnswer"
    assert actual["propagation"]["status"] == "pending"


@pytest.mark.asyncio
async def test_subject_id_cannot_silently_disagree_with_uri(setup):
    service = domain(setup)
    uri = "viking://user/alice/memories/people/ethan/memory.md"
    service.write.read_files[uri] = MemoryFile(uri=uri, memory_type="people", content="Ethan")
    with pytest.raises(ValueError, match="does not match"):
        await service.submit(
            {
                "subjectKind": "person",
                "subjectId": "bob",
                "subjectMemoryUri": uri,
                "entries": [proposal()],
            }
        )
    assert await service.store.list() == []


@pytest.mark.asyncio
async def test_failed_propagation_yields_to_other_questions(setup, monkeypatch):
    from openviking.server.routers import questions as router

    service = domain(setup)
    records = await service.submit(
        {"subjectKind": "unassigned", "entries": [proposal(), proposal(topicKey="another_role")]}
    )
    for q in records:
        await service.record(event(q, "resolved", evidenceText="I work on Spark"))
    monkeypatch.setattr(router, "store", lambda ctx: service.store)
    before = (await router.pending_propagation(setup[1])).result["questions"]
    broken_id = before[0]["questionId"]
    sessions = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError):
        await service.propagate(broken_id, sessions)
    after = (await router.pending_propagation(setup[1])).result["questions"]
    assert after[0]["questionId"] != broken_id
    assert after[-1]["propagation"]["lastAttemptAt"]
