"""Questions are evidence-backed knowledge; asking preferences are independent."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.models.vlm.base import ToolCall, VLMResponse
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.question_context import QuestionContext
from openviking.session.memory.question_contract import QuestionOperations
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking.session.memory.utils.json_parser import parse_json_with_stability
from openviking_cli.exceptions import InvalidArgumentError
from tests.test_question_store import event

pytest_plugins = ["tests.test_canonical_people_extraction", "tests.test_question_store"]


REF = "session-message:original-1"


def proposal(**changes):
    return {
        "topicKey": "spark_role",
        "text": "What is your role in Spark?",
        "context": {
            "summary": "Spark appears in the user's conversation",
            "uncertainty": "The user's role is unclear",
            "knownFacts": [],
            "candidates": [],
        },
        "sourceRefs": [REF],
        "evidence": [{"sourceRef": REF, "quote": "I work on Spark"}],
        "importance": {"level": "later", "reason": "Background understanding"},
        **changes,
    }


@pytest.mark.asyncio
async def test_muted_question_can_be_resolved_by_source_without_asking_again(setup):
    fs, ctx, store = setup
    subject = {"kind": "self", "id": "self"}
    uri = question_uri(ctx, subject)
    q = (await store.discover(uri, subject, [proposal()]))[0]
    q = (await store.record(event(q, "dismissed")))["question"]
    resolution = proposal(
        action="resolve",
        questionId=q["questionId"],
        expectedRevision=q["revision"],
        resolution="The user co-founded Spark",
        evidence=[{"sourceRef": REF, "quote": "I co-founded Spark"}],
    )
    q = (await store.discover(uri, subject, [resolution]))[0]
    assert q["state"] == "resolved" and q["asking"] == "muted"
    assert q["resolutionRecord"]["kind"] == "sourceEvidence"
    assert q["propagation"]["status"] == "pending"
    assert await store.candidates() == []
    # A lost response can replay the exact operation after its revision changed.
    again = (await QuestionStore(fs, ctx).discover(uri, subject, [resolution]))[0]
    assert again["revision"] == q["revision"]
    assert len(again["events"]) == 2


@pytest.mark.asyncio
async def test_stale_evidence_cannot_overwrite_a_user_answer(setup):
    _, ctx, store = setup
    subject = {"kind": "self", "id": "self"}
    uri = question_uri(ctx, subject)
    q = (await store.discover(uri, subject, [proposal()]))[0]
    await store.record(event(q, "resolved", evidenceText="I am an advisor, not a founder"))
    with pytest.raises(InvalidArgumentError):
        await store.discover(
            uri,
            subject,
            [
                proposal(
                    action="resolve",
                    questionId=q["questionId"],
                    expectedRevision=q["revision"],
                    resolution="Founder",
                )
            ],
        )
    assert (await store.get(q["questionId"]))["resolution"] == "I am an advisor, not a founder"


@pytest.mark.asyncio
async def test_identity_resolution_still_requires_explicit_confirmation(setup):
    _, ctx, store = setup
    subject = {"kind": "self", "id": "self"}
    uri = question_uri(ctx, subject)
    q = (await store.discover(uri, subject, [proposal(purpose="externalIdentity")]))[0]
    with pytest.raises(InvalidArgumentError, match="explicit user confirmation"):
        await store.discover(
            uri,
            subject,
            [
                proposal(
                    action="resolve",
                    questionId=q["questionId"],
                    expectedRevision=q["revision"],
                    resolution="Slack Bob is People Bob",
                )
            ],
        )


@pytest.mark.asyncio
async def test_native_chat_discovers_then_resolves_question_and_preserves_context(env):
    qstore = QuestionStore(env.fs, env.ctx)

    async def run(text, responses):
        provider = env.provider(text)
        env.config.vlm.get_vlm_instance = lambda: SimpleNamespace(
            model="test-model", get_completion_async=AsyncMock(side_effect=responses)
        )
        return await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
            messages=provider.messages, ctx=env.ctx, strict_extract_errors=True
        )

    def output(entry):
        return json.dumps({"questions": [{"subjectKind": "self", "entries": [entry]}]})

    await run("I work on Spark", [output(proposal())])
    q = (await qstore.list())[0]
    assert q["context"]["uncertainty"] == "The user's role is unclear"
    assert q["evidence"][0]["quote"] == "I work on Spark"
    assert q["state"] == "open"
    await run(
        "I co-founded Spark",
        [
            VLMResponse(
                tool_calls=[
                    ToolCall("read-question", "readQuestion", {"questionId": q["questionId"]})
                ]
            ),
            output(
                proposal(
                    action="resolve",
                    questionId=q["questionId"],
                    expectedRevision=q["revision"],
                    resolution="The user co-founded Spark",
                    evidence=[{"sourceRef": REF, "quote": "I co-founded Spark"}],
                )
            ),
        ],
    )
    resolved = await qstore.get(q["questionId"])
    assert resolved["state"] == "resolved"
    assert resolved["resolutionRecord"]["kind"] == "userAnswer"
    assert resolved["propagation"]["status"] == "pending"
    assert await qstore.candidates() == []


@pytest.mark.asyncio
async def test_shared_search_includes_muted_but_not_other_user_questions(env):
    context = QuestionContext(env.provider("I work on Spark"))
    subject = {"kind": "self", "id": "self"}
    q = (await context.store.discover(question_uri(env.ctx, subject), subject, [proposal()]))[0]
    await context.store.record(event(q, "dismissed"))
    assert (await context.execute("searchQuestions", {"query": "Spark"}))["questions"][0][
        "questionId"
    ] == q["questionId"]
    full = await context.execute("readQuestion", {"questionId": q["questionId"]})
    assert full["asking"] == "muted"
    assert len(full["events"]) == 1


@pytest.mark.asyncio
async def test_evidence_requires_actual_quote_and_rejects_fabricated_source(env):
    provider = env.provider("I work on Spark")
    context = QuestionContext(provider)
    op = ResolvedOperation(
        memory_type="questions",
        uris=[],
        memory_fields={"subjectKind": "self", "entries": [proposal()]},
    )
    context.route(op)
    context.validate(ResolvedOperations(upsert_operations=[op], delete_file_contents=[], errors=[]))
    op.memory_fields["entries"] = [proposal(evidence=[{"sourceRef": REF, "quote": "I am CEO"}])]
    with pytest.raises(ValueError, match="not supplied/read"):
        context.validate(
            ResolvedOperations(upsert_operations=[op], delete_file_contents=[], errors=[])
        )
    context.observe({"sourceRef": "email:invented", "body": "I am CEO"})
    assert "email:invented" not in context.sources


def test_invalid_question_does_not_disappear_in_tolerant_parser():
    from pydantic import Field, create_model

    model = create_model(
        "Operations", questions=(list[QuestionOperations], Field(default_factory=list))
    )
    parsed, error = parse_json_with_stability(
        json.dumps({"questions": [{"subjectKind": "self", "entries": [{"text": "Who?"}]}]}), model
    )
    assert parsed is None and error


@pytest.mark.asyncio
async def test_source_resolution_propagates_without_impersonating_user(setup):
    fs, ctx, store = setup
    subject = {"kind": "self", "id": "self"}
    uri = question_uri(ctx, subject)
    q = (await store.discover(uri, subject, [proposal()]))[0]
    await store.discover(
        uri,
        subject,
        [
            proposal(
                action="resolve",
                questionId=q["questionId"],
                expectedRevision=q["revision"],
                resolution="Advisor",
            )
        ],
    )
    session = SimpleNamespace(
        uri="viking://user/alice/sessions/answer",
        meta=SimpleNamespace(),
        messages=[],
        _save_meta=AsyncMock(),
        _write_to_agfs_async=AsyncMock(),
        add_message=lambda role, parts: captured.append((role, parts)),
        commit_async=AsyncMock(return_value={"task_id": "task", "archive_uri": "archive"}),
    )
    captured = []
    sessions = SimpleNamespace(get=AsyncMock(return_value=session))
    await store.propagate(q["questionId"], sessions)
    assert session.meta.question_context["resolutionKind"] == "sourceEvidence"
    payload = json.loads(captured[0][1][0].text.split("\n", 1)[1])
    assert payload["resolution"]["kind"] == "sourceEvidence"
    assert payload["context"]["uncertainty"] and payload["evidence"]
    assert payload["sourceRefs"] == [REF]
