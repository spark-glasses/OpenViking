"""Observable evidence, identity, ownership and recovery boundaries for meetings."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import NAMESPACE_URL, uuid5

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.email_context_provider import EmailContextProvider
from openviking.session.memory.meeting_context import MEETING_MEMORY_TYPES
from openviking.session.memory.meeting_context_provider import MeetingContextProvider
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig
from tests.test_email_memory import MemoryFS

MID = "00000000-0000-4000-8000-000000000099"
REF = f"transcript:{MID}"
VERSION = "sha256:" + "a" * 64
ROOT = "viking://user/alice/memories/"
PERSON = ROOT + "people/ethan.md"
MEETING = ROOT + f"events/meetings/{MID}.md"
ARCHIVE = "viking://user/alice/sessions/meeting-job/history/archive_001"
SPEAKER = MID + ":" + VERSION + ":SPEAKER_00"


@pytest.fixture
def setup(monkeypatch):
    fs = MemoryFS()
    config = SimpleNamespace(memory=MemoryConfig(), vlm=SimpleNamespace(is_available=lambda: False))
    for module in (
        "openviking_cli.utils.config",
        "openviking.session.memory.session_extract_context_provider",
        "openviking.session.memory.meeting_context_provider",
        "openviking.session.memory.extract_loop",
        "openviking.session.compressor_v2",
    ):
        monkeypatch.setattr(module + ".get_openviking_config", lambda: config)
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    spec = {
        "jobId": "job-1",
        "meetingId": MID,
        "inputHash": "hash-1",
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "people": [
            {
                "contactId": "contact-1",
                "anchorId": "ethan",
                "name": "Ethan",
                "emails": ["ethan@example.test"],
                "personMemoryUri": PERSON,
                "calendarEventIds": ["event-1"],
            }
        ],
        "mediaWindows": [
            {
                "startMs": 0,
                "endMs": 60000,
                "startedAt": "2026-09-22T12:00:00Z",
                "endedAt": "2026-09-22T12:01:00Z",
                "calendarEventIds": ["event-1"],
            },
            {
                "startMs": 70000,
                "endMs": 90000,
                "startedAt": "2026-09-22T14:00:00Z",
                "endedAt": "2026-09-22T14:00:20Z",
                "calendarEventIds": ["event-2"],
            },
        ],
        "calendarEvents": [],
        "referenceEndedAt": "2026-09-22T14:00:20Z",
        "captureTimingSource": "server-aligned",
        "meetingMemoryUri": MEETING,
    }
    evidence = {
        "kind": "meetingEvidence",
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "tokens": [
            {
                "text": "I am Ethan. I will deliver the prototype Friday.",
                "startMs": 100,
                "endMs": 1000,
                "speakerRef": SPEAKER,
                "speakerLabel": "SPEAKER_00",
            },
            {
                "text": "Next topic.",
                "startMs": 71000,
                "endMs": 72000,
                "speakerRef": SPEAKER,
                "speakerLabel": "SPEAKER_00",
            },
        ],
        "coverage": {"complete": True, "hasMore": False},
    }

    def provider(changes=None, evidence_changes=None):
        message = Message(
            id="seed",
            role="user",
            parts=[TextPart(json.dumps({**evidence, **(evidence_changes or {})}))],
        )
        result = MeetingContextProvider(
            meeting_context={**spec, **(changes or {})},
            archive_uri=ARCHIVE,
            attempt_id="attempt-1",
            messages=[message],
            ctx=ctx,
            viking_fs=fs,
        )
        result.validate_source = AsyncMock()
        return result

    return SimpleNamespace(
        fs=fs, config=config, ctx=ctx, spec=spec, evidence=evidence, provider=provider
    )


def operations(*ops):
    return ResolvedOperations(upsert_operations=list(ops), delete_file_contents=[], errors=[])


def op(uri, kind, text="Useful understanding."):
    return ResolvedOperation(
        memory_type=kind,
        uris=[uri],
        memory_fields={"content": {"blocks": [{"search": "", "replace": text}]}},
    )


def proposal(confidence=95, **changes):
    return {
        "speakerRef": SPEAKER,
        "contactId": "contact-1",
        "confidence": confidence,
        "evidence": [
            {
                "sourceRef": REF,
                "sourceVersion": VERSION,
                "startMs": 100,
                "endMs": 1000,
                "quote": "I am Ethan.",
            }
        ],
        **changes,
    }


async def read_target(p, uri):
    await p._execute("read", {"uri": uri}, count_call=False)


def test_spec_owner_and_tool_sets_are_separate(setup):
    p = setup.provider()
    assert "readTranscript" in p.get_tools()
    assert "proposeSpeakerAssignments" in p.get_tools()
    assert {s.memory_type for s in p.get_memory_schemas(setup.ctx)} == MEETING_MEMORY_TYPES
    assert set(EmailContextProvider.get_tools(None)) == {
        "read",
        "search",
        "searchEmails",
        "readEmail",
    }
    with pytest.raises(ValueError):
        setup.provider({"meetingMemoryUri": MEETING.replace("alice", "bob")})
    with pytest.raises(ValueError):
        setup.provider({"maxToolCalls": 25})
    with pytest.raises(ValueError):
        setup.provider(evidence_changes={"sourceVersion": "sha256:" + "b" * 64})


@pytest.mark.asyncio
async def test_all_confidences_require_user_confirmation_and_actual_read_evidence(setup):
    p = setup.provider()
    await read_target(p, PERSON)
    for confidence in (0, 90, 90.1, 99, 100):
        assignment = p.propose_assignments([proposal(confidence)])["assignments"][0]
        assert assignment["status"] == "unresolved"
        assert assignment["confidence"] == confidence
        with pytest.raises(ValueError, match="user-confirmed"):
            p.validate_operations(operations(op(PERSON, "people")))
        with pytest.raises(ValueError, match="scoped meeting question"):
            p.validate_operations(operations())
    bad = proposal()
    bad["evidence"][0]["quote"] = "Invented words"
    with pytest.raises(ValueError):
        p.propose_assignments([bad])
    bad = proposal()
    bad["evidence"][0]["sourceVersion"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError):
        p.propose_assignments([bad])
    with pytest.raises(ValueError):
        p.propose_assignments([proposal(speakerRef="UNKNOWN")])


@pytest.mark.asyncio
async def test_same_label_in_another_window_does_not_inherit_roster(setup):
    p = setup.provider()
    await read_target(p, PERSON)
    item = proposal()
    item["evidence"] = [
        {
            "sourceRef": REF,
            "sourceVersion": VERSION,
            "startMs": 71000,
            "endMs": 72000,
            "quote": "Next topic.",
        }
    ]
    assert p.propose_assignments([item])["assignments"][0]["status"] == "unresolved"


@pytest.mark.asyncio
async def test_person_writes_require_valid_mapping_and_existing_related_reads(setup):
    p = setup.provider()
    await read_target(p, PERSON)
    with pytest.raises(ValueError):
        p.validate_operations(operations(op(PERSON, "people")))
    p.propose_assignments([proposal(100)])
    with pytest.raises(ValueError, match="user-confirmed"):
        p.validate_operations(operations(op(PERSON, "people")))
    with pytest.raises(ValueError):
        p.validate_operations(operations(op(ROOT + "events/invented.md", "events")))
    with pytest.raises(ValueError):
        p.validate_operations(
            operations(op(PERSON, "people", "Source email:00000000-0000-4000-8000-000000000002"))
        )


@pytest.mark.asyncio
async def test_memory_changed_after_read_fails_before_apply(setup):
    p = setup.provider()
    await read_target(p, MEETING)
    setup.fs.files[MEETING] = "Concurrent writer created this"
    with pytest.raises(RuntimeError, match="changed after read"):
        await p.before_apply(operations(op(MEETING, "meetings")))


@pytest.mark.asyncio
async def test_auto_continuation_reads_current_pages_and_records_coverage(setup):
    p = setup.provider(evidence_changes={"nextCursor": "page2", "coverage": {"hasMore": True}})
    p._source_request = AsyncMock(
        return_value={
            **setup.evidence,
            "success": True,
            "coverage": {"hasMore": False, "complete": True},
        }
    )
    await p.prefetch()
    assert not p.coverage()["hasMore"]
    assert p._source_request.await_args.args[0] == "readTranscript"
    assert p._source_request.await_args.args[1]["cursor"] == "page2"
    stored = json.loads(setup.fs.files[ARCHIVE + "/meeting_evidence.json"])
    assert any(e["tool"] == "readTranscript" for e in stored["calls"])


@pytest.mark.asyncio
async def test_oversized_unread_suffix_is_durable_and_does_not_complete(setup):
    p = setup.provider(
        {"maxSourceChars": 10000}, {"nextCursor": "remaining", "coverage": {"hasMore": True}}
    )
    with pytest.raises(RuntimeError, match="remaining nextCursor"):
        await p.prefetch()
    assert (
        json.loads(setup.fs.files[ARCHIVE + "/meeting_evidence.json"])["coverage"]["nextCursor"]
        == "remaining"
    )
    with pytest.raises(RuntimeError, match="unread pages"):
        await p.before_apply(operations())


@pytest.mark.asyncio
async def test_speaker_question_belongs_to_matter_original_window_and_version(setup):
    p = setup.provider()
    await read_target(p, MEETING)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    uri = question_uri(setup.ctx, subject)
    await read_target(p, uri)
    q = ResolvedOperation(
        memory_type="questions",
        uris=[uri],
        memory_fields={
            "subjectKind": "matter",
            "subjectMemoryUri": MEETING,
            "entries": json.dumps(
                [
                    {
                        "topicKey": "speaker",
                        "text": "Was that you or Ethan?",
                        "sourceRefs": [REF],
                        "purpose": "speakerIdentity",
                        "scope": {"speakerRef": SPEAKER, "startMs": 100, "endMs": 1000},
                    }
                ]
            ),
        },
    )
    p.validate_operations(operations(q, op(MEETING, "meetings")))
    entry = json.loads(q.memory_fields["entries"])[0]
    assert entry["scope"]["sourceVersion"] == VERSION
    assert entry["scope"]["endMs"] == 60000
    assert entry["delivery"]["expiresAt"] == "2026-09-22T14:01:00+00:00"
    assert q.memory_fields["meeting_require_subject"] == MEETING
    assert entry["sourceRefs"] == [REF]


@pytest.mark.asyncio
async def test_question_cannot_attach_to_missing_meeting_without_creation(setup):
    p = setup.provider()
    await read_target(p, MEETING)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    uri = question_uri(setup.ctx, subject)
    await read_target(p, uri)
    q = ResolvedOperation(
        memory_type="questions",
        uris=[uri],
        memory_fields={"subjectKind": "matter", "subjectMemoryUri": MEETING, "entries": "[]"},
    )
    with pytest.raises(ValueError, match="Create useful meeting"):
        p.validate_operations(operations(q))


@pytest.mark.asyncio
async def test_user_confirmation_preserves_frozen_scope_and_rejects_cross_user(setup):
    fs, ctx = setup.fs, setup.ctx
    fs.files[ARCHIVE + "/meeting_context.json"] = json.dumps(setup.spec)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    store = QuestionStore(fs, ctx)
    scope = {
        "speakerRef": SPEAKER,
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "startMs": 0,
        "endMs": 60000,
        "meetingContextUri": ARCHIVE + "/meeting_context.json",
        "meetingMemoryUri": MEETING,
    }
    q = (
        await store.discover(
            question_uri(ctx, subject),
            subject,
            [
                {
                    "topicKey": "speaker",
                    "text": "Was that Ethan?",
                    "sourceRefs": [REF],
                    "purpose": "speakerIdentity",
                    "scope": scope,
                }
            ],
        )
    )[0]
    data = {
        "questionId": q["questionId"],
        "action": "resolved",
        "evidenceRole": "user",
        "evidenceText": "Yes, that was Ethan.",
        "conversationId": "chat",
        "messageId": "answer",
        "confirmedSpeakerAssignment": {
            "personMemoryUri": "viking://user/bob/memories/people/ethan.md"
        },
    }
    with pytest.raises(InvalidArgumentError):
        await store.record(data)
    data["confirmedSpeakerAssignment"] = {"contactId": "contact-1"}
    result = await store.record(data)
    mapped = result["question"]["confirmedSpeakerAssignment"]
    assert (
        mapped["status"] == "confirmed"
        and mapped["sourceVersion"] == VERSION
        and mapped["endMs"] == 60000
    )
    assert mapped["personMemoryUri"] == PERSON
    assert result["question"]["propagation"]["status"] == "pending"


@pytest.mark.asyncio
async def test_native_loop_writes_meeting_then_canonical_scoped_question(setup, monkeypatch):
    from openviking.models.vlm.base import ToolCall, VLMResponse
    from openviking.session.compressor_v2 import SessionCompressorV2

    for module in ("openviking.session.compressor_v2", "openviking.session.memory.memory_updater"):
        monkeypatch.setattr(module + ".get_viking_fs", lambda: setup.fs)
    monkeypatch.setattr(MeetingContextProvider, "validate_source", AsyncMock())
    entry = {
        "topicKey": "speaker",
        "text": "Was that Ethan speaking?",
        "sourceRefs": [REF],
        "purpose": "speakerIdentity",
        "scope": {"speakerRef": SPEAKER, "startMs": 100, "endMs": 1000},
    }
    final = json.dumps(
        {
            "meetings": [
                {
                    "page_id": 100,
                    "content": {
                        "blocks": [
                            {
                                "search": "",
                                "replace": "The prototype is due Friday. Attribution remains unresolved. "
                                + REF,
                            }
                        ]
                    },
                }
            ],
            "people": [],
            "entities": [],
            "events": [],
            "questions": [
                {
                    "page_id": 101,
                    "subjectKind": "matter",
                    "subjectId": "meeting",
                    "subjectMemoryUri": MEETING,
                    "entries": json.dumps([entry]),
                }
            ],
            "delete_uris": [],
        }
    )
    model = SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(
            side_effect=[
                VLMResponse(
                    tool_calls=[
                        ToolCall(
                            "candidate",
                            "proposeSpeakerAssignments",
                            {"assignments": [proposal(100)]},
                        )
                    ]
                ),
                VLMResponse(content=final),
            ]
        ),
    )
    setup.config.vlm.get_vlm_instance = lambda: model
    messages = [Message(id="seed", role="user", parts=[TextPart(json.dumps(setup.evidence))])]
    result = await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=messages,
        ctx=setup.ctx,
        archive_uri=ARCHIVE,
        meeting_context=setup.spec,
        meeting_attempt_id="native-1",
        strict_extract_errors=True,
    )
    assert len(result) == 2
    outcome = json.loads(setup.fs.files[ARCHIVE + "/meeting_result.json"])
    assert outcome["outcome"] == "applied" and not outcome["coverage"]["hasMore"]
    assert outcome["writtenUris"][0] == MEETING
    assert len(outcome["questionRefs"]) == 1
    question = await QuestionStore(setup.fs, setup.ctx).get(
        outcome["questionRefs"][0]["questionId"]
    )
    assert question["subject"]["memoryUri"] == MEETING
    assert question["state"] == "open" and question["delivery"]["mode"] == "timeBound"
    assert outcome["speakerAssignments"][0]["status"] == "unresolved"
    assert outcome["speakerAssignments"][0]["confidence"] == 100
    assert PERSON not in setup.fs.files
    assert ARCHIVE + "/attempts/native-1/memory_diff.json" in setup.fs.files


@pytest.mark.asyncio
async def test_speaker_answer_propagates_original_frozen_recording_and_mapping(setup):
    from openviking.session.session import SessionMeta

    fs, ctx = setup.fs, setup.ctx
    fs.files[ARCHIVE + "/meeting_context.json"] = json.dumps(setup.spec)
    seed = Message(id="seed", role="user", parts=[TextPart(json.dumps(setup.evidence))])
    fs.files[ARCHIVE + "/messages.jsonl"] = seed.to_jsonl()
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    scope = {
        "speakerRef": SPEAKER,
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "startMs": 0,
        "endMs": 60000,
        "meetingContextUri": ARCHIVE + "/meeting_context.json",
        "meetingMemoryUri": MEETING,
    }
    store = QuestionStore(fs, ctx)
    q = (
        await store.discover(
            question_uri(ctx, subject),
            subject,
            [
                {
                    "topicKey": "speaker",
                    "text": "Was that Ethan?",
                    "sourceRefs": [REF],
                    "purpose": "speakerIdentity",
                    "scope": scope,
                }
            ],
        )
    )[0]
    await store.record(
        {
            "questionId": q["questionId"],
            "action": "resolved",
            "evidenceRole": "user",
            "evidenceText": "Yes, Ethan.",
            "conversationId": "chat",
            "messageId": "answer",
            "confirmedSpeakerAssignment": {"contactId": "contact-1"},
        }
    )
    session = SimpleNamespace(
        meta=SessionMeta(session_id="answer"),
        uri="viking://user/alice/sessions/answer",
        messages=[],
        _save_meta=AsyncMock(),
        _write_to_agfs_async=AsyncMock(),
        commit_async=AsyncMock(
            return_value={"task_id": "answer-task", "archive_uri": "answer-archive"}
        ),
    )

    def add_message(role, parts):
        session.messages.append(Message(id=str(len(session.messages)), role=role, parts=parts))

    session.add_message = add_message
    sessions = SimpleNamespace(get=AsyncMock(return_value=session), get_commit_task=AsyncMock())
    result = await store.propagate(q["questionId"], sessions)
    assert result["propagation"]["status"] == "submitted"
    assert session.meta.meeting_context["sourceVersion"] == VERSION
    assert session.meta.meeting_context["confirmedAssignments"][0]["personMemoryUri"] == PERSON
    assert json.loads(session.messages[0].parts[0].text)["kind"] == "meetingEvidence"
    assert len(session.messages) == 2
    session.commit_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_source_bridge_uses_trusted_user_and_job_and_checks_version(setup, monkeypatch):
    import httpx

    setup.config.memory.email_source_base_url = "http://backend.test"
    setup.config.memory.email_source_api_key = "test-only-key"
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={**setup.evidence, "success": True})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "openviking.session.memory.meeting_context_provider.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    p = setup.provider()
    await p._source_request(
        "readTranscript",
        {
            "sourceRef": REF,
            "sourceVersion": VERSION,
            "startMs": 0,
            "endMs": 60000,
            "cursor": "page2",
            "limit": 999999,
        },
    )
    assert requests[0].url.path == "/internal/memory/meetings/read-transcript"
    assert requests[0].headers["X-Spark-User-Id"] == "alice"
    assert requests[0].headers["X-Spark-Meeting-Job-Id"] == "job-1"
    assert json.loads(requests[0].content)["limit"] == 24000
    with pytest.raises(ValueError):
        await p._source_request("searchTranscripts", {"userId": "bob"})
    with pytest.raises(RuntimeError, match="version"):
        await p._source_request(
            "readTranscript",
            {"sourceRef": REF, "sourceVersion": "sha256:" + "b" * 64, "startMs": 0, "endMs": 60000},
        )


@pytest.mark.asyncio
async def test_truncated_memory_cannot_authorize_writes(setup):
    p = setup.provider({"maxSourceChars": 3000})
    setup.fs.files[PERSON] = "x" * 10000
    with pytest.raises(RuntimeError, match="budget"):
        await read_target(p, PERSON)
    assert PERSON not in p._fully_read and PERSON not in p._snapshots
    assert PERSON not in p.read_file_contents


def test_offsets_are_compared_as_instants(setup):
    from openviking.session.memory.meeting_context import MediaWindow

    with pytest.raises(ValueError):
        MediaWindow(
            startMs=0,
            endMs=1,
            startedAt="2026-09-22T12:00:00-07:00",
            endedAt="2026-09-22T13:00:00Z",
            calendarEventIds=[],
        )


@pytest.mark.asyncio
async def test_readable_profile_cannot_be_retyped_as_person_or_entity(setup):
    p = setup.provider()
    profile = ROOT + "profile.md"
    setup.fs.files[profile] = "The user knows Ethan."
    await read_target(p, profile)
    with pytest.raises(ValueError):
        p.validate_operations(operations(op(profile, "entities")))
    with pytest.raises(ValueError):
        p.propose_assignments(
            [proposal(contactId=None, personMemoryUri=profile, identifiedName="Ethan")]
        )
    p.assignments = [{"personMemoryUri": profile, "status": "inferred"}]
    with pytest.raises(ValueError):
        p.validate_operations(operations(op(profile, "people")))


def test_missing_coverage_never_certifies_complete_recording(setup):
    with pytest.raises(ValueError, match="hasMore"):
        setup.provider(evidence_changes={"coverage": {}})


@pytest.mark.asyncio
async def test_read_snapshot_must_match_bytes_delivered_to_model(setup):
    p = setup.provider()
    setup.fs.files[PERSON] = "Old fact"
    original = setup.fs.read_file
    count = 0

    async def racing_read(uri, **kwargs):
        nonlocal count
        value = await original(uri, **kwargs)
        if uri == PERSON:
            count += 1
            if count == 1:
                setup.fs.files[uri] = "Changed fact"
        return value

    setup.fs.read_file = racing_read
    with pytest.raises(RuntimeError, match="changed during read"):
        await read_target(p, PERSON)
    assert PERSON not in p._snapshots and PERSON not in p.read_file_contents


@pytest.mark.asyncio
async def test_later_negative_answer_revokes_active_confirmed_mapping(setup):
    fs, ctx = setup.fs, setup.ctx
    fs.files[ARCHIVE + "/meeting_context.json"] = json.dumps(setup.spec)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    scope = {
        "speakerRef": SPEAKER,
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "startMs": 0,
        "endMs": 60000,
        "meetingContextUri": ARCHIVE + "/meeting_context.json",
        "meetingMemoryUri": MEETING,
    }
    store = QuestionStore(fs, ctx)
    q = (
        await store.discover(
            question_uri(ctx, subject),
            subject,
            [
                {
                    "topicKey": "speaker",
                    "text": "Was that Ethan?",
                    "sourceRefs": [REF],
                    "purpose": "speakerIdentity",
                    "scope": scope,
                }
            ],
        )
    )[0]
    data = {
        "questionId": q["questionId"],
        "action": "resolved",
        "evidenceRole": "user",
        "evidenceText": "Yes.",
        "conversationId": "chat",
        "messageId": "answer-1",
        "confirmedSpeakerAssignment": {"contactId": "contact-1"},
    }
    await store.record(data)
    data.pop("confirmedSpeakerAssignment")
    data.update(messageId="answer-2", evidenceText="Actually, that was not Ethan.")
    corrected = (await store.record(data))["question"]
    assert "confirmedSpeakerAssignment" not in corrected
    assert corrected["events"][-1]["revokedSpeakerAssignment"]["personMemoryUri"] == PERSON
    assert corrected["events"][0]["confirmedSpeakerAssignment"]["personMemoryUri"] == PERSON


@pytest.mark.asyncio
async def test_frozen_confirmation_cannot_restore_a_revoked_canonical_answer(setup):
    assignment = {
        "speakerRef": SPEAKER,
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "startMs": 0,
        "endMs": 60000,
        "personMemoryUri": PERSON,
        "contactId": "contact-1",
        "isSelf": False,
        "status": "confirmed",
        "questionId": "00000000-0000-4000-8000-000000000111",
        "answerSourceRef": "conversation:chat/message:old-answer",
    }
    p = setup.provider({"confirmedAssignments": [assignment]})
    with pytest.raises(RuntimeError, match="superseded"):
        await p.prefetch()


@pytest.mark.asyncio
async def test_current_confirmations_are_reloaded_from_canonical_question_records(setup):
    fs, ctx = setup.fs, setup.ctx
    fs.files[ARCHIVE + "/meeting_context.json"] = json.dumps(setup.spec)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    scope = {
        "speakerRef": SPEAKER,
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "startMs": 0,
        "endMs": 60000,
        "meetingContextUri": ARCHIVE + "/meeting_context.json",
        "meetingMemoryUri": MEETING,
    }
    store = QuestionStore(fs, ctx)
    q = (
        await store.discover(
            question_uri(ctx, subject),
            subject,
            [
                {
                    "topicKey": "speaker",
                    "text": "Was that Ethan?",
                    "sourceRefs": [REF],
                    "purpose": "speakerIdentity",
                    "scope": scope,
                }
            ],
        )
    )[0]
    recorded = await store.record(
        {
            "questionId": q["questionId"],
            "action": "resolved",
            "evidenceRole": "user",
            "evidenceText": "Yes, Ethan.",
            "conversationId": "chat",
            "messageId": "answer",
            "confirmedSpeakerAssignment": {"contactId": "contact-1"},
        }
    )
    mapping = recorded["question"]["confirmedSpeakerAssignment"]
    p = setup.provider({"confirmedAssignments": [mapping]})
    await p.prefetch()
    assert p._confirmed == [mapping]
    ordinary = setup.provider()
    await ordinary.prefetch()
    assert ordinary._confirmed == [mapping]
    assert ordinary.propose_assignments([proposal(100)])["assignments"][0]["status"] == "confirmed"
    # A replay can update this person's memory without creating another question.
    ordinary.validate_operations(operations(op(PERSON, "people")))
    page = await store._load_page(question_uri(ctx, subject))
    assert len(page.extra_fields["questions"]) == 1
    assert page.extra_fields["questions"][0]["state"] == "resolved"
    later = proposal()
    later["evidence"] = [
        {
            "sourceRef": REF,
            "sourceVersion": VERSION,
            "startMs": 71000,
            "endMs": 72000,
            "quote": "Next topic.",
        }
    ]
    assert ordinary.propose_assignments([later])["assignments"][0]["status"] == "unresolved"


@pytest.mark.asyncio
async def test_invalid_source_arguments_are_persisted_and_repairable(setup, monkeypatch):
    import httpx

    from openviking.models.vlm.base import ToolCall

    setup.config.memory.email_source_base_url = "http://backend.test"
    setup.config.memory.email_source_api_key = "test-only-key"
    requests = []

    def handler(request):
        args = json.loads(request.content)
        requests.append(args)
        if args.get("participantEmail") == "Ethan":
            return httpx.Response(400, json={"error": "participantEmail must be an email address"})
        return httpx.Response(200, json={"success": True, "results": []})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "openviking.session.memory.meeting_context_provider.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    p = setup.provider()
    result = await p.execute_tool(
        ToolCall(
            "bad",
            "searchEmails",
            {
                "participantEmail": "Ethan",
                "query": "",
                "threadId": None,
                "dateRange": {"after": "", "before": None},
            },
        )
    )
    assert result["recoverable"] and "participantEmail" in result["error"]
    assert requests[0] == {"participantEmail": "Ethan", "maxResults": 10}
    result = await p.execute_tool(
        ToolCall(
            "fixed",
            "searchEmails",
            {
                "participantEmail": "ethan@example.test",
                "dateRange": {"after": "2026-09-01T00:00:00Z"},
            },
        )
    )
    assert result["success"]
    assert requests[1]["dateRange"]["after"] == "2026-09-01"
    evidence = json.loads(setup.fs.files[ARCHIVE + "/meeting_evidence.json"])
    assert evidence["calls"][0]["result"]["recoverable"]
    assert evidence["toolCalls"] == 2


@pytest.mark.asyncio
async def test_source_version_conflict_remains_fatal(setup, monkeypatch):
    import httpx

    from openviking.models.vlm.base import ToolCall

    setup.config.memory.email_source_base_url = "http://backend.test"
    setup.config.memory.email_source_api_key = "test-only-key"
    original = httpx.AsyncClient
    monkeypatch.setattr(
        "openviking.session.memory.meeting_context_provider.httpx.AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(409, json={"error": "source changed"})
            ),
            **kwargs,
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await setup.provider().execute_tool(
            ToolCall(
                "r",
                "readTranscript",
                {"sourceRef": REF, "sourceVersion": VERSION, "startMs": 0, "endMs": 60000},
            )
        )


def chunk_spec(setup, index=0):
    window = {
        **setup.spec["mediaWindows"][0],
        "startMs": 100,
        "endMs": 1000,
        "startedAt": "2026-09-22T12:00:00.100Z",
        "endedAt": "2026-09-22T12:00:01Z",
        "originalStartMs": 0,
        "originalEndMs": 60000,
        "referenceEndedAt": "2026-09-22T12:01:00Z",
    }
    return {"chunkIndex": index, "mediaWindows": [window]}


def chunk_evidence(setup):
    return {
        "tokens": [setup.evidence["tokens"][0]],
        "coverage": {"hasMore": False, "complete": True},
    }


def test_chunk_scope_never_claims_whole_recording_completion(setup):
    p = setup.provider(chunk_spec(setup), chunk_evidence(setup))
    assert p.spec.chunkIndex == 0
    assert p.coverage()["scope"] == "chunk" and p.coverage()["chunkIndex"] == 0
    assert p.coverage()["hasMore"] is False
    assert p.partial is True
    assert p.coverage()["requestedWindows"] == [{"startMs": 100, "endMs": 1000}]
    with pytest.raises(ValueError):
        setup.provider({"chunkIndex": -1})
    with pytest.raises(ValueError):
        setup.provider({"chunkIndex": False})


@pytest.mark.asyncio
async def test_chunk_header_comes_only_from_frozen_context(setup, monkeypatch):
    import httpx

    setup.config.memory.email_source_base_url = "http://backend.test"
    setup.config.memory.email_source_api_key = "test-only-key"
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"success": True, "jobId": "job-1", "inputHash": "hash-1"})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "openviking.session.memory.meeting_context_provider.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    await setup.provider(chunk_spec(setup), chunk_evidence(setup))._source_request("validate", {})
    await setup.provider()._source_request("validate", {})
    assert requests[0].headers["X-Spark-Meeting-Chunk-Index"] == "0"
    assert "X-Spark-Meeting-Chunk-Index" not in requests[1].headers
    with pytest.raises(ValueError):
        await setup.provider()._source_request("searchTranscripts", {"chunkIndex": 3})


@pytest.mark.asyncio
async def test_chunk_question_uses_original_window_deadline_and_identity(setup):
    p = setup.provider(chunk_spec(setup, 2), chunk_evidence(setup))
    await p.prefetch()
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    uri = question_uri(setup.ctx, subject)
    entry = {
        "topicKey": "speaker",
        "text": "Was that Ethan?",
        "sourceRefs": [REF],
        "purpose": "speakerIdentity",
        "scope": {"speakerRef": SPEAKER, "startMs": 100, "endMs": 1000},
    }
    q = ResolvedOperation(
        memory_type="questions",
        uris=[uri],
        memory_fields={
            "subjectKind": "matter",
            "subjectMemoryUri": MEETING,
            "entries": json.dumps([entry]),
        },
    )
    meeting = op(MEETING, "meetings")
    p.validate_operations(operations(meeting, q))
    saved = json.loads(q.memory_fields["entries"])[0]
    assert saved["scope"]["startMs"] == 0 and saved["scope"]["endMs"] == 60000
    assert saved["scope"]["evidenceStartMs"] == 100 and saved["scope"]["evidenceEndMs"] == 1000
    assert saved["scope"]["chunkIndex"] == 2
    assert saved["delivery"]["expiresAt"] == "2026-09-22T14:01:00+00:00"
    assert meeting.memory_fields["meeting_chunk_index"] == 2
    assert meeting.memory_fields["meeting_analyzed_ranges"][0]["startMs"] == 100


@pytest.mark.asyncio
async def test_chunk_preserves_prior_analyzed_ranges(setup):
    from openviking.session.memory.dataclass import MemoryFile

    previous = {
        "sourceRef": REF,
        "sourceVersion": VERSION,
        "chunkIndex": 0,
        "startMs": 0,
        "endMs": 99,
    }
    setup.fs.files[MEETING] = MemoryFileUtils.write(
        MemoryFile(
            uri=MEETING,
            content="Earlier understanding.",
            extra_fields={"meeting_analyzed_ranges": [previous]},
        )
    )
    p = setup.provider(chunk_spec(setup, 1), chunk_evidence(setup))
    await p.prefetch()
    update = op(MEETING, "meetings")
    update.old_memory_file_content = p.read_file_contents[MEETING]
    p.validate_operations(operations(update))
    assert update.memory_fields["meeting_analyzed_ranges"][0] == previous
    assert update.memory_fields["meeting_analyzed_ranges"][1]["chunkIndex"] == 1


@pytest.mark.asyncio
async def test_native_chunk_result_retains_chunk_zero_without_whole_recording_claim(
    setup, monkeypatch
):
    from openviking.session.compressor_v2 import SessionCompressorV2

    for module in ("openviking.session.compressor_v2", "openviking.session.memory.memory_updater"):
        monkeypatch.setattr(module + ".get_viking_fs", lambda: setup.fs)
    monkeypatch.setattr(MeetingContextProvider, "validate_source", AsyncMock())
    model = SimpleNamespace(
        model="test-model",
        get_completion_async=AsyncMock(
            return_value=json.dumps(
                {
                    "meetings": [],
                    "people": [],
                    "entities": [],
                    "events": [],
                    "questions": [],
                    "delete_uris": [],
                }
            )
        ),
    )
    setup.config.vlm.get_vlm_instance = lambda: model
    message = Message(
        id="seed",
        role="user",
        parts=[TextPart(json.dumps({**setup.evidence, **chunk_evidence(setup)}))],
    )
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=[message],
        ctx=setup.ctx,
        archive_uri=ARCHIVE,
        meeting_context={**setup.spec, **chunk_spec(setup)},
        meeting_attempt_id="chunk-attempt",
        strict_extract_errors=True,
    )
    result = json.loads(setup.fs.files[ARCHIVE + "/meeting_result.json"])
    assert result["chunkIndex"] == 0 and result["outcome"] == "no_change"
    assert result["partial"] and result["coverage"]["scope"] == "chunk"
    assert result["coverage"]["hasMore"] is False


@pytest.mark.asyncio
async def test_same_speaker_question_is_deduplicated_across_chunks_without_deadline_refresh(setup):
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    uri = question_uri(setup.ctx, subject)
    store = QuestionStore(setup.fs, setup.ctx)

    async def discover(index, start, end):
        changes = chunk_spec(setup, index)
        changes["mediaWindows"][0].update(startMs=start, endMs=end)
        evidence = chunk_evidence(setup)
        evidence["tokens"] = [{**setup.evidence["tokens"][0], "startMs": start, "endMs": end}]
        p = setup.provider(changes, evidence)
        await p.prefetch()
        entry = {
            "topicKey": "speaker",
            "text": "Was that Ethan?",
            "sourceRefs": [REF],
            "purpose": "speakerIdentity",
            "scope": {"speakerRef": SPEAKER, "startMs": start, "endMs": end},
        }
        q = ResolvedOperation(
            memory_type="questions",
            uris=[uri],
            memory_fields={
                "subjectKind": "matter",
                "subjectMemoryUri": MEETING,
                "entries": json.dumps([entry]),
            },
        )
        p.validate_operations(operations(op(MEETING, "meetings"), q))
        return (await store.discover(uri, subject, json.loads(q.memory_fields["entries"])))[0]

    first = await discover(0, 100, 1000)
    later = await discover(1, 5000, 6000)
    assert first["questionId"] == later["questionId"]
    assert later["scope"]["chunkIndex"] == 0
    assert later["scope"]["evidenceStartMs"] == 100
    assert later["delivery"] == first["delivery"]
