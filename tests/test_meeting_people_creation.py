"""Creating memory for existing contact anchors without requiring email or prior memory."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import NAMESPACE_URL, uuid5

import pytest

from openviking.message import Message, TextPart
from openviking.models.vlm.base import VLMResponse
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.meeting_context_provider import MeetingContextProvider
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from tests.test_meeting_memory import (
    ARCHIVE,
    MEETING,
    PERSON,
    REF,
    ROOT,
    SPEAKER,
    VERSION,
    op,
    operations,
    proposal,
)
from tests.test_meeting_memory import (
    setup as meeting_setup,
)


@pytest.fixture
def setup(monkeypatch):
    result = meeting_setup.__wrapped__(monkeypatch)
    result.spec["people"][0]["emails"] = []
    return result


def person_operations():
    return json.dumps(
        {
            "people": [
                {
                    "page_id": 100,
                    "anchorId": "ethan",
                    "content": {
                        "blocks": [
                            {
                                "search": "",
                                "replace": f"Ethan will deliver the prototype Friday. {REF} {VERSION}",
                            }
                        ]
                    },
                }
            ],
            "meetings": [],
            "entities": [],
            "events": [],
            "questions": [],
            "delete_uris": [],
        }
    )


async def record_confirmation(setup):
    setup.fs.files[ARCHIVE + "/meeting_context.json"] = json.dumps(setup.spec)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    store = QuestionStore(setup.fs, setup.ctx)
    question = (
        await store.discover(
            question_uri(setup.ctx, subject),
            subject,
            [
                {
                    "topicKey": "speaker",
                    "text": "Was that Ethan?",
                    "sourceRefs": [REF],
                    "purpose": "speakerIdentity",
                    "scope": {
                        "speakerRef": SPEAKER,
                        "sourceRef": REF,
                        "sourceVersion": VERSION,
                        "startMs": 0,
                        "endMs": 60000,
                        "meetingContextUri": ARCHIVE + "/meeting_context.json",
                        "meetingMemoryUri": MEETING,
                    },
                }
            ],
        )
    )[0]
    result = await store.record(
        {
            "questionId": question["questionId"],
            "action": "resolved",
            "evidenceRole": "user",
            "evidenceText": "Yes, that was Ethan.",
            "conversationId": "chat-confirmation",
            "messageId": "answer-1",
            "confirmedSpeakerAssignment": {"contactId": "contact-1"},
        }
    )
    return result["question"]["confirmedSpeakerAssignment"]


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen_mapping", [False, True])
async def test_native_loop_creates_missing_authorized_person_without_email(
    setup, monkeypatch, frozen_mapping
):
    for module in ("openviking.session.compressor_v2", "openviking.session.memory.memory_updater"):
        monkeypatch.setattr(module + ".get_viking_fs", lambda: setup.fs)
    monkeypatch.setattr(MeetingContextProvider, "validate_source", AsyncMock())
    mapping = await record_confirmation(setup)
    if frozen_mapping:
        setup.spec["confirmedAssignments"] = [mapping]
    responses = []
    responses.append(VLMResponse(content=person_operations()))
    model = SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(side_effect=responses)
    )
    setup.config.vlm.get_vlm_instance = lambda: model
    assert PERSON not in setup.fs.files
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=[Message(id="seed", role="user", parts=[TextPart(json.dumps(setup.evidence))])],
        ctx=setup.ctx,
        archive_uri=ARCHIVE,
        meeting_context=setup.spec,
        meeting_attempt_id="create-person",
        strict_extract_errors=True,
    )
    saved = MemoryFileUtils.read(setup.fs.files[PERSON], uri=PERSON)
    assert saved.extra_fields["anchorId"] == "ethan"
    assert "prototype Friday" in saved.content
    assert REF in saved.content and VERSION in saved.content
    outcome = json.loads(setup.fs.files[ARCHIVE + "/meeting_result.json"])
    assert outcome["outcome"] == "applied" and not outcome["errors"]
    assert outcome["writtenUris"] == [PERSON]
    assert outcome["editedUris"] == []
    assert outcome["speakerAssignments"][0]["status"] == "confirmed"
    assert outcome["speakerAssignments"][0]["personMemoryUri"] == PERSON
    assert outcome["speakerAssignments"][0]["answerSourceRef"] == mapping["answerSourceRef"]
    replay = setup.provider()
    await replay.prefetch()
    assert replay.propose_assignments([proposal(100)])["assignments"][0]["status"] == "confirmed"
    replay.validate_operations(
        operations(op(PERSON, "people", "Additional supported understanding."))
    )
    assert replay.read_file_contents[PERSON].content == saved.content


@pytest.mark.asyncio
async def test_confirmed_absence_is_not_registered_as_existing_person_memory(setup):
    provider = setup.provider()
    for _ in range(2):
        value = await provider._execute("read", {"uri": PERSON})
        assert value["exists"] is False and value["canCreate"] is True
        assert value["anchorId"] == "ethan" and value["attributionRequired"] is True
        assert PERSON not in provider.read_file_contents
        assert PERSON not in provider._fully_read
    assert PERSON in provider._missing_uris
    assert provider._snapshots[PERSON] is None


@pytest.mark.asyncio
async def test_missing_candidate_cannot_be_written_without_validated_attribution(setup):
    provider = setup.provider()
    await provider._execute("read", {"uri": PERSON})
    with pytest.raises(ValueError, match="user-confirmed speaker mapping"):
        provider.validate_operations(operations(op(PERSON, "people")))
    assert PERSON not in setup.fs.files


@pytest.mark.asyncio
async def test_missing_unknown_anchor_cannot_be_created_from_name_or_known_person_assignment(setup):
    provider = setup.provider()
    unknown = ROOT + "people/not-a-contact.md"
    value = await provider._execute("read", {"uri": unknown})
    assert not value.get("canCreate")
    unknown_proposal = proposal(
        contactId="unknown-contact", personMemoryUri=unknown, identifiedName="Ethan"
    )
    assert (
        provider.propose_assignments([unknown_proposal])["assignments"][0]["status"] == "unresolved"
    )
    with pytest.raises(ValueError, match="user-confirmed speaker mapping"):
        provider.validate_operations(operations(op(unknown, "people")))
    await provider._execute("read", {"uri": PERSON})
    assert provider.propose_assignments([proposal(100)])["assignments"][0]["status"] == "unresolved"
    with pytest.raises(ValueError, match="user-confirmed speaker mapping"):
        provider.validate_operations(operations(op(unknown, "people")))
    assert unknown not in setup.fs.files and PERSON not in setup.fs.files


@pytest.mark.asyncio
async def test_missing_question_cache_does_not_trigger_native_refetch(setup):
    from openviking.session.memory.extract_loop import ExtractLoop

    provider = setup.provider()
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    uri = question_uri(setup.ctx, subject)
    loop = object.__new__(ExtractLoop)
    loop.context_provider = provider
    for _ in range(2):
        value = await provider._execute("read", {"uri": uri})
        assert value["exists"] is False
        assert uri not in provider.read_file_contents
        assert await loop._check_unread_existing_files(operations(op(uri, "questions"))) == {}
    assert provider._snapshots[uri] is None


@pytest.mark.asyncio
async def test_native_pre_read_cannot_hide_exhausted_tool_budget(setup):
    from openviking.session.memory.extract_loop import ExtractLoop

    provider = setup.provider({"maxToolCalls": 1})
    await provider._execute("read", {"uri": PERSON})
    loop = object.__new__(ExtractLoop)
    loop.context_provider = provider
    pending = operations(op(PERSON, "people"))
    # The native helper logs and catches read errors; final application must still fail.
    assert await loop._check_unread_existing_files(pending) == {}
    with pytest.raises(RuntimeError, match="tool-call budget"):
        await provider.before_apply(pending)
    assert PERSON not in setup.fs.files


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", [99, 100])
@pytest.mark.parametrize("existing", [False, True])
async def test_model_confirmed_flag_and_high_confidence_cannot_authorize_person_write(
    setup, confidence, existing
):
    if existing:
        setup.fs.files[PERSON] = "Earlier trusted person memory."
    provider = setup.provider()
    await provider.prefetch()
    before = setup.fs.files.get(PERSON)
    candidate = provider.propose_assignments(
        [proposal(confidence, status="confirmed", answerSourceRef="conversation:fake/message:fake")]
    )["assignments"][0]
    assert candidate["status"] == "unresolved"
    with pytest.raises(ValueError, match="user-confirmed"):
        provider.validate_operations(operations(op(PERSON, "people")))
    with pytest.raises(ValueError, match="scoped meeting question"):
        provider.validate_operations(operations(op(MEETING, "meetings")))
    assert setup.fs.files.get(PERSON) == before


@pytest.mark.asyncio
async def test_self_introduction_outside_calendar_is_only_a_candidate(setup):
    setup.fs.files[PERSON] = "Ethan is a founder."
    provider = setup.provider({"people": []})
    await provider._execute("read", {"uri": PERSON})
    candidate = provider.propose_assignments(
        [proposal(100, contactId=None, personMemoryUri=PERSON, identifiedName="Ethan")]
    )["assignments"][0]
    assert candidate["candidateEvidenceSupported"] is True
    assert candidate["status"] == "unresolved"
    with pytest.raises(ValueError, match="user-confirmed"):
        provider.validate_operations(operations(op(PERSON, "people")))


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_status", ["inferred", "confirmed"])
async def test_legacy_meeting_metadata_cannot_grant_person_write_permission(setup, legacy_status):
    from openviking.session.memory.dataclass import MemoryFile

    setup.fs.files[MEETING] = MemoryFileUtils.write(
        MemoryFile(
            uri=MEETING,
            content="Earlier meeting understanding.",
            extra_fields={
                "speaker_assignments": [
                    {
                        "speakerRef": SPEAKER,
                        "sourceRef": REF,
                        "sourceVersion": VERSION,
                        "startMs": 0,
                        "endMs": 60000,
                        "personMemoryUri": PERSON,
                        "status": legacy_status,
                    }
                ]
            },
        )
    )
    provider = setup.provider()
    await provider.prefetch()
    assert provider._confirmed == []
    with pytest.raises(ValueError, match="user-confirmed"):
        provider.validate_operations(operations(op(PERSON, "people")))
    update = op(MEETING, "meetings")
    provider.validate_operations(operations(update))
    assert all(a["status"] == "unresolved" for a in update.memory_fields["speaker_assignments"])


@pytest.mark.asyncio
async def test_confirmation_requires_canonical_answer_history_not_only_metadata(setup):

    mapping = await record_confirmation(setup)
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, MEETING).hex, "memoryUri": MEETING}
    uri = question_uri(setup.ctx, subject)
    page = MemoryFileUtils.read(setup.fs.files[uri], uri=uri)
    page.extra_fields["questions"][0]["answers"] = []
    setup.fs.files[uri] = MemoryFileUtils.write(page)
    provider = setup.provider({"confirmedAssignments": [mapping]})
    with pytest.raises(ValueError, match="user-confirmed"):
        provider.validate_operations(operations(op(PERSON, "people")))
    with pytest.raises(RuntimeError, match="canonical answer"):
        await provider.prefetch()
    ordinary = setup.provider()
    await ordinary.prefetch()
    assert ordinary._confirmed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entities", "events"])
async def test_related_matters_wait_for_all_current_speaker_confirmations(setup, kind):
    await record_confirmation(setup)
    uri = ROOT + kind + "/prototype.md"
    setup.fs.files[uri] = "The prototype project."
    provider = setup.provider()
    await provider.prefetch()
    await provider._execute("read", {"uri": uri})
    # The first window is confirmed, but the same label in a second window is not.
    with pytest.raises(ValueError, match="all current speakers"):
        provider.validate_operations(operations(op(uri, kind)))
    # Restricting analysis to the confirmed window now allows propagation.
    confirmed = setup.provider(
        {"mediaWindows": [setup.spec["mediaWindows"][0]]},
        {"tokens": [setup.evidence["tokens"][0]]},
    )
    await confirmed.prefetch()
    await confirmed._execute("read", {"uri": uri})
    confirmed.validate_operations(operations(op(uri, kind)))
