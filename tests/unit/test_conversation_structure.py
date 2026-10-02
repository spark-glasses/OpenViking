"""Conversation provenance contracts, independent of model prompt wording."""

from openviking.message import Message, TextPart
from openviking.message.conversation_structure import conversation_structure


def test_work_structure_survives_message_archive_round_trip():
    metadata = {
        "agentKind": "work", "runtimeId": "work", "turnId": "w1",
        "parentTurnId": "u1", "parentToolCallId": "delegate1",
        "messageId": "original1", "sourceRef": "conversation:c/message:original1",
        "originalRole": "user", "source": "delegatedInstruction",
    }
    original = Message(id="archived1", role="assistant", parts=[TextPart("task")],
                       metadata={**metadata, "providerMetadata": {"internal": "excluded"}})
    restored = Message.from_dict(original.to_dict())
    assert restored.role == "assistant"
    assert conversation_structure(restored) == metadata


def test_structure_keeps_interleaved_execution_ownership_and_excludes_arbitrary_metadata():
    messages = [
        Message(id="m1", role="assistant", parts=[], metadata={"agentKind": "work", "turnId": "w1", "parentToolCallId": "d1"}),
        Message(id="m2", role="assistant", parts=[], metadata={"agentKind": "interaction", "turnId": "u2"}),
        Message(id="m3", role="tool", parts=[], metadata={"agentKind": "work", "turnId": "w2", "parentToolCallId": "d2", "secret": "excluded"}),
    ]
    assert [conversation_structure(m) for m in messages] == [
        {"agentKind": "work", "turnId": "w1", "parentToolCallId": "d1"},
        {"agentKind": "interaction", "turnId": "u2"},
        {"agentKind": "work", "turnId": "w2", "parentToolCallId": "d2"},
    ]


def test_unstructured_sources_remain_valid():
    assert conversation_structure(Message(id="m", role="user", parts=[TextPart("hello")])) == {}
