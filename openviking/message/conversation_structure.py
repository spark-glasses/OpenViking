"""Allowlisted provenance for a conversation containing multiple agent streams."""

from typing import Any, Dict


def conversation_structure(message: Any) -> Dict[str, str]:
    """Expose execution links without copying arbitrary runtime/provider metadata.

    turnId identifies a turn within runtimeId; parentToolCallId links a Work
    instruction to Interaction's delegation. These are provenance, not facts
    about the user or additional independent evidence.
    """
    metadata = getattr(message, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in (
            "sourceRef", "messageId", "agentKind", "runtimeId", "agentRunId",
            "turnId", "parentTurnId", "parentToolCallId", "originalRole", "source",
        )
        if isinstance(metadata.get(key), str) and metadata[key]
    }
