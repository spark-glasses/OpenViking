"""Immutable, server supplied input for an explicit contextual memory update."""

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openviking.session.memory.question_store import memory_root

UPDATE_MEMORY_TYPES = frozenset(
    {"profile", "preferences", "people", "entities", "events", "questions"}
)
IDENTIFIER = r"^[A-Za-z0-9_-]+$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpdateOrigin(StrictModel):
    conversationId: str = Field(min_length=1, max_length=200)
    turnId: str = Field(min_length=1, max_length=200)
    toolCallId: str = Field(min_length=1, max_length=200)


class UpdateTarget(StrictModel):
    kind: Literal["person", "memory"]
    memoryUri: str = Field(min_length=1, max_length=1000)
    personId: str | None = Field(default=None, max_length=128, pattern=IDENTIFIER)
    anchorId: str | None = Field(default=None, max_length=128, pattern=IDENTIFIER)
    displayName: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def check_person(self):
        if self.kind == "person" and not (self.personId and self.anchorId):
            raise ValueError("Person targets require resolved personId and anchorId")
        return self


class UpdateMessage(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    role: Literal["user", "assistant", "tool", "summary"]
    content: str = Field(max_length=2000000)
    createdAt: str | None = Field(default=None, max_length=100)
    sourceRef: str | None = Field(default=None, max_length=1000)


class MemoryUpdateContext(StrictModel):
    operationId: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER)
    text: str = Field(min_length=1, max_length=50000)
    origin: UpdateOrigin
    targets: list[UpdateTarget] = Field(default_factory=list, max_length=20)
    messages: list[UpdateMessage] = Field(default_factory=list, max_length=10000)
    sourceKinds: list[Literal["email", "transcript"]] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def check_snapshot(self):
        if len(json.dumps(self.model_dump(), ensure_ascii=False).encode()) > 8000000:
            raise ValueError("Context exceeds the persisted operation snapshot limit")
        self.sourceKinds = sorted(set(self.sourceKinds))
        return self

    def validate_owner(self, ctx):
        root = memory_root(ctx)
        for target in self.targets:
            check_memory_uri(target.memoryUri, root)
            if target.kind == "person" and target.memoryUri != f"{root}people/{target.anchorId}.md":
                raise ValueError("Person target must use its resolved stable anchor")
        for message in self.messages:
            if message.sourceRef and not re.fullmatch(
                r"conversation:[^\s]+/message:[^\s]+", message.sourceRef
            ):
                raise ValueError("Context messages require conversation message source references")

    def input_hash(self):
        payload = json.dumps(
            self.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def check_memory_uri(uri, root):
    if not isinstance(uri, str) or not uri.startswith(root):
        raise ValueError("Memory URI is outside the authenticated user's memories")
    relative = uri[len(root) :]
    if (
        not relative
        or any(p in ("", ".", "..") for p in relative.split("/"))
        or any(c in uri for c in ("%", "?", "#", "\\"))
    ):
        raise ValueError("Invalid memory URI")
    if not uri.endswith(".md"):
        raise ValueError("Memory target must be a Markdown document")
