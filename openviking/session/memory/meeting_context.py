"""Frozen, trusted scope for extraction from versioned recording evidence."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openviking.session.memory.person_paths import person_anchor_from_uri
from openviking.core.namespace import user_space_fragment

MEETING_MEMORY_TYPES = frozenset({"people", "meetings", "entities", "events", "focuses", "questions"})


class MeetingPerson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    personId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    anchorId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=256)
    emails: list[str] = Field(default_factory=list, max_length=32)
    personMemoryUri: str
    calendarEventIds: list[str] = Field(default_factory=list, max_length=100)


class MediaWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    startMs: int = Field(ge=0)
    endMs: int = Field(gt=0)
    startedAt: str
    endedAt: str
    calendarEventIds: list[str] = Field(default_factory=list, max_length=100)
    originalStartMs: int | None = Field(default=None, ge=0)
    originalEndMs: int | None = Field(default=None, gt=0)
    referenceEndedAt: str | None = None

    @property
    def identity_start(self):
        return self.originalStartMs if self.originalStartMs is not None else self.startMs

    @property
    def identity_end(self):
        return self.originalEndMs if self.originalEndMs is not None else self.endMs

    @model_validator(mode="after")
    def valid_window(self):
        if self.endMs <= self.startMs:
            raise ValueError("Media window end must follow start")
        if (self.originalStartMs is None) != (self.originalEndMs is None):
            raise ValueError("Original meeting window requires both endpoints")
        if not self.identity_start <= self.startMs < self.endMs <= self.identity_end:
            raise ValueError("Chunk must remain inside its original meeting window")
        for value in (self.startedAt, self.endedAt, self.referenceEndedAt):
            if value is None:
                continue
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Media window timestamps require a timezone")
        if datetime.fromisoformat(self.endedAt.replace("Z", "+00:00")) <= datetime.fromisoformat(
            self.startedAt.replace("Z", "+00:00")
        ):
            raise ValueError("Media window clock end must follow start")
        return self


class MeetingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    meetingId: str
    inputHash: str = Field(min_length=1, max_length=256)
    chunkIndex: int | None = Field(default=None, ge=0, strict=True)
    sourceRef: str
    sourceVersion: str = Field(pattern=r"^sha256:[0-9a-fA-F]{64}$")
    people: list[MeetingPerson] = Field(default_factory=list, max_length=100)
    mediaWindows: list[MediaWindow] = Field(min_length=1, max_length=100)
    calendarEvents: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    referenceEndedAt: str
    captureTimingSource: str = Field(min_length=1, max_length=128)
    meetingMemoryUri: str
    maxToolCalls: int = Field(default=20, ge=1, le=24)
    maxSourceChars: int = Field(default=200000, ge=1000, le=200000)
    # Filled only by OV's confirmed question propagation, never by a model tool.
    confirmedAssignments: list[dict[str, Any]] = Field(default_factory=list, max_length=100)

    def validate_owner(self, ctx):
        UUID(self.meetingId)
        if self.sourceRef != f"transcript:{self.meetingId}":
            raise ValueError("Meeting source must identify this recording")
        root = f"viking://user/{user_space_fragment(ctx)}/memories/"
        if self.meetingMemoryUri != root + f"events/meetings/{self.meetingId}.md":
            raise ValueError("Meeting URI must identify this user's fixed meeting anchor")
        if len({p.anchorId for p in self.people}) != len(self.people):
            raise ValueError("Duplicate person anchors")
        for person in self.people:
            if person.personMemoryUri != root + f"people/{person.anchorId}/memory.md":
                raise ValueError("Person URI must identify this user's fixed person anchor")
        for assignment in self.confirmedAssignments:
            if (
                assignment.get("status") != "confirmed"
                or assignment.get("sourceRef") != self.sourceRef
                or assignment.get("sourceVersion") != self.sourceVersion
            ):
                raise ValueError("Confirmed assignments must retain their exact source version")
            uri = assignment.get("personMemoryUri")
            if uri is not None and (
                person_anchor_from_uri(uri, root) is None
            ):
                raise ValueError("Confirmed speaker target must be a same-user person")
            if not any(
                w.identity_start == assignment.get("startMs")
                and w.identity_end == assignment.get("endMs")
                for w in self.mediaWindows
            ):
                raise ValueError("Confirmed assignment must match one frozen meeting window")
        if len(str(self.calendarEvents)) > 60000:
            raise ValueError("Calendar evidence exceeds context limit")


def meeting_memory_policy():
    return {
        "self": {"enabled": True},
        "peer": {"enabled": False},
        "memory_types": sorted(MEETING_MEMORY_TYPES),
    }
