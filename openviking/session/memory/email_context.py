"""Persisted, server-validated input for one email extraction batch."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from openviking.core.namespace import user_space_fragment

EMAIL_MEMORY_TYPES = frozenset({"people", "entities", "events", "focuses", "questions"})


class EmailContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batchId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    personId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    anchorId: Optional[str] = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    personName: str = Field(min_length=1, max_length=256)
    emails: list[str] = Field(min_length=1, max_length=32)
    personMemoryUri: str
    sourceRefs: list[str] = Field(default_factory=list, max_length=100)
    maxToolCalls: int = Field(default=20, ge=1, le=24)
    # Total initial conversation plus prefetched and dynamically read source context.
    maxSourceChars: int = Field(default=200000, ge=1000, le=200000)

    def validate_owner(self, ctx) -> None:
        anchor = self.anchorId or self.personId
        expected = f"viking://user/{user_space_fragment(ctx)}/memories/people/{anchor}/memory.md"
        if self.personMemoryUri != expected:
            raise ValueError("personMemoryUri must identify this user's stable person anchor")
        if any(len(address) > 320 or "@" not in address for address in self.emails):
            raise ValueError("emails must contain email addresses")
        from uuid import UUID

        for ref in self.sourceRefs:
            if not ref.startswith("email:"):
                raise ValueError("sourceRefs must be persistent email references")
            UUID(ref[6:])


def email_memory_policy() -> dict:
    return {
        "self": {"enabled": True},
        "peer": {"enabled": False},
        "memory_types": sorted(EMAIL_MEMORY_TYPES),
    }
