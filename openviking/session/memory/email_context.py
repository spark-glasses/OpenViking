"""Persisted, server-validated input for one email extraction batch."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from openviking.core.namespace import user_space_fragment

EMAIL_MEMORY_TYPES = frozenset(
    {"profile", "preferences", "people", "entities", "events", "focuses", "questions"}
)


class EmailPerson(BaseModel):
    """A person in the mail whom the application already knows."""

    model_config = ConfigDict(extra="forbid")

    personId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    anchorId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=256)
    emails: list[str] = Field(min_length=1, max_length=32)
    personMemoryUri: str


class EmailContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batchId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    # Mail from someone the application does not know is still evidence, so none is valid.
    people: list[EmailPerson] = Field(default_factory=list, max_length=20)
    sourceRefs: list[str] = Field(default_factory=list, max_length=100)
    maxToolCalls: int = Field(default=20, ge=1, le=24)
    # Total initial conversation plus prefetched and dynamically read source context.
    maxSourceChars: int = Field(default=200000, ge=1000, le=200000)

    def validate_owner(self, ctx) -> None:
        root = f"viking://user/{user_space_fragment(ctx)}/memories/"
        if len({person.anchorId for person in self.people}) != len(self.people):
            raise ValueError("Duplicate person anchors")
        for person in self.people:
            if person.personMemoryUri != root + f"people/{person.anchorId}/memory.md":
                raise ValueError("personMemoryUri must identify this user's stable person anchor")
            if any(len(address) > 320 or "@" not in address for address in person.emails):
                raise ValueError("emails must contain email addresses")
        for ref in self.sourceRefs:
            if not ref.startswith("email:"):
                raise ValueError("sourceRefs must be persistent email references")
            UUID(ref[6:])

    def person_uris(self) -> set[str]:
        return {person.personMemoryUri for person in self.people}


def email_memory_policy() -> dict:
    return {
        "self": {"enabled": True},
        "peer": {"enabled": False},
        "memory_types": sorted(EMAIL_MEMORY_TYPES),
    }
