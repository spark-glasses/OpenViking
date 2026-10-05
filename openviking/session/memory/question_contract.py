"""Typed extraction contract. Questions are operations, never arbitrary Markdown patches."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QuestionContextData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=3000)
    uncertainty: str = Field(min_length=1, max_length=2000)
    knownFacts: list[str] = Field(default_factory=list, max_length=20)
    candidates: list[str] = Field(default_factory=list, max_length=20)


class QuestionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sourceRef: str = Field(min_length=1, max_length=1000)
    sourceVersion: str | None = Field(default=None, max_length=200)
    quote: str = Field(min_length=1, max_length=4000)
    author: str | None = Field(default=None, max_length=200)
    occurredAt: str | None = None

    @field_validator("quote")
    @classmethod
    def meaningful_quote(cls, value):
        if not value.strip():
            raise ValueError("Evidence quote must contain source text")
        return value


class QuestionImportance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["soon", "later"] = "later"
    reason: str = Field(default="", max_length=1000)


class QuestionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["discover", "addEvidence", "resolve", "obsolete", "relocate"] = "discover"
    questionId: str | None = None
    expectedRevision: int | None = Field(default=None, ge=0)
    topicKey: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=1000)
    context: QuestionContextData
    sourceRefs: list[str] = Field(min_length=1, max_length=20)
    evidence: list[QuestionEvidence] = Field(min_length=1, max_length=20)
    importance: QuestionImportance = Field(default_factory=QuestionImportance)
    relatedSubjectUris: list[str] = Field(default_factory=list, max_length=20)
    ownershipUncertain: bool = False
    purpose: str | None = Field(default=None, max_length=160)
    scope: dict | None = None
    resolution: str | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def transition_contract(self):
        if self.action != "discover" and (not self.questionId or self.expectedRevision is None):
            raise ValueError(
                "Existing question operations require questionId and expectedRevision from readQuestion"
            )
        if self.action in ("resolve", "obsolete") and not self.resolution:
            raise ValueError("Resolution requires a supported answer or obsolescence reason")
        if any(e.sourceRef not in self.sourceRefs for e in self.evidence):
            raise ValueError("Evidence must belong to sourceRefs")
        return self


class QuestionOperations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subjectKind: Literal["self", "person", "project", "focus", "matter", "unassigned"]
    subjectId: str = ""
    subjectMemoryUri: str = ""
    entries: list[QuestionProposal] = Field(min_length=1, max_length=20)


QUESTION_INSTRUCTION = """
Questions are durable unresolved knowledge, not a notification command. Save only
supported facts in memory; put useful unresolved interpretations in structured
questions with their causal context and exact evidence. Do not manufacture a
question for every missing field. Ambiguities already clarified during the user's
current task need no durable question. Inspect related open questions: new evidence
may add support, resolve one, or demonstrate it is obsolete. A muted/snoozed question
can still be resolved by evidence. Never mistake an assistant restatement, a memory
summary or a repeated source for independent confirmation. Same-name identities and
speaker assignments still require the established user-confirmation workflow.
Question entries are typed objects, not JSON strings, and do not use page_id.
For discover include summary, knownFacts, uncertainty, candidate explanations and
an exact source quote. For existing questions use readQuestion's questionId and
revision. When ownership is uncertain use subjectKind=unassigned rather than guessing a person. Importance is soon or later with a reason; asking time/channel is decided
by Spark policy. Never set asked, user preferences, delivery times or user answers.
"""
