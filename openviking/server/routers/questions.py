"""Subject-owned questions and answer lifecycle, persisted entirely in OV."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.memory_update_context import MemoryUpdateContext, check_memory_uri
from openviking.session.memory.memory_write_context import MemoryWriteContext
from openviking.session.memory.question_contract import QuestionOperations
from openviking.session.memory.question_service import QuestionService
from openviking.session.memory.question_store import (
    QuestionStore,
    memory_root,
    topic_key,
)

router = APIRouter(prefix="/api/v1/questions", tags=["questions"])


class CandidatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=3, ge=1, le=20)
    relatedSubjectIds: list[str] = Field(default_factory=list, max_length=50)
    conversationId: str | None = None
    recentText: str = Field(default="", max_length=20000)


class DueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=3, ge=1, le=20)


class DeliveryReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deliveryId: UUID
    channel: Literal["glasses", "phone"]
    receivedAt: datetime
    messageId: str = Field(min_length=1, max_length=200)


class RecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questionId: UUID
    action: Literal["asked", "resolved", "deferred", "dismissed", "partial"]
    evidenceText: str = Field(min_length=1, max_length=50000)
    conversationId: str = Field(min_length=1, max_length=200)
    messageId: str = Field(min_length=1, max_length=200)
    evidenceRole: Literal["user", "assistant"]
    evidenceAt: datetime | None = None
    turnId: str | None = None
    notBefore: datetime | None = None
    resolution: str | None = Field(default=None, max_length=5000)
    deliveryReceipt: DeliveryReceipt | None = None
    confirmedSpeakerAssignment: dict | None = None


def store(ctx):
    service = get_service()
    return QuestionStore(service.viking_fs, ctx, service.vikingdb_manager)


def domain(ctx):
    backend = store(ctx)
    return QuestionService(MemoryWriteContext(backend.fs, ctx), backend.db)


@router.get("")
async def list_questions(ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result={"questions": await store(ctx).list()})


@router.post("/candidates")
async def candidates(body: CandidatesRequest, ctx: RequestContext = Depends(get_request_context)):
    questions = await store(ctx).candidates(
        limit=body.limit,
        related_subject_ids=body.relatedSubjectIds,
        conversation_id=body.conversationId,
        recent_text=body.recentText,
    )
    return Response(status="ok", result={"questions": questions})


@router.post("/due")
async def due(body: DueRequest, ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result={"questions": await store(ctx).due(limit=body.limit)})


@router.get("/pending-propagation")
async def pending_propagation(ctx: RequestContext = Depends(get_request_context)):
    questions = [
        q
        for q in await store(ctx).list()
        if q.get("propagation")
        and q["propagation"]["status"] in ("pending", "submitted", "unknown", "failed")
    ]
    # Retry the least recently attempted question first. A broken source or
    # unavailable session must not starve other answers owned by this user.
    questions.sort(key=lambda q: (q["propagation"].get("lastAttemptAt", ""), q["createdAt"]))
    return Response(status="ok", result={"questions": questions})


@router.post("/record")
async def record(body: RecordRequest, ctx: RequestContext = Depends(get_request_context)):
    result = await domain(ctx).record(body.model_dump(mode="json", exclude_none=True))
    return Response(status="ok", result=result)


class EmailIdentityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clusterId: UUID
    addresses: list[str] = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=800)
    sourceRefs: list[str] = Field(min_length=1, max_length=20)


@router.post("/email-identity")
async def discover_email_identity(
    body: EmailIdentityRequest, ctx: RequestContext = Depends(get_request_context)
):
    # Authenticated Spark ingestion owns source access/validation. This endpoint
    # only writes an uncertain unassigned question, never a People identity.
    from openviking_cli.exceptions import InvalidArgumentError

    try:
        for ref in body.sourceRefs:
            if not ref.startswith("email:") or str(UUID(ref[6:])) != ref[6:]:
                raise ValueError("invalid email reference")
    except ValueError as error:
        raise InvalidArgumentError(
            "Email identity evidence must use persistent email references"
        ) from error
    subject = {"kind": "unassigned", "id": "unassigned"}
    topic = "email-identity:" + str(body.clusterId)
    records = await domain(ctx).discover_business_event(
        subject,
        {
            "topicKey": topic,
            "text": f"You have exchanged emails with {body.description}. Who is this person?",
            "sourceRefs": body.sourceRefs,
            "context": {
                "summary": body.description,
                "uncertainty": "Which person owns these email addresses?",
                "knownFacts": body.addresses,
                "candidates": [],
            },
            "importance": {"level": "later", "reason": "Confirm an unresolved email identity"},
            "purpose": "emailIdentity",
            "scope": {
                "kind": "emailIdentity",
                "clusterId": str(body.clusterId),
                "addresses": body.addresses,
            },
            "ownershipUncertain": True,
        },
        event_ref="email-identity-observation:" + str(body.clusterId),
        event=body.model_dump(mode="json"),
    )
    return Response(
        status="ok",
        result={"question": next(q for q in records if q["topicKey"] == topic_key(topic))},
    )


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="", max_length=2000)
    subjectIds: list[str] = Field(default_factory=list, max_length=30)
    includeResolved: bool = False
    limit: int = Field(default=10, ge=1, le=20)


@router.post("/search")
async def search_questions(body: SearchRequest, ctx: RequestContext = Depends(get_request_context)):
    service = get_service()
    domain = QuestionService(MemoryWriteContext(service.viking_fs, ctx))
    return Response(
        status="ok",
        result={
            "questions": await domain.search(
                body.query, body.subjectIds, body.includeResolved, body.limit
            )
        },
    )


class WriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: MemoryUpdateContext
    operation: QuestionOperations


@router.post("/write")
async def write_question(body: WriteRequest, ctx: RequestContext = Depends(get_request_context)):
    from openviking_cli.exceptions import InvalidArgumentError

    try:
        return await apply_question_write(body, ctx)
    except ValueError as error:
        raise InvalidArgumentError(str(error)) from error


async def apply_question_write(body: WriteRequest, ctx: RequestContext):
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

    service = get_service()
    body.context.validate_owner(ctx)
    write = MemoryWriteContext(service.viking_fs, ctx, origin=body.context.origin.model_dump())
    # This snapshot is the caller runtime's actual visible context, not the
    # native provider's frozen-but-unread long history. Assistant prose is not evidence.
    write.add_messages(
        body.context.messages,
        structured=True,
        source_only=getattr(body.context.origin, "kind", None) == "automation",
    )
    domain = QuestionService(write, service.vikingdb_manager)
    fields = body.operation.model_dump(exclude_none=True)
    uri = fields.get("subjectMemoryUri")
    identifier = fields.get("subjectId")
    if not uri and identifier and fields["subjectKind"] in ("person", "project"):
        directory = "people" if fields["subjectKind"] == "person" else "projects"
        uri = memory_root(ctx) + f"{directory}/{identifier}/memory.md"
        fields["subjectMemoryUri"] = uri
    uris = {uri} if uri else set()
    for entry in fields["entries"]:
        uris.update(entry.get("relatedSubjectUris", []))
        if entry.get("questionId"):
            item = await domain.store.get(entry["questionId"])
            uris.add(item["questionUri"])
    for uri in uris:
        check_memory_uri(uri, memory_root(ctx))
        content = await service.viking_fs.read_file(uri, ctx=ctx)
        write.read_files[uri] = MemoryFileUtils.read(content, uri=uri)
    result = await domain.submit(fields)
    topics = {p["topicKey"] for p in fields["entries"]}
    topics = {topic_key(t) for t in topics}
    return Response(
        status="ok",
        result={
            "questions": [
                q
                for q in result
                if q["topicKey"] in topics
                or q["questionId"] in {p.get("questionId") for p in fields["entries"]}
            ]
        },
    )


@router.get("/{question_id}")
async def get_question(question_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result=await store(ctx).get(str(question_id)))


@router.post("/{question_id}/propagate")
async def propagate(question_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    question = await domain(ctx).propagate(str(question_id), get_service().sessions)
    return Response(status="ok", result={"question": question})
