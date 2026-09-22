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
from openviking.session.memory.question_store import QuestionStore

router = APIRouter(prefix="/api/v1/questions", tags=["questions"])


class CandidatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=3, ge=1, le=20)
    relatedSubjectIds: list[str] = Field(default_factory=list, max_length=50)
    conversationId: str | None = None
    recentText: str = Field(default="", max_length=20000)


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


def store(ctx):
    service = get_service()
    return QuestionStore(service.viking_fs, ctx, service.vikingdb_manager)


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


@router.get("/pending-propagation")
async def pending_propagation(ctx: RequestContext = Depends(get_request_context)):
    questions = [
        q
        for q in await store(ctx).list()
        if q.get("propagation") and q["propagation"]["status"] in ("pending", "submitted")
    ]
    questions.sort(key=lambda q: (q["propagation"]["status"] != "pending", q["updatedAt"]))
    return Response(status="ok", result={"questions": questions})


@router.post("/record")
async def record(body: RecordRequest, ctx: RequestContext = Depends(get_request_context)):
    result = await store(ctx).record(body.model_dump(mode="json", exclude_none=True))
    return Response(status="ok", result=result)


@router.get("/{question_id}")
async def get_question(question_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result=await store(ctx).get(str(question_id)))


@router.post("/{question_id}/propagate")
async def propagate(question_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    question = await store(ctx).propagate(str(question_id), get_service().sessions)
    return Response(status="ok", result={"question": question})
