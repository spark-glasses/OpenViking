"""Profile writes are deterministic; only native extraction may write learned text."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.profile_store import ProfileStore

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])


class Identity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preferredName: str = Field(max_length=255)
    names: list[str] = Field(max_length=50)
    additionalEmails: list[str] = Field(max_length=50)


class IdentityEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operationId: UUID
    expectedRevision: int = Field(ge=0)
    identity: Identity
    initialize: bool = False


class BlockEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operationId: UUID
    expectedRevision: int = Field(ge=0)
    blockId: UUID | None = None
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=30000)
    reason: str | None = Field(default=None, max_length=3000)
    action: Literal["edit", "delete", "unlock"] = "edit"


async def store(ctx):
    s = get_service()
    await s.initialize_user_directories(ctx)
    return ProfileStore(s.viking_fs, ctx, s.vikingdb_manager)


@router.get("")
async def get(ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result=await (await store(ctx)).get())


@router.post("/identity")
async def identity(body: IdentityEdit, ctx: RequestContext = Depends(get_request_context)):
    return Response(
        status="ok",
        result=await (await store(ctx)).identity(
            str(body.operationId),
            body.expectedRevision,
            body.identity.model_dump(),
            body.initialize,
        ),
    )


@router.post("/blocks")
async def edit(body: BlockEdit, ctx: RequestContext = Depends(get_request_context)):
    return Response(
        status="ok",
        result=await (await store(ctx)).edit(
            str(body.operationId),
            body.expectedRevision,
            str(body.blockId) if body.blockId else None,
            body.title,
            body.content,
            body.reason,
            body.action,
        ),
    )


@router.get("/history")
async def history(ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result={"edits": await (await store(ctx)).history()})


@router.post("/history/{edit_id}/receipt")
async def receipt(edit_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    await (await store(ctx)).review_receipt(str(edit_id))
    return Response(status="ok", result={"accepted": True})
