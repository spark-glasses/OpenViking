"""Profile writes are deterministic: the application sends the identity and an
agent's sections; native extraction writes what it learns."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.profile_store import ProfileStore

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])


_SHORT = Annotated[str, StringConstraints(min_length=1, max_length=500)]


class ConnectedAccount(BaseModel):
    """Who the user is in one connected app, as far as that app says."""

    model_config = ConfigDict(extra="forbid")
    email: _SHORT | None = None
    name: _SHORT | None = None
    handle: _SHORT | None = None
    userId: _SHORT | None = None
    title: _SHORT | None = None
    workspace: _SHORT | None = None
    workspaceUrl: _SHORT | None = None
    teams: list[_SHORT] | None = Field(default=None, max_length=100)
    company: _SHORT | None = None
    tradeNames: list[_SHORT] | None = Field(default=None, max_length=20)


class Identity(BaseModel):
    """Who the user is: the application's database holds it, this is its copy.

    An account is an address when only the address is known, a short note when
    the app says nothing about the user, and otherwise what the app says.
    """

    model_config = ConfigDict(extra="forbid")
    names: list[Annotated[str, StringConstraints(min_length=1, max_length=255)]] = Field(
        max_length=50
    )
    connectedAccounts: dict[_SHORT, list[_SHORT | ConnectedAccount]]


class IdentitySync(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1, le=9007199254740991, strict=True)
    identity: Identity


class BodyWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expectedRevision: int = Field(ge=0)
    content: str = Field(max_length=200000)


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
async def identity(body: IdentitySync, ctx: RequestContext = Depends(get_request_context)):
    return Response(
        status="ok",
        result=await (await store(ctx)).sync_identity(
            body.revision, body.identity.model_dump(exclude_none=True)
        ),
    )


@router.post("/body")
async def body(edit: BodyWrite, ctx: RequestContext = Depends(get_request_context)):
    return Response(
        status="ok",
        result=await (await store(ctx)).write_body(edit.expectedRevision, edit.content),
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
