"""Authenticated Focus catalog; user actions have no connector dependency."""

from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.focus_store import FocusStore

router = APIRouter(prefix="/api/v1/focuses", tags=["focuses"])


class CreateFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    createKey: str = Field(min_length=1, max_length=200)
    userIntent: str = Field(default="", max_length=4000)


class FocusFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    userIntent: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "archived"] | None = None


class EditFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expectedRevision: int = Field(ge=1)
    operationId: UUID
    fields: FocusFields


def store(ctx):
    service = get_service()
    return FocusStore(service.viking_fs, ctx, service.vikingdb_manager)


@router.get("")
async def listing(
    cursor: UUID | None = None,
    limit: int = Query(50, ge=1, le=100),
    status: Literal["active", "archived", "forming", "all"] = "active",
    ctx: RequestContext = Depends(get_request_context),
):
    return Response(
        status="ok",
        result=await store(ctx).list(str(cursor) if cursor else None, limit, status),
    )


@router.post("")
async def create(body: CreateFocus, ctx: RequestContext = Depends(get_request_context)):
    await get_service().initialize_user_directories(ctx)
    return Response(
        status="ok",
        result=await store(ctx).ensure(
            name=body.name, create_key=body.createKey, origin="user", user_intent=body.userIntent
        ),
    )


@router.get("/{focus_id}")
async def get(focus_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    value = store(ctx)
    return Response(status="ok", result=value.view(await value.get(focus_id)))


@router.patch("/{focus_id}")
async def edit(focus_id: UUID, body: EditFocus, ctx: RequestContext = Depends(get_request_context)):
    return Response(
        status="ok",
        result=await store(ctx).update(
            focus_id,
            expected_revision=body.expectedRevision,
            fields=body.fields.model_dump(exclude_none=True),
            operation_id=str(body.operationId),
        ),
    )
