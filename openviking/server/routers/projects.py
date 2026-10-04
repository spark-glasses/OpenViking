"""Authenticated Project catalog; its only persistence is OV memory files."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.project_store import ProjectStore

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


class EnsureProject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    createKey: str = Field(min_length=1, max_length=200)
    source: dict | None = None


class EditProject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expectedRevision: int = Field(ge=1)
    fields: dict


def store(ctx):
    service = get_service()
    return ProjectStore(service.viking_fs, ctx, service.vikingdb_manager)


@router.get("")
async def listing(
    cursor: UUID | None = None,
    limit: int = Query(50, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context),
):
    return Response(
        status="ok", result=await store(ctx).list(str(cursor) if cursor else None, limit)
    )


@router.post("")
async def ensure(body: EnsureProject, ctx: RequestContext = Depends(get_request_context)):
    await get_service().initialize_user_directories(ctx)
    return Response(
        status="ok",
        result=await store(ctx).ensure(
            name=body.name, create_key=body.createKey, source=body.source
        ),
    )


@router.get("/{project_id}")
async def get(project_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    value = store(ctx)
    return Response(status="ok", result=value.view(await value.get(project_id)))


@router.patch("/{project_id}")
async def edit(
    project_id: UUID, body: EditProject, ctx: RequestContext = Depends(get_request_context)
):
    return Response(
        status="ok",
        result=await store(ctx).update(
            project_id, expected_revision=body.expectedRevision, fields=body.fields
        ),
    )
