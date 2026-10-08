"""Authenticated deterministic contact projections into canonical People memory."""

from fastapi import APIRouter, Depends

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.person_contact_store import PersonContactStore, SyncPersonRequest

router = APIRouter(prefix="/api/v1/people", tags=["people"])


@router.post("/sync")
async def sync_person(body: SyncPersonRequest, ctx: RequestContext = Depends(get_request_context)):
    service = get_service()
    await service.initialize_user_directories(ctx)
    return Response(
        status="ok",
        result=await PersonContactStore(service.viking_fs, ctx, service.vikingdb_manager).sync(
            body
        ),
    )


from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from uuid import UUID
from openviking.session.memory.people_operation import change_gate


class PeopleOperationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operationId: UUID
    action: Literal["freeze", "finish"]
    kind: Literal["merge", "reassign"]
    attempt: int = Field(ge=0)
    memoryOperationId: str = Field(pattern=r"^[a-f0-9]{64}$")
    anchors: list[UUID] = Field(min_length=2, max_length=2)


@router.post("/operation")
async def person_operation(
    body: PeopleOperationRequest, ctx: RequestContext = Depends(get_request_context)
):
    service = get_service()
    await service.initialize_user_directories(ctx)
    result = await change_gate(
        service.viking_fs,
        ctx,
        str(body.operationId),
        body.action,
        [str(anchor) for anchor in body.anchors],
        body.kind,
        str(body.memoryOperationId),
        body.attempt,
        service.vikingdb_manager,
    )
    return Response(status="ok", result=result)


@router.get("/directory")
async def person_directory(ctx: RequestContext = Depends(get_request_context)):
    from openviking.session.memory.person_identity import load_identity_directory

    directory = await load_identity_directory(get_service().viking_fs, ctx)
    shown = ("anchorId", "displayName", "deleted", "aliases", "emails", "phones", "artifactId")
    return Response(
        status="ok",
        result={
            "people": [{key: p[key] for key in shown} for p in directory["people"].values()]
        },
    )
