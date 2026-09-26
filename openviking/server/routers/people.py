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
