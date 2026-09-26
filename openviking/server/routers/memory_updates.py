"""Contextual memory updates use one immutable operation and native extraction."""

from fastapi import APIRouter, Depends

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory.memory_update_context import MemoryUpdateContext
from openviking.session.memory.memory_update_store import MemoryUpdateStore

router = APIRouter(prefix="/api/v1/memory-updates", tags=["memory-updates"])


def store(ctx):
    service = get_service()
    return MemoryUpdateStore(service.viking_fs, ctx, service.session_compressor, service.sessions)


@router.post("")
async def submit(body: MemoryUpdateContext, ctx: RequestContext = Depends(get_request_context)):
    await get_service().initialize_user_directories(ctx)
    return Response(status="ok", result=await store(ctx).submit(body))


@router.get("/{operation_id}")
async def get(operation_id: str, ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result=await store(ctx).get(operation_id))


@router.post("/{operation_id}/retry")
async def retry(operation_id: str, ctx: RequestContext = Depends(get_request_context)):
    return Response(status="ok", result=await store(ctx).retry(operation_id))
