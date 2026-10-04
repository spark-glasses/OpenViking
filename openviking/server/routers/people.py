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


from openviking.session.memory.people_profile_proposals import pending, queue_root, receipt_uri
from openviking.session.memory.question_service import QuestionService
from openviking.session.memory.memory_write_context import MemoryWriteContext
import json
import hashlib
from openviking_cli.exceptions import NotFoundError


@router.get("/profile-proposals")
async def profile_proposals(ctx: RequestContext = Depends(get_request_context)):
    service = get_service()
    return Response(status="ok", result={"proposals": await pending(service.viking_fs, ctx)})


class ProfileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome: Literal["applied", "conflict", "obsolete"]
    currentValue: str | None = Field(default=None, max_length=2000)


@router.post("/profile-proposals/receipt")
async def profile_receipt(body: ProfileReceipt, ctx: RequestContext = Depends(get_request_context)):
    service = get_service()
    uri = queue_root(ctx) + body.id + ".json"
    receipt = receipt_uri(ctx, body.id)
    try:
        completed = json.loads(await service.viking_fs.read_file(receipt, ctx=ctx))
        try:
            await service.viking_fs.rm(uri, ctx=ctx)
        except NotFoundError:
            pass
        return Response(status="ok", result={"status": completed["status"]})
    except NotFoundError:
        pass
    record = json.loads(await service.viking_fs.read_file(uri, ctx=ctx))
    if record.get("status") != "pending":
        return Response(status="ok", result={"status": record["status"]})
    if body.outcome == "conflict":
        subject = {"kind": "person", "id": record["anchorId"], "memoryUri": record["memoryUri"]}
        # Stable per-observation topic preserves a denial instead of re-asking
        # every time an unchanged document is projected back to the model.
        await QuestionService(MemoryWriteContext(service.viking_fs, ctx), service.vikingdb_manager).discover_business_event(
            subject,
                {
                    "topicKey": "profile-"
                    + hashlib.sha256(
                        json.dumps(
                            [
                                record["personId"],
                                record["field"],
                                body.currentValue,
                                record["value"],
                            ],
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                    "text": f"You previously set {record['field']} to {(body.currentValue or '(empty)')[:220]}. A new source says {record['value'][:220]}. Should I update it?",
                    "sourceRefs": record["sourceRefs"],
                    "context": {"summary": "New evidence conflicts with a user-maintained profile field.",
                                "uncertainty": "Should the user's value be updated?",
                                "knownFacts": [f"User value: {body.currentValue}", f"Source proposal: {record['value']}"],
                                "candidates": []},
                    "importance": {"level": "later", "reason": "Preserve user correction until explicitly approved"},
                    "purpose": "profileCorrection",
                    "scope": {
                        "kind": "profileCorrection",
                        "personId": record["personId"],
                        "field": record["field"],
                        "proposedValue": record["value"],
                        "previousValue": body.currentValue,
                    },
                },
            event_ref="profile-proposal:" + body.id,
            event={"proposal": record, "currentValue": body.currentValue},
        )
    record["status"] = body.outcome
    await service.viking_fs.write_file(receipt, json.dumps(record, ensure_ascii=False), ctx=ctx)
    await service.viking_fs.rm(uri, ctx=ctx)
    return Response(status="ok", result={"status": body.outcome})


@router.get("/directory")
async def person_directory(ctx: RequestContext = Depends(get_request_context)):
    from openviking.session.memory.person_identity import load_identity_directory

    directory = await load_identity_directory(get_service().viking_fs, ctx)
    return Response(status="ok", result={"people": [
        {"anchorId": p["anchorId"], "displayName": p["displayName"], "deleted": p["deleted"]}
        for p in directory["people"].values()
    ]})
