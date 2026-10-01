"""A durable per-user identity-change barrier shared by every extraction writer.

The exact gate lock is also acquired by CompressorV2. A successful freeze thus
means earlier writers finished; later writers must observe the frozen operation.
No DB/OV distributed transaction is assumed. Spark retries the same operation ID.
"""

import json
import re
from contextlib import asynccontextmanager
from uuid import UUID
from openviking.session.memory.person_identity import memory_root, person_memory_uri
from openviking_cli.exceptions import ConflictError, NotFoundError


def gate_uri(ctx):
    return memory_root(ctx) + "people/.identity-operation.json"


async def read_gate(fs, ctx):
    try:
        return json.loads(await fs.read_file(gate_uri(ctx), ctx=ctx))
    except NotFoundError:
        return {"operationId": None}


async def check_gate(fs, ctx, operation_id=None, expected_epoch=None):
    gate = await read_gate(fs, ctx)
    if expected_epoch is not None and gate.get("epoch", 0) != expected_epoch:
        raise ConflictError("People identity changed while preparing context; rebuild context")
    if gate.get("operationId") and gate.get("memoryOperationId") != operation_id:
        raise ConflictError(
            "Person identity change in progress; retry extraction after reconciliation"
        )


@asynccontextmanager
async def gate_lock(fs, ctx):
    from openviking.storage.transaction import LockContext, get_lock_manager, init_lock_manager

    if not getattr(fs, "agfs", None):
        raise RuntimeError("Identity changes require the shared storage lock manager")
    try:
        manager = get_lock_manager()
    except RuntimeError:
        manager = init_lock_manager(fs.agfs)
    async with LockContext(manager, [fs._uri_to_path(gate_uri(ctx), ctx)], lock_mode="exact"):
        yield


async def change_gate(
    fs, ctx, operation_id, action, anchors, kind, memory_operation_id, attempt=0, db=None
):
    operation_id = str(UUID(operation_id))
    if not re.fullmatch(r"[a-f0-9]{64}", memory_operation_id):
        raise ValueError("Invalid memory operation ID")
    if kind not in ("merge", "reassign"):
        raise ValueError("Invalid identity operation")
    anchors = [str(UUID(anchor)) for anchor in anchors]
    archive = memory_root(ctx) + f"people/.identity-operations/{operation_id}.json"
    async with gate_lock(fs, ctx):
        gate = await read_gate(fs, ctx)
        if gate.get("operationId") not in (None, operation_id):
            raise ConflictError("Another People operation owns the identity barrier")
        try:
            record = json.loads(await fs.read_file(archive, ctx=ctx))
        except NotFoundError:
            record = None
        if record and (record["anchors"] != anchors or record["kind"] != kind):
            raise ConflictError("People operation ID reused with different identity change")
        if record and (
            attempt < record.get("attempt", 0)
            or (
                attempt == record.get("attempt", 0)
                and record.get("memoryOperationId") not in (None, memory_operation_id)
            )
        ):
            raise ConflictError("Stale People reconciliation attempt")
        if action == "freeze":
            if record and record["anchors"] != anchors:
                raise ConflictError("People operation ID reused with different anchors")
            if record and record.get("state") == "done":
                return record
            if record is None:
                originals = {}
                for anchor in anchors:
                    uri = person_memory_uri(ctx, anchor)
                    try:
                        originals[uri] = await fs.read_file(uri, ctx=ctx)
                    except NotFoundError:
                        originals[uri] = None
                record = {
                    "operationId": operation_id,
                    "anchors": anchors,
                    "state": "frozen",
                    "originals": originals,
                    "kind": kind,
                }
                await fs.write_file(archive, json.dumps(record, ensure_ascii=False), ctx=ctx)
            record.update(memoryOperationId=memory_operation_id, attempt=attempt)
            await fs.write_file(archive, json.dumps(record, ensure_ascii=False), ctx=ctx)
            epoch = gate.get("epoch", 0) + int(
                (gate.get("operationId"), gate.get("memoryOperationId"))
                != (operation_id, memory_operation_id)
            )
            await fs.write_file(
                gate_uri(ctx),
                json.dumps(
                    {
                        "operationId": operation_id,
                        "memoryOperationId": memory_operation_id,
                        "epoch": epoch,
                    }
                ),
                ctx=ctx,
            )
        elif action == "finish":
            if record is None:
                raise ConflictError("Identity operation was not frozen")
            if gate.get("operationId") and gate.get("memoryOperationId") != memory_operation_id:
                raise ConflictError("A newer reconciliation owns the barrier")
            if kind == "merge":
                await finish_merge(fs, ctx, anchors, operation_id, db)
            record["state"] = "done"
            await fs.write_file(archive, json.dumps(record, ensure_ascii=False), ctx=ctx)
            await fs.write_file(
                gate_uri(ctx),
                json.dumps(
                    {
                        "operationId": None,
                        "epoch": gate.get("epoch", 0) + int(bool(gate.get("operationId"))),
                    }
                ),
                ctx=ctx,
            )
        else:
            raise ValueError("Unknown People operation action")
        return record


async def finish_merge(fs, ctx, anchors, operation_id, db):
    """Keep historical content readable, with an explicit canonical destination.

    Questions keep their IDs, answers and delivery history when ownership moves.
    This is not a rollback of arbitrary facts propagated into other memories.
    """
    from openviking.session.memory.question_store import QuestionStore
    from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
    from openviking.session.memory.memory_updater import MemoryUpdater

    source, target = (person_memory_uri(ctx, anchor) for anchor in anchors)
    try:
        page = MemoryFileUtils.read(await fs.read_file(source, ctx=ctx), uri=source)
    except NotFoundError:
        page = None
    if page is not None:
        page.extra_fields["redirectUri"] = target
        page.extra_fields["identityOperationId"] = operation_id
        await fs.write_file(source, MemoryFileUtils.write(page), ctx=ctx)
        if db is not None:
            await MemoryUpdater.refresh_file_embedding(
                viking_fs=fs, vikingdb=db, uri=source, memory_type="people", ctx=ctx
            )
    await QuestionStore(fs, ctx, db).move_person(anchors[0], anchors[1], target)
