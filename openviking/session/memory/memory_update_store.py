"""Durable operation receipt around the existing native extraction loop.

A frozen input is accepted once. A lost request response can be looked up by the
same operation ID. Ambiguous writes are exposed for recovery, never replayed.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from openviking.core.namespace import canonical_session_uri
from openviking.message import Message, TextPart
from openviking.session.memory.memory_update_context import MemoryUpdateContext
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError

_BACKGROUND_TASKS = set()
_LOCAL_LOCKS = {}
_ACTIVE_SECONDS = 480


class MemoryUpdateStore:
    def __init__(self, fs, ctx, compressor, sessions=None):
        self.fs, self.ctx, self.compressor, self.sessions = fs, ctx, compressor, sessions

    def location(self, operation_id):
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", operation_id):
            raise InvalidArgumentError("Invalid memory operation ID")
        session_id = "memory-update-" + operation_id
        uri = canonical_session_uri(self.ctx, session_id)
        return session_id, uri, uri + "/history/archive_001"

    @asynccontextmanager
    async def _lock(self, operation_id):
        _, uri, _ = self.location(operation_id)
        if getattr(self.fs, "agfs", None):
            from openviking.storage.transaction import (
                LockContext,
                get_lock_manager,
                init_lock_manager,
            )

            try:
                manager = get_lock_manager()
            except RuntimeError:
                manager = init_lock_manager(self.fs.agfs)
            async with LockContext(
                manager,
                [self.fs._uri_to_path(uri + "/memory_update.json", self.ctx)],
                lock_mode="exact",
            ):
                yield
        else:
            key = (id(self.fs), self.ctx.user.account_id, self.ctx.user.user_id, operation_id)
            async with _LOCAL_LOCKS.setdefault(key, asyncio.Lock()):
                yield

    async def _load(self, operation_id):
        _, uri, _ = self.location(operation_id)
        return json.loads(await self.fs.read_file(uri + "/memory_update.json", ctx=self.ctx))

    async def _save(self, record):
        _, uri, _ = self.location(record["operationId"])
        record["updatedAt"] = time.time()
        await self.fs.write_file(
            uri + "/memory_update.json",
            json.dumps(record, ensure_ascii=False, indent=2),
            ctx=self.ctx,
        )

    async def get(self, operation_id):
        record = await self._load(operation_id)
        # GET stays read-only, including when called by the source bridge.
        if record["status"] == "running" and time.time() > record["deadlineAt"]:
            record = {
                **record,
                "status": "needsRecovery" if record.get("phase") == "applying" else "failed",
                "errors": record.get("errors", [])
                + ["Operation interrupted or exceeded its deadline"],
                "retryable": record.get("phase") != "applying",
            }
        return record

    async def submit(self, value):
        spec = MemoryUpdateContext.model_validate(value)
        spec.validate_owner(self.ctx)
        async with self._lock(spec.operationId):
            try:
                record = await self._load(spec.operationId)
            except NotFoundError:
                record = None
            digest = spec.input_hash()
            if record:
                if record["inputHash"] != digest:
                    raise AlreadyExistsError(
                        "Memory operation ID already belongs to a different input"
                    )
                if record["status"] == "accepted":
                    self._launch(spec.operationId)
                return await self.get(spec.operationId)
            session_id, _, archive_uri = self.location(spec.operationId)
            if self.sessions is not None:
                try:
                    await self.sessions.create(self.ctx, session_id)
                except AlreadyExistsError:
                    pass
            # Input precedes the receipt. Repeating a crash before receipt simply
            # replaces the same still-unaccepted snapshot; no memory ran yet.
            await self.fs.write_file(
                archive_uri + "/memory_update_context.json",
                json.dumps(spec.model_dump(), ensure_ascii=False),
                ctx=self.ctx,
            )
            await self.fs.write_file(
                archive_uri + "/messages.jsonl",
                "\n".join(
                    json.dumps(m.model_dump(exclude_none=True), ensure_ascii=False)
                    for m in spec.messages
                ),
                ctx=self.ctx,
            )
            record = {
                "operationId": spec.operationId,
                "inputHash": digest,
                "userId": self.ctx.user.user_id,
                "sourceKinds": spec.sourceKinds,
                "status": "accepted",
                "phase": "accepted",
                "sessionId": session_id,
                "archiveUri": archive_uri,
                "origin": spec.origin.model_dump(),
                "collaboration": spec.collaboration.model_dump(exclude_none=True) if spec.collaboration else None,
                "sourceMessageRefs": sorted({m.sourceRef for m in spec.messages if m.sourceRef}),
                "appliedUris": [],
                "errors": [],
                "partial": False,
                "primaryChangeStatus": "unverified",
                "createdAt": time.time(),
                "retryable": False,
            }
            await self._save(record)
        self._launch(spec.operationId)
        return record

    def _launch(self, operation_id):
        task = asyncio.create_task(self.run(operation_id))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
        # run persists ordinary failures; retrieve a storage failure to avoid
        # unobserved task exceptions while leaving an inspectable stale receipt.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())

    async def retry(self, operation_id):
        async with self._lock(operation_id):
            record = await self.get(operation_id)
            if record["status"] != "failed" or not record.get("retryable"):
                raise InvalidArgumentError(
                    "Only failed pre-apply operations can be retried; inspect ambiguous or partial writes first"
                )
            record.update(status="accepted", phase="accepted", errors=[], retryable=False)
            await self._save(record)
        self._launch(operation_id)
        return record

    async def run(self, operation_id):
        async with self._lock(operation_id):
            record = await self._load(operation_id)
            if record["status"] != "accepted":
                return
            attempt_id = uuid4().hex
            record.update(
                status="running",
                phase="reasoning",
                attemptId=attempt_id,
                deadlineAt=time.time() + _ACTIVE_SECONDS,
            )
            await self._save(record)
        archive_uri = record["archiveUri"]
        try:
            spec = MemoryUpdateContext.model_validate(
                json.loads(
                    await self.fs.read_file(
                        archive_uri + "/memory_update_context.json", ctx=self.ctx
                    )
                )
            )
            if spec.input_hash() != record["inputHash"]:
                raise ValueError("Frozen operation context changed")
            if self.compressor is None:
                raise RuntimeError("Memory extraction is unavailable")

            async def before_apply(operations):
                async with self._lock(operation_id):
                    live = await self._load(operation_id)
                    if live.get("attemptId") != attempt_id or live["status"] != "running":
                        raise RuntimeError("Memory operation attempt was superseded")
                    await self.fs.write_file(
                        archive_uri + "/memory_update_operations.json",
                        operations.model_dump_json(),
                        ctx=self.ctx,
                    )
                    live.update(
                        phase="applying",
                        potentialUris=list(
                            dict.fromkeys(
                                uri for op in operations.upsert_operations for uri in op.uris
                            )
                        ),
                    )
                    await self._save(live)

            async def after_apply(result):
                async with self._lock(operation_id):
                    live = await self._load(operation_id)
                    if live.get("attemptId") != attempt_id:
                        raise RuntimeError("Memory update receipt attempt mismatch")
                    applied = list(dict.fromkeys(result["writtenUris"] + result["editedUris"]))
                    errors = result["errors"]
                    needs_clarification = bool(result.get("unresolvedItems")) and not result.get(
                        "nonQuestionUris", applied
                    )
                    status = (
                        "needsRecovery"
                        if errors
                        else "needsClarification"
                        if needs_clarification
                        else "completed"
                        if result["changed"]
                        else "noChange"
                    )
                    live.update(
                        status=status,
                        phase="finished",
                        appliedUris=applied,
                        errors=errors,
                        partial=bool(errors and applied),
                        questionRefs=result["questionRefs"],
                        unresolvedItems=result.get("unresolvedItems", []),
                        sourceRefs=result["sourceRefs"],
                        retryable=False,
                    )
                    await self._save(live)

            # One native extraction call; no working-memory or skill extraction.
            # Original roles are preserved in the Provider's snapshot instead of
            # flattening tool receipts into fabricated user utterances.
            messages = [Message(id=operation_id, role="user", parts=[TextPart(spec.text)])]
            await asyncio.wait_for(
                self.compressor.extract_long_term_memories(
                    messages=messages,
                    user=self.ctx.user,
                    session_id=record["sessionId"],
                    ctx=self.ctx,
                    strict_extract_errors=True,
                    archive_uri=archive_uri,
                    memory_update_context=spec.model_dump(),
                    memory_update_before_apply=before_apply,
                    memory_update_after_apply=after_apply,
                ),
                timeout=420,
            )
            final = await self._load(operation_id)
            if final["status"] == "running":
                raise RuntimeError("Native extraction returned without a persisted outcome")
        except (Exception, asyncio.CancelledError) as exc:
            async with self._lock(operation_id):
                live = await self._load(operation_id)
                if live.get("attemptId") != attempt_id or live["status"] != "running":
                    return
                applying = live.get("phase") == "applying"
                live.update(
                    status="needsRecovery" if applying else "failed",
                    errors=[type(exc).__name__ + ": " + str(exc)],
                    partial=applying,
                    retryable=not applying,
                )
                await self._save(live)
