"""Revisioned contact-to-People projections with repairable indexing receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater, MemoryUpdateResult
from openviking.session.memory.person_identity import (
    MAX_DIRECTORY_BYTES,
    MAX_PEOPLE,
    apply_person_identity,
    compact_identity,
    identity_directory_uri,
    identity_uri,
    load_identity_directory,
    load_person_identity,
    person_aliases,
    person_memory_uri,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import ConflictError, InvalidArgumentError, NotFoundError

_SHORT = Annotated[str, StringConstraints(max_length=500)]
_NAME = Annotated[str, StringConstraints(max_length=2000)]
_LOCKS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class ContactProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    displayName: str = Field(min_length=1, max_length=2000)
    aliases: list[_NAME] = Field(default_factory=list, max_length=100)
    givenName: _SHORT | None = None
    middleName: _SHORT | None = None
    familyName: _SHORT | None = None
    nickname: _SHORT | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    organization: _SHORT | None = None
    jobTitle: _SHORT | None = None
    notes: str | None = None

    @field_validator("displayName")
    @classmethod
    def nonblank_name(cls, value):
        if not value.strip():
            raise ValueError("Contact displayName cannot be blank")
        return value.strip()


class SyncPersonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contactId: UUID
    anchorId: UUID
    revision: int = Field(ge=1, le=9007199254740991, strict=True)
    profile: ContactProfile
    deleted: bool = False


class PersonContactStore:
    def __init__(self, viking_fs, ctx, vikingdb=None):
        self.fs, self.ctx, self.db = viking_fs, ctx, vikingdb

    @asynccontextmanager
    async def _lock(self, anchor):
        # The directory exact lock serializes identity remaps; the canonical md
        # exact lock conflicts with native extraction's people subtree lock.
        uris = [
            identity_directory_uri(self.ctx),
            identity_uri(self.ctx, anchor),
            person_memory_uri(self.ctx, anchor),
        ]
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
                manager, [self.fs._uri_to_path(uri, self.ctx) for uri in uris], lock_mode="exact"
            ):
                yield
        else:
            key = (id(self.fs), self.ctx.account_id, self.ctx.user.user_id)
            lock = _LOCKS.get(key)
            if lock is None:
                lock = asyncio.Lock()
                _LOCKS[key] = lock
            async with lock:
                yield

    async def _save_record(self, record):
        await self.fs.write_file(
            identity_uri(self.ctx, record["anchorId"]),
            json.dumps(record, ensure_ascii=False, indent=2),
            ctx=self.ctx,
        )

    def _result(self, record, status):
        return {
            key: record[key]
            for key in ("memoryUri", "contactId", "anchorId", "revision", "deleted", "indexStatus")
        } | {"status": status}

    async def _index(self, uri):
        if not self.db or not bool(getattr(self.db, "has_queue_manager", False)):
            return "unavailable"
        updater = MemoryUpdater(registry=create_default_registry(), vikingdb=self.db)
        updater._viking_fs = self.fs
        updater.strict_merge_errors = True
        result = MemoryUpdateResult()
        result.add_written(uri)
        count = await updater._vectorize_memories(
            result, self.ctx, uri_memory_type_map={uri: "people"}
        )
        if result.errors or count != 1:
            raise RuntimeError("Contact projection embedding enqueue failed")
        return "queued"

    async def sync(self, value):
        request = SyncPersonRequest.model_validate(value)
        payload = request.model_dump(mode="json")
        contact_id, anchor = payload["contactId"], payload["anchorId"]
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        async with self._lock(anchor):
            directory = await load_identity_directory(self.fs, self.ctx)
            previous = await load_person_identity(self.fs, self.ctx, anchor)
            bound = directory["contacts"].get(contact_id)
            if bound and bound != anchor:
                raise ConflictError(
                    "This contact is already mapped to another stable person anchor"
                )
            if previous and request.revision < previous["revision"]:
                return self._result(previous, "stale")
            if (
                previous
                and request.revision == previous["revision"]
                and digest != previous["payloadHash"]
            ):
                raise ConflictError(
                    "Contact revision already belongs to a different profile snapshot"
                )
            uri = person_memory_uri(self.ctx, anchor)
            try:
                raw = await self.fs.read_file(uri, ctx=self.ctx)
                memory = MemoryFileUtils.read(raw, uri=uri)
            except NotFoundError:
                raw = None
                memory = MemoryFile.from_parsed(
                    uri=uri, parsed={"content": "", "memory_type": "people"}
                )
            if previous and request.revision == previous["revision"]:
                record = previous
            else:
                if anchor not in directory["people"] and len(directory["people"]) >= MAX_PEOPLE:
                    raise InvalidArgumentError("Contact projection directory capacity exceeded")
                profile = request.profile.model_dump(mode="json", exclude_none=True)
                record = {
                    "formatVersion": 1,
                    "accountId": self.ctx.account_id,
                    "userId": self.ctx.user.user_id,
                    "contactId": contact_id,
                    "anchorId": anchor,
                    "revision": request.revision,
                    "memoryUri": uri,
                    "deleted": request.deleted,
                    "profile": profile,
                    "displayName": profile["displayName"],
                    "aliases": person_aliases(profile),
                    "payloadHash": digest,
                    "projectionState": "pending",
                    "indexStatus": "unavailable",
                }
            expected = MemoryFileUtils.write(apply_person_identity(memory, record))
            compact = compact_identity(record)
            already_complete = (
                record["projectionState"] == "complete"
                and raw == expected
                and directory["people"].get(anchor) == compact
                and directory["contacts"].get(contact_id) == anchor
                and (
                    record["indexStatus"] == "queued"
                    or not self.db
                    or not getattr(self.db, "has_queue_manager", False)
                )
            )
            if already_complete:
                return self._result(record, "unchanged")
            # Validate bounded discovery data before accepting the revision.
            final_directory = {
                **directory,
                "people": {**directory["people"], anchor: compact},
                "contacts": {**directory["contacts"], contact_id: anchor},
            }
            encoded_directory = json.dumps(final_directory, ensure_ascii=False, indent=2)
            if len(encoded_directory.encode()) > MAX_DIRECTORY_BYTES:
                raise InvalidArgumentError("Contact identity directory exceeds its byte limit")
            # Reserve the stable contact-to-anchor mapping before the profile
            # record. A crash cannot let a different anchor claim the same ID.
            if directory["contacts"].get(contact_id) != anchor:
                directory["contacts"][contact_id] = anchor
                await self.fs.write_file(
                    identity_directory_uri(self.ctx),
                    json.dumps(directory, ensure_ascii=False, indent=2),
                    ctx=self.ctx,
                )
            # Persist the authoritative latest revision before the md/index.
            # Retrying the same revision repairs any later interrupted step.
            record["projectionState"] = "pending"
            await self._save_record(record)
            if raw != expected:
                await self.fs.write_file(uri, expected, ctx=self.ctx)
            directory["people"][anchor] = compact
            directory["contacts"][contact_id] = anchor
            # Keep prior contact IDs bound to their historical anchor so delayed
            # uploads cannot silently reassign an old identity to another person.
            await self.fs.write_file(
                identity_directory_uri(self.ctx), encoded_directory, ctx=self.ctx
            )
            record["indexStatus"] = await self._index(uri)
            record["projectionState"] = "complete"
            await self._save_record(record)
            return self._result(record, "synced")
