"""Personal priorities in OV, with conditional writes and recoverable indexing.

User intent and lifecycle choices are authoritative metadata. The extraction
model maintains the narrative and can activate grounded forming discoveries,
but cannot undo a user's archiving decision.
"""

import hashlib
import json
from contextlib import asynccontextmanager
from uuid import NAMESPACE_URL, UUID, uuid5

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.focus_paths import focus_directory_uri, focus_metadata_uri, focus_uri
from openviking.session.memory.question_store import memory_root, now_iso, owner_lock
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError


class FocusStore:
    def __init__(self, fs, ctx, db=None, lock_handle=None):
        self.fs, self.ctx, self.db = fs, ctx, db
        self.lock_handle = lock_handle

    @asynccontextmanager
    async def _write_guard(self):
        # Native extraction already owns the directory lock. API writes acquire
        # the same lock before mutating either file; a busy model never causes a
        # half-applied UI correction. Isolated in-memory test FS has no AGFS locks.
        if self.lock_handle is not None or not hasattr(self.fs, "_uri_to_path"):
            async with owner_lock(self.ctx):
                yield
            return
        from openviking.storage.transaction import LockContext, get_lock_manager
        from openviking.storage.errors import LockAcquisitionError

        try:
            async with LockContext(
                get_lock_manager(),
                [self.fs._uri_to_path(memory_root(self.ctx) + "focuses", ctx=self.ctx)],
                lock_mode="tree",
            ) as handle:
                self.lock_handle = handle
                try:
                    async with owner_lock(self.ctx):
                        yield
                finally:
                    self.lock_handle = None
        except LockAcquisitionError as error:
            raise AlreadyExistsError(
                "Focus is being updated; retry after the current memory update"
            ) from error

    async def get(self, identifier):
        # Reads can finish an interrupted two-file write, so they use the same
        # cross-process guard as writers, not only the in-process owner lock.
        async with self._write_guard():
            return await self._get(identifier)

    async def _get(self, identifier):
        identifier = str(UUID(str(identifier)))
        await self._recover(identifier)
        uri = focus_uri(self.ctx, identifier)
        page = MemoryFileUtils.read(await self.fs.read_file(uri, ctx=self.ctx), uri=uri)
        fields = json.loads(
            await self.fs.read_file(focus_metadata_uri(self.ctx, identifier), ctx=self.ctx)
        )
        if fields.get("focusId") != identifier:
            raise InvalidArgumentError("Focus identity does not match its URI")
        page.extra_fields.update(fields)
        return page

    @staticmethod
    def view(page, include_content=True):
        fields = page.extra_fields
        return (
            {
                key: fields.get(key)
                for key in (
                    "focusId",
                    "name",
                    "aliases",
                    "userIntent",
                    "origin",
                    "status",
                    "summary",
                    "discoveryReason",
                    "sourceRefs",
                    "revision",
                    "createdAt",
                    "updatedAt",
                )
            }
            | {
                "uri": page.uri,
                "directoryUri": page.uri.rsplit("/", 1)[0],
                "metadataUri": page.uri.rsplit("/", 1)[0] + "/focus.json",
                "questionsUri": page.uri.rsplit("/", 1)[0] + "/questions.md",
            }
            | ({"content": page.content} if include_content else {})
        )

    async def list(self, after=None, limit=50, status="all"):
        async with self._write_guard():
            return await self._list(after, limit, status)

    async def _list(self, after=None, limit=50, status="all"):
        if status not in ("all", "forming", "active", "archived"):
            raise InvalidArgumentError("Invalid Focus status filter")
        if after:
            after = str(UUID(after))
        root = memory_root(self.ctx) + "focuses"
        try:
            entries = await self.fs.ls(root, node_limit=10001, ctx=self.ctx)
            if len(entries) >= 10001:
                raise InvalidArgumentError("Focus catalog exceeds listing limit")
        except NotFoundError:
            entries = []
        identifiers = set()
        for entry in entries:
            stem = (
                (entry.get("uri") or root + "/" + entry.get("name", ""))
                .removeprefix(root + "/")
                .rstrip("/")
            )
            try:
                identifier = str(UUID(stem))
            except ValueError:
                continue
            if not after or identifier > after:
                identifiers.add(identifier)
        limit = max(1, min(100, limit))
        pages = []
        for identifier in sorted(identifiers):
            page = self.view(await self._get(identifier), include_content=False)
            if status == "all" or page["status"] == status:
                pages.append(page)
            if len(pages) > limit:
                break
        return {
            "focuses": pages[:limit],
            "nextCursor": pages[limit - 1]["focusId"] if len(pages) > limit else None,
        }

    @staticmethod
    def _validate(fields):
        for key, maximum in (
            ("name", 200),
            ("userIntent", 4000),
            ("summary", 600),
            ("discoveryReason", 2000),
            ("content", 200000),
        ):
            if key in fields and (
                not isinstance(fields[key], str)
                or len(fields[key]) > maximum
                or (key == "name" and not fields[key].strip())
            ):
                raise InvalidArgumentError(f"Invalid Focus {key}")
        if "status" in fields and fields["status"] not in ("forming", "active", "archived"):
            raise InvalidArgumentError("Invalid Focus status")
        if "aliases" in fields and (
            not isinstance(fields["aliases"], list)
            or len(fields["aliases"]) > 50
            or any(not isinstance(a, str) or not 1 <= len(a) <= 200 for a in fields["aliases"])
        ):
            raise InvalidArgumentError("Invalid Focus aliases")

    async def ensure(
        self,
        *,
        name,
        create_key,
        origin="discovered",
        user_intent="",
        discovery_reason="",
        source_refs=(),
    ):
        self._validate(
            {"name": name, "userIntent": user_intent, "discoveryReason": discovery_reason}
        )
        if (
            origin not in ("user", "discovered")
            or not isinstance(create_key, str)
            or not 1 <= len(create_key) <= 500
        ):
            raise InvalidArgumentError("Invalid Focus creation")
        if origin == "discovered" and (
            user_intent or not discovery_reason.strip() or not source_refs
        ):
            raise InvalidArgumentError(
                "A discovered Focus requires evidence and cannot invent user intent"
            )
        identity = str(uuid5(NAMESPACE_URL, memory_root(self.ctx) + "focus:" + create_key))
        creation_hash = self._hash(
            json.dumps([name.strip(), origin, user_intent], ensure_ascii=False)
        )
        async with self._write_guard():
            try:
                existing = await self._get(identity)
                if existing.extra_fields.get("creationHash") != creation_hash:
                    raise AlreadyExistsError(
                        "Creation key already belongs to another Focus request"
                    )
                return self.view(existing)
            except NotFoundError:
                pass
            # Discovery must respect archived identities too.
            if origin == "discovered":
                cursor = None
                matches = []
                while True:
                    page = await self._list(cursor, 100)
                    matches.extend(
                        f
                        for f in page["focuses"]
                        if name.strip().casefold()
                        in [n.casefold() for n in [f["name"], *(f["aliases"] or [])]]
                    )
                    cursor = page["nextCursor"]
                    if not cursor:
                        break
                if len(matches) > 1:
                    raise InvalidArgumentError(
                        "Ambiguous Focus name; read and choose an existing identity"
                    )
                if matches:
                    return self.view(await self._get(matches[0]["focusId"]))
            page = MemoryFile(
                uri=focus_uri(self.ctx, identity),
                memory_type="focuses",
                content=f"# {name.strip()}\n",
                extra_fields={
                    "focusId": identity,
                    "name": name.strip(),
                    "aliases": [],
                    "userIntent": user_intent,
                    "origin": origin,
                    "status": "active" if origin == "user" else "forming",
                    "summary": "",
                    "discoveryReason": discovery_reason,
                    "sourceRefs": sorted(set(source_refs)),
                    "protectedFields": ["name", "userIntent", "status"] if origin == "user" else [],
                    "creationHash": creation_hash,
                    "revision": 0,
                    "createdAt": now_iso(),
                },
            )
            await self._save(page)
            return self.view(page)

    async def update(
        self,
        identifier,
        *,
        expected_revision,
        fields,
        actor="user",
        patch_content=None,
        metadata=None,
        operation_id=None,
        links=(),
        backlinks=(),
    ):
        allowed = {
            "name",
            "aliases",
            "userIntent",
            "status",
            "summary",
            "content",
            "discoveryReason",
        }
        if actor not in ("user", "model") or set(fields) - allowed:
            raise InvalidArgumentError("Focus identity, origin and provenance are protected")
        self._validate(fields)
        request_hash = self._hash(json.dumps(fields, sort_keys=True, ensure_ascii=False))
        async with self._write_guard():
            page = await self._get(identifier)
            if operation_id and page.extra_fields.get("lastOperationId") == operation_id:
                if page.extra_fields.get("lastOperationHash") != request_hash:
                    raise AlreadyExistsError("Operation ID was reused with different fields")
                return self.view(page)
            if page.extra_fields["revision"] != expected_revision:
                raise AlreadyExistsError("Focus changed; read the latest revision before editing")
            protected = set(page.extra_fields.get("protectedFields", []))
            if actor == "model":
                if any(
                    key in fields and fields[key] != page.extra_fields.get(key)
                    for key in protected | {"userIntent"}
                ):
                    raise InvalidArgumentError("Model cannot override the user's Focus choices")
                if "status" in fields and fields["status"] != page.extra_fields["status"]:
                    # Native extraction can only activate an unprotected forming
                    # discovery. Archiving/restoring an established Focus belongs
                    # to the user; new evidence may still extend its narrative.
                    if (
                        fields["status"] != "active"
                        or page.extra_fields["status"] != "forming"
                        or not fields.get("summary", page.extra_fields.get("summary"))
                        or not page.extra_fields.get("sourceRefs")
                    ):
                        raise InvalidArgumentError(
                            "Model may only activate a grounded forming Focus"
                        )
            else:
                if fields.get("status") == "forming":
                    raise InvalidArgumentError("Forming is an internal discovery state")
                protected.update(set(fields) & {"name", "userIntent", "status"})
            if "name" in fields:
                lines = page.content.splitlines()
                if lines and lines[0] == "# " + page.extra_fields["name"]:
                    lines[0] = "# " + fields["name"]
                    page.content = "\n".join(lines)
            if "content" in fields:
                page.content = fields["content"]
            if patch_content:
                page.content = patch_content(page.plain_content())
            from openviking.session.memory.merge_op.link_merge import merge_links

            page.links = merge_links(page.links, [link.model_dump() for link in links])
            page.backlinks = merge_links(page.backlinks, [link.model_dump() for link in backlinks])
            page.extra_fields.update({k: v for k, v in fields.items() if k != "content"})
            page.extra_fields["protectedFields"] = sorted(protected)
            page.extra_fields.update(
                {
                    k: v
                    for k, v in (metadata or {}).items()
                    if k.startswith(("memory_update_", "email_", "meeting_"))
                }
            )
            if operation_id:
                page.extra_fields.update(
                    lastOperationId=operation_id, lastOperationHash=request_hash
                )
            await self._save(page)
            return self.view(page)

    async def apply_native(self, operation, schema):
        from openviking.session.memory.merge_op import MergeOpFactory

        old = operation.old_memory_file_content
        if old is None or "revision" not in old.extra_fields:
            raise InvalidArgumentError("Read the complete Focus before updating it")
        fields = operation.memory_fields
        if (
            operation.uris != [focus_uri(self.ctx, old.extra_fields["focusId"])]
            or fields.get("focusId") != old.extra_fields["focusId"]
        ):
            raise InvalidArgumentError("Focus identity does not match its target")
        content = next(f for f in schema.fields if f.name == "content")
        patch = (
            (lambda value: MergeOpFactory.from_field(content).apply(value, fields["content"]))
            if "content" in fields
            else None
        )
        editable = {
            f.name: MergeOpFactory.from_field(f).apply(old.extra_fields.get(f.name), fields[f.name])
            for f in schema.fields
            if f.name in ("name", "summary", "status") and fields.get(f.name)
        }
        await self.update(
            old.extra_fields["focusId"],
            expected_revision=old.extra_fields["revision"],
            fields=editable,
            actor="model",
            patch_content=patch,
            metadata=fields,
            links=getattr(operation, "_incoming_links_by_uri", {}).get(old.uri, []),
            backlinks=getattr(operation, "_incoming_backlinks_by_uri", {}).get(old.uri, []),
        )

    async def add_links(self, identifier, links, backlinks):
        page = await self.get(identifier)
        await self.update(
            identifier,
            expected_revision=page.extra_fields["revision"],
            fields={},
            actor="model",
            links=links,
            backlinks=backlinks,
        )

    async def _save(self, page):
        from openviking.session.memory.utils.resource_refs import (
            RESOURCE_REF_SOURCE_SESSION_COMMIT,
            sync_memory_resource_refs,
        )

        page.content = page.content.strip()
        sync_memory_resource_refs(page, source=RESOURCE_REF_SOURCE_SESSION_COMMIT)
        page.extra_fields["revision"] += 1
        page.extra_fields["updatedAt"] = now_iso()
        identifier = page.extra_fields["focusId"]
        narrative = MemoryFile(
            uri=page.uri,
            memory_type="focuses",
            content=page.content.strip(),
            links=page.links,
            backlinks=page.backlinks,
        )
        previous = await self._read_or_none(self._journal_uri(identifier))
        pending = {
            "focusId": identifier,
            "files": [],
            "indexPending": self.db is not None
            or bool(previous and json.loads(previous).get("indexPending")),
        }
        for uri, after in {
            focus_uri(self.ctx, identifier): MemoryFileUtils.write(narrative),
            focus_metadata_uri(self.ctx, identifier): json.dumps(
                page.extra_fields, ensure_ascii=False, indent=2
            ),
        }.items():
            pending["files"].append(
                {
                    "uri": uri,
                    "beforeHash": self._hash(await self._read_or_none(uri)),
                    "after": after,
                    "afterHash": self._hash(after),
                }
            )
        await self.fs.write_file(
            self._journal_uri(identifier), json.dumps(pending, ensure_ascii=False), ctx=self.ctx
        )
        await self._recover(identifier)

    def _journal_uri(self, identifier):
        return focus_directory_uri(self.ctx, identifier) + "/.pending-write.json"

    @staticmethod
    def _hash(value):
        return None if value is None else hashlib.sha256(value.encode()).hexdigest()

    async def _read_or_none(self, uri):
        try:
            return await self.fs.read_file(uri, ctx=self.ctx)
        except NotFoundError:
            return None

    async def _recover(self, identifier):
        raw = await self._read_or_none(self._journal_uri(identifier))
        if raw is None:
            return
        pending = json.loads(raw)
        expected = {focus_uri(self.ctx, identifier), focus_metadata_uri(self.ctx, identifier)}
        if pending.get("focusId") != identifier or {f["uri"] for f in pending["files"]} != expected:
            raise InvalidArgumentError("Invalid Focus write journal")
        # Validate the whole write before repairing either file.
        for file in pending["files"]:
            current = await self._read_or_none(file["uri"])
            if self._hash(file["after"]) != file["afterHash"] or self._hash(current) not in (
                file["beforeHash"],
                file["afterHash"],
            ):
                raise AlreadyExistsError("Focus changed during interrupted write")
        for file in pending["files"]:
            await self.fs.write_file(file["uri"], file["after"], ctx=self.ctx)
        if pending["indexPending"]:
            if self.db is None:
                return
            from openviking.session.memory.memory_updater import MemoryUpdater

            indexed = await MemoryUpdater.refresh_file_embedding(
                viking_fs=self.fs,
                vikingdb=self.db,
                ctx=self.ctx,
                uri=focus_uri(self.ctx, identifier),
                memory_type="focuses",
            )
            if not indexed:
                raise RuntimeError("Focus persisted; indexing awaits recovery")
        await self.fs.rm(self._journal_uri(identifier), ctx=self.ctx, lock_handle=self.lock_handle)
