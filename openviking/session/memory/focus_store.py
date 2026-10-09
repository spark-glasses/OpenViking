"""Personal priorities in OV: two files per Focus and one writer of both.

``focus.json`` is the record of a Focus and ``memory.md`` its narrative. Every
write of either goes through this store, whether it arrives as a file write
from the user's side (the user, or an agent acting for them) or from memory
extraction. The two sides have different rights:

- The user's side names a Focus, states what it means to them, writes its
  summary and narrative, and moves it between active and archived.
- Extraction may discover a Focus, which starts as forming: its draft. It may
  rename only a forming Focus, may only guess an intent the user has not
  stated, and may only move a forming Focus to active. It writes the summary
  and narrative of any Focus.

A Focus is never deleted; the user archives it.
"""

import hashlib
import json
from contextlib import asynccontextmanager
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.focus_paths import (
    FOCUS_RECORD_FILE,
    focus_directory_uri,
    focus_metadata_uri,
    focus_uri,
)
from openviking.session.memory.question_store import memory_root, now_iso, owner_lock
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import (
    AlreadyExistsError,
    ConflictError,
    InvalidArgumentError,
    NotFoundError,
)

# What a record holds, in the order it is shown.
RECORD_FIELDS = (
    "focusId",
    "name",
    "intent",
    "intentSource",
    "origin",
    "status",
    "summary",
    "sourceRefs",
    "revision",
    "createdAt",
    "updatedAt",
)
# The fields of a record that a file write from the user's side may change.
USER_FIELDS = ("name", "intent", "status", "summary")
STATUSES = ("forming", "active", "archived")


class FocusStore:
    def __init__(self, fs, ctx, db=None, lock_handle=None):
        self.fs, self.ctx, self.db = fs, ctx, db
        self.lock_handle = lock_handle

    @asynccontextmanager
    async def _write_guard(self):
        # Native extraction already owns the directory lock. Other writers
        # acquire the same lock before mutating either file, so a busy model
        # never leaves a half-applied change. Isolated in-memory test FS has no
        # AGFS locks.
        if self.lock_handle is not None or not hasattr(self.fs, "_uri_to_path"):
            async with owner_lock(self.ctx):
                yield
            return
        from openviking.storage.errors import LockAcquisitionError
        from openviking.storage.transaction import LockContext, get_lock_manager

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
            raise ConflictError(
                "resource is busy and cannot be read or written now: "
                "this user's Focuses are being updated"
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
        directory = page.uri.rsplit("/", 1)[0]
        return (
            {key: fields.get(key) for key in RECORD_FIELDS}
            | {
                "uri": page.uri,
                "directoryUri": directory,
                "metadataUri": directory + "/" + FOCUS_RECORD_FILE,
                "questionsUri": directory + "/questions.md",
            }
            | ({"content": page.content} if include_content else {})
        )

    async def list(self, after=None, limit=50, status="all"):
        async with self._write_guard():
            return await self._list(after, limit, status)

    async def _identifiers(self):
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
                identifiers.add(str(UUID(stem)))
            except ValueError:
                continue
        return sorted(identifiers)

    async def _list(self, after=None, limit=50, status="all"):
        if status not in ("all", *STATUSES):
            raise InvalidArgumentError("Invalid Focus status filter")
        if after:
            after = str(UUID(after))
        limit = max(1, min(100, limit))
        pages = []
        for identifier in await self._identifiers():
            if after and identifier <= after:
                continue
            page = self.view(await self._get(identifier), include_content=False)
            if status == "all" or page["status"] == status:
                pages.append(page)
            if len(pages) > limit:
                break
        return {
            "focuses": pages[:limit],
            "nextCursor": pages[limit - 1]["focusId"] if len(pages) > limit else None,
        }

    async def _named(self, name):
        """Every Focus that goes by this name, whatever its status."""
        wanted = name.strip().casefold()
        found = []
        for identifier in await self._identifiers():
            page = await self._get(identifier)
            if page.extra_fields.get("name", "").casefold() == wanted:
                found.append(page)
        return found

    @staticmethod
    def _validate(fields):
        for key, maximum in (
            ("name", 200),
            ("intent", 4000),
            ("summary", 600),
            ("content", 200000),
        ):
            if key in fields and (
                not isinstance(fields[key], str)
                or len(fields[key]) > maximum
                or (key == "name" and not fields[key].strip())
            ):
                raise InvalidArgumentError(f"Invalid Focus {key}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise InvalidArgumentError("Invalid Focus status")

    async def ensure(self, *, name, origin="discovered", intent="", source_refs=(), create_key=None):
        """Create a Focus, or return the one extraction already discovered.

        The user's side creates an active Focus. A forming Focus of the same
        name becomes theirs; any other Focus of that name is a refusal.
        Extraction creates a forming one from evidence, with its guess at why
        the Focus matters, and gets back any existing Focus of the same name.
        """
        self._validate({"name": name, "intent": intent})
        if origin not in ("user", "discovered"):
            raise InvalidArgumentError("Invalid Focus creation")
        if origin == "discovered" and (
            not intent.strip()
            or not source_refs
            or not isinstance(create_key, str)
            or not 1 <= len(create_key) <= 500
        ):
            raise InvalidArgumentError(
                "A discovered Focus needs evidence and a grounded guess at why it matters"
            )
        async with self._write_guard():
            same = await self._named(name)
            if origin == "user":
                if same and same[0].extra_fields["status"] != "forming":
                    raise AlreadyExistsError(
                        f"{same[0].extra_fields['name']} at "
                        + same[0].uri.rsplit("/", 1)[0]
                        + "/"
                        + FOCUS_RECORD_FILE,
                        "focus",
                    )
                if same:
                    fields = {"status": "active"} | ({"intent": intent} if intent.strip() else {})
                    return self.view(await self._change(same[0], fields, actor="user"))
                identity = str(uuid4())
            else:
                if len(same) > 1:
                    raise InvalidArgumentError(
                        "Ambiguous Focus name; read and choose an existing identity"
                    )
                if same:
                    return self.view(same[0])
                identity = str(uuid5(NAMESPACE_URL, memory_root(self.ctx) + "focus:" + create_key))
                try:
                    return self.view(await self._get(identity))
                except NotFoundError:
                    pass
            source = "user" if origin == "user" else "guessed"
            page = MemoryFile(
                uri=focus_uri(self.ctx, identity),
                memory_type="focuses",
                content=f"# {name.strip()}\n",
                extra_fields={
                    "focusId": identity,
                    "name": name.strip(),
                    "intent": intent,
                    "intentSource": source if intent.strip() else None,
                    "origin": origin,
                    "status": "active" if origin == "user" else "forming",
                    "summary": "",
                    "sourceRefs": sorted(set(source_refs)),
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
        fields,
        actor="user",
        expected_revision=None,
        patch_content=None,
        metadata=None,
        links=(),
        backlinks=(),
    ):
        async with self._write_guard():
            page = await self._get(identifier)
            return self.view(
                await self._change(
                    page,
                    fields,
                    actor=actor,
                    expected_revision=expected_revision,
                    patch_content=patch_content,
                    metadata=metadata,
                    links=links,
                    backlinks=backlinks,
                )
            )

    async def _change(
        self,
        page,
        fields,
        *,
        actor,
        expected_revision=None,
        patch_content=None,
        metadata=None,
        links=(),
        backlinks=(),
    ):
        if actor not in ("user", "model") or set(fields) - {*USER_FIELDS, "content"}:
            raise InvalidArgumentError("Focus identity, origin and provenance are kept by the system")
        self._validate(fields)
        record = page.extra_fields
        if expected_revision is not None and record["revision"] != expected_revision:
            raise ConflictError("Focus changed; read the latest revision before editing")
        changed = {
            key: value
            for key, value in fields.items()
            if key != "content" and value != record.get(key)
        }
        if actor == "model":
            if "name" in changed and record["status"] != "forming":
                raise InvalidArgumentError("Extraction may rename only a forming Focus")
            if "intent" in changed and record.get("intentSource") == "user":
                raise InvalidArgumentError(
                    "The user stated this intent; extraction cannot replace it"
                )
            if "status" in changed and (
                changed["status"] != "active"
                or record["status"] != "forming"
                or not fields.get("summary", record.get("summary"))
                or not record.get("sourceRefs")
            ):
                # Archiving and restoring belong to the user; new evidence may
                # still extend the narrative of a Focus in any state.
                raise InvalidArgumentError("Extraction may only activate a grounded forming Focus")
        else:
            if changed.get("status") == "forming":
                raise InvalidArgumentError("Forming is the draft state of a discovered Focus")
            # A forming Focus is extraction's draft. Naming it or saying what
            # it means makes it the user's.
            if (
                record["status"] == "forming"
                and "status" not in changed
                and {"name", "intent"} & set(changed)
            ):
                changed["status"] = "active"
        if "intent" in changed:
            source = "user" if actor == "user" else "guessed"
            changed["intentSource"] = source if changed["intent"].strip() else None
        if "name" in changed:
            lines = page.content.splitlines()
            if lines and lines[0] == "# " + record["name"]:
                lines[0] = "# " + changed["name"]
                page.content = "\n".join(lines)
        if "content" in fields:
            page.content = fields["content"]
        if patch_content:
            page.content = patch_content(page.plain_content())
        from openviking.session.memory.merge_op.link_merge import merge_links

        page.links = merge_links(page.links, [link.model_dump() for link in links])
        page.backlinks = merge_links(page.backlinks, [link.model_dump() for link in backlinks])
        record.update(changed)
        record.update(
            {
                k: v
                for k, v in (metadata or {}).items()
                if k.startswith(("memory_update_", "email_", "meeting_"))
            }
        )
        await self._save(page)
        return page

    async def write_file(self, folder, name, content, mode):
        """A file write of a Focus's record or page, from the user's side.

        Returns the URI written. Creating the record of a Focus that does not
        exist creates the Focus; its folder gets a new stable name, so the
        returned URI is where it actually lives.
        """
        try:
            page = await self.get(folder)
        except (ValueError, NotFoundError):
            page = None
        if name == FOCUS_RECORD_FILE:
            if page is None:
                if mode != "create":
                    raise NotFoundError(folder, "focus")
                return (await self.create_record(content))["metadataUri"]
            if mode == "create":
                raise AlreadyExistsError(page.extra_fields["name"], "focus")
            if mode == "append":
                raise InvalidArgumentError(
                    "focus.json is one JSON object: replace it, or change a value in it"
                )
            return (await self.write_record(folder, content))["metadataUri"]
        if page is None:
            raise NotFoundError(folder, "focus")
        if mode == "create":
            raise AlreadyExistsError(page.uri, "file")
        return (await self.write_page(folder, content, mode))["uri"]

    @staticmethod
    def _submitted_record(text):
        try:
            submitted = json.loads(text)
        except ValueError as error:
            raise InvalidArgumentError(f"focus.json must be one JSON object: {error}") from error
        if not isinstance(submitted, dict):
            raise InvalidArgumentError("focus.json must be one JSON object")
        return submitted

    async def create_record(self, text):
        """A new Focus from the record the user's side wrote for it."""
        submitted = self._submitted_record(text)
        if set(submitted) - {*USER_FIELDS} or submitted.get("status", "active") != "active":
            raise InvalidArgumentError(
                "A new Focus takes name, and optionally intent and summary; it starts active"
            )
        if not isinstance(submitted.get("name"), str):
            raise InvalidArgumentError("A new Focus needs a name")
        self._validate(submitted)
        created = await self.ensure(
            name=submitted["name"], origin="user", intent=submitted.get("intent", "")
        )
        if submitted.get("summary"):
            return await self.update(created["focusId"], fields={"summary": submitted["summary"]})
        return created

    async def write_record(self, identifier, text):
        """Replace a Focus's record with what the user's side wrote.

        The submitted record is compared with the stored one: only the fields
        the user's side owns may differ, and its revision must be the stored
        one, so a record read before another change is not written over it.
        """
        submitted = self._submitted_record(text)
        async with self._write_guard():
            page = await self._get(identifier)
            record = page.extra_fields
            unknown = sorted(set(submitted) - set(record))
            if unknown:
                raise InvalidArgumentError("focus.json has no field " + ", ".join(unknown))
            different = {key: value for key, value in submitted.items() if value != record[key]}
            if "revision" in different:
                raise ConflictError("Focus changed since it was read; read focus.json again")
            kept = sorted(set(different) - set(USER_FIELDS))
            if kept:
                raise InvalidArgumentError(
                    ", ".join(kept)
                    + " is kept by the system; change only name, intent, status or summary"
                )
            if not different:
                return self.view(page)
            return self.view(await self._change(page, different, actor="user"))

    async def write_page(self, identifier, text, mode="replace"):
        """Replace a Focus's narrative, or add to its end, from the user's side."""
        if mode not in ("replace", "append"):
            raise InvalidArgumentError("Invalid Focus page write")
        # A page read with its fields attached comes back as the narrative alone.
        content = MemoryFileUtils.read(text).content if "MEMORY_FIELDS" in text else text
        async with self._write_guard():
            page = await self._get(identifier)
            if mode == "append":
                # Added text starts its own block unless it brings its own spacing.
                content = page.content + ("" if content.startswith("\n") else "\n\n") + content
            return self.view(await self._change(page, {"content": content}, actor="user"))

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
            if f.name in USER_FIELDS and fields.get(f.name)
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
                raise ConflictError("Focus changed during interrupted write")
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
