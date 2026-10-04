"""OV is the sole authority for Project identity, source bindings and narrative.

The directory is enumerated for listing; semantic search is never a catalog.
All supported writes (UI and native extraction) share the owner lock and revision
check. The lock has the same single-workspace-process boundary as QuestionStore.
"""

import hashlib
import json
from uuid import NAMESPACE_URL, UUID, uuid5

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.project_paths import (
    project_directory_uri,
    project_metadata_uri,
    project_uri,
)
from openviking.session.memory.question_store import memory_root, now_iso, owner_lock
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError


class ProjectStore:
    def __init__(self, fs, ctx, db=None):
        self.fs, self.ctx, self.db = fs, ctx, db

    async def get(self, identifier):
        async with owner_lock(self.ctx):
            return await self._get(identifier)

    async def _get(self, identifier):
        await self._recover(identifier)
        uri = project_uri(self.ctx, identifier)
        page = MemoryFileUtils.read(await self.fs.read_file(uri, ctx=self.ctx), uri=uri)
        fields = json.loads(
            await self.fs.read_file(project_metadata_uri(self.ctx, identifier), ctx=self.ctx)
        )
        if fields.get("projectId") != str(UUID(str(identifier))):
            raise InvalidArgumentError("Project identity does not match its URI")
        # Metadata is authoritative in project.json. The MemoryFile is a read
        # projection for existing native extraction/revision checks, not a copy.
        page.extra_fields.update(fields)
        return page

    @staticmethod
    def view(page, include_content=True):
        fields = page.extra_fields
        return (
            {
                key: fields.get(key)
                for key in (
                    "projectId",
                    "name",
                    "aliases",
                    "status",
                    "revision",
                    "sourceBindings",
                    "createdAt",
                    "updatedAt",
                )
            }
            | {
                "uri": page.uri,
                "directoryUri": page.uri.rsplit("/", 1)[0],
                "metadataUri": page.uri.rsplit("/", 1)[0] + "/project.json",
                "questionsUri": page.uri.rsplit("/", 1)[0] + "/questions.md",
            }
            | ({"content": page.content} if include_content else {})
        )

    async def list(self, after=None, limit=50):
        async with owner_lock(self.ctx):
            return await self._list(after, limit)

    async def _list(self, after=None, limit=50):
        if after:
            after = str(UUID(after))
        root = memory_root(self.ctx) + "projects"
        try:
            # VikingFS defaults to 1,000 entries with silent truncation. Catalog
            # pagination must never mistake that truncated listing for the end.
            entries = await self.fs.ls(root, node_limit=10001, ctx=self.ctx)
            if len(entries) >= 10001:
                raise InvalidArgumentError("Project catalog exceeds the 10,000-entry listing limit")
        except NotFoundError:
            entries = []
        identifiers = set()
        for entry in entries:
            uri = entry.get("uri") or root + "/" + entry.get("name", "")
            stem = uri.removeprefix(root + "/").rstrip("/")
            try:
                identifier = str(UUID(stem))
            except ValueError:
                continue
            if uri.rstrip("/") == root + "/" + identifier and (not after or identifier > after):
                identifiers.add(identifier)
        ids = sorted(identifiers)
        limit = max(1, min(100, limit))
        pages = [
            self.view(await self._get(identifier), include_content=False)
            for identifier in ids[:limit]
        ]
        return {"projects": pages, "nextCursor": ids[limit - 1] if len(ids) > limit else None}

    async def ensure(self, *, name, create_key, source=None):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
            raise InvalidArgumentError("Project name must contain 1-200 characters")
        if source and (
            source.get("provider") not in ("slack", "linear")
            or not isinstance(source.get("workspaceId"), str)
            or not source["workspaceId"]
        ):
            raise InvalidArgumentError("Invalid Project source")
        async with owner_lock(self.ctx):
            if source:
                cursor = None
                while True:
                    listing = await self._list(cursor, 100)
                    for project in listing["projects"]:
                        if any(
                            binding.get("provider") == source["provider"]
                            and binding.get("workspaceId") == source["workspaceId"]
                            and not binding.get("resourceId")
                            for binding in project["sourceBindings"]
                        ):
                            return self.view(await self._get(project["projectId"]))
                    cursor = listing["nextCursor"]
                    if not cursor:
                        break
            key = f"source:{source['provider']}:{source['workspaceId']}" if source else create_key
            identifier = str(uuid5(NAMESPACE_URL, "spark-project:" + key))
            try:
                return self.view(await self._get(identifier))
            except NotFoundError:
                pass
            page = MemoryFile(
                uri=project_uri(self.ctx, identifier),
                memory_type="projects",
                content=f"# {name.strip()}\n",
                extra_fields={
                    "projectId": identifier,
                    "name": name.strip(),
                    "aliases": [],
                    "status": "active",
                    "revision": 0,
                    "createdAt": now_iso(),
                    "sourceBindings": [{**source, "confirmationStatus": "inferred"}]
                    if source
                    else [],
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
        patch_content=None,
        metadata=None,
        links=(),
        backlinks=(),
    ):
        async with owner_lock(self.ctx):
            page = await self._get(identifier)
            if page.extra_fields["revision"] != expected_revision:
                raise AlreadyExistsError(
                    "Project changed; read its current revision before editing"
                )
            if set(fields) - {"name", "aliases", "status", "content"}:
                raise InvalidArgumentError("Project identity and source bindings are protected")
            if "name" in fields and (
                not isinstance(fields["name"], str) or not 1 <= len(fields["name"].strip()) <= 200
            ):
                raise InvalidArgumentError("Invalid Project name")
            if "aliases" in fields and (
                not isinstance(fields["aliases"], list)
                or len(fields["aliases"]) > 50
                or any(
                    not isinstance(alias, str) or not 1 <= len(alias) <= 200
                    for alias in fields["aliases"]
                )
            ):
                raise InvalidArgumentError("Invalid Project aliases")
            if "status" in fields and fields["status"] not in (
                "active",
                "paused",
                "completed",
                "archived",
            ):
                raise InvalidArgumentError("Invalid Project status")
            if "content" in fields:
                if not isinstance(fields["content"], str) or len(fields["content"]) > 200000:
                    raise InvalidArgumentError("Invalid Project content")
                page.content = fields["content"]
            if patch_content:
                page.content = patch_content(page.plain_content())
            from openviking.session.memory.merge_op.link_merge import merge_links

            page.links = merge_links(page.links, [link.model_dump() for link in links])
            page.backlinks = merge_links(page.backlinks, [link.model_dump() for link in backlinks])
            page.extra_fields.update({k: v for k, v in fields.items() if k != "content"})
            page.extra_fields.update(
                {k: v for k, v in (metadata or {}).items() if k.startswith("memory_update_")}
            )
            await self._save(page)
            return self.view(page)

    async def apply_native(self, operation, schema):
        from openviking.session.memory.merge_op import MergeOpFactory

        old = operation.old_memory_file_content
        if old is None or "revision" not in old.extra_fields:
            raise InvalidArgumentError("Read the Project before updating it")
        fields = operation.memory_fields
        if (
            operation.uris != [project_uri(self.ctx, old.extra_fields["projectId"])]
            or fields.get("projectId") != old.extra_fields["projectId"]
        ):
            raise InvalidArgumentError("Project identity does not match its update target")
        content_field = next(field for field in schema.fields if field.name == "content")
        patch = (
            (
                lambda content: MergeOpFactory.from_field(content_field).apply(
                    content, fields["content"]
                )
            )
            if "content" in fields
            else None
        )
        # Native replace fields use None/"" to mean "leave unchanged". Apply the
        # schema's merge semantics before validating the final Project values;
        # treating an omitted rename as an empty new name breaks content updates.
        editable = {
            field.name: MergeOpFactory.from_field(field).apply(
                old.extra_fields.get(field.name), fields[field.name]
            )
            for field in schema.fields
            if field.name in ("name", "status") and field.name in fields
        }
        await self.update(
            old.extra_fields["projectId"],
            expected_revision=old.extra_fields["revision"],
            fields=editable,
            patch_content=patch,
            metadata=fields,
            links=getattr(operation, "_incoming_links_by_uri", {}).get(old.uri, []),
            backlinks=getattr(operation, "_incoming_backlinks_by_uri", {}).get(old.uri, []),
        )

    async def add_links(self, identifier, links, backlinks):
        from openviking.session.memory.merge_op.link_merge import merge_links

        async with owner_lock(self.ctx):
            page = await self._get(identifier)
            page.links = merge_links(page.links, [link.model_dump() for link in links])
            page.backlinks = merge_links(page.backlinks, [link.model_dump() for link in backlinks])
            await self._save(page)

    async def _save(self, page):
        from openviking.session.memory.utils.resource_refs import (
            RESOURCE_REF_SOURCE_SESSION_COMMIT,
            sync_memory_resource_refs,
        )

        page.content = page.content.strip()
        # Maintain citation metadata inside the same journaled, revision-checked
        # write. The generic updater must not rewrite this narrative afterwards.
        sync_memory_resource_refs(page, source=RESOURCE_REF_SOURCE_SESSION_COMMIT)
        page.extra_fields["revision"] += 1
        page.extra_fields["updatedAt"] = now_iso()
        identifier = page.extra_fields["projectId"]
        fields = dict(page.extra_fields)
        # Neither identity nor revision is duplicated in the narrative file.
        narrative = MemoryFile(
            uri=page.uri,
            memory_type="projects",
            content=page.content,
            links=page.links,
            backlinks=page.backlinks,
        )
        files = {
            project_uri(self.ctx, identifier): MemoryFileUtils.write(narrative),
            project_metadata_uri(self.ctx, identifier): json.dumps(
                fields, ensure_ascii=False, indent=2
            ),
        }
        previous = await self._journal(identifier)
        pending = {
            "projectId": identifier,
            "files": [],
            "indexPending": self.db is not None or bool(previous and previous.get("indexPending")),
        }
        for uri, after in files.items():
            before = await self._read_or_none(uri)
            pending["files"].append(
                {
                    "uri": uri,
                    "beforeHash": self._hash(before),
                    "after": after,
                    "afterHash": self._hash(after),
                }
            )
        await self.fs.write_file(
            self._journal_uri(identifier), json.dumps(pending, ensure_ascii=False), ctx=self.ctx
        )
        await self._recover(identifier)

    def _journal_uri(self, identifier):
        return project_directory_uri(self.ctx, identifier) + "/.pending-write.json"

    @staticmethod
    def _hash(value):
        return None if value is None else hashlib.sha256(value.encode()).hexdigest()

    async def _read_or_none(self, uri):
        try:
            return await self.fs.read_file(uri, ctx=self.ctx)
        except NotFoundError:
            return None

    async def _journal(self, identifier):
        raw = await self._read_or_none(self._journal_uri(identifier))
        return json.loads(raw) if raw is not None else None

    async def _recover(self, identifier):
        pending = await self._journal(identifier)
        if pending is None:
            return
        identifier = str(UUID(str(identifier)))
        expected = {project_uri(self.ctx, identifier), project_metadata_uri(self.ctx, identifier)}
        if (
            pending.get("projectId") != identifier
            or {f["uri"] for f in pending["files"]} != expected
        ):
            raise InvalidArgumentError("Invalid Project write journal")
        for file in pending["files"]:
            current = await self._read_or_none(file["uri"])
            if self._hash(file["after"]) != file["afterHash"] or self._hash(current) not in (
                file["beforeHash"],
                file["afterHash"],
            ):
                raise AlreadyExistsError(
                    "Project changed during interrupted write; refusing to overwrite"
                )
            if current != file["after"]:
                await self.fs.write_file(file["uri"], file["after"], ctx=self.ctx)
        # Keep the durable journal until the index update has been accepted.
        # A read-only Store may repair bytes; a Store with DB access finishes it.
        if pending.get("indexPending") and self.db is None:
            return
        if pending.get("indexPending"):
            from openviking.session.memory.memory_updater import MemoryUpdater

            indexed = await MemoryUpdater.refresh_file_embedding(
                viking_fs=self.fs,
                vikingdb=self.db,
                uri=project_uri(self.ctx, identifier),
                memory_type="projects",
                ctx=self.ctx,
            )
            if not indexed:
                raise RuntimeError("Project write persisted; index enqueue is pending recovery")
        await self.fs.rm(self._journal_uri(identifier), ctx=self.ctx)
