"""One authoritative Profile with protected user edits and recoverable writes.

The Markdown body is a deterministic view. The same file's MEMORY_FIELDS holds
stable blocks and revision history; history is never part of semantic retrieval.
Owner locking has the same single-process boundary as ProjectStore.
"""

import copy
import hashlib
import json
import re
from uuid import uuid4

from openviking.session.memory.question_store import memory_root, now_iso, owner_lock
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError


def profile_uri(ctx):
    return memory_root(ctx) + "profile.md"


def managed_profile_uri(uri, ctx):
    return uri in (profile_uri(ctx), memory_root(ctx) + ".profile-pending.json")


def protected_profile_path(uri, ctx):
    """Reject replacing/deleting the document, its journal, or their parent tree."""
    path = uri.rstrip("/")
    return any(
        target == path or target.startswith(path + "/")
        for target in (profile_uri(ctx), memory_root(ctx) + ".profile-pending.json")
    )


def split_blocks(content):
    """Stable section IDs are matched by exact title, never inferred by a model."""
    if "<!-- MEMORY_FIELDS" in content:
        raise InvalidArgumentError("Profile content cannot contain storage metadata")
    parts = re.split(r"(?m)^## (.+)\n", content.strip())
    result = []
    if parts[0].strip():
        result.append(("About me", parts[0].strip()))
    result.extend((parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2))
    if len({title for title, _ in result}) != len(result):
        raise InvalidArgumentError("Profile section titles must be unique")
    return result


class ProfileStore:
    def __init__(self, fs, ctx, db=None):
        self.fs, self.ctx, self.db = fs, ctx, db
        self.uri = profile_uri(ctx)
        self.journal = memory_root(ctx) + ".profile-pending.json"

    async def _raw(self, uri):
        try:
            return await self.fs.read_file(uri, ctx=self.ctx)
        except NotFoundError:
            return None

    async def _get(self):
        await self._recover()
        raw = await self._raw(self.uri)
        page = MemoryFileUtils.read(raw or "", uri=self.uri)
        data = page.extra_fields.get("profileDocument")
        if data is None:
            data = {
                "version": 1,
                "documentRevision": 0,
                "identityRevision": 0,
                "identity": None,
                "blocks": [
                    {
                        "id": str(uuid4()),
                        "title": "Identity notes" if title == "Identity" else title,
                        "content": content,
                        "authority": "import",
                        "deleted": False,
                    }
                    for title, content in split_blocks(page.plain_content())
                ],
                "history": [],
                "operations": {},
            }
            page.memory_type = "profile"
            page.extra_fields["profileDocument"] = data
            await self._save(page)
        return page

    @staticmethod
    def identity_text(data):
        """The identity as the page shows it, under the revision it was sent with."""
        return json.dumps(
            {"revision": data["identityRevision"], **data["identity"]},
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def render(data):
        sections = [
            f"## {b['title']}\n{b['content']}" for b in data["blocks"] if not b.get("deleted")
        ]
        if data["identity"] is not None:
            sections.insert(0, "## Identity\n" + ProfileStore.identity_text(data))
        return "\n\n".join(sections).strip()

    @staticmethod
    def view(page):
        d = page.extra_fields["profileDocument"]
        return {
            "documentRevision": d["documentRevision"],
            "identityRevision": d["identityRevision"],
            "identity": copy.deepcopy(d["identity"]),
            "blocks": copy.deepcopy(d["blocks"]),
            "content": page.content,
        }

    async def read(self):
        async with owner_lock(self.ctx):
            return await self._get()

    async def get(self):
        async with owner_lock(self.ctx):
            return self.view(await self._get())

    async def history(self, block_id=None):
        async with owner_lock(self.ctx):
            d = (await self._get()).extra_fields["profileDocument"]
            return copy.deepcopy(
                [h for h in d["history"] if block_id is None or h["blockId"] == block_id]
            )

    @staticmethod
    def _replay(d, operation_id, payload):
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        old = d["operations"].get(operation_id)
        if old and old != digest:
            raise AlreadyExistsError("Profile operation ID was reused with different input")
        return old is not None, digest

    async def sync_identity(self, revision, identity):
        """Take the identity the application holds at `revision`.

        The application's database is the authority: a later revision replaces
        what is here, an earlier one is answered with what is kept.
        """
        async with owner_lock(self.ctx):
            page = await self._get()
            d = page.extra_fields["profileDocument"]
            held = d["identityRevision"]
            if revision < held:
                return {"status": "stale", "identityRevision": held}
            if revision == held and d["identity"] is not None:
                if d["identity"] != identity:
                    raise AlreadyExistsError(
                        "Identity revision already belongs to a different snapshot"
                    )
                return {"status": "unchanged", "identityRevision": held}
            before = copy.deepcopy(d["identity"])
            d["identity"] = copy.deepcopy(identity)
            d["identityRevision"] = revision
            self._record(d, str(uuid4()), "identity", before, identity, "application", None)
            await self._save(page)
            return {"status": "synced", "identityRevision": revision}

    @staticmethod
    def _record(d, edit_id, block_id, before, after, actor, reason):
        d["history"].append(
            {
                "editId": edit_id,
                "blockId": block_id,
                "baseRevision": d["documentRevision"],
                "before": copy.deepcopy(before),
                "after": copy.deepcopy(after),
                "actor": actor,
                "createdAt": now_iso(),
                "reason": reason,
                "reviewStatus": "pending" if actor == "user" else "notRequired",
            }
        )

    async def edit(
        self, operation_id, expected_revision, block_id, title, content, reason=None, action="edit"
    ):
        title = title.strip()
        async with owner_lock(self.ctx):
            page = await self._get()
            d = page.extra_fields["profileDocument"]
            replay, digest = self._replay(
                d, operation_id, [expected_revision, block_id, title, content, reason, action]
            )
            if replay:
                return self.view(page)
            if d["documentRevision"] != expected_revision:
                raise AlreadyExistsError("Profile changed; read the current version before saving")
            block = next((b for b in d["blocks"] if b["id"] == block_id), None)
            if block_id and block is None:
                raise NotFoundError("Profile section not found")
            if action == "unlock" and block is None:
                raise InvalidArgumentError("Select an existing section to unlock")
            if (
                title.strip() == "Identity"
                or not title.strip()
                or "\n" in title
                or re.search(r"(?m)^## ", content)
                or "<!-- MEMORY_FIELDS" in content
                or "<!-- MEMORY_FIELDS" in title
            ):
                raise InvalidArgumentError("Edit one Profile section at a time")
            if any(b["title"] == title and b is not block for b in d["blocks"]):
                raise InvalidArgumentError("Profile section title already exists")
            before = copy.deepcopy(block)
            if block is None:
                block = {"id": str(uuid4())}
                d["blocks"].append(block)
            if action == "unlock":
                block["authority"] = "learned"
            else:
                block.update(
                    title=title.strip(),
                    content=content.strip(),
                    authority="user",
                    deleted=action == "delete",
                )
            block["updatedAt"] = now_iso()
            self._record(d, operation_id, block["id"], before, block, "user", reason)
            d["operations"][operation_id] = digest
            await self._save(page)
            return self.view(page)

    async def review_receipt(self, edit_id):
        async with owner_lock(self.ctx):
            page = await self._get()
            record = next(
                (
                    h
                    for h in page.extra_fields["profileDocument"]["history"]
                    if h["editId"] == edit_id
                ),
                None,
            )
            if record is None:
                raise NotFoundError("Profile revision not found")
            record["reviewStatus"] = "submitted"
            await self._save(page, bump=False)

    async def apply_native(self, operation, schema):
        from openviking.session.memory.merge_op import MergeOpFactory

        if operation.uris != [self.uri]:
            raise InvalidArgumentError("Profile update must target the current user's profile")
        old = operation.old_memory_file_content
        if old is None:
            # A first native write still has to observe absence rather than replace
            # a file that appeared after extraction started.
            old_text = ""
        else:
            old_text = old.plain_content()
        async with owner_lock(self.ctx):
            page = await self._get()
            d = page.extra_fields["profileDocument"]
            old_d = old.extra_fields.get("profileDocument") if old else None
            if (
                old_d and old_d["documentRevision"] != d["documentRevision"]
            ) or old_text != page.plain_content():
                raise AlreadyExistsError(
                    "Profile changed during extraction; reread before updating"
                )
            field = next(f for f in schema.fields if f.name == "content")
            new = MergeOpFactory.from_field(field).apply(
                page.plain_content(), operation.memory_fields.get("content")
            )
            incoming = dict(split_blocks(new))
            if d["identity"] is not None:
                if incoming.pop("Identity", None) != self.identity_text(d):
                    raise InvalidArgumentError(
                        "Profile identity is set by the application, not by a model"
                    )
            elif "Identity" in incoming:
                raise InvalidArgumentError("A model cannot create verified Profile identity")
            changed = self._apply_sections(d, incoming, "model")
            links_changed = self._merge_links(
                page,
                getattr(operation, "_incoming_links_by_uri", {}).get(self.uri, []),
                getattr(operation, "_incoming_backlinks_by_uri", {}).get(self.uri, []),
            )
            if changed or links_changed:
                await self._save(page, bump=changed)

    async def write_body(self, expected_revision, content):
        """Replace the sections below the identity with an agent's version.

        The sections stay open to later learning. A section the user edited
        by hand is theirs: it must come back as it is.
        """
        async with owner_lock(self.ctx):
            page = await self._get()
            d = page.extra_fields["profileDocument"]
            if d["documentRevision"] != expected_revision:
                raise AlreadyExistsError("Profile changed; read the current version before saving")
            incoming = dict(split_blocks(content))
            if "Identity" in incoming:
                raise InvalidArgumentError("Profile identity is set by the application")
            if self._apply_sections(d, incoming, "agent"):
                await self._save(page)
            return self.view(page)

    def _apply_sections(self, d, incoming, actor):
        """Make the learned sections match `incoming`; True when any changed."""
        from openviking.session.memory.utils.link_renderer import LinkRenderer

        for b in d["blocks"]:
            if b["authority"] == "user":
                if b.get("deleted") and b["title"] in incoming:
                    raise InvalidArgumentError("A user-deleted Profile section cannot be restored")
                if not b.get("deleted") and incoming.get(b["title"]) != LinkRenderer.strip_links(
                    b["content"]
                ):
                    raise InvalidArgumentError("User-edited Profile sections are protected")
        changed = []
        for b in d["blocks"]:
            if b["authority"] == "user":
                incoming.pop(b["title"], None)
                continue
            before = copy.deepcopy(b)
            content = incoming.pop(b["title"], None)
            b.update(
                content=content if content is not None else b["content"],
                deleted=content is None,
                authority="learned",
            )
            if before != b:
                changed.append((before, copy.deepcopy(b)))
        for title, content in incoming.items():
            b = {
                "id": str(uuid4()),
                "title": title,
                "content": content,
                "authority": "learned",
                "deleted": False,
            }
            d["blocks"].append(b)
            changed.append((None, b))
        for before, after in changed:
            self._record(d, str(uuid4()), after["id"], before, after, actor, None)
        return bool(changed)

    @staticmethod
    def _merge_links(page, links, backlinks):
        from openviking.session.memory.merge_op.link_merge import merge_links

        before = copy.deepcopy((page.links, page.backlinks))
        page.links = merge_links(page.links, [link.model_dump() for link in links])
        page.backlinks = merge_links(page.backlinks, [link.model_dump() for link in backlinks])
        return before != (page.links, page.backlinks)

    async def add_links(self, links, backlinks):
        # Graph metadata must not overwrite a concurrent user edit or replace
        # its history. Merge against the latest file under the same owner lock.
        async with owner_lock(self.ctx):
            page = await self._get()
            if self._merge_links(page, links, backlinks):
                await self._save(page, bump=False)

    async def _save(self, page, bump=True):
        d = page.extra_fields["profileDocument"]
        if bump:
            d["documentRevision"] += 1
        page.content = self.render(d)
        page.extra_fields["profileRevision"] = d["documentRevision"]
        # Synchronize references under the same lock without rewriting the user
        # text: the canonical body must remain the deterministic block rendering.
        from openviking.session.memory.utils.resource_refs import sync_memory_resource_refs

        reference_view = copy.deepcopy(page)
        sync_memory_resource_refs(reference_view, source="profile.update")
        if "resource_refs" in reference_view.extra_fields:
            page.extra_fields["resource_refs"] = reference_view.extra_fields["resource_refs"]
        else:
            page.extra_fields.pop("resource_refs", None)
        raw = MemoryFileUtils.write(page)
        before = await self._raw(self.uri)
        previous_journal = await self._raw(self.journal)
        pending_index = bool(previous_journal and json.loads(previous_journal)["indexPending"])
        journal = {
            "beforeHash": self._hash(before),
            "after": raw,
            "afterHash": self._hash(raw),
            # A metadata-only writer may not have the vector DB dependency. It
            # must carry forward an earlier failed index enqueue, not erase it.
            "indexPending": pending_index
            or (self.db is not None and bool(page.content.strip()) and bump),
        }
        await self.fs.write_file(
            self.journal, json.dumps(journal, ensure_ascii=False), ctx=self.ctx
        )
        await self._recover()

    @staticmethod
    def _hash(value):
        return None if value is None else hashlib.sha256(value.encode()).hexdigest()

    async def _recover(self):
        raw = await self._raw(self.journal)
        if raw is None:
            return
        pending = json.loads(raw)
        current = await self._raw(self.uri)
        if self._hash(pending["after"]) != pending["afterHash"] or self._hash(current) not in (
            pending["beforeHash"],
            pending["afterHash"],
        ):
            raise AlreadyExistsError("Profile changed during interrupted write")
        if current != pending["after"]:
            await self.fs.write_file(self.uri, pending["after"], ctx=self.ctx)
        if (
            pending["indexPending"]
            and MemoryFileUtils.read(pending["after"], uri=self.uri).content.strip()
        ):
            if self.db is None:
                return
            from openviking.session.memory.memory_updater import MemoryUpdater

            if not await MemoryUpdater.refresh_file_embedding(
                viking_fs=self.fs,
                vikingdb=self.db,
                uri=self.uri,
                memory_type="profile",
                ctx=self.ctx,
            ):
                raise RuntimeError("Profile index update awaits recovery")
        await self.fs.rm(self.journal, ctx=self.ctx)
