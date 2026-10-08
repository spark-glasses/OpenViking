"""Take a merged-away person out of memory once another person stands for them.

The application merges two people in its own records, has their memory merged,
and then retires the one that is gone. Every step here can be repeated: a step
that meets a lock raises, the caller calls again later, and what was done stays
done.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.person_contact_store import PersonContactStore
from openviking.session.memory.person_identity import person_memory_uri
from openviking.session.memory.question_store import QuestionStore, memory_root
from openviking_cli.exceptions import InvalidArgumentError


@asynccontextmanager
async def _exact_lock(fs, ctx, uri):
    """Hold one file against extraction, which locks the folder above it."""
    if not getattr(fs, "agfs", None):
        yield
        return
    from openviking.storage.transaction import LockContext, get_lock_manager, init_lock_manager

    try:
        manager = get_lock_manager()
    except RuntimeError:
        manager = init_lock_manager(fs.agfs)
    async with LockContext(manager, [fs._uri_to_path(uri, ctx)], lock_mode="exact"):
        yield


async def _repoint_references(fs, ctx, db, anchor, into_anchor):
    """Make every page that points at the retired person point at the other one."""
    root = memory_root(ctx)
    retired_folder = root + f"people/{anchor}/"
    # A person's id occurs nowhere but in their own paths, so the path segment
    # is rewritten wherever it stands: in a full address and in a relative link.
    old, new = f"people/{anchor}/", f"people/{into_anchor}/"
    repointed = []
    for entry in await fs.tree(root, node_limit=None, level_limit=None, ctx=ctx):
        uri = entry["uri"]
        if entry.get("isDir") or not uri.endswith(".md") or uri.startswith(retired_folder):
            continue
        if old not in await fs.read_file(uri, ctx=ctx):
            continue
        async with _exact_lock(fs, ctx, uri):
            raw = await fs.read_file(uri, ctx=ctx)
            await fs.write_file(uri, raw.replace(old, new), ctx=ctx)
        await MemoryUpdater.refresh_file_embedding(
            viking_fs=fs,
            vikingdb=db,
            uri=uri,
            memory_type=MemoryUpdater.memory_type_from_uri(uri),
            ctx=ctx,
        )
        repointed.append(uri)
    return repointed


async def retire_person(fs, ctx, db, anchor, into_anchor):
    """Leave one person where there were two.

    The caller has already had the retired person's memory merged into the
    other's. Here the retired person leaves the directory first, so nothing
    chooses them from then on; their questions move; pages that pointed at
    them point at the other person; their folder goes last.
    """
    anchor, into_anchor = str(UUID(str(anchor))), str(UUID(str(into_anchor)))
    if anchor == into_anchor:
        raise InvalidArgumentError("A person cannot be retired into themself")
    await PersonContactStore(fs, ctx, db).retire(anchor, into_anchor)
    await QuestionStore(fs, ctx, db).move_person(
        anchor, into_anchor, person_memory_uri(ctx, into_anchor)
    )
    repointed = await _repoint_references(fs, ctx, db, anchor, into_anchor)
    await fs.rm(memory_root(ctx) + f"people/{anchor}/", recursive=True, ctx=ctx)
    return {"anchorId": anchor, "intoAnchorId": into_anchor, "repointed": repointed}
