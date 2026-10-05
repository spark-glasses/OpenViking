"""Explicit, single-owner People migration. Never runs on server startup.

Run under the workspace process lock with API/queue writers stopped. A local
journal keeps original and replacement bytes, allowing inspection and recovery.
Neither names, source IDs nor question lifecycle states are changed.
"""

import hashlib
import json
import os
import posixpath
import re
from pathlib import Path

from openviking.core.namespace import canonical_user_root
from openviking.session.memory.memory_update_context import MemoryUpdateContext
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking_cli.exceptions import NotFoundError


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def relocated_uri(uri, root):
    prefix = root + "/memories/people/"
    if not uri.startswith(prefix):
        return uri
    name = uri[len(prefix):]
    if name == ".contacts/directory.json":
        return prefix + ".directory.json"
    profile = re.fullmatch(r"\.contacts/([A-Za-z0-9_-]+)\.json", name)
    person = re.fullmatch(r"([A-Za-z0-9_-]+)\.md", name)
    if profile:
        return prefix + profile.group(1) + "/profile.json"
    if person:
        return prefix + person.group(1) + "/memory.md"
    return uri


def rewrite_document(raw, old_uri, new_uri, moves):
    if old_uri.endswith(".md"):
        def link(match):
            target = match.group(2)
            if ":" in target or target.startswith(("/", "#")):
                return match.group(0)
            path, sep, fragment = target.partition("#")
            absolute = "viking://" + posixpath.normpath(posixpath.join(
                posixpath.dirname(old_uri.removeprefix("viking://")), path))
            destination = moves.get(absolute, absolute)
            if old_uri == new_uri and destination == absolute:
                return match.group(0)
            relative = posixpath.relpath(destination.removeprefix("viking://"),
                                         posixpath.dirname(new_uri.removeprefix("viking://")))
            return match.group(1) + relative + (sep + fragment if sep else "") + ")"
        raw = re.sub(r"(\[[^\]]*\]\()([^\s)]+)\)", link, raw)
    if moves:
        pattern = re.compile("|".join(re.escape(uri) for uri in sorted(moves, key=len, reverse=True)))
        raw = pattern.sub(lambda match: moves[match.group(0)], raw)
    return raw


async def read_or_none(fs, uri, ctx):
    try:
        return await fs.read_file(uri, ctx=ctx)
    except (NotFoundError, FileNotFoundError):
        return None


async def plan_migration(fs, ctx, reference_map=None):
    root = canonical_user_root(ctx)
    entries = await fs.tree(root, show_all_hidden=True, node_limit=None, level_limit=None, ctx=ctx)
    uris = [e["uri"] for e in entries if not e.get("isDir")
            and e["uri"].startswith(root + "/")
            and e["uri"].endswith((".md", ".json", ".jsonl"))]
    moves = {uri: relocated_uri(uri, root) for uri in uris if relocated_uri(uri, root) != uri}
    occupied = set(uris)
    for source, target in moves.items():
        if target in occupied:
            raise RuntimeError(f"People migration collision: {source} -> {target}")
    replacements = {**(reference_map or {}), **moves}
    files = []
    for uri in uris if replacements else []:
        raw = await fs.read_file(uri, ctx=ctx)
        target = moves.get(uri, uri)
        updated = rewrite_document(raw, uri, target, replacements)
        if uri.endswith("/memory_update.json"):
            receipt = json.loads(updated)
            if archive := receipt.get("archiveUri"):
                context_uri = archive + "/memory_update_context.json"
                context = await read_or_none(fs, context_uri, ctx)
                if context is not None:
                    migrated = rewrite_document(context, context_uri, context_uri, replacements)
                    if migrated != context:
                        receipt["inputHash"] = MemoryUpdateContext.model_validate(json.loads(migrated)).input_hash()
                        updated = json.dumps(receipt, ensure_ascii=False, indent=2)
        if updated != raw or target != uri:
            files.append({"source": uri, "target": target, "before": raw, "after": updated,
                          "beforeHash": digest(raw), "afterHash": digest(updated)})
    return {"version": 1, "accountId": ctx.account_id, "userId": ctx.user.user_id,
            "root": root, "status": "prepared", "moves": moves, "files": files}


async def apply_migration(fs, ctx, plan, journal_path, db=None):
    root = canonical_user_root(ctx)
    if (plan["accountId"], plan["userId"], plan["root"]) != (ctx.account_id, ctx.user.user_id, root):
        raise ValueError("Migration journal belongs to a different owner")
    if plan["status"] == "complete":
        return
    for item in plan["files"]:
        if any(not item[key].startswith(root + "/") for key in ("source", "target")):
            raise ValueError("Migration path outside selected user")
        if digest(item["before"]) != item["beforeHash"] or digest(item["after"]) != item["afterHash"]:
            raise ValueError("Migration journal failed integrity check")
    # The complete pre-migration snapshot is durable BEFORE any live write.
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
            json.dump(plan, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o600)
    for item in plan["files"]:
        source, target = item["source"], item["target"]
        current = await read_or_none(fs, target, ctx)
        allowed = {item["afterHash"]} | ({item["beforeHash"]} if source == target else set())
        if current is not None and digest(current) not in allowed:
            raise RuntimeError(f"Migration destination changed: {target}")
        original = await read_or_none(fs, source, ctx)
        if source != target and original is not None and digest(original) != item["beforeHash"]:
            raise RuntimeError(f"Migration source changed: {source}")
        if current is None and original is None:
            raise RuntimeError(f"Migration source and destination missing: {source}")
        await fs.write_file(target, item["after"], ctx=ctx)
        if await fs.read_file(target, ctx=ctx) != item["after"]:
            raise RuntimeError(f"Migration verification failed: {target}")
        if db is not None and "/memories/" in target and target.endswith(".md") and "/." not in target:
            indexed = await MemoryUpdater.refresh_file_embedding(viking_fs=fs, vikingdb=db,
                uri=target, memory_type=MemoryUpdater.memory_type_from_uri(target), ctx=ctx)
            if not indexed:
                raise RuntimeError(f"Migration could not enqueue index: {target}")
        if source != target and original is not None:
            # Only retire a source after backup, read-after-write verification
            # and successful index enqueue; VikingFS also removes old vectors.
            await fs.rm(source, ctx=ctx)
    plan["status"] = "complete"
    # Original bytes remain in this receipt for recovery; do not delete it.
    pending = path.with_suffix(".tmp")
    with os.fdopen(os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as output:
        json.dump(plan, output, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    pending.replace(path)
