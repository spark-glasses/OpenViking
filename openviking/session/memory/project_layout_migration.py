"""Explicit offline Project migration; no runtime aliases or automatic migration."""

import hashlib
import json
import os
import re
from pathlib import Path

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.person_layout_migration import read_or_none, rewrite_document
from openviking.session.memory.project_paths import (
    project_metadata_uri,
    project_questions_uri,
    project_uri,
)
from openviking.session.memory.question_store import memory_root, render_questions
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils


def digest(value):
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


async def plan_migration(fs, ctx):
    root = memory_root(ctx)
    entries = await fs.tree(
        root.rstrip("/"), show_all_hidden=True, node_limit=None, level_limit=None, ctx=ctx
    )
    originals = {
        e["uri"]: await fs.read_file(e["uri"], ctx=ctx)
        for e in entries
        if not e.get("isDir") and e["uri"].endswith((".md", ".json"))
    }
    outputs, moves, ids = {}, {}, {}
    for uri, raw in originals.items():
        match = re.fullmatch(re.escape(root) + r"projects/([0-9a-f-]{36})\.md", uri)
        if not match:
            continue
        page = MemoryFileUtils.read(raw, uri=uri)
        identifier = match.group(1)
        if page.extra_fields.get("projectId") != identifier:
            raise ValueError("Project identity does not match migration path")
        target = project_uri(ctx, identifier)
        moves[uri], ids[uri] = target, identifier
        outputs[target] = MemoryFileUtils.write(
            MemoryFile(
                uri=target,
                memory_type="projects",
                content=page.content,
                links=page.links,
                backlinks=page.backlinks,
            )
        )
        outputs[project_metadata_uri(ctx, identifier)] = json.dumps(
            page.extra_fields, ensure_ascii=False, indent=2
        )
    for uri, raw in originals.items():
        if not uri.endswith("/questions.md"):
            continue
        page = MemoryFileUtils.read(raw, uri=uri)
        subject = page.extra_fields.get("subject", {})
        old = subject.get("memoryUri")
        if old not in ids:
            continue
        subject.update(kind="project", id=ids[old], memoryUri=moves[old])
        target = project_questions_uri(ctx, ids[old])
        page.uri = target
        page.content = render_questions(subject, page.extra_fields["questions"])
        moves[uri], outputs[target] = target, MemoryFileUtils.write(page)
    if any(target in originals for target in outputs):
        raise ValueError("Project migration destination already exists")
    # Active memory references move; immutable session evidence is deliberately
    # outside this tree and remains interpretable using this migration manifest.
    for uri, raw in originals.items():
        if uri not in moves:
            updated = rewrite_document(raw, uri, uri, moves)
            if updated != raw:
                outputs[uri] = updated
    reverse_moves = {target: source for source, target in moves.items()}
    outputs = {
        uri: rewrite_document(raw, reverse_moves.get(uri, uri), uri, moves)
        for uri, raw in outputs.items()
    }
    return {
        "userId": ctx.user.user_id,
        "accountId": ctx.account_id,
        "root": root,
        "status": "prepared",
        "moves": moves,
        "sources": {uri: originals[uri] for uri in moves},
        "files": [
            {
                "uri": uri,
                "before": originals.get(uri),
                "after": raw,
                "beforeHash": digest(originals.get(uri)),
                "afterHash": digest(raw),
            }
            for uri, raw in outputs.items()
        ],
    }


async def apply_migration(fs, ctx, plan, journal_path, db=None):
    if (plan["userId"], plan["accountId"], plan["root"]) != (
        ctx.user.user_id,
        ctx.account_id,
        memory_root(ctx),
    ):
        raise ValueError("Migration belongs to another owner")
    if plan["status"] == "complete":
        return
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
            json.dump(plan, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
    # Validate every precondition before mutating any file.
    for file in plan["files"]:
        if (
            not file["uri"].startswith(plan["root"])
            or digest(file["after"]) != file["afterHash"]
            or digest(file["before"]) != file["beforeHash"]
        ):
            raise ValueError("Invalid migration entry")
        current = await read_or_none(fs, file["uri"], ctx)
        if digest(current) not in (file["beforeHash"], file["afterHash"]):
            raise ValueError("Migration target changed")
    for uri, before in plan["sources"].items():
        if not uri.startswith(plan["root"]) or any(part in (".", "..") for part in uri.split("/")):
            raise ValueError("Migration source outside selected owner")
        current = await read_or_none(fs, uri, ctx)
        if current is not None and current != before:
            raise ValueError("Migration source changed")
    for file in plan["files"]:
        await fs.write_file(file["uri"], file["after"], ctx=ctx)
        if await fs.read_file(file["uri"], ctx=ctx) != file["after"]:
            raise RuntimeError("Migration verification failed")
        # Derived directory abstracts/overviews are not L2 memory documents.
        # Their filename deliberately has no canonical people/entity schema.
        if db is not None and file["uri"].endswith(".md") and not file["uri"].rsplit("/", 1)[-1].startswith("."):
            from openviking.session.memory.memory_updater import MemoryUpdater

            indexed = await MemoryUpdater.refresh_file_embedding(
                viking_fs=fs,
                vikingdb=db,
                ctx=ctx,
                uri=file["uri"],
                memory_type="questions"
                if file["uri"].endswith("/questions.md")
                else MemoryUpdater.memory_type_from_uri(file["uri"]),
            )
            if not indexed:
                raise RuntimeError("Migration index enqueue failed; keep journal for retry")
    for uri in plan["moves"]:
        if await read_or_none(fs, uri, ctx) is not None:
            await fs.rm(uri, ctx=ctx)
    plan["status"] = "complete"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as output:
        json.dump(plan, output, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
