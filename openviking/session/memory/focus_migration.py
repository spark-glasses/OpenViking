"""Offline semantic migration. Review classification before applying with writers stopped.

This is intentionally not a runtime alias: old project IDs are explicitly mapped
to personal focuses or native entities. The existing migration writer journals
before/after content and verifies all preconditions before deleting old files.
Immutable source/session archives are left untouched; the manifest maps their URIs.
"""

import json
from uuid import NAMESPACE_URL, uuid5

from openviking.session.memory.focus_paths import focus_uri, focus_metadata_uri
from openviking.session.memory.person_layout_migration import rewrite_document
from openviking.session.memory.project_layout_migration import digest
from openviking.session.memory.project_layout_migration import apply_migration as apply_migration
from openviking.session.memory.project_store import ProjectStore
from openviking.session.memory.question_store import (
    memory_root,
    now_iso,
    question_uri,
    render_questions,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import NotFoundError


async def plan_lifecycle_migration(fs, ctx):
    """One-time dev conversion of visibility + status; never a runtime fallback.

    Hidden means archived, preserving the user's decision not to surface it.
    Existing archives win over discovery visibility. Identity, narrative and
    question history stay in place. Revision bumps invalidate stale UI edits.
    """
    root = memory_root(ctx)
    try:
        entries = await fs.tree(
            root + "focuses", show_all_hidden=True, node_limit=None, level_limit=None, ctx=ctx
        )
    except NotFoundError:
        entries = []
    if any(e.get("uri", "").endswith("/.pending-write.json") for e in entries):
        raise ValueError("Recover pending Focus writes before migration")
    files = []
    for entry in entries:
        uri = entry.get("uri", "")
        if entry.get("isDir") or not uri.endswith("/focus.json"):
            continue
        before = await fs.read_file(uri, ctx=ctx)
        fields = json.loads(before)
        if "visibility" not in fields:
            continue
        visibility = fields.pop("visibility")
        if visibility not in ("visible", "hidden", "forming") or fields["status"] not in (
            "active",
            "archived",
        ):
            raise ValueError("Unknown legacy Focus lifecycle; review before migration")
        fields["status"] = (
            "archived"
            if fields["status"] == "archived" or visibility == "hidden"
            else "forming"
            if visibility == "forming"
            else "active"
        )
        protected = set(fields.get("protectedFields", []))
        if "visibility" in protected:
            protected.remove("visibility")
            protected.add("status")
        fields["protectedFields"] = sorted(protected)
        fields["revision"] += 1
        fields["updatedAt"] = now_iso()
        fields.pop("lastOperationId", None)
        fields.pop("lastOperationHash", None)
        after = json.dumps(fields, ensure_ascii=False, indent=2)
        files.append(
            dict(
                uri=uri,
                before=before,
                after=after,
                beforeHash=digest(before),
                afterHash=digest(after),
            )
        )
    return dict(
        userId=ctx.user.user_id,
        accountId=ctx.account_id,
        root=root,
        status="prepared",
        moves={},
        sources={},
        files=files,
    )


async def plan_migration(fs, ctx, decisions):
    """decisions: {projectId: {kind: entity|focus, userIntent?: str}}.

    Every old Project must have an explicit decision. Connection-owned records
    should become entities; do not classify unknown intent from message volume.
    """
    root = memory_root(ctx)
    entries = await fs.tree(
        root.rstrip("/"), show_all_hidden=True, node_limit=None, level_limit=None, ctx=ctx
    )
    originals = {
        e["uri"]: await fs.read_file(e["uri"], ctx=ctx)
        for e in entries
        if not e.get("isDir") and e["uri"].endswith((".md", ".json"))
    }
    if any("/projects/" in uri and uri.endswith("/.pending-write.json") for uri in originals):
        raise ValueError("Recover pending Project writes before migration")
    projects = ProjectStore(fs, ctx)
    cursor, catalog = None, []
    while True:
        page = await projects.list(cursor, 100)
        catalog.extend(page["projects"])
        cursor = page["nextCursor"]
        if not cursor:
            break
    if set(decisions) != {p["projectId"] for p in catalog}:
        raise ValueError("Every Project needs one reviewed classification")
    moves, outputs, subjects = {}, {}, {}
    for item in catalog:
        identifier = item["projectId"]
        choice = decisions[identifier]
        page = await projects.get(identifier)
        old_uri = page.uri
        metadata = page.extra_fields
        if choice["kind"] == "focus":
            if not isinstance(choice.get("userIntent"), str) or not choice["userIntent"].strip():
                raise ValueError("A migrated Focus needs reviewed personal meaning")
            target = focus_uri(ctx, identifier)
            fields = {
                "focusId": identifier,
                "name": metadata["name"],
                "aliases": metadata.get("aliases", []),
                "userIntent": choice["userIntent"],
                "origin": "user",
                "status": metadata.get("status", "active"),
                "summary": "",
                "discoveryReason": "",
                "sourceRefs": metadata.get("memory_update_source_refs", []),
                "protectedFields": ["name", "userIntent", "status"],
                "revision": metadata.get("revision", 1),
                "createdAt": metadata.get("createdAt"),
                "updatedAt": metadata.get("updatedAt"),
                "migratedFrom": old_uri,
                "previousMetadata": metadata,
            }
            outputs[focus_metadata_uri(ctx, identifier)] = json.dumps(
                fields, ensure_ascii=False, indent=2
            )
            moves[item["metadataUri"]] = focus_metadata_uri(ctx, identifier)
            kind = "focus"
            subject = {"kind": "focus", "id": identifier, "memoryUri": target}
            page.extra_fields = {}
        elif choice["kind"] == "entity":
            target = root + "entities/undertakings/" + identifier + ".md"
            # Keep all old metadata and source bindings, even when it is not a
            # first-class field in the entity extraction schema.
            page.extra_fields.update(
                category="undertakings",
                name=identifier,
                displayName=metadata["name"],
                migratedFrom=old_uri,
            )
            moves[item["metadataUri"]] = target
            kind = "entity"
            subject = {
                "kind": "matter",
                "id": uuid5(NAMESPACE_URL, target).hex,
                "memoryUri": target,
            }
        else:
            raise ValueError("Unknown migration classification")
        subjects[old_uri] = subject
        page.uri, page.memory_type = target, "focuses" if kind == "focus" else "entities"
        moves[old_uri], outputs[target] = target, MemoryFileUtils.write(page)

    for uri, raw in originals.items():
        if not uri.endswith("/questions.md"):
            continue
        page = MemoryFileUtils.read(raw, uri=uri)
        old_subject = page.extra_fields.get("subject", {}).get("memoryUri")
        if old_subject not in subjects:
            continue
        subject = subjects[old_subject]
        page.extra_fields["subject"] = subject
        # Question IDs, answers, asking preferences and history are preserved.
        for question in page.extra_fields.get("questions", []):
            if "subject" in question:
                question["subject"] = subject
        page.uri = question_uri(ctx, subject)
        page.content = render_questions(subject, page.extra_fields["questions"])
        moves[uri], outputs[page.uri] = page.uri, MemoryFileUtils.write(page)
    if any(target in originals and target not in moves for target in outputs):
        raise ValueError("Migration destination already exists")
    for uri, raw in originals.items():
        if uri not in moves:
            updated = rewrite_document(raw, uri, uri, moves)
            if updated != raw:
                outputs[uri] = updated
    # Narratives resolve relative links against their original location. Metadata
    # has no relative Markdown links; all absolute links are rewritten as well.
    for source, target in moves.items():
        if source.endswith(".md"):
            outputs[target] = rewrite_document(outputs[target], source, target, moves)
    return {
        "userId": ctx.user.user_id,
        "accountId": ctx.account_id,
        "root": root,
        "status": "prepared",
        "decisions": decisions,
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
