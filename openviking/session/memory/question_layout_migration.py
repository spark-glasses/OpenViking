"""Offline upgrade: asking preferences are independent of knowledge state."""

from openviking.session.memory.project_layout_migration import digest
from openviking.session.memory.question_store import is_question_uri, memory_root, render_questions
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils


async def plan_migration(fs, ctx):
    root = memory_root(ctx)
    entries = await fs.tree(root.rstrip("/"), ctx=ctx, node_limit=None, level_limit=None)
    files = []
    for entry in entries:
        uri = entry.get("uri", "")
        if not is_question_uri(uri, ctx):
            continue
        before = await fs.read_file(uri, ctx=ctx)
        page = MemoryFileUtils.read(before, uri=uri)
        version = page.extra_fields.get("questionFormatVersion")
        if version == 2:
            continue
        if version != 1:
            raise ValueError("Unsupported question format; refusing lossy migration")
        for question in page.extra_fields["questions"]:
            old = question["state"]
            if old not in ("open", "asked", "deferred", "dismissed", "resolved"):
                raise ValueError("Unknown question state")
            question["state"] = "resolved" if old == "resolved" else "open"
            question["asking"] = (
                "muted" if old == "dismissed" else "snoozed" if old == "deferred" else "allowed"
            )
            question.setdefault(
                "importance",
                {
                    "level": "soon" if question.get("purpose") == "speakerIdentity" else "later",
                    "reason": "Migrated from existing question policy",
                },
            )
            # Existing records may have no original quote. Keep their references
            # and history, never manufacture evidence or claim we recovered it.
            question.setdefault("evidence", [])
            question.setdefault(
                "context",
                {
                    "summary": question["text"],
                    "uncertainty": question["text"],
                    "knownFacts": [],
                    "candidates": [],
                },
            )
            if old == "resolved" and question.get("resolution"):
                event = next(
                    (e for e in reversed(question.get("events", [])) if e["action"] == "resolved"),
                    None,
                )
                if event:
                    question["resolutionRecord"] = {
                        "kind": "userAnswer",
                        "text": question["resolution"],
                        "sourceRef": event.get("sourceRef"),
                        "eventId": event["eventId"],
                    }
        page.extra_fields["questionFormatVersion"] = 2
        page.extra_fields["revision"] = page.extra_fields.get("revision", 0) + 1
        page.content = render_questions(
            page.extra_fields["subject"], page.extra_fields["questions"]
        )
        after = MemoryFileUtils.write(page)
        files.append(
            {
                "uri": uri,
                "before": before,
                "after": after,
                "beforeHash": digest(before),
                "afterHash": digest(after),
            }
        )
    return {
        "userId": ctx.user.user_id,
        "accountId": ctx.account_id,
        "root": root,
        "status": "prepared",
        "moves": {},
        "sources": {},
        "files": files,
    }
