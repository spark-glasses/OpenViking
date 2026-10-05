"""Question domain entry point, shared by extraction, Spark tools and business writes."""

import hashlib
import json
import re
from uuid import NAMESPACE_URL, uuid5

from openviking.session.memory.person_paths import person_anchor_from_uri
from openviking.session.memory.project_paths import project_id_from_uri
from openviking.session.memory.focus_paths import focus_id_from_uri
from openviking.session.memory.question_store import (
    QuestionStore,
    memory_root,
    question_uri,
    validate_proposals,
)


def resolve_question_subject(write, fields):
    root = memory_root(write.ctx)
    kind, uri = fields.get("subjectKind"), fields.get("subjectMemoryUri", "")
    if isinstance(uri, str) and uri and not uri.startswith(root):
        raise ValueError("Question subject belongs to another user")
    if not uri and fields.get("subjectId"):
        registered = write.subjects.get((kind, fields["subjectId"]))
        if registered:
            uri = registered.get("memoryUri", "")
    if kind == "self":
        return {"kind": "self", "id": "self"}
    if kind == "unassigned":
        return {"kind": "unassigned", "id": "unassigned"}
    supplied = next((s for s in write.subjects.values() if s.get("memoryUri") == uri), None)
    if uri not in write.read_files and not supplied:
        raise ValueError("Read the question's existing subject before assigning ownership")
    if supplied:
        if supplied["kind"] != kind:
            raise ValueError("Question subject kind does not match the registered identity")
        return supplied
    if kind == "person" and (identifier := person_anchor_from_uri(uri, root)):
        page = write.read_files.get(uri)
        if page and (
            page.extra_fields.get("contact_projection_deleted")
            or page.extra_fields.get("merged_into")
        ):
            raise ValueError("Question requires an active canonical person")
        return {"kind": kind, "id": identifier, "memoryUri": uri}
    if kind == "project" and (identifier := project_id_from_uri(uri, root)):
        return {"kind": kind, "id": identifier, "memoryUri": uri}
    if kind == "focus" and (identifier := focus_id_from_uri(uri, root)):
        return {"kind": kind, "id": identifier, "memoryUri": uri}
    if (
        kind == "matter"
        and isinstance(uri, str)
        and any(
            uri.startswith(root + directory + "/")
            for directory in ("entities", "events", "matters")
        )
        and uri.endswith(".md")
        and not uri.endswith("/questions.md")
        and not any(x in uri for x in ("..", "%", "?", "#", "\\"))
    ):
        return {"kind": kind, "id": uuid5(NAMESPACE_URL, uri).hex, "memoryUri": uri}
    raise ValueError("Invalid question subject; use its canonical People, Focus or matter identity")


class QuestionService:
    def __init__(self, write, db=None):
        self.write = write
        self.store = QuestionStore(write.fs, write.ctx, db)

    def route(self, operation):
        subject = resolve_question_subject(self.write, operation.memory_fields)
        if subject["kind"] != "matter" and operation.memory_fields.get("subjectId") not in (
            None,
            "",
            subject["id"],
        ):
            raise ValueError("Question subject ID does not match its canonical memory URI")
        uri = question_uri(self.write.ctx, subject)
        operation.uris = [uri]
        operation.old_memory_file_content = self.write.read_files.get(uri)
        operation.memory_fields["question_subject"] = subject

    def approve(self, operation):
        """Seal source-specific validation, including server-derived meeting timing."""
        self.write.approved_questions[id(operation)] = self._digest(operation)

    @staticmethod
    def _digest(operation):
        return hashlib.sha256(
            json.dumps(
                [
                    operation.uris,
                    operation.memory_fields.get("entries"),
                    operation.memory_fields.get("question_subject"),
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    async def apply(self, operation):
        from openviking.session.memory.dataclass import ResolvedOperations

        if self.write.approved_questions.get(id(operation)) != self._digest(operation):
            self.route(operation)
            self.validate(
                ResolvedOperations(
                    upsert_operations=[operation], delete_file_contents=[], errors=[]
                )
            )
        entries = json.loads(operation.memory_fields["entries"])
        moves = [p for p in entries if p.get("action") == "relocate"]
        if moves:
            if len(entries) != 1:
                raise ValueError("Relocate one question per operation")
            proposal = moves[0]
            return [
                await self.store.relocate(
                    proposal["questionId"],
                    proposal["expectedRevision"],
                    operation.memory_fields["question_subject"],
                    proposal["evidence"],
                )
            ]
        return await self.store.discover(
            operation.uris[0],
            operation.memory_fields["question_subject"],
            json.loads(operation.memory_fields["entries"]),
            metadata=operation.memory_fields,
        )

    def validate(self, operations):
        allowed = set(self.write.sources) | self.write.accepted_refs
        from openviking.session.memory.question_store import is_question_uri

        if any(
            is_question_uri(page.uri, self.write.ctx) for page in operations.delete_file_contents
        ):
            raise ValueError("Question history cannot be deleted by extraction")
        for op in operations.upsert_operations:
            if op.memory_type != "questions":
                if any(is_question_uri(uri, self.write.ctx) for uri in op.uris):
                    raise ValueError("Question records require structured question operations")
                continue
            proposals = validate_proposals(op.memory_fields.get("entries"), allowed)
            if any(not p.get("context") or not p.get("evidence") for p in proposals):
                raise ValueError("Question writes require causal context and quoted evidence")
            kinds = {}
            for proposal in proposals:
                read = self.write.read_files
                if any(uri not in read for uri in proposal.get("relatedSubjectUris", [])):
                    raise ValueError("Read related question subjects before linking them")
                evidence_kinds = []
                # A matching source ID alone is insufficient. Verify quotations
                # against the exact material actually supplied/read in this run.
                for evidence in proposal.get("evidence", []):
                    evidence_kinds.append(self.write.verify(evidence))
                if proposal.get("action") in ("resolve", "obsolete", "relocate"):
                    known = next(
                        (
                            q
                            for page in self.write.read_files.values()
                            for q in page.extra_fields.get("questions", [])
                            if q["questionId"] == proposal.get("questionId")
                        ),
                        None,
                    )
                    if not known:
                        raise ValueError("Read the existing question before resolving it")
                    if (
                        proposal["action"] == "relocate"
                        and op.memory_fields.get("subjectKind") == "person"
                        and any(k != "userAnswer" for k in evidence_kinds)
                    ):
                        raise ValueError(
                            "Assigning question ownership to a person requires user confirmation"
                        )
                    if proposal["action"] == "resolve" and known.get("purpose") in (
                        "speakerIdentity",
                        "personIdentity",
                        "externalIdentity",
                        "emailIdentity",
                        "profileCorrection",
                    ):
                        raise ValueError(
                            "Use the explicit user-confirmation workflow for this identity/profile question"
                        )
                    kinds[proposal["questionId"]] = (
                        "userAnswer"
                        if evidence_kinds and all(k == "userAnswer" for k in evidence_kinds)
                        else "sourceEvidence"
                    )
            op.memory_fields["entries"] = json.dumps(proposals, ensure_ascii=False)
            op.memory_fields["question_resolution_kinds"] = kinds

    async def search(self, query="", subject_ids=(), include_resolved=False, limit=6):
        terms = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for item in await self.store.list():
            if not include_resolved and item["state"] != "open":
                continue
            related = item["subject"]["id"] in subject_ids or any(
                any(identifier in uri for identifier in subject_ids)
                for uri in item.get("relatedSubjectUris", [])
            )
            overlap = len(
                terms
                & set(
                    re.findall(
                        r"\w+",
                        json.dumps(
                            [item["text"], item.get("context", {})], ensure_ascii=False
                        ).casefold(),
                    )
                )
            )
            if (terms or subject_ids) and not (related or overlap):
                continue
            scored.append((20 * int(related) + overlap, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["questionId"]))
        return [
            {
                key: item.get(key)
                for key in (
                    "questionId",
                    "revision",
                    "subject",
                    "text",
                    "context",
                    "importance",
                    "state",
                    "asking",
                )
            }
            for _, item in scored[: max(1, min(limit, 20))]
        ]

    async def submit(self, fields):
        """Structured write entry point for non-loop writers; no model call."""
        from openviking.session.memory.dataclass import ResolvedOperation

        operation = ResolvedOperation(memory_type="questions", uris=[], memory_fields=fields)
        result = await self.apply(operation)
        for uri in operation.uris:
            await self.store.refresh(uri)
        return result

    async def discover_business_event(self, subject, proposal, *, event_ref, event):
        """A validated backend observation is evidence of a conflict, not identity truth.

        Keep original references for investigation, but quote the actual structured
        receipt. Do not invent a quotation from an email we have never read here.
        """
        text = json.dumps(event, ensure_ascii=False, sort_keys=True)
        self.write.remember(event_ref, text)
        self.write.accepted_refs.update(proposal.get("sourceRefs", []))
        self.write.register_subject(subject)
        proposal = {
            **proposal,
            "sourceRefs": list(dict.fromkeys([event_ref, *proposal.get("sourceRefs", [])[:19]])),
            "evidence": [{"sourceRef": event_ref, "quote": text[:4000]}],
        }
        return await self.submit(
            {
                "subjectKind": subject["kind"],
                "subjectId": subject["id"],
                "subjectMemoryUri": subject.get("memoryUri", ""),
                "entries": [proposal],
            }
        )

    async def record(self, event):
        """Trusted user/display events remain distinct from model proposals."""
        return await self.store.record(event)

    async def propagate(self, question_id, sessions):
        return await self.store.propagate(question_id, sessions)
