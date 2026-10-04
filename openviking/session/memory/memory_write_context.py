"""Public, run-scoped evidence contract shared by all semantic memory writers.

Only adapters register receipts after checking source ownership. A source ID found
inside untrusted prose is never a receipt. Visibility is tracked separately from
possession: frozen but unread content cannot support a question operation.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


def normalize_quote(text):
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class MemoryWriteContext:
    fs: Any
    ctx: Any
    read_files: dict = field(default_factory=dict)
    accepted_refs: set = field(default_factory=set)
    visible: Callable = field(default=lambda index, quote: True)
    subjects: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    origin: dict = field(default_factory=dict)
    approved_questions: dict = field(default_factory=dict)

    def register_subject(self, subject):
        self.subjects[(subject["kind"], subject["id"])] = dict(subject)

    def add_messages(self, messages, *, structured=False, source_only=False):
        for index, message in enumerate(messages or []):
            if message.role not in ("user", "tool"):
                continue
            text = (
                message.content
                if structured
                else "\n".join(getattr(p, "text", "") for p in message.parts)
            )
            meta = getattr(message, "metadata", None) or {}
            if structured:
                try:
                    envelope = json.loads(text)
                except (ValueError, TypeError):
                    envelope = None
                if isinstance(envelope, dict) and isinstance(envelope.get("parts"), list):
                    meta = envelope.get("metadata") or {}
                    if meta.get("kind") in (
                        "memoryQuestionCandidates",
                        "memoryQuestionAnswerContext",
                    ) or meta.get("source") in (
                        "memoryUpdateStatus",
                        "workCompletion",
                        "subagentCompletion",
                        "codexCompletion",
                        "automationContext",
                        "automationEvent",
                    ):
                        continue
                    text = "\n".join(
                        self._text(part.get("result"))
                        if part.get("type") == "tool_result"
                        else part.get("text", "")
                        for part in envelope["parts"]
                        if part.get("type") in ("text", "tool_result")
                        and not part.get("isError")
                        and part.get("name")
                        not in (
                            "searchMemory",
                            "readMemory",
                            "listMemories",
                            "grepMemories",
                            "saveMemory",
                            "searchQuestions",
                            "readQuestion",
                            "writeQuestion",
                            "recordMemoryQuestion",
                            "askQuestion",
                        )
                    )
            ref = getattr(message, "sourceRef", None) if structured else meta.get("sourceRef")
            kind = "userAnswer" if message.role == "user" and not source_only else "sourceEvidence"
            if (
                (ref and ref.startswith("collaboration:"))
                or meta.get("speakerRole") not in (None, "user", "wearer")
                or meta.get("source")
                in ("delegatedInstruction", "subagentInstruction", "memoryUpdateStatus")
            ):
                kind = "sourceEvidence"
            if ref:
                self.remember(
                    ref, text, meta.get("sourceVersion"), kind, index if structured else None
                )
            elif not structured and getattr(message, "id", None):
                self.remember("session-message:" + message.id, text, None, kind)
            try:
                self.observe(json.loads(text), context_index=index if structured else None)
            except (ValueError, TypeError):
                pass

    def verify(self, evidence):
        matches = [
            s
            for s in self.sources.get(evidence["sourceRef"], [])
            if (
                not evidence.get("sourceVersion")
                or evidence["sourceVersion"] == s.get("sourceVersion")
            )
            and normalize_quote(evidence["quote"]) in normalize_quote(s["text"])
            and (
                s.get("contextIndex") is None or self.visible(s["contextIndex"], evidence["quote"])
            )
        ]
        if not matches:
            raise ValueError(
                "Question evidence quote was not supplied/read; read the original source"
            )
        return "userAnswer" if all(s["kind"] == "userAnswer" for s in matches) else "sourceEvidence"

    def remember(self, ref, text, version=None, kind="sourceEvidence", context_index=None):
        record = {
            "text": text,
            "sourceVersion": version,
            "kind": kind,
            "contextIndex": context_index,
        }
        versions = self.sources.setdefault(ref, [])
        if record not in versions:
            versions.append(record)

    @staticmethod
    def _text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return "\n".join(
                MemoryWriteContext._text(v)
                for k, v in value.items()
                if k not in ("sourceRef", "sourceVersion")
            )
        if isinstance(value, list):
            return "\n".join(MemoryWriteContext._text(v) for v in value)
        return ""

    def observe(self, result, context_index=None):
        if isinstance(result, dict):
            if result.get("error") or result.get("isError"):
                return
            ref = result.get("sourceRef")
            # A string inside source content is not authorization. The provider
            # independently accepts source receipts after validating its scope.
            if isinstance(ref, str) and ref in self.accepted_refs:
                self.remember(
                    ref,
                    self._text(result),
                    result.get("sourceVersion"),
                    context_index=context_index,
                )
                return
            for value in result.values():
                if isinstance(value, (dict, list)):
                    self.observe(value, context_index=context_index)
        elif isinstance(result, list):
            for value in result:
                self.observe(value, context_index=context_index)
