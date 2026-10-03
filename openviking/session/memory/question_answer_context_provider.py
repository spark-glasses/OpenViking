"""Native extraction of a persisted user clarification, including person anchors."""

from pathlib import Path

from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.question_store import is_question_uri, memory_root
from openviking.session.memory.session_extract_context_provider import SessionExtractContextProvider
from openviking.session.memory.tools import add_tool_call_pair_to_messages, get_tool


def answer_registry(subject):
    registry = create_default_registry()
    registry.load_from_yaml(
        str(Path(__file__).parents[2] / "prompts/templates/memory/email/people.yaml"), replace=True
    )
    registry.get("people").filename_template = (
        (subject["id"] + "/memory.md") if subject["kind"] == "person" else "person.md"
    )
    return registry


class QuestionAnswerContextProvider(SessionExtractContextProvider):
    supports_links = False

    def __init__(self, *, question_context, **kwargs):
        super().__init__(**kwargs)
        self.subject = question_context["subject"]
        self._registry = answer_registry(self.subject)
        self.root = memory_root(self._ctx)
        self._link_enabled = False

    def get_tools(self):
        return ["read", "search"]

    def create_tool_context(self, default_search_uris=None):
        return super().create_tool_context([self.root.rstrip("/")])

    def instruction(self):
        return (
            super().instruction()
            + "\nThis input is an actual user clarification of a persisted question. Read and update its existing subject memory using the answer, including negative corrections. The known subject is "
            + str(self.subject)
            + ". Do not duplicate an existing person as a new entity. Preserve the question page: its answer and lifecycle are already managed by OV. Do not delete memories or create new matter cards."
        )

    async def execute_tool(self, call):
        args = dict(call.arguments or {})
        if call.name == "read":
            uri = args.get("uri", "")
            if (
                not uri.startswith(self.root)
                or any(p in ("", ".", "..") for p in uri[len(self.root) :].split("/"))
                or any(c in uri for c in ("%", "?", "#", "\\"))
            ):
                raise ValueError("Question clarification read outside user memory")
            args = {"uri": uri}
        elif call.name == "search":
            args = {
                "query": str(args.get("query", ""))[:2000],
                "limit": min(10, max(1, int(args.get("limit", 5)))),
            }
        else:
            raise ValueError("Unsupported clarification tool")
        return await get_tool(call.name).execute(self.create_tool_context(), **args)

    async def prefetch(self):
        from openviking.models.vlm.base import ToolCall

        messages = [self._build_conversation_message()]
        for uri in dict.fromkeys([self.root + "profile.md", self.subject.get("memoryUri")]):
            if uri:
                value = await self.execute_tool(ToolCall("prefetch", "read", {"uri": uri}))
                add_tool_call_pair_to_messages(messages, len(messages), "read", {"uri": uri}, value)
        return messages

    def validate_operations(self, operations):
        if operations.errors or operations.delete_file_contents or operations.resolved_links:
            raise ValueError("Clarification cannot delete memories or mutate implicit targets")
        for operation in operations.upsert_operations:
            for uri in operation.uris:
                if not uri.startswith(self.root) or is_question_uri(uri, self._ctx):
                    raise ValueError("Clarification cannot change question records or other users")
                if uri not in self.read_file_contents and uri != self.root + "profile.md":
                    raise ValueError("Clarification must update an existing fully read subject")
