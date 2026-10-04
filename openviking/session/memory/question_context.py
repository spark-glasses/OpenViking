"""Shared question capability; source providers retain their own tool/identity scope."""

from openviking.session.memory.question_service import QuestionService
from openviking.session.memory.tools import MemoryTool, register_tool
from openviking_cli.exceptions import NotFoundError


class QuestionTool(MemoryTool):
    def __init__(self, name, description, properties, required=()):
        self._name, self._description = name, description
        self._parameters = {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        }

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def parameters(self):
        return self._parameters

    async def execute(self, ctx, **kwargs):
        raise RuntimeError("Question tools require the shared extraction context")


register_tool(
    QuestionTool(
        "searchQuestions",
        "Find related memory questions, including muted questions that new evidence may resolve. Results are summaries; readQuestion loads the full evidence/history.",
        {
            "query": {"type": "string", "maxLength": 2000},
            "subjectIds": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
            "includeResolved": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    )
)
register_tool(
    QuestionTool(
        "readQuestion",
        "Read one question's authoritative revision, causal context, evidence, answer and user asking preferences.",
        {"questionId": {"type": "string"}},
        ["questionId"],
    )
)


class QuestionContext:
    tools = ("searchQuestions", "readQuestion")

    def __init__(self, provider):
        self.provider = provider
        self.write = provider.get_memory_write_context()
        self.service = QuestionService(self.write)
        self.store = self.service.store
        self.sources = self.write.sources
        self._calls = 0

    def observe(self, result, context_index=None):
        self.provider.get_memory_write_context()
        self.write.observe(result, context_index=context_index)

    async def search(self, query="", subject_ids=(), include_resolved=False, limit=6):
        return await self.service.search(query, subject_ids, include_resolved, limit)

    async def prefetch(self):
        text = self.provider.get_conversation_text()
        ids = [
            uri.rsplit("/", 2)[-2] for uri in self.write.read_files if uri.endswith("/memory.md")
        ]
        return {
            "kind": "relatedMemoryQuestions",
            "questions": await self.search(text[-12000:], ids),
            "inputSources": [
                {"sourceRef": ref, "sourceVersion": value.get("sourceVersion")}
                for ref, values in self.sources.items()
                for value in values
            ],
        }

    async def execute(self, name, args):
        self._calls += 1
        if self._calls > 12:
            return {"error": "Question read budget exhausted"}
        if name == "searchQuestions":
            return {
                "questions": await self.search(
                    args.get("query", ""),
                    args.get("subjectIds", []),
                    args.get("includeResolved", False),
                    args.get("limit", 10),
                )
            }
        try:
            item = await self.store.get(args["questionId"])
        except NotFoundError:
            return {"error": "Question not found for this user"}
        page = await self.store._load_page(item["questionUri"])
        self.write.read_files[page.uri] = page
        self.provider.register_memory_read(page.uri, page)
        return item

    def route(self, operation):
        self.provider.get_memory_write_context()
        self.service.route(operation)

    def validate(self, operations):
        self.provider.get_memory_write_context()
        self.service.validate(operations)
