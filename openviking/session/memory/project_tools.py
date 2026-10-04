"""Project tool definitions shared by ordinary and task-specific extraction."""

from openviking.session.memory.tools import MemoryTool, register_tool


class ProjectTool(MemoryTool):
    def __init__(self, name, description, properties, required):
        self._name, self._description = name, description
        self._parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
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
        raise RuntimeError("Project tools require an extraction context")


register_tool(
    ProjectTool(
        "listProjects",
        "List canonical Projects and their stable IDs; continue using nextCursor.",
        {"cursor": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        [],
    )
)
register_tool(
    ProjectTool(
        "ensureProject",
        "Create a stable anchor for a supported ongoing undertaking. Search/list existing Projects first and reuse matches. Read the resulting file before proposing narrative updates.",
        {"name": {"type": "string", "minLength": 1, "maxLength": 200}},
        ["name"],
    )
)
