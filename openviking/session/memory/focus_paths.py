"""Stable user-owned Focus paths; display names never determine identity."""

import re
from uuid import UUID

from openviking.core.namespace import user_space_fragment


def focus_directory_uri(ctx, identifier):
    return f"viking://user/{user_space_fragment(ctx)}/memories/focuses/{UUID(str(identifier))}"


def focus_metadata_uri(ctx, identifier):
    return focus_directory_uri(ctx, identifier) + "/focus.json"


def focus_uri(ctx, identifier):
    return focus_directory_uri(ctx, identifier) + "/memory.md"


def focus_questions_uri(ctx, identifier):
    return focus_directory_uri(ctx, identifier) + "/questions.md"


def focus_id_from_uri(uri, root_uri):
    if not isinstance(uri, str):
        return None
    match = re.fullmatch(re.escape(root_uri) + r"focuses/([0-9a-f-]{36})/memory\.md", uri)
    if not match:
        return None
    try:
        return str(UUID(match.group(1)))
    except ValueError:
        return None
