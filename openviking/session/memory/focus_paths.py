"""Stable user-owned Focus paths; display names never determine identity."""

import re
from uuid import UUID

from openviking.core.namespace import user_space_fragment

FOCUS_RECORD_FILE = "focus.json"
FOCUS_PAGE_FILE = "memory.md"


def focus_directory_uri(ctx, identifier):
    return f"viking://user/{user_space_fragment(ctx)}/memories/focuses/{UUID(str(identifier))}"


def focus_metadata_uri(ctx, identifier):
    return focus_directory_uri(ctx, identifier) + "/" + FOCUS_RECORD_FILE


def focus_uri(ctx, identifier):
    return focus_directory_uri(ctx, identifier) + "/" + FOCUS_PAGE_FILE


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


def focus_file_from_uri(uri, root_uri):
    """The folder and file a URI names when it is a Focus's record or page.

    The folder need not be a Focus yet: writing the record of a Focus that
    does not exist is how one is created.
    """
    if not isinstance(uri, str):
        return None
    match = re.fullmatch(
        re.escape(root_uri)
        + r"focuses/([^/]+)/("
        + re.escape(FOCUS_RECORD_FILE)
        + "|"
        + re.escape(FOCUS_PAGE_FILE)
        + ")",
        uri,
    )
    return (match.group(1), match.group(2)) if match else None
