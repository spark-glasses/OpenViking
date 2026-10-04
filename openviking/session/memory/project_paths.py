"""Stable, owner-scoped Project paths. Display names are never path keys."""

import re
from uuid import UUID

from openviking.core.namespace import user_space_fragment


def project_directory_uri(ctx, identifier):
    return f"viking://user/{user_space_fragment(ctx)}/memories/projects/{UUID(str(identifier))}"


def project_metadata_uri(ctx, identifier):
    return project_directory_uri(ctx, identifier) + "/project.json"


def project_uri(ctx, identifier):
    return project_directory_uri(ctx, identifier) + "/memory.md"


def project_questions_uri(ctx, identifier):
    return project_directory_uri(ctx, identifier) + "/questions.md"


def project_id_from_uri(uri, root_uri):
    if not isinstance(uri, str):
        return None
    match = re.fullmatch(re.escape(root_uri) + r"projects/([0-9a-f-]{36})/memory\.md", uri)
    if not match:
        return None
    try:
        return str(UUID(match.group(1)))
    except ValueError:
        return None
