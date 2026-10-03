"""Canonical person layout shared by all source providers and identity guards."""

import re


def person_anchor_from_uri(uri, root_uri):
    """Accept only a same-owner narrative, never profile.json or questions.md."""
    if not isinstance(uri, str):
        return None
    match = re.fullmatch(re.escape(root_uri) + r"people/([A-Za-z0-9_-]+)/memory\.md", uri)
    return match.group(1) if match else None
