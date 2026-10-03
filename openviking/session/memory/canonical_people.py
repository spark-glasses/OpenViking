"""Deterministic discovery and routing of application-owned person anchors."""

import re

from openviking.session.memory.person_identity import normalize_person_alias

MAX_PERSON_CANDIDATES = 20
MAX_PERSON_PREFETCH_READS = 5
_PERSON_CATEGORIES = {
    "person",
    "people",
    "persons",
    "human",
    "individual",
    "contact",
    "contacts",
    "friend",
    "friends",
    "family",
    "colleague",
    "colleagues",
    "人物",
    "人",
    "个人",
    "联系人",
    "朋友",
    "同事",
    "家人",
    "亲友",
    "人脉",
}


def _alias_key(value):
    return normalize_person_alias(str(value or ""))


def _contains_alias(text, alias):
    # Chinese names occur without spaces inside ordinary prose. Latin names need
    # token boundaries: Ann must not match anniversary, nor Bob match Bobby.
    if len(alias) < 2:
        return False
    left = r"(?<![a-z0-9_])" if alias[0].isascii() and alias[0].isalnum() else ""
    right = r"(?![a-z0-9_])" if alias[-1].isascii() and alias[-1].isalnum() else ""
    return re.search(left + re.escape(alias) + right, text) is not None


class CanonicalPeople:
    def __init__(self, records):
        self.records = {p["memoryUri"]: p for p in records if not p.get("deleted")}
        self.aliases = {}
        for uri, record in self.records.items():
            for alias in record.get("aliases", []) + [record.get("displayName", "")]:
                key = _alias_key(alias)
                if len(key) >= 2:
                    self.aliases.setdefault(key, set()).add(uri)

    def matching(self, text):
        text = _alias_key(text)
        matched = {}
        for alias, uris in self.aliases.items():
            if _contains_alias(text, alias):
                for uri in uris:
                    matched.setdefault(uri, set()).add(alias)
        return matched

    def exact(self, value):
        # The native entity schema may render an English name with underscores.
        key = _alias_key(value)
        return self.aliases.get(key, set()) | self.aliases.get(key.replace("_", " "), set())

    def unambiguous(self, text):
        matched = self.matching(text)
        return {
            uri
            for uri, aliases in matched.items()
            if any(len(self.aliases[alias]) == 1 for alias in aliases)
        }

    def candidates(self, text, limit=MAX_PERSON_CANDIDATES):
        matched = self.matching(text)
        unique = self.unambiguous(text)
        candidates = []
        for uri in sorted(matched, key=lambda uri: (uri not in unique, uri))[:limit]:
            person = self.records[uri]
            profile = person.get("profile") or {}
            candidates.append(
                {
                    "uri": uri,
                    "memoryType": "people",
                    "anchorId": person["anchorId"],
                    "personId": person["personId"],
                    "displayName": person.get("displayName", ""),
                    "matchedAliases": sorted(matched[uri]),
                    "identityAmbiguous": uri not in unique,
                    "organization": profile.get("organization"),
                    "jobTitle": profile.get("jobTitle"),
                }
            )
        return {
            "people": candidates,
            "totalMatches": len(matched),
            "truncated": len(matched) > limit,
        }

    def validate(self, operations, root_uri):
        """Common guard; task providers still apply their narrower permissions."""
        for operation in operations.upsert_operations:
            for uri in operation.uris:
                in_people = uri.startswith(root_uri + "people/")
                if operation.memory_type == "people" or in_people:
                    match = re.fullmatch(re.escape(root_uri) + r"people/([A-Za-z0-9_-]+)/memory\.md", uri)
                    if not match or operation.memory_type != "people":
                        raise ValueError(
                            "Person writes require memory_type=people and a canonical person URI"
                        )
                    anchor = match.group(1)
                    if operation.memory_fields.get("anchorId") not in (None, anchor):
                        raise ValueError("Person anchorId conflicts with its canonical URI")
                    operation.memory_fields["anchorId"] = anchor
                is_person_entity = (
                    _alias_key(operation.memory_fields.get("category")) in _PERSON_CATEGORIES
                    or bool(operation.memory_fields.get("personId"))
                    or bool(operation.memory_fields.get("anchorId"))
                )
                if operation.memory_type == "entities" and is_person_entity:
                    name = operation.memory_fields.get("name", "")
                    # A title or company suffix does not create a new identity.
                    # Keep the same token-boundary matching as source discovery,
                    # and retain every candidate when a short name is ambiguous.
                    matches = self.exact(name) | set(self.matching(str(name).replace("_", " ")))
                    if matches:
                        targets = ", ".join(sorted(matches)[:MAX_PERSON_CANDIDATES])
                        raise ValueError(
                            f"{name!r} matches existing contact identities. Do not create or update "
                            f"a duplicate entity person. Read and use memory_type=people at {targets}. "
                            "If these candidates remain ambiguous, omit person-specific writes."
                        )
        for memory in operations.delete_file_contents:
            if (
                memory.uri in self.records
                or memory.extra_fields.get("contact_projection_revision") is not None
            ):
                raise ValueError("Application-owned person anchors cannot be deleted by extraction")
