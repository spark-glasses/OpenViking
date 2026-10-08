"""Deterministic contact projections and protected identity sections in People.

The JSON record is the contact projection authority; the Markdown section is a
readable view. Narrative memory remains owned by native extraction. No model is
called to create or refresh these identity records.
"""

from __future__ import annotations

import json
import re
import unicodedata
from uuid import UUID

from openviking.session.memory.question_store import memory_root
from openviking.session.memory.person_paths import person_anchor_from_uri
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError

CONTACT_SECTION_START = "<!-- SPARK_CONTACT_PROFILE_START -->"
CONTACT_SECTION_END = "<!-- SPARK_CONTACT_PROFILE_END -->"
_SECTION_PATTERN = re.compile(
    re.escape(CONTACT_SECTION_START) + r".*?" + re.escape(CONTACT_SECTION_END) + r"\n*",
    re.DOTALL,
)
MAX_PEOPLE = 10000
MAX_DIRECTORY_BYTES = 8000000


def normalize_person_alias(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def person_aliases(profile):
    """Exact name variants only; never infer a translated name or identity."""
    given, middle, family = (
        profile.get(key) or "" for key in ("givenName", "middleName", "familyName")
    )
    supplied = [
        profile.get("displayName"),
        profile.get("givenName"),
        profile.get("nickname"),
        *profile.get("aliases", []),
    ]
    for parts in (
        [given, middle, family],
        [given, family],
        [family, given],
        [family, given, middle],
    ):
        present = [part.strip() for part in parts if part and part.strip()]
        if len(present) >= 2:
            supplied.extend([" ".join(present), "".join(present)])
    values, seen = [], set()
    for value in supplied:
        if not value or not value.strip():
            continue
        normalized = normalize_person_alias(value)
        if normalized not in seen:
            seen.add(normalized)
            values.append(value.strip())
    return values


def person_memory_uri(ctx, anchor_id):
    return memory_root(ctx) + "people/" + str(UUID(str(anchor_id))) + "/memory.md"


def identity_root(ctx):
    return memory_root(ctx) + "people/"


def identity_uri(ctx, anchor_id):
    return identity_root(ctx) + str(UUID(str(anchor_id))) + "/profile.json"


def identity_directory_uri(ctx):
    return identity_root(ctx) + ".directory.json"


def _validate_owner(record, ctx):
    if (
        record.get("formatVersion") != 1
        or record.get("userId") != ctx.user.user_id
        or record.get("accountId") != ctx.account_id
    ):
        raise InvalidArgumentError("Contact identity projection has invalid owner or format")


async def load_person_identity(viking_fs, ctx, anchor_id):
    try:
        record = json.loads(await viking_fs.read_file(identity_uri(ctx, anchor_id), ctx=ctx))
    except NotFoundError:
        return None
    _validate_owner(record, ctx)
    if record.get("anchorId") != str(UUID(str(anchor_id))) or record.get(
        "memoryUri"
    ) != person_memory_uri(ctx, anchor_id):
        raise InvalidArgumentError("Contact projection anchor does not match its location")
    return record


async def load_identity_directory(viking_fs, ctx):
    try:
        raw = await viking_fs.read_file(identity_directory_uri(ctx), ctx=ctx)
        if len(raw.encode()) > MAX_DIRECTORY_BYTES:
            raise InvalidArgumentError("Contact identity directory exceeds its byte limit")
        directory = json.loads(raw)
    except NotFoundError:
        return {
            "formatVersion": 1,
            "userId": ctx.user.user_id,
            "accountId": ctx.account_id,
            "people": {},
            "contacts": {},
        }
    _validate_owner(directory, ctx)
    if (
        not isinstance(directory.get("people"), dict)
        or not isinstance(directory.get("contacts"), dict)
        or not isinstance(directory.get("retired", {}), dict)
        or len(directory["people"]) > MAX_PEOPLE
    ):
        raise InvalidArgumentError("Invalid or oversized contact identity directory")
    return directory


async def retired_into(viking_fs, ctx, anchor_id):
    """The person who stands for a merged-away one, or None for anyone else."""
    directory = await load_identity_directory(viking_fs, ctx)
    return directory.get("retired", {}).get(str(UUID(str(anchor_id))))


async def load_active_people(viking_fs, ctx):
    """Return compact deterministic identities; caller selects what reaches a model."""
    directory = await load_identity_directory(viking_fs, ctx)
    result = []
    for anchor, entry in directory["people"].items():
        if entry.get("anchorId") != anchor or entry.get("memoryUri") != person_memory_uri(
            ctx, anchor
        ):
            raise InvalidArgumentError("Contact directory contains a mismatched anchor")
        if not entry.get("deleted"):
            result.append(dict(entry))
    return sorted(
        result, key=lambda item: (normalize_person_alias(item["displayName"]), item["anchorId"])
    )


def compact_identity(record):
    profile = record["profile"]
    result = {
        key: record[key]
        for key in (
            "personId",
            "anchorId",
            "memoryUri",
            "revision",
            "displayName",
            "aliases",
            "deleted",
        )
    }
    result["confirmationStatus"] = profile.get("confirmationStatus", "confirmed")
    result["artifactId"] = profile.get("artifactId")
    result["emails"] = [email["address"] for email in profile["emails"]]
    result["phones"] = [phone["number"] for phone in profile["phones"]]
    return result


# The basic information the page shows, in the order it is shown.
_SHOWN_PROFILE_FIELDS = (
    "displayName",
    "givenName",
    "middleName",
    "familyName",
    "namePrefix",
    "nameSuffix",
    "nickname",
    "organization",
    "departmentName",
    "jobTitle",
    "contactType",
    "birthday",
    "phones",
    "emails",
    "postalAddresses",
    "urlAddresses",
    "socialProfiles",
    "instantMessageAddresses",
    "dates",
    "relations",
    "notes",
)
# Lists every page shows, empty or not; the others appear once they hold something.
_ALWAYS_SHOWN_LISTS = {"phones": "number", "emails": "address"}
_MAX_SHOWN_ITEMS = 32


def model_contact_profile(identity):
    """Bound the readable projection; full imported values remain in the sidecar."""
    source = identity["profile"]
    profile = {
        key: source[key]
        for key in _SHOWN_PROFILE_FIELDS
        if key in source and (source[key] != [] or key in _ALWAYS_SHOWN_LISTS)
    }
    truncated = {}
    if len(profile.get("notes") or "") > 4000:
        truncated["notes"] = {"totalChars": len(profile["notes"]), "shownChars": 4000}
        profile["notes"] = profile["notes"][:4000]
    for key, original in list(profile.items()):
        if not isinstance(original, list):
            continue
        value_field = _ALWAYS_SHOWN_LISTS.get(key)
        shown = [
            {**item, value_field: item[value_field][:1000]} if value_field else item
            for item in original[:_MAX_SHOWN_ITEMS]
        ]
        if shown != original:
            truncated[key] = {"totalItems": len(original), "shownItems": len(shown)}
            if value_field:
                truncated[key]["maxShownItemChars"] = 1000
        profile[key] = shown
    if truncated:
        profile["projectionTruncated"] = truncated
        profile["fullProfileUri"] = identity["memoryUri"].rsplit("/", 1)[0] + "/profile.json"
    return profile


def is_person_identity_uri(uri, ctx):
    if not isinstance(uri, str) or not uri.startswith(identity_root(ctx)):
        return False
    name = uri[len(identity_root(ctx)) :]
    try:
        return name.endswith("/profile.json") and name == str(UUID(name.removesuffix("/profile.json"))) + "/profile.json"
    except ValueError:
        return False


def render_contact_section(identity):
    artifact_id = identity["profile"].get("artifactId")
    data = {
        "personId": identity["personId"],
        "anchorId": identity["anchorId"],
        **({"artifactId": artifact_id} if artifact_id else {}),
        "revision": identity["revision"],
        "deleted": identity["deleted"],
        **model_contact_profile(identity),
    }
    # Contact text cannot inject a managed-section delimiter or escape JSON lines.
    encoded = (
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    status = (
        "Contact removed; historical memory retained."
        if identity["deleted"]
        else "Current contact information."
    )
    return f"{CONTACT_SECTION_START}\n## Contact profile\n{status}\n\n```json\n{encoded}\n```\n{CONTACT_SECTION_END}"


def _contact_narrative(content, trusted_section):
    """Strip only delimited content or an exact known section with one marker missing."""
    starts, ends = content.count(CONTACT_SECTION_START), content.count(CONTACT_SECTION_END)
    if not starts and not ends:
        return content.strip()
    if starts == ends == 1 and content.index(CONTACT_SECTION_START) < content.index(
        CONTACT_SECTION_END
    ):
        return _SECTION_PATTERN.sub("", content).strip()
    if (starts, ends) in ((1, 0), (0, 1)):
        missing = CONTACT_SECTION_END if starts else CONTACT_SECTION_START
        incomplete = trusted_section.replace(missing, "").strip()
        if incomplete and content.count(incomplete) == 1:
            remainder = content.replace(incomplete, "", 1)
            if CONTACT_SECTION_START not in remainder and CONTACT_SECTION_END not in remainder:
                return remainder.strip()
    raise ValueError(
        "Incomplete managed contact section; preserve the entire contact section or return only narrative memory"
    )


def validate_person_identity_patch(operation, old_memory):
    """Preview the actual merge before any writes; malformed boundaries are repairable."""
    if not old_memory or "contact_projection_revision" not in old_memory.extra_fields:
        return
    if "content" not in operation.memory_fields:
        return
    from openviking.session.memory.merge_op.base import FieldType
    from openviking.session.memory.merge_op.patch import PatchOp

    original = old_memory.plain_content()
    match = _SECTION_PATTERN.search(original)
    if not match:
        raise ValueError("Existing contact identity section is incomplete; cannot safely update it")
    section = match.group(0).strip()
    proposed = PatchOp(FieldType.STRING).apply(original, operation.memory_fields["content"])
    narrative = _contact_narrative(proposed, section)
    # Reattach the immutable section now so the native updater never sees a broken
    # delimiter. It refreshes contact ownership again from the authoritative record.
    operation.memory_fields["content"] = section + ("\n\n" + narrative if narrative else "")


def apply_person_identity(memory, identity):
    """Apply a deterministic contact-owned section, preserving all other narrative."""
    section = render_contact_section(identity)
    try:
        narrative = _contact_narrative(memory.content or "", section)
    except ValueError as error:
        raise InvalidArgumentError(str(error)) from error
    memory.content = section + ("\n\n" + narrative if narrative else "")
    memory.memory_type = "people"
    memory.extra_fields.update(
        {
            "anchorId": identity["anchorId"],
            "personId": identity["personId"],
            "contact_projection_revision": identity["revision"],
            "contact_projection_deleted": identity["deleted"],
        }
    )
    return memory


async def preserve_person_identity(viking_fs, ctx, uri, memory):
    """Native updater hook. Registered contact metadata is never model writable."""
    anchor = person_anchor_from_uri(uri, memory_root(ctx))
    if anchor is None:
        return memory
    try:
        UUID(anchor)
    except ValueError:
        return memory  # Unregistered upstream OV people remain supported.
    survivor = await retired_into(viking_fs, ctx, anchor)
    if survivor:
        raise InvalidArgumentError(
            "This person was merged into another; write to "
            + person_memory_uri(ctx, survivor)
        )
    identity = await load_person_identity(viking_fs, ctx, anchor)
    if identity is None:
        if "contact_projection_revision" in memory.extra_fields:
            raise InvalidArgumentError(
                "Registered contact metadata is missing; refusing to erase its identity"
            )
        return memory
    return apply_person_identity(memory, identity)
