"""Known contacts stay on their canonical People pages during ordinary extraction."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from openviking.message import Message, TextPart
from openviking.models.vlm.base import ToolCall
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.canonical_people import MAX_PERSON_CANDIDATES, CanonicalPeople
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import ExtractContext
from openviking.session.memory.person_identity import identity_directory_uri, person_aliases
from openviking.session.memory.session_extract_context_provider import SessionExtractContextProvider
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig

ROOT = "viking://user/alice/memories/"


def person(number, display, **profile):
    anchor = str(UUID(int=number))
    profile = {"displayName": display, **profile}
    return {
        "personId": f"contact-{number}",
        "anchorId": anchor,
        "memoryUri": ROOT + f"people/{anchor}/memory.md",
        "revision": 1,
        "displayName": display,
        "aliases": person_aliases(profile),
        "deleted": False,
    }


class FS:
    agfs = None

    def __init__(self):
        self.files = {}
        self.reads = []

    async def read_file(self, uri, **kwargs):
        self.reads.append(uri)
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content

    async def search(self, query, **kwargs):
        return SimpleNamespace(to_dict=lambda: {"memories": []})

    async def ls(self, uri, **kwargs):
        return []

    async def tree(self, uri, **kwargs):
        return [{"uri": key} for key in self.files if key.startswith(uri + "/")]


@pytest.fixture
def env(monkeypatch):
    config = SimpleNamespace(memory=MemoryConfig(), vlm=SimpleNamespace(is_available=lambda: False))
    for module in (
        "openviking_cli.utils.config",
        "openviking.session.memory.session_extract_context_provider",
        "openviking.session.memory.extract_loop",
        "openviking.session.compressor_v2",
    ):
        monkeypatch.setattr(module + ".get_openviking_config", lambda: config)
    fs = FS()
    for module in (
        "openviking.session.compressor_v2",
        "openviking.session.memory.memory_updater",
        "openviking.storage.viking_fs",
    ):
        monkeypatch.setattr(module + ".get_viking_fs", lambda: fs)
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    people = [
        person(1, "Sungryull Sohn", givenName="Sungryull", familyName="Sohn"),
        person(2, "禧 贺", givenName="禧", familyName="贺"),
        person(3, "Unmentioned Person"),
    ]

    def seed(records):
        fs.files[identity_directory_uri(ctx)] = json.dumps(
            {
                "formatVersion": 1,
                "userId": ctx.user.user_id,
                "accountId": ctx.account_id,
                "people": {p["anchorId"]: p for p in records},
                "contacts": {p["personId"]: p["anchorId"] for p in records},
            }
        )
        for p in records:
            fs.files[p["memoryUri"]] = MemoryFileUtils.write(
                MemoryFile(
                    uri=p["memoryUri"],
                    memory_type="people",
                    content=f"# {p['displayName']}\n",
                    extra_fields={"anchorId": p["anchorId"]},
                )
            )

    def provider(text):
        messages = [Message(id="original-1", role="user", parts=[TextPart(text)])]
        isolation = MemoryIsolationHandler(ctx, ExtractContext(messages))
        return SessionExtractContextProvider(
            messages=messages,
            ctx=ctx,
            viking_fs=fs,
            isolation_handler=isolation,
        )

    seed(people)
    return SimpleNamespace(
        config=config, fs=fs, ctx=ctx, people=people, seed=seed, provider=provider
    )


def ops(kind, uri, **fields):
    return ResolvedOperations(
        upsert_operations=[ResolvedOperation(memory_type=kind, uris=[uri], memory_fields=fields)],
        delete_file_contents=[],
        errors=[],
    )


def test_exact_names_chinese_compact_latin_boundaries_and_ambiguity():
    records = [
        person(1, "Sungryull Sohn", givenName="Sungryull", familyName="Sohn"),
        person(2, "禧 贺", givenName="禧", familyName="贺"),
        person(3, "Bob Smith", givenName="Bob", familyName="Smith"),
        person(4, "Bob Jones", givenName="Bob", familyName="Jones"),
    ]
    directory = CanonicalPeople(records)
    assert directory.unambiguous("Sungryull喜欢陶艺，贺禧在学大提琴") == {
        records[0]["memoryUri"],
        records[1]["memoryUri"],
    }
    assert not directory.matching("Bobby has an anniversary，恭禧")
    assert directory.unambiguous("Bob likes pottery") == set()
    assert len(directory.candidates("Bob likes pottery")["people"]) == 2
    assert directory.unambiguous("Bob Jones likes pottery") == {records[3]["memoryUri"]}


@pytest.mark.asyncio
async def test_casual_conversation_discovers_people_without_save_memory_or_embedding(env):
    provider = env.provider("Sungryull现在每周六做陶艺，贺禧每周三学大提琴。")
    await provider.prepare_extraction_messages()
    results = await provider.prefetch()
    reads = {
        json.loads(m["content"])["args"]["uri"]
        for m in results
        if m["content"].startswith("{") and json.loads(m["content"]).get("tool_call_name") == "read"
    }
    assert {p["memoryUri"] for p in env.people[:2]} <= reads
    assert env.people[2]["memoryUri"] not in reads
    assert set(provider.get_tools()) == {"read", "search"}
    assert MemoryTypeRegistry().get("people").filename_template == "{{ anchorId }}/memory.md"


@pytest.mark.asyncio
async def test_identity_context_is_bounded_and_remaining_people_searchable(env):
    records = [person(i + 10, f"Person {i:03}") for i in range(35)]
    env.seed(records)
    provider = env.provider("; ".join(p["displayName"] for p in records))
    await provider.prepare_extraction_messages()
    results = await provider.prefetch()
    lookup = next(
        json.loads(m["content"])["result"]
        for m in results
        if m["content"].startswith("{")
        and isinstance(json.loads(m["content"]).get("result"), dict)
        and "people" in json.loads(m["content"])["result"]
    )
    assert len(lookup["people"]) == MAX_PERSON_CANDIDATES
    assert lookup["totalMatches"] == 35
    assert lookup["truncated"] is True
    assert len(set(provider.read_file_contents) & {p["memoryUri"] for p in records}) == 5
    last = await provider.execute_tool(ToolCall("lookup", "search", {"query": "Person 034"}))
    assert last["contactCandidates"]["people"][0]["uri"] == records[-1]["memoryUri"]


@pytest.mark.asyncio
async def test_known_person_entities_rejected_but_same_name_project_allowed(env):
    provider = env.provider("Sungryull最近开始陶艺")
    await provider.prepare_extraction_messages()
    await provider.prefetch()
    duplicate = ops(
        "entities", ROOT + "entities/人物/Sungryull.md", name="Sungryull", category="人物"
    )
    with pytest.raises(ValueError):
        provider.validate_canonical_people_operations(duplicate)
    project = ops(
        "entities", ROOT + "entities/project/Sungryull.md", name="Sungryull", category="project"
    )
    provider.validate_canonical_people_operations(project)
    canonical = ops("people", env.people[0]["memoryUri"], content="New fact")
    provider.validate_canonical_people_operations(canonical)
    provider.validate_operations(canonical)
    assert canonical.upsert_operations[0].memory_fields["anchorId"] == env.people[0]["anchorId"]


@pytest.mark.parametrize(
    "name",
    [
        "Bob Smith (Acme)",
        "Bob Smith — founder",
        "bob_smith_founder",
        "Bob (founder)",
        "贺禧（大提琴学生）",
    ],
)
def test_decorated_person_entity_names_cannot_bypass_existing_identity(name):
    records = [
        person(5, "Bob Smith", givenName="Bob", familyName="Smith"),
        person(6, "Bob Jones", givenName="Bob", familyName="Jones"),
        person(7, "禧 贺", givenName="禧", familyName="贺"),
    ]
    directory = CanonicalPeople(records)
    proposed = ops("entities", ROOT + "entities/people/duplicate.md", name=name, category="people")
    with pytest.raises(ValueError):
        directory.validate(proposed, ROOT)
    # Rejection must not silently route the operation to either same-name person.
    assert proposed.upsert_operations[0].uris == [ROOT + "entities/people/duplicate.md"]
    assert directory.unambiguous("Bob (founder)") == set()


def test_person_title_matching_preserves_boundaries_and_nonperson_collisions():
    directory = CanonicalPeople([person(5, "Bob Smith", givenName="Bob", familyName="Smith")])
    for name, category in [
        ("Bobby (Acme)", "people"),
        ("Bob Smith (Acme)", "company"),
        ("Bob Smith — founder", "project"),
    ]:
        directory.validate(
            ops("entities", ROOT + "entities/other/card.md", name=name, category=category), ROOT
        )


@pytest.mark.asyncio
async def test_ambiguous_person_is_not_selected_even_if_model_reads_one_candidate(env):
    records = [
        person(5, "Bob Smith", givenName="Bob", familyName="Smith"),
        person(6, "Bob Jones", givenName="Bob", familyName="Jones"),
    ]
    env.seed(records)
    provider = env.provider("Bob plays cello")
    await provider.prepare_extraction_messages()
    await provider.prefetch()
    assert not provider.read_file_contents
    await provider.read_file(records[0]["memoryUri"])
    with pytest.raises(ValueError):
        provider.validate_operations(ops("people", records[0]["memoryUri"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,uri,fields",
    [
        ("people", ROOT + "people/invented/memory.md", {}),
        ("people", "viking://user/other/memories/people/other/memory.md", {}),
        ("people", ROOT + "people/../entities/person.md", {}),
        ("entities", ROOT + "people/00000000-0000-0000-0000-000000000001/memory.md", {}),
        ("people", ROOT + "people/00000000-0000-0000-0000-000000000001/memory.md", {"anchorId": "wrong"}),
    ],
)
async def test_invalid_or_invented_anchor_cannot_be_written(env, kind, uri, fields):
    provider = env.provider("Sungryull plays cello")
    await provider.prepare_extraction_messages()
    await provider.prefetch()
    with pytest.raises(ValueError):
        proposed = ops(kind, uri, **fields)
        provider.validate_canonical_people_operations(proposed)
        provider.validate_operations(proposed)


@pytest.mark.asyncio
async def test_native_loop_repairs_duplicate_person_to_existing_anchor(env):
    person_record = env.people[0]
    provider = env.provider("Sungryull now studies pottery on Saturdays.")
    entity = json.dumps(
        {
            "entities": [
                {
                    "page_id": 100,
                    "category": "people",
                    "name": "Sungryull",
                    "content": "Sungryull studies pottery on Saturdays.",
                }
            ]
        }
    )
    canonical = json.dumps(
        {
            "people": [
                {
                    "page_id": 100,
                    "anchorId": person_record["anchorId"],
                    "content": {
                        "blocks": [
                            {
                                "search": "# Sungryull Sohn",
                                "replace": "# Sungryull Sohn\nStudies pottery on Saturdays.",
                            }
                        ]
                    },
                }
            ]
        }
    )
    model = SimpleNamespace(
        model="test-model", get_completion_async=AsyncMock(side_effect=[entity, canonical])
    )
    env.config.vlm.get_vlm_instance = lambda: model
    await SessionCompressorV2(vikingdb=None).extract_long_term_memories(
        messages=provider.messages,
        ctx=env.ctx,
        strict_extract_errors=True,
    )
    saved = MemoryFileUtils.read(env.fs.files[person_record["memoryUri"]])
    assert "Studies pottery on Saturdays." in saved.content
    assert model.get_completion_async.await_count == 2
    assert not any(uri.startswith(ROOT + "entities/") for uri in env.fs.files)
