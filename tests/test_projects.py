"""Project identity and conditional writes are filesystem contracts, not prompts."""

import asyncio
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import MemoryField
from openviking.session.memory.project_store import ProjectStore
from openviking_cli.exceptions import AlreadyExistsError, InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from tests.test_question_store import FS


@pytest.fixture
def store():
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    return ProjectStore(FS(), ctx)


@pytest.mark.asyncio
async def test_workspace_identity_survives_retry_and_name_changes(store):
    first = await store.ensure(
        name="Workspace", create_key="one", source={"provider": "slack", "workspaceId": "T1"}
    )
    second = await store.ensure(
        name="New name", create_key="two", source={"provider": "slack", "workspaceId": "T1"}
    )
    assert first == second
    updated = await store.update(
        first["projectId"], expected_revision=1, fields={"name": "Spark", "content": "Product work"}
    )
    assert updated["revision"] == 2
    assert updated["sourceBindings"] == first["sourceBindings"]
    listing = await store.list()
    assert listing["projects"] == [{k: v for k, v in updated.items() if k != "content"}]


@pytest.mark.asyncio
async def test_conditional_writes_do_not_overwrite_concurrent_changes(store):
    page = await store.ensure(name="Study", create_key="study")
    results = await asyncio.gather(
        *[
            store.update(page["projectId"], expected_revision=1, fields={"name": name})
            for name in ("A", "B")
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(result, AlreadyExistsError) for result in results) == 1
    assert (await store.get(page["projectId"])).extra_fields["revision"] == 2


@pytest.mark.asyncio
async def test_catalog_paginates_and_never_crosses_owner(store):
    pages = [await store.ensure(name=str(i), create_key=str(i)) for i in range(3)]
    first = await store.list(limit=2)
    second = await store.list(after=first["nextCursor"], limit=2)
    assert {p["projectId"] for p in first["projects"] + second["projects"]} == {
        p["projectId"] for p in pages
    }
    assert second["nextCursor"] is None
    other = ProjectStore(
        store.fs, RequestContext(user=UserIdentifier("account", "bob"), role=Role.ROOT)
    )
    assert (await other.list())["projects"] == []
    with pytest.raises(NotFoundError):
        await other.get(pages[0]["projectId"])


@pytest.mark.asyncio
async def test_protected_identity_and_native_target_cannot_be_rewritten(store):
    page = await store.ensure(name="Memory", create_key="memory")
    with pytest.raises(InvalidArgumentError):
        await store.update(page["projectId"], expected_revision=1, fields={"sourceBindings": []})
    op = SimpleNamespace(
        old_memory_file_content=await store.get(page["projectId"]),
        uris=[page["uri"].replace("alice", "bob")],
        memory_fields={"projectId": page["projectId"]},
    )
    with pytest.raises(InvalidArgumentError):
        await store.apply_native(op, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new_name,expected", [(None, "Memory"), ("", "Memory"), ("Spark Memory", "Spark Memory")]
)
async def test_native_update_uses_replace_semantics_for_optional_rename(store, new_name, expected):
    page = await store.ensure(name="Memory", create_key="memory")
    schema = SimpleNamespace(
        fields=[
            MemoryField(name="name", field_type="string", merge_op="replace"),
            MemoryField(name="content", field_type="string", merge_op="replace"),
        ]
    )
    op = SimpleNamespace(
        old_memory_file_content=await store.get(page["projectId"]),
        uris=[page["uri"]],
        memory_fields={
            "projectId": page["projectId"],
            "name": new_name,
            "content": "New supported progress",
        },
    )
    await store.apply_native(op, schema)
    result = store.view(await store.get(page["projectId"]))
    assert result["name"] == expected
    assert result["content"] == "New supported progress"
    assert result["revision"] == 2
