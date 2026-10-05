import json
import pytest

from openviking.session.memory.focus_migration import (
    plan_migration,
    plan_lifecycle_migration,
    apply_migration,
)
from openviking.session.memory.focus_store import FocusStore
from openviking.session.memory.project_store import ProjectStore
from openviking.session.memory.question_store import QuestionStore, question_uri
from tests.test_question_store import proposal

pytest_plugins = ["tests.test_focus_store"]


@pytest.mark.asyncio
async def test_migration_preserves_questions_and_moves_only_reviewed_priority(store, tmp_path):
    old = ProjectStore(store.fs, store.ctx)
    workspace = await old.ensure(
        name="Company Slack",
        create_key="workspace",
        source={"provider": "slack", "workspaceId": "T1"},
    )
    health = await old.ensure(name="Be healthier", create_key="health")
    qs = QuestionStore(store.fs, store.ctx)
    subject = {"kind": "project", "id": workspace["projectId"], "memoryUri": workspace["uri"]}
    q = (await qs.discover(question_uri(store.ctx, subject), subject, [proposal()]))[0]
    with pytest.raises(ValueError, match="classification"):
        await plan_migration(store.fs, store.ctx, {})
    plan = await plan_migration(
        store.fs,
        store.ctx,
        {
            workspace["projectId"]: {"kind": "entity"},
            health["projectId"]: {"kind": "focus", "userIntent": "Have more energy"},
        },
    )
    await apply_migration(store.fs, store.ctx, plan, tmp_path / "focus-migration.json")
    await apply_migration(store.fs, store.ctx, plan, tmp_path / "focus-migration.json")
    focuses = await FocusStore(store.fs, store.ctx).list()
    assert [f["name"] for f in focuses["focuses"]] == ["Be healthier"]
    assert focuses["focuses"][0]["focusId"] == health["projectId"]
    actual = await qs.get(q["questionId"])
    assert actual["subject"]["kind"] == "matter"
    assert actual["state"] == q["state"] and actual["events"] == q["events"]
    assert actual["answers"] == q["answers"]
    assert workspace["uri"] not in store.fs.files
    assert "T1" in store.fs.files[plan["moves"][workspace["uri"]]]


@pytest.mark.asyncio
async def test_migration_refuses_newer_concurrent_data_before_any_write(store, tmp_path):
    old = await ProjectStore(store.fs, store.ctx).ensure(name="Work", create_key="work")
    plan = await plan_migration(store.fs, store.ctx, {old["projectId"]: {"kind": "entity"}})
    store.fs.files[old["uri"]] += "\nNew concurrent content"
    before = dict(store.fs.files)
    with pytest.raises(ValueError, match="changed"):
        await apply_migration(store.fs, store.ctx, plan, tmp_path / "changed.json")
    assert before == store.fs.files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "visibility,previous,expected",
    [
        ("visible", "active", "active"),
        ("forming", "active", "forming"),
        ("hidden", "active", "archived"),
        ("forming", "archived", "archived"),
        ("visible", "archived", "archived"),
    ],
)
async def test_lifecycle_migration_preserves_memory_and_is_resumable(
    store, tmp_path, visibility, previous, expected
):
    focus = await store.ensure(name="Study", create_key="study", origin="user")
    metadata = json.loads(store.fs.files[focus["metadataUri"]])
    metadata.update(
        visibility=visibility,
        status=previous,
        protectedFields=["name", "visibility"],
        lastOperationId="legacy",
        lastOperationHash="old",
    )
    store.fs.files[focus["metadataUri"]] = json.dumps(metadata)
    before = dict(store.fs.files)
    plan = await plan_lifecycle_migration(store.fs, store.ctx)
    assert store.fs.files == before
    assert [f["uri"] for f in plan["files"]] == [focus["metadataUri"]]
    journal = tmp_path / "lifecycle.json"
    await apply_migration(store.fs, store.ctx, plan, journal)
    await apply_migration(store.fs, store.ctx, plan, journal)
    result = json.loads(store.fs.files[focus["metadataUri"]])
    assert result["status"] == expected
    assert result["revision"] == metadata["revision"] + 1
    assert result["focusId"] == focus["focusId"]
    assert "visibility" not in result and "lastOperationId" not in result
    assert set(result["protectedFields"]) == {"name", "status"}
    assert store.fs.files[focus["uri"]] == before[focus["uri"]]
    assert (await plan_lifecycle_migration(store.fs, store.ctx))["files"] == []


@pytest.mark.asyncio
async def test_lifecycle_migration_refuses_pending_writes(store):
    focus = await store.ensure(name="Study", create_key="study", origin="user")
    store.fs.files[focus["directoryUri"] + "/.pending-write.json"] = "{}"
    with pytest.raises(ValueError, match="pending"):
        await plan_lifecycle_migration(store.fs, store.ctx)
