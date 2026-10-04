import json

import pytest

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.project_store import ProjectStore
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from tests.test_question_store import proposal

pytest_plugins = ["tests.test_projects"]


@pytest.mark.asyncio
async def test_generic_write_cannot_create_or_replace_managed_files(store):
    from openviking.storage.content_write import ContentWriteCoordinator
    from openviking_cli.exceptions import InvalidArgumentError

    value = await store.ensure(name="Spark", create_key="spark")
    before = dict(store.fs.files)
    store.fs._ensure_mutable_access = lambda *args: None
    for uri in (
        value["uri"],
        value["metadataUri"],
        value["questionsUri"],
        question_uri(store.ctx, {"kind": "self", "id": "self"}),
    ):
        for mode in ("create", "replace"):
            with pytest.raises(InvalidArgumentError):
                await ContentWriteCoordinator(store.fs).write(
                    uri=uri, content="bypass", ctx=store.ctx, mode=mode
                )
    assert store.fs.files == before


@pytest.mark.asyncio
async def test_metadata_has_one_authority_and_questions_share_folder(store):
    value = await store.ensure(name="Spark", create_key="spark")
    assert value["uri"].endswith("/memory.md")
    metadata = json.loads(store.fs.files[value["metadataUri"]])
    assert metadata["name"] == "Spark"
    assert '"revision"' not in store.fs.files[value["uri"]]
    subject = {"kind": "project", "id": value["projectId"], "memoryUri": value["uri"]}
    questions = QuestionStore(store.fs, store.ctx)
    item = (await questions.discover(question_uri(store.ctx, subject), subject, [proposal()]))[0]
    assert item["questionUri"] == value["questionsUri"]
    assert (await questions.get(item["questionId"]))["subject"] == subject


@pytest.mark.asyncio
async def test_interrupted_write_recovers_without_losing_metadata(store):
    value = await store.ensure(name="Spark", create_key="spark")
    write = store.fs.write_file

    async def fail(uri, content, **kwargs):
        if uri == value["metadataUri"]:
            raise OSError("interrupted")
        await write(uri, content, **kwargs)

    store.fs.write_file = fail
    with pytest.raises(OSError):
        await store.update(
            value["projectId"],
            expected_revision=1,
            fields={"name": "Spark Memory", "content": "Grounded progress"},
        )
    store.fs.write_file = write
    recovered = await ProjectStore(store.fs, store.ctx).get(value["projectId"])
    assert recovered.content == "Grounded progress"
    assert recovered.extra_fields["name"] == "Spark Memory"
    assert recovered.extra_fields["revision"] == 2
    assert not any(key.endswith(".pending-write.json") for key in store.fs.files)


@pytest.mark.asyncio
async def test_same_name_different_create_keys_remain_distinct(store):
    first = await store.ensure(name="Spark", create_key="one")
    second = await store.ensure(name="Spark", create_key="two")
    assert first["projectId"] != second["projectId"]
    assert len((await store.list())["projects"]) == 2


@pytest.mark.asyncio
async def test_offline_migration_preserves_id_question_history_and_relative_links(store, tmp_path):
    from uuid import NAMESPACE_URL, UUID, uuid5

    from openviking.session.memory.project_layout_migration import apply_migration, plan_migration
    from openviking.session.memory.question_layout_migration import plan_migration as question_plan

    identifier = str(UUID(int=55))
    root = "viking://user/alice/memories/"
    old = root + "projects/" + identifier + ".md"
    fields = {
        "projectId": identifier,
        "name": "Spark",
        "revision": 4,
        "status": "active",
        "aliases": [],
        "sourceBindings": [],
    }
    store.fs.files[old] = MemoryFileUtils.write(
        MemoryFile(
            uri=old,
            memory_type="projects",
            content="[Self](./" + identifier + ".md)\n[Profile](../profile.md)",
            extra_fields=fields,
        )
    )
    subject = {"kind": "matter", "id": uuid5(NAMESPACE_URL, old).hex, "memoryUri": old}
    qs = QuestionStore(store.fs, store.ctx)
    q = (await qs.discover(question_uri(store.ctx, subject), subject, [proposal()]))[0]
    # Emulate the previous on-disk format without adding runtime compatibility.
    page = MemoryFileUtils.read(store.fs.files[q["questionUri"]], uri=q["questionUri"])
    page.extra_fields["questionFormatVersion"] = 1
    page.extra_fields["questions"][0]["state"] = "dismissed"
    store.fs.files[q["questionUri"]] = MemoryFileUtils.write(page)
    plan = await plan_migration(store.fs, store.ctx)
    await apply_migration(store.fs, store.ctx, plan, tmp_path / "projects.json")
    await apply_migration(store.fs, store.ctx, plan, tmp_path / "projects.json")
    qplan = await question_plan(store.fs, store.ctx)
    await apply_migration(store.fs, store.ctx, qplan, tmp_path / "questions.json")
    project = await store.get(identifier)
    assert project.extra_fields["revision"] == 4
    assert "[Self](./memory.md)" in project.content or "[Self](memory.md)" in project.content
    assert "../../profile.md" in project.content
    assert old not in store.fs.files
    actual = await qs.get(q["questionId"])
    assert actual["subject"]["kind"] == "project"
    assert actual["questionUri"].endswith(identifier + "/questions.md")
    assert actual["asking"] == "muted" and actual["state"] == "open"


@pytest.mark.asyncio
async def test_failed_index_enqueue_keeps_project_recovery_journal(store, monkeypatch):
    from unittest.mock import AsyncMock

    from openviking.session.memory.memory_updater import MemoryUpdater

    store.db = object()
    index = AsyncMock(return_value=False)
    monkeypatch.setattr(MemoryUpdater, "refresh_file_embedding", index)
    with pytest.raises(RuntimeError, match="index enqueue"):
        await store.ensure(name="Spark", create_key="index")
    assert any(key.endswith(".pending-write.json") for key in store.fs.files)
    index.return_value = True
    page = await store.ensure(name="Spark", create_key="index")
    assert page["revision"] == 1
    assert not any(key.endswith(".pending-write.json") for key in store.fs.files)
