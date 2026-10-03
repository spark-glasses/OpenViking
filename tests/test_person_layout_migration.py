import copy
import json
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.exceptions import NotFoundError
from openviking.session.memory.person_layout_migration import plan_migration, apply_migration
from openviking.session.memory.memory_update_context import MemoryUpdateContext
from openviking.session.memory.person_paths import person_anchor_from_uri

ROOT = "viking://user/alice"
OLD = ROOT + "/memories/people/11111111-1111-4111-8111-111111111111.md"
NEW = OLD.removesuffix(".md") + "/memory.md"
CTX = RequestContext(user=UserIdentifier("default", "alice"), role=Role.ROOT)

class FS:
    def __init__(self, files):
        self.files = dict(files)
        self.fail_remove = False
    async def tree(self, root, **kwargs):
        return [{"uri": uri, "isDir": False} for uri in self.files if uri.startswith(root + "/")]
    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]
    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content
    async def rm(self, uri, **kwargs):
        if self.fail_remove:
            self.fail_remove = False
            raise OSError("crash after destination write")
        del self.files[uri]


def fixture():
    profile = ROOT + "/memories/people/.contacts/11111111-1111-4111-8111-111111111111.json"
    question = NEW.replace("memory.md", "questions.md")
    return FS({OLD: "# 贺禧\n[Project](../projects/p.md)\nemail:original\n",
        profile: json.dumps({"memoryUri": OLD, "displayName": "贺禧"}),
        ROOT + "/memories/people/.contacts/directory.json": json.dumps({"people": {"a": {"memoryUri": OLD}}}),
        ROOT + "/memories/projects/p.md": "[贺禧](../people/11111111-1111-4111-8111-111111111111.md)",
        question: json.dumps({"subject": {"memoryUri": OLD}, "questions": [{"questionId":"unchanged", "state":"resolved", "resolution":"Yes", "events":[{"action":"answered"}]}]}),
        "viking://user/bob/memories/people/other.md": "unrelated"})


@pytest.mark.asyncio
async def test_moves_profile_memory_and_preserves_questions_sources_and_links(tmp_path):
    fs = fixture()
    before = copy.deepcopy(fs.files)
    plan = await plan_migration(fs, CTX)
    assert fs.files == before  # dry run has no writes
    path = tmp_path / "backup.json"
    await apply_migration(fs, CTX, plan, path)
    assert OLD not in fs.files
    assert fs.files[NEW] == "# 贺禧\n[Project](../../projects/p.md)\nemail:original\n"
    assert json.loads(fs.files[NEW.replace("memory.md", "profile.json")])["displayName"] == "贺禧"
    assert json.loads(fs.files[NEW.replace("memory.md", "questions.md")])["questions"] == json.loads(before[NEW.replace("memory.md", "questions.md")])["questions"]
    assert fs.files["viking://user/bob/memories/people/other.md"] == "unrelated"
    assert "../people/11111111-1111-4111-8111-111111111111/memory.md" in fs.files[ROOT + "/memories/projects/p.md"]
    assert json.loads(path.read_text())["files"][0]["before"] == before[OLD]
    assert (await plan_migration(fs, CTX))["files"] == []
    await apply_migration(fs, CTX, plan, path)


@pytest.mark.asyncio
async def test_interrupted_move_resumes_from_durable_journal(tmp_path):
    fs = fixture()
    plan = await plan_migration(fs, CTX)
    fs.fail_remove = True
    path = tmp_path / "backup.json"
    with pytest.raises(OSError):
        await apply_migration(fs, CTX, plan, path)
    assert OLD in fs.files and NEW in fs.files
    await apply_migration(fs, CTX, json.loads(path.read_text()), path)
    assert OLD not in fs.files and NEW in fs.files


@pytest.mark.asyncio
async def test_collision_and_cross_user_fail_without_writes(tmp_path):
    fs = fixture()
    fs.files[NEW] = "different narrative"
    before = dict(fs.files)
    with pytest.raises(RuntimeError, match="collision"):
        await plan_migration(fs, CTX)
    assert fs.files == before
    fs.files.pop(NEW)
    plan = await plan_migration(fs, CTX)
    other = RequestContext(user=UserIdentifier("default", "bob"), role=Role.ROOT)
    with pytest.raises(ValueError, match="different owner"):
        await apply_migration(fs, other, plan, tmp_path / "journal.json")


@pytest.mark.asyncio
async def test_frozen_update_receipt_hash_tracks_migrated_context(tmp_path):
    fs = fixture()
    archive = ROOT + "/session/memory-update-test/history/archive_001"
    context = {"operationId":"test", "text":"Remember this", "origin":{"conversationId":"c", "turnId":"t", "toolCallId":"call"},
        "targets":[{"kind":"person", "personId":"p", "anchorId":"11111111-1111-4111-8111-111111111111", "memoryUri":OLD}], "messages":[], "sourceKinds":[]}
    fs.files[archive + "/memory_update_context.json"] = json.dumps(context)
    receipt = ROOT + "/session/memory-update-test/memory_update.json"
    fs.files[receipt] = json.dumps({"archiveUri":archive, "inputHash":"old", "status":"failed"})
    plan = await plan_migration(fs, CTX)
    await apply_migration(fs, CTX, plan, tmp_path / "backup.json")
    actual = MemoryUpdateContext.model_validate(json.loads(fs.files[archive + "/memory_update_context.json"]))
    actual.validate_owner(CTX)
    assert json.loads(fs.files[receipt])["inputHash"] == actual.input_hash()


def test_canonical_parser_rejects_other_files_and_owners():
    root = ROOT + "/memories/"
    assert person_anchor_from_uri(NEW, root) == "11111111-1111-4111-8111-111111111111"
    for uri in [OLD, NEW.replace("memory.md", "profile.json"), NEW.replace("memory.md", "questions.md"), NEW.replace("alice", "bob"), NEW.replace("/memory.md", "/../memory.md")]:
        assert person_anchor_from_uri(uri, root) is None


@pytest.mark.asyncio
async def test_model_can_discover_person_subdirectories():
    from unittest.mock import AsyncMock
    from openviking.session.memory.tools import MemoryLsTool
    fs = SimpleNamespace(ls=AsyncMock(return_value=[{"uri": NEW.rsplit("/", 1)[0], "isDir": True}]))
    result = await MemoryLsTool().execute(SimpleNamespace(viking_fs=fs, request_ctx=CTX), uri=ROOT + "/memories/people")
    assert "11111111-1111-4111-8111-111111111111/" in result


@pytest.mark.asyncio
async def test_failed_index_enqueue_keeps_original_recoverable(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from openviking.session.memory.memory_updater import MemoryUpdater
    fs = fixture()
    plan = await plan_migration(fs, CTX)
    path = tmp_path / "backup.json"
    enqueue = AsyncMock(return_value=False)
    monkeypatch.setattr(MemoryUpdater, "refresh_file_embedding", enqueue)
    with pytest.raises(RuntimeError, match="enqueue index"):
        await apply_migration(fs, CTX, plan, path, db=object())
    assert OLD in fs.files
    assert path.exists()
    enqueue.return_value = True
    await apply_migration(fs, CTX, json.loads(path.read_text()), path, db=object())
    assert OLD not in fs.files and NEW in fs.files


@pytest.mark.asyncio
async def test_refresh_does_not_report_failed_enqueue_as_success():
    from unittest.mock import AsyncMock
    from openviking.session.memory.memory_updater import MemoryUpdater
    db = SimpleNamespace(has_queue_manager=True, enqueue_embedding_msg=AsyncMock(return_value=False))
    assert not await MemoryUpdater.refresh_file_embedding(viking_fs=fixture(), vikingdb=db, uri=OLD, memory_type="people", ctx=CTX)
    db.enqueue_embedding_msg.return_value = True
    assert await MemoryUpdater.refresh_file_embedding(viking_fs=fixture(), vikingdb=db, uri=OLD, memory_type="people", ctx=CTX)


def test_question_answer_propagation_targets_same_person_directory():
    from openviking.session.memory.question_answer_context_provider import answer_registry
    from openviking.session.memory.utils.uri import generate_uri
    registry = answer_registry({"kind": "person", "id": "11111111-1111-4111-8111-111111111111", "memoryUri": NEW})
    assert generate_uri(registry.get("people"), {}, user_space="alice") == NEW
