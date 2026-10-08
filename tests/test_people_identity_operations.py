"""Identity barriers and durable moves."""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4
import pytest
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.exceptions import ConflictError, NotFoundError
from openviking.session.memory import people_operation as operations
from openviking.session.memory.person_identity import person_memory_uri
from openviking.session.memory.person_contact_store import PersonContactStore
from openviking.session.memory.question_store import QuestionStore, question_uri
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.dataclass import MemoryFile


class FS:
    agfs = None

    def __init__(self):
        self.files = {}

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content

    async def ls(self, uri, **kwargs):
        return [{"uri": key} for key in self.files if key.startswith(uri.rstrip("/") + "/")]

    async def tree(self, uri, **kwargs):
        return await self.ls(uri, **kwargs)

    async def rm(self, uri, **kwargs):
        self.files.pop(uri, None)


@pytest.fixture
def setup(monkeypatch):
    fs = FS()
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    lock = asyncio.Lock()

    @asynccontextmanager
    async def locked(fs, ctx):
        async with lock:
            yield

    monkeypatch.setattr(operations, "gate_lock", locked)
    return fs, ctx


@pytest.mark.asyncio
async def test_barrier_blocks_other_writers_and_fences_stale_context_after_finish(setup):
    fs, ctx = setup
    op, anchors, attempt = str(uuid4()), [str(uuid4()), str(uuid4())], "a" * 64
    await operations.change_gate(fs, ctx, op, "freeze", anchors, "reassign", attempt)
    with pytest.raises(ConflictError):
        await operations.check_gate(fs, ctx, "b" * 64)
    await operations.check_gate(fs, ctx, attempt)
    await operations.change_gate(fs, ctx, op, "finish", anchors, "reassign", attempt)
    with pytest.raises(ConflictError):
        await operations.check_gate(fs, ctx, "b" * 64, expected_epoch=0)
    await operations.check_gate(fs, ctx, "b" * 64)
    assert (await operations.change_gate(fs, ctx, op, "freeze", anchors, "reassign", attempt))[
        "state"
    ] == "done"
    assert (await operations.read_gate(fs, ctx))["operationId"] is None


@pytest.mark.asyncio
async def test_recovery_rejects_a_stale_worker_and_changed_operation_parameters(setup):
    fs, ctx = setup
    op, anchors = str(uuid4()), [str(uuid4()), str(uuid4())]
    await operations.change_gate(fs, ctx, op, "freeze", anchors, "reassign", "a" * 64)
    await operations.change_gate(fs, ctx, op, "freeze", anchors, "reassign", "b" * 64, 1)
    with pytest.raises(ConflictError):
        await operations.change_gate(fs, ctx, op, "finish", anchors, "reassign", "a" * 64, 0)
    with pytest.raises(ConflictError):
        await operations.change_gate(fs, ctx, op, "freeze", anchors[::-1], "reassign", "b" * 64, 1)
    await operations.check_gate(fs, ctx, "b" * 64)


@pytest.mark.asyncio
async def test_merge_keeps_question_ids_answers_and_historical_memory(setup):
    fs, ctx = setup
    anchors = [str(uuid4()), str(uuid4())]
    op = str(uuid4())
    source, target = [person_memory_uri(ctx, a) for a in anchors]
    fs.files[source] = MemoryFileUtils.write(
        MemoryFile.from_parsed(
            uri=source, parsed={"memory_type": "people", "content": "Historical source fact."}
        )
    )
    store = QuestionStore(fs, ctx)
    subject = {"kind": "person", "id": anchors[0], "memoryUri": source}
    q = (
        await store.discover(
            question_uri(ctx, subject),
            subject,
            [{"topicKey": "company", "text": "Which company?", "sourceRefs": ["email:source"]}],
        )
    )[0]
    await store.record(
        {
            "questionId": q["questionId"],
            "action": "resolved",
            "evidenceText": "Acme",
            "conversationId": "chat",
            "messageId": "answer",
            "evidenceRole": "user",
        }
    )
    before = await store.get(q["questionId"])
    await operations.change_gate(fs, ctx, op, "freeze", anchors, "merge", "a" * 64)
    await operations.change_gate(fs, ctx, op, "finish", anchors, "merge", "a" * 64)
    after = await QuestionStore(fs, ctx).get(q["questionId"])
    assert after["subject"]["id"] == anchors[1]
    assert after["answers"] == before["answers"] and after["state"] == "resolved"
    assert after["deliveryCycleId"] == before["deliveryCycleId"]
    page = MemoryFileUtils.read(fs.files[source], uri=source)
    assert page.extra_fields["redirectUri"] == target and page.content == "Historical source fact."
    await operations.change_gate(fs, ctx, op, "finish", anchors, "merge", "a" * 64)
    assert len(await store.list()) == 1
