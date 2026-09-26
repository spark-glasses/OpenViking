"""Meeting context persistence, native task outcomes, retries and crash recovery."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.server.routers.sessions import CreateSessionRequest
from openviking.service.session_service import SessionService
from openviking.session.memory.meeting_context import MeetingContext, meeting_memory_policy
from openviking.session.session import Session, SessionMeta
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig

MEETING_ID = "00000000-0000-4000-8000-000000000001"
MEETING_URI = f"viking://user/alice/memories/events/meetings/{MEETING_ID}.md"
ARCHIVE = "viking://session/meeting-job/history/archive_001"


class MemoryFS:
    agfs = None

    def __init__(self):
        self.files = {}
        self.vector_store = object()

    def _uri_to_path(self, uri, **kwargs):
        return uri

    async def exists(self, uri, **kwargs):
        return uri in self.files

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content

    async def rm(self, uri, **kwargs):
        self.files.pop(uri, None)

    async def ls(self, uri, **kwargs):
        return [
            {"uri": key, "name": key[len(uri) + 1 :]}
            for key in self.files
            if key.startswith(uri + "/") and "/" not in key[len(uri) + 1 :]
        ]


@pytest.fixture
def setup(monkeypatch):
    fs = MemoryFS()
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    config = SimpleNamespace(memory=MemoryConfig(), vlm=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("openviking.session.session.get_openviking_config", lambda: config)
    spec = MeetingContext.model_validate(
        {
            "jobId": "job-1",
            "meetingId": MEETING_ID,
            "inputHash": "input-1",
            "sourceRef": f"transcript:{MEETING_ID}",
            "sourceVersion": "sha256:" + "a" * 64,
            "people": [
                {
                    "contactId": "contact-1",
                    "anchorId": "anchor-1",
                    "name": "Ethan",
                    "personMemoryUri": "viking://user/alice/memories/people/anchor-1.md",
                }
            ],
            "mediaWindows": [
                {
                    "startMs": 1000,
                    "endMs": 20000,
                    "startedAt": "2026-09-22T10:00:00Z",
                    "endedAt": "2026-09-22T10:00:19Z",
                }
            ],
            "referenceEndedAt": "2026-09-22T10:00:19Z",
            "captureTimingSource": "hardware",
            "meetingMemoryUri": MEETING_URI,
        }
    ).model_dump()
    messages = [Message(id="input-1", role="user", parts=[TextPart("Scoped recording evidence")])]
    session = Session(viking_fs=fs, ctx=ctx, session_id="meeting-job")
    archive = session.uri + "/history/archive_001"
    return SimpleNamespace(
        fs=fs,
        ctx=ctx,
        config=config,
        spec=spec,
        messages=messages,
        session=session,
        archive=archive,
    )


def email_spec():
    return {
        "batchId": "batch-1",
        "contactId": "contact-1",
        "contactName": "Ethan",
        "emails": ["ethan@example.test"],
        "personMemoryUri": "viking://user/alice/memories/people/contact-1.md",
        "sourceRefs": ["email:00000000-0000-4000-8000-000000000002"],
    }


async def prepare_failed_archive(setup):
    setup.fs.files[setup.archive + "/meeting_context.json"] = json.dumps(setup.spec)
    setup.fs.files[setup.archive + "/.failed.json"] = "{}"
    setup.fs.files[setup.archive + "/messages.jsonl"] = setup.messages[0].to_jsonl()


def test_meeting_meta_roundtrip_and_create_request_exclusion(setup):
    meta = SessionMeta(session_id="s", meeting_context=setup.spec)
    assert SessionMeta.from_dict(meta.to_dict()).meeting_context == setup.spec
    assert SessionMeta.from_dict({"session_id": "old"}).meeting_context is None
    request = CreateSessionRequest(meeting_context=setup.spec)
    assert request.meeting_context.jobId == "job-1"
    with pytest.raises(ValueError):
        CreateSessionRequest(email_context=email_spec(), meeting_context=setup.spec)


@pytest.mark.asyncio
async def test_create_meeting_validates_owner_and_forces_task_policy(setup, monkeypatch):
    service = SessionService()
    session = Mock(meta=SessionMeta(session_id="s"))
    session.exists = AsyncMock(return_value=False)
    session.ensure_exists = AsyncMock()
    monkeypatch.setattr(service, "session", lambda *args: session)
    await service.create(
        setup.ctx, "s", memory_policy={"peer": {"enabled": True}}, meeting_context=setup.spec
    )
    assert session.meta.meeting_context == setup.spec
    assert session.meta.memory_policy == meeting_memory_policy()
    assert session.meta.email_context is None
    with pytest.raises(InvalidArgumentError):
        await service.create(setup.ctx, "s", email_context=email_spec(), meeting_context=setup.spec)
    with pytest.raises(ValueError):
        await service.create(
            setup.ctx,
            "s",
            meeting_context={**setup.spec, "meetingMemoryUri": MEETING_URI.replace("alice", "bob")},
        )
    assert session.ensure_exists.await_count == 1


@pytest.fixture
def commit_env(setup, monkeypatch):
    @asynccontextmanager
    async def lock(*args, **kwargs):
        yield

    tracker = SimpleNamespace(
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="attempt-1"))
    )
    monkeypatch.setattr("openviking.storage.transaction.LockContext", lock)
    monkeypatch.setattr(
        "openviking.storage.transaction.get_lock_manager", lambda: SimpleNamespace()
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    session = setup.session
    session._save_meta = AsyncMock()
    session._write_to_agfs_async = AsyncMock()
    session._run_memory_extraction = AsyncMock()
    session._messages = list(setup.messages)
    session.meta.meeting_context = setup.spec
    return tracker


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_index", [None, 0, 4])
async def test_commit_freezes_nested_context_and_uses_one_task(setup, commit_env, chunk_index):
    if chunk_index is not None:
        setup.spec["chunkIndex"] = chunk_index
    result = await setup.session.commit_async()
    setup.session.meta.meeting_context["people"][0]["anchorId"] = "later-live-change"
    setup.session.meta.meeting_context["chunkIndex"] = 99
    await asyncio.sleep(0)
    frozen = json.loads(setup.fs.files[setup.archive + "/meeting_context.json"])
    assert frozen["people"][0]["anchorId"] == "anchor-1"
    assert frozen.get("chunkIndex") == chunk_index
    called = setup.session._run_memory_extraction.await_args.kwargs
    assert called["meeting_context"] == frozen
    assert called["memory_policy"] == meeting_memory_policy()
    assert called["email_context"] is None
    assert result["archive_uri"] == setup.archive
    assert result["task_id"] == "attempt-1"
    assert commit_env.create_if_no_running.await_args.kwargs["require_no_existing"] is True
    assert setup.session.messages == []


@pytest.mark.asyncio
async def test_context_persistence_failure_does_not_drop_messages(setup, commit_env):
    setup.fs.write_file = AsyncMock(side_effect=OSError("archive unavailable"))
    with pytest.raises(OSError):
        await setup.session.commit_async()
    assert setup.session.messages == setup.messages
    assert setup.session._compression.compression_index == 0
    commit_env.create_if_no_running.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_rejects_conflicting_contexts_before_archiving(setup, commit_env):
    setup.session.meta.email_context = email_spec()
    with pytest.raises(ValueError):
        await setup.session.commit_async()
    assert setup.fs.files == {}
    commit_env.create_if_no_running.assert_not_awaited()


def prepare_phase2(setup, monkeypatch, *, failure=None):
    tracker = SimpleNamespace(start=AsyncMock(), complete=AsyncMock(), fail=AsyncMock())
    waiting = SimpleNamespace(
        register_request=Mock(),
        cleanup=Mock(),
        wait_for_request=AsyncMock(),
        build_queue_status=Mock(return_value={}),
    )
    redo = SimpleNamespace(write_pending_async=AsyncMock(), mark_done_async=AsyncMock())
    manager = SimpleNamespace(redo_recovery_enabled=True, redo_log=redo)
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr("openviking.storage.transaction.get_lock_manager", lambda: manager)
    monkeypatch.setattr("openviking.session.session.get_request_wait_tracker", lambda: waiting)
    session = setup.session
    session._wait_for_previous_archive_done = AsyncMock()
    session._get_latest_completed_archive_overview = AsyncMock(return_value=None)
    session._generate_archive_summary_async = AsyncMock(return_value="Meeting summary")
    session._merge_and_save_commit_meta = AsyncMock()

    async def extract(**kwargs):
        result = {
            "jobId": setup.spec["jobId"],
            "taskId": "attempt-1",
            "outcome": "applied",
            "writtenUris": [MEETING_URI],
            "editedUris": [],
            "sourceRefs": [setup.spec["sourceRef"]],
            "coverage": [{"startMs": 1000, "endMs": 20000}],
            "questionRefs": [{"questionId": "q1"}],
            "errors": [],
        }
        if failure == "wrong-attempt":
            result["taskId"] = "old-attempt"
        if failure == "failed-result":
            result["outcome"] = "failed"
        if failure != "missing-result":
            setup.fs.files[setup.archive + "/meeting_result.json"] = json.dumps(
                None if failure == "null-result" else result
            )
        return []

    session._session_compressor = SimpleNamespace(
        extract_long_term_memories=AsyncMock(side_effect=extract)
    )
    if failure == "queue":
        waiting.build_queue_status.return_value = {"embedding": {"error_count": 1}}
    elif failure == "timeout":
        waiting.wait_for_request.side_effect = TimeoutError("index timeout")
    elif failure == "summary":
        session._generate_archive_summary_async.side_effect = ValueError("invalid summary")
    elif failure == "task-store":
        tracker.complete.side_effect = RuntimeError("task persistence failed")
    elif failure == "cancel":
        waiting.wait_for_request.side_effect = asyncio.CancelledError()
    return SimpleNamespace(tracker=tracker, waiting=waiting, redo=redo)


async def run_phase2(setup):
    await setup.session._run_memory_extraction(
        task_id="attempt-1",
        archive_uri=setup.archive,
        messages=setup.messages,
        usage_records=[],
        first_message_id="input-1",
        last_message_id="input-1",
        memory_policy=meeting_memory_policy(),
        meeting_context=setup.spec,
    )


@pytest.mark.asyncio
async def test_phase2_passes_frozen_scope_and_publishes_result_with_attempt_snapshot(
    setup, monkeypatch
):
    runtime = prepare_phase2(setup, monkeypatch)
    await run_phase2(setup)
    runtime.tracker.fail.assert_not_awaited()
    result = runtime.tracker.complete.await_args.args[1]
    assert result["meeting_result"]["outcome"] == "applied"
    assert result["meeting_result_uri"] == setup.archive + "/meeting_result.json"
    assert result["speaker_assignments_uri"] == setup.archive + "/speaker_assignments.json"
    assert setup.archive + "/.done" in setup.fs.files
    assert (
        json.loads(setup.fs.files[setup.archive + "/attempts/attempt-1/meeting_context.json"])
        == setup.spec
    )
    called = setup.session._session_compressor.extract_long_term_memories.await_args.kwargs
    assert called["meeting_context"] == setup.spec
    assert called["meeting_attempt_id"] == "attempt-1"
    assert "email_context" not in called
    assert runtime.redo.write_pending_async.await_args.args[1]["meeting_context"] == setup.spec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "queue",
        "timeout",
        "summary",
        "task-store",
        "wrong-attempt",
        "failed-result",
        "missing-result",
        "null-result",
    ],
)
async def test_phase2_failure_cannot_publish_success_or_leave_done_marker(
    setup, monkeypatch, failure
):
    runtime = prepare_phase2(setup, monkeypatch, failure=failure)
    await run_phase2(setup)
    runtime.tracker.fail.assert_awaited_once()
    assert setup.archive + "/.done" not in setup.fs.files
    failed = json.loads(setup.fs.files[setup.archive + "/meeting_result.json"])
    assert failed["outcome"] == "failed"
    assert failed["jobId"] == setup.spec["jobId"]
    assert failed["taskId"] == "attempt-1"
    assert failed["questionRefs"] == []
    assert failed["errors"]
    if failure not in ("missing-result", "null-result"):
        assert failed["writtenUris"] == [MEETING_URI]
        assert failed["coverage"] == [{"startMs": 1000, "endMs": 20000}]
        assert failed["partial"] is True
    assert failed == json.loads(
        setup.fs.files[setup.archive + "/attempts/attempt-1/meeting_result.json"]
    )


@pytest.mark.asyncio
async def test_phase2_cancel_persists_failed_attempt_before_raising(setup, monkeypatch):
    runtime = prepare_phase2(setup, monkeypatch, failure="cancel")
    with pytest.raises(asyncio.CancelledError):
        await run_phase2(setup)
    runtime.tracker.fail.assert_awaited_once()
    assert json.loads(setup.fs.files[setup.archive + "/meeting_result.json"])["outcome"] == "failed"
    assert setup.archive + "/.done" not in setup.fs.files


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_index", [None, 0, 4])
async def test_retry_reads_frozen_context_and_never_rearchives(setup, monkeypatch, chunk_index):
    if chunk_index is not None:
        setup.spec["chunkIndex"] = chunk_index
    await prepare_failed_archive(setup)
    setup.session.meta.meeting_context = {
        **setup.spec,
        "inputHash": "new-live-hash",
        "chunkIndex": 99,
    }
    setup.session._run_memory_extraction = AsyncMock()
    tracker = SimpleNamespace(
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="retry-1"))
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    original = setup.fs.files[setup.archive + "/messages.jsonl"]
    result = await setup.session.retry_meeting_archive("archive_001")
    await asyncio.sleep(0)
    assert result["archive_uri"] == setup.archive
    assert setup.fs.files[setup.archive + "/messages.jsonl"] == original
    assert setup.session._run_memory_extraction.await_args.kwargs["meeting_context"] == setup.spec
    assert (
        setup.session._run_memory_extraction.await_args.kwargs["memory_policy"]
        == meeting_memory_policy()
    )
    assert setup.session.meta.commit_count == 0
    tracker.create_if_no_running.return_value = None
    with pytest.raises(InvalidArgumentError):
        await setup.session.retry_meeting_archive("archive_001")
    with pytest.raises(InvalidArgumentError):
        await setup.session.retry_meeting_archive("../archive_001")


@pytest.mark.asyncio
async def test_retry_orphan_waits_for_original_request_and_atomically_requires_no_tasks(
    setup, monkeypatch
):
    await prepare_failed_archive(setup)
    del setup.fs.files[setup.archive + "/.failed.json"]
    setup.session.meta.commit_count = 1
    setup.session.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
    setup.session._run_memory_extraction = AsyncMock()
    tracker = SimpleNamespace(
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="recovered"))
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    with pytest.raises(InvalidArgumentError):
        await setup.session.retry_meeting_archive("archive_001")
    tracker.create_if_no_running.assert_not_awaited()
    setup.session.meta.last_commit_at = (
        datetime.now(timezone.utc) - timedelta(seconds=61)
    ).isoformat()
    await setup.session.retry_meeting_archive("archive_001")
    await asyncio.sleep(0)
    assert tracker.create_if_no_running.await_args.kwargs["require_no_existing"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [("completed", "completed"), ("failed", "accepted")])
async def test_retry_done_marker_requires_completed_task(setup, monkeypatch, status, expected):
    await prepare_failed_archive(setup)
    setup.fs.files[setup.archive + "/.done"] = "{}"
    setup.session._run_memory_extraction = AsyncMock()
    tracker = SimpleNamespace(
        list_tasks=AsyncMock(return_value=[SimpleNamespace(status=status, task_id="original")]),
        create_if_no_running=AsyncMock(return_value=SimpleNamespace(task_id="retry")),
    )
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    assert (await setup.session.retry_meeting_archive("archive_001"))["status"] == expected
    await asyncio.sleep(0)
    assert setup.session._run_memory_extraction.await_count == int(status != "completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [None, "{}", "not-json", "null", "[]"])
async def test_redo_never_falls_back_if_frozen_meeting_context_is_missing_or_invalid(
    setup, monkeypatch, context
):
    from openviking.storage.transaction.lock_manager import LockManager

    if context is not None:
        setup.fs.files[setup.archive + "/meeting_context.json"] = context
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: setup.fs)
    manager = SimpleNamespace(
        _async_agfs=SimpleNamespace(cat=AsyncMock(return_value=setup.messages[0].to_jsonl())),
        _enqueue_semantic=AsyncMock(),
    )
    with pytest.raises((RuntimeError, ValueError)):
        await LockManager._redo_session_memory(
            manager,
            {
                "archive_uri": setup.archive,
                "session_uri": setup.session.uri,
                "account_id": "account",
                "user_id": "alice",
                "role": "root",
                "meeting_context": setup.spec,
            },
        )
    manager._enqueue_semantic.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_index", [None, 0, 4])
async def test_redo_uses_frozen_scope_native_phase2_and_real_vector_store(
    setup, monkeypatch, chunk_index
):
    from openviking.service.task_tracker import TaskStatus
    from openviking.storage.transaction.lock_manager import LockManager

    if chunk_index is not None:
        setup.spec["chunkIndex"] = chunk_index
    await prepare_failed_archive(setup)
    tracker = SimpleNamespace(
        get=AsyncMock(
            side_effect=[
                SimpleNamespace(status=TaskStatus.FAILED),
                SimpleNamespace(status=TaskStatus.COMPLETED),
            ]
        )
    )
    session = SimpleNamespace(load=AsyncMock(), _run_memory_extraction=AsyncMock())
    constructor = Mock(return_value=session)
    compressor = Mock(return_value=object())
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: setup.fs)
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr("openviking.session.Session", constructor)
    monkeypatch.setattr("openviking.session.create_session_compressor", compressor)
    manager = SimpleNamespace(
        _async_agfs=SimpleNamespace(cat=AsyncMock(return_value=setup.messages[0].to_jsonl())),
        _enqueue_semantic=AsyncMock(),
    )
    await LockManager._redo_session_memory(
        manager,
        {
            "archive_uri": setup.archive,
            "session_uri": setup.session.uri,
            "account_id": "account",
            "user_id": "alice",
            "role": "root",
            "task_id": "t1",
            "meeting_context": {**setup.spec, "inputHash": "stale-redo-copy", "chunkIndex": 99},
        },
    )
    called = session._run_memory_extraction.await_args.kwargs
    assert called["meeting_context"] == setup.spec
    assert called["memory_policy"] == meeting_memory_policy()
    assert "email_context" not in called
    assert constructor.call_args.kwargs["vikingdb_manager"] is setup.fs.vector_store
    assert compressor.call_args.kwargs["vikingdb"] is setup.fs.vector_store
    manager._enqueue_semantic.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_route_loads_authorized_session_and_returns_native_result(setup, monkeypatch):
    from openviking.server.routers import sessions

    native = {"status": "accepted", "task_id": "retry", "archive_uri": setup.archive}
    session = SimpleNamespace(retry_meeting_archive=AsyncMock(return_value=native))
    get = AsyncMock(return_value=session)
    monkeypatch.setattr(
        sessions, "get_service", lambda: SimpleNamespace(sessions=SimpleNamespace(get=get))
    )
    response = await sessions.retry_meeting_archive("meeting-job", "archive_001", setup.ctx)
    assert response.result == native
    get.assert_awaited_once_with("meeting-job", setup.ctx, auto_create=False)
    session.retry_meeting_archive.assert_awaited_once_with("archive_001")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected,actual,accepted",
    [
        (0, 0, True),
        (4, 4, True),
        (None, None, True),
        (0, None, False),
        (4, 0, False),
        (0, 4, False),
        (None, 0, False),
        (0, False, False),
        (1, True, False),
        (1, 1.0, False),
        (1, "1", False),
    ],
)
async def test_phase2_result_requires_original_chunk_identity(
    setup, monkeypatch, expected, actual, accepted
):
    if expected is not None:
        setup.spec["chunkIndex"] = expected
    runtime = prepare_phase2(setup, monkeypatch)
    original = setup.session._session_compressor.extract_long_term_memories.side_effect

    async def extract(**kwargs):
        result = await original(**kwargs)
        path = setup.archive + "/meeting_result.json"
        outcome = json.loads(setup.fs.files[path])
        if actual is not None:
            outcome["chunkIndex"] = actual
        setup.fs.files[path] = json.dumps(outcome)
        return result

    setup.session._session_compressor.extract_long_term_memories.side_effect = extract
    await run_phase2(setup)
    if accepted:
        runtime.tracker.fail.assert_not_awaited()
        runtime.tracker.complete.assert_awaited_once()
        assert (
            runtime.tracker.complete.await_args.args[1]["meeting_result"].get("chunkIndex")
            == expected
        )
    else:
        runtime.tracker.complete.assert_not_awaited()
        runtime.tracker.fail.assert_awaited_once()
        assert setup.archive + "/.done" not in setup.fs.files
        failed = json.loads(setup.fs.files[setup.archive + "/meeting_result.json"])
        assert failed["outcome"] == "failed"
        assert failed.get("chunkIndex") == expected
        assert type(failed.get("chunkIndex")) is type(expected)
        assert failed["writtenUris"] == failed["editedUris"] == failed["reindexedUris"] == []
        assert failed["questionRefs"] == failed["sourceRefs"] == failed["speakerAssignments"] == []
        assert failed["previousResult"].get("chunkIndex") == actual
        assert failed["previousResult"]["writtenUris"] == [MEETING_URI]
        assert failed["previousResult"]["coverage"] == [{"startMs": 1000, "endMs": 20000}]
        assert failed["partial"] is True
        assert failed["coverage"]["hasMore"] is True
        assert failed["coverage"].get("chunkIndex") == expected
        assert failed["coverage"]["scope"] == ("chunk" if expected is not None else "recording")
        assert failed["coverage"]["readTokenRanges"] == failed["coverage"]["pages"] == []
        assert failed["coverage"]["requestedWindows"] == [{"startMs": 1000, "endMs": 20000}]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_index", [None, 0, 4])
async def test_early_meeting_failure_keeps_original_chunk_without_prior_result(setup, chunk_index):
    if chunk_index is not None:
        setup.spec["chunkIndex"] = chunk_index
    await setup.session._record_meeting_phase_failure(
        setup.archive, "failed-attempt", setup.spec, "source unavailable"
    )
    failed = json.loads(setup.fs.files[setup.archive + "/meeting_result.json"])
    assert failed.get("chunkIndex") == chunk_index
    assert failed == json.loads(
        setup.fs.files[setup.archive + "/attempts/failed-attempt/meeting_result.json"]
    )
