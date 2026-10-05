import pytest

from openviking.session.memory.dataclass import ResolvedOperation
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.merge_op import SearchReplaceBlock, StrPatch
from openviking.session.memory.question_store import memory_root
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

pytest_plugins = ["tests.test_focus_store"]


@pytest.mark.asyncio
async def test_later_evidence_updates_same_occurrence_without_inventing_time(store):
    updater = MemoryUpdater(registry=create_default_registry())
    updater._viking_fs = store.fs
    updater.strict_merge_errors = True
    uri = memory_root(store.ctx) + "events/launch-review.md"
    first = ResolvedOperation(
        memory_type="events",
        uris=[uri],
        memory_fields={
            "event_key": "launch-review",
            "event_name": "Launch review",
            "occurred_at": "",
            "summary": "Discussed the launch",
            "content": StrPatch(
                blocks=[SearchReplaceBlock(search="", replace="A demo was proposed.")]
            ),
        },
    )
    await updater._apply_upsert(first, store.ctx)
    second = ResolvedOperation(
        memory_type="events",
        uris=[uri],
        memory_fields={
            "event_key": "launch-review",
            "event_name": "Launch review",
            "occurred_at": "",
            "summary": "Agreed on a demo",
            "content": StrPatch(
                blocks=[
                    SearchReplaceBlock(
                        search="A demo was proposed.",
                        replace="A demo was proposed. The team agreed after review.",
                    )
                ]
            ),
        },
    )
    await updater._apply_upsert(second, store.ctx)
    assert list(store.fs.files) == [uri]
    result = MemoryFileUtils.read(store.fs.files[uri])
    # Empty metadata is omitted by the Markdown serializer; no date is invented.
    assert not result.extra_fields.get("occurred_at")
    assert result.extra_fields["event_key"] == "launch-review"
    assert result.content == "A demo was proposed. The team agreed after review."
