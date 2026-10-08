"""Retiring a merged-away person: one person is left where there were two."""

from uuid import uuid4

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.person_contact_store import PersonContactStore
from openviking.session.memory.person_identity import (
    load_identity_directory,
    person_memory_uri,
    preserve_person_identity,
)
from openviking.session.memory.person_retire import retire_person
from openviking.session.memory.question_store import QuestionStore, memory_root, question_uri
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import ConflictError, InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier


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

    async def tree(self, uri, **kwargs):
        return [{"uri": key, "isDir": False} for key in self.files if key.startswith(uri)]

    async def ls(self, uri, **kwargs):
        return await self.tree(uri.rstrip("/") + "/")

    async def rm(self, uri, recursive=False, **kwargs):
        for key in [key for key in self.files if key == uri or key.startswith(uri)]:
            del self.files[key]


def person(contact_id, anchor, name, deleted=False):
    return {
        "personId": contact_id,
        "anchorId": anchor,
        "revision": 2 if deleted else 1,
        "deleted": deleted,
        "profile": {"displayName": name, "givenName": name, "phones": [], "emails": []},
    }


@pytest.fixture
async def two_people():
    """Ada stays. Countess was merged into her and is already marked removed."""
    fs = FS()
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    store = PersonContactStore(fs, ctx)
    ada, countess = str(uuid4()), str(uuid4())
    await store.sync(person(str(uuid4()), ada, "Ada"))
    countess_contact = str(uuid4())
    await store.sync(person(countess_contact, countess, "Countess"))
    for anchor, note in ((ada, "Wrote the first program."), (countess, "Met at the salon.")):
        uri = person_memory_uri(ctx, anchor)
        page = MemoryFileUtils.read(fs.files[uri], uri=uri)
        page.content = page.content + "\n\n" + note
        fs.files[uri] = MemoryFileUtils.write(page)
    await store.sync(person(countess_contact, countess, "Countess", deleted=True))
    return fs, ctx, ada, countess, countess_contact


@pytest.mark.asyncio
async def test_the_retired_person_leaves_and_everything_points_at_the_other(two_people):
    fs, ctx, ada, countess, _ = two_people
    root = memory_root(ctx)
    event = root + "events/salon.md"
    fs.files[event] = MemoryFileUtils.write(
        MemoryFile.from_parsed(
            uri=event,
            parsed={
                "memory_type": "events",
                "content": f"Dinner with [Countess]({person_memory_uri(ctx, countess)}) "
                f"and [Ada]({person_memory_uri(ctx, ada)}); see ../people/{countess}/memory.md.",
            },
        )
    )
    questions = QuestionStore(fs, ctx)
    subject = {"kind": "person", "id": countess, "memoryUri": person_memory_uri(ctx, countess)}
    asked = (
        await questions.discover(
            question_uri(ctx, subject),
            subject,
            [{"topicKey": "title", "text": "Which title?", "sourceRefs": ["email:source"]}],
        )
    )[0]

    result = await retire_person(fs, ctx, None, countess, ada)

    assert not [uri for uri in fs.files if f"people/{countess}/" in uri]
    assert countess not in fs.files[event] and fs.files[event].count(f"people/{ada}/memory.md") == 3
    assert result["repointed"].count(event) == 1
    assert (await QuestionStore(fs, ctx).get(asked["questionId"]))["subject"]["id"] == ada
    directory = await load_identity_directory(fs, ctx)
    assert countess not in directory["people"] and directory["retired"] == {countess: ada}
    # Merging what the pages say is the caller's part; Ada's page is as it was.
    ada_page = fs.files[person_memory_uri(ctx, ada)]
    assert "Wrote the first program." in ada_page and "Met at the salon." not in ada_page


@pytest.mark.asyncio
async def test_retiring_again_changes_nothing(two_people):
    fs, ctx, ada, countess, _ = two_people
    await retire_person(fs, ctx, None, countess, ada)
    after = dict(fs.files)
    assert (await retire_person(fs, ctx, None, countess, ada))["repointed"] == []
    assert fs.files == after


@pytest.mark.asyncio
async def test_a_retired_person_cannot_come_back(two_people):
    fs, ctx, ada, countess, countess_contact = two_people
    await retire_person(fs, ctx, None, countess, ada)

    with pytest.raises(ConflictError):
        await PersonContactStore(fs, ctx).sync(person(countess_contact, countess, "Countess"))
    uri = person_memory_uri(ctx, countess)
    with pytest.raises(InvalidArgumentError):
        await preserve_person_identity(
            fs, ctx, uri, MemoryFile.from_parsed(uri=uri, parsed={"content": "New fact."})
        )
    assert not [key for key in fs.files if f"people/{countess}/" in key]


@pytest.mark.asyncio
async def test_retiring_needs_a_current_person_to_stand_for_the_other(two_people):
    fs, ctx, ada, countess, _ = two_people
    with pytest.raises(InvalidArgumentError):
        await retire_person(fs, ctx, None, countess, countess)
    with pytest.raises(InvalidArgumentError):
        await retire_person(fs, ctx, None, countess, str(uuid4()))
    # Ada cannot retire into Countess, who is removed.
    with pytest.raises(InvalidArgumentError):
        await retire_person(fs, ctx, None, ada, countess)
    await retire_person(fs, ctx, None, countess, ada)
    other = str(uuid4())
    await PersonContactStore(fs, ctx).sync(person(str(uuid4()), other, "Byron"))
    with pytest.raises(ConflictError):
        await retire_person(fs, ctx, None, countess, other)
