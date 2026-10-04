"""Email identity ingestion persists uncertain questions, never a People record."""
from uuid import uuid4

import pytest
from openviking.server.identity import RequestContext, Role
from openviking.server.routers import questions
from openviking.session.memory.question_store import QuestionStore
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier


class MemoryFS:
    def __init__(self):
        self.files = {}

    async def read_file(self, uri, **kwargs):
        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        self.files[uri] = content

    async def ls(self, uri, **kwargs):
        children = {uri + "/" + key[len(uri) + 1:].split("/", 1)[0]
                    for key in self.files if key.startswith(uri + "/")}
        return [{"uri": key, "isDir": key not in self.files} for key in children]


@pytest.mark.asyncio
async def test_email_identity_question_is_unassigned_and_replays_same_cluster(monkeypatch):
    fs = MemoryFS()
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    store = QuestionStore(fs, ctx)
    monkeypatch.setattr(questions, "store", lambda scope: store)
    cluster = uuid4()
    body = questions.EmailIdentityRequest(clusterId=cluster, addresses=["maya@example.test"],
        description="the founder discussing a prototype", sourceRefs=["email:" + str(uuid4())])
    first = (await questions.discover_email_identity(body, ctx)).result["question"]
    replay = (await questions.discover_email_identity(body, ctx)).result["question"]
    assert first["questionId"] == replay["questionId"]
    assert first["subject"] == {"kind": "unassigned", "id": "unassigned"}
    assert first["purpose"] == "emailIdentity"
    assert first["scope"] == {"kind": "emailIdentity", "clusterId": str(cluster), "addresses": body.addresses}
    assert first["ownershipUncertain"] is True
    assert set(first["sourceRefs"]) == set(body.sourceRefs + ["email-identity-observation:" + str(cluster)])
    assert first["evidence"][0]["sourceRef"] == "email-identity-observation:" + str(cluster)
    assert not any("/people/" in uri for uri in fs.files)


@pytest.mark.asyncio
@pytest.mark.parametrize("ref", ["em_short", "email:invalid", "transcript:" + str(uuid4())])
async def test_invalid_sources_do_not_write_question(monkeypatch, ref):
    def unexpected_store(ctx):
        raise AssertionError("invalid source must be rejected before storage")
    monkeypatch.setattr(questions, "store", unexpected_store)
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.ROOT)
    body = questions.EmailIdentityRequest(clusterId=uuid4(), addresses=["maya@example.test"],
        description="a founder", sourceRefs=[ref])
    with pytest.raises(InvalidArgumentError):
        await questions.discover_email_identity(body, ctx)
