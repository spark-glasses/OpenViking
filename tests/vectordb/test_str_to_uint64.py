# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The id a record is stored under is the same under every xxhash release."""

import pytest

from openviking.storage.vectordb.utils.str_to_uint64 import str_to_uint64


@pytest.mark.parametrize(
    "text, stored_id",
    [
        ("abc", 4952883123889572249),
        ("郭一森", 18439688206828658805),
        ("viking://user/u/memories/people/08c74a4a/memory.md", 4025495341808660549),
        ("", 17241709254077376921),
    ],
)
def test_a_string_keeps_the_id_it_was_stored_under(text, stored_id):
    assert str_to_uint64(text) == stored_id
