# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
import xxhash


def str_to_uint64(input_string: str) -> int:
    """
    Generate a 64-bit unsigned integer hash from a string using xxHash.

    The string is hashed as its UTF-8 bytes. xxhash releases differ on whether
    they take a str at all; bytes give the same value under every one, so the
    ids already stored stay valid.
    """
    return xxhash.xxh64(input_string.encode("utf-8")).intdigest()
