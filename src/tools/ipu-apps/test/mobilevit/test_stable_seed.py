"""Filler-stream seeds of the full-layer tests are the same in every process.

``hash()`` of a string is salted per process, so a seed taken from it cannot be
replayed from a failure log; ``_stable_seed`` must not depend on
``PYTHONHASHSEED``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib

from fixture_full_layer_chain import _stable_seed


def _seed_in_subprocess(hash_seed: str) -> tuple[int, int]:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fixture_full_layer_chain import _stable_seed; "
            "print(_stable_seed('a', 'filler'), hash(('a', 'filler')))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return int(out[0]), int(out[1])


def test_stable_seed_ignores_the_hash_salt():
    (seed_1, hash_1), (seed_2, hash_2) = _seed_in_subprocess("1"), _seed_in_subprocess("2")
    assert hash_1 != hash_2, "the two subprocesses must use different hash salts"
    assert seed_1 == seed_2 == _stable_seed("a", "filler")


def test_stable_seed_is_a_32_bit_digest_of_the_joined_parts():
    assert _stable_seed("a", "filler") == zlib.crc32(b"a/filler")
    assert 0 <= _stable_seed("q_block", "filler_km") <= 0xFFFFFFFF
    assert _stable_seed("a", "filler") != _stable_seed("a", "filler_km")
