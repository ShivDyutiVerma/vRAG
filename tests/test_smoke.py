"""Trivial green test so `pytest` is never red on an empty codebase (CLAUDE.md hard rule)."""

import vrag


def test_package_imports() -> None:
    assert vrag is not None
