#!/usr/bin/env python3
"""Dependency-free test runner for the plain-assert test suite."""

from __future__ import annotations

import inspect

import test_core
import test_string_methods
import test_v2_line_search_semantics


class _MonkeyPatch:
    def __init__(self):
        self._changes = []

    def setattr(self, target, name, value):
        self._changes.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, original in reversed(self._changes):
            setattr(target, name, original)


def main() -> None:
    tests = []
    for module in (
        test_core,
        test_string_methods,
        test_v2_line_search_semantics,
    ):
        tests.extend(
            (name, function)
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        )
    for name, function in tests:
        parameters = inspect.signature(function).parameters
        if not parameters:
            function()
        elif list(parameters) == ["monkeypatch"]:
            monkeypatch = _MonkeyPatch()
            try:
                function(monkeypatch)
            finally:
                monkeypatch.undo()
        else:
            raise TypeError(
                f"Unsupported test fixtures for {name}: {list(parameters)}"
            )
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
