#!/usr/bin/env python3
"""Zero-OpenMM exact-history verifier for the native all-path R2 replay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import verify_selected_ivm_full_history_replay as BASE


SCHEMA = "bb-irivm-selected-all-paths-replay-verification-v2"
REGISTRATION_SCHEMA = "bb-irivm-selected-all-paths-output-only-replay-v2"
AUTH_SCHEMA = "bb-irivm-selected-all-paths-output-only-replay-authorization-v2"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_allpaths_r2"


@contextmanager
def _specialize_base():
    replacements = {
        "SCHEMA": SCHEMA,
        "REGISTRATION_SCHEMA": REGISTRATION_SCHEMA,
        "AUTH_SCHEMA": AUTH_SCHEMA,
        "LABEL": LABEL,
    }
    previous = {name: getattr(BASE, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(BASE, name, value)


def main(argv: Iterable[str] | None = None) -> None:
    tokens = list(argv) if argv is not None else None
    identity = argparse.ArgumentParser(add_help=False)
    identity.add_argument("--registration", type=Path, required=True)
    args, _ = identity.parse_known_args(tokens)
    registration_path = args.registration.expanduser().resolve(strict=True)
    registration = BASE.read_json(registration_path)
    BASE.require(
        registration.get("source_sha256", {}).get(Path(__file__).name)
        == BASE.sha256_file(Path(__file__).resolve(strict=True)),
        "R2 verifier source drift",
    )
    BASE.require(
        registration.get("full_state_observer_installed") is False,
        "R2 registration unexpectedly installs a full-state observer",
    )
    with _specialize_base():
        BASE.main(tokens)


if __name__ == "__main__":
    main()
