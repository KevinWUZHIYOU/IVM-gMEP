#!/usr/bin/env python3
"""R3 identity wrapper around the hash-bound zero-OpenMM shard verifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import verify_selected_ivm_full_state_history_replay as BASE


SCHEMA = "bb-irivm-selected-full-state-history-verification-v3"
REGISTRATION_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-v3"
AUTHORIZATION_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-authorization-v3"
REGISTRATION_STATUS = "FROZEN_BEFORE_FIRST_R3_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r3"


def main(argv: Iterable[str] | None = None) -> None:
    tokens = list(argv) if argv is not None else None
    identity_parser = argparse.ArgumentParser(add_help=False)
    identity_parser.add_argument("--registration", type=Path, required=True)
    identity, _ = identity_parser.parse_known_args(tokens)
    registration_path = identity.registration.expanduser().resolve(strict=True)
    registration = BASE.read_json(registration_path)
    BASE.require(
        registration.get("source_sha256", {}).get(Path(__file__).name)
        == BASE.sha256_file(Path(__file__).resolve(strict=True)),
        "R3 verifier wrapper source drift",
    )
    overrides = {
        "SCHEMA": SCHEMA,
        "REGISTRATION_SCHEMA": REGISTRATION_SCHEMA,
        "AUTHORIZATION_SCHEMA": AUTHORIZATION_SCHEMA,
        "REGISTRATION_STATUS": REGISTRATION_STATUS,
        "LABEL": LABEL,
    }
    prior = {name: getattr(BASE, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(BASE, name, value)
        BASE.main(tokens)
    finally:
        for name, value in prior.items():
            setattr(BASE, name, value)


if __name__ == "__main__":
    main()
