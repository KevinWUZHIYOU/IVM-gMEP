#!/usr/bin/env python3
"""NF30000 configuration of the audited zero-physical ODE batch observer.

The implementation is deliberately reused from the completed NF15000
campaign and is hash-pinned below.  Only the terminal batch budget and the
small set of retained plotting states change.  Importing this wrapper neither
imports OpenMM nor acquires a Barnase--Barstar compute lock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True

BASE = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf15000_20260903_r1/analysis/"
    "cache_formal_discrete32_incrementally.py"
)
BASE_SHA256 = "5c0e0917b8937fe45d711a78a3d6c22de99a1257371d3326de5d21dbf4f33963"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if BASE.resolve(strict=True) != BASE or sha256_file(BASE) != BASE_SHA256:
    raise RuntimeError("hash-pinned NF15000 observer implementation changed")

SPEC = importlib.util.spec_from_file_location("bb_nf15000_incremental_cache_base", BASE)
MODULE = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("cannot load hash-pinned observer implementation")
SPEC.loader.exec_module(MODULE)

# The production source guarantees 30 paid interior Force evaluations per
# published batch.  Retain the states needed by the R2 profile/evolution plots;
# every one of the 1000 scalar metric rows remains in the final cache.
MODULE.TARGET_BATCHES = 1000
MODULE.TARGET_NF = 30000
MODULE.TARGET_STATE_NF = (240, 2040, 15000, 30000)
_BASE_IMPORT_FROZEN_SOURCE = MODULE.import_frozen_source

CACHE_SCHEMA = MODULE.CACHE_SCHEMA
CACHE_PARTIAL = MODULE.CACHE_PARTIAL
CACHE_FINAL = MODULE.CACHE_FINAL
TARGET_BATCHES = MODULE.TARGET_BATCHES
TARGET_NF = MODULE.TARGET_NF
TARGET_STATE_NF = MODULE.TARGET_STATE_NF

MUTAGEN_MIRROR_RUN = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/runs/"
    "fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1"
)
CAMPAIGN_ROOT = MUTAGEN_MIRROR_RUN.parent.parent
RUN_LABEL = MUTAGEN_MIRROR_RUN.name
MUTAGEN_ARCHIVE = MUTAGEN_MIRROR_RUN / "paid_batch_full_state"
MUTAGEN_PRESTART = MUTAGEN_MIRROR_RUN / "PRESTART.json"
MUTAGEN_COMPLETION = MUTAGEN_ARCHIVE / "FULL_STATE_COMPLETE.json"
MUTAGEN_TERMINAL_AUDIT = (
    CAMPAIGN_ROOT / "terminal_audits" / f"{RUN_LABEL}.terminal-audit.json"
)
TRANSPORT_RECEIPT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/"
    "LOCAL_MUTAGEN_REPLICA_TRANSPORT_RECEIPT_R2.json"
)
TRANSPORT_RECEIPT_SHA256 = (
    "7164b8b2593ae1d6a592e11da8b8d99b94578ab631b8527f9f752ec301a41213"
)
MUTAGEN_SESSION_ID = "sync_wluhxwR03mFsMVwykAr1wJZEhU2OBSLb9h1LhC0Dus0"
MUTAGEN_SESSION_NAME = "bb-aseode-nf30000-output-20260903-r1"
MUTAGEN_ALPHA = (
    "wuzhiyou@100.74.85.3:/Users/wuzhiyou/Downloads/IVM_for_MEP/"
    "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1"
)
MUTAGEN_BETA = str(CAMPAIGN_ROOT)
PRESTART_SHA256 = "1c5aa970ab806eccb785e41cb273c52bf5d38b087c834d5dda5fc09555522422"
PRESTART_BYTES = 5648
EXEMPLAR_BATCH = MUTAGEN_ARCHIVE / "batch_0333_nf09990.npz"
EXEMPLAR_BATCH_SHA256 = (
    "850780637149ab34bd6dba36cd01ee7ca76ad03cbc4caa4d2f756a3ee7fb524d"
)
EXEMPLAR_BATCH_BYTES = 4949896
BATCH_NAME = re.compile(r"batch_(\d{4})_nf(\d{5})\.npz")

_BASE_LATEST_CACHE = MODULE.latest_cache
_BASE_WRITE_JSON_NEW = MODULE.write_json_new
_BASE_READONLY_REGULAR = MODULE.readonly_regular
_TRANSPORT_RECEIPT_CACHE: dict[str, Any] | None = None

FROZEN_SOURCE_ROOT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1"
)
FORMAL_OUTPUT_DIR = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/"
    "formal_discrete32_cache_r2_audited_v2_local_mutagen"
)


def _exact_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _read_transport_receipt() -> dict[str, Any]:
    """Load the sealed independent transport receipt and verify local evidence."""

    global _TRANSPORT_RECEIPT_CACHE
    if _TRANSPORT_RECEIPT_CACHE is not None:
        return _TRANSPORT_RECEIPT_CACHE
    receipt_path = TRANSPORT_RECEIPT.expanduser().resolve(strict=True)
    status = receipt_path.lstat()
    MODULE.require(
        stat.S_ISREG(status.st_mode)
        and not receipt_path.is_symlink()
        and stat.S_IMODE(status.st_mode) == 0o444
        and sha256_file(receipt_path) == TRANSPORT_RECEIPT_SHA256,
        "local Mutagen transport receipt is not the frozen registered file",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    session = receipt.get("mutagen_session", {})
    exact_run = receipt.get("exact_run", {})
    prestart = receipt.get("prestart", {})
    exemplar = receipt.get("published_shard_exemplar", {})
    diagnostics = receipt.get("preaudit_diagnostic_local_caches", [])
    MODULE.require(
        receipt.get("schema")
        == "bb-nf30000-local-mutagen-replica-transport-receipt-v1"
        and receipt.get("status")
        == "PASS_REMOTE_IMMUTABLE_LOCAL_MODE_NORMALIZED"
        and receipt.get("physical_calls") == 0
        and session.get("identifier") == MUTAGEN_SESSION_ID
        and session.get("name") == MUTAGEN_SESSION_NAME
        and session.get("mode") == "One Way Safe"
        and session.get("alpha") == MUTAGEN_ALPHA
        and session.get("beta") == MUTAGEN_BETA
        and session.get("alpha_connected") is True
        and session.get("beta_connected") is True
        and exact_run.get("run_label") == RUN_LABEL
        and exact_run.get("remote_path") == str(MUTAGEN_MIRROR_RUN)
        and exact_run.get("local_beta_path") == str(MUTAGEN_MIRROR_RUN)
        and prestart.get("relative_path") == "PRESTART.json"
        and prestart.get("bytes") == PRESTART_BYTES
        and prestart.get("remote_mode_octal") == "0444"
        and prestart.get("local_beta_mode_octal") == "0600"
        and prestart.get("remote_sha256") == PRESTART_SHA256
        and prestart.get("local_beta_sha256") == PRESTART_SHA256
        and prestart.get("remote_local_bytes_and_sha256_equal") is True
        and exemplar.get("relative_path")
        == "paid_batch_full_state/batch_0333_nf09990.npz"
        and exemplar.get("batch_id") == 333
        and exemplar.get("optimization_N_F") == 9990
        and exemplar.get("bytes") == EXEMPLAR_BATCH_BYTES
        and exemplar.get("remote_mode_octal") == "0444"
        and exemplar.get("local_beta_mode_octal") == "0600"
        and exemplar.get("remote_sha256") == EXEMPLAR_BATCH_SHA256
        and exemplar.get("local_beta_sha256") == EXEMPLAR_BATCH_SHA256
        and exemplar.get("remote_local_bytes_and_sha256_equal") is True
        and isinstance(diagnostics, list)
        and len(diagnostics) == 2
        and all(row.get("formal_input_eligible") is False for row in diagnostics),
        "local Mutagen transport receipt contents differ",
    )
    local_prestart = MUTAGEN_PRESTART.expanduser().resolve(strict=True)
    local_status = local_prestart.lstat()
    MODULE.require(
        stat.S_ISREG(local_status.st_mode)
        and not local_prestart.is_symlink()
        and stat.S_IMODE(local_status.st_mode) == 0o600
        and local_status.st_size == PRESTART_BYTES
        and sha256_file(local_prestart) == PRESTART_SHA256,
        "local Mutagen PRESTART does not match the frozen remote-0444 receipt",
    )
    exemplar_path = EXEMPLAR_BATCH.expanduser().resolve(strict=True)
    exemplar_status = exemplar_path.lstat()
    MODULE.require(
        stat.S_ISREG(exemplar_status.st_mode)
        and not exemplar_path.is_symlink()
        and stat.S_IMODE(exemplar_status.st_mode) == 0o600
        and exemplar_status.st_size == EXEMPLAR_BATCH_BYTES
        and sha256_file(exemplar_path) == EXEMPLAR_BATCH_SHA256,
        "local Mutagen exemplar shard does not match the frozen remote receipt",
    )
    _TRANSPORT_RECEIPT_CACHE = receipt
    return receipt


def _registered_writable_transport_role(path: Path, role: str) -> bool:
    """Return whether one exact mode-0600 replica path has the expected role."""

    resolved = path.expanduser().resolve(strict=False)
    if resolved == MUTAGEN_PRESTART:
        return role == "formal PRESTART"
    match = BATCH_NAME.fullmatch(resolved.name)
    if resolved.parent == MUTAGEN_ARCHIVE and match is not None:
        batch_id, nf = (int(value) for value in match.groups())
        return (
            1 <= batch_id <= TARGET_BATCHES
            and nf == 30 * batch_id
            and role in {f"published shard {batch_id}", "full-state shard"}
        )
    if resolved == MUTAGEN_COMPLETION:
        return role == "full-state completion" and MUTAGEN_TERMINAL_AUDIT.is_file()
    if resolved == MUTAGEN_TERMINAL_AUDIT:
        return role == "canonical terminal audit"
    return False


def transport_provenance() -> dict[str, Any]:
    _read_transport_receipt()
    wrapper = Path(__file__).expanduser().resolve(strict=True)
    return {
        "schema": "bb-nf30000-local-mutagen-replica-cache-binding-v1",
        "status": "PASS_EXACT_REPLICA_ADAPTER_BOUND",
        "wrapper": _exact_binding(wrapper),
        "transport_receipt": _exact_binding(TRANSPORT_RECEIPT),
        "mutagen_session_identifier": MUTAGEN_SESSION_ID,
        "exact_run_dir": str(MUTAGEN_MIRROR_RUN),
        "prestart": {
            "path": str(MUTAGEN_PRESTART),
            "sha256": PRESTART_SHA256,
            "bytes": PRESTART_BYTES,
            "local_mode_octal": "0600",
            "remote_receipt_mode_octal": "0444",
        },
        "preaudit_diagnostic_cache_reused": False,
        "physical_calls": 0,
    }


def mutagen_transport_readonly_regular(path: Path, role: str) -> None:
    """Accept mode-normalized local replicas only behind the WAL/hash gates.

    Mutagen preserves content but normalizes the remote immutable files to
    mode 0600 on this host.  The reused observer subsequently requires a
    matching ``batch_published`` record, optimizer ``batch_commit``, and shard
    SHA-256 before accepting any scientific state.  This narrow transport
    exception applies only to PRESTART and published batch shards in the exact
    registered local mirror; arbitrary writable files still fail closed.
    """

    _read_transport_receipt()
    resolved = path.expanduser().resolve(strict=True)
    status = resolved.lstat()
    MODULE.require(
        stat.S_ISREG(status.st_mode) and not resolved.is_symlink(),
        f"{role} is not a regular nonsymlink file",
    )
    if stat.S_IMODE(status.st_mode) & 0o222 == 0:
        return
    MODULE.require(
        stat.S_IMODE(status.st_mode) == 0o600
        and _registered_writable_transport_role(resolved, role),
        f"{role} is writable outside the exact Mutagen transport mirror policy",
    )


def import_mutagen_transport_source(root: Path):
    recorder, selection = _BASE_IMPORT_FROZEN_SOURCE(root)
    native_regular_file = recorder._regular_file

    def transport_regular_file(path: Path, label: str, *, readonly: bool) -> None:
        if readonly:
            mutagen_transport_readonly_regular(path, label)
        else:
            native_regular_file(path, label, readonly=False)

    recorder._regular_file = transport_regular_file
    return recorder, selection


def latest_bound_cache(output_dir: Path):
    path, value = _BASE_LATEST_CACHE(output_dir)
    if value is not None:
        MODULE.require(
            value.get("local_mutagen_replica_transport") == transport_provenance(),
            "previous incremental cache lacks the exact current replica binding",
        )
    return path, value


def write_bound_cache_json(path: Path, value: Mapping[str, Any]) -> None:
    document = dict(value)
    if document.get("schema") == CACHE_SCHEMA:
        MODULE.require(
            "local_mutagen_replica_transport" not in document,
            "base cache unexpectedly supplied replica provenance",
        )
        document["local_mutagen_replica_transport"] = transport_provenance()
    _BASE_WRITE_JSON_NEW(path, document)


def validate_runtime_adapter() -> None:
    wrapper = Path(__file__).expanduser().resolve(strict=True)
    status = wrapper.lstat()
    MODULE.require(
        stat.S_ISREG(status.st_mode)
        and not wrapper.is_symlink()
        and stat.S_IMODE(status.st_mode) == 0o444,
        "replica adapter wrapper must be frozen mode 0444 before cache creation",
    )
    _read_transport_receipt()


def validate_cli_scope(argv: list[str]) -> None:
    """Fail before installing the adapter unless every production path is exact."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terminal-audit", type=Path)
    args, unknown = parser.parse_known_args(argv)
    MODULE.require(not unknown, "unexpected cache-wrapper CLI arguments")
    MODULE.require(
        args.source_root.is_absolute()
        and args.source_root == FROZEN_SOURCE_ROOT
        and args.run_dir.is_absolute()
        and args.run_dir == MUTAGEN_MIRROR_RUN
        and args.output_dir.is_absolute()
        and args.output_dir == FORMAL_OUTPUT_DIR,
        "cache wrapper CLI is outside the exact registered source/run/output scope",
    )
    if args.terminal_audit is not None:
        MODULE.require(
            args.terminal_audit.is_absolute()
            and args.terminal_audit == MUTAGEN_TERMINAL_AUDIT,
            "cache wrapper terminal audit is not the exact canonical target",
        )


def install_adapter() -> None:
    MODULE.readonly_regular = mutagen_transport_readonly_regular
    MODULE.import_frozen_source = import_mutagen_transport_source
    MODULE.latest_cache = latest_bound_cache
    MODULE.write_json_new = write_bound_cache_json


def main() -> None:
    validate_cli_scope(sys.argv[1:])
    validate_runtime_adapter()
    install_adapter()
    MODULE.main()


if __name__ == "__main__":
    main()
