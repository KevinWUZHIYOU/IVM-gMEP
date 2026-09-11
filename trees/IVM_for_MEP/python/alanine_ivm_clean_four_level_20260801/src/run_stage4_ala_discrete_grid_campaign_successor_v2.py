#!/usr/bin/env python3
"""Isolated successor campaign for exact/no-reparam runtime v2.

The successor does not mutate or reinterpret the preregistered v1 campaign.
It revalidates the 240 completed primary artifacts into fresh host reports and
a fresh primary aggregate, then runs eight exact cells in new v2 namespaces.
The failed v1 exact attempts and the completed predecessor primary aggregate
are pinned as immutable provenance in every successor preregistration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional

import run_stage4_ala_discrete_grid_campaign as legacy
import run_stage4_ala_discrete_inexact_reparam_grid_member as primary_member
import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member_v2 as secondary_member


PROTOCOL_ID = "stage4_ala_discrete_grid_campaign_successor_v2"
PREDECESSOR_PROTOCOL_ID = legacy.PROTOCOL_ID
PRIMARY_FAMILY = legacy.PRIMARY_FAMILY
SECONDARY_FAMILY = legacy.SECONDARY_FAMILY
FAMILIES = legacy.FAMILIES
HOSTS = legacy.HOSTS
DEFAULT_GLOBAL_LOCK = legacy.DEFAULT_GLOBAL_LOCK
TOTAL_INTERNAL_NF_BUDGET = legacy.TOTAL_INTERNAL_NF_BUDGET
PRIMARY_CELL_COUNT = legacy.PRIMARY_CELL_COUNT
SECONDARY_CELL_COUNT = legacy.SECONDARY_CELL_COUNT
CampaignSafetyError = legacy.CampaignSafetyError
PartialCellError = legacy.PartialCellError

PREDECESSOR_PREREGISTRATION = (
    "reports/stage4_ala_discrete_grid_campaign_v1_"
    "1787512668957303000_preregistered.json"
)
PREDECESSOR_PREREGISTRATION_SHA256 = (
    "fbdd9ff93dc971841950dd42123f5caf414b59e83a8ba37116c9ecb5f72e83fd"
)
PREDECESSOR_PRIMARY_AGGREGATE = (
    "reports/stage4_ala_discrete_grid_campaign_v1_"
    "1787512668957303000_inexact_reparam_aggregate.json"
)
PREDECESSOR_PRIMARY_AGGREGATE_SHA256 = (
    "5d9a606a03d59f6ddb2168f7dbcea3e93715a7e7a72490b9a1b1f12a5119a1c2"
)
PREDECESSOR_SCIENTIFIC_RUNTIME_SHA256 = (
    "d34e76f312ac1ca097dcb13fef70f685911f00bb739916cbeb195147d034b6e2"
)

FAILED_V1_EVIDENCE = (
    {
        "cell_id": "f897abbf6a432fbd",
        "host": "local",
        "alpha_text": "1e-1",
        "member_log": (
            "logs/stage4_ala_discrete_exact_noreparam_alpha_grid_v1/members/"
            "local/stage4_ala_discrete_exact_noreparam_alpha_grid_v1_"
            "exact_no_reparam_local_f897abbf6a432fbd_a1em1_r01_unbounded_"
            "NF5000_r1"
        ),
        "worker_log": (
            "logs/stage4_ala_discrete_grid_campaign_v1/"
            "stage4_ala_discrete_grid_campaign_v1_1787512668957303000/"
            "exact_no_reparam/local/f897abbf6a432fbd"
        ),
        "files": {
            "member/command.json": (
                "df96e3a030119f4f727a152b2416eafb39a8d4c836cb6a361d5e49e69aea8924"
            ),
            "member/stderr.txt": (
                "599e8fa50271c07b88227cc8d7497b6f46a570d8f0013a12454320f1996dc428"
            ),
            "member/stdout.txt": (
                "3b3305c8cee63567e4eda17ce91bea5b05dedd9992452d027e9348004eaa0f4a"
            ),
            "worker/command.json": (
                "0529d91431d9acbab84f5423d18b7e1f900171334e3cb4bea740e06c6d9fece2"
            ),
            "worker/exit.json": (
                "afbff3d16d64e2540a2f72878bcdc286cd1bf0e28ef02ee3eff10253e4d52d79"
            ),
            "worker/stderr.txt": (
                "22960e456639659a126d7eb232e8255be8e9b3318e0e77f23bca6d6678d7fbdd"
            ),
            "worker/stdout.txt": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
    },
    {
        "cell_id": "0ec50c1aeffbb910",
        "host": "remote",
        "alpha_text": "1e0",
        "member_log": (
            "logs/stage4_ala_discrete_exact_noreparam_alpha_grid_v1/members/"
            "remote/stage4_ala_discrete_exact_noreparam_alpha_grid_v1_"
            "exact_no_reparam_remote_0ec50c1aeffbb910_a1e0_r01_unbounded_"
            "NF5000_r1"
        ),
        "worker_log": (
            "logs/stage4_ala_discrete_grid_campaign_v1/"
            "stage4_ala_discrete_grid_campaign_v1_1787512668957303000/"
            "exact_no_reparam/remote/0ec50c1aeffbb910"
        ),
        "files": {
            "member/command.json": (
                "91bbd70d5b4625b35bbfa7b31a646177200d4ac1934e75d24baa421d46cd6a57"
            ),
            "member/stderr.txt": (
                "599e8fa50271c07b88227cc8d7497b6f46a570d8f0013a12454320f1996dc428"
            ),
            "member/stdout.txt": (
                "67800b4c3775afafea1c920530cb1ae8678c40768c843b9564506cd5ed1edc9f"
            ),
            "worker/command.json": (
                "6a0c93a22d902fd19d1d028a5395ccaf95406f93daddead1b4d955467840b376"
            ),
            "worker/exit.json": (
                "1a6bd83f30b38a235ac622df12177d1e56f2d9fb0b97d80d92a93611d55bffbc"
            ),
            "worker/stderr.txt": (
                "41721f10cc220e36dc3abbe7948af7f7683c54d61bd4c11d08bafdd31eef7de5"
            ),
            "worker/stdout.txt": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
    },
)

SOURCE_PATHS = dict(legacy.SOURCE_PATHS)
SOURCE_PATHS.update(
    {
        "campaign": "src/run_stage4_ala_discrete_grid_campaign_successor_v2.py",
        "secondary_member": (
            "src/run_stage4_ala_discrete_exact_noreparam_alpha_grid_member_v2.py"
        ),
        "exact_no_reparam_runtime_v2": (
            "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim_v2.py"
        ),
        "superseded_exact_no_reparam_runtime_v1": (
            "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim.py"
        ),
        "superseded_secondary_member_v1": (
            "src/run_stage4_ala_discrete_exact_noreparam_alpha_grid_member.py"
        ),
        "predecessor_campaign_v1": (
            "src/run_stage4_ala_discrete_grid_campaign.py"
        ),
    }
)
SOURCE_PATHS.pop("exact_no_reparam_runtime", None)


def _members() -> tuple[Any, Any]:
    return primary_member, secondary_member


@contextmanager
def _successor_context() -> Iterator[None]:
    previous_protocol = legacy.PROTOCOL_ID
    previous_paths = legacy.SOURCE_PATHS
    previous_members = legacy._members
    legacy.PROTOCOL_ID = PROTOCOL_ID
    legacy.SOURCE_PATHS = SOURCE_PATHS
    legacy._members = _members
    try:
        yield
    finally:
        legacy._members = previous_members
        legacy.SOURCE_PATHS = previous_paths
        legacy.PROTOCOL_ID = previous_protocol


def _failed_evidence_payload(root: Path, verify: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for frozen in FAILED_V1_EVIDENCE:
        member_dir = (root / str(frozen["member_log"])).resolve()
        worker_dir = (root / str(frozen["worker_log"])).resolve()
        files: List[Dict[str, str]] = []
        for logical_name, expected_hash in dict(frozen["files"]).items():
            scope, filename = logical_name.split("/", 1)
            path = (member_dir if scope == "member" else worker_dir) / filename
            if verify and (
                not path.is_file()
                or legacy.file_sha256(path) != str(expected_hash)
            ):
                raise CampaignSafetyError(
                    f"failed-v1 evidence drifted: {path}"
                )
            files.append(
                {
                    "logical_name": logical_name,
                    "path": str(path),
                    "sha256": str(expected_hash),
                }
            )
        records.append(
            {
                "cell_id": frozen["cell_id"],
                "host": frozen["host"],
                "alpha_text": frozen["alpha_text"],
                "member_log": str(member_dir),
                "worker_log": str(worker_dir),
                "failure": {
                    "exception": "KeyError: 'path'",
                    "source": (
                        "run_ivm_v9_discrete_ffrbk_takeover_"
                        "exact_noreparam_shim.py:110"
                    ),
                    "root_cause": (
                        "completion audit bound the variadic outer wrapper "
                        "instead of the frozen candidate call signature"
                    ),
                },
                "files": files,
            }
        )
    return records


def _validate_predecessor(root: Path) -> Dict[str, Any]:
    root = legacy._root(root)
    prereg_path = (root / PREDECESSOR_PREREGISTRATION).resolve()
    aggregate_path = (root / PREDECESSOR_PRIMARY_AGGREGATE).resolve()
    for path, expected in (
        (prereg_path, PREDECESSOR_PREREGISTRATION_SHA256),
        (aggregate_path, PREDECESSOR_PRIMARY_AGGREGATE_SHA256),
    ):
        if not path.is_file() or legacy.file_sha256(path) != expected:
            raise CampaignSafetyError(f"predecessor provenance drifted: {path}")
    preregistration = legacy.load_and_validate_preregistration(root, prereg_path)
    aggregate = legacy._validate_aggregate_file(
        root,
        prereg_path,
        preregistration,
        PRIMARY_FAMILY,
    )
    if (
        aggregate.get("status") != "complete"
        or int(aggregate.get("expected_cell_count", -1)) != PRIMARY_CELL_COUNT
        or int(aggregate.get("unique_cell_count", -1)) != PRIMARY_CELL_COUNT
        or aggregate.get("scientific_runtime_sha256")
        != PREDECESSOR_SCIENTIFIC_RUNTIME_SHA256
    ):
        raise CampaignSafetyError("predecessor primary aggregate is not reusable")
    failed = _failed_evidence_payload(root, verify=True)
    return {
        "protocol_id": PREDECESSOR_PROTOCOL_ID,
        "preregistration": {
            "path": str(prereg_path),
            "sha256": PREDECESSOR_PREREGISTRATION_SHA256,
        },
        "completed_primary_aggregate": {
            "path": str(aggregate_path),
            "sha256": PREDECESSOR_PRIMARY_AGGREGATE_SHA256,
            "scientific_runtime_sha256": (
                PREDECESSOR_SCIENTIFIC_RUNTIME_SHA256
            ),
            "complete_cell_count": PRIMARY_CELL_COUNT,
        },
        "primary_reuse_policy": (
            "same preregistered scientific cells; freshly revalidate every "
            "artifact, record validated_skip rows, and create successor host "
            "reports plus successor primary aggregate"
        ),
        "failed_exact_v1": failed,
        "failed_v1_reuse_allowed": False,
        "failed_v1_artifacts_mutated": False,
    }


def _stable_primary_cell(cell: Mapping[str, Any]) -> Dict[str, Any]:
    stable = dict(cell)
    paths = dict(stable["paths"])
    paths.pop("worker_log")
    stable["paths"] = paths
    return stable


def build_primary_cells(
    root: Path,
    campaign_stem: str,
    single_host_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with _successor_context():
        return legacy.build_primary_cells(
            root, campaign_stem, single_host_override
        )


def build_secondary_cells(
    root: Path,
    campaign_stem: str,
    single_host_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with _successor_context():
        return legacy.build_secondary_cells(
            root, campaign_stem, single_host_override
        )


def build_preregistration(
    root: Path,
    preregistration_path: Path,
    primary_single_host_override: Optional[str] = None,
    secondary_single_host_override: Optional[str] = None,
    global_lock_path: Path = Path(DEFAULT_GLOBAL_LOCK),
) -> Dict[str, Any]:
    root = legacy._root(root)
    predecessor = _validate_predecessor(root)
    with _successor_context():
        payload = legacy.build_preregistration(
            root,
            preregistration_path,
            primary_single_host_override=primary_single_host_override,
            secondary_single_host_override=secondary_single_host_override,
            global_lock_path=global_lock_path,
        )
    old = legacy._read_json(
        Path(str(predecessor["preregistration"]["path"]))
    )
    observed = payload["cells"][PRIMARY_FAMILY]
    expected = old["cells"][PRIMARY_FAMILY]
    if len(observed) != len(expected) or any(
        _stable_primary_cell(left) != _stable_primary_cell(right)
        for left, right in zip(observed, expected)
    ):
        raise CampaignSafetyError(
            "successor primary cells differ from predecessor scientific cells"
        )
    payload["successor_protocol"] = {
        "version": 2,
        "reason": "exact-v1-completion-audit-call-signature-failure",
        "primary_artifact_action": "validated_skip_only",
        "primary_reports_and_aggregate": "fresh_successor_namespace",
        "secondary_runtime_policy_id": secondary_member.runtime.POLICY_ID,
        "secondary_member_policy_version": secondary_member.MEMBER_POLICY_VERSION,
    }
    payload["predecessor_provenance"] = predecessor
    return payload


def _validate_successor_fields(root: Path, payload: Mapping[str, Any]) -> None:
    expected_predecessor = _validate_predecessor(root)
    expected_successor = {
        "version": 2,
        "reason": "exact-v1-completion-audit-call-signature-failure",
        "primary_artifact_action": "validated_skip_only",
        "primary_reports_and_aggregate": "fresh_successor_namespace",
        "secondary_runtime_policy_id": secondary_member.runtime.POLICY_ID,
        "secondary_member_policy_version": secondary_member.MEMBER_POLICY_VERSION,
    }
    if payload.get("successor_protocol") != expected_successor:
        raise CampaignSafetyError("successor protocol metadata drifted")
    if payload.get("predecessor_provenance") != expected_predecessor:
        raise CampaignSafetyError("predecessor provenance drifted")


def preregister(
    root: Path,
    primary_single_host_override: Optional[str] = None,
    secondary_single_host_override: Optional[str] = None,
    global_lock_path: Path = Path(DEFAULT_GLOBAL_LOCK),
) -> Path:
    root = legacy._root(root)
    if global_lock_path.expanduser().resolve() != Path(DEFAULT_GLOBAL_LOCK).resolve():
        raise CampaignSafetyError(
            "formal preregistration requires the host-local Ala heavy-task lock"
        )
    stem = f"{PROTOCOL_ID}_{time.time_ns()}"
    path = (root / "reports" / f"{stem}_preregistered.json").resolve()
    payload = build_preregistration(
        root,
        path,
        primary_single_host_override=primary_single_host_override,
        secondary_single_host_override=secondary_single_host_override,
        global_lock_path=global_lock_path,
    )
    legacy._write_json_exclusive(path, payload)
    return path


def load_and_validate_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    root = legacy._root(root)
    with _successor_context():
        payload = legacy.load_and_validate_preregistration(root, path)
    _validate_successor_fields(root, payload)
    return payload


def require_primary_barrier(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Dict[str, Any]:
    with _successor_context():
        return legacy.require_primary_barrier(
            root, preregistration_path, preregistration
        )


def run_host_worker(
    root: Path,
    preregistration_path: Path,
    family: str,
    host: str,
    jobs: int = 1,
    wait_for_lock: bool = False,
) -> Path:
    load_and_validate_preregistration(root, preregistration_path)
    with _successor_context():
        return legacy.run_host_worker(
            root,
            preregistration_path,
            family,
            host,
            jobs=jobs,
            wait_for_lock=wait_for_lock,
        )


def aggregate_family(root: Path, preregistration_path: Path, family: str) -> Path:
    load_and_validate_preregistration(root, preregistration_path)
    with _successor_context():
        return legacy.aggregate_family(root, preregistration_path, family)


def argument_parser() -> argparse.ArgumentParser:
    return legacy.argument_parser()


def main(argv: Optional[List[str]] = None) -> None:
    args = argument_parser().parse_args(argv)
    if args.action == "preregister":
        path = preregister(
            args.root,
            primary_single_host_override=args.primary_single_host_override,
            secondary_single_host_override=args.secondary_single_host_override,
        )
    elif args.action == "run-shard":
        path = run_host_worker(
            args.root,
            args.preregistration,
            args.family,
            args.host,
            jobs=args.jobs,
            wait_for_lock=args.wait_for_lock,
        )
    elif args.action == "aggregate":
        path = aggregate_family(args.root, args.preregistration, args.family)
    else:
        raise AssertionError("unreachable action")
    print(path)


if __name__ == "__main__":
    main()
