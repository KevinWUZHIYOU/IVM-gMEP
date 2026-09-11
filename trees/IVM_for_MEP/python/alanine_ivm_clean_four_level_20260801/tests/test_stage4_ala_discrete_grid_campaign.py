from __future__ import annotations

import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC = STUDY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_stage4_ala_discrete_grid_campaign as campaign


class GridConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.primary = campaign.build_primary_cells(STUDY_ROOT, "unit-grid")
        cls.secondary = campaign.build_secondary_cells(STUDY_ROOT, "unit-grid")

    def test_primary_is_exact_registered_240_cell_product(self) -> None:
        self.assertEqual(len(self.primary), 240)
        triples = {
            (
                row["controls"]["alpha"],
                row["controls"]["residual_ratio"],
                row["controls"]["inner_max_steps"],
            )
            for row in self.primary
        }
        self.assertEqual(len(triples), 240)
        self.assertEqual(len({row["cell_id"] for row in self.primary}), 240)
        self.assertTrue(
            all(
                row["controls"]["total_internal_nf_budget"] == 5000
                for row in self.primary
            )
        )

    def test_balanced_formula_has_no_parameter_level_host_confound(self) -> None:
        totals = collections.Counter(row["assigned_host"] for row in self.primary)
        self.assertEqual(totals, {"local": 120, "remote": 120})

        alpha = collections.Counter()
        ratio = collections.Counter()
        cap = collections.Counter()
        pair = collections.Counter()
        for row in self.primary:
            indices = row["grid_indices"]
            host = row["assigned_host"]
            alpha[(indices["alpha"], host)] += 1
            ratio[(indices["residual_ratio"], host)] += 1
            cap[(indices["maxstep"], host)] += 1
            pair[(indices["alpha"], indices["residual_ratio"], host)] += 1
        self.assertEqual(set(alpha.values()), {15})
        self.assertEqual(set(ratio.values()), {24})
        self.assertEqual(set(cap.values()), {20})
        self.assertEqual(set(pair.values()), {3})

    def test_single_host_override_is_total_and_explicit(self) -> None:
        rows = campaign.build_primary_cells(
            STUDY_ROOT, "unit-single", single_host_override="remote"
        )
        self.assertEqual({row["assigned_host"] for row in rows}, {"remote"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unit_preregistered.json"
            payload = campaign.build_preregistration(
                STUDY_ROOT,
                path,
                primary_single_host_override="remote",
                global_lock_path=Path(temporary) / "host.lock",
            )
        self.assertEqual(
            payload["primary_execution"]["mode"],
            "single_host_override_preregistered",
        )
        self.assertEqual(payload["primary_execution"]["required_hosts"], ["remote"])
        self.assertEqual(
            payload["secondary_execution"]["required_hosts"],
            ["local", "remote"],
        )
        self.assertEqual(payload["concurrency_policy"], campaign.CONCURRENCY_POLICY)
        self.assertEqual(
            payload["known_audit_limitations"], campaign.KNOWN_AUDIT_LIMITATIONS
        )
        limitation = payload["known_audit_limitations"][
            "inexact_terminal_partial_band_phase_decomposition"
        ]
        self.assertIn("all terminal cache misses", limitation["accounting"])
        self.assertIn("excluded from complete-state CTP", limitation["impact"])
        self.assertTrue(payload["concurrency_policy"]["bb_and_ala_may_overlap"])
        self.assertFalse(
            payload["concurrency_policy"]["bb_lock_consulted_by_ala_worker"]
        )

    def test_secondary_is_frozen_eight_alpha_four_four_shard(self) -> None:
        self.assertEqual(len(self.secondary), 8)
        self.assertEqual(
            collections.Counter(row["assigned_host"] for row in self.secondary),
            {"local": 4, "remote": 4},
        )
        self.assertEqual(len({row["cell_id"] for row in self.secondary}), 8)
        self.assertTrue(
            all(
                row["priority"] == "secondary_after_primary_aggregate"
                for row in self.secondary
            )
        )

    def test_secondary_single_host_override_preserves_scientific_identity(self) -> None:
        overridden = campaign.build_secondary_cells(
            STUDY_ROOT,
            "unit-secondary-single",
            single_host_override="local",
        )
        self.assertEqual({row["assigned_host"] for row in overridden}, {"local"})
        self.assertEqual(
            [row["cell_id"] for row in overridden],
            [row["cell_id"] for row in self.secondary],
        )
        self.assertTrue(
            all(
                row["wrapper_command"][-2:] == ["--execution-host", "local"]
                for row in overridden
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unit_preregistered.json"
            payload = campaign.build_preregistration(
                STUDY_ROOT,
                path,
                secondary_single_host_override="local",
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                campaign.load_and_validate_preregistration(STUDY_ROOT, path),
                payload,
            )
        self.assertEqual(
            payload["secondary_execution"],
            {
                "mode": "single_host_override_preregistered",
                "assignment_formula": None,
                "single_host_override": "local",
                "required_hosts": ["local"],
                "cell_count": 8,
                "primary_barrier": "validated_primary_aggregate",
            },
        )
        self.assertEqual(
            set(payload["planned_outputs"]["host_reports"][campaign.SECONDARY_FAMILY]),
            {"local"},
        )

    def test_host_reports_are_independent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unit_preregistered.json"
            payload = campaign.build_preregistration(
                STUDY_ROOT,
                path,
            )
        reports = payload["planned_outputs"]["host_reports"]
        observed = {
            reports[family][host]
            for family in campaign.FAMILIES
            for host in campaign.HOSTS
        }
        self.assertEqual(len(observed), 4)

    def test_preregistration_round_trip_rebuilds_every_registered_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roundtrip_preregistered.json"
            payload = campaign.build_preregistration(
                STUDY_ROOT,
                path,
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = campaign.load_and_validate_preregistration(STUDY_ROOT, path)
            self.assertEqual(loaded, payload)

            payload["cells"][campaign.PRIMARY_FAMILY][0][
                "scientific_command_sha256"
            ] = ("0" * 64)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                campaign.CampaignSafetyError, "registered primary cells drifted"
            ):
                campaign.load_and_validate_preregistration(STUDY_ROOT, path)

    def test_preregistration_rejects_execution_plan_and_source_path_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strict_preregistered.json"
            payload = campaign.build_preregistration(STUDY_ROOT, path)
            payload["secondary_execution"]["required_hosts"] = ["local"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                campaign.CampaignSafetyError, "secondary execution plan drifted"
            ):
                campaign.load_and_validate_preregistration(STUDY_ROOT, path)

            payload = campaign.build_preregistration(STUDY_ROOT, path)
            payload["source_manifest"]["campaign"]["path"] = str(
                STUDY_ROOT
                / "src/run_stage4_ala_discrete_inexact_reparam_grid_member.py"
            )
            payload["source_manifest"]["campaign"]["sha256"] = campaign.file_sha256(
                STUDY_ROOT
                / "src/run_stage4_ala_discrete_inexact_reparam_grid_member.py"
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                campaign.CampaignSafetyError, "source path drifted"
            ):
                campaign.load_and_validate_preregistration(STUDY_ROOT, path)


class SequentialExecutionTests(unittest.TestCase):
    def test_jobs_is_hard_fixed_to_one(self) -> None:
        self.assertEqual(campaign.validate_jobs(1), 1)
        for value in (0, 2, 8):
            with self.assertRaises(campaign.CampaignSafetyError):
                campaign.validate_jobs(value)

    def test_formal_preregister_cannot_bypass_ala_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(campaign.CampaignSafetyError, "host-local Ala"):
                campaign.preregister(
                    STUDY_ROOT,
                    global_lock_path=Path(temporary) / "private.lock",
                )

    def test_verified_cells_skip_and_absent_cells_execute_once(self) -> None:
        cells = [{"cell_id": "one"}, {"cell_id": "two"}]
        complete = {"one"}
        launched = []

        def inspect(cell):
            if cell["cell_id"] in complete:
                return "verified", {
                    "cell_id": cell["cell_id"],
                    "assigned_host": "local",
                }
            return "absent", None

        def launch(cell):
            launched.append(cell["cell_id"])
            complete.add(cell["cell_id"])

        rows = campaign.execute_cell_sequence(cells, inspect, launch)
        self.assertEqual(launched, ["two"])
        self.assertEqual(
            [row["action"] for row in rows], ["validated_skip", "executed"]
        )

    def test_launched_cell_must_validate_before_next_cell(self) -> None:
        cells = [{"cell_id": "bad"}]

        def inspect(unused):
            return "absent", None

        with self.assertRaises(campaign.CampaignSafetyError):
            campaign.execute_cell_sequence(cells, inspect, lambda unused: None)

    def test_exact_order_coverage_rejects_duplicates_or_reordering(self) -> None:
        cells = [{"cell_id": "a"}, {"cell_id": "b"}]
        campaign._assert_report_coverage(cells, [{"cell_id": "a"}, {"cell_id": "b"}])
        with self.assertRaises(campaign.CampaignSafetyError):
            campaign._assert_report_coverage(
                cells, [{"cell_id": "a"}, {"cell_id": "a"}]
            )
        with self.assertRaises(campaign.CampaignSafetyError):
            campaign._assert_report_coverage(
                cells, [{"cell_id": "b"}, {"cell_id": "a"}]
            )

    def test_json_outputs_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            campaign._write_json_exclusive(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                campaign._write_json_exclusive(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text())["value"], 1)

    def test_secondary_artifact_quarantines_charged_incomplete_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "results"
            result_dir.mkdir()
            result = result_dir / "result.npz"
            companion = result_dir / "result.audit.json"
            result.write_bytes(b"synthetic-result")
            companion.write_text("{}", encoding="utf-8")
            member_log = directory / "member-log"
            member_log.mkdir()
            member_audit = directory / "member-audit.json"
            command = ["python", "synthetic-exact-member.py"]
            command_sha256 = campaign.payload_sha256(command)
            cell = {
                "cell_id": "exact-tail-cell",
                "assigned_host": "local",
                "execution_host_override": None,
                "controls": {"alpha": 1.0},
                "paths": {
                    "result_dir": str(result_dir),
                    "result_glob": "*.npz",
                    "companion_glob": "*.audit.json",
                    "audit": str(member_audit),
                    "member_log": str(member_log),
                },
                "scientific_command": command,
                "scientific_command_sha256": command_sha256,
            }
            member_audit.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "result": str(result.resolve()),
                        "result_sha256": campaign.file_sha256(result),
                        "command": command,
                        "scientific_command_sha256": command_sha256,
                        "assigned_host": "local",
                        "cell_id": cell["cell_id"],
                        "default_assigned_host": "local",
                        "execution_host_override": None,
                    }
                ),
                encoding="utf-8",
            )
            secondary = mock.Mock()
            secondary.assigned_host.return_value = "local"
            secondary.validate_result.return_value = {
                "runner_counter_audit": str(companion.resolve()),
                "runner_counter_audit_sha256": campaign.file_sha256(companion),
                "configuration_sha256": "c" * 64,
                "final_optimization_NF": 5000,
                "last_exact_complete_NF": 4960,
                "trajectory_record_count": 249,
            }
            with mock.patch.object(
                campaign, "_members", return_value=(mock.Mock(), secondary)
            ):
                artifact = campaign._validate_secondary_artifact(STUDY_ROOT, cell)

            self.assertEqual(artifact["final_optimization_NF"], 5000)
            self.assertEqual(artifact["last_complete_state_NF"], 4960)


class LockAndBarrierTests(unittest.TestCase):
    @staticmethod
    def _identity_for_host(host: str, machine_suffix: str = ""):
        identity = campaign.worker_runtime_identity("local")
        identity = json.loads(json.dumps(identity))
        identity["worker_host_label"] = host
        if machine_suffix:
            identity["machine"]["hostname"] += machine_suffix
            identity["machine"]["platform_node"] += machine_suffix
        identity["machine_fingerprint"] = campaign.payload_sha256(identity["machine"])
        without_hash = {
            key: value for key, value in identity.items() if key != "identity_sha256"
        }
        identity["identity_sha256"] = campaign.payload_sha256(without_hash)
        return identity

    def test_dual_host_attestation_requires_distinct_machine_and_same_stack(
        self,
    ) -> None:
        local = self._identity_for_host("local")
        remote = self._identity_for_host("remote", "-remote")
        fingerprints, runtime_hash = campaign._runtime_set_attestation(
            {"local": local, "remote": remote}
        )
        self.assertEqual(len(set(fingerprints.values())), 2)
        self.assertEqual(runtime_hash, campaign._scientific_runtime_sha256(local))

        duplicate_machine = self._identity_for_host("remote")
        with self.assertRaisesRegex(campaign.CampaignSafetyError, "distinct machines"):
            campaign._runtime_set_attestation(
                {"local": local, "remote": duplicate_machine}
            )

        mismatched_stack = self._identity_for_host("remote", "-remote")
        mismatched_stack["scientific_distributions"]["OpenMM"] = "different"
        without_hash = {
            key: value
            for key, value in mismatched_stack.items()
            if key != "identity_sha256"
        }
        mismatched_stack["identity_sha256"] = campaign.payload_sha256(without_hash)
        with self.assertRaisesRegex(
            campaign.CampaignSafetyError, "runtime inventories differ"
        ):
            campaign._runtime_set_attestation(
                {"local": local, "remote": mismatched_stack}
            )

    def test_secondary_runtime_must_match_primary_aggregate(self) -> None:
        primary = {"scientific_runtime_sha256": "a" * 64}
        campaign._require_runtime_matches_primary(
            primary, "a" * 64, "secondary aggregate"
        )
        with self.assertRaisesRegex(
            campaign.CampaignSafetyError, "differs from primary campaign"
        ):
            campaign._require_runtime_matches_primary(
                primary, "b" * 64, "secondary aggregate"
            )

    def _child_lock(self, path: Path) -> subprocess.CompletedProcess:
        source = (
            "from pathlib import Path; "
            "import run_stage4_ala_discrete_grid_campaign as c; "
            f"p=Path({str(path)!r}); "
            "\ntry:\n"
            "  with c.HostLocalFcntlLock(p, wait=False): print('acquired')\n"
            "except c.HostLockHeldError:\n"
            "  print('blocked')\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SRC)
        return subprocess.run(
            [sys.executable, "-c", source],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_host_lock_excludes_other_process_for_whole_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "heavy.lock"
            with campaign.HostLocalFcntlLock(path, wait=False):
                blocked = self._child_lock(path)
                self.assertEqual(blocked.returncode, 0, blocked.stderr)
                self.assertEqual(blocked.stdout.strip(), "blocked")
            acquired = self._child_lock(path)
            self.assertEqual(acquired.returncode, 0, acquired.stderr)
            self.assertEqual(acquired.stdout.strip(), "acquired")

    def test_bb_lock_does_not_block_independent_ala_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bb_lock = directory / ".bb_global_heavy_task.lock"
            ala_lock = directory / ".ala_global_heavy_task.lock"
            with campaign.HostLocalFcntlLock(bb_lock, wait=False):
                acquired = self._child_lock(ala_lock)
                self.assertEqual(acquired.returncode, 0, acquired.stderr)
                self.assertEqual(acquired.stdout.strip(), "acquired")

    def test_secondary_fails_closed_without_primary_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-primary.json"
            protocol = {
                "planned_outputs": {"primary_aggregate": str(missing)},
                "primary_execution": {"mode": "single_host_override_preregistered"},
            }
            with self.assertRaisesRegex(
                campaign.CampaignSafetyError, "secondary is blocked"
            ):
                campaign.require_primary_barrier(
                    STUDY_ROOT,
                    Path(temporary) / "preregistered.json",
                    protocol,
                )

    def test_host_report_is_revalidated_against_fresh_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            preregistration = directory / "preregistered.json"
            preregistration.write_text("{}", encoding="utf-8")
            report_path = directory / "worker.json"
            cell = {"cell_id": "cell-a", "assigned_host": "local"}
            artifact = {
                "cell_id": "cell-a",
                "assigned_host": "local",
                "result_sha256": "a" * 64,
            }
            worker_identity = campaign.worker_runtime_identity("local")
            protocol = {
                "cells": {campaign.PRIMARY_FAMILY: [cell]},
                "primary_execution": {"required_hosts": ["local"]},
                "planned_outputs": {
                    "host_reports": {
                        campaign.PRIMARY_FAMILY: {"local": str(report_path)}
                    }
                },
                "host_local_global_lock": str(directory / "heavy.lock"),
            }
            report = {
                "status": "complete",
                "protocol_id": campaign.PROTOCOL_ID,
                "family": campaign.PRIMARY_FAMILY,
                "host_shard": "local",
                "preregistration": str(preregistration.resolve()),
                "preregistration_sha256": campaign.file_sha256(preregistration),
                "jobs": 1,
                "expected_cell_count": 1,
                "host_local_global_lock": str(directory / "heavy.lock"),
                "provenance_policy": campaign.PROVENANCE_POLICY,
                "worker_runtime_identity": worker_identity,
                "rows": [
                    {
                        "action": "executed",
                        "producer_provenance": "attested_worker_machine",
                        "producer_machine_fingerprint": worker_identity[
                            "machine_fingerprint"
                        ],
                        **artifact,
                    }
                ],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(
                campaign,
                "inspect_cell",
                return_value=("verified", artifact),
            ):
                observed = campaign.validate_host_report(
                    STUDY_ROOT,
                    preregistration,
                    protocol,
                    campaign.PRIMARY_FAMILY,
                    "local",
                )
            self.assertEqual(observed, report)

            stale = dict(artifact)
            stale["result_sha256"] = "b" * 64
            with mock.patch.object(
                campaign,
                "inspect_cell",
                return_value=("verified", stale),
            ):
                with self.assertRaisesRegex(
                    campaign.CampaignSafetyError, "artifact record drifted"
                ):
                    campaign.validate_host_report(
                        STUDY_ROOT,
                        preregistration,
                        protocol,
                        campaign.PRIMARY_FAMILY,
                        "local",
                    )

    def test_primary_aggregate_binds_host_report_hash_and_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            preregistration = directory / "preregistered.json"
            preregistration.write_text("{}", encoding="utf-8")
            report_path = directory / "worker.json"
            aggregate_path = directory / "aggregate.json"
            cells = [
                {"cell_id": "a", "assigned_host": "local"},
                {"cell_id": "b", "assigned_host": "local"},
            ]
            worker_identity = campaign.worker_runtime_identity("local")
            report = {
                "worker_runtime_identity": worker_identity,
                "rows": [{"cell_id": "a"}, {"cell_id": "b"}],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            protocol = {
                "cells": {campaign.PRIMARY_FAMILY: cells},
                "primary_execution": {"required_hosts": ["local"]},
                "planned_outputs": {
                    "host_reports": {
                        campaign.PRIMARY_FAMILY: {"local": str(report_path)}
                    },
                    "primary_aggregate": str(aggregate_path),
                },
            }
            aggregate = {
                "status": "complete",
                "protocol_id": campaign.PROTOCOL_ID,
                "family": campaign.PRIMARY_FAMILY,
                "preregistration": str(preregistration.resolve()),
                "preregistration_sha256": campaign.file_sha256(preregistration),
                "required_hosts": ["local"],
                "expected_cell_count": 2,
                "host_reports": {
                    "local": {
                        "path": str(report_path.resolve()),
                        "sha256": campaign.file_sha256(report_path),
                    }
                },
                "host_machine_fingerprints": {
                    "local": worker_identity["machine_fingerprint"]
                },
                "scientific_runtime_sha256": campaign._scientific_runtime_sha256(
                    worker_identity
                ),
            }
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
            with mock.patch.object(
                campaign, "validate_host_report", return_value=report
            ):
                observed = campaign._validate_aggregate_file(
                    STUDY_ROOT,
                    preregistration,
                    protocol,
                    campaign.PRIMARY_FAMILY,
                )
            self.assertEqual(observed, aggregate)

            report_path.write_text(
                json.dumps(
                    {
                        "worker_runtime_identity": worker_identity,
                        "rows": [{"cell_id": "a"}, {"cell_id": "a"}],
                    }
                ),
                encoding="utf-8",
            )
            report["rows"] = [{"cell_id": "a"}, {"cell_id": "a"}]
            aggregate["host_reports"]["local"]["sha256"] = campaign.file_sha256(
                report_path
            )
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
            with mock.patch.object(
                campaign, "validate_host_report", return_value=report
            ):
                with self.assertRaisesRegex(
                    campaign.CampaignSafetyError, "coverage is incomplete"
                ):
                    campaign._validate_aggregate_file(
                        STUDY_ROOT,
                        preregistration,
                        protocol,
                        campaign.PRIMARY_FAMILY,
                    )


if __name__ == "__main__":
    unittest.main()
