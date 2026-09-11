"""Zero-physical tests for the R5B-bound cost-only late-six selector."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/select_bb_irivm_cost_only_late_six_r3.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "bb_irivm_cost_only_late_six_r3_tested", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


S = load_module()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class GuardedRow(Mapping):
    """Expose forbidden keys but fail if a selector tries to read their value."""

    def __init__(self, allowed: dict):
        self.allowed = dict(allowed)
        self.extra = {
            "i_perp": object(),
            "densified_Delta_E": object(),
            "energy": object(),
            "force": object(),
            "barrier": object(),
        }

    def __getitem__(self, key):
        if key in self.extra:
            raise AssertionError("selector read forbidden field {}".format(key))
        return self.allowed[key]

    def __iter__(self):
        return iter(tuple(self.allowed) + tuple(self.extra))

    def __len__(self):
        return len(self.allowed) + len(self.extra)


class CostOnlyLateSixR3Test(unittest.TestCase):
    def base_row(self, call: int, nf: int, **changes) -> dict:
        value = {
            "history_index": call - 1,
            "call_per_image": call,
            "optimization_N_F": nf,
            "selected_path_sha256": "{:064x}".format(call),
            "source_full_state_shard": "/synthetic/shard_{:04d}.npz".format(call),
            "source_full_state_shard_sha256": "{:064x}".format(1000 + call),
            "reparameterization_committed": True,
            "fully_evaluated_current_path": True,
            "stable_last": True,
            "formally_reportable": True,
            "terminal_uncommitted_charged_tail": False,
        }
        value.update(changes)
        return value

    def candidate_inventory(self):
        nfs = [
            60,
            150,
            300,
            1329,
            1900,
            2100,
            3900,
            4100,
            6000,
            8000,
            8982,
            14970,
            14990,
        ]
        rows = [self.base_row(index + 1, nf) for index, nf in enumerate(nfs)]
        rows[-1]["formally_reportable"] = False
        rows[-1]["terminal_uncommitted_charged_tail"] = True
        return rows

    def make_final_closure(self, directory: Path):
        rows = self.candidate_inventory()
        calls = np.arange(1, len(rows) + 1, dtype=np.int64)
        nfs = np.asarray([row["optimization_N_F"] for row in rows], dtype=np.int64)
        committed = np.ones(len(rows), dtype=bool)
        result = directory / "r5b_result.npz"
        np.savez(
            result,
            calls_per_image=calls,
            total_internal_joint_evaluations=nfs,
            reparam_committed=committed,
            total_internal_nf_last_complete_state_index=np.asarray(
                len(rows) - 2, dtype=np.int64
            ),
            i_perp=np.full(len(rows), -999.0),
            barrier_kcal=np.full(len(rows), -888.0),
            energy_eV=np.full((len(rows), 32), -777.0),
            force_eV_per_A=np.full((len(rows), 32, 2), -666.0),
        )
        records = []
        primary_inventory = []
        for row in rows:
            records.append(
                {
                    "history_index": row["history_index"],
                    "call_per_image": row["call_per_image"],
                    "optimization_N_F": row["optimization_N_F"],
                    "path_sha256": row["selected_path_sha256"],
                    "shard": row["source_full_state_shard"],
                    "shard_sha256": row["source_full_state_shard_sha256"],
                }
            )
            primary_inventory.append(
                {
                    **records[-1],
                    "formally_reportable": row["formally_reportable"],
                    "terminal_uncommitted_charged_tail": row[
                        "terminal_uncommitted_charged_tail"
                    ],
                    "energy_sha256": "{:064x}".format(2000 + row["call_per_image"]),
                    "force_sha256": "{:064x}".format(3000 + row["call_per_image"]),
                    "i_perp": -12345.0,
                }
            )
        index = directory / "full_state_history.canonical_index.json"
        write_json(
            index,
            {
                "schema": S.INDEX_SCHEMA,
                "status": "COMPLETE_PENDING_EXACT_RESULT_VERIFICATION",
                "canonical_record_count": len(records),
                "records": records,
            },
        )
        manifest = directory / "full_state_history.manifest.json"
        write_json(
            manifest,
            {
                "schema": S.MANIFEST_SCHEMA,
                "status": "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION",
                "canonical_record_count": len(records),
                "canonical_index": str(index.resolve()),
                "canonical_index_sha256": S.sha256_file(index),
                "formally_reportable_call_range": [1, len(rows) - 1],
                "canonical_records": records,
            },
        )
        capture_audit = directory / "full_state_history.audit.json"
        write_json(capture_audit, {"status": "synthetic"})
        wal = directory / "full_state_history.wal.jsonl"
        wal.write_text('{"event":"synthetic"}\n', encoding="utf-8")
        prereg = S.read_json(S.CANONICAL_PREREGISTRATION)
        primary = directory / "primary.json"
        write_json(
            primary,
            {
                "schema": S.PRIMARY_SCHEMA,
                "status": S.PRIMARY_STATUS,
                "actual_canonical_call_count": len(records),
                "last_complete_call": len(rows) - 1,
                "path_inventory": primary_inventory,
                "bindings": {
                    "registration": prereg["bindings"]["registration"],
                    "authorization": prereg["bindings"]["authorization"],
                    "capture_manifest": S.bind(manifest),
                    "canonical_index": S.bind(index),
                    "capture_audit": S.bind(capture_audit),
                    "replay_result": S.bind(result),
                },
            },
        )
        supplemental = directory / "supplemental.json"
        write_json(
            supplemental,
            {
                "schema": S.SUPPLEMENTAL_SCHEMA,
                "status": S.SUPPLEMENTAL_STATUS,
                "canonical_record_count": len(records),
                "bindings": {
                    "registration": prereg["bindings"]["registration"],
                    "authorization": prereg["bindings"]["authorization"],
                    "primary_verification": S.bind(primary),
                    "manifest": S.bind(manifest),
                    "canonical_index": S.bind(index),
                    "capture_audit": S.bind(capture_audit),
                    "wal": S.bind(wal),
                },
            },
        )
        return {
            "result": result,
            "index": index,
            "manifest": manifest,
            "primary": primary,
            "supplemental": supplemental,
            "capture_audit": capture_audit,
            "wal": wal,
        }

    def build_synthetic_selection(self, closure):
        return S.build_selection(
            S.CANONICAL_PREREGISTRATION,
            S.CANONICAL_SPLIT_TEMPLATE,
            closure["primary"],
            closure["supplemental"],
            closure["manifest"],
            closure["index"],
            enforce_canonical=False,
        )

    def test_frozen_scientific_policy_and_balanced_split(self):
        prereg = S.validate_preregistration(S.CANONICAL_PREREGISTRATION)
        template = S.validate_split_template(
            S.CANONICAL_SPLIT_TEMPLATE, S.CANONICAL_PREREGISTRATION
        )
        self.assertEqual(prereg["early_four_optimization_N_F"], [60, 150, 300, 1329])
        self.assertEqual(
            prereg["selection_policy"]["anchor_targets_N_F"],
            [2000, 4000, 6000, 8000, 8982],
        )
        self.assertFalse(
            prereg["selection_policy"]["anchors_changed_from_old_preregistration"]
        )
        self.assertEqual(
            prereg["selection_policy"]["maximum_legal_optimization_N_F"], 15000
        )
        self.assertEqual(template["remote_four"]["selection_roles"], list(S.REMOTE_ROLES))
        self.assertEqual(template["local_two"]["selection_roles"], list(S.LOCAL_ROLES))
        self.assertEqual(template["workload"]["expected_fresh_calls_remote"], 1992)
        self.assertEqual(template["workload"]["expected_fresh_calls_local"], 996)
        self.assertEqual(template["workload"]["expected_fresh_calls_total"], 2988)
        self.assertEqual(template["workload"]["optimization_N_F_contribution"], 0)

    def test_preregistration_reads_only_frozen_pre_final_inputs(self):
        original = S.read_json
        seen = []

        def observed(path):
            seen.append(Path(path).resolve())
            return original(path)

        with mock.patch.object(S, "read_json", side_effect=observed):
            payload = S.build_preregistration()
        self.assertFalse(payload["R5B_final_artifacts_read"])
        self.assertFalse(payload["metric_values_read_by_selector_or_this_freezer"])
        self.assertEqual(
            set(seen),
            {
                S.R5B_REGISTRATION,
                S.R5B_AUTHORIZATION,
                S.R5B_MUTAGEN_GATE,
                S.OLD_SELECTION_PREREG,
            },
        )
        self.assertNotIn(S.CANONICAL_PRIMARY, seen)
        self.assertNotIn(S.CANONICAL_MANIFEST, seen)

    def test_selection_projection_does_not_read_metric_fields(self):
        guarded = [GuardedRow(row) for row in self.candidate_inventory()]
        selected = S.select_rows(guarded)
        self.assertEqual(len(selected), 6)
        self.assertNotIn("i_perp", selected[0])
        self.assertNotIn("energy", selected[0])
        self.assertNotIn("force", selected[0])

    def test_highest_first_old_anchors_and_terminal_tail_exclusion(self):
        selected = S.select_rows(self.candidate_inventory())
        by_role = {row["selection_role"]: row for row in selected}
        self.assertEqual(by_role[S.HIGHEST_ROLE]["optimization_N_F"], 14970)
        self.assertEqual(by_role["nearest_unused_to_NF2000"]["optimization_N_F"], 1900)
        self.assertEqual(by_role["nearest_unused_to_NF4000"]["optimization_N_F"], 3900)
        self.assertEqual(by_role["nearest_unused_to_NF6000"]["optimization_N_F"], 6000)
        self.assertEqual(by_role["nearest_unused_to_NF8000"]["optimization_N_F"], 8000)
        self.assertEqual(by_role["nearest_unused_to_NF8982"]["optimization_N_F"], 8982)
        self.assertNotIn(14990, [row["optimization_N_F"] for row in selected])
        self.assertEqual(
            len({row["selected_path_sha256"] for row in selected}), 6
        )

    def test_uncommitted_nearest_path_is_ineligible(self):
        rows = self.candidate_inventory()
        rows[4]["reparameterization_committed"] = False
        selected = S.select_rows(rows)
        by_role = {row["selection_role"]: row for row in selected}
        self.assertEqual(by_role["nearest_unused_to_NF2000"]["optimization_N_F"], 2100)

    def test_replay_loader_accesses_only_allowed_eligibility_arrays(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            closure = self.make_final_closure(directory)
            real_load = np.load
            accessed = []

            class Tracking:
                def __init__(self, path, **kwargs):
                    self.saved = real_load(path, **kwargs)
                    self.files = self.saved.files

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.saved.close()

                def __getitem__(self, key):
                    accessed.append(key)
                    if key not in S.REPLAY_ELIGIBILITY_KEYS:
                        raise AssertionError("forbidden array read: {}".format(key))
                    return self.saved[key]

            with mock.patch.object(S.np, "load", side_effect=Tracking):
                state = S.load_replay_eligibility(closure["result"])
            self.assertEqual(set(accessed), set(S.REPLAY_ELIGIBILITY_KEYS))
            self.assertEqual(state["last_complete_call"], 12)

    def test_dual_pass_closure_selects_and_binds_remote4_local2(self):
        with tempfile.TemporaryDirectory() as raw:
            closure = self.make_final_closure(Path(raw))
            payload = self.build_synthetic_selection(closure)
        self.assertEqual(payload["status"], S.SELECTION_STATUS)
        self.assertFalse(payload["selection_metric_values_consulted"])
        self.assertTrue(
            payload["prior_metric_values_may_have_been_observed_outside_selector"]
        )
        self.assertEqual(payload["forbidden_selection_fields_read"], [])
        self.assertEqual(len(payload["host_assignments"]["remote"]), 4)
        self.assertEqual(len(payload["host_assignments"]["local"]), 2)
        self.assertEqual(
            {
                row["selection_role"]
                for rows in payload["host_assignments"].values()
                for row in rows
            },
            set(S.ROLE_ORDER),
        )
        self.assertEqual(payload["expected_fresh_dense_joint_EF_calls"]["total"], 2988)
        self.assertEqual(payload["optimization_N_F_contribution"], 0)

    def test_primary_status_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            closure = self.make_final_closure(Path(raw))
            primary = S.read_json(closure["primary"])
            primary["status"] = "NOT_PASS"
            write_json(closure["primary"], primary)
            supplemental = S.read_json(closure["supplemental"])
            supplemental["bindings"]["primary_verification"] = S.bind(
                closure["primary"]
            )
            write_json(closure["supplemental"], supplemental)
            with self.assertRaisesRegex(ValueError, "primary R5B verification"):
                self.build_synthetic_selection(closure)

    def test_supplemental_primary_binding_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            closure = self.make_final_closure(Path(raw))
            supplemental = S.read_json(closure["supplemental"])
            supplemental["bindings"]["primary_verification"]["sha256"] = "0" * 64
            write_json(closure["supplemental"], supplemental)
            with self.assertRaisesRegex(ValueError, "supplemental closure"):
                self.build_synthetic_selection(closure)

    def test_manifest_index_binding_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            closure = self.make_final_closure(Path(raw))
            manifest = S.read_json(closure["manifest"])
            manifest["canonical_index_sha256"] = "0" * 64
            write_json(closure["manifest"], manifest)
            primary = S.read_json(closure["primary"])
            primary["bindings"]["capture_manifest"] = S.bind(closure["manifest"])
            write_json(closure["primary"], primary)
            supplemental = S.read_json(closure["supplemental"])
            supplemental["bindings"]["primary_verification"] = S.bind(
                closure["primary"]
            )
            supplemental["bindings"]["manifest"] = S.bind(closure["manifest"])
            write_json(closure["supplemental"], supplemental)
            with self.assertRaisesRegex(ValueError, "manifest/index"):
                self.build_synthetic_selection(closure)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "frozen.json"
            S.write_exclusive(path, {"first": True})
            with self.assertRaises(FileExistsError):
                S.write_exclusive(path, {"second": True})

    def test_preflight_is_zero_physical_and_does_not_require_final_result(self):
        payload = S.preflight(
            S.CANONICAL_PREREGISTRATION, S.CANONICAL_SPLIT_TEMPLATE
        )
        self.assertEqual(
            payload["status"],
            "PASS_ZERO_PHYSICAL_R5B_COST_ONLY_LATE_SIX_PREFLIGHT",
        )
        self.assertFalse(payload["R5B_final_artifacts_read"])
        self.assertFalse(payload["selection_frozen"])
        self.assertFalse(payload["metric_values_consulted"])
        self.assertFalse(payload["OpenMM_imported_or_called"])


if __name__ == "__main__":
    unittest.main()
