"""Zero-OpenMM tests for the preregistered metric-blind IR selector."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/select_bb_irivm_late_milestones_blind.py"
SPEC = importlib.util.spec_from_file_location("bb_ir_late_blind_selector_test", SCRIPT)
SELECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SELECTOR)


class MetricPoisonedRow(dict):
    def __getitem__(self, key):
        if key in SELECTOR.FORBIDDEN_SELECTION_FIELDS:
            raise AssertionError(f"selector inspected forbidden metric {key}")
        return super().__getitem__(key)


def row(index, nf, *, committed=True, reportable=True, tail=False):
    return MetricPoisonedRow(
        {
            "history_index": index,
            "call_per_image": index + 1,
            "optimization_N_F": nf,
            "selected_path_sha256": f"{index + 1:064x}",
            "reparameterization_committed": committed,
            "fully_evaluated_current_path": True,
            "formally_reportable": reportable,
            "terminal_uncommitted_charged_tail": tail,
            "exclusion_reason": "tail" if tail else None,
            "discrete_I_perp_eV2_per_A": -1.0e99,
            "densified_Delta_E_kcal_per_mol": -1.0e99,
        }
    )


class BlindSelectorTest(unittest.TestCase):
    def inventory(self):
        return [
            row(0, 60),
            row(1, 150),
            row(2, 300),
            row(3, 1329),
            row(4, 1800),
            row(5, 2050),
            row(6, 3990),
            row(7, 6010),
            row(8, 8020),
            row(9, 8950),
            row(10, 8982),
            row(11, 9985, committed=False, reportable=False, tail=True),
        ]

    def test_selection_is_cost_only_distinct_and_excludes_terminal_tail(self):
        selected = SELECTOR.select_rows(self.inventory())
        self.assertEqual(
            [item["optimization_N_F"] for item in selected],
            [2050, 3990, 6010, 8020, 8950, 8982],
        )
        self.assertEqual(len({item["selected_path_sha256"] for item in selected}), 6)
        self.assertFalse(any(item["optimization_N_F"] == 9985 for item in selected))
        highest = next(
            item
            for item in selected
            if item["selection_role"]
            == "highest_legal_optimization_N_F_not_above_9990"
        )
        self.assertEqual(highest["optimization_N_F"], 8982)

    def test_real_preregistration_and_future_freeze_are_zero_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prereg = directory / "prereg.json"
            SELECTOR.freeze_preregistration(
                prereg,
                SELECTOR.DEFAULT_REPLAY_REGISTRATION,
                SELECTOR.DEFAULT_REPLAY_AUTHORIZATION,
            )
            report = SELECTOR.preflight(prereg)
            self.assertEqual(report["status"], "PASS_ZERO_OPENMM_SELECTION_PREFLIGHT")
            self.assertFalse(report["metric_values_consulted"])
            self.assertFalse(report["terminal_partial_tail_eligible"])
            replay_result = directory / "all_paths.npz"
            replay_result.write_bytes(b"synthetic-result")
            replay_audit = directory / "all_paths.audit.json"
            replay_audit.write_text("{}\n", encoding="utf-8")
            prereg_payload = json.loads(prereg.read_text())
            verification = directory / "verification.json"
            verification.write_text(
                json.dumps(
                    {
                        "schema": SELECTOR.VERIFICATION_SCHEMA,
                        "status": SELECTOR.VERIFICATION_STATUS,
                        "bindings": {
                            "registration": prereg_payload["bindings"]["all_paths_replay_registration"],
                            "authorization": prereg_payload["bindings"]["all_paths_replay_authorization"],
                            "replay_result": SELECTOR.bind(replay_result),
                            "replay_audit": SELECTOR.bind(replay_audit),
                        },
                        "path_inventory": [dict(item) for item in self.inventory()],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            selection = directory / "selection.json"
            SELECTOR.freeze_selection(prereg, verification, selection)
            frozen = json.loads(selection.read_text())
            self.assertEqual(frozen["schema"], SELECTOR.SELECTION_SCHEMA)
            self.assertEqual(frozen["selected_path_count"], 6)
            self.assertFalse(frozen["selection_metric_values_consulted"])
            self.assertFalse(frozen["terminal_partial_tail_selected"])
            self.assertEqual(
                frozen["future_joint_EF_protocol"]["number_of_points_per_selected_path"],
                500,
            )
            self.assertTrue(
                frozen["future_joint_EF_protocol"]["store_all_full_force_vectors"]
            )


if __name__ == "__main__":
    unittest.main()
