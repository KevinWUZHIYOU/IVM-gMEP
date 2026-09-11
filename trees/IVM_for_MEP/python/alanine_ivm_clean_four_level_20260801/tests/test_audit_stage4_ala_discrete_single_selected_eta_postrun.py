from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_stage4_ala_discrete_single_selected_eta_postrun as audit


class SelectedEtaPostrunAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.formal_report = cls.root / audit.FORMAL_REPORT_RELATIVE

    def test_completed_formal_campaign_passes_strong_readonly_audit(self):
        payload = audit.build_audit(self.root, self.formal_report)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["barrier_values_accessed"])
        parity = payload["eta_one_all_common_safe_field_parity"]
        self.assertEqual(parity["safe_common_field_count"], 253)
        self.assertEqual(parity["safe_common_difference_count"], 0)
        self.assertEqual(
            set(parity["barrier_fields_seen_by_name_not_indexed"]),
            set(audit.PARITY_BARRIER_EXCLUSIONS),
        )
        mechanisms = payload["mechanism_recomputation"]
        self.assertEqual([row["eta"] for row in mechanisms], list(audit.ETA_VALUES))
        failure_keys = [
            key
            for key in mechanisms[0]
            if key.endswith("_mismatch_count")
            or key in {
                "base_exceeds_dynamic_cap_count",
                "applied_below_base_count",
            }
        ]
        for row in mechanisms:
            for key in failure_keys:
                self.assertEqual(row[key], 0, (row["eta"], key))

    @staticmethod
    def _mechanism_payload(eta: float = 1.5):
        base = np.asarray([0.2, 0.8])
        dynamic = np.asarray([1.0, 1.0])
        requested = eta * base
        applied = np.minimum(requested, dynamic)
        selection = np.asarray(["H0", "LBFGS"], dtype="U8")
        return {
            "configuration_sha256": np.asarray("configuration"),
            "discrete_audit_selection": selection,
            "discrete_audit_h0_capped_predicted_decrease": np.asarray(
                [2.0, 1.0]
            ),
            "discrete_audit_lbfgs_capped_predicted_decrease": np.asarray(
                [1.0, 2.0]
            ),
            "discrete_step_multiplier_eta": np.asarray(eta),
            "discrete_step_multiplier_selector_isolated": np.asarray(True),
            "discrete_step_multiplier_dynamic_cap_reapplied": np.asarray(True),
            "discrete_step_multiplier_energy_backtracking": np.asarray(False),
            "discrete_step_multiplier_extra_physical_evaluations": np.asarray(0),
            "discrete_step_multiplier_audit_selection": selection.copy(),
            "discrete_step_multiplier_audit_requested": np.full(2, eta),
            "discrete_step_multiplier_audit_applied": applied / base,
            "discrete_step_multiplier_audit_cap_hit": requested > dynamic,
            "discrete_step_multiplier_audit_dynamic_max_step": dynamic,
            "discrete_step_multiplier_audit_base_max_atomic_displacement": base,
            "discrete_step_multiplier_audit_requested_max_atomic_displacement": (
                requested
            ),
            "discrete_step_multiplier_audit_applied_max_atomic_displacement": (
                applied
            ),
            "discrete_step_multiplier_uncommitted_selection_count": np.asarray(0),
            "discrete_step_multiplier_aborted_selection_count": np.asarray(0),
            # If the auditor indexes this value, allow_pickle=False raises.
            "barrier_kcal": np.asarray(["embargoed"], dtype=object),
        }

    def test_mechanism_recomputation_rejects_selector_and_recap_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            valid = temporary / "valid.npz"
            payload = self._mechanism_payload()
            np.savez(valid, **payload)
            record = audit.audit_mechanism_arm(valid, 1.5)
            self.assertEqual(record["selector_winner_mismatch_count"], 0)
            self.assertEqual(record["applied_recap_target_mismatch_count"], 0)
            self.assertFalse(record["barrier_values_accessed"])

            selector = temporary / "selector.npz"
            selector_payload = self._mechanism_payload()
            selector_payload["discrete_audit_selection"] = np.asarray(
                ["LBFGS", "LBFGS"], dtype="U8"
            )
            selector_payload["discrete_step_multiplier_audit_selection"] = (
                selector_payload["discrete_audit_selection"].copy()
            )
            np.savez(selector, **selector_payload)
            with self.assertRaisesRegex(RuntimeError, "Mechanism invariant"):
                audit.audit_mechanism_arm(selector, 1.5)

            recap = temporary / "recap.npz"
            recap_payload = self._mechanism_payload()
            recap_payload[
                "discrete_step_multiplier_audit_applied_max_atomic_displacement"
            ] = np.asarray([0.3, 0.9])
            recap_payload["discrete_step_multiplier_audit_applied"] = np.asarray(
                [1.5, 0.9 / 0.8]
            )
            np.savez(recap, **recap_payload)
            with self.assertRaisesRegex(RuntimeError, "Mechanism invariant"):
                audit.audit_mechanism_arm(recap, 1.5)

    def test_safe_parity_never_indexes_object_barrier_fields(self):
        common = {
            "safe": np.asarray([1.0, np.nan]),
            "run_label": np.asarray("old"),
            "configuration_sha256": np.asarray("old-configuration"),
            "discrete_implementation_id": np.asarray("old-implementation"),
            "wall_seconds": np.asarray([1.0]),
            "optimization_barrier_is_sparse": np.asarray(
                ["embargoed"], dtype=object
            ),
            "barrier_metric_for_figures": np.asarray(
                ["embargoed"], dtype=object
            ),
            "barrier_kcal": np.asarray(["embargoed"], dtype=object),
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            old = temporary / "old.npz"
            new = temporary / "new.npz"
            np.savez(old, **common)
            new_payload = dict(common)
            new_payload.update(
                {
                    key: np.asarray(index)
                    for index, key in enumerate(sorted(audit.EXPECTED_ETA_ONLY_FIELDS))
                }
            )
            new_payload["run_label"] = np.asarray("new")
            new_payload["configuration_sha256"] = np.asarray("new-configuration")
            new_payload["discrete_implementation_id"] = np.asarray(
                "new-implementation"
            )
            new_payload["wall_seconds"] = np.asarray([2.0])
            np.savez(new, **new_payload)
            result = audit.compare_eta_one_safe_fields(old, new)
        self.assertEqual(result["safe_common_fields_compared"], ["safe"])
        self.assertEqual(result["safe_common_difference_count"], 0)
        self.assertFalse(result["barrier_values_accessed"])

    def test_sidecar_write_is_exclusive_and_json_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "audit.json"
            audit.write_exclusive(target, {"status": "passed", "value": 1.0})
            self.assertEqual(json.loads(target.read_text())["status"], "passed")
            with self.assertRaises(FileExistsError):
                audit.write_exclusive(target, {"status": "overwritten"})
            invalid = Path(temporary) / "invalid.json"
            with self.assertRaises(ValueError):
                audit.write_exclusive(invalid, {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
