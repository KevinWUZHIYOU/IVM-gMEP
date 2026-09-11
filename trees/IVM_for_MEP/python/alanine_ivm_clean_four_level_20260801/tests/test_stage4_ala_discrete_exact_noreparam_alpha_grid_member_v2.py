from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest

import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member as v1
import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member_v2 as v2


ROOT = Path(__file__).resolve().parents[1]


class ExactNoReparamMemberV2Tests(unittest.TestCase):
    def test_commands_change_only_runtime_and_routing(self) -> None:
        observed_ids = set()
        for alpha in v2.ALPHAS:
            host = v2.assigned_host(alpha)
            old, _ = v1.single_run_command(ROOT, alpha, host)
            new, label = v2.single_run_command(ROOT, alpha, host)
            normalized = list(new)
            normalized[1] = old[1]
            v2._replace_option(
                normalized,
                "--output-dir",
                f"runs/{v1.MEMBER_POLICY_VERSION}/members/{host}",
            )
            v2._replace_option(normalized, "--run-label", v1.run_label(alpha, host))
            self.assertEqual(normalized, old)
            self.assertTrue(new[1].endswith("exact_noreparam_shim_v2.py"))
            self.assertIn(v2.MEMBER_POLICY_VERSION, label)
            self.assertNotEqual(v2.cell_id(alpha, host), v1.cell_id(alpha, host))
            observed_ids.add(v2.cell_id(alpha, host))
        self.assertEqual(len(observed_ids), len(v2.ALPHAS))

    def test_paths_are_isolated_from_failed_v1_namespaces(self) -> None:
        for alpha in v2.ALPHAS:
            host = v2.assigned_host(alpha)
            old = v1.member_paths(ROOT, alpha, host)
            new = v2.member_paths(ROOT, alpha, host)
            for key in ("result_dir", "audit", "member_log", "worker_log"):
                self.assertNotEqual(Path(old[key]), Path(new[key]))
            self.assertNotEqual(
                v1.expected_result_glob(alpha, host),
                v2.expected_result_glob(alpha, host),
            )

    def test_canonical_identity_pins_successor_runtime(self) -> None:
        spec = v2.canonical_cell_spec(0.1, "local")
        self.assertEqual(spec["member_policy_version"], v2.MEMBER_POLICY_VERSION)
        self.assertEqual(spec["runtime_policy_id"], v2.runtime.POLICY_ID)
        self.assertEqual(
            spec["runtime_implementation_policy_id"],
            v2.runtime.IMPLEMENTATION_POLICY_ID,
        )
        self.assertEqual(
            spec["successor_of_member_policy_version"],
            v1.MEMBER_POLICY_VERSION,
        )

    def test_validation_delegates_under_v2_runtime_and_restores(self) -> None:
        original = v1.runtime

        def fake_validate(result, value, execution_host=None):
            self.assertIs(v1.runtime, v2.runtime)
            return {"ok": True, "value": value, "host": execution_host}

        with mock.patch.object(v1, "validate_result", side_effect=fake_validate), mock.patch.object(
            v2,
            "_validate_v2_implementation_identity",
            return_value="base+ala-exact-completion-audit-call-signature-v2",
        ):
            observed = v2.validate_result(Path("unused.npz"), 0.1, "local")
        self.assertTrue(observed["ok"])
        self.assertTrue(observed["runtime_v2_implementation_policy_verified"])
        self.assertIn(
            v2.runtime.IMPLEMENTATION_POLICY_ID,
            observed["discrete_implementation_id"],
        )
        self.assertIs(v1.runtime, original)

    def test_v2_implementation_token_is_fail_closed(self) -> None:
        fake_saved = mock.MagicMock()
        fake_saved.files = ["discrete_implementation_id"]
        fake_saved.__getitem__.return_value = "base+candidate"
        context = mock.MagicMock()
        context.__enter__.return_value = fake_saved
        with mock.patch.object(v1.np, "load", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "omits the v2"):
                v2._validate_v2_implementation_identity(Path("unused.npz"))


if __name__ == "__main__":
    unittest.main()
