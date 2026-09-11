from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import run_stage4_ala_discrete_grid_campaign as v1
import run_stage4_ala_discrete_grid_campaign_successor_v2 as v2


ROOT = Path(__file__).resolve().parents[1]


def stable_primary(cell):
    return v2._stable_primary_cell(cell)


class SuccessorCampaignV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            ROOT
            / "reports"
            / "stage4_ala_discrete_grid_campaign_successor_v2_unittest_"
            "preregistered.json"
        )
        cls.payload = v2.build_preregistration(ROOT, cls.path)

    def test_preregistration_and_source_hashes(self) -> None:
        payload = self.payload
        self.assertEqual(payload["protocol_id"], v2.PROTOCOL_ID)
        self.assertEqual(len(payload["cells"][v2.PRIMARY_FAMILY]), 240)
        self.assertEqual(len(payload["cells"][v2.SECONDARY_FAMILY]), 8)
        manifest = payload["source_manifest"]
        self.assertIn("exact_no_reparam_runtime_v2", manifest)
        self.assertIn("superseded_exact_no_reparam_runtime_v1", manifest)
        self.assertIn("superseded_secondary_member_v1", manifest)
        self.assertIn("predecessor_campaign_v1", manifest)
        for record in manifest.values():
            path = Path(record["path"])
            self.assertEqual(v1.file_sha256(path), record["sha256"])

    def test_primary_science_is_identical_but_worker_routing_is_fresh(self) -> None:
        predecessor = v1._read_json(
            ROOT / v2.PREDECESSOR_PREREGISTRATION
        )["cells"][v2.PRIMARY_FAMILY]
        successor = self.payload["cells"][v2.PRIMARY_FAMILY]
        self.assertEqual(
            [stable_primary(cell) for cell in successor],
            [stable_primary(cell) for cell in predecessor],
        )
        self.assertTrue(
            all(
                left["paths"]["worker_log"] != right["paths"]["worker_log"]
                for left, right in zip(successor, predecessor)
            )
        )

    def test_secondary_commands_and_all_artifact_paths_are_v2_unique(self) -> None:
        predecessor = v1._read_json(
            ROOT / v2.PREDECESSOR_PREREGISTRATION
        )["cells"][v2.SECONDARY_FAMILY]
        successor = self.payload["cells"][v2.SECONDARY_FAMILY]
        self.assertEqual(len({cell["cell_id"] for cell in successor}), 8)
        self.assertTrue(
            all(
                cell["scientific_command"][1].endswith(
                    "exact_noreparam_shim_v2.py"
                )
                for cell in successor
            )
        )
        old_ids = {cell["cell_id"] for cell in predecessor}
        old_paths = {
            value
            for cell in predecessor
            for value in cell["paths"].values()
        }
        self.assertFalse(old_ids & {cell["cell_id"] for cell in successor})
        self.assertFalse(
            old_paths
            & {
                value
                for cell in successor
                for value in cell["paths"].values()
            }
        )

    def test_predecessor_and_failed_v1_provenance_is_pinned(self) -> None:
        provenance = self.payload["predecessor_provenance"]
        self.assertEqual(
            provenance["preregistration"]["sha256"],
            v2.PREDECESSOR_PREREGISTRATION_SHA256,
        )
        self.assertEqual(
            provenance["completed_primary_aggregate"]["sha256"],
            v2.PREDECESSOR_PRIMARY_AGGREGATE_SHA256,
        )
        self.assertEqual(
            {record["cell_id"] for record in provenance["failed_exact_v1"]},
            {"f897abbf6a432fbd", "0ec50c1aeffbb910"},
        )
        self.assertFalse(provenance["failed_v1_reuse_allowed"])

    def test_preregistration_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temporary:
            path = Path(temporary) / (
                "stage4_ala_discrete_grid_campaign_successor_v2_roundtrip_"
                "preregistered.json"
            )
            payload = v2.build_preregistration(ROOT, path)
            v1._write_json_exclusive(path, payload)
            self.assertEqual(
                v2.load_and_validate_preregistration(ROOT, path), payload
            )

    def test_all_primary_cells_can_take_verified_skip_branch(self) -> None:
        cells = self.payload["cells"][v2.PRIMARY_FAMILY]
        launched = []

        def inspector(cell):
            return "verified", {
                "cell_id": cell["cell_id"],
                "assigned_host": cell["assigned_host"],
            }

        rows = v1.execute_cell_sequence(cells, inspector, launched.append)
        self.assertEqual(len(rows), 240)
        self.assertFalse(launched)
        self.assertEqual({row["action"] for row in rows}, {"validated_skip"})

    def test_secondary_barrier_fails_closed_without_successor_primary_aggregate(self) -> None:
        payload = copy.deepcopy(self.payload)
        with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temporary:
            absent = Path(temporary) / "absent_primary_aggregate.json"
            payload["planned_outputs"]["primary_aggregate"] = str(absent)
            with self.assertRaisesRegex(
                v2.CampaignSafetyError,
                "secondary is blocked",
            ):
                v2.require_primary_barrier(ROOT, self.path, payload)

    def test_planned_reports_and_aggregates_do_not_overwrite_predecessor(self) -> None:
        planned = self.payload["planned_outputs"]
        all_outputs = {
            planned["primary_aggregate"],
            planned["secondary_aggregate"],
            *planned["host_reports"][v2.PRIMARY_FAMILY].values(),
            *planned["host_reports"][v2.SECONDARY_FAMILY].values(),
        }
        self.assertNotIn(
            str((ROOT / v2.PREDECESSOR_PRIMARY_AGGREGATE).resolve()),
            all_outputs,
        )
        self.assertTrue(all(v2.PROTOCOL_ID in path for path in all_outputs))


if __name__ == "__main__":
    unittest.main()
