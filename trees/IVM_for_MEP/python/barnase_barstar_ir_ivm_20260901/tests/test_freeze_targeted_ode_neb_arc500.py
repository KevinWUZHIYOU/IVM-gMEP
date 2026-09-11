from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_targeted_ode_neb_arc500 as selector  # noqa: E402


def row(nf: int, path_hash: str, source_batch: int | None = None) -> dict:
    batch = nf // 30
    return {
        "paid_batch_id": batch,
        "optimization_N_F": nf,
        "disposition": "accepted",
        "source_batch_id": batch if source_batch is None else source_batch,
        "coordinate_archive": f"/sealed/batch_{batch:04d}.npz",
        "coordinate_archive_sha256": str(batch) * 64,
        "official_path32_sha256": path_hash,
    }


class FirstPassageTests(unittest.TestCase):
    def test_exact_first_passage_requires_every_earlier_distinct_path(self):
        rows = [row(30, "a"), row(60, "b"), row(90, "c")]
        exact = selector.classify_first_passage(
            rows, {"a": 9.0, "b": 4.0, "c": 2.0}, 3.0
        )
        self.assertEqual(exact["claim"], "exact_first_passage")
        self.assertEqual(exact["possible_exact_first_passage_N_F_interval"], [90, 90])

        observed = selector.classify_first_passage(rows, {"a": 9.0, "c": 2.0}, 3.0)
        self.assertEqual(observed["claim"], "earliest_observed_first_passage")
        self.assertEqual(observed["possible_exact_first_passage_N_F_interval"], [60, 90])

    def test_rejected_duplicate_needs_no_second_measurement(self):
        rows = [row(30, "a"), row(60, "a", source_batch=1), row(90, "b")]
        result = selector.classify_first_passage(rows, {"a": 8.0, "b": 1.0}, 2.0)
        self.assertEqual(result["claim"], "exact_first_passage")
        self.assertEqual(result["observed_first_passage_N_F"], 90)

    def test_no_observed_passage_is_not_reported_as_exact(self):
        rows = [row(30, "a"), row(60, "b")]
        result = selector.classify_first_passage(rows, {"a": 8.0}, 2.0)
        self.assertEqual(
            result["claim"], "no_observed_first_passage_exact_status_undetermined"
        )
        self.assertEqual(result["possible_exact_first_passage_N_F_interval"], [60, 60])


class TargetSelectionTests(unittest.TestCase):
    def test_observed_mode_uses_exact_iperp_not_ase_residual(self):
        rows = [row(30, "a"), row(60, "b"), row(90, "c"), row(120, "d")]
        selected, rationale = selector.choose_paths_to_densify(
            rows,
            delta_by_hash={"a": 20.0, "d": 10.0},
            exact_i_by_hash={"b": 4.0, "c": 1.5},
            i_threshold=2.0,
            delta_threshold=5.0,
            claim_mode="observed",
            max_new_paths=1,
        )
        self.assertEqual([entry["official_path32_sha256"] for entry in selected], ["c"])
        self.assertEqual(rationale["policy"], "targeted_observed_search")

    def test_exact_mode_is_chronological_and_skips_exact_duplicates(self):
        rows = [row(30, "a"), row(60, "a", source_batch=1), row(90, "b"), row(120, "c")]
        selected, rationale = selector.choose_paths_to_densify(
            rows,
            delta_by_hash={},
            exact_i_by_hash={"a": 9.0, "b": 4.0, "c": 2.0},
            i_threshold=3.0,
            delta_threshold=5.0,
            claim_mode="exact",
            max_new_paths=2,
        )
        self.assertEqual(
            [entry["official_path32_sha256"] for entry in selected], ["a", "b"]
        )
        self.assertEqual(rationale["remaining_unmeasured_paths_in_required_prefix_after_this_freeze"], 1)

    def test_observed_mode_refuses_ase_only_ranking(self):
        rows = [row(30, "a"), row(60, "b")]
        selected, rationale = selector.choose_paths_to_densify(
            rows,
            delta_by_hash={"a": 20.0},
            exact_i_by_hash={},
            i_threshold=2.0,
            delta_threshold=5.0,
            claim_mode="observed",
            max_new_paths=1,
        )
        self.assertEqual(selected, [])
        self.assertEqual(
            rationale["status"], "NO_UNMEASURED_PATH_WITH_EXACT_PAID_FORCE_I_PERP"
        )


if __name__ == "__main__":
    unittest.main()
