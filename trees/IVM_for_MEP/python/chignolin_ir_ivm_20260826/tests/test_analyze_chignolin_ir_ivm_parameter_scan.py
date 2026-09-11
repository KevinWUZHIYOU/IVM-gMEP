from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_chignolin_ir_ivm_parameter_scan as analysis


def _passed_row(
    label: str,
    alpha: float,
    ratio: float,
    cap: int,
    reached: bool,
) -> Dict[str, Any]:
    if reached:
        terminal_internal = 1500
        terminal_total = 1502
        last_index = 79
        last_internal = 1382
        last_total = 1384
        last_iperp = 0.10842149768934348
        first_index = 50
        first_kind = "reparam_committed"
        first_internal = 905
        first_total = 907
        first_iperp = 0.18194143923859238
        tail = 118
    else:
        terminal_internal = 1484
        terminal_total = 1486
        last_index = 76
        last_internal = 1464
        last_total = 1466
        last_iperp = 0.27392876673955896
        first_index = None
        first_kind = None
        first_internal = None
        first_total = None
        first_iperp = None
        tail = 20
    marker = "a" if reached else "b"
    return {
        "audit_status": "passed",
        "run_label": label,
        "alpha": alpha,
        "residual_ratio_r": ratio,
        "inner_max_steps_m_max": cap,
        "eta": 1.1,
        "selector_mode": "one_way_takeover",
        "reparam_mode": "state_curve",
        "internal_nf_budget": 1500,
        "terminal_internal_nf": terminal_internal,
        "terminal_total_nf": terminal_total,
        "i_perp_threshold": analysis.IPERP_THRESHOLD,
        "first_threshold_reached": reached,
        "first_threshold_state_index": first_index,
        "first_threshold_state_kind": first_kind,
        "first_threshold_internal_nf": first_internal,
        "first_threshold_total_nf": first_total,
        "first_threshold_i_perp": first_iperp,
        "usable_state_rule": "record_index_0_or_reparam_committed",
        "usable_state_count": 9 if reached else 20,
        "last_complete_state_index": last_index,
        "last_complete_state_kind": "reparam_committed",
        "last_complete_internal_nf": last_internal,
        "last_complete_total_nf": last_total,
        "last_complete_i_perp": last_iperp,
        "terminal_partial_tail_present": True,
        "terminal_partial_tail_quarantined": True,
        "terminal_partial_tail_internal_nf": tail,
        "reparam_attempt_count": 9 if reached else 19,
        "reparam_accept_count": 8 if reached else 19,
        "recovery_event_count": 8 if reached else 19,
        "recovery_joint_evaluations": 502 if reached else 384,
        "result_npz": "/runs/{}.npz".format(label),
        "result_npz_sha256": marker * 64,
        "configuration_sha256": marker.upper() * 64,
    }


def _failed_row() -> Dict[str, Any]:
    return {
        "audit_status": "failed",
        "audit_failure_reason": "companion counter mismatch",
        "run_label": "chig_irivm_failed",
        "alpha": 100.0,
        "residual_ratio_r": 0.3,
        "inner_max_steps_m_max": 8,
        "internal_nf_budget": 1500,
        "i_perp_threshold": analysis.IPERP_THRESHOLD,
    }


class AnalyzeChignolinIRIVMParameterScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [
            _passed_row("chig_irivm_reached", 10.0, 0.5, 5, True),
            _passed_row("chig_irivm_unreached", 1.0, 0.5, 5, False),
            _failed_row(),
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_json(
        self, name: str = "audit.json", rows: Sequence[Dict[str, Any]] = ()
    ) -> Path:
        selected = list(rows) if rows else list(self.rows)
        path = self.root / name
        path.write_text(
            json.dumps(
                {
                    "schema_version": "chignolin-ir-ivm-result-audit-v1",
                    "i_perp_threshold": analysis.IPERP_THRESHOLD,
                    "result_count": len(selected),
                    "results": selected,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _write_csv(
        self, name: str = "audit.csv", rows: Sequence[Dict[str, Any]] = ()
    ) -> Path:
        selected = list(rows) if rows else list(self.rows)
        fields: List[str] = []
        for row in selected:
            for key in row:
                if key not in fields:
                    fields.append(key)
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        return path

    def test_statuses_and_complete_reparameterization_summary(self) -> None:
        source = self._write_json()
        settings, sources = analysis.load_parameter_settings([source])
        by_status = {row["analysis_status"]: row for row in settings}

        reached = by_status[analysis.STATUS_REACHED]
        self.assertEqual(reached["first_threshold_total_nf"], 907)
        self.assertEqual(reached["last_complete_reparam_total_nf"], 1384)
        self.assertAlmostEqual(reached["last_complete_reparam_i_perp"], 0.1084215)

        unreached = by_status[analysis.STATUS_NOT_REACHED]
        self.assertIsNone(unreached["first_threshold_total_nf"])
        self.assertEqual(unreached["right_censoring_total_nf"], 1466)
        self.assertEqual(unreached["last_complete_reparam_total_nf"], 1466)

        failed = by_status[analysis.STATUS_AUDIT_FAILED]
        self.assertIsNone(failed["last_complete_reparam_total_nf"])
        self.assertEqual(failed["audit_failure_reason"], "companion counter mismatch")
        self.assertEqual(len(sources), 1)

    def test_parameter_effect_table_marks_first_passage_and_censoring(self) -> None:
        settings, _ = analysis.load_parameter_settings([self._write_json()])
        effects = analysis.parameter_effect_records(settings)

        self.assertEqual(len(effects), 9)
        reached = [
            row
            for row in effects
            if row["analysis_status"] == analysis.STATUS_REACHED
            and row["parameter_symbol"] == "alpha"
        ][0]
        self.assertEqual(reached["observation_type"], "first_passage")
        self.assertEqual(reached["number_of_forces_for_plot"], 907)
        self.assertEqual(reached["parameter_value_log10"], 1.0)

        unreached = [
            row
            for row in effects
            if row["analysis_status"] == analysis.STATUS_NOT_REACHED
            and row["parameter_symbol"] == "r"
        ][0]
        self.assertEqual(unreached["observation_type"], "right_censored")
        self.assertIsNone(unreached["first_threshold_total_nf"])
        self.assertEqual(unreached["number_of_forces_for_plot"], 1466)

    def test_duplicate_csv_and_json_sources_are_deduplicated(self) -> None:
        json_path = self._write_json()
        csv_path = self._write_csv()
        settings, sources = analysis.load_parameter_settings([json_path, csv_path])

        self.assertEqual(len(settings), 3)
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(len(row["source_audit_files"]) == 2 for row in settings))

    def test_inconsistent_nof_and_false_unreached_claim_are_rejected(self) -> None:
        bad_total = dict(self.rows[0])
        bad_total["terminal_total_nf"] = 1503
        with self.assertRaisesRegex(analysis.AnalysisError, "internal NF plus two"):
            analysis.load_parameter_settings(
                [self._write_json("bad_total.json", [bad_total])]
            )

        bad_unreached = dict(self.rows[1])
        bad_unreached["last_complete_i_perp"] = 0.20
        with self.assertRaisesRegex(analysis.AnalysisError, "already satisfies"):
            analysis.load_parameter_settings(
                [self._write_json("bad_unreached.json", [bad_unreached])]
            )

    def test_cli_writes_tidy_csv_json_and_article_language_summary(self) -> None:
        source = self._write_json()
        output_dir = self.root / "analysis"
        analysis.main([str(source), "--output-dir", str(output_dir)])

        settings_path = output_dir / analysis.SETTINGS_FILENAME
        effects_path = output_dir / analysis.EFFECTS_FILENAME
        json_path = output_dir / analysis.JSON_FILENAME
        self.assertTrue(settings_path.is_file())
        self.assertTrue(effects_path.is_file())
        self.assertTrue(json_path.is_file())

        with settings_path.open(newline="", encoding="utf-8") as handle:
            settings_rows = list(csv.DictReader(handle))
        with effects_path.open(newline="", encoding="utf-8") as handle:
            effect_rows = list(csv.DictReader(handle))
        document = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(len(settings_rows), 3)
        self.assertEqual(len(effect_rows), 9)
        self.assertEqual(document["status_counts"]["reached"], 1)
        self.assertEqual(
            document["best_reached_setting"]["first_threshold_total_nf"], 907
        )
        self.assertFalse(document["dense_barrier_computed"])
        self.assertFalse(document["pdf_generated"])
        joined_summary = " ".join(document["article_summary_zh"])
        self.assertIn("完整重参数化路径状态", joined_summary)
        self.assertIn("Number of Forces", joined_summary)


if __name__ == "__main__":
    unittest.main()
