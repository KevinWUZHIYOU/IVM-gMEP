from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_bb_discrete32_action_cost_pack as B
import freeze_bb_discrete32_action_cost_pack as F
import plot_bb_discrete32_action_cost as P


class T(unittest.TestCase):
    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def make_r5b_pair(
        self,
        directory: Path,
        *,
        primary_schema: str = B.PRIMARY_SCHEMA,
        primary_status: str = B.PRIMARY_STATUS,
        supplemental_schema: str = B.SUPPLEMENTAL_SCHEMA,
        supplemental_status: str = B.SUPPLEMENTAL_STATUS,
        highest_reportable_nf: int = 14950,
        canonical_count: int = 321,
        last_complete: int = 320,
    ) -> tuple[Path, Path]:
        manifest = directory / "manifest.json"
        index = directory / "index.json"
        audit = directory / "audit.json"
        registration = directory / "registration.json"
        authorization = directory / "authorization.json"
        reference = directory / "reference.npz"
        replay = directory / "ivm_bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_synthetic.npz"
        replay_audit = replay.with_suffix(".audit.json")
        for path in (manifest, index, audit, registration, authorization, reference, replay):
            path.write_text("{}\n", encoding="utf-8")
        self._write(
            replay_audit,
            {"consistent": True, "result_file": str(replay.resolve())},
        )

        prefix = [
            max(1, round(B.OLD_COMPLETE_PREFIX_NF * call / B.OLD_COMPLETE_PREFIX_CALLS))
            for call in range(1, B.OLD_COMPLETE_PREFIX_CALLS + 1)
        ]
        prefix[-1] = B.OLD_COMPLETE_PREFIX_NF
        post_complete = last_complete - B.OLD_COMPLETE_PREFIX_CALLS
        self.assertGreater(post_complete, 0)
        post = [
            B.OLD_COMPLETE_PREFIX_NF
            + round((highest_reportable_nf - B.OLD_COMPLETE_PREFIX_NF) * step / post_complete)
            for step in range(1, post_complete + 1)
        ]
        tail = [
            min(B.R5B_BUDGET, highest_reportable_nf + 30 * step)
            for step in range(1, canonical_count - last_complete + 1)
        ]
        nfs = prefix + post + tail
        inventory = [
            {
                "history_index": call - 1,
                "call_per_image": call,
                "optimization_N_F": nfs[call - 1],
                "path_sha256": f"path-{call}",
                "energy_sha256": f"energy-{call}",
                "force_sha256": f"force-{call}",
                "shard": str(directory / f"call-{call}.npz"),
                "shard_sha256": f"shard-{call}",
                "formally_reportable": call <= last_complete,
                "terminal_uncommitted_charged_tail": call > last_complete,
            }
            for call in range(1, canonical_count + 1)
        ]
        primary = directory / "primary.json"
        self._write(
            primary,
            {
                "schema": primary_schema,
                "status": primary_status,
                "exact_old_9990_complete_prefix_call_range": [
                    1,
                    B.OLD_COMPLETE_PREFIX_CALLS,
                ],
                "exact_old_9990_complete_prefix_last_N_F": B.OLD_COMPLETE_PREFIX_NF,
                "old_9990_terminal_tail_required_equal": False,
                "actual_canonical_call_count": canonical_count,
                "last_complete_call": last_complete,
                "terminal_optimization_N_F": max(B.OLD_BUDGET + 1, nfs[-1]),
                "path_inventory": inventory,
                "bindings": {
                    "registration": B.binding(registration),
                    "authorization": B.binding(authorization),
                    "reference_result": B.binding(reference),
                    "replay_result": B.binding(replay),
                    "replay_audit": B.binding(replay_audit),
                    "capture_manifest": B.binding(manifest),
                    "canonical_index": B.binding(index),
                    "capture_audit": B.binding(audit),
                },
            },
        )
        supplemental = directory / "supplemental.json"
        self._write(
            supplemental,
            {
                "schema": supplemental_schema,
                "status": supplemental_status,
                "canonical_record_count": canonical_count,
                "inventory": [
                    {
                        "call_per_image": row["call_per_image"],
                        "optimization_N_F": row["optimization_N_F"],
                    }
                    for row in inventory
                ],
                "bindings": {
                    "primary_verification": B.binding(primary),
                    "registration": B.binding(registration),
                    "authorization": B.binding(authorization),
                    "manifest": B.binding(manifest),
                    "canonical_index": B.binding(index),
                    "capture_audit": B.binding(audit),
                },
            },
        )
        return primary, supplemental

    def test_lower_envelope(self):
        self.assertTrue(np.array_equal(B.lower_envelope([5, 7, 3, 4]), [5, 5, 3, 3]))

    def test_dynamic_nf15000_r5b_pair_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, supplemental = self.make_r5b_pair(Path(temporary))
            _, _, inventory, last, *_ = B.validate_r5b_verifiers(
                primary, supplemental, require_canonical=False
            )
            self.assertEqual(last, 320)
            self.assertEqual(len(inventory), 321)
            self.assertEqual(inventory[last - 1]["optimization_N_F"], 14950)

    def test_old_r2_primary_is_rejected_even_if_status_starts_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, supplemental = self.make_r5b_pair(
                Path(temporary),
                primary_schema="bb-irivm-all-paths-verification-v2",
                primary_status="PASS_R2_ALL_PATHS_VERIFIED",
            )
            with self.assertRaisesRegex(RuntimeError, "exact NF15000 R5B"):
                B.validate_r5b_verifiers(primary, supplemental, require_canonical=False)

    def test_fake_pass_supplemental_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, supplemental = self.make_r5b_pair(
                Path(temporary), supplemental_status="PASS_FAKE"
            )
            with self.assertRaisesRegex(RuntimeError, "supplemental verifier"):
                B.validate_r5b_verifiers(primary, supplemental, require_canonical=False)

    def test_history_truncated_at_9990_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, supplemental = self.make_r5b_pair(
                Path(temporary), highest_reportable_nf=B.OLD_BUDGET
            )
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                B.validate_r5b_verifiers(primary, supplemental, require_canonical=False)

    def test_noncanonical_verifier_paths_are_rejected_by_production_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, supplemental = self.make_r5b_pair(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "noncanonical"):
                B.validate_r5b_verifiers(primary, supplemental, require_canonical=True)

    def test_primary_replay_result_binding_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            primary, supplemental = self.make_r5b_pair(directory)
            replay = directory / "ivm_bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_synthetic.npz"
            replay.write_bytes(b"tampered-after-primary-freeze")
            with self.assertRaisesRegex(RuntimeError, "binding drift replay_result"):
                B.validate_r5b_verifiers(primary, supplemental, require_canonical=False)

    def test_sealed_ode_action_pack_and_audit_pass_semantic_validation(self):
        nf, action, audit = B.validate_ode_pack_and_audit()
        self.assertTrue(np.array_equal(nf, B.O.EXPECTED_NF))
        self.assertTrue(np.allclose(action, B.O.EXPECTED_S, rtol=0, atol=1e-9))
        self.assertEqual(audit["optimization_N_F_contribution"], 0)

    def test_fake_ode_pack_is_rejected_even_with_hash_consistent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pack = directory / "fake_ode.npz"
            np.savez(
                pack,
                schema=np.asarray("bb-ode8-discrete32-action-pack-v0-fake"),
                status=np.asarray("PASS"),
                optimization_N_F=B.O.EXPECTED_NF,
                total_action_eV=B.O.EXPECTED_S,
                total_action_kcal_per_mol=B.O.EXPECTED_S * B.O.KCAL,
                path_sha256=np.asarray([f"path-{i}" for i in range(8)]),
                force32_sha256=np.asarray([f"force-{i}" for i in range(8)]),
                metric_images=np.asarray(32),
                metric_segments=np.asarray(31),
                metric_interpolation=np.asarray("none"),
                optimization_N_F_contribution=np.asarray(0),
            )
            audit = directory / "fake_ode.audit.json"
            self._write(
                audit,
                {
                    "schema": "bb-ode8-discrete32-action-pack-audit-v1",
                    "status": "PASS",
                    "result": str(pack),
                    "result_sha256": B.O.sha_file(pack),
                    "contract": str(B.O.CONTRACT.resolve()),
                    "contract_sha256": B.O.sha_file(B.O.CONTRACT),
                    "arc500_used_by_metric": False,
                    "original_images": 32,
                    "segments": 31,
                    "physical_calls": 0,
                    "optimization_N_F_contribution": 0,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "ODE action pack is not PASS"):
                B.validate_ode_pack_and_audit(pack, audit, require_canonical=False)

    def test_final_pack_is_zero_query_and_original32(self):
        text = Path(B.__file__).read_text()
        self.assertNotIn("OpenMMBackend", text)
        self.assertNotIn("dense_force_eV_per_A", text)
        self.assertIn("force_eV_per_A'", text)

    def test_freezer_requires_exact_canonical_nf15000_r5b_pair(self):
        text = (ROOT / "scripts/freeze_bb_discrete32_action_cost_pack.py").read_text()
        self.assertIn("--primary-verifier", text)
        self.assertIn("--supplemental-verifier", text)
        self.assertIn("validate_r5b_verifiers(pri,sup,require_canonical=True)", text)
        self.assertNotIn("startswith('PASS_')", text)

    def test_plot_loader_accepts_dynamic_nf15000_highest_and_binds_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            primary, supplemental = self.make_r5b_pair(directory)
            contract = directory / "contract.json"
            output = directory / "result.npz"
            audit = directory / "result.audit.json"
            inf = np.asarray(
                json.loads(primary.read_text())["path_inventory"][:320], dtype=object
            )
            inf = np.asarray([row["optimization_N_F"] for row in inf], dtype=np.int64)
            ivm_action = np.linspace(10.0, 1.0, 320)
            ode_nf = np.asarray(B.O.EXPECTED_NF, dtype=np.int64)
            ode_action = np.asarray(B.O.EXPECTED_S, dtype=np.float64)
            np.savez(
                output,
                schema=np.asarray("bb-discrete32-action-cost-pack-v1"),
                status=np.asarray("PASS"),
                ivm_optimization_N_F=inf,
                ivm_total_action_eV=ivm_action,
                ivm_observed_lower_envelope_eV=B.lower_envelope(ivm_action),
                ode_optimization_N_F=ode_nf,
                ode_total_action_eV=ode_action,
                ode_observed_lower_envelope_eV=B.lower_envelope(ode_action),
                R5B_budget=np.asarray(B.R5B_BUDGET),
                R5B_dynamic_last_complete_call=np.asarray(320),
                R5B_dynamic_highest_reportable_optimization_N_F=np.asarray(14950),
                metric_images=np.asarray(32),
                metric_segments=np.asarray(31),
                metric_interpolation=np.asarray("none"),
                optimization_N_F_contribution=np.asarray(0),
            )
            paths = {
                "builder": Path(B.__file__),
                "freezer": ROOT / "scripts/freeze_bb_discrete32_action_cost_pack.py",
                "plotter": Path(P.__file__),
                "test": Path(__file__),
                "ODE_pack": B.ODE,
                "ODE_audit": B.ODE_AUDIT,
                "R5B_primary_verifier": primary,
                "R5B_supplemental_verifier": supplemental,
            }
            payload = {
                "schema": "bb-discrete32-action-cost-pack-contract-v1",
                "status": "FROZEN_AFTER_R5B_PRIMARY_AND_SUPPLEMENTAL_PASS",
                "x_axis": {
                    "label": r"Path-dependent geometric action, $\hat S$ (eV)",
                    "quantity": "original32_path_dependent_geometric_action",
                    "scale": "log",
                    "direction": "larger_to_smaller_left_to_right",
                    "inverted": True,
                },
                "y_axis": {
                    "label": "Number of Forces",
                    "quantity": "optimization_N_F",
                    "scale": "linear",
                    "formatter": "thousands_k",
                },
                "curve_semantics": "observed lower-envelope cost-to-precision frontier; no metric interpolation",
                "R5B_budget": B.R5B_BUDGET,
                "R5B_dynamic_canonical_call_count": 321,
                "R5B_dynamic_last_complete_call": 320,
                "R5B_dynamic_highest_reportable_optimization_N_F": 14950,
                "output": str(output),
                "audit": str(audit),
                "figure_output": str(directory / "figure.pdf"),
                "figure_provenance": str(directory / "figure_provenance.json"),
                **{role: B.binding(path) for role, path in paths.items()},
            }
            self._write(contract, payload)
            self._write(
                audit,
                {
                    "schema": "bb-discrete32-action-cost-pack-audit-v1",
                    "status": "PASS",
                    "result": B.binding(output),
                    "contract": B.binding(contract),
                    "R5B_primary_verifier": payload["R5B_primary_verifier"],
                    "R5B_supplemental_verifier": payload["R5B_supplemental_verifier"],
                    "R5B_dynamic_canonical_call_count": 321,
                    "R5B_last_complete_call": 320,
                    "R5B_dynamic_highest_reportable_optimization_N_F": 14950,
                    "R5B_terminal_tail_excluded": True,
                },
            )
            with mock.patch.object(B, "CONTRACT", contract), mock.patch.object(
                B, "OUTPUT", output
            ), mock.patch.object(B, "AUDIT", audit), mock.patch.object(
                B, "PRIMARY", primary
            ), mock.patch.object(B, "SUPPLEMENTAL", supplemental), mock.patch.object(
                B, "R5B_REGISTRATION", directory / "registration.json"
            ), mock.patch.object(B, "R5B_AUTHORIZATION", directory / "authorization.json"), mock.patch.object(
                B, "R5B_MANIFEST", directory / "manifest.json"
            ), mock.patch.object(B, "R5B_INDEX", directory / "index.json"), mock.patch.object(
                B, "R5B_CAPTURE_AUDIT", directory / "audit.json"
            ), mock.patch.object(
                B, "R5B_REFERENCE_RESULT", directory / "reference.npz"
            ), mock.patch.object(
                B, "R5B_OUTPUT_DIR", directory
            ), mock.patch.object(
                P, "FIG", directory / "figure.pdf"
            ), mock.patch.object(P, "PROV", directory / "figure_provenance.json"):
                data, _, _, _ = P.load_verified_pack()
                payload["x_axis"]["inverted"] = False
                self._write(contract, payload)
                changed_audit = json.loads(audit.read_text())
                changed_audit["contract"] = B.binding(contract)
                self._write(audit, changed_audit)
                with self.assertRaisesRegex(RuntimeError, "x-axis semantics drift"):
                    P.load_verified_pack()
            self.assertEqual(len(data["IVM"][0]), 320)
            self.assertEqual(int(data["IVM"][0][-1]), 14950)

    def test_plotter_article_labels_and_two_item_legend(self):
        text = (ROOT / "scripts/plot_bb_discrete32_action_cost.py").read_text()
        self.assertIn(r"set_xlabel(r'Path-dependent geometric action, $\hat S$ (eV)')", text)
        self.assertIn("set_ylabel('Number of Forces')", text)
        self.assertIn("set_xscale('log')", text)
        self.assertIn("invert_xaxis()", text)
        self.assertIn("FuncFormatter", text)
        self.assertIn("label='_nolegend_'", text)
        self.assertIn("label=name)", text)
        self.assertNotIn("observations'", text)
        self.assertNotIn("observed lower envelope'", text)

    def test_plot_loader_rejects_axis_direction_regression(self):
        text = (ROOT / "scripts/plot_bb_discrete32_action_cost.py").read_text()
        self.assertIn("action-cost x-axis semantics drift", text)
        self.assertIn("action-cost y-axis semantics drift", text)
        self.assertIn("action-cost curve semantics drift", text)
        freezer = (ROOT / "scripts/freeze_bb_discrete32_action_cost_pack.py").read_text()
        self.assertIn("larger_to_smaller_left_to_right", freezer)
        self.assertNotIn("'y_scale':'log'", freezer)

    def test_freezer_and_plotter_bind_the_same_final_figure_path(self):
        self.assertEqual(F.FIG, P.FIG)
        self.assertEqual(
            P.FIG.name,
            "barnase_barstar_ivm_vs_ode_neb_discrete32_geometric_action_cost_to_precision.pdf",
        )

    def test_plot_publication_is_atomic_pair_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);fig=directory/'figure.pdf';prov=directory/'figure_provenance.json'
            def render(path): Path(path).write_bytes(b'%PDF-fixture')
            payload={'schema':'fixture','status':'PASS'}
            with mock.patch.object(P,'FIG',fig),mock.patch.object(P,'PROV',prov):
                P.publish_pair(render,payload)
                self.assertTrue(fig.exists() and prov.exists())
                self.assertEqual(json.loads(prov.read_text())['figure']['path'],str(fig))
                fig.unlink();prov.unlink()
                for occupied in (fig,prov):
                    occupied.write_bytes(b'occupied')
                    with self.assertRaisesRegex(RuntimeError,'overwrite'):P.publish_pair(render,payload)
                    self.assertFalse((prov if occupied==fig else fig).exists())
                    occupied.unlink()
                real_link=os.link;count=[0]
                def fail_second(src,dst):
                    count[0]+=1
                    if count[0]==2:raise OSError('second link failure')
                    return real_link(src,dst)
                with mock.patch.object(P.os,'link',side_effect=fail_second):
                    with self.assertRaisesRegex(OSError,'second link failure'):P.publish_pair(render,payload)
                self.assertFalse(fig.exists() or prov.exists())


if __name__ == "__main__":
    unittest.main()
