#!/usr/bin/env python3
"""Freeze every input of the non-overwriting densified BB cost figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PLOTTER = ROOT / "scripts" / "plot_bb_cost_to_precision_densified.py"
DIRECT_PLOT_DEPENDENCIES = (
    ROOT / "scripts" / "plot_bb_cost_to_precision_final.py",
    ROOT / "scripts" / "plot_bb_cost_to_precision_preview.py",
    ROOT / "scripts" / "evaluate_discrete_iperp_densified_barrier.py",
    ROOT / "src" / "bb_core.py",
)
EXPECTED_STEM = "barnase_barstar_ivm_vs_ode_neb_cost_to_precision_densified"
TIMING_CORRECTION_SHA256 = "39998097ea221de11da491b5880e95e6293267e2a0a3dd33386516c19f303ac1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path):
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="contract JSON")
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--ivm-result", type=Path, required=True)
    parser.add_argument("--ivm-audit", type=Path, required=True)
    parser.add_argument("--ivm-terminal-result", type=Path, required=True)
    parser.add_argument("--ivm-terminal-audit", type=Path, required=True)
    parser.add_argument("--ivm-long-verification", type=Path, required=True)
    parser.add_argument("--ivm-prefix-verification", type=Path, required=True)
    parser.add_argument("--ivm-dense-result", type=Path, required=True)
    parser.add_argument("--ivm-dense-audit", type=Path, required=True)
    parser.add_argument("--ivm-dense-contract", type=Path, required=True)
    parser.add_argument("--ode-result", type=Path, action="append", required=True)
    parser.add_argument("--ode-audit", type=Path, action="append", required=True)
    parser.add_argument("--ode-target-selection", type=Path)
    parser.add_argument("--ode-targeted-bundle", nargs=3, type=Path, action="append", default=[])
    parser.add_argument("--ode-dense-result", type=Path, required=True)
    parser.add_argument("--ode-dense-audit", type=Path, required=True)
    parser.add_argument("--ode-dense-contract", type=Path, required=True)
    parser.add_argument("--timing-correction", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    figure = args.figure_output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if figure.suffix.lower() != ".pdf" or figure.stem != EXPECTED_STEM:
        raise ValueError("figure output must use the registered new densified stem")
    if figure.exists() or figure.with_suffix(".json").exists():
        raise FileExistsError("refusing to bind an existing figure/provenance output")
    if len(args.ode_result) != len(args.ode_audit):
        raise ValueError("each existing ODE result requires one audit")
    timing_path = args.timing_correction.expanduser().resolve(strict=True)
    if sha256(timing_path) != TIMING_CORRECTION_SHA256:
        raise ValueError("timing correction R2 SHA mismatch")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    if timing.get("status") != "R1_SUPERSEDED_TIMING_CORRECTION_NO_NEW_SELECTION" or timing.get("prospective_selection_authority") is not False:
        raise ValueError("timing correction R2 status/authority mismatch")
    authorities = timing.get("authoritative_pre_metric_materials", {})
    if len(authorities) != 4:
        raise ValueError("timing correction must bind four earlier authorities")
    authority_bindings = {}
    superseded_path = Path(timing["correction"]["superseded_file"]["path"]).expanduser().resolve()
    for role, record in authorities.items():
        authority = Path(record["path"]).expanduser().resolve(strict=True)
        if authority == superseded_path or sha256(authority) != str(record["sha256"]):
            raise ValueError(f"invalid timing authority: {role}")
        authority_bindings[role] = bind(authority)
    bindings = {
        "plotter": bind(PLOTTER),
        "direct_plot_dependencies": [bind(path) for path in DIRECT_PLOT_DEPENDENCIES],
        "existing_IVM": {
            "arc500_result": bind(args.ivm_result), "arc500_audit": bind(args.ivm_audit),
            "terminal_result": bind(args.ivm_terminal_result), "terminal_audit": bind(args.ivm_terminal_audit),
            "long_verification": bind(args.ivm_long_verification), "prefix_verification": bind(args.ivm_prefix_verification),
        },
        "new_IVM_four": {"result": bind(args.ivm_dense_result), "audit": bind(args.ivm_dense_audit), "contract": bind(args.ivm_dense_contract)},
        "existing_ODE_NEB": [{"result": bind(result), "audit": bind(audit)} for result, audit in zip(args.ode_result, args.ode_audit)],
        "new_ODE_NEB_four": {"result": bind(args.ode_dense_result), "audit": bind(args.ode_dense_audit), "contract": bind(args.ode_dense_contract)},
        "timing_correction_R2": bind(timing_path),
        "authoritative_pre_metric_materials": authority_bindings,
        "ODE_NEB_targeted_rounds": [{"selection": bind(selection), "result": bind(result), "audit": bind(audit)} for selection, result, audit in args.ode_targeted_bundle],
    }
    if args.ode_target_selection is not None:
        bindings["legacy_ODE_NEB_target_selection"] = bind(args.ode_target_selection)
    selection_candidates = [Path(bundle[0]).expanduser().resolve(strict=True) for bundle in args.ode_targeted_bundle]
    if args.ode_target_selection is not None:
        selection_candidates.append(args.ode_target_selection.expanduser().resolve(strict=True))
    nf30_authority = None
    for selection_path in reversed(selection_candidates):
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        receipts = selection.get("bindings", {}).get("paid_force_receipts", [])
        matches = [record for record in receipts if int(record["optimization_N_F"]) == 30]
        if len(matches) == 1:
            paid_states = [record for record in selection["official_paid_states"] if int(record["optimization_N_F"]) == 30]
            if len(paid_states) != 1:
                raise ValueError("NF30 official paid state absent/nonunique")
            receipt = Path(matches[0]["path"]).expanduser().resolve(strict=True)
            archive = Path(paid_states[0]["coordinate_archive"]).expanduser().resolve(strict=True)
            if sha256(receipt) != "afcdddd09457d74d20b3c6fd885b7eb0b92f96333003890a876c127223c2f4f6" or str(matches[0]["official_path32_sha256"]) != "8fadf37e320c936deeac7d7d24bfd8a9b789916d156c9ecf185b55fc6dbe8ae7":
                raise ValueError("NF30 paid receipt identity mismatch")
            nf30_authority = {
                "targeted_R2_selection": bind(selection_path),
                "paid_force_receipt": bind(receipt),
                "coordinate_archive": bind(archive),
                "optimization_audit": bind(Path(selection["bindings"]["optimization_audit"]["path"])),
                "frozen_endpoints": bind(Path(selection["bindings"]["frozen_endpoints"]["path"])),
                "official_path32_sha256": "8fadf37e320c936deeac7d7d24bfd8a9b789916d156c9ecf185b55fc6dbe8ae7",
                "optimization_N_F": 30,
            }
            break
    if nf30_authority is None:
        raise ValueError("no bound targeted selection supplies the NF30 paid receipt")
    ivm_dense_contract = json.loads(args.ivm_dense_contract.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    ode_dense_contract = json.loads(args.ode_dense_contract.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    for role in ("topology", "bb_core"):
        if ivm_dense_contract.get(role) != ode_dense_contract.get(role):
            raise ValueError(f"IVM/ODE dense contracts bind different {role}")
        record = ivm_dense_contract[role]
        path = Path(record["path"]).expanduser().resolve(strict=True)
        if sha256(path) != str(record["sha256"]):
            raise ValueError(f"dense-contract {role} binding drift")
        nf30_authority[role] = bind(path)
    nf30_authority["system_identity_enforcement"] = "shared frozen bb_core OpenMMBackend source validates topology SHA, OpenMM version, System XML SHA/bytes, force-class sequence, CPU/double, and one thread"
    bindings["NF30_common_initial_reuse_authority"] = nf30_authority
    payload = {
        "schema": "bb-densified-cost-to-precision-figure-contract-v1",
        "status": "FROZEN_DENSIFIED_INPUT_BINDINGS",
        "figure_output": str(figure),
        "figure_output_policy": {"overwrite": False, "required_stem": EXPECTED_STEM},
        "claim_policy": {
            "main_curve": "observed lower envelope",
            "show_all_raw_points_faintly": True,
            "Delta_E_monotonicity_assumed": False,
            "interpolation": "forbidden",
            "same_path_discrete_I_perp_and_arc500_Delta_E": True,
            "ODE_NEB_NF30_reuse": "only after byte-identical equality with IVM common initial path",
            "allowed_passage_term": "earliest observed first passage unless complete earlier-path coverage certifies exactness",
            "all_cost_ratios": "observed only",
            "both_methods_have_incomplete_earlier_path_coverage": True,
        },
        "expected_point_inventory_before_NF30_common_reuse": {"IVM_existing": 5, "IVM_new": 4, "ODE_NEB_existing": 7, "ODE_NEB_new": 4},
        "metric_protocol": {"discrete_metric": "I_perp", "densified_metric": "Delta E", "coordinate_space": "unweighted Cartesian", "sampling": "uniform arclength", "points": 500, "optimization_N_F_contribution": 0},
        "bindings": bindings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(json.dumps({"status": payload["status"], "contract": str(destination), "sha256": sha256(destination), "figure_output": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
