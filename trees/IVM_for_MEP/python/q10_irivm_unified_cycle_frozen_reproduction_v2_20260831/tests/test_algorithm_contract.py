from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import unittest


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_ROOT))

from src.static_contract import (
    DIRECT_POTENTIAL_CALLS,
    FORBIDDEN_GEOMETRY_FRAGMENTS,
    FORBIDDEN_OPTIMIZER_DATA_FRAGMENTS,
    _comparison_names_and_numbers,
    _definitions,
    _direct_calls,
    _identifiers_and_strings,
    _reachable_local,
    analyze_source_texts,
)


REPOSITORY_ROOT = CAMPAIGN_ROOT.parents[1]
BROAD_CORE = REPOSITORY_ROOT / (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
EXPECTED_OPTIMIZER_SHA256 = (
    "1b9aa22908256a725baec013127edb07f9bff6615513182e9c317a69e66ea1ab"
)
EXPECTED_BROAD_CORE_SHA256 = (
    "2e12b8817730b56ddef0594c932cfc0c3f4642795af25c623ca82c7a726a66a0"
)


COMPLIANT_OPTIMIZER = '''
from . import irivm_runtime
FORMAL_OPTIMIZER_ENTRYPOINT = "run_registered_member"
FORMAL_RUNTIME_MODULE = "irivm_runtime"
FORMAL_REPARAMETERIZATION_ENTRYPOINT = "hj_spring_reparameterize"
FORMAL_TANGENT_ENTRYPOINT = "hj_improved_tangents"
def run_registered_member(n_internal, path, energies):
    return irivm_runtime.recovery_cycle(path, energies)
'''

COMPLIANT_RUNTIME = '''
def hj_improved_tangents(path, energies):
    return path[1:-1]
def hj_spring_reparameterize(path, energies):
    tangents = hj_improved_tangents(path, energies)
    return path, tangents
def recovery_cycle(path, energies):
    return hj_spring_reparameterize(path, energies)
'''


class ContractAnalyzerTests(unittest.TestCase):
    def test_compliant_fixture_passes(self) -> None:
        report = analyze_source_texts(COMPLIANT_OPTIMIZER, COMPLIANT_RUNTIME)
        self.assertTrue(report.passed, report.violations)

    def test_resolution_specific_branch_fails(self) -> None:
        optimizer = COMPLIANT_OPTIMIZER.replace(
            "return irivm_runtime.recovery_cycle(path, energies)",
            "\n    if n_internal == 20:\n        path = path.copy()\n    return irivm_runtime.recovery_cycle(path, energies)",
        )
        report = analyze_source_texts(optimizer, COMPLIANT_RUNTIME)
        self.assertFalse(report.passed)
        self.assertTrue(any("resolution-specific" in item for item in report.violations))

    def test_article_dependency_fails(self) -> None:
        optimizer = COMPLIANT_OPTIMIZER.replace(
            "return irivm_runtime.recovery_cycle(path, energies)",
            "article_checkpoint = 20\n    return irivm_runtime.recovery_cycle(path, energies)",
        )
        report = analyze_source_texts(optimizer, COMPLIANT_RUNTIME)
        self.assertFalse(report.passed)
        self.assertTrue(any("offline data" in item for item in report.violations))

    def test_alternate_polyline_reparameterization_fails(self) -> None:
        runtime = COMPLIANT_RUNTIME.replace(
            "return hj_spring_reparameterize(path, energies)",
            "return polyline_reparameterize(path, energies)",
        ) + "\ndef polyline_reparameterize(path, energies):\n    return path\n"
        report = analyze_source_texts(COMPLIANT_OPTIMIZER, runtime)
        self.assertFalse(report.passed)
        self.assertTrue(any("polyline" in item for item in report.violations))

    def test_materialized_candidate_and_hash_pinned_broad_core(self) -> None:
        optimizer = CAMPAIGN_ROOT / "src" / "q10_unified_cycle_optimizer.py"
        self.assertTrue(optimizer.is_file(), "formal optimizer is not materialized")
        self.assertTrue(BROAD_CORE.is_file(), "hash-pinned broad core is missing")
        self.assertEqual(
            hashlib.sha256(optimizer.read_bytes()).hexdigest(),
            EXPECTED_OPTIMIZER_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(BROAD_CORE.read_bytes()).hexdigest(),
            EXPECTED_BROAD_CORE_SHA256,
        )

        optimizer_tree = ast.parse(
            optimizer.read_text(encoding="utf-8"), filename=str(optimizer)
        )
        runtime_tree = ast.parse(
            BROAD_CORE.read_text(encoding="utf-8"), filename=str(BROAD_CORE)
        )
        optimizer_definitions = _definitions(optimizer_tree)
        runtime_definitions = _definitions(runtime_tree)
        optimizer_reachable, _ = _reachable_local(
            optimizer_definitions, ["run_clean"]
        )
        self.assertIn(
            "hj_nearest_neighbor_spring_reparameterize", optimizer_reachable
        )

        optimizer_calls = []
        for name in sorted(optimizer_reachable):
            optimizer_calls.extend(_direct_calls(optimizer_definitions[name]))
        runtime_seeds = {
            target.split(".", 1)[1]
            for target in optimizer_calls
            if target.startswith("alg.")
        }
        self.assertFalse(runtime_seeds - set(runtime_definitions))
        runtime_reachable, _ = _reachable_local(runtime_definitions, runtime_seeds)

        imports = [
            alias
            for node in optimizer_tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "q10_irivm_atomic_tangent_secant_core"
        ]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].asname, "alg")

        runtime_calls = []
        for name in sorted(runtime_reachable):
            runtime_calls.extend(_direct_calls(runtime_definitions[name]))
        all_reachable_calls = optimizer_calls + runtime_calls
        reparameterization_calls = [
            target
            for target in all_reachable_calls
            if "reparam" in target.rsplit(".", 1)[-1].lower()
        ]
        self.assertEqual(
            reparameterization_calls,
            ["hj_nearest_neighbor_spring_reparameterize"] * 5,
        )
        improved_tangent_calls = [
            target
            for target in all_reachable_calls
            if target.rsplit(".", 1)[-1] == "improved_tangents"
        ]
        self.assertEqual(improved_tangent_calls, ["core.improved_tangents"] * 6)

        spring_source = ast.get_source_segment(
            optimizer.read_text(encoding="utf-8"),
            optimizer_definitions["hj_nearest_neighbor_spring_reparameterize"],
        )
        self.assertIsNotNone(spring_source)
        self.assertIn(
            "length_differences = lengths[1:] - lengths[:-1]", spring_source
        )
        self.assertIn(
            "spring_force = length_differences[:, None] * tangents", spring_source
        )
        self.assertIn(
            "result[1:-1] += alg.SPRING_DTAU * spring_force", spring_source
        )
        spring_calls = _direct_calls(
            optimizer_definitions["hj_nearest_neighbor_spring_reparameterize"]
        )
        self.assertEqual(spring_calls.count("core.improved_tangents"), 1)

        dormant_forbidden = {
            name
            for name in runtime_definitions
            if any(fragment in name.lower() for fragment in FORBIDDEN_GEOMETRY_FRAGMENTS)
        }
        self.assertTrue(
            dormant_forbidden,
            "broad-core test must distinguish dormant from reachable surfaces",
        )
        self.assertFalse(dormant_forbidden & runtime_reachable)
        reachable_nodes = [
            optimizer_definitions[name] for name in sorted(optimizer_reachable)
        ] + [runtime_definitions[name] for name in sorted(runtime_reachable)]
        forbidden_geometry = [
            (value, line)
            for value, line in _identifiers_and_strings(reachable_nodes)
            if any(fragment in value for fragment in FORBIDDEN_GEOMETRY_FRAGMENTS)
        ]
        self.assertEqual(forbidden_geometry, [])

        direct_potential = [
            target
            for target in optimizer_calls
            if target.rsplit(".", 1)[-1].lower() in DIRECT_POTENTIAL_CALLS
        ]
        self.assertEqual(direct_potential, [])
        optimizer_offline_tokens = [
            (value, line)
            for value, line in _identifiers_and_strings(
                [optimizer_definitions[name] for name in sorted(optimizer_reachable)]
            )
            if any(
                fragment in value
                for fragment in FORBIDDEN_OPTIMIZER_DATA_FRAGMENTS
            )
        ]
        self.assertEqual(optimizer_offline_tokens, [])

        # The broad core may write the historical ``article_nf`` trace label,
        # but no reachable optimizer/runtime decision may read Article,
        # checkpoint, reference-barrier, or offline-metric data.
        forbidden_decisions = []
        for owner, definitions, names in (
            ("optimizer", optimizer_definitions, optimizer_reachable),
            ("runtime", runtime_definitions, runtime_reachable),
        ):
            for name in sorted(names):
                for node in ast.walk(definitions[name]):
                    if not isinstance(node, (ast.If, ast.IfExp)):
                        continue
                    identifiers, _ = _comparison_names_and_numbers(node.test)
                    if any(
                        fragment in identifier
                        for identifier in identifiers
                        for fragment in FORBIDDEN_OPTIMIZER_DATA_FRAGMENTS
                    ):
                        forbidden_decisions.append((owner, name, node.lineno))
        self.assertEqual(forbidden_decisions, [])


if __name__ == "__main__":
    unittest.main()
