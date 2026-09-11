"""Deterministic zero-NoF article reparameterization for Ala2D.

This is the spring-only ``reparam_by_neb`` map used by the article's Ala2D
code, specialized to a periodic two-coordinate path.  At every geometry-only
iteration it uses shortest-branch segment vectors, the article centered
tangent ``X[i+1]-X[i-1]``, and

    F_sp[i] = k_s (|d_plus|-|d_minus|) t_centered[i].

The endpoints are immutable.  The registered fixed ``dtau``, tolerance, and
iteration cap determine one reproducible map.  Failure to meet the tolerance
is typed and cannot be treated as a paid state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Dict

import numpy as np

from riemannian_torus_geometry import (
    arclength_residual,
    periodic_difference,
    unwrap_path,
)


Array = np.ndarray
REPARAMETERIZER_ID = "ala2d-article-centered-tangent-spring-neb-v13"
REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD = 1.0e-14
REPARAMETERIZATION_NORMALIZATION_EPSILON = 1.0e-16


class ArticleReparameterizationError(RuntimeError):
    """The registered article spring map is undefined or did not converge."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    header = _canonical_json(
        {"schema": "Binary64Array/ala2d-v13", "shape": list(array.shape)}
    )
    return hashlib.sha256(header + b"\x00" + array.tobytes()).hexdigest()


@dataclass(frozen=True)
class ArticleReparameterizationResult:
    path: Array
    iteration_count: int
    stopping_pre_update_spring_residual: float
    output_spring_residual: float
    segment_lengths: Array
    maximum_neighbor_length_difference: float
    actual_chord_relative_deviation: float
    actual_I_arc: float
    audit_sha256: str

    def audit_payload(
        self,
        *,
        source_path: Array,
        k_s: float,
        dtau: float,
        tolerance: float,
        max_iterations: int,
    ) -> Dict[str, object]:
        return {
            "schema": "Ala2DArticleNEBReparameterizationAudit/v13",
            "reparameterizer_id": REPARAMETERIZER_ID,
            "source_path_sha256": array_sha256(source_path),
            "output_path_sha256": array_sha256(self.path),
            "segment_lengths_sha256": array_sha256(self.segment_lengths),
            "k_s_binary64": float(k_s).hex(),
            "dtau_binary64": float(dtau).hex(),
            "tolerance_binary64": float(tolerance).hex(),
            "max_iterations": int(max_iterations),
            "reparameterization_zero_segment_threshold_binary64": float(
                REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
            ).hex(),
            "reparameterization_normalization_epsilon_binary64": float(
                REPARAMETERIZATION_NORMALIZATION_EPSILON
            ).hex(),
            "iteration_count": int(self.iteration_count),
            "stopping_pre_update_spring_residual_binary64": float(
                self.stopping_pre_update_spring_residual
            ).hex(),
            "output_spring_residual_binary64": float(
                self.output_spring_residual
            ).hex(),
            "maximum_neighbor_length_difference_binary64": float(
                self.maximum_neighbor_length_difference
            ).hex(),
            "actual_chord_relative_deviation_binary64": float(
                self.actual_chord_relative_deviation
            ).hex(),
            "actual_I_arc_binary64": float(self.actual_I_arc).hex(),
            "fixed_endpoints": True,
            "shortest_periodic_branch": True,
            "centered_tangent_formula": "X[i+1]-X[i-1]",
            "article_operation_order": (
                "compute_pre_update_residual_then_update_then_test_pre_update_residual"
            ),
            "additional_correct_path_gate": (
                "post_update_output_residual_strictly_less_than_tolerance"
            ),
            "potential_calls": 0,
            "mean_force_evaluations": 0,
        }


def reparameterize_article_neb(
    preliminary_trial: Array,
    *,
    k_s: float = 1.0,
    dtau: float = 0.5,
    tolerance: float = 1.0e-6,
    max_iterations: int = 20000,
) -> ArticleReparameterizationResult:
    """Apply the article centered-tangent spring update until registered tol."""

    source = np.asarray(preliminary_trial, dtype=np.float64)
    if source.shape != (22, 2) or not np.all(np.isfinite(source)):
        raise ValueError("the Ala2D preliminary trial must be finite shape (22,2)")
    if not math.isfinite(float(k_s)) or float(k_s) <= 0.0:
        raise ValueError("spring constant must be finite and positive")
    if not math.isfinite(float(dtau)) or float(dtau) <= 0.0:
        raise ValueError("reparameterization dtau must be finite and positive")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("reparameterization tolerance must be finite and positive")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("reparameterization max_iterations must be positive")

    path = unwrap_path(source)
    start = path[0].copy()
    end = path[-1].copy()
    stopping_pre_update_residual = math.inf
    output_residual = math.inf
    iteration_count = 0
    for iteration in range(max_iterations):
        segments = periodic_difference(path[1:], path[:-1])
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(
            lengths <= REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
        ):
            raise ArticleReparameterizationError(
                "article spring map encountered a zero or non-finite segment"
            )
        centered = periodic_difference(path[2:], path[:-2])
        tangent_norms = np.linalg.norm(centered, axis=1)
        if np.any(~np.isfinite(tangent_norms)) or np.any(
            tangent_norms <= REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
        ):
            raise ArticleReparameterizationError(
                "article centered tangent is zero or non-finite"
            )
        tangents = centered / (
            tangent_norms[:, None] + REPARAMETERIZATION_NORMALIZATION_EPSILON
        )
        spring = (
            float(k_s)
            * (lengths[1:] - lengths[:-1])[:, None]
            * tangents
        )
        pre_update_residual = float(np.max(np.linalg.norm(spring, axis=1)))
        if not math.isfinite(pre_update_residual):
            raise ArticleReparameterizationError("spring residual is non-finite")

        # Match the article operation order: update first, then test the
        # residual computed from the beginning of this iteration.
        path[1:-1] += float(dtau) * spring
        path = unwrap_path(path)
        path[0], path[-1] = start, end
        iteration_count = iteration + 1

        # The article tests the residual computed immediately before the
        # update above.  Independently audit the actual output path: the
        # protocol's correct-path gate applies to this post-update residual.
        output_segments = periodic_difference(path[1:], path[:-1])
        output_lengths = np.linalg.norm(output_segments, axis=1)
        output_centered = periodic_difference(path[2:], path[:-2])
        output_tangent_norms = np.linalg.norm(output_centered, axis=1)
        if (
            np.any(~np.isfinite(output_lengths))
            or np.any(
                output_lengths <= REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
            )
            or np.any(~np.isfinite(output_tangent_norms))
            or np.any(
                output_tangent_norms
                <= REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
            )
        ):
            raise ArticleReparameterizationError(
                "article spring output has a zero or non-finite geometry"
            )
        output_tangents = output_centered / (
            output_tangent_norms[:, None]
            + REPARAMETERIZATION_NORMALIZATION_EPSILON
        )
        output_spring = (
            float(k_s)
            * (output_lengths[1:] - output_lengths[:-1])[:, None]
            * output_tangents
        )
        output_residual = float(np.max(np.linalg.norm(output_spring, axis=1)))
        if not math.isfinite(output_residual):
            raise ArticleReparameterizationError(
                "article spring output residual is non-finite"
            )
        if pre_update_residual < float(tolerance) and output_residual < float(tolerance):
            stopping_pre_update_residual = pre_update_residual
            break
    else:
        raise ArticleReparameterizationError(
            "article spring map did not converge within the registered cap; "
            f"last pre-update residual={pre_update_residual:.9e}, "
            f"last output residual={output_residual:.9e}"
        )

    if not np.array_equal(path[[0, -1]], np.vstack((start, end))):
        raise ArticleReparameterizationError("article spring map moved an endpoint")
    segments = periodic_difference(path[1:], path[:-1])
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(
        lengths <= REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
    ):
        raise ArticleReparameterizationError("article spring output is degenerate")
    neighbor_difference = float(np.max(np.abs(np.diff(lengths))))
    chord_deviation = float(
        np.max(np.abs(lengths / float(np.mean(lengths)) - 1.0))
    )
    i_arc = float(arclength_residual(path))
    if not all(math.isfinite(x) for x in (neighbor_difference, chord_deviation, i_arc)):
        raise ArticleReparameterizationError("article spring diagnostics are non-finite")
    if not stopping_pre_update_residual < float(tolerance):
        raise ArticleReparameterizationError(
            "article spring stopping pre-update residual missed tolerance"
        )
    if not output_residual < float(tolerance):
        raise ArticleReparameterizationError(
            "article spring output residual missed tolerance"
        )

    provisional = ArticleReparameterizationResult(
        np.ascontiguousarray(path),
        int(iteration_count),
        stopping_pre_update_residual,
        output_residual,
        np.ascontiguousarray(lengths),
        neighbor_difference,
        chord_deviation,
        i_arc,
        "",
    )
    audit_sha = hashlib.sha256(
        _canonical_json(
            provisional.audit_payload(
                source_path=source,
                k_s=k_s,
                dtau=dtau,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        )
    ).hexdigest()
    return ArticleReparameterizationResult(
        provisional.path,
        provisional.iteration_count,
        provisional.stopping_pre_update_spring_residual,
        provisional.output_spring_residual,
        provisional.segment_lengths,
        provisional.maximum_neighbor_length_difference,
        provisional.actual_chord_relative_deviation,
        provisional.actual_I_arc,
        audit_sha,
    )


__all__ = [
    "Array",
    "ArticleReparameterizationError",
    "ArticleReparameterizationResult",
    "REPARAMETERIZER_ID",
    "REPARAMETERIZATION_NORMALIZATION_EPSILON",
    "REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD",
    "array_sha256",
    "reparameterize_article_neb",
]
