"""Auditable surrogate and restrained-MD oracles for the 2D Ala MFEP.

One successful query at one internal image returns the mean-force gradient and
the collective-variable tensor from the same restrained trajectory.  It is
therefore exactly one image-wise mean-force evaluation (one NoF).  Coordinate
cache hits are deliberately forbidden for MD: a repeated center is a new
statistical sample and is charged again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from riemannian_torus_geometry import TWO_PI, wrap_periodic


Array = np.ndarray
KAPPA_KJ_PER_MOL_RAD2 = 1000.0
KJ_PER_KCAL = 4.184


class OracleError(RuntimeError):
    """Fail-closed oracle error."""


@dataclass(frozen=True)
class OracleBand:
    """One complete band containing 20 image-wise mean-force evaluations."""

    centers: Array
    mean_coordinates: Array
    grad_F: Array
    M: Array
    grad_F_block_means: Array
    M_block_means: Array
    seeds: Array
    thermostat_seeds: Array
    frame_counts: Array
    discarded_frame_counts: Array
    wall_seconds: Array
    nof: int
    metadata: Dict[str, object]


def _atomic_save_npz(path: Path, **arrays: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_once_json(path: Path, payload: Dict[str, object]) -> None:
    """Create *path* once, or require byte-for-byte-equivalent JSON content.

    A hard-link publish gives an exclusive, atomic create on the local
    filesystem.  The registered campaign runs formal MD on one named local
    host, so this exclusive publish prevents two local writers from silently
    choosing different first-observed expanded topologies.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise OracleError(
                    f"invalid write-once campaign binding {path}"
                ) from error
            if existing != payload:
                raise OracleError(
                    "expanded topology differs from the write-once campaign binding"
                )
    finally:
        temporary_path.unlink(missing_ok=True)


def campaign_topology_binding_payload(
    processed_topology_sha256: str, registration_sha256: str
) -> Dict[str, object]:
    """Canonical payload shared by the oracle and campaign-level audit."""

    if not re.fullmatch(r"[0-9a-f]{64}", processed_topology_sha256):
        raise OracleError("invalid processed-topology SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", registration_sha256):
        raise OracleError("invalid registration SHA256 for topology binding")
    return {
        "binding_kind": "campaign-global-expanded-topology-sha256",
        "registration_sha256": registration_sha256,
        "processed_topology_sha256": processed_topology_sha256,
        "scope": "every pilot, confirmation, and terminal-diagnostic MD evaluation",
        "meaning": (
            "SHA256 of the fully expanded topology emitted by grompp -pp; "
            "binds dynamically included force-field files across the campaign"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_seed(
    master_seed: int,
    band_id: int,
    image_index: int,
    stream: str,
) -> int:
    payload = (
        f"ala2d-irivm|{int(master_seed)}|{int(band_id)}|"
        f"{int(image_index)}|{stream}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (2**31 - 2)


def _contiguous_blocks(n_items: int, n_blocks: int) -> List[Array]:
    if n_items < 1:
        raise ValueError("cannot block an empty trajectory")
    count = min(int(n_blocks), int(n_items))
    return [block for block in np.array_split(np.arange(n_items), count) if block.size]


def _version_line(command: Sequence[str], prefix: str) -> str:
    completed = subprocess.run(
        [str(item) for item in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped
    return completed.stdout.strip().splitlines()[-1]


class SurrogateOracle:
    """Deterministic prescreening oracle; never valid as formal MD evidence.

    The free-energy gradient comes from the independent metadynamics grid.
    The position-dependent tensor is a smooth inverse-distance interpolation
    of tensors already sampled in the historical current-IVM trajectory.
    """

    oracle_id = "fes-spline-plus-historical-M-knn-prescreen-v1"
    formal_evidence = False

    def __init__(
        self,
        fes_npz: Path,
        metric_bank_npz: Path,
        metric_neighbors: int = 32,
        metric_bandwidth_rad: float = 0.25,
    ) -> None:
        self.fes_npz = Path(fes_npz).resolve()
        self.metric_bank_npz = Path(metric_bank_npz).resolve()
        with np.load(self.fes_npz) as loaded:
            x_deg = np.asarray(loaded["X"], dtype=np.float64)[0]
            y_deg = np.asarray(loaded["Y"], dtype=np.float64)[:, 0]
            # PLUMED/GROMACS writes kJ mol^-1; public curves use kcal mol^-1.
            free_energy = np.asarray(loaded["Z"], dtype=np.float64) / KJ_PER_KCAL
        self.x = np.deg2rad(x_deg)
        self.y = np.deg2rad(y_deg)
        self.spline = RectBivariateSpline(
            self.y, self.x, free_energy, kx=3, ky=3, s=0.0
        )

        with np.load(self.metric_bank_npz, allow_pickle=False) as loaded:
            positions = np.asarray(loaded["oracle_sample_z"], dtype=np.float64)
            metrics = np.asarray(loaded["oracle_sample_M"], dtype=np.float64)
        shifts = np.asarray(
            [(i * TWO_PI, j * TWO_PI) for i in (-1, 0, 1) for j in (-1, 0, 1)],
            dtype=np.float64,
        )
        tiled_positions = np.concatenate(
            [wrap_periodic(positions) + shift for shift in shifts], axis=0
        )
        self.metric_values = np.concatenate([metrics for _ in shifts], axis=0)
        self.metric_tree = cKDTree(tiled_positions)
        self.metric_neighbors = int(metric_neighbors)
        self.metric_bandwidth_rad = float(metric_bandwidth_rad)
        self.call_count = 0

    def provenance(self) -> Dict[str, object]:
        return {
            "oracle_id": self.oracle_id,
            "formal_evidence": False,
            "fes_npz": str(self.fes_npz),
            "fes_npz_sha256": _sha256_file(self.fes_npz),
            "metric_bank_npz": str(self.metric_bank_npz),
            "metric_bank_npz_sha256": _sha256_file(self.metric_bank_npz),
            "metric_neighbors": self.metric_neighbors,
            "metric_bandwidth_rad": self.metric_bandwidth_rad,
        }

    def _gradient(self, centers: Array) -> Array:
        wrapped = wrap_periodic(centers)
        result = np.empty_like(wrapped)
        # RectBivariateSpline arguments are (y, x); dx differentiates y.
        for index, (phi, psi) in enumerate(wrapped):
            result[index, 0] = self.spline.ev(psi, phi, dx=0, dy=1)
            result[index, 1] = self.spline.ev(psi, phi, dx=1, dy=0)
        return result

    def _metric(self, centers: Array) -> Array:
        wrapped = wrap_periodic(centers)
        distances, indices = self.metric_tree.query(
            wrapped, k=self.metric_neighbors, workers=1
        )
        distances = np.atleast_2d(distances)
        indices = np.atleast_2d(indices)
        if distances.shape[0] != wrapped.shape[0]:
            distances = distances.T
            indices = indices.T
        weights = np.exp(
            -0.5 * (distances / max(self.metric_bandwidth_rad, 1.0e-12)) ** 2
        )
        weights /= np.sum(weights, axis=1, keepdims=True)
        values = self.metric_values[indices]
        metric = np.einsum("nk,nkij->nij", weights, values)
        return 0.5 * (metric + np.swapaxes(metric, 1, 2))

    def evaluate_band(
        self,
        full_path: Array,
        band_id: int,
        band_directory: Optional[Path] = None,
    ) -> OracleBand:
        del band_directory
        centers = np.asarray(full_path, dtype=np.float64)[1:-1]
        grad_F = self._gradient(centers)
        metric = self._metric(centers)
        self.call_count += centers.shape[0]
        return OracleBand(
            centers=centers.copy(),
            mean_coordinates=centers.copy(),
            grad_F=grad_F,
            M=metric,
            grad_F_block_means=grad_F[:, None, :],
            M_block_means=metric[:, None, :, :],
            seeds=np.full(centers.shape[0], -1, dtype=np.int64),
            thermostat_seeds=np.full(centers.shape[0], -1, dtype=np.int64),
            frame_counts=np.ones(centers.shape[0], dtype=np.int64),
            discarded_frame_counts=np.zeros(centers.shape[0], dtype=np.int64),
            wall_seconds=np.zeros(centers.shape[0], dtype=np.float64),
            nof=int(centers.shape[0]),
            metadata={
                "oracle_id": self.oracle_id,
                "formal_evidence": False,
                "band_id": int(band_id),
            },
        )


class RestrainedMDOracle:
    """Serial GROMACS/PLUMED oracle with deterministic velocity seeds."""

    oracle_id = "ala2d-gromacs-plumed-restrained-md-v1"
    formal_evidence = True
    mdrun_options = (
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-ntmpi",
        "1",
        "-ntomp",
        "1",
        "-pin",
        "off",
    )

    def __init__(
        self,
        input_directory: Path,
        run_directory: Path,
        master_seed: int,
        *,
        gmx_executable: Path = Path("/usr/local/gromacs/bin/gmx"),
        plumed_kernel: Path = Path(
            "/Users/wuzhiyou/plumed/lib/libplumedKernel.dylib"
        ),
        discard_fraction: float = 0.05,
        block_count: int = 10,
        expected_production_frame_count: Optional[int] = 2501,
        maximum_attempts_per_logical_evaluation: int = 3,
        campaign_topology_binding_path: Optional[Path] = None,
        expected_processed_topology_sha256: Optional[str] = None,
        quiet: bool = True,
    ) -> None:
        self.input_directory = Path(input_directory).resolve()
        self.run_directory = Path(run_directory).resolve()
        self.master_seed = int(master_seed)
        self.gmx_executable = Path(gmx_executable)
        self.plumed_kernel = Path(plumed_kernel)
        self.discard_fraction = float(discard_fraction)
        self.block_count = int(block_count)
        self.expected_production_frame_count = (
            None
            if expected_production_frame_count is None
            else int(expected_production_frame_count)
        )
        self.maximum_attempts_per_logical_evaluation = int(
            maximum_attempts_per_logical_evaluation
        )
        self.campaign_topology_binding_path = (
            None
            if campaign_topology_binding_path is None
            else Path(campaign_topology_binding_path).resolve()
        )
        self.expected_processed_topology_sha256 = (
            None
            if expected_processed_topology_sha256 is None
            else str(expected_processed_topology_sha256)
        )
        self.quiet = bool(quiet)
        if not 0.0 <= self.discard_fraction < 1.0:
            raise ValueError("discard_fraction must lie in [0,1)")
        if self.block_count < 2:
            raise ValueError("block_count must be at least two")
        if (
            self.expected_production_frame_count is not None
            and self.expected_production_frame_count < self.block_count
        ):
            raise ValueError("expected production frame count is too small")
        if self.maximum_attempts_per_logical_evaluation < 1:
            raise ValueError("maximum attempts must be positive")
        if (
            self.expected_processed_topology_sha256 is not None
            and not re.fullmatch(
                r"[0-9a-f]{64}", self.expected_processed_topology_sha256
            )
        ):
            raise ValueError("expected processed-topology SHA256 is malformed")
        for required in (
            "ala2.pdb",
            "nvt.tpr",
            "nvt.mdp",
            "topol.top",
            "posre.itp",
            "processed_topology_reference.top",
        ):
            if not (self.input_directory / required).is_file():
                raise FileNotFoundError(self.input_directory / required)
        if not self.gmx_executable.is_file():
            raise FileNotFoundError(self.gmx_executable)
        if not self.plumed_kernel.is_file():
            raise FileNotFoundError(self.plumed_kernel)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.attempt_ledger = self.run_directory / "md_attempts.jsonl"
        self.atomic_attempt_directory = self.run_directory / "md_attempt_records"
        self.oracle_binding = self.run_directory / "oracle_binding.json"
        self.registration_binding = self.run_directory / "registration_binding.json"
        self._input_hashes = {
            name: _sha256_file(self.input_directory / name)
            for name in (
                "ala2.pdb",
                "nvt.tpr",
                "nvt.mdp",
                "topol.top",
                "posre.itp",
                "processed_topology_reference.top",
            )
        }
        self._gmx_sha256 = _sha256_file(self.gmx_executable)
        self._plumed_kernel_sha256 = _sha256_file(self.plumed_kernel)

    def provenance(self) -> Dict[str, object]:
        plumed_executable = self.plumed_kernel.parent.parent / "bin" / "plumed"
        return {
            "oracle_id": self.oracle_id,
            "formal_evidence": True,
            "master_seed": self.master_seed,
            "discard_fraction": self.discard_fraction,
            "block_count": self.block_count,
            "expected_production_frame_count": self.expected_production_frame_count,
            "maximum_attempts_per_logical_evaluation": (
                self.maximum_attempts_per_logical_evaluation
            ),
            "campaign_topology_binding_path": (
                None
                if self.campaign_topology_binding_path is None
                else str(self.campaign_topology_binding_path)
            ),
            "expected_processed_topology_sha256": (
                self.expected_processed_topology_sha256
            ),
            "quiet": self.quiet,
            "gmx_executable": str(self.gmx_executable.resolve()),
            "gmx_sha256": self._gmx_sha256,
            "gmx_version": _version_line(
                [self.gmx_executable, "--version"], "GROMACS version:"
            ),
            "plumed_kernel": str(self.plumed_kernel.resolve()),
            "plumed_kernel_sha256": self._plumed_kernel_sha256,
            "plumed_version": _version_line(
                [plumed_executable, "info", "--version"], ""
            ),
            "input_directory": str(self.input_directory),
            "input_sha256": dict(self._input_hashes),
            "preparation_seed_protocol": (
                "frozen nvt.tpr with embedded velocities and fixed ld_seed; "
                "the registered TPR SHA256 binds this stage"
            ),
            "production_seed_protocol": (
                "independent deterministic SHA256-derived gen_seed and ld_seed "
                "for each master-seed/band/image tuple"
            ),
            "gromacs_mdrun_threading": {
                "ntmpi": 1,
                "ntomp": 1,
                "pin": "off",
            },
            "bitwise_reproducibility_scope": (
                "auditable retries use the same registered executable, inputs, "
                "seeds, and thread settings; bitwise identity across different "
                "CPU models or hosts is not claimed"
            ),
            "one_cv_point_mean_force_vector_equals_one_nof": True,
        }

    def _bind_campaign_processed_topology(
        self, processed_topology_sha256: str, registration_sha256: str
    ) -> None:
        if self.campaign_topology_binding_path is None:
            return
        if (
            self.expected_processed_topology_sha256 is not None
            and processed_topology_sha256
            != self.expected_processed_topology_sha256
        ):
            raise OracleError(
                "expanded topology differs from the preregistered preserved reference"
            )
        payload = campaign_topology_binding_payload(
            processed_topology_sha256, registration_sha256
        )
        _atomic_write_once_json(self.campaign_topology_binding_path, payload)

    def _run_command(self, command: Sequence[str], cwd: Path, label: str) -> None:
        environment = dict(os.environ)
        environment["PLUMED_KERNEL"] = str(self.plumed_kernel)
        environment["PATH"] = (
            str(self.gmx_executable.parent)
            + os.pathsep
            + environment.get("PATH", "")
        )
        log_path = cwd / f"{label}.log"
        with log_path.open("wb") as log_stream:
            completed = subprocess.run(
                [str(item) for item in command],
                cwd=str(cwd),
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = log_path.read_text(errors="replace")[-8000:]
            raise OracleError(
                f"{label} failed with code {completed.returncode}\n{tail}"
            )
        if self.quiet:
            log_path.unlink(missing_ok=True)

    def _seeded_mdp(self, velocity_seed: int, thermostat_seed: int) -> str:
        source = (self.input_directory / "nvt.mdp").read_text(encoding="utf-8")
        gen_pattern = re.compile(r"(?m)^\s*gen_seed\s*=.*$")
        if not gen_pattern.search(source):
            raise OracleError("nvt.mdp has no gen_seed field")
        result = gen_pattern.sub(f"gen_seed        = {int(velocity_seed)}", source)
        ld_pattern = re.compile(r"(?m)^\s*ld_seed\s*=.*$")
        if ld_pattern.search(result):
            result = ld_pattern.sub(f"ld_seed         = {int(thermostat_seed)}", result)
        else:
            if not result.endswith("\n"):
                result += "\n"
            result += f"ld_seed         = {int(thermostat_seed)}\n"
        return result if result.endswith("\n") else result + "\n"

    @staticmethod
    def _plumed_files(center: Array) -> Dict[str, str]:
        phi, psi = (float(center[0]), float(center[1]))
        common = "MOLINFO STRUCTURE=./ala2.pdb\n\nphi: TORSION ATOMS=@phi-2\npsi: TORSION ATOMS=@psi-2\n"
        initial = (
            common
            + f"restraint: RESTRAINT ARG=phi,psi AT={phi:.17g},{psi:.17g} KAPPA=100.0,100.0\n\n"
            + "PRINT ARG=phi,psi,restraint.bias FILE=COLVAR_initial STRIDE=100\n"
        )
        production = (
            common
            + f"restraint: RESTRAINT ARG=phi,psi AT={phi:.17g},{psi:.17g} KAPPA=1000.0,1000.0\n\n"
            + "DUMPPROJECTIONS ARG=phi,psi FILE=PROJ STRIDE=20\n"
            + "PRINT ARG=phi,psi,restraint.bias FILE=COLVAR_restrained STRIDE=20\n"
        )
        return {"plumed_initial.dat": initial, "plumed.dat": production}

    def _parse_samples(self, center: Array, work_directory: Path) -> Dict[str, Array]:
        colvar = np.loadtxt(work_directory / "COLVAR_restrained", comments="#")
        projection = np.loadtxt(work_directory / "PROJ", comments="#")
        colvar = np.atleast_2d(colvar)
        projection = np.atleast_2d(projection)
        if colvar.shape[1] < 3 or projection.shape[1] < 5:
            raise OracleError(
                f"malformed PLUMED output: COLVAR={colvar.shape}, PROJ={projection.shape}"
            )
        if colvar.shape[0] != projection.shape[0]:
            raise OracleError(
                "COLVAR/PROJ frame counts differ: "
                f"{colvar.shape[0]} != {projection.shape[0]}"
            )
        n_total = int(colvar.shape[0])
        if (
            self.expected_production_frame_count is not None
            and n_total != self.expected_production_frame_count
        ):
            raise OracleError(
                "unexpected synchronized production frame count: "
                f"{n_total} != {self.expected_production_frame_count}"
            )
        if not np.allclose(
            colvar[:, 0], projection[:, 0], rtol=0.0, atol=1.0e-9
        ):
            maximum_time_error = float(np.max(np.abs(colvar[:, 0] - projection[:, 0])))
            raise OracleError(
                "COLVAR/PROJ time columns are not synchronized: "
                f"max error={maximum_time_error:.6e} ps"
            )
        if n_total < self.block_count:
            raise OracleError(f"too few synchronized frames: {n_total}")
        discard_count = int(np.floor(n_total * self.discard_fraction))
        if discard_count >= n_total:
            raise OracleError("discard rule removed every trajectory frame")
        dihedrals = colvar[:n_total, 1:3][discard_count:]
        projection_values = projection[:n_total, 1:5][discard_count:]

        displacement = wrap_periodic(dihedrals - center[None, :])
        mean_coordinate = center + np.mean(displacement, axis=0)
        gradient_frames = -KAPPA_KJ_PER_MOL_RAD2 * displacement / KJ_PER_KCAL
        metric_frames = np.empty((projection_values.shape[0], 2, 2), dtype=np.float64)
        metric_frames[:, 0, 0] = projection_values[:, 0]
        metric_frames[:, 1, 1] = projection_values[:, 3]
        off_diagonal = 0.5 * (projection_values[:, 1] + projection_values[:, 2])
        metric_frames[:, 0, 1] = off_diagonal
        metric_frames[:, 1, 0] = off_diagonal
        blocks = _contiguous_blocks(gradient_frames.shape[0], self.block_count)
        gradient_blocks = np.stack(
            [np.mean(gradient_frames[block], axis=0) for block in blocks], axis=0
        )
        metric_blocks = np.stack(
            [np.mean(metric_frames[block], axis=0) for block in blocks], axis=0
        )
        mean_force = np.mean(gradient_frames, axis=0)
        if mean_force.shape != (2,) or np.any(~np.isfinite(mean_force)):
            raise OracleError("restrained trajectory did not return a finite 2D mean-force vector")

        metric = np.mean(metric_frames, axis=0)
        metric = 0.5 * (metric + metric.T)
        try:
            eigenvalues = np.linalg.eigvalsh(metric)
        except np.linalg.LinAlgError:
            eigenvalues = np.full(2, np.nan, dtype=np.float64)
        metric_valid = bool(
            eigenvalues.shape == (2,)
            and np.all(np.isfinite(eigenvalues))
            and np.all(eigenvalues > 1.0e-10)
        )
        return {
            "mean_coordinate": mean_coordinate,
            "grad_F": mean_force,
            "mean_force_vector_complete": np.asarray(True),
            "M": metric,
            "metric_valid": np.asarray(metric_valid),
            "grad_F_block_means": gradient_blocks,
            "M_block_means": metric_blocks,
            "frame_count": np.asarray(n_total, dtype=np.int64),
            "discarded_frame_count": np.asarray(discard_count, dtype=np.int64),
            "metric_eigenvalues": eigenvalues,
        }

    def _evaluate_image(
        self,
        center: Array,
        velocity_seed: int,
        thermostat_seed: int,
        scratch_root: Path,
    ) -> Dict[str, Array]:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix="image_", dir=str(scratch_root)
        ) as temporary_directory:
            work_directory = Path(temporary_directory)
            for name in ("ala2.pdb", "nvt.tpr", "topol.top", "posre.itp"):
                (work_directory / name).symlink_to(self.input_directory / name)
            for name, content in self._plumed_files(center).items():
                (work_directory / name).write_text(content, encoding="utf-8")
            (work_directory / "nvt_seeded.mdp").write_text(
                self._seeded_mdp(velocity_seed, thermostat_seed), encoding="utf-8"
            )
            common_mdrun = list(self.mdrun_options)
            self._run_command(
                [
                    self.gmx_executable,
                    "mdrun",
                    "-s",
                    "nvt.tpr",
                    "-plumed",
                    "plumed_initial.dat",
                    *common_mdrun,
                    "-deffnm",
                    "initial",
                ],
                work_directory,
                "initial_mdrun",
            )
            self._run_command(
                [
                    self.gmx_executable,
                    "grompp",
                    "-f",
                    "nvt_seeded.mdp",
                    "-c",
                    "initial.gro",
                    "-r",
                    "initial.gro",
                    "-p",
                    "topol.top",
                    "-o",
                    "restrained.tpr",
                    "-pp",
                    "processed.top",
                ],
                work_directory,
                "grompp",
            )
            self._run_command(
                [
                    self.gmx_executable,
                    "mdrun",
                    "-s",
                    "restrained.tpr",
                    "-plumed",
                    "plumed.dat",
                    *common_mdrun,
                    "-deffnm",
                    "restrained",
                ],
                work_directory,
                "restrained_mdrun",
            )
            result = self._parse_samples(center, work_directory)
            result["seeded_mdp_sha256"] = np.asarray(
                _sha256_file(work_directory / "nvt_seeded.mdp")
            )
            result["processed_topology_sha256"] = np.asarray(
                _sha256_file(work_directory / "processed.top")
            )
            result["restrained_tpr_sha256"] = np.asarray(
                _sha256_file(work_directory / "restrained.tpr")
            )
        result["wall_seconds"] = np.asarray(time.perf_counter() - start)
        return result

    def _append_attempt(self, record: Dict[str, object]) -> None:
        attempt_id = record.get("attempt_id")
        status = record.get("status")
        if not isinstance(attempt_id, str) or not re.fullmatch(
            r"b\d{6}-i\d{2}-t\d+-p\d+", attempt_id
        ):
            raise OracleError("invalid MD attempt identifier")
        if status not in {
            "started",
            "failed_before_mean_force",
            "succeeded",
            "recovered_success",
            "mean_force_returned_but_ineligible",
        }:
            raise OracleError(f"invalid MD attempt status {status!r}")

        self.atomic_attempt_directory.mkdir(parents=True, exist_ok=True)
        atomic_record = self.atomic_attempt_directory / f"{attempt_id}.{status}.json"
        if atomic_record.exists():
            existing = json.loads(atomic_record.read_text(encoding="utf-8"))
            if existing != record:
                raise OracleError(f"conflicting atomic attempt record {atomic_record}")
        else:
            _atomic_save_json(atomic_record, record)

        self.attempt_ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.attempt_ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _attempt_identity(
        self,
        *,
        attempt_id: str,
        band_id: int,
        image_index: int,
        center: Array,
        velocity_seed: int,
        thermostat_seed: int,
    ) -> Dict[str, object]:
        """Return the immutable identity shared by every record of one attempt."""

        if not self.registration_binding.is_file():
            raise OracleError(
                "formal restrained-MD evaluation requires registration_binding.json"
            )
        try:
            binding = json.loads(self.registration_binding.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OracleError("invalid registration_binding.json") from error
        parameter_id = binding.get("parameter_id")
        registration_sha256 = binding.get("registration_sha256")
        if not isinstance(parameter_id, str) or not isinstance(
            registration_sha256, str
        ):
            raise OracleError("incomplete registration binding for MD attempt")
        return {
            "attempt_id": str(attempt_id),
            "parameter_id": parameter_id,
            "registration_sha256": registration_sha256,
            "registration_binding_sha256": _sha256_file(self.registration_binding),
            "band_id": int(band_id),
            "image_index": int(image_index),
            "master_seed": int(self.master_seed),
            "seed": int(velocity_seed),
            "thermostat_seed": int(thermostat_seed),
            "center": [float(center[0]), float(center[1])],
            "seeded_mdp_sha256": hashlib.sha256(
                self._seeded_mdp(velocity_seed, thermostat_seed).encode("utf-8")
            ).hexdigest(),
            "gmx_sha256": self._gmx_sha256,
            "plumed_kernel_sha256": self._plumed_kernel_sha256,
            "input_sha256": dict(self._input_hashes),
        }

    def _atomic_attempt_records(self) -> List[Dict[str, object]]:
        """Load the independently atomic source-of-truth attempt records."""

        if not self.atomic_attempt_directory.is_dir():
            return []
        records: List[Dict[str, object]] = []
        for path in sorted(self.atomic_attempt_directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise OracleError(f"invalid atomic MD attempt record {path}") from error
            if not isinstance(record, dict):
                raise OracleError(f"non-object atomic MD attempt record {path}")
            expected_name = f"{record.get('attempt_id')}.{record.get('status')}.json"
            if path.name != expected_name:
                raise OracleError(f"attempt record filename/content mismatch: {path}")
            records.append(record)
        return records

    @staticmethod
    def _same_logical_identity(
        observed: Dict[str, object], expected: Dict[str, object]
    ) -> bool:
        fields = (
            "parameter_id",
            "registration_sha256",
            "registration_binding_sha256",
            "band_id",
            "image_index",
            "master_seed",
            "seed",
            "thermostat_seed",
            "center",
            "seeded_mdp_sha256",
            "gmx_sha256",
            "plumed_kernel_sha256",
            "input_sha256",
        )
        return all(observed.get(field) == expected.get(field) for field in fields)

    def _records_for_logical_evaluation(
        self, expected: Dict[str, object]
    ) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for record in self._atomic_attempt_records():
            if (
                record.get("band_id") == expected["band_id"]
                and record.get("image_index") == expected["image_index"]
            ):
                if not self._same_logical_identity(record, expected):
                    raise OracleError(
                        "MD attempt identity/hash conflict for registered band/image"
                    )
                records.append(record)
        by_attempt: Dict[str, List[Dict[str, object]]] = {}
        for record in records:
            attempt_id = record.get("attempt_id")
            if not isinstance(attempt_id, str) or not re.fullmatch(
                r"b\d{6}-i\d{2}-t\d+-p\d+", attempt_id
            ):
                raise OracleError("invalid attempt identifier in atomic audit record")
            by_attempt.setdefault(attempt_id, []).append(record)
        terminal_statuses = {
            "failed_before_mean_force",
            "succeeded",
            "recovered_success",
            "mean_force_returned_but_ineligible",
        }
        for attempt_id, attempt_records in by_attempt.items():
            started = [row for row in attempt_records if row.get("status") == "started"]
            terminal = [
                row for row in attempt_records if row.get("status") in terminal_statuses
            ]
            if len(started) != 1 or len(terminal) > 1:
                raise OracleError(f"invalid audit state sequence for attempt {attempt_id}")
            if terminal:
                expected_nof = (
                    0 if terminal[0]["status"] == "failed_before_mean_force" else 1
                )
                if terminal[0].get("nof_returned") != expected_nof:
                    raise OracleError(f"invalid NoF result for attempt {attempt_id}")
        return records

    def _assert_new_attempt_allowed(self, expected: Dict[str, object]) -> None:
        records = self._records_for_logical_evaluation(expected)
        attempts = {
            str(record.get("attempt_id"))
            for record in records
            if record.get("status") == "started"
        }
        if any(
            record.get("status")
            in {
                "succeeded",
                "recovered_success",
                "mean_force_returned_but_ineligible",
            }
            for record in records
        ):
            raise OracleError(
                "terminal mean-force record exists but its result artifact is missing"
            )
        if len(attempts) >= self.maximum_attempts_per_logical_evaluation:
            raise OracleError("maximum same-request MD attempt count exhausted")

    def evaluate_band(
        self,
        full_path: Array,
        band_id: int,
        band_directory: Optional[Path] = None,
    ) -> OracleBand:
        full_path = np.asarray(full_path, dtype=np.float64)
        centers = full_path[1:-1]
        if centers.shape != (20, 2):
            raise ValueError(f"expected 20 internal images, got {centers.shape}")
        if band_directory is None:
            band_directory = self.run_directory / "bands" / f"band_{int(band_id):06d}"
        band_directory = Path(band_directory)
        band_directory.mkdir(parents=True, exist_ok=True)
        scratch_root = self.run_directory / "scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)

        image_results: List[Dict[str, Array]] = []
        for image_index, center in enumerate(centers):
            image_path = band_directory / f"image_{image_index:02d}.npz"
            seed = _deterministic_seed(
                self.master_seed, int(band_id), image_index, "velocity"
            )
            thermostat_seed = _deterministic_seed(
                self.master_seed, int(band_id), image_index, "thermostat"
            )
            logical_identity = self._attempt_identity(
                attempt_id="logical-evaluation",
                band_id=int(band_id),
                image_index=int(image_index),
                center=center,
                velocity_seed=seed,
                thermostat_seed=thermostat_seed,
            )
            if not image_path.is_file():
                self._assert_new_attempt_allowed(logical_identity)
                attempt_id = (
                    f"b{int(band_id):06d}-i{int(image_index):02d}-"
                    f"t{time.time_ns()}-p{os.getpid()}"
                )
                attempt = {
                    **self._attempt_identity(
                        attempt_id=attempt_id,
                        band_id=int(band_id),
                        image_index=int(image_index),
                        center=center,
                        velocity_seed=seed,
                        thermostat_seed=thermostat_seed,
                    ),
                    "started_unix": time.time(),
                    "status": "started",
                }
                self._append_attempt(attempt)
                attempt_perf_counter = time.perf_counter()
                try:
                    result = self._evaluate_image(
                        center, seed, thermostat_seed, scratch_root
                    )
                except BaseException as error:
                    attempt.update(
                        {
                            "finished_unix": time.time(),
                            "status": "failed_before_mean_force",
                            "error_type": type(error).__name__,
                            "error": str(error)[-8000:],
                            "wall_seconds": float(
                                time.perf_counter() - attempt_perf_counter
                            ),
                            "nof_returned": 0,
                        }
                    )
                    self._append_attempt(attempt)
                    raise
                _atomic_save_npz(
                    image_path,
                    center=np.asarray(center),
                    seed=np.asarray(seed, dtype=np.int64),
                    thermostat_seed=np.asarray(thermostat_seed, dtype=np.int64),
                    attempt_id=np.asarray(attempt_id),
                    parameter_id=np.asarray(attempt["parameter_id"]),
                    registration_sha256=np.asarray(attempt["registration_sha256"]),
                    registration_binding_sha256=np.asarray(
                        attempt["registration_binding_sha256"]
                    ),
                    gmx_sha256=np.asarray(attempt["gmx_sha256"]),
                    plumed_kernel_sha256=np.asarray(attempt["plumed_kernel_sha256"]),
                    input_sha256_json=np.asarray(
                        json.dumps(attempt["input_sha256"], sort_keys=True)
                    ),
                    **result,
                )
                metric_valid = bool(np.asarray(result["metric_valid"]).item())
                processed_topology_sha256 = str(
                    result["processed_topology_sha256"].item()
                )
                topology_valid = True
                topology_error = None
                try:
                    self._bind_campaign_processed_topology(
                        processed_topology_sha256,
                        str(attempt["registration_sha256"]),
                    )
                except Exception as error:
                    topology_valid = False
                    topology_error = f"{type(error).__name__}: {error}"[-8000:]
                eligibility_valid = metric_valid and topology_valid
                attempt.update(
                    {
                        "finished_unix": time.time(),
                        "status": (
                            "succeeded"
                            if eligibility_valid
                            else "mean_force_returned_but_ineligible"
                        ),
                        "wall_seconds": float(result["wall_seconds"]),
                        "result_npz_sha256": _sha256_file(image_path),
                        "processed_topology_sha256": processed_topology_sha256,
                        "restrained_tpr_sha256": str(
                            result["restrained_tpr_sha256"].item()
                        ),
                        "nof_returned": 1,
                    }
                )
                if topology_error is not None:
                    attempt["eligibility_error"] = topology_error
                self._append_attempt(attempt)
            with np.load(image_path, allow_pickle=False) as loaded:
                stored_center = np.asarray(loaded["center"], dtype=np.float64)
                stored_seed = int(np.asarray(loaded["seed"]).item())
                stored_thermostat_seed = int(
                    np.asarray(loaded["thermostat_seed"]).item()
                )
                stored_attempt_id = str(np.asarray(loaded["attempt_id"]).item())
                expected_seeded_mdp_hash = hashlib.sha256(
                    self._seeded_mdp(seed, thermostat_seed).encode("utf-8")
                ).hexdigest()
                stored_seeded_mdp_hash = str(
                    np.asarray(loaded["seeded_mdp_sha256"]).item()
                )
                stored_input_hashes = json.loads(
                    str(np.asarray(loaded["input_sha256_json"]).item())
                )
                complete_mean_force = bool(
                    np.asarray(loaded["mean_force_vector_complete"]).item()
                )
                stored_gradient = np.asarray(loaded["grad_F"], dtype=np.float64)
                if (
                    not np.array_equal(stored_center, center)
                    or stored_seed != seed
                    or stored_thermostat_seed != thermostat_seed
                    or stored_seeded_mdp_hash != expected_seeded_mdp_hash
                    or str(np.asarray(loaded["parameter_id"]).item())
                    != logical_identity["parameter_id"]
                    or str(np.asarray(loaded["registration_sha256"]).item())
                    != logical_identity["registration_sha256"]
                    or str(
                        np.asarray(loaded["registration_binding_sha256"]).item()
                    )
                    != logical_identity["registration_binding_sha256"]
                    or str(np.asarray(loaded["gmx_sha256"]).item())
                    != logical_identity["gmx_sha256"]
                    or str(np.asarray(loaded["plumed_kernel_sha256"]).item())
                    != logical_identity["plumed_kernel_sha256"]
                    or stored_input_hashes != logical_identity["input_sha256"]
                    or not complete_mean_force
                    or stored_gradient.shape != (2,)
                    or np.any(~np.isfinite(stored_gradient))
                ):
                    raise OracleError(
                        f"partial-band identity mismatch for {image_path}"
                    )
                loaded_result = {
                    key: np.array(loaded[key], copy=True) for key in loaded.files
                }

            records = self._records_for_logical_evaluation(logical_identity)
            terminal_records = [
                record
                for record in records
                if record.get("status")
                in {
                    "failed_before_mean_force",
                    "succeeded",
                    "recovered_success",
                    "mean_force_returned_but_ineligible",
                }
                and record.get("attempt_id") == stored_attempt_id
            ]
            successful_records = [
                record
                for record in records
                if record.get("status")
                in {
                    "succeeded",
                    "recovered_success",
                    "mean_force_returned_but_ineligible",
                }
            ]
            if len(successful_records) > 1 or len(terminal_records) > 1:
                raise OracleError("multiple terminal mean-force records for one request")
            result_hash = _sha256_file(image_path)
            metric_valid = bool(np.asarray(loaded_result["metric_valid"]).item())
            topology_valid = True
            topology_error = None
            try:
                self._bind_campaign_processed_topology(
                    str(loaded_result["processed_topology_sha256"].item()),
                    str(logical_identity["registration_sha256"]),
                )
            except Exception as error:
                topology_valid = False
                topology_error = f"{type(error).__name__}: {error}"[-8000:]
            eligibility_valid = metric_valid and topology_valid
            if successful_records:
                terminal = successful_records[0]
                if terminal.get("attempt_id") != stored_attempt_id:
                    raise OracleError("result artifact belongs to a different successful attempt")
                if terminal.get("result_npz_sha256") != result_hash:
                    raise OracleError("successful result artifact hash mismatch")
                expected_status = (
                    "succeeded"
                    if eligibility_valid
                    else "mean_force_returned_but_ineligible"
                )
                if terminal.get("status") == "recovered_success":
                    expected_status = (
                        "recovered_success" if eligibility_valid else "invalid"
                    )
                if terminal.get("status") != expected_status:
                    raise OracleError("mean-force artifact eligibility/status mismatch")
            elif terminal_records:
                raise OracleError("failed-before-mean-force record conflicts with result artifact")
            else:
                started = [
                    record
                    for record in records
                    if record.get("status") == "started"
                    and record.get("attempt_id") == stored_attempt_id
                ]
                if len(started) != 1:
                    raise OracleError("orphan result artifact lacks one exact started record")
                recovered = {
                    **self._attempt_identity(
                        attempt_id=stored_attempt_id,
                        band_id=int(band_id),
                        image_index=int(image_index),
                        center=center,
                        velocity_seed=seed,
                        thermostat_seed=thermostat_seed,
                    ),
                    "started_unix": started[0]["started_unix"],
                    "finished_unix": time.time(),
                    "status": (
                        "recovered_success"
                        if eligibility_valid
                        else "mean_force_returned_but_ineligible"
                    ),
                    "wall_seconds": float(loaded_result["wall_seconds"]),
                    "result_npz_sha256": result_hash,
                    "processed_topology_sha256": str(
                        loaded_result["processed_topology_sha256"].item()
                    ),
                    "restrained_tpr_sha256": str(
                        loaded_result["restrained_tpr_sha256"].item()
                    ),
                    "nof_returned": 1,
                    "reconciled_from_atomic_result": True,
                }
                if topology_error is not None:
                    recovered["eligibility_error"] = topology_error
                self._append_attempt(recovered)
            if not eligibility_valid:
                failed_gate = (
                    "sampled M"
                    if not metric_valid
                    else "campaign-global expanded topology binding"
                )
                raise OracleError(
                    "mean-force vector returned and costs 1 NoF, but "
                    f"{failed_gate} is ineligible"
                )
            image_results.append(loaded_result)

        gradient_blocks = np.stack(
            [result["grad_F_block_means"] for result in image_results], axis=0
        )
        metric_blocks = np.stack(
            [result["M_block_means"] for result in image_results], axis=0
        )
        processed_topology_hashes = [
            str(result["processed_topology_sha256"].item())
            for result in image_results
        ]
        if len(set(processed_topology_hashes)) != 1:
            raise OracleError("processed topology changed within one complete band")
        binding = {
            "oracle_id": self.oracle_id,
            "processed_topology_sha256": processed_topology_hashes[0],
            "meaning": (
                "fully expanded topology generated by grompp -pp; this binds "
                "the dynamically included GROMACS force-field files"
            ),
        }
        if self.oracle_binding.exists():
            existing_binding = json.loads(
                self.oracle_binding.read_text(encoding="utf-8")
            )
            if existing_binding != binding:
                raise OracleError(
                    "processed topology changed across complete bands in one run"
                )
        else:
            _atomic_save_json(self.oracle_binding, binding)
        return OracleBand(
            centers=centers.copy(),
            mean_coordinates=np.stack(
                [result["mean_coordinate"] for result in image_results], axis=0
            ),
            grad_F=np.stack([result["grad_F"] for result in image_results], axis=0),
            M=np.stack([result["M"] for result in image_results], axis=0),
            grad_F_block_means=gradient_blocks,
            M_block_means=metric_blocks,
            seeds=np.asarray([int(result["seed"]) for result in image_results], dtype=np.int64),
            thermostat_seeds=np.asarray(
                [int(result["thermostat_seed"]) for result in image_results],
                dtype=np.int64,
            ),
            frame_counts=np.asarray(
                [int(result["frame_count"]) for result in image_results], dtype=np.int64
            ),
            discarded_frame_counts=np.asarray(
                [int(result["discarded_frame_count"]) for result in image_results],
                dtype=np.int64,
            ),
            wall_seconds=np.asarray(
                [float(result["wall_seconds"]) for result in image_results],
                dtype=np.float64,
            ),
            nof=int(centers.shape[0]),
            metadata={
                "oracle_id": self.oracle_id,
                "formal_evidence": True,
                "band_id": int(band_id),
                "master_seed": int(self.master_seed),
                "discard_fraction": self.discard_fraction,
                "block_count": self.block_count,
                "jobs": 1,
                "coordinate_cache": False,
                "processed_topology_sha256": processed_topology_hashes[0],
                "seeded_mdp_sha256_by_image": [
                    str(result["seeded_mdp_sha256"].item())
                    for result in image_results
                ],
                "restrained_tpr_sha256_by_image": [
                    str(result["restrained_tpr_sha256"].item())
                    for result in image_results
                ],
            },
        )
