#!/usr/bin/env python3
"""Deterministic ball-and-stick rendering of the registered 22-atom Ala system.

Coordinates are Angstrom, in the exact atom order of ``ala2_fixed.pdb``.
The topology is fixed explicitly below; no bonds are inferred from a snapshot.
All configurations receive proper rigid Kabsch alignment (no reflection),
one common orthographic camera, and one common scale. Spheres and finite rods
are intersected analytically. Surface normals drive diffuse/specular light;
cosine-weighted hemisphere rays against the atom spheres provide soft ambient
occlusion. Bond cylinders are not included among the ambient occluders.

This is a coordinate renderer, not molecular simulation or image generation.
The unmodified input coordinates and each rendering transform are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[5]
PDB = REPO / "python/alanine_ivm_clean_four_level_20260801/inputs/ala2_fixed.pdb"
# Native PDB zero-based atom indices, 22 atoms and 21 covalent connections.
BONDS = [(1, 0), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6),
         (6, 7), (6, 8), (8, 9), (8, 10), (10, 11), (10, 12),
         (10, 13), (8, 14), (14, 15), (14, 16), (16, 17),
         (16, 18), (18, 19), (18, 20), (18, 21)]
BACKBONE = np.array([4, 6, 8, 14, 16])
COLORS = {"C": [0.265, 0.295, 0.330], "N": [0.165, 0.365, 0.760],
          "O": [0.870, 0.160, 0.180], "H": [0.860, 0.875, 0.900]}
RADII = {"C": 0.370, "N": 0.360, "O": 0.355, "H": 0.220}
ROD_RADIUS = 0.118


def read_topology():
    atoms = [line for line in PDB.read_text().splitlines()
             if line.startswith(("ATOM  ", "HETATM"))]
    symbols = [line[76:78].strip() for line in atoms]
    if len(atoms) != 22 or symbols != ["H", "C", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]:
        raise ValueError("Native Ala PDB atom order changed")
    return symbols, [line[12:16].strip() for line in atoms]


def _normalize(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def _align(coordinates, reference):
    ref_center = reference[BACKBONE].mean(axis=0)
    target = reference[BACKBONE] - ref_center
    aligned, transforms = [], []
    for xyz in coordinates:
        center = xyz[BACKBONE].mean(axis=0)
        left, _, right = np.linalg.svd((xyz[BACKBONE] - center).T @ target)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right)
        rotation = left @ correction @ right
        aligned.append((xyz - center) @ rotation)
        transforms.append({"input_center_angstrom": center.tolist(),
                           "proper_rotation": rotation.tolist(),
                           "determinant": float(np.linalg.det(rotation))})
    return np.asarray(aligned), transforms


def _camera(aligned, reference):
    """Use a common view with ACE left, NME right, and exposed backbone bonds."""
    axis_x = _normalize(reference[18] - reference[1])
    sidechain = reference[8] - reference[10]
    axis_y = _normalize(sidechain - np.dot(sidechain, axis_x) * axis_x)
    axis_z = _normalize(np.cross(axis_x, axis_y))
    initial = np.column_stack([axis_x, axis_y, axis_z])
    heavy = np.array([1, 4, 5, 6, 8, 10, 14, 15, 16, 18])
    # Deterministic, jointly optimized azimuth about the long molecular axis.
    # The objective only chooses the camera; it does not alter coordinates.
    candidates = []
    for degrees in np.linspace(-55, 55, 23):
        angle = np.deg2rad(degrees)
        roll_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
        rotation = initial @ roll_x
        projected = aligned @ rotation
        score = 0.0
        for xyz in projected:
            for a, b in [(6, 8), (8, 14)]:
                score += 8.0 * np.linalg.norm(xyz[a, :2] - xyz[b, :2])
            for i, a in enumerate(heavy):
                for b in heavy[i + 1:]:
                    if (a, b) in BONDS or (b, a) in BONDS:
                        continue
                    distance = np.linalg.norm(xyz[a, :2] - xyz[b, :2])
                    score -= 4.0 * max(0.0, 0.85 - distance) ** 2
        candidates.append((score, degrees, rotation))
    _, degrees, rotation = max(candidates, key=lambda x: x[0])
    return rotation, degrees


def _surface(xyz, symbols, width, height, bounds):
    xmin, xmax, ymin, ymax = bounds
    xs = xmin + (np.arange(width) + 0.5) / width * (xmax - xmin)
    ys = ymax - (np.arange(height) + 0.5) / height * (ymax - ymin)
    xx, yy = np.meshgrid(xs, ys)
    depth = np.full((height, width), -np.inf, dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    color = np.zeros_like(normal)
    zfront = float(xyz[:, 2].max() + 3.0)

    def region(a, b, radius):
        x0 = max(0, int(np.floor((min(a[0], b[0]) - radius - xmin) / (xmax - xmin) * width)))
        x1 = min(width, int(np.ceil((max(a[0], b[0]) + radius - xmin) / (xmax - xmin) * width)))
        y0 = max(0, int(np.floor((ymax - max(a[1], b[1]) - radius) / (ymax - ymin) * height)))
        y1 = min(height, int(np.ceil((ymax - min(a[1], b[1]) + radius) / (ymax - ymin) * height)))
        return np.s_[y0:y1, x0:x1]

    for a_index, b_index in BONDS:
        a, b = xyz[a_index], xyz[b_index]
        radius = ROD_RADIUS * (0.87 if "H" in (symbols[a_index], symbols[b_index]) else 1.0)
        rgn = region(a, b, radius)
        u = _normalize(b - a)
        length = float(np.linalg.norm(b - a))
        qx, qy, qz = xx[rgn] - a[0], yy[rgn] - a[1], zfront - a[2]
        dot = qx * u[0] + qy * u[1] + qz * u[2]
        aa = 1.0 - u[2] ** 2
        if aa < 1e-10:
            continue
        bb = 2.0 * (-qz + dot * u[2])
        cc = qx ** 2 + qy ** 2 + qz ** 2 - dot ** 2 - radius ** 2
        disc = bb ** 2 - 4.0 * aa * cc
        t = (-bb - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * aa)
        h = dot - t * u[2]
        hit_z = zfront - t
        valid = (disc >= 0) & (h >= 0) & (h <= length) & (hit_z > depth[rgn])
        n = np.stack([qx - h * u[0], qy - h * u[1], qz - t - h * u[2]], axis=-1) / radius
        base = np.where((h / length < 0.5)[..., None], COLORS[symbols[a_index]], COLORS[symbols[b_index]])
        depth[rgn][valid] = hit_z[valid]
        normal[rgn][valid] = n[valid]
        color[rgn][valid] = base[valid]

    for atom, symbol in zip(xyz, symbols):
        radius = RADII[symbol]
        rgn = region(atom, atom, radius)
        dx, dy = xx[rgn] - atom[0], yy[rgn] - atom[1]
        radicand = radius ** 2 - dx ** 2 - dy ** 2
        dz = np.sqrt(np.maximum(radicand, 0.0))
        hit_z = atom[2] + dz
        valid = (radicand >= 0) & (hit_z > depth[rgn])
        n = np.stack([dx, dy, dz], axis=-1) / radius
        depth[rgn][valid] = hit_z[valid]
        normal[rgn][valid] = n[valid]
        color[rgn][valid] = COLORS[symbol]
    return xx, yy, depth, normal, color


def _ambient_visibility(points, normals, xyz, symbols, samples):
    """Cosine-weighted hemispherical AO, using exact ray/sphere intersection."""
    output = np.empty(len(points), dtype=np.float32)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    numbers = np.arange(samples)
    azimuth = numbers * golden
    height = np.sqrt(1.0 - (numbers + 0.5) / samples)
    radius = np.sqrt((numbers + 0.5) / samples)
    local = np.column_stack([radius * np.cos(azimuth), radius * np.sin(azimuth), height])
    atom_radii = np.array([RADII[s] for s in symbols])
    for first in range(0, len(points), 4096):
        p, n = points[first:first + 4096], normals[first:first + 4096]
        tangent = np.cross(np.where((np.abs(n[:, 2]) < 0.9)[:, None], [0., 0., 1.], [0., 1., 0.]), n)
        tangent = _normalize(tangent)
        bitangent = np.cross(n, tangent)
        directions = (tangent[:, None, :] * local[None, :, 0, None]
                      + bitangent[:, None, :] * local[None, :, 1, None]
                      + n[:, None, :] * local[None, :, 2, None])
        origin = p + 0.008 * n
        blocked = np.zeros((len(p), samples), dtype=bool)
        for atom, atom_radius in zip(xyz, atom_radii):
            vec = atom - origin
            projection = np.einsum("ijk,ik->ij", directions, vec)
            discriminant = projection ** 2 - np.einsum("ij,ij->i", vec, vec)[:, None] + atom_radius ** 2
            distance = projection - np.sqrt(np.maximum(discriminant, 0.0))
            blocked |= (discriminant > 0) & (distance > 0.001) & (distance < 3.2)
        output[first:first + len(p)] = 1.0 - blocked.mean(axis=1)
    return output


def _render(xyz, symbols, size, bounds, samples):
    width, height = size
    xx, yy, depth, normals, colors = _surface(xyz, symbols, width, height, bounds)
    mask = np.isfinite(depth)
    positions = np.column_stack([xx[mask], yy[mask], depth[mask]])
    n = normals[mask]
    visibility = _ambient_visibility(positions, n, xyz, symbols, samples)
    key = _normalize(np.array([-0.42, 0.65, 0.78]))
    fill = _normalize(np.array([0.55, 0.15, 0.82]))
    half = _normalize(key + np.array([0.0, 0.0, 1.0]))
    diffuse = 0.48 * np.maximum(n @ key, 0.0) + 0.13 * np.maximum(n @ fill, 0.0)
    ambient = 0.45 * (0.43 + 0.57 * visibility)
    specular = 0.23 * np.maximum(n @ half, 0.0) ** 40.0 * visibility
    pixels = colors[mask] * (ambient + diffuse)[:, None] + specular[:, None]
    # Subtle grazing-angle lift on glossy atom surfaces, without outlines.
    pixels += (0.018 * (1.0 - np.maximum(n[:, 2], 0.0)) ** 3)[:, None]
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    rgb[mask] = np.rint(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def render_configurations(coordinates, output_dir, *, names=None,
                          image_size=1600, reference=None, quality="final"):
    """Render n×22×3 Å snapshots, returning images and lower-left xy metadata."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (22, 3) or not np.isfinite(coordinates).all():
        raise ValueError("Expected finite coordinates with shape (n,22,3), Angstrom")
    symbols, atom_names = read_topology()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = names or [f"configuration_{i:02d}" for i in range(len(coordinates))]
    if len(names) != len(coordinates) or len(set(names)) != len(names):
        raise ValueError("One unique output name is required for every snapshot")
    reference = coordinates[0] if reference is None else np.asarray(reference, dtype=float)
    aligned, transforms = _align(coordinates, reference)
    camera, camera_angle = _camera(aligned, reference)
    projected = aligned @ camera
    # Shared scale and bounds; fixed-size atoms remain comparable across frames.
    extent = projected[:, :, :2].reshape(-1, 2)
    low, high = extent.min(axis=0) - 0.60, extent.max(axis=0) + 0.60
    midpoint = (low + high) / 2
    width, height = int(image_size), round(int(image_size) * 0.75)
    span_x = max(high[0] - low[0], (high[1] - low[1]) * width / height)
    span_y = span_x * height / width
    bounds = [midpoint[0] - span_x / 2, midpoint[0] + span_x / 2,
              midpoint[1] - span_y / 2, midpoint[1] + span_y / 2]
    result = {"renderer": "analytic orthographic spheres and finite cylinders",
              "renderer_source": str(Path(__file__).resolve()),
              "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "ambient_occlusion": "cosine-weighted hemisphere rays, atom-sphere occluders, 3.2 Angstrom range",
              "ambient_occlusion_samples": 32 if quality == "final" else 12,
              "coordinate_units": "Angstrom", "topology_file": str(PDB),
              "topology_sha256": hashlib.sha256(PDB.read_bytes()).hexdigest(),
              "bonds_zero_based": BONDS, "bond_order_display": "single covalent rods",
              "atom_names": atom_names, "symbols": symbols,
              "atom_radii_angstrom": RADII, "rod_radius_angstrom": ROD_RADIUS,
              "element_colors_rgb": COLORS, "alignment_atom_indices": BACKBONE.tolist(),
              "camera_rotation": camera.tolist(), "camera_roll_about_long_axis_degrees": camera_angle,
              "camera_bounds_angstrom": bounds, "pixel_size": [width, height],
              "input_coordinates_sha256": hashlib.sha256(np.ascontiguousarray(coordinates).tobytes()).hexdigest(),
              "images": []}
    for index, (xyz, name) in enumerate(zip(projected, names)):
        path = output_dir / f"{name}.png"
        picture = _render(xyz, symbols, (width, height), bounds, result["ambient_occlusion_samples"])
        picture.save(path, dpi=(600, 600))
        normalized = np.column_stack([(xyz[:, 0] - bounds[0]) / span_x,
                                      (xyz[:, 1] - bounds[2]) / span_y])
        entry = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "atom_xy_normalized": normalized.tolist(),
                 "phi_bond_xy": normalized[[6, 8]].mean(axis=0).tolist(),
                 "psi_bond_xy": normalized[[8, 14]].mean(axis=0).tolist(),
                 "rigid_alignment": transforms[index],
                 "camera_coordinates_angstrom": xyz.tolist()}
        result["images"].append(entry)
    (output_dir / "render_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPY array n×22×3 in Angstrom")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-size", type=int, default=1600)
    parser.add_argument("--quality", choices=["preview", "final"], default="final")
    args = parser.parse_args()
    metadata = render_configurations(np.load(args.input, allow_pickle=False), args.output_dir,
                                     image_size=args.image_size, quality=args.quality)
    print(json.dumps({"images": [entry["path"] for entry in metadata["images"]],
                      "pixel_size": metadata["pixel_size"]}, indent=2))
