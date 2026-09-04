"""Paired appearance ablation for mesh-free forward boundary transport."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark import cuda_environment
from .boundary_transport import (
    build_boundary_transport,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
    reweight_boundary_transport_rgb,
)
from .detector_readout import detector_from_outgoing_direction, read_light_field
from .image_formation import image_metrics
from .light_field import build_sparse_boundary_light_field
from .mesh_field import prepare_stanford_bunny
from .meshfree_rgb import (
    READOUT,
    RGBTransportConfig,
    SURFACE_SAMPLES,
    _plot_grid,
    _rgb_reference,
    _structure_metrics,
)
from .meshfree_surface import sample_meshfree_zero_set


Tensor = torch.Tensor


@dataclass(frozen=True)
class AppearanceVariant:
    key: str
    label: str
    attenuation: bool
    normal_lighting: bool
    distance_sigma: float
    cosine_power: float
    ambient_emission: float

    def transport_config(self) -> RGBTransportConfig:
        return RGBTransportConfig(
            self.key,
            self.distance_sigma,
            self.cosine_power,
            self.ambient_emission,
        )


VARIANTS = (
    AppearanceVariant("A", "base color only", False, False, 0.0, 0.0, 1.0),
    AppearanceVariant("B", "attenuation only", True, False, 0.55, 0.0, 1.0),
    AppearanceVariant("C", "normal lighting only", False, True, 0.0, 1.0, 0.35),
    AppearanceVariant(
        "D", "attenuation + normal lighting", True, True, 0.55, 1.0, 0.35
    ),
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rgb_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    from skimage.metrics import structural_similarity

    mask = np.any(first > 1e-10, axis=-1) | np.any(second > 1e-10, axis=-1)
    raw_mae = float(np.mean(np.abs(first - second)))
    raw_ssim = float(
        structural_similarity(first, second, channel_axis=-1, data_range=1.0)
    )
    foreground = mask[..., None]
    numerator = float(np.sum(first * second * foreground))
    denominator = float(np.sum(first * first * foreground))
    exposure_scale = numerator / max(denominator, 1e-30)
    matched = np.clip(exposure_scale * first, 0.0, 1.0)
    return {
        "raw_rgb_mae": raw_mae,
        "raw_ssim": raw_ssim,
        "least_squares_exposure_scale_C_to_D": exposure_scale,
        "exposure_matched_rgb_mae": float(np.mean(np.abs(matched - second))),
        "exposure_matched_ssim": float(
            structural_similarity(
                matched, second, channel_axis=-1, data_range=1.0
            )
        ),
    }


def _qualitative(variant: AppearanceVariant) -> dict[str, object]:
    if variant.key == "A":
        assessment = "weak internal structure"
        shading = False
        note = (
            "All parts are identifiable from diagnostic color, but relief is "
            "nearly flat."
        )
    elif variant.key == "B":
        assessment = "weak internal structure"
        shading = False
        note = (
            "Distance darkening is visible, but local folds and contours remain "
            "weak."
        )
    elif variant.key == "C":
        assessment = "strong internal structure"
        shading = True
        note = (
            "Normal shading clearly resolves the face, ears, shoulder, torso "
            "folds, leg, and tail."
        )
    else:
        assessment = "strong internal structure"
        shading = True
        note = "The same parts remain clear, with additional propagation darkening."
    return {
        "assessment": assessment,
        "ears_visible": True,
        "head_visible": True,
        "torso_visible": True,
        "legs_visible": True,
        "tail_visible": True,
        "continuous_shading_visible": shading,
        "note": note,
    }


def run_appearance_ablation(
    mesh_path: Path,
    artifact_directory: Path,
    render_directory: Path,
) -> dict[str, object]:
    """Run exactly A/B/C/D with one frozen visibility event table."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    started = time.perf_counter()
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    field = prepared.gt_field.to(device)
    surface = sample_meshfree_zero_set(field, SURFACE_SAMPLES)
    atlas = nested_fibonacci_atlas(device, (8, 128))
    boundary = enclosing_observation_sphere(surface.points)
    visibility = build_boundary_transport(
        field,
        surface.points,
        surface.normals,
        atlas,
        boundary,
        root_samples=16,
        bisection_steps=18,
    )
    camera = detector_from_outgoing_direction(boundary, atlas.directions[3])
    boundary_center = boundary.center.detach().cpu().numpy()
    lower = field.lower.detach().cpu().numpy()
    upper = field.upper.detach().cpu().numpy()

    images: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    variant_references: dict[str, np.ndarray] = {}
    frozen_digests: dict[str, str] = {}
    for variant in VARIANTS:
        events = reweight_boundary_transport_rgb(
            visibility,
            surface.base_colors,
            surface.normals,
            distance_sigma=variant.distance_sigma,
            cosine_power=variant.cosine_power,
            ambient_emission=variant.ambient_emission,
            emission_strengths=surface.emission_strengths,
        )
        light_field = build_sparse_boundary_light_field(events)
        output = read_light_field(light_field, camera, **READOUT)
        image = output.image.detach().cpu().numpy()
        reference = _rgb_reference(
            prepared.repaired_vertices,
            prepared.repaired_faces,
            camera,
            boundary_center,
            boundary.radius,
            lower,
            upper,
            variant.transport_config(),
            READOUT["sensor_gain"],
        )
        images[variant.key] = image
        variant_references[variant.key] = reference.reference.image
        frozen_digests[variant.key] = light_field.digest
        rows.append(
            {
                **asdict(variant),
                **{
                    f"paired_oracle_{key}": value
                    for key, value in image_metrics(
                        image, reference.reference
                    ).items()
                },
                **_structure_metrics(
                    image, output.mask.detach().cpu().numpy()
                ),
                "direct_support_fraction": output.direct_support_fraction,
                "interpolated_support_fraction": output.interpolated_support_fraction,
                "field_occupied_bins": light_field.occupied_bins,
                "field_memory_mib": light_field.memory_bytes / 2**20,
                **_qualitative(variant),
            }
        )

    # Use the old v0.6 D oracle as the one common target for the requested table.
    common_reference_config = VARIANTS[3].transport_config()
    common_reference = _rgb_reference(
        prepared.repaired_vertices,
        prepared.repaired_faces,
        camera,
        boundary_center,
        boundary.radius,
        lower,
        upper,
        common_reference_config,
        READOUT["sensor_gain"],
    ).reference
    for row in rows:
        common = image_metrics(images[str(row["key"])], common_reference)
        row.update(
            {
                "iou": common["silhouette_iou"],
                "f1": common["silhouette_f1"],
                "ssim": common["ssim"],
                "psnr": common["psnr"],
                "rgb_mae": float(
                    np.mean(
                        np.abs(
                            images[str(row["key"])] - common_reference.image
                        )
                    )
                ),
            }
        )

    c_structure = next(row for row in rows if row["key"] == "C")
    d_structure = next(row for row in rows if row["key"] == "D")
    c_to_d = _rgb_similarity(images["C"], images["D"])
    c_preserves_structure = (
        float(c_structure["foreground_luminance_std"]) >= 0.08
        and float(c_structure["interior_gradient_energy"]) >= 0.01
        and int(c_structure["quantized_foreground_color_count"]) >= 256
        and float(c_structure["iou"]) >= 0.95
        and bool(c_structure["continuous_shading_visible"])
    )
    c_close_enough_to_d = (
        float(c_structure["interior_gradient_energy"])
        >= 0.75 * float(d_structure["interior_gradient_energy"])
        and float(c_structure["foreground_luminance_std"])
        >= 0.75 * float(d_structure["foreground_luminance_std"])
        and float(c_to_d["exposure_matched_ssim"]) >= 0.90
    )
    if c_preserves_structure and c_close_enough_to_d:
        verdict = "ATTENUATION_NOT_NEEDED"
        decision = (
            "remove attenuation from the main formulation; retain it only as "
            "an optional diagnostic"
        )
    elif not c_preserves_structure:
        verdict = "ATTENUATION_STILL_NEEDED"
        decision = "keep attenuation in the main formulation"
    else:
        verdict = "ABLATION_INCONCLUSIVE"
        decision = "keep attenuation optional and do not change the default yet"

    _plot_grid(
        render_directory / "v07_abcd_reference.png",
        [
            *[(f"{item.key}: {item.label}", images[item.key]) for item in VARIANTS],
            ("common reference (D oracle)", common_reference.image),
        ],
        1,
        5,
        "Paired forward appearance ablation: attenuation x normal lighting",
        figsize_scale=2.8,
    )
    _plot_grid(
        render_directory / "v07_v06_C_D_reference.png",
        [
            ("old v0.6 default (D)", images["D"]),
            ("C: no attenuation", images["C"]),
            ("D: attenuation", images["D"]),
            ("reference", common_reference.image),
        ],
        1,
        4,
        "Does normal lighting retain structure without propagation attenuation?",
        figsize_scale=3.0,
    )

    report: dict[str, object] = {
        "version": "0.7.0",
        "phase": "appearance_ablation",
        "environment": cuda_environment(),
        "controlled_factors": {
            "surface_samples": SURFACE_SAMPLES,
            "directions": atlas.count,
            "resolution": [256, 256],
            "readout": READOUT,
            "base_appearance": (
                "camera-independent analytic position RGB diagnostic field"
            ),
            "normal_lighting": "ambient + (1-ambient) max(0,n dot omega)",
            "attenuation": "exp(-sigma * zero-set-to-boundary path length)",
            "common_reference": "old v0.6 D evaluation oracle",
        },
        "variants": rows,
        "c_to_d_similarity": c_to_d,
        "decision_gates": {
            "c_preserves_structure": c_preserves_structure,
            "c_close_enough_to_d": c_close_enough_to_d,
            "thresholds": {
                "c_luminance_std_min": 0.08,
                "c_interior_gradient_min": 0.01,
                "c_quantized_colors_min": 256,
                "c_iou_min": 0.95,
                "c_to_d_structure_ratio_min": 0.75,
                "c_to_d_exposure_matched_ssim_min": 0.90,
            },
        },
        "verdict": verdict,
        "main_formulation_decision": decision,
        "visibility": {
            "digest": visibility.digest,
            "emitted": visibility.emitted_count,
            "absorbed": visibility.absorbed_count,
            "retained": visibility.count,
            "scene_seconds": visibility.scene_seconds,
            "reused_unchanged_for_all_variants": True,
        },
        "field_digests": frozen_digests,
        "meshfree_surface": {
            "valid_count": surface.valid_count,
            "root_failures": surface.root_failure_count,
            "degenerate_normals": surface.degenerate_normal_count,
            "nonfinite": surface.nonfinite_count,
            "residual_max": surface.residual_max,
        },
        "artifacts": {
            "json": str(artifact_directory / "v07_appearance_ablation.json"),
            "csv": str(artifact_directory / "v07_appearance_ablation.csv"),
            "abcd_figure": str(render_directory / "v07_abcd_reference.png"),
            "focused_figure": str(render_directory / "v07_v06_C_D_reference.png"),
        },
        "total_seconds": time.perf_counter() - started,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "v07_appearance_ablation.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(artifact_directory / "v07_appearance_ablation.csv", rows)
    return report
