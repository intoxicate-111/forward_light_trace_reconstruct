#!/usr/bin/env python3
"""Command-line demo and deterministic production-code verification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Keep the requested `python demo.py` workflow usable before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from zlt import (  # noqa: E402
    PhotonBatch,
    PlanarCamera,
    SphereField,
    TorusField,
    TraceResult,
    make_scene,
    render_first_arrival,
    trace_photons,
    unit_normals,
)


def camera_for_scene(scene: str, resolution: int) -> PlanarCamera:
    if scene == "sphere":
        center = torch.tensor([0.0, 0.0, 3.0])
        normal = torch.tensor([0.0, 0.0, -1.0])
        right = torch.tensor([1.0, 0.0, 0.0])
        up = torch.tensor([0.0, 1.0, 0.0])
        extent = 4.0
    else:
        # A side detector makes the torus hole an explicit self-occlusion test:
        # inward-facing emitters can cross the hole and hit the opposite tube.
        center = torch.tensor([3.0, 0.0, 0.0])
        normal = torch.tensor([-1.0, 0.0, 0.0])
        right = torch.tensor([0.0, 1.0, 0.0])
        up = torch.tensor([0.0, 0.0, 1.0])
        extent = 4.5
    return PlanarCamera(
        center=center,
        normal=normal,
        right=right,
        up=up,
        width=extent,
        height=extent,
        resolution=(resolution, resolution),
    )


def run_demo(args: argparse.Namespace) -> tuple[dict[str, object], torch.Tensor]:
    field = SphereField() if args.scene == "sphere" else TorusField()
    emitter_count = args.emitters or (1024 if args.scene == "sphere" else 2048)
    camera = camera_for_scene(args.scene, args.resolution)
    started = time.perf_counter()
    points, normals, photons, trace, render = make_scene(
        field,
        camera,
        emitter_count=emitter_count,
        packets_per_emitter=args.packets,
        cone_power=args.cone_power,
        emission_interval=args.emission_interval,
        seed=args.seed,
        root_samples=args.root_samples,
        mode=args.mode,
        beta=args.beta,
    )
    runtime = time.perf_counter() - started
    nnz = render.transport._nnz()
    pixel_count = camera.pixel_count
    active_pixels = int(torch.unique(trace.pixels).numel())
    summary: dict[str, object] = {
        "scene": args.scene,
        "mode": args.mode,
        "seed": args.seed,
        "emitters": emitter_count,
        "emitted_photons": photons.count,
        "camera_candidates": trace.camera_candidate_count,
        "camera_hits": trace.camera_hit_count,
        "camera_hit_fraction": trace.camera_hit_count / photons.count,
        "absorbed": trace.absorbed_count,
        "absorption_fraction": trace.absorbed_count / photons.count,
        "candidate_absorption_fraction": (
            trace.absorbed_count / trace.camera_candidate_count
            if trace.camera_candidate_count
            else 0.0
        ),
        "active_pixels": active_pixels,
        "max_abs_field_at_emitters": float(field.value(points).abs().max()),
        "max_normal_length_error": float(
            (torch.linalg.vector_norm(normals, dim=-1) - 1.0).abs().max()
        ),
        "min_direction_normal_dot": float(
            (photons.directions * normals[photons.emitter_ids]).sum(dim=-1).min()
        ),
        "transport_shape": list(render.transport.shape),
        "transport_nnz": nnz,
        "transport_density": nnz / (pixel_count * emitter_count),
        "mean_active_emitters_per_pixel": nnz / pixel_count,
        "mean_affected_pixels_per_emitter": nnz / emitter_count,
        "direct_sparse_max_difference": render.max_difference,
        "runtime_seconds": runtime,
    }
    if args.output:
        import matplotlib.pyplot as plt

        image = render.direct_image.reshape(*camera.resolution, 3).clamp(0.0, 1.0)
        plt.imsave(args.output, image.numpy())
    return summary, render.direct_image


def run_verification() -> dict[str, object]:
    """Exercise acceptance gates through the same classes/functions as the demo."""
    generator = torch.Generator().manual_seed(19)
    sphere, torus = SphereField(), TorusField()
    sphere_points = sphere.sample_surface(256, generator)
    torus_points = torus.sample_surface(256, generator)
    sphere_normals = unit_normals(sphere, sphere_points)
    torus_normals = unit_normals(torus, torus_points)

    sphere_zero_error = float(sphere.value(sphere_points).abs().max())
    torus_zero_error = float(torus.value(torus_points).abs().max())
    normal_error = max(
        float((torch.linalg.vector_norm(sphere_normals, dim=-1) - 1.0).abs().max()),
        float((torch.linalg.vector_norm(torus_normals, dim=-1) - 1.0).abs().max()),
    )
    sphere_reference_error = float(
        (sphere_normals - sphere_points / sphere.radius).abs().max()
    )
    assert sphere_zero_error < 1e-12 and torus_zero_error < 1e-12
    assert normal_error < 1e-12 and sphere_reference_error < 1e-12

    # Known finite-plane intersections: center, upper-right, outside, parallel.
    check_camera = PlanarCamera(
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        2.0,
        2.0,
        (2, 2),
    )
    origins = torch.zeros((4, 3), dtype=torch.float64)
    directions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.5, 0.5, 1.0], [2.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    valid, times, pixels, local = check_camera.intersect(origins, directions)
    assert valid.tolist() == [True, True, False, False]
    assert torch.allclose(times[:2], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(local[:2], torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64))
    assert pixels[:2].tolist() == [3, 1]

    # A diameter ray re-enters the sphere at t=2 and is absorbed; an outward
    # ray from the opposite point reaches the same x=3 detector.
    side_camera = PlanarCamera(
        torch.tensor([3.0, 0.0, 0.0]),
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        4.0,
        4.0,
        (4, 4),
    )
    absorption_photons = PhotonBatch(
        origins=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        directions=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        energies=torch.ones(2, dtype=torch.float64),
        emit_times=torch.zeros(2, dtype=torch.float64),
        emitter_ids=torch.tensor([0, 1]),
    )
    absorption_trace = trace_photons(sphere, side_camera, absorption_photons, root_samples=48)
    assert absorption_trace.absorbed_count == 1
    assert absorption_trace.camera_hit_count == 1
    assert absorption_trace.emitter_ids.tolist() == [1]

    # Arrival competition uses two packets in one pixel.
    competition = TraceResult(
        emitted_count=2,
        camera_candidate_count=2,
        absorbed_count=0,
        pixels=torch.tensor([0, 0]),
        arrival_times=torch.tensor([1.0, 2.0], dtype=torch.float64),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        energies=torch.ones(2, dtype=torch.float64),
        emitter_ids=torch.tensor([0, 1]),
    )
    emitter_colors = competition.colors.clone()
    hard = render_first_arrival(competition, 1, emitter_colors, mode="hard")
    soft_low = render_first_arrival(competition, 1, emitter_colors, mode="soft", beta=1.0)
    soft_high = render_first_arrival(competition, 1, emitter_colors, mode="soft", beta=20.0)
    assert torch.equal(hard.direct_image[0], emitter_colors[0])
    high_error = torch.linalg.vector_norm(soft_high.direct_image - hard.direct_image)
    low_error = torch.linalg.vector_norm(soft_low.direct_image - hard.direct_image)
    assert high_error < low_error
    assert max(hard.max_difference, soft_low.max_difference, soft_high.max_difference) < 1e-12

    # Full-pipeline determinism and sparse equivalence.
    deterministic_camera = camera_for_scene("sphere", 24)
    run_kwargs = dict(
        emitter_count=128,
        packets_per_emitter=4,
        cone_power=16.0,
        emission_interval=0.0,
        seed=23,
        root_samples=32,
        mode="hard",
        beta=50.0,
    )
    first = make_scene(sphere, deterministic_camera, **run_kwargs)
    second = make_scene(sphere, deterministic_camera, **run_kwargs)
    first_trace, first_render = first[-2], first[-1]
    second_trace, second_render = second[-2], second[-1]
    assert first_trace.emitted_count == second_trace.emitted_count
    assert first_trace.camera_candidate_count == second_trace.camera_candidate_count
    assert first_trace.absorbed_count == second_trace.absorbed_count
    assert torch.equal(first_trace.pixels, second_trace.pixels)
    assert torch.equal(first_render.direct_image, second_render.direct_image)
    assert torch.equal(first_trace.arrival_times, second_trace.arrival_times)
    assert torch.equal(first_render.transport.indices(), second_render.transport.indices())
    assert torch.equal(first_render.transport.values(), second_render.transport.values())
    assert first_render.max_difference < 1e-6

    return {
        "gate_a_zero_set": {
            "sphere_max_abs_F": sphere_zero_error,
            "torus_max_abs_F": torus_zero_error,
        },
        "gate_b_normals": {
            "max_unit_length_error": normal_error,
            "sphere_reference_error": sphere_reference_error,
        },
        "gate_c_camera": "passed",
        "gate_d_absorption": {"absorbed": 1, "reached_camera": 1},
        "gate_e_first_arrival": "passed",
        "gate_f_sparse": {"max_difference": first_render.max_difference},
        "gate_g_determinism": "passed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("sphere", "torus"), default="sphere")
    parser.add_argument("--mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--emitters", type=int, default=None)
    parser.add_argument("--packets", type=int, default=16, help="photon packets per emitter")
    parser.add_argument("--cone-power", type=float, default=16.0)
    parser.add_argument("--emission-interval", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--root-samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None, help="optional PNG path")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        print("verification:")
        print(json.dumps(run_verification(), indent=2, sort_keys=True))
    summary, _ = run_demo(args)
    print("render:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
