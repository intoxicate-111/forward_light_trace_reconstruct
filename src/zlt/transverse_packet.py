"""v0.8.11: bounded quadrature of the existing finite-ball packet operator.

Offsets remain world-space ball offsets as in finite_packet._transmission.
They are not reinterpreted as a disk perpendicular to the ray.  Count one is
the historical zero-offset control, separate from the nested ball sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .finite_packet import _compact_shell, _micro_offsets, _transmission

COUNTS = (1, 4, 8, 16, 32, 64, 128)
RADII = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class FusedGridLookup:
    """Same border-clamped trilinear values/gradient grid in one CUDA lookup.

    The existing gradient grid is interpolated, not replaced with derivatives
    of the scalar interpolant. Float64 and align_corners=True preserve the
    historical coordinates. Equivalence is checked before any experiment.
    """
    def __init__(self, field):
        self.field = field
        self.lower, self.upper = field.lower, field.upper
        self.channels = torch.cat((field.grid[..., None], field.gradient_grid), -1)
        self.channels = self.channels.permute(3, 0, 1, 2).contiguous()[None]

    def evaluate(self, points):
        xyz = 2 * (points - self.lower) / (self.upper - self.lower) - 1
        grid = xyz[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        result = functional.grid_sample(self.channels, grid, mode="bilinear",
                                       padding_mode="border", align_corners=True)
        return result[0, :, 0, 0].T

    def value(self, points):
        return self.evaluate(points)[:, 0]

    def gradient(self, points):
        return self.evaluate(points)[:, 1:]


def transmission_prefixes(field, origins, directions, maximum_times, *, radius,
                          epsilon, path_step, offsets, counts=COUNTS, eta=1e-6,
                          kappa=-math.log(.01), launch_exclusion_factor=1.05,
                          path_chunk=32, micro_chunk=8, launch_start=None):
    """Integrate each offset once; read nested prefix means at requested M.

    Live samples are bounded by [emitter_chunk,path_chunk,micro_chunk,3].
    The [emitter_chunk,M] table contains only integrated scalar tau, no path
    history or geometry-parameter axis. Source energy is never multiplied by M.
    """
    if epsilon <= 0 or path_step <= 0 or min(path_chunk, micro_chunk) <= 0:
        raise ValueError("positive quadrature widths and chunks required")
    counts = tuple(sorted(set(counts)))
    if min(counts) < 1 or max(counts) > len(offsets):
        raise ValueError("invalid micro counts")
    # The separate first entry is exactly zero, including when the ball's
    # first Sobol radius is the historical 1e-4 clamped value.
    maximum = max((m for m in counts if m > 1), default=0) if radius != 0 else 0
    selected = torch.cat((origins.new_zeros((1, 3)), offsets[:maximum]), 0)
    start = (launch_exclusion_factor * (radius + epsilon)
             if launch_start is None else launch_start)
    available = (maximum_times - start).clamp_min(0)
    steps = max(1, math.ceil(float(available.max()) / path_step))
    integrated = origins.new_zeros((len(origins), len(selected)))
    evaluations = 0
    for p in range(0, steps, path_chunk):
        t = start + (torch.arange(p, min(p + path_chunk, steps),
                                  device=origins.device, dtype=origins.dtype) + .5) * path_step
        valid = t[None] < maximum_times[:, None]
        centers = origins[:, None] + t[None, :, None] * directions[:, None]
        for m in range(0, len(selected), micro_chunk):
            part = selected[m:m + micro_chunk]
            samples = centers[:, :, None] + radius * part[None, None]
            flat = samples.reshape(-1, 3)
            if isinstance(field, FusedGridLookup):
                values = field.evaluate(flat)
                value, gradient = values[:, 0], values[:, 1:]
            else:
                value, gradient = field.value(flat), field.gradient(flat)
            norm = torch.sqrt(gradient.square().sum(-1) + eta * eta).clamp_min(1e-30)
            distance = value / norm
            unit = (gradient / norm[:, None]).reshape(*samples.shape)
            barrier = (unit * directions[:, None, None]).sum(-1).abs()
            influence = (_compact_shell(distance / epsilon) / epsilon).reshape(samples.shape[:-1])
            integrated[:, m:m + len(part)] += (influence * barrier * valid[..., None]).sum(1)
            evaluations += len(flat)
    integrated *= kappa * path_step
    if radius == 0:
        # All requested offsets query identical positions. Reuse that exact
        # integrand, without changing the requested quadrature expectation.
        tau = integrated.expand(-1, len(counts))
    else:
        cumulative = integrated[:, 1:].cumsum(1)
        tau = torch.stack([integrated[:, 0] if m == 1 else cumulative[:, m - 1] / m
                           for m in counts], 1)
    return torch.exp(-tau), tau, evaluations


class ToyField:
    def __init__(self, kind, offset):
        self.kind, self.offset = kind, offset

    def value(self, x):
        if self.kind == "filament":
            return (x[:, 0] - self.offset).square() + x[:, 2].square() - .10**2
        if self.kind == "grazing":
            return x[:, 0] + .12 * x[:, 2] - self.offset
        if self.kind == "two_surfaces":
            a = (x[:, 0] - self.offset).square() + x[:, 2].square() - .10**2
            b = (x[:, 0] + self.offset).square() + (x[:, 2] - .3).square() - .10**2
            return a * b
        return (x[:, 0] - self.offset).square() + (x[:, 2] / .35).square() - .15**2

    def gradient(self, x):
        if self.kind == "grazing":
            return torch.tensor([1., 0., .12], device=x.device, dtype=x.dtype).expand_as(x)
        first = torch.stack((2 * (x[:, 0] - self.offset), torch.zeros_like(x[:, 0]),
                             2 * x[:, 2]), 1)
        if self.kind == "filament":
            return first
        if self.kind == "two_surfaces":
            a = (x[:, 0] - self.offset).square() + x[:, 2].square() - .10**2
            b = (x[:, 0] + self.offset).square() + (x[:, 2] - .3).square() - .10**2
            second = torch.stack((2 * (x[:, 0] + self.offset), torch.zeros_like(x[:, 0]),
                                  2 * (x[:, 2] - .3)), 1)
            return first * b[:, None] + second * a[:, None]
        first[:, 2] /= .35**2
        return first


def preflight(device):
    offsets = _micro_offsets(128, device, torch.float64)
    origin = torch.tensor([[0., 0., -2.]], device=device, dtype=torch.float64)
    direction = torch.tensor([[0., 0., 1.]], device=device, dtype=torch.float64)
    maximum = origin.new_tensor([4.])
    rows, equivalence = [], []
    for kind in ("slab", "filament", "grazing", "two_surfaces"):
        for d in np.linspace(0, 1.5, 31):
            field = ToyField(kind, float(d))
            t, tau, _ = transmission_prefixes(field, origin, direction, maximum,
                radius=1., epsilon=.04, path_step=.01, offsets=offsets,
                counts=(1, 8, 32, 128), launch_start=0., path_chunk=32, micro_chunk=8)
            rows.append({"kind": kind, "d_over_r": float(d), "transmission": t[0].tolist(),
                         "tau": tau[0].tolist(), "micro_counts": [1, 8, 32, 128]})
        field = ToyField(kind, .5)
        for count in (1, 8, 32, 128):
            kwargs = dict(radius=1., epsilon=.04, path_step=.01, eta=1e-6,
                          kappa=-math.log(.01), launch_exclusion_factor=0.)
            dense_t, dense_tau, _ = _transmission(field, origin, direction, maximum,
                offsets=_micro_offsets(count, device, torch.float64), surface_barrier=True, **kwargs)
            small_t, small_tau, _ = transmission_prefixes(field, origin, direction, maximum,
                offsets=offsets, counts=(count,), path_chunk=17, micro_chunk=3, **kwargs)
            equivalence.append({"kind": kind, "micro": count,
                "tau_error": float((dense_tau - small_tau[:, 0]).abs().max()),
                "transmission_error": float((dense_t - small_t[:, 0]).abs().max())})
    stats = []
    for m in COUNTS:
        # These are the actual common prefixes used by transmission_prefixes,
        # not independently evaluated pow/sin kernels of different sizes.
        u = torch.zeros_like(offsets[:1]) if m == 1 else offsets[:m]
        lengths = u.norm(dim=1)
        angles = torch.atan2(u[:, 2], u[:, 0])
        bins = ((angles + math.pi) / (2 * math.pi) * 8).long().clamp(0, 7)
        stats.append({"micro": m, "offsets": u.tolist(), "digest": digest(u),
            "mean_norm": float(lengths.mean()), "rms_norm": float(lengths.square().mean().sqrt()),
            "norm_p50_p90_p95": torch.quantile(lengths, lengths.new_tensor([.5, .9, .95])).tolist(),
            "max_norm": float(lengths.max()), "azimuth_occupancy": torch.bincount(bins, minlength=8).tolist(),
            "radial_occupancy": torch.bincount((lengths * 4).long().clamp(0, 3), minlength=4).tolist()})
    center = origin[:, None] + origin.new_tensor([.5, 1.])[None, :, None] * direction[:, None]
    actual = center[:, :, None] + 2. * _micro_offsets(1, device, torch.float64)[None, None]
    return {"micro_offset_statistics": stats, "sequence_digest": digest(offsets),
        "nested_prefix_exact": all(torch.equal(torch.tensor(row["offsets"], device=device,
            dtype=torch.float64), offsets[:row["micro"]]) for row in stats if row["micro"] > 1),
        "historical_independent_draw_max_error": max(float((_micro_offsets(m, device,
            torch.float64)-offsets[:m]).abs().max()) for m in COUNTS if m > 1),
        "centerline_trace": {"origins": origin.tolist(), "directions": direction.tolist(),
            "times": [.5, 1.], "radius": 2., "offsets": [[0., 0., 0.]],
            "sample_positions": actual.tolist(), "maximum_centerline_difference": float((actual[:, :, 0] - center).abs().max())},
        "toy_rows": rows, "streaming_equivalence": equivalence}


def write_json(path, report):
    # Experiment outputs are machine-generated artifacts, never hand-edited measurements.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def progress(phase, **kwargs):
    print(json.dumps({"phase": "v0811_" + phase, **kwargs}), flush=True)


def build_scene(cache):
    from .corrected_birth import CorrectedBirthConfig, _build_context
    from .mesh_field import prepare_stanford_bunny
    from .emitter_scaling import _center_bank
    from .million_emitter import MillionEmitterConfig, _template, _chart_masses
    from .sparse_geodesic import prepare_centers
    from .stratified_polar import polar_layout, polar_weights, _shared_ray_map_from_prepared
    from .finite_packet import _surface_spacing
    from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
    mesh = prepare_stanford_bunny(Path("data/stanford_bunny/cache/bun_zipper.ply"), build_surface_scaffold=False)
    context = _build_context(mesh, CorrectedBirthConfig(dictionary_count=128, initial_count=32,
        surface_samples=4096, views=4, resolution=256, surface_scramble_seed=101))
    device, dtype = context.reference_points.device, context.reference_points.dtype
    bank = _center_bank(context, 1024)
    template = _template(context, bank, MillionEmitterConfig())
    area = json.loads(Path("artifacts/v086_continuous_source_field.json").read_text())["source_measure"]["reference_area_calibration"]
    mass = _chart_masses(context, template, area)["HIERARCHICAL_MASS_CONSERVING"]
    layout = polar_layout(1024, 32, 32, 101, 0., 0., "AREA_STRATIFIED", device, dtype)
    source_path = cache / "source.pt"
    if source_path.exists():
        source = torch.load(source_path, weights_only=True)
        if source["layout_digest"] != layout.digest:
            raise RuntimeError("cached source identity mismatch")
    else:
        prepared = prepare_centers(context.base, template)
        positions, normals = [], []
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, 1024, 8):
                ids = torch.arange(start, start + 8, device=device)
                x, n, _ = _shared_ray_map_from_prepared(context.base, template, prepared,
                    ids, layout.theta[start:start+8], layout.rho[start:start+8], 32)
                positions.append(x.reshape(-1, 3).cpu())
                normals.append(n.reshape(-1, 3).cpu())
        source = {"positions": torch.cat(positions), "normals": torch.cat(normals),
            "weights": polar_weights(layout, mass).reshape(-1).cpu() / area,
            "source_mass": float(mass.sum()), "layout_digest": layout.digest,
            "mapping_seconds": time.perf_counter() - started}
        torch.save(source, source_path)
    scene = dict(context=context, source=source, device=device, dtype=dtype,
                 h=_surface_spacing(context.reference_points), eta=template.eta,
                 atlas=nested_fibonacci_atlas(device, (4,)),
                 boundary=enclosing_observation_sphere(context.reference_points))
    scene["field"] = FusedGridLookup(context.base)
    source["emitter_count"] = len(source["positions"])
    source["normalized_weight_sum"] = float(source["weights"].sum())
    source["measured_raw_source_mass"] = float(source["weights"].sum()) * area
    source["mass_error"] = abs(source["measured_raw_source_mass"]-source["source_mass"])
    source["weights_digest"] = digest(source["weights"])
    source["positions_digest"] = digest(source["positions"])
    source["normals_digest"] = digest(source["normals"])
    source["geometry_grid_digest"] = digest(context.base.grid)
    if source["mass_error"] > 1e-10:
        raise RuntimeError("source mass does not match the frozen hierarchy")
    return scene


def grid_validation(scene):
    field, original = scene["field"], scene["context"].base
    points = scene["source"]["positions"][::1024].to(scene["device"])
    # Include points well outside the grid to test identical border clamping.
    points = torch.cat((points, points + 10, points - 10))
    values = field.evaluate(points)
    errors = {"value_max_error": float((values[:, 0] - original.value(points)).abs().max()),
              "gradient_max_error": float((values[:, 1:] - original.gradient(points)).abs().max())}
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"fused field interpolation equivalence failed: {errors}")
    origins = points[:32]
    directions = scene["atlas"].directions[0].expand_as(origins)
    maximum = scene["boundary"].exit_times(origins, directions)
    kwargs = dict(radius=scene["h"], epsilon=scene["h"], path_step=.5*scene["h"],
                  eta=scene["eta"], kappa=-math.log(.01), launch_exclusion_factor=1.05)
    offsets = _micro_offsets(128, origins.device, origins.dtype)
    for m in (1, 8, 128):
        t0, tau0, _ = _transmission(original, origins, directions, maximum,
            offsets=_micro_offsets(m, origins.device, origins.dtype), surface_barrier=True, **kwargs)
        t, tau, _ = transmission_prefixes(field, origins, directions, maximum,
            offsets=offsets, counts=(m,), **kwargs)
        errors[f"micro{m}_transmission_error"] = float((t[:, 0]-t0).abs().max())
        errors[f"micro{m}_tau_error"] = float((tau[:, 0]-tau0).abs().max())
    small_source = {**scene["source"], **{key:scene["source"][key][::1024][:32]
        for key in ("positions", "normals", "weights")}}
    small_scene = {**scene, "source": small_source}
    original_tau, streamed_tau = [], []
    for direction in scene["atlas"].directions:
        directions = direction.expand_as(origins)
        maximum = scene["boundary"].exit_times(origins, directions)
        _, tau0, _ = _transmission(original, origins, directions, maximum,
            offsets=_micro_offsets(8, scene["device"], scene["dtype"]),
            surface_barrier=True, **kwargs)
        _, tau, _ = transmission_prefixes(field, origins, directions, maximum,
            offsets=_micro_offsets(128, scene["device"], scene["dtype"]),
            counts=(8,), **kwargs)
        original_tau.append(tau0.cpu()); streamed_tau.append(tau[:, 0].cpu())
    a = readout(small_scene, torch.stack(original_tau), 32)
    b = readout(small_scene, torch.stack(streamed_tau), 32)
    errors["four_view_rgb_max_error"] = float(np.max(np.abs(np.stack(a)-np.stack(b))))
    if max(errors.values()) > 1e-9:
        raise RuntimeError(errors)
    return errors


def profile_micro(scene):
    source, device = scene["source"], scene["device"]
    origins = source["positions"][::256].to(device)
    direction = scene["atlas"].directions[0].expand_as(origins)
    maximum = scene["boundary"].exit_times(origins, direction)
    offsets = _micro_offsets(128, device, origins.dtype)
    rows = []
    for pc, mc in ((16, 4), (32, 8), (64, 16)):
        for m in COUNTS:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            t, tau, evaluations = transmission_prefixes(scene["field"], origins, direction, maximum,
                radius=scene["h"], epsilon=scene["h"], path_step=.5*scene["h"],
                offsets=offsets, counts=(m,), eta=scene["eta"], path_chunk=pc, micro_chunk=mc)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            rows.append({"micro": m, "emitter_chunk": len(origins), "path_chunk": pc,
                "micro_chunk": mc, "seconds": elapsed, "field_evaluations": evaluations,
                "interactions_per_second": evaluations/elapsed, "packets_per_second": len(origins)/elapsed,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
                "cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                "transmission_mean": float(t.mean()), "tau_mean": float(tau.mean())})
            progress("profile", **rows[-1])
    return rows


def trace_family(scene, cache, label, radius, *, epsilon_ratio=1., step_ratio=.5,
                 counts=COUNTS, launch_start=None, chunks=(4096, 32, 8)):
    """Cache scalar tau only, so detector resolution can reuse scene transport."""
    path = cache / (label + ".pt")
    h, device, source = scene["h"], scene["device"], scene["source"]
    settings = {"radius_over_h": radius, "epsilon_over_h": epsilon_ratio,
        "path_step_over_epsilon": step_ratio, "micro_counts": list(counts),
        "emitter_chunk": chunks[0], "path_chunk": chunks[1], "micro_chunk": chunks[2],
        "launch_start_override": launch_start, "layout_digest": source["layout_digest"]}
    if path.exists():
        result = torch.load(path, weights_only=True)
        if result["settings"] != settings:
            raise RuntimeError(f"cache settings mismatch: {label}")
        progress("cache_hit", label=label)
        return result
    epsilon = epsilon_ratio * h
    offsets = _micro_offsets(128, device, scene["dtype"])
    count = len(source["positions"])
    # Only one scalar optical depth per emitter/view/micro count, on CPU.
    tau_cpu = torch.empty((4, count, len(counts)), dtype=scene["dtype"])
    evaluations = 0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for view in range(4):
        for start in range(0, count, chunks[0]):
            end = min(start + chunks[0], count)
            positions = source["positions"][start:end].to(device)
            directions = scene["atlas"].directions[view].expand_as(positions)
            maximum = scene["boundary"].exit_times(positions, directions)
            _, tau, evals = transmission_prefixes(scene["field"], positions, directions, maximum,
                radius=radius*h, epsilon=epsilon, path_step=step_ratio*epsilon,
                offsets=offsets, counts=counts, eta=scene["eta"],
                path_chunk=chunks[1], micro_chunk=chunks[2], launch_start=launch_start)
            tau_cpu[view, start:end] = tau.cpu()
            evaluations += evals
            if start % (chunks[0]*32) == 0:
                progress("trace", label=label, view=view, emitters=end, total=count,
                         elapsed_seconds=time.perf_counter()-started)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    result = {"settings": settings, "tau": tau_cpu, "runtime_seconds": elapsed,
        "attempted_packets":4*count, "requested_micro_counts":list(counts),
        "unique_offsets_evaluated":1 if radius==0 else 1+max((m for m in counts if m>1), default=0),
        "field_evaluations": evaluations, "interactions_per_second": evaluations/elapsed,
        "packets_per_second": 4*count/elapsed, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
        "cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "actual_launch_start": 1.05*(radius*h+epsilon) if launch_start is None else launch_start}
    torch.save(result, path)
    progress("family_complete", label=label, runtime_seconds=elapsed)
    return result


def readout(scene, tau, resolution=256):
    from .finite_packet import _accumulate_continuous
    from .continuous_source import _emission_factor
    from .meshfree_surface import meshfree_base_color
    resolution = (resolution, resolution) if isinstance(resolution, int) else resolution
    device, source = scene["device"], scene["source"]
    images = []
    for view in range(4):
        accumulator = torch.zeros((math.prod(resolution), 3), device=device, dtype=torch.float32)
        for start in range(0, len(source["positions"]), 8192):
            end = start+8192
            positions = source["positions"][start:end].to(device)
            normals = source["normals"][start:end].to(device)
            weights = source["weights"][start:end].to(device)
            cosine = normals @ scene["atlas"].directions[view]
            colors = meshfree_base_color(positions, scene["field"].lower, scene["field"].upper)
            energy = (weights * torch.exp(-tau[view, start:end].to(device)) *
                      _emission_factor(cosine, .05) * (.35+.65*cosine.clamp_min(0)))[:, None]*colors
            _accumulate_continuous(accumulator, positions.float(), energy.float(),
                scene["atlas"].right[view].float(), scene["atlas"].up[view].float(),
                scene["boundary"].center.float(), 2.8, resolution)
        images.append((accumulator * (1.5*math.prod(resolution))).reshape(*resolution, 3).cpu().numpy())
    return images


def reference_images(scene, resolution):
    from .finite_packet import _hard_images
    resolution = (resolution, resolution) if isinstance(resolution, int) else resolution
    context = scene["context"]
    images, _ = _hard_images(context.base, context.reference_points, context.reference_normals,
        context, scene["atlas"], scene["boundary"], resolution, recompute_visibility=True)
    return images


def difference(a, b):
    delta = (a-b).double()
    return {"mae": float(delta.abs().mean()), "rms": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm()/b.double().norm().clamp_min(1e-30)),
            "max": float(delta.abs().max())}


def family_metrics(scene, family, reference):
    from .finite_packet import _image_metrics
    counts, tau = family["settings"]["micro_counts"], family["tau"]
    last = readout(scene, tau[:, :, -1])
    last_tensor = torch.from_numpy(np.stack(last))
    rows = []
    for index, m in enumerate(counts):
        images = last if index == len(counts)-1 else readout(scene, tau[:, :, index])
        row = {"micro": m, **family["settings"], **_image_metrics(images, reference),
            "actual_launch_start":family["actual_launch_start"],
            "source_mass": scene["source"]["source_mass"],
            "rgb_difference_to_max_micro": difference(torch.from_numpy(np.stack(images)), last_tensor),
            "tau_difference_to_max_micro": difference(tau[:, :, index], tau[:, :, -1]),
            "transmission_difference_to_max_micro": difference(torch.exp(-tau[:, :, index]), torch.exp(-tau[:, :, -1])),
            "tau_mean": float(tau[:, :, index].mean()), "transmission_mean": float(torch.exp(-tau[:, :, index]).mean())}
        rows.append(row)
    return rows


def forensic(scene, pre):
    import ast
    calls = []
    for filename in ("million_emitter", "stratified_polar", "geodesic_source", "emitter_scaling", "continuous_source"):
        path = Path("src/zlt") / (filename + ".py")
        code = path.read_text()
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                    "_transmission", "_micro_offsets", "_render_fixed_sources", "_render_chart"):
                calls.append({"file": str(path), "line": node.lineno,
                              "call": ast.get_source_segment(code, node)})
    h = scene["h"]
    paths = []
    for name, micro, chunk in (("million_emitter._stream_render", 1, 8192),
        ("stratified_polar._shared_stream_render", 1, 8192),
        ("geodesic_source._render_chart default", 4, 256),
        ("geodesic_source FD/multiseed controls via _render_chart", 1, 256),
        ("geodesic_source._multiview_reuse", 1, "all active emitters"),
        ("geodesic_source._energy_checks", 1, "all active emitters"),
        ("emitter_scaling primary and FullHD", 1, 256)):
        paths.append({"path": name, "micro_samples": micro, "radius": h, "epsilon": h,
            "path_step": .5*h, "launch_exclusion_factor": 1.05,
            "launch_exclusion_world": 2.1*h, "emitter_chunk": chunk,
            "offsets": _micro_offsets(micro, torch.device("cpu"), torch.float64).tolist()})
    offsets = _micro_offsets(128, scene["device"], scene["dtype"])
    production_offsets = []
    for m in COUNTS:
        u = torch.zeros_like(offsets[:1]) if m == 1 else offsets[:m]
        length = u.norm(dim=1)
        bins = ((torch.atan2(u[:, 2], u[:, 0]) + math.pi)/(2*math.pi)*8).long().clamp(0, 7)
        polar_bins = ((u[:, 1]/length.clamp_min(1e-30) + 1)*4).long().clamp(0, 7)
        solid_bins = torch.bincount(polar_bins*8 + bins, minlength=64) if m>1 else torch.zeros(64, dtype=torch.long)
        production_offsets.append({"micro":m, "offsets":u.tolist(), "digest":digest(u),
            "angular_direction_defined":m>1, "solid_angle_occupancy_8x8":solid_bins.reshape(8, 8).tolist(),
            "mean_norm":float(length.mean()), "rms_norm":float(length.square().mean().sqrt()),
            "norm_p50_p90_p95":torch.quantile(length, length.new_tensor([.5, .9, .95])).tolist(),
            "max_norm":float(length.max()), "azimuth_occupancy":torch.bincount(bins, minlength=8).tolist(),
            "radial_occupancy":torch.bincount((length*4).long().clamp(0, 3), minlength=4).tolist()})
    world = []
    for ratio in RADII:
        for m in COUNTS:
            u = torch.zeros_like(offsets[:1]) if m == 1 else offsets[:m]
            for view, w in enumerate(scene["atlas"].directions):
                perpendicular = u - (u @ w)[:, None] * w
                lengths = perpendicular.norm(dim=1) * ratio * h
                world.append({"radius_over_h": ratio, "micro": m, "view": view,
                    "rms_transverse_world": float(lengths.square().mean().sqrt()),
                    "p95_transverse_world": float(torch.quantile(lengths, .95)),
                    "max_transverse_world": float(lengths.max())})
    class RecordingField:
        def value(self, points):
            self.samples = points[:8].detach().cpu()
            return scene["context"].base.value(points)

        def gradient(self, points):
            return scene["context"].base.gradient(points)

    recorder = RecordingField()
    origin = scene["source"]["positions"][:1].to(scene["device"])
    direction = scene["atlas"].directions[:1]
    center_offsets = _micro_offsets(1, scene["device"], scene["dtype"])
    _transmission(recorder, origin, direction, scene["boundary"].exit_times(origin, direction),
        radius=h, epsilon=h, path_step=.5*h, offsets=center_offsets,
        eta=scene["eta"], kappa=-math.log(.01), surface_barrier=True,
        launch_exclusion_factor=1.05)
    times = 2.1*h + (torch.arange(len(recorder.samples), dtype=scene["dtype"]) + .5)*.5*h
    expected = origin.cpu() + times[:, None]*direction.cpu()
    actual_error = float((recorder.samples-expected).abs().max())
    return {"call_sites": calls, "production_paths": paths, "surface_spacing_h": h,
        "offset_definition": "world-space 3D ball; transverse statistics project perpendicular to each ray",
        "production_sequence_digest": digest(offsets), "production_micro_offset_statistics": production_offsets,
        "micro1_zero": bool(torch.count_nonzero(_micro_offsets(1, scene["device"], scene["dtype"])) == 0),
        "centerline_sample_error": actual_error,
        "actual_production_query_trace": {"origin":origin.tolist(), "direction":direction.tolist(),
            "times":times.tolist(), "radius":h, "offsets":center_offsets.tolist(),
            "queried_positions":recorder.samples.tolist(), "centerline_max_error":actual_error},
        "effective_world_support": world}


def convergence(rows):
    # M=128 is a finite reference, never automatic evidence of convergence.
    accepted = [r["micro"] for r in rows if 1 < r["micro"] < 128 and
                r["rgb_difference_to_max_micro"]["relative_l2"] <= .01 and
                r["transmission_difference_to_max_micro"]["rms"] <= .01 and
                r["tau_difference_to_max_micro"]["relative_l2"] <= .05]
    minimum = next((m for m in accepted if all(n in accepted for n in COUNTS if m <= n < 128)), None)
    zero_radius = rows[0]["radius_over_h"] == 0.
    if zero_radius and minimum is not None:
        minimum = 1  # Every offset is exactly the same point when r=0.
    return {"minimum_converged_count": minimum, "converged": minimum is not None,
        "zero_radius_degenerate":zero_radius,
        "criteria": {"rgb_relative_l2": .01, "transmission_rms": .01, "tau_relative_l2": .05},
        "reference_micro_count": 128, "reference_is_not_infinite_quadrature": True}


def streaming_validation(scene):
    """Check per-packet chunk invariance and fixed-workspace count scaling."""
    source, device = scene["source"], scene["device"]
    x = source["positions"][::256].to(device)
    w = scene["atlas"].directions[0].expand_as(x)
    offsets = _micro_offsets(128, device, x.dtype)
    kwargs = dict(radius=scene["h"], epsilon=scene["h"], path_step=.5*scene["h"],
                  offsets=offsets, counts=COUNTS, eta=scene["eta"])
    reference = None
    errors = []
    for pc, mc in ((16, 4), (32, 8), (64, 16)):
        _, tau, _ = transmission_prefixes(scene["field"], x, w,
            scene["boundary"].exit_times(x, w), path_chunk=pc, micro_chunk=mc, **kwargs)
        if reference is None:
            reference = tau.cpu()
        errors.append({"path_chunk": pc, "micro_chunk": mc,
                       "tau_max_error": float((tau.cpu()-reference).abs().max())})
    del x, w, tau
    rows = []
    for count in (4096, 32768):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for start in range(0, count, 4096):
            x = source["positions"][start:start+4096].to(device)
            w = scene["atlas"].directions[0].expand_as(x)
            _, tau, _ = transmission_prefixes(scene["field"], x, w,
                scene["boundary"].exit_times(x, w), path_chunk=32, micro_chunk=8, **kwargs)
            del x, w, tau
        torch.cuda.synchronize()
        rows.append({"packets": count, "emitter_chunk": 4096, "path_chunk": 32,
                     "micro_chunk": 8, "micro_counts": list(COUNTS),
                     "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20})
    equivalent = max(r["tau_max_error"] for r in errors) < 1e-10
    bounded = rows[-1]["peak_cuda_allocated_mib"] <= rows[0]["peak_cuda_allocated_mib"] * 1.1
    if not equivalent or not bounded:
        raise RuntimeError(f"streaming validation failed: {errors}, {rows}")
    return {"chunk_equivalence": errors, "emitter_count_scaling": rows,
            "equivalent": equivalent, "memory_bounded": bounded}


def regional_changes(scene, tau, chosen):
    from scipy.spatial import cKDTree
    source, context = scene["source"], scene["context"]
    ref = context.reference_points.cpu().numpy()
    rn = context.reference_normals.cpu().numpy()
    neighbors = cKDTree(ref).query(ref, k=9)[1][:, 1:]
    curvature = (1 - (rn[:, None]*rn[neighbors]).sum(-1)).mean(1)
    owners = cKDTree(ref).query(source["positions"].numpy())[1]
    high = curvature[owners] >= np.quantile(curvature, .75)
    thin = source["positions"][:, 2].numpy() >= np.quantile(ref[:, 2], .82)
    rows = []
    for view in range(4):
        direction = scene["atlas"].directions[view].cpu()
        silhouette = (source["normals"] @ direction).abs().numpy() <= .15
        for name, mask in (("all", np.ones(len(owners), bool)), ("thin", thin),
                           ("silhouette", silhouette), ("high_curvature", high),
                           ("broad_smooth", ~(thin | high | silhouette))):
            a, b = tau[view, mask, 0], tau[view, mask, chosen]
            delta = (torch.exp(-a)-torch.exp(-b)).abs()
            rows.append({"view": view, "region": name, "packets": len(a),
                "center_tau_mean": float(a.mean()), "finite_tau_mean": float(b.mean()),
                "tau_absolute_difference_mean": float((a-b).abs().mean()),
                "transmission_absolute_difference_mean": float(delta.mean()),
                "fraction_transmission_change_gt_001": float((delta > .01).double().mean()),
                "fraction_center_clear_finite_attenuated": float(((torch.exp(-a)>.99)&(torch.exp(-b)<.95)).double().mean())})
    return rows


def small_controls(scene, best_radius, micro):
    """4096 spatially distributed emitters x all four production directions."""
    source, device, h = scene["source"], scene["device"], scene["h"]
    # Integer stride 256 would always select radial stop zero in the 32x32
    # layout. Endpoint-spanning indices cover different radial stops as well.
    ids = torch.linspace(0, len(source["positions"])-1, 4096).long()
    x = source["positions"][ids].to(device)
    offsets = _micro_offsets(128, device, x.dtype)
    path_rows, eps_rows, calibration = [], [], []
    arrays = []
    for ratio in (1., .5, .25, .125):
        tau_views = []
        for w in scene["atlas"].directions:
            _, tau, _ = transmission_prefixes(scene["field"], x, w.expand_as(x),
                scene["boundary"].exit_times(x, w.expand_as(x)),
                radius=best_radius*h, epsilon=h, path_step=ratio*h,
                offsets=offsets, counts=(micro,), eta=scene["eta"], launch_start=2.1*h)
            tau_views.append(tau.cpu())
        arrays.append(torch.cat(tau_views))
    for ratio, tau in zip((1., .5, .25, .125), arrays):
        path_rows.append({"path_step_over_epsilon": ratio, "micro": micro,
            "radius_over_h": best_radius, "packets": 4*len(x),
            "tau_error_to_0125": difference(tau, arrays[-1]),
            "transmission_error_to_0125": difference(torch.exp(-tau), torch.exp(-arrays[-1]))})
    valid_steps = [row["path_step_over_epsilon"] for row in path_rows[:-1]
                  if row["transmission_error_to_0125"]["rms"] < .005 and row["tau_error_to_0125"]["relative_l2"] < .02]
    best_step = max(valid_steps) if valid_steps else .125
    # Explicit shell control at nonzero radius; if radius zero wins, use h as
    # the declared shell-control radius because epsilon/r is undefined at r=0.
    shell_radius = best_radius if best_radius > 0 else 1.
    for ratio in (.25, .5, 1., 2.):
        tau_views = []
        for w in scene["atlas"].directions:
            _, tau, _ = transmission_prefixes(scene["field"], x, w.expand_as(x),
                scene["boundary"].exit_times(x, w.expand_as(x)), radius=shell_radius*h,
                epsilon=ratio*shell_radius*h, path_step=best_step*ratio*shell_radius*h,
                offsets=offsets, counts=(micro,), eta=scene["eta"], launch_start=2.1*h)
            tau_views.append(tau.cpu())
        tau = torch.cat(tau_views)
        eps_rows.append({"epsilon_over_radius": ratio, "radius_over_h": shell_radius,
            "micro": micro, "path_step_over_epsilon": best_step,
            "tau_mean": float(tau.mean()), "transmission_mean": float(torch.exp(-tau).mean()),
            "scope": "16384 packet shell-sensitivity control; no image-optimal epsilon inferred"})
    class Plane:
        def value(self, p): return p[:, 2]
        def gradient(self, p):
            return p.new_tensor([0., 0., 1.]).expand_as(p)
    origin = x.new_tensor([[0., 0., -8*h]])
    direction = x.new_tensor([[0., 0., 1.]])
    for r in RADII:
        t, tau, _ = transmission_prefixes(Plane(), origin, direction, origin.new_tensor([16*h]),
            radius=r*h, epsilon=h, path_step=.125*h, offsets=offsets, counts=COUNTS,
            eta=scene["eta"])
        for i, m in enumerate(COUNTS):
            calibration.append({"radius_over_h": r, "micro": m, "transmission": float(t[0, i]),
                "tau": float(tau[0, i]), "target": .01, "kappa": -math.log(.01)})
    return {"path_rows": path_rows, "epsilon_rows": eps_rows, "calibration": calibration,
        "source_indices_digest":digest(ids), "source_indices":ids.tolist(),
        "path_converged": bool(valid_steps), "best_step": best_step,
        "best_epsilon_over_radius": None,
        "epsilon_selection_reason": "shell sensitivity alone cannot identify a fidelity-optimal epsilon"}


def make_figures(report, images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path("figures"); destination.mkdir(exist_ok=True)
    paths = []
    def save(name, figure):
        path = destination / ("v0811_"+name+".png")
        figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)
        paths.append(str(path))
    pre = report["preflight"]
    fig, ax = plt.subplots(figsize=(6, 5))
    u = np.asarray(pre["micro_offset_statistics"][-1]["offsets"])
    ax.scatter(u[:, 0], u[:, 2], c=np.arange(len(u)), s=18)
    ax.set(xlabel="u_x", ylabel="u_z", title="Nested 3D-ball offsets (projection)", aspect="equal")
    save("micro_offsets", fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in (1, 8, 32, 128):
        rows = [r for r in report["forensic"]["effective_world_support"] if r["micro"]==m and r["view"]==0]
        ax.plot([r["radius_over_h"] for r in rows], [r["rms_transverse_world"] for r in rows], "o-", label=str(m))
    ax.set(xlabel="r/h", ylabel="RMS perpendicular offset, world units"); ax.legend(title="micro")
    save("effective_transverse_support", fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, kind in zip(axes.flat, ("slab", "filament", "grazing", "two_surfaces")):
        rows = [r for r in pre["toy_rows"] if r["kind"]==kind]
        for i, m in enumerate((1, 8, 32, 128)):
            ax.plot([r["d_over_r"] for r in rows], [1-r["transmission"][i] for r in rows], label=str(m))
        ax.set(title=kind, xlabel="lateral d/r", ylabel="1 - transmission"); ax.legend()
    save("thin_surface_offset_toy", fig)
    screen = report["screening"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for r in RADII:
        rows = [x for x in screen if x["radius_over_h"]==r and x["micro"]<128]
        for ax, key in zip(axes, ("rgb_difference_to_max_micro", "transmission_difference_to_max_micro", "tau_difference_to_max_micro")):
            ax.plot([x["micro"] for x in rows], [x[key]["rms"] for x in rows], "o-", label=str(r))
            ax.set(xscale="log", xlabel="micro count", ylabel=key+" RMS")
    axes[0].legend(title="r/h"); save("micro_convergence", fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("whole_image_mse", "silhouette_0_8px_mse")):
        for m in (1, 8, 32, 128):
            rows = [x for x in screen if x["micro"]==m]
            ax.plot([x["radius_over_h"] for x in rows], [x[key] for x in rows], "o-", label=str(m))
        ax.set(xlabel="r/h", ylabel=key); ax.legend(title="micro")
    save("radius_ablation", fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    values = np.asarray([[next(x["whole_image_mse"] for x in screen if x["radius_over_h"]==r and x["micro"]==m) for m in COUNTS] for r in RADII])
    display = ax.imshow(values, aspect="auto"); fig.colorbar(display, ax=ax)
    ax.set(xticks=range(7), xticklabels=COUNTS, yticks=range(6), yticklabels=RADII, xlabel="micro", ylabel="r/h", title="256-square whole MSE")
    save("micro_radius_heatmap", fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    regions = [x for x in report["offcenter_regions"] if x["view"]==0]
    axis = np.arange(len(regions))
    axes[0].bar(axis-.18, [x["center_tau_mean"] for x in regions], .36, label="centerline")
    axes[0].bar(axis+.18, [x["finite_tau_mean"] for x in regions], .36, label="finite support")
    axes[0].set(xticks=axis, xticklabels=[x["region"] for x in regions], ylabel="mean optical depth")
    axes[0].legend()
    axes[1].bar([x["region"] for x in regions], [x["fraction_transmission_change_gt_001"] for x in regions])
    axes[1].set(ylabel="fraction with |delta T| > 0.01", title="View 0 off-center contribution")
    for ax in axes: ax.tick_params(axis="x", rotation=25)
    save("offcenter_tau_contribution", fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = report["controls"]["path_rows"]
    ax.plot([x["path_step_over_epsilon"] for x in rows], [x["transmission_error_to_0125"]["rms"] for x in rows], "o-")
    ax.set(xlabel="path step / epsilon", ylabel="T RMS error to 0.125", title="16384-packet convergence control")
    save("path_step_convergence", fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = report["controls"]["epsilon_rows"]
    ax.plot([x["epsilon_over_radius"] for x in rows], [x["transmission_mean"] for x in rows], "o-")
    ax.set(xlabel="epsilon/r", ylabel="mean transmission", title="Shell sensitivity (kappa fixed)")
    save("epsilon_control", fig)
    for filename, metric in (("thin_response", "thin_feature_response_ratio"), ("high_frequency_response", "edge_sharpness_ratio")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in (1, 8, 32, 128):
            rows = [x for x in screen if x["micro"]==m]
            ax.plot([x["radius_over_h"] for x in rows], [x[metric] for x in rows], "o-", label=str(m))
        ax.set(xlabel="r/h", ylabel=metric); ax.legend(title="micro")
        save(filename, fig)
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    for i, (label, current) in enumerate(images.items()):
        setting = next(row for row in report["fullhd"] if row["label"]==label)
        for view in range(4):
            axes[i, view].imshow(np.clip(current[view], 0, 1)); axes[i, view].axis("off")
            axes[i, view].set_title(f"{label}: r/h={setting['radius_over_h']:g}, M={setting['micro']} / view {view}", fontsize=9)
    save("fullhd_comparison", fig)
    for filename, metric in (("runtime_vs_micro_count", "seconds"), ("vram_vs_micro_count", "peak_cuda_allocated_mib")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for pc in (16, 32, 64):
            rows = [x for x in report["performance"] if x["path_chunk"]==pc]
            ax.plot([x["micro"] for x in rows], [x[metric] for x in rows], "o-", label=f"path={pc}, micro={pc//4}")
        ax.set(xscale="log", xlabel="micro count", ylabel=metric, title="4096-packet standalone profile"); ax.legend()
        save(filename, fig)
    return paths


def run_experiment(profile_only=False):
    if not torch.cuda.is_available():
        raise RuntimeError("The formal million-emitter experiment requires local CUDA; use --preflight for CPU controls.")
    cache = Path("runs/v0811_transverse")
    cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    historical = subprocess.check_output(["git", "ls-tree", "-r", "3965ebf", "artifacts"], text=True)
    progress("setup")
    with torch.no_grad():
        scene = build_scene(cache)
        validation = grid_validation(scene)
        write_json(cache / "grid_validation.json", validation)
        progress("grid_validation", **validation)
        if (cache / "profile.json").exists():
            profile = json.loads((cache / "profile.json").read_text())
        else:
            profile = profile_micro(scene)
            write_json(cache / "profile.json", profile)
    if profile_only:
        return
    from .finite_packet import _image_metrics, _scalar_csv_rows
    from .high_sample import _write_csv
    preflight_path = Path("artifacts/v0811_preflight.json")
    if not preflight_path.exists():
        write_json(preflight_path, preflight(torch.device("cpu")))
    pre = json.loads(preflight_path.read_text())
    fastest = min((r for r in profile if r["micro"]==128), key=lambda r:r["seconds"])
    chunks = (4096, fastest["path_chunk"], fastest["micro_chunk"])
    report = {"version": "0.8.11", "starting_commit": "3965ebf555bed14ea2321478fb54801a37b47d30",
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(), "dtype": str(scene["dtype"])},
        "historical_artifact_tree_sha256": hashlib.sha256(historical.encode()).hexdigest(),
        "preflight": pre, "forensic": forensic(scene, pre), "grid_lookup_validation": validation,
        "performance": profile, "configuration": {"K_chart":1024, "N_theta":32, "N_r":32,
        "jitter":0, "radial_scheme":"AREA_STRATIFIED", "source_measure":"HIERARCHICAL_MASS_CONSERVING",
        "micro_counts":list(COUNTS), "radius_ratios":list(RADII), "primary_epsilon_over_h":1.,
        "primary_path_step_over_epsilon":.5, "primary_launch_start":2.1*scene["h"],
        "primary_launch_start_over_h":2.1, "kappa":-math.log(.01), "eta":scene["eta"],
        "sensor_gain":1.5, "ambient":.35, "emission_transition_width":.05,
        "detector_extent":2.8, "views":4, "screening_resolution":[256,256],
        "fullhd_resolution":[1080,1920], "chunks":list(chunks)},
        "source": {k:v for k,v in scene["source"].items() if not isinstance(v, torch.Tensor)},
        "workspace_contract": {"maximum_query_shape":[chunks[0],chunks[1],chunks[2],3],
            "maximum_query_tensor_bytes":math.prod(chunks)*3*8,
            "parameter_axis_present":False, "production_field":"fixed GridZeroSetField, fused trilinear lookup",
            "source_storage":"CPU positions/normals/weights; geodesic construction workspace released before transport",
            "optical_depth_storage":"CPU [4, N_emitters, 7]; no all-emitter path-by-micro tensor"},
        "screening": [], "radius_convergence": [], "family_performance": []}
    with torch.no_grad():
        report["streaming_validation"] = streaming_validation(scene)
        reference = reference_images(scene, 256)
        old_report_path = Path("artifacts/v0811_transverse_packet_integration.json")
        if old_report_path.exists():
            old_report = json.loads(old_report_path.read_text())
            report["historical_launch_auxiliary_screening"] = old_report.get(
                "historical_launch_auxiliary_screening", old_report["screening"]
                if "primary_launch_start_over_h" not in old_report["configuration"] else [])
        for r in RADII:
            # At r=h, the historical rule already starts at exactly 2.1h.
            # Reuse that measured family rather than trace identical paths.
            label = "radius_1" if r==1. else f"fixed_launch_radius_{r:g}"
            family = trace_family(scene, cache, label, r, chunks=chunks,
                                  launch_start=None if r==1. else 2.1*scene["h"])
            rows = family_metrics(scene, family, reference)
            report["screening"].extend(rows)
            report["radius_convergence"].append({"radius_over_h":r, **convergence(rows)})
            report["family_performance"].append({k:v for k,v in family.items() if k!="tau"})
            write_json(cache / "screening_partial.json", report)
            del family
        # A declared rank across image fidelity metrics selects support scale.
        center_rows = [row for row in report["screening"] if row["micro"]==1]
        report["centerline_radius_control"] = {
            "tau_mean_range":max(r["tau_mean"] for r in center_rows)-min(r["tau_mean"] for r in center_rows),
            "transmission_mean_range":max(r["transmission_mean"] for r in center_rows)-min(r["transmission_mean"] for r in center_rows),
            "whole_mse_range":max(r["whole_image_mse"] for r in center_rows)-min(r["whole_image_mse"] for r in center_rows)}
        if report["centerline_radius_control"]["tau_mean_range"] > 1e-10:
            raise RuntimeError("radius changed the centerline control despite fixed launch")
        selected = [row for row in report["screening"] if row["micro"]==128]
        narrow = next(row for row in selected if row["radius_over_h"]==.25)
        wide = next(row for row in selected if row["radius_over_h"]==2.)
        report["radius_bandwidth_control"] = {"comparison_radii_over_h":[.25, 2.],
            "reference_micro_count":128, "mean_tau_increase":wide["tau_mean"]-narrow["tau_mean"],
            "thin_response_ratio_wide_over_narrow":wide["thin_feature_response_ratio"]/narrow["thin_feature_response_ratio"],
            "edge_response_ratio_wide_over_narrow":wide["edge_sharpness_ratio"]/narrow["edge_sharpness_ratio"],
            "wide_radius_converged":next(row["converged"] for row in report["radius_convergence"] if row["radius_over_h"]==2.),
            "note":"Increased attenuation is not itself improved geometric fidelity; an unconverged wide radius remains reference-limited."}
        scores = {r:0 for r in RADII}
        for key, reverse in (("whole_image_mse", False), ("silhouette_0_8px_mse", False),
                             ("thin_feature_response_ratio", True), ("edge_sharpness_ratio", True)):
            for rank, row in enumerate(sorted(selected, key=lambda x:x[key], reverse=reverse)):
                scores[row["radius_over_h"]] += rank
        best_radius = min(scores, key=lambda r:(scores[r], r))
        best_nonzero_radius = min((r for r in scores if r>0), key=lambda r:(scores[r], r))
        nonzero_cv = next(row for row in report["radius_convergence"] if row["radius_over_h"]==best_nonzero_radius)
        report["best_nonzero_support"] = {"radius_over_h":best_nonzero_radius,
            "minimum_converged_micro_count":nonzero_cv["minimum_converged_count"],
            "epsilon_over_h":1., "epsilon_over_radius":1./best_nonzero_radius,
            "selection_scope":"256-square fixed-launch screen, not a separate FullHD winner"}
        cv = next(r for r in report["radius_convergence"] if r["radius_over_h"]==best_radius)
        current_cv = next(r for r in report["radius_convergence"] if r["radius_over_h"]==1.)
        # Use a count meeting convergence at BOTH candidate radii, else 128
        # explicitly labelled reference-limited, not converged.
        converged = cv["converged"] and current_cv["converged"]
        micro = max(cv["minimum_converged_count"], current_cv["minimum_converged_count"]) if converged else 128
        progress("selected", radius=best_radius, micro=micro, convergence_supported=converged)
        report["selection_rank"] = scores
        report["controls"] = small_controls(scene, best_radius, micro)
        if best_radius != 1.:
            report["controls_at_production_radius"] = small_controls(scene, 1., micro)
        current = torch.load(cache / "radius_1.pt", weights_only=True)
        report["offcenter_regions"] = regional_changes(scene, current["tau"], COUNTS.index(micro))
        # Contrast the fixed primary start with the historical variable start
        # on the same 4096 packets, exposing the integration-domain confound.
        source = scene["source"]
        control_ids = torch.linspace(0, len(source["positions"])-1, 4096).long()
        x = source["positions"][control_ids].to(scene["device"])
        w = scene["atlas"].directions[0].expand_as(x)
        launch_rows=[]
        for r in RADII:
            for start in (2.1*scene["h"], None):
                _, tau, _ = transmission_prefixes(scene["field"], x, w, scene["boundary"].exit_times(x,w),
                    radius=r*scene["h"], epsilon=scene["h"], path_step=.5*scene["h"],
                    offsets=_micro_offsets(128, scene["device"], x.dtype), counts=(1,micro),
                    eta=scene["eta"], launch_start=start)
                launch_rows.append({"radius_over_h":r, "launch_start":2.1*scene["h"] if start is not None else 1.05*(r+1)*scene["h"],
                    "launch_rule":"fixed" if start is not None else "historical_radius_dependent",
                    "center_tau_mean":float(tau[:,0].mean()), "finite_tau_mean":float(tau[:,-1].mean())})
        report["launch_exclusion_control"] = launch_rows
        ref_full = reference_images(scene, (1080,1920))
        images, full_rows = {}, []
        for label, r, m in (("centerline",1.,1), ("same_radius",1.,micro), ("best_radius",best_radius,micro)):
            family = current if r==1. else torch.load(cache/f"fixed_launch_radius_{r:g}.pt", weights_only=True)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin=time.perf_counter()
            images[label] = readout(scene, family["tau"][:,:,COUNTS.index(m)], (1080,1920))
            torch.cuda.synchronize()
            detector_seconds = time.perf_counter()-begin
            detector_peak = torch.cuda.max_memory_allocated()/2**20
            full_rows.append({"label":label, "radius_over_h":r, "micro":m,
                "quadrature_role":("ZERO_RADIUS_CONTROL" if r==0. else "CENTERLINE_CONTROL" if m==1 else
                    "CONVERGED_AGAINST_128" if converged else "128_REFERENCE_LIMITED"),
                "path_step_over_epsilon":.5, "epsilon_over_h":1., "launch_start_over_h":2.1,
                **_image_metrics(images[label],ref_full), "detector_readout_seconds":detector_seconds,
                "detector_peak_cuda_allocated_mib":detector_peak,
                "peak_cuda_allocated_mib":max(detector_peak, family["peak_cuda_allocated_mib"]),
                "peak_scope":"upper bound from measured all-prefix transport and selected-M detector readout",
                "transport_plus_readout_upper_bound_seconds":family["runtime_seconds"]+detector_seconds,
                "transport_reused_from_screening":True,
                "source_mass":source["source_mass"], "family_transport_seconds":family["runtime_seconds"]})
            progress("fullhd_complete", label=label)
        report["fullhd"] = full_rows
        historical_report = json.loads(Path("artifacts/v0810_stratified_geodesic_polar.json").read_text())
        baseline = next(row["stratified"] for row in historical_report["resolution_fidelity"]
                        if row["resolution"] == [1080, 1920])
        report["historical_fullhd_control"] = {key:{"v0810":baseline[key], "v0811":full_rows[0][key],
            "relative_difference":abs(full_rows[0][key]-baseline[key])/max(abs(baseline[key]), 1e-30)}
            for key in ("whole_image_mse", "foreground_mse", "silhouette_0_8px_mse",
                        "thin_feature_response_ratio", "edge_sharpness_ratio", "source_mass")}
        if max(row["relative_difference"] for row in report["historical_fullhd_control"].values()) > 1e-5:
            raise RuntimeError("historical FullHD control changed; inspect detector/source equivalence")
    control, candidate, best = full_rows
    thin_gain = candidate["thin_feature_response_ratio"]/control["thin_feature_response_ratio"]
    hf_gain = candidate["edge_sharpness_ratio"]/control["edge_sharpness_ratio"]
    improves = [best[k]<.99*control[k] for k in ("whole_image_mse","foreground_mse","silhouette_0_8px_mse")]
    improves += [best[k]>1.01*control[k] for k in ("thin_feature_response_ratio","edge_sharpness_ratio")]
    report["fullhd_improvement_rule"] = "at least three of five fidelity metrics improve by more than 1%; sub-percent/atomic-rounding differences do not count"
    micro_primary = thin_gain>1.1 and hf_gain>1.1 and candidate["whole_image_mse"]<control["whole_image_mse"]
    radius_primary = best_radius>1. and best["thin_feature_response_ratio"]>1.1*candidate["thin_feature_response_ratio"] and best["whole_image_mse"]<candidate["whole_image_mse"]
    micro1 = next(x for x in report["screening"] if x["micro"]==1 and x["radius_over_h"]==1.)
    memory_ratio = max(row["peak_cuda_allocated_mib"] for row in report["family_performance"])/fastest["peak_cuda_allocated_mib"]
    report["full_scale_memory"] = {"million_emitter_to_4096_packet_peak_ratio":memory_ratio,
        "profile_peak_mib":fastest["peak_cuda_allocated_mib"],
        "maximum_family_peak_mib":max(row["peak_cuda_allocated_mib"] for row in report["family_performance"]),
        "matched_chunks":list(chunks), "within_ten_percent":memory_ratio<=1.1}
    toy_active = any(x["kind"]=="filament" and .3<=x["d_over_r"]<=.8 and
                     x["transmission"][0]>.99 and x["transmission"][-1]<.95 for x in pre["toy_rows"])
    report["verdicts"]={"PRODUCTION_TRANSVERSE_MC_COLLAPSED":True,
        "MICRO1_OFFSET_IS_ZERO":report["forensic"]["micro1_zero"],
        "MICRO1_COLLAPSES_TO_CENTERLINE":report["forensic"]["centerline_sample_error"]==0,
        "MICRO1_IS_BIASED_CENTERLINE_ESTIMATOR":micro1["transmission_difference_to_max_micro"]["rms"]>.01,
        "TRANSVERSE_SUPPORT_PHYSICALLY_ACTIVE":toy_active,
        "OFFCENTER_MICRO_SAMPLES_MATTER":any(x["fraction_transmission_change_gt_001"]>.05 for x in report["offcenter_regions"]),
        "TRANSVERSE_MC_CONVERGED":converged, "PATH_QUADRATURE_CONVERGED":report["controls"]["path_converged"] and
            report.get("controls_at_production_radius", report["controls"])["path_converged"],
        "PACKET_RADIUS_TOO_SMALL":radius_primary,
        "TRANSVERSE_PACKET_SUPPORT_NOT_PRIMARY":not micro_primary and not radius_primary,
        "THIN_FEATURE_RECOVERED_BY_MICRO_INTEGRATION":candidate["thin_feature_response_ratio"]>=.75,
        "HIGH_FREQUENCY_RECOVERED_BY_MICRO_INTEGRATION":candidate["edge_sharpness_ratio"]>=.75,
        "FULLHD_TRANSVERSE_INTEGRATION_BETTER":sum(improves)>=3,
        "STREAMING_MICRO_INTEGRATION_MEMORY_BOUNDED":report["streaming_validation"]["memory_bounded"] and memory_ratio<=1.1,
        "NO_DENSE_POINT_BY_LAMBDA_TENSOR":True}
    report.update(MINIMUM_CONVERGED_MICRO_COUNT=micro if converged else None,
        SELECTED_REFERENCE_MICRO_COUNT=micro, BEST_PACKET_RADIUS_OVER_H=best_radius,
        BEST_EPSILON_OVER_RADIUS=None, BEST_PATH_STEP_OVER_EPSILON=min(report["controls"]["best_step"],
            report.get("controls_at_production_radius", report["controls"])["best_step"]),
        PRIMARY_FAILURE=("UNDERINTEGRATED_TRANSVERSE_PACKET_SUPPORT" if micro_primary else
                         "PACKET_SUPPORT_RADIUS_TOO_SMALL" if radius_primary else "TRANSVERSE_PACKET_SUPPORT_NOT_PRIMARY"))
    report["limitations"]=["M=128 is a finite reference; convergence can fail.",
        "The preserved 3D-ball rule includes longitudinal as well as perpendicular offsets; this is not a pure transverse-disk operator.",
        "Path and shell controls use 4096 spatially distributed emitters and all four views (16384 packets).",
        "No optimal epsilon inferred from attenuation-only shell control; BEST_EPSILON_OVER_RADIUS is null.",
        "Primary radius and shell controls fix start=2.1h; the historical radius-dependent start is measured separately and is not used to attribute support gains.",
        "FullHD reuses exact cached scene optical depths; detector cost and shared family transport cost reported separately.",
        "High-frequency metric is the historical edge-sharpness proxy, not an isolated frequency transfer function."]
    report["runtime_seconds"]=time.perf_counter()-started
    report["runtime_scope"] = "wall time of this invocation; cached transport is not charged again"
    report["sum_measured_family_transport_seconds"] = sum(row["runtime_seconds"]
        for row in report["family_performance"])
    previous_output = Path("artifacts/v0811_transverse_packet_integration.json")
    if previous_output.exists():
        previous = json.loads(previous_output.read_text())
        report["prior_complete_invocation_runtime_seconds"] = previous.get(
            "prior_complete_invocation_runtime_seconds", previous["runtime_seconds"])
    report["figures"] = make_figures(report, images)
    output=Path("artifacts/v0811_transverse_packet_integration.json")
    write_json(output, report)
    _write_csv(Path("artifacts/v0811_transverse_packet_integration.csv"), _scalar_csv_rows(report))
    progress("complete", verdicts=report["verdicts"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        report = preflight(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        write_json(Path("artifacts/v0811_preflight.json"), report)
        print(max(row["tau_error"] for row in report["streaming_equivalence"]))
    else:
        run_experiment(profile_only=args.profile)
