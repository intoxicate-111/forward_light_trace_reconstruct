"""Piecewise-smooth geometry-to-image Jacobian analysis for v0.2.

Visibility, absorption, first-arrival ownership, and footprint indices define a
fixed transport cell.  Derivatives within that cell are smooth; changes to any
of those discrete choices are measured separately as transport events.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .camera import PlanarCamera
from .fields import (
    LocalBasisField,
    ZeroSetField,
    deform_reference_surface,
    implicit_position_jacobian,
    unit_normals,
)
from .tracer import PhotonBatch, trace_photons


Tensor = torch.Tensor


@dataclass
class FixedTransportCell:
    camera: PlanarCamera
    packets_per_emitter: int
    cone_power: float
    normal_mode: str
    emitter_colors: Tensor
    photon_ids: Tensor
    emitter_ids: Tensor
    fixed_directions: Tensor
    footprint_pixels: Tensor
    footprint_valid: Tensor
    footprint_base_xy: Tensor
    photon_state: Tensor
    owner_map: Tensor


@dataclass
class GeometryJacobian:
    image: Tensor
    matrix: Tensor
    sparse_matrix: Tensor
    position_jacobian: Tensor
    stable_emitters: Tensor
    cell: FixedTransportCell


def make_local_basis_field(
    base: ZeroSetField,
    parameter_count: int,
    support_radius: float,
    *,
    seed: int = 101,
) -> LocalBasisField:
    generator = torch.Generator().manual_seed(seed)
    centers = base.sample_surface(parameter_count, generator)
    radii = torch.full((parameter_count,), support_radius, dtype=torch.float64)
    coefficients = torch.zeros(parameter_count, dtype=torch.float64)
    return LocalBasisField(base, centers, radii, coefficients)


def _tangent_basis(normals: Tensor) -> tuple[Tensor, Tensor]:
    z_axis = torch.tensor(
        [0.0, 0.0, 1.0], dtype=normals.dtype, device=normals.device
    )
    y_axis = torch.tensor(
        [0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device
    )
    helper = torch.where((normals[:, 2].abs() > 0.9)[:, None], y_axis, z_axis)
    tangent = torch.linalg.cross(helper, normals, dim=-1)
    tangent = tangent / torch.linalg.vector_norm(
        tangent, dim=-1, keepdim=True
    ).clamp_min(1e-15)
    return tangent, torch.linalg.cross(normals, tangent, dim=-1)


def deterministic_directions(
    normals: Tensor, packets_per_emitter: int, cone_power: float
) -> Tensor:
    """Reusable low-discrepancy cone directions, ordered emitter-major."""
    emitters = normals.shape[0]
    packet = torch.arange(
        packets_per_emitter, dtype=normals.dtype, device=normals.device
    )
    emitter = torch.arange(
        emitters, dtype=normals.dtype, device=normals.device
    )[:, None]
    u = (packet + 0.5) / packets_per_emitter
    cos_theta = u.pow(1.0 / (cone_power + 1.0)).expand(emitters, -1)
    sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp_min(0.0))
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    phase = torch.frac(packet[None, :] / golden_ratio + emitter / golden_ratio**2)
    azimuth = 2.0 * torch.pi * phase
    tangent, bitangent = _tangent_basis(normals)
    directions = (
        cos_theta[..., None] * normals[:, None, :]
        + (sin_theta * torch.cos(azimuth))[..., None] * tangent[:, None, :]
        + (sin_theta * torch.sin(azimuth))[..., None] * bitangent[:, None, :]
    )
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    )
    return directions.reshape(-1, 3)


def _photon_batch(
    points: Tensor, directions: Tensor, colors: Tensor, packets_per_emitter: int
) -> PhotonBatch:
    emitters = points.shape[0]
    emitter_ids = torch.arange(
        emitters, device=points.device
    ).repeat_interleave(packets_per_emitter)
    count = emitter_ids.numel()
    return PhotonBatch(
        origins=points[emitter_ids],
        directions=directions,
        colors=colors[emitter_ids],
        energies=torch.full(
            (count,),
            1.0 / packets_per_emitter,
            dtype=points.dtype,
            device=points.device,
        ),
        emit_times=torch.zeros(count, dtype=points.dtype, device=points.device),
        emitter_ids=emitter_ids,
    )


def _owners(trace_pixels: Tensor, arrivals: Tensor, photon_ids: Tensor, pixels: int) -> Tensor:
    missing = torch.iinfo(torch.long).max
    owners = torch.full(
        (pixels,), missing, dtype=torch.long, device=trace_pixels.device
    )
    minimum_times = torch.full(
        (pixels,), torch.inf, dtype=arrivals.dtype, device=arrivals.device
    )
    minimum_times.scatter_reduce_(0, trace_pixels, arrivals, reduce="amin")
    winners = arrivals == minimum_times[trace_pixels]
    owners.scatter_reduce_(
        0, trace_pixels[winners], photon_ids[winners], reduce="amin"
    )
    owners[owners == missing] = -1
    return owners


def _continuous_sensor(
    camera: PlanarCamera, origins: Tensor, directions: Tensor
) -> tuple[Tensor, Tensor]:
    denominator = directions @ camera.normal
    times = ((camera.center - origins) * camera.normal).sum(dim=-1) / denominator
    hits = origins + times[:, None] * directions
    relative = hits - camera.center
    local_right = relative @ camera.right
    local_up = relative @ camera.up
    rows, columns = camera.resolution
    x = (local_right / camera.width + 0.5) * columns - 0.5
    y = (0.5 - local_up / camera.height) * rows - 0.5
    return torch.stack((x, y), dim=-1), times


def _footprint(
    sensor_xy: Tensor, resolution: tuple[int, int]
) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = resolution
    base = torch.floor(sensor_xy).to(torch.long)
    x0, y0 = base.unbind(dim=-1)
    pixels_xy = torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x0 + 1, y0), dim=-1),
            torch.stack((x0, y0 + 1), dim=-1),
            torch.stack((x0 + 1, y0 + 1), dim=-1),
        ),
        dim=1,
    )
    valid = (
        (pixels_xy[..., 0] >= 0)
        & (pixels_xy[..., 0] < columns)
        & (pixels_xy[..., 1] >= 0)
        & (pixels_xy[..., 1] < rows)
    )
    pixels = pixels_xy[..., 1] * columns + pixels_xy[..., 0]
    pixels = torch.where(valid, pixels, torch.zeros_like(pixels))
    return pixels, valid, base


def _topology(
    field: ZeroSetField,
    camera: PlanarCamera,
    points: Tensor,
    directions: Tensor,
    colors: Tensor,
    packets_per_emitter: int,
    root_samples: int,
) -> tuple[Tensor, Tensor, Tensor]:
    photons = _photon_batch(points, directions, colors, packets_per_emitter)
    valid, _, camera_pixels, _ = camera.intersect(photons.origins, directions)
    state = torch.full(
        (photons.count,), -1, dtype=torch.long, device=points.device
    )
    state[valid] = camera_pixels[valid]
    trace = trace_photons(field, camera, photons, root_samples=root_samples)
    candidate = torch.nonzero(valid, as_tuple=False).flatten()
    survived = torch.zeros(
        photons.count, dtype=torch.bool, device=points.device
    )
    survived[trace.photon_ids] = True
    state[candidate[~survived[candidate]]] = -2
    owners = _owners(
        trace.pixels, trace.arrival_times, trace.photon_ids, camera.pixel_count
    )
    return state, owners, trace.photon_ids


def build_fixed_transport_cell(
    field: LocalBasisField,
    camera: PlanarCamera,
    points: Tensor,
    reference_normals: Tensor,
    colors: Tensor,
    *,
    packets_per_emitter: int = 8,
    cone_power: float = 32.0,
    normal_mode: str = "reference",
    root_samples: int = 64,
) -> FixedTransportCell:
    if normal_mode not in {"reference", "current"}:
        raise ValueError("normal_mode must be 'reference' or 'current'")
    orientation = (
        reference_normals if normal_mode == "reference" else unit_normals(field, points)
    )
    directions = deterministic_directions(orientation, packets_per_emitter, cone_power)
    state, owners, _ = _topology(
        field,
        camera,
        points,
        directions,
        colors,
        packets_per_emitter,
        root_samples,
    )
    photon_ids = owners[owners >= 0]
    emitter_ids = torch.div(photon_ids, packets_per_emitter, rounding_mode="floor")
    selected_directions = directions[photon_ids]
    sensor_xy, _ = _continuous_sensor(
        camera, points[emitter_ids], selected_directions
    )
    footprint_pixels, footprint_valid, footprint_base = _footprint(
        sensor_xy, camera.resolution
    )
    return FixedTransportCell(
        camera,
        packets_per_emitter,
        cone_power,
        normal_mode,
        colors,
        photon_ids,
        emitter_ids,
        selected_directions,
        footprint_pixels,
        footprint_valid,
        footprint_base,
        state,
        owners,
    )


def render_fixed_transport_cell(
    cell: FixedTransportCell,
    points: Tensor,
    orientation_normals: Tensor | None = None,
) -> Tensor:
    """Bilinearly splat fixed first-arrival owners to a four-pixel footprint."""
    if orientation_normals is None:
        directions = cell.fixed_directions
    else:
        all_directions = deterministic_directions(
            orientation_normals, cell.packets_per_emitter, cell.cone_power
        )
        directions = all_directions[cell.photon_ids]
    sensor_xy, _ = _continuous_sensor(
        cell.camera, points[cell.emitter_ids], directions
    )
    fractional = sensor_xy - cell.footprint_base_xy.to(sensor_xy.dtype)
    dx, dy = fractional.unbind(dim=-1)
    weights = torch.stack(
        ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy),
        dim=-1,
    )
    weights = weights * cell.footprint_valid
    image = torch.zeros(
        (cell.camera.pixel_count, 3), dtype=points.dtype, device=points.device
    )
    contributions = weights[..., None] * cell.emitter_colors[cell.emitter_ids, None, :]
    return image.index_add(
        0, cell.footprint_pixels.reshape(-1), contributions.reshape(-1, 3)
    )


def sparse_bilinear_transport(
    cell: FixedTransportCell,
    points: Tensor,
) -> tuple[Tensor, Tensor]:
    """Construct local detector transport directly as sparse COO."""
    sensor_xy, _ = _continuous_sensor(
        cell.camera, points[cell.emitter_ids], cell.fixed_directions
    )
    fractional = sensor_xy - cell.footprint_base_xy.to(sensor_xy.dtype)
    dx, dy = fractional.unbind(dim=-1)
    weights = torch.stack(
        ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy),
        dim=-1,
    )
    valid = cell.footprint_valid & (weights != 0.0)
    rows = cell.footprint_pixels[valid]
    columns = cell.emitter_ids[:, None].expand_as(weights)[valid]
    transport = torch.sparse_coo_tensor(
        torch.stack((rows, columns)),
        weights[valid],
        size=(cell.camera.pixel_count, cell.emitter_colors.shape[0]),
        dtype=points.dtype,
        device=points.device,
    ).coalesce()
    image = torch.sparse.mm(transport, cell.emitter_colors)
    return transport, image


def sparse_geometry_image_jacobian(
    field: LocalBasisField,
    points: Tensor,
    reference_normals: Tensor,
    cell: FixedTransportCell,
    *,
    threshold: float = 1e-12,
) -> Tensor:
    """Build the RGB geometry Jacobian from touched local entries only.

    The largest candidate tensor is proportional to
    ``winning_photons * 4 footprints * K * 3 channels``; no tensor has a
    ``3 * image_pixels * K`` shape.  v0.2.1 keeps the validated primary
    reference-direction transport semantics for this scaling path.
    """
    if cell.normal_mode != "reference":
        raise ValueError("sparse scaling Jacobian requires reference normal mode")
    dp_dlambda, stable, _ = implicit_position_jacobian(
        field, points, reference_normals
    )
    selected_dp = dp_dlambda[cell.emitter_ids]
    selected_dp = torch.where(
        stable[cell.emitter_ids][:, None, None],
        selected_dp,
        torch.zeros_like(selected_dp),
    )

    directions = cell.fixed_directions
    denominator = directions @ cell.camera.normal
    right_gradient = (
        cell.camera.right
        - (directions @ cell.camera.right)[:, None]
        * cell.camera.normal
        / denominator[:, None]
    )
    up_gradient = (
        cell.camera.up
        - (directions @ cell.camera.up)[:, None]
        * cell.camera.normal
        / denominator[:, None]
    )
    rows, columns = cell.camera.resolution
    sensor_gradient = torch.stack(
        (
            columns / cell.camera.width * right_gradient,
            -rows / cell.camera.height * up_gradient,
        ),
        dim=1,
    )
    dxy_dlambda = torch.einsum("qac,qkc->qka", sensor_gradient, selected_dp)

    sensor_xy, _ = _continuous_sensor(
        cell.camera, points[cell.emitter_ids], directions
    )
    fractional = sensor_xy - cell.footprint_base_xy.to(sensor_xy.dtype)
    dx, dy = fractional.unbind(dim=-1)
    weight_gradient = torch.stack(
        (
            torch.stack((-(1.0 - dy), -(1.0 - dx)), dim=-1),
            torch.stack((1.0 - dy, -dx), dim=-1),
            torch.stack((-dy, 1.0 - dx), dim=-1),
            torch.stack((dy, dx), dim=-1),
        ),
        dim=1,
    )
    dweight_dlambda = torch.einsum(
        "qfa,qka->qfk", weight_gradient, dxy_dlambda
    )
    values = (
        dweight_dlambda[..., None]
        * cell.emitter_colors[cell.emitter_ids, None, None, :]
    )
    parameter_count = field.parameter_count
    channels = torch.arange(3, device=points.device)
    output_rows = (
        cell.footprint_pixels[:, :, None, None] * 3
        + channels[None, None, None, :]
    ).expand(-1, -1, parameter_count, -1)
    parameter_columns = torch.arange(
        parameter_count, device=points.device
    )[None, None, :, None].expand_as(output_rows)
    touched = cell.footprint_valid[:, :, None, None] & (values != 0.0)
    sparse = torch.sparse_coo_tensor(
        torch.stack((output_rows[touched], parameter_columns[touched])),
        values[touched],
        size=(3 * cell.camera.pixel_count, parameter_count),
        dtype=points.dtype,
        device=points.device,
    ).coalesce()
    if threshold > 0.0:
        keep = sparse.values().abs() > threshold
        sparse = torch.sparse_coo_tensor(
            sparse.indices()[:, keep],
            sparse.values()[keep],
            size=sparse.shape,
            dtype=points.dtype,
            device=points.device,
        ).coalesce()
    return sparse


def geometry_image_jacobian(
    field: LocalBasisField,
    points: Tensor,
    reference_normals: Tensor,
    cell: FixedTransportCell,
) -> GeometryJacobian:
    """Chain the explicit implicit-root derivative through fixed-cell transport."""
    dp_dlambda, stable, _ = implicit_position_jacobian(
        field, points, reference_normals
    )
    zero = torch.zeros(
        field.parameter_count, dtype=points.dtype, device=points.device
    )

    def image_from_delta(delta: Tensor) -> Tensor:
        linear_points = points + torch.einsum("ekd,k->ed", dp_dlambda, delta)
        orientation = None
        if cell.normal_mode == "current":
            perturbed_field = field.with_coefficients(field.coefficients + delta)
            orientation = unit_normals(perturbed_field, linear_points)
        return render_fixed_transport_cell(cell, linear_points, orientation).reshape(-1)

    matrix = torch.autograd.functional.jacobian(
        image_from_delta,
        zero,
        vectorize=True,
        strategy="forward-mode",
    )
    image = image_from_delta(zero).reshape(cell.camera.pixel_count, 3)
    sparse_mask = matrix.abs() > 1e-12
    sparse_indices = torch.nonzero(sparse_mask, as_tuple=False).T
    sparse_matrix = torch.sparse_coo_tensor(
        sparse_indices,
        matrix[sparse_mask],
        size=matrix.shape,
        dtype=matrix.dtype,
    ).coalesce()
    return GeometryJacobian(
        image, matrix, sparse_matrix, dp_dlambda, stable, cell
    )


def exact_deformation(
    field: LocalBasisField,
    reference_points: Tensor,
    reference_normals: Tensor,
    coefficients: Tensor,
) -> tuple[LocalBasisField, Tensor, Tensor]:
    perturbed = field.with_coefficients(coefficients)
    points, success, _ = deform_reference_surface(
        perturbed, reference_points, reference_normals
    )
    if not bool(success.all()):
        raise RuntimeError(
            f"normal-line root failure fraction: {float((~success).double().mean()):.6f}"
        )
    return perturbed, points, success


def topology_event_pixels(
    field: LocalBasisField,
    reference_points: Tensor,
    reference_normals: Tensor,
    colors: Tensor,
    cell: FixedTransportCell,
    coefficients: Tensor,
    *,
    root_samples: int = 64,
) -> Tensor:
    perturbed, points, _ = exact_deformation(
        field, reference_points, reference_normals, coefficients
    )
    orientation = (
        reference_normals
        if cell.normal_mode == "reference"
        else unit_normals(perturbed, points)
    )
    directions = deterministic_directions(
        orientation, cell.packets_per_emitter, cell.cone_power
    )
    state, owners, _ = _topology(
        perturbed,
        cell.camera,
        points,
        directions,
        colors,
        cell.packets_per_emitter,
        root_samples,
    )
    changed_owner = owners != cell.owner_map
    changed_photons = torch.nonzero(state != cell.photon_state, as_tuple=False).flatten()
    event_pixels = changed_owner.clone()
    old_pixels = cell.photon_state[changed_photons]
    new_pixels = state[changed_photons]
    event_pixels[old_pixels[old_pixels >= 0]] = True
    event_pixels[new_pixels[new_pixels >= 0]] = True
    selected_directions = directions[cell.photon_ids]
    sensor_xy, _ = _continuous_sensor(
        cell.camera, points[cell.emitter_ids], selected_directions
    )
    new_footprint, new_valid, new_base = _footprint(
        sensor_xy, cell.camera.resolution
    )
    changed_footprint = (new_base != cell.footprint_base_xy).any(dim=-1)
    for owner in torch.nonzero(changed_footprint, as_tuple=False).flatten().tolist():
        old_support = cell.footprint_pixels[owner][cell.footprint_valid[owner]]
        new_support = new_footprint[owner][new_valid[owner]]
        event_pixels[old_support] = True
        event_pixels[new_support] = True
    return event_pixels


def finite_difference_report(
    field: LocalBasisField,
    reference_points: Tensor,
    reference_normals: Tensor,
    colors: Tensor,
    result: GeometryJacobian,
    *,
    parameter_ids: Tensor,
    steps: tuple[float, ...] = (1e-3, 3e-4, 1e-4, 3e-5),
) -> dict[str, object]:
    errors: list[float] = []
    best_steps: list[float] = []
    event_fractions: list[float] = []
    maximum_event_counts: list[int] = []
    for parameter in parameter_ids.tolist():
        analytic = result.matrix[:, parameter].reshape(-1, 3)
        best_error = torch.inf
        best_step = steps[0]
        best_events = 0.0
        maximum_events = 0
        for step in steps:
            plus = field.coefficients.clone()
            minus = field.coefficients.clone()
            plus[parameter] += step
            minus[parameter] -= step
            plus_field, plus_points, _ = exact_deformation(
                field, reference_points, reference_normals, plus
            )
            minus_field, minus_points, _ = exact_deformation(
                field, reference_points, reference_normals, minus
            )
            plus_orientation = minus_orientation = None
            if result.cell.normal_mode == "current":
                plus_orientation = unit_normals(plus_field, plus_points)
                minus_orientation = unit_normals(minus_field, minus_points)
            plus_image = render_fixed_transport_cell(
                result.cell, plus_points, plus_orientation
            )
            minus_image = render_fixed_transport_cell(
                result.cell, minus_points, minus_orientation
            )
            finite_difference = (plus_image - minus_image) / (2.0 * step)
            events = topology_event_pixels(
                field,
                reference_points,
                reference_normals,
                colors,
                result.cell,
                plus,
            ) | topology_event_pixels(
                field,
                reference_points,
                reference_normals,
                colors,
                result.cell,
                minus,
            )
            stable = ~events
            maximum_events = max(maximum_events, int(events.sum()))
            numerator = torch.linalg.vector_norm((finite_difference - analytic)[stable])
            denominator = torch.linalg.vector_norm(analytic[stable]).clamp_min(1e-15)
            relative_error = numerator / denominator
            if relative_error < best_error:
                best_error = relative_error
                best_step = step
                best_events = float(events.double().mean())
        errors.append(float(best_error))
        best_steps.append(best_step)
        event_fractions.append(best_events)
        maximum_event_counts.append(maximum_events)
    error_tensor = torch.tensor(errors, dtype=torch.float64)
    return {
        "parameters_tested": parameter_ids.tolist(),
        "median_relative_error": float(error_tensor.median()),
        "max_stable_relative_error": float(error_tensor.max()),
        "best_steps": best_steps,
        "transport_event_pixel_fraction": float(
            torch.tensor(event_fractions).mean()
        ),
        "maximum_tested_transport_event_pixels": max(maximum_event_counts),
        "maximum_tested_transport_event_pixel_fraction": (
            max(maximum_event_counts) / result.cell.camera.pixel_count
        ),
    }


def support_report(
    field: LocalBasisField,
    points: Tensor,
    result: GeometryJacobian,
    *,
    absolute_thresholds: tuple[float, ...] = (1e-12, 1e-9, 1e-6, 1e-4),
) -> dict[str, object]:
    pixels = result.cell.camera.pixel_count
    parameters = field.parameter_count
    response = torch.linalg.vector_norm(
        result.matrix.reshape(pixels, 3, parameters), dim=1
    )
    summaries: dict[str, object] = {}
    for threshold in absolute_thresholds:
        active = response > threshold
        nnz = int(active.sum())
        per_parameter = active.sum(dim=0).to(torch.float64)
        per_pixel = active.sum(dim=1).to(torch.float64)
        summaries[f"absolute_{threshold:g}"] = {
            "nnz_pixel_parameter": nnz,
            "density": nnz / active.numel(),
            "mean_affected_pixels_per_parameter": float(per_parameter.mean()),
            "median_affected_pixels_per_parameter": float(per_parameter.median()),
            "mean_active_parameters_per_pixel": float(per_pixel.mean()),
        }
    column_max = response.max(dim=0).values
    relative_active = response > (1e-3 * column_max)[None, :]
    relative_active[:, column_max == 0.0] = False
    summaries["relative_1e-3"] = {
        "nnz_pixel_parameter": int(relative_active.sum()),
        "density": float(relative_active.double().mean()),
    }

    emitter_support = field.basis_values(points) > 0.0
    predicted = torch.zeros(
        (pixels, parameters), dtype=torch.bool, device=points.device
    )
    for owner, emitter in enumerate(result.cell.emitter_ids.tolist()):
        footprint = result.cell.footprint_pixels[owner][result.cell.footprint_valid[owner]]
        supported_parameters = torch.nonzero(
            emitter_support[emitter], as_tuple=False
        ).flatten()
        if footprint.numel() and supported_parameters.numel():
            predicted[footprint[:, None], supported_parameters[None, :]] = True
    actual = response > 1e-12
    intersection = predicted & actual
    precision = intersection.sum(dim=0) / predicted.sum(dim=0).clamp_min(1)
    recall = intersection.sum(dim=0) / actual.sum(dim=0).clamp_min(1)
    areas = actual.sum(dim=0).to(torch.float64)
    responsive = actual.any(dim=0)
    predicted_nonempty = predicted.any(dim=0)
    metric_columns = responsive | predicted_nonempty
    rows, columns = result.cell.camera.resolution
    parameter_details: list[dict[str, object]] = []
    for parameter in range(parameters):
        active_pixels = torch.nonzero(actual[:, parameter], as_tuple=False).flatten()
        if active_pixels.numel():
            active_rows = torch.div(active_pixels, columns, rounding_mode="floor")
            active_columns = active_pixels % columns
            bounding_box: list[int] | None = [
                int(active_columns.min()),
                int(active_rows.min()),
                int(active_columns.max()),
                int(active_rows.max()),
            ]
        else:
            bounding_box = None
        parameter_details.append(
            {
                "parameter": parameter,
                "affected_emitters": int(emitter_support[:, parameter].sum()),
                "affected_pixels": int(active_pixels.numel()),
                "maximum_image_response": float(response[:, parameter].max()),
                "jacobian_energy": float((result.matrix[:, parameter] ** 2).sum()),
                "image_support_bbox_xyxy": bounding_box,
            }
        )
    summaries["locality"] = {
        "responsive_parameters": int(responsive.sum()),
        "mean_support_precision": float(
            precision[metric_columns].to(torch.float64).mean()
        ),
        "mean_support_recall": float(recall[responsive].to(torch.float64).mean()),
        "mean_image_support_area": float(areas.mean()),
        "median_image_support_area": float(areas.median()),
        "affected_emitters_per_parameter": emitter_support.sum(dim=0).tolist(),
        "parameter_details": parameter_details,
    }
    return summaries


def locality_perturbation_report(
    field: LocalBasisField,
    reference_points: Tensor,
    reference_normals: Tensor,
    result: GeometryJacobian,
    parameter_ids: Tensor,
    *,
    delta: float = 1e-3,
) -> dict[str, float]:
    baseline = result.image
    response = torch.linalg.vector_norm(
        result.matrix.reshape(result.cell.camera.pixel_count, 3, -1), dim=1
    )
    ratios: list[float] = []
    for parameter in parameter_ids.tolist():
        coefficients = field.coefficients.clone()
        coefficients[parameter] = delta
        perturbed_field, points, _ = exact_deformation(
            field, reference_points, reference_normals, coefficients
        )
        orientation = None
        if result.cell.normal_mode == "current":
            orientation = unit_normals(perturbed_field, points)
        change = render_fixed_transport_cell(result.cell, points, orientation) - baseline
        predicted = response[:, parameter] > 1e-12
        local_energy = (change[predicted] ** 2).sum()
        total_energy = (change**2).sum().clamp_min(1e-30)
        ratios.append(float(torch.sqrt(local_energy / total_energy)))
    values = torch.tensor(ratios, dtype=torch.float64)
    return {
        "mean_local_energy_norm_fraction": float(values.mean()),
        "median_local_energy_norm_fraction": float(values.median()),
        "minimum_local_energy_norm_fraction": float(values.min()),
    }


def observability_report(
    jacobians: list[Tensor], view_counts: tuple[int, ...] = (1, 2, 4, 8)
) -> dict[str, object]:
    one_view_scale = float(torch.linalg.svdvals(jacobians[0]).max())
    threshold = one_view_scale * 1e-6
    report: dict[str, object] = {"singular_value_threshold": threshold}
    for view_count in view_counts:
        stacked = torch.cat(jacobians[:view_count], dim=0)
        singular_values = torch.linalg.svdvals(stacked)
        significant = singular_values > threshold
        rank = int(significant.sum())
        condition = (
            float(singular_values[significant].max() / singular_values[significant].min())
            if rank
            else torch.inf
        )
        report[str(view_count)] = {
            "shape": list(stacked.shape),
            "nnz_at_1e-12": int((stacked.abs() > 1e-12).sum()),
            "density_at_1e-12": float((stacked.abs() > 1e-12).double().mean()),
            "rank": rank,
            "near_null_directions": stacked.shape[1] - rank,
            "condition_number": condition,
            "singular_values": singular_values.tolist(),
        }

    _, _, right_vectors = torch.linalg.svd(jacobians[0], full_matrices=False)
    weak_direction = right_vectors[-1]
    responses = [float(torch.linalg.vector_norm(matrix @ weak_direction)) for matrix in jacobians]
    report["camera_1_weak_mode_response_by_view"] = responses
    return report
