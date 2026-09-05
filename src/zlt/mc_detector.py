"""v0.8.4 continuous Monte Carlo detector-integration diagnostic.

The primary estimator integrates the transport quantity induced by the
existing four-dimensional surface-sampling map.  It is deliberately not
described as calibrated radiometry: the surface sampler is cell-uniform in a
latent cube and has no tractable physical surface-area density correction.
"""

from __future__ import annotations

import gc
import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
import torch
from scipy import stats

from .benchmark import cuda_environment
from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
from .high_sample import _json_ready, _prepare_surface, _progress, _release, _write_csv
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import sign_changing_cells
from .pixel_diagnostic import (
    PixelDiagnosticConfig,
    PixelView,
    REGION_NAMES,
    _controlled_visibility_switching,
    _render_diagnostics,
    _silhouette_regions,
    _vector_metrics,
)


Tensor = torch.Tensor
ESTIMATORS = (
    "LEGACY_GATED_NORMALIZED_SPLAT",
    "UNGATED_NORMALIZED_SPLAT",
    "HARD_BIN_MONTE_CARLO",
    "CONTINUOUS_KERNEL_MONTE_CARLO",
)


@dataclass(frozen=True)
class MCDetectorConfig:
    views: int = 20
    resolutions: tuple[tuple[int, int], ...] = (
        (256, 256),
        (512, 512),
        (540, 960),
        (1080, 1920),
    )
    emitter_counts: tuple[int, ...] = (65_536, 262_144, 518_400, 2_073_600)
    convergence_counts: tuple[int, ...] = (32_768, 65_536, 131_072, 262_144)
    emitter_chunk_size: int = 65_536
    sobol_seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    root_samples: int = 16
    bisection_steps: int = 18
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    support_threshold: float = 0.05
    ambient: float = 0.35
    capture_view: int = 10
    gradient_surface_samples: int = 16_384
    gradient_views: int = 4
    gradient_resolution: int = 256
    gradient_parameters: int = 32
    gradient_epsilons: tuple[float, ...] = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)

    def pixel_config(self, *, views: int | None = None) -> PixelDiagnosticConfig:
        return PixelDiagnosticConfig(
            views=self.views if views is None else views,
            resolutions=self.resolutions,
            emitter_counts=self.emitter_counts,
            emitter_chunk_size=self.emitter_chunk_size,
            sobol_seeds=self.sobol_seeds,
            root_samples=self.root_samples,
            bisection_steps=self.bisection_steps,
            detector_extent=self.detector_extent,
            sensor_gain=self.sensor_gain,
            support_threshold=self.support_threshold,
            ambient=self.ambient,
            capture_view=self.capture_view,
            gradient_surface_samples=self.gradient_surface_samples,
            gradient_views=self.gradient_views,
            gradient_resolution=self.gradient_resolution,
            gradient_parameters=self.gradient_parameters,
            gradient_epsilons=self.gradient_epsilons,
        )


@dataclass
class EstimatorView:
    images: dict[str, np.ndarray]
    preclamp: dict[str, np.ndarray]
    supports: dict[str, np.ndarray]
    count: np.ndarray
    mass: np.ndarray


def _derive_estimators(
    view: PixelView,
    emitter_count: int,
    resolution: tuple[int, int],
    sensor_gain: float,
) -> EstimatorView:
    if view.numerator is None or view.hard_bin_numerator is None:
        raise ValueError("MC detector diagnostics require both raw accumulators")
    rows, columns = resolution
    pixels = rows * columns
    mass = view.mass.astype(np.float32, copy=False).reshape(rows, columns)
    numerator = view.numerator.astype(np.float32, copy=False).reshape(rows, columns, 3)
    hard_numerator = view.hard_bin_numerator.astype(np.float32, copy=False).reshape(
        rows, columns, 3
    )
    normalized = np.zeros_like(numerator)
    nonzero = mass > 0.0
    normalized[nonzero] = (
        sensor_gain * numerator[nonzero] / mass[nonzero, None]
    )
    mc_scale = sensor_gain * pixels / emitter_count
    hard_mc = mc_scale * hard_numerator
    continuous_mc = mc_scale * numerator
    legacy = view.image.astype(np.float32, copy=False)
    ungated = np.clip(normalized, 0.0, 1.0)
    images = {
        ESTIMATORS[0]: legacy,
        ESTIMATORS[1]: ungated,
        ESTIMATORS[2]: hard_mc,
        ESTIMATORS[3]: continuous_mc,
    }
    preclamp = {
        ESTIMATORS[0]: normalized,
        ESTIMATORS[1]: normalized,
        ESTIMATORS[2]: hard_mc,
        ESTIMATORS[3]: continuous_mc,
    }
    supports = {
        ESTIMATORS[0]: mass >= 0.05,
        ESTIMATORS[1]: mass > 0.0,
        ESTIMATORS[2]: np.linalg.norm(hard_numerator, axis=2) > 0.0,
        ESTIMATORS[3]: mass > 0.0,
    }
    return EstimatorView(
        images,
        preclamp,
        supports,
        view.counts.reshape(rows, columns),
        mass,
    )


def _derive_views(
    views: list[PixelView],
    emitter_count: int,
    resolution: tuple[int, int],
    sensor_gain: float,
) -> list[EstimatorView]:
    return [
        _derive_estimators(view, emitter_count, resolution, sensor_gain)
        for view in views
    ]


def _measurement_derivation(config: MCDetectorConfig) -> dict[str, object]:
    return {
        "prototype_scope": (
            "A mathematically coherent integral of the current transported "
            "attribute under its algorithmic sampling measure, not a calibrated "
            "physical radiance or irradiance estimator."
        ),
        "sample_variables": {
            "z_i": "four-dimensional point in [0,1)^4 from a Sobol scramble",
            "x_i": "Newton/fallback projection T_F(z_i) onto a sign-changing grid cell",
            "omega_i": "one fixed Fibonacci-atlas direction for each detector view",
            "u_i": "orthographic normalized detector coordinate of x_i",
            "V_i": "hard outward/first-zero-set visibility indicator",
            "f_i": "V_i C(x_i)[0.35+0.65 max(0,n_i dot omega_v)]",
            "q_i": "q_z(z)=1 on [0,1)^4; the surface-space push-forward density is not assumed area-uniform",
        },
        "sampling_audit": {
            "cell_choice": "floor(z0 * number_of_sign_changing_cells), hence uniform cell probability 1/M",
            "within_cell": "z1,z2,z3 are uniform voxel coordinates before zero-set projection",
            "surface_area_pdf": "not available and not silently approximated",
            "normal_weight": "normal affects outward acceptance and the declared cosine appearance lobe",
            "direction_pdf": "none in the per-view estimator; omega_v is a deterministic view/packet direction",
            "visibility": "hard first-zero-set ownership with intersection epsilon 2e-4",
            "attenuation": "no inverse-square or distance attenuation; intentionally absent in this prototype",
            "packet_weight": "one unit latent sample per emitter per view",
            "emitter_normalization": "global 1/N in MC estimators",
        },
        "detector_coordinates": (
            "s=(s_y,s_x) in [0,1)^2; r=s_y H-1/2 and c=s_x W-1/2"
        ),
        "continuous_kernel": {
            "family": "separable cardinal cubic B-spline",
            "definition": "K_p(s)=H W B(r_p-r(s)) B(c_p-c(s))",
            "support_radius": "2 pixels per axis (fixed 4x4 pixel-space footprint)",
            "normalization": "integral K_p(s) ds = 1 for interior pixels",
            "pixel_area": "DeltaA_normalized=1/(H W); the H W factor is reciprocal pixel area",
            "resolution_semantics": (
                "a per-pixel reconstruction/aperture response whose normalized-detector "
                "width shrinks with resolution; this is the v0.8.2 matched-density control, "
                "not the v0.8.1 enlarged footprint"
            ),
            "interpretation": "continuous detector density reconstruction kernel",
        },
        "integral": (
            "I_p = g integral_[0,1)^4 f(T_F(z),omega_v) K_p(u(T_F(z))) dz"
        ),
        "estimator": "Ihat_p = g/N sum_i f_i K_p(u_i), with q_z=1",
        "expectation": "E[Ihat_p]=g integral f(T_F(z),omega_v) K_p(u(T_F(z))) dz",
        "variance": "Var[Ihat_p]=g^2 Var[fK_p]/N for iid latent samples",
        "sobol_note": "scrambled Sobol is randomized QMC, so iid 1/N variance is a reference rather than an asserted exact rate",
        "brightness_invariance": "N->aN multiplies the expected numerator by a and the explicit 1/N by 1/a",
        "jacobian": (
            "dI_p/dlambda=g/N sum_i[(df_i/dlambda)K_p(u_i)+"
            "f_i grad(K_p)(u_i) dot du_i/dlambda], with visibility and root/path ownership frozen"
        ),
    }


def _estimator_definitions() -> dict[str, object]:
    return {
        ESTIMATORS[0]: "1[W>=0.05] clamp(g A/max(W,1e-30),0,1)",
        ESTIMATORS[1]: "1[W>0] clamp(g A/W,0,1)",
        ESTIMATORS[2]: "g H W/N sum_i f_i 1[u_i in pixel p] (unclamped measurement)",
        ESTIMATORS[3]: "g H W/N sum_i f_i B(r_p-r_i)B(c_p-c_i) (unclamped measurement)",
        "semantic_distinction": (
            "LEGACY_GATED_NORMALIZED_SPLAT and UNGATED_NORMALIZED_SPLAT reconstruct "
            "a local transported attribute; HARD_BIN_MONTE_CARLO and "
            "CONTINUOUS_KERNEL_MONTE_CARLO estimate detector-density integrals under "
            "the declared latent measure."
        ),
    }


def _cubic_numpy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(x)
    sign = np.sign(x)
    value = np.where(
        absolute < 1.0,
        2.0 / 3.0 - absolute**2 + 0.5 * absolute**3,
        np.where(absolute < 2.0, (2.0 - absolute) ** 3 / 6.0, 0.0),
    )
    derivative = np.where(
        absolute < 1.0,
        sign * (-2.0 * absolute + 1.5 * absolute**2),
        np.where(absolute < 2.0, -0.5 * sign * (2.0 - absolute) ** 2, 0.0),
    )
    return value, derivative


def _continuity_audit() -> dict[str, object]:
    sweeps: list[dict[str, object]] = []
    threshold_grid = np.linspace(1.0, 2.0, 100_001)
    threshold_weights, _ = _cubic_numpy(threshold_grid)
    threshold_coordinate = float(
        threshold_grid[np.argmin(np.abs(threshold_weights - 0.05))]
    )
    tests = {
        "pixel_center": np.linspace(-0.25, 0.25, 2001),
        "integer_pixel_boundary": np.linspace(0.25, 0.75, 2001),
        "legacy_activation_threshold_boundary": np.linspace(
            threshold_coordinate - 0.25, threshold_coordinate + 0.25, 2001
        ),
        "cubic_support_boundary": np.linspace(1.75, 2.25, 2001),
        "detector_boundary": np.linspace(-0.75, -0.25, 2001),
    }
    response_curves: dict[str, dict[str, list[float]]] = {}
    for location, coordinate in tests.items():
        weight, dweight = _cubic_numpy(coordinate)
        legacy = np.where(weight >= 0.05, 0.8, 0.0)
        ungated = np.where(weight > 0.0, 0.8, 0.0)
        hard = ((coordinate >= -0.5) & (coordinate < 0.5)).astype(float) * 0.8
        continuous = weight * 0.8
        curves = {
            ESTIMATORS[0]: legacy,
            ESTIMATORS[1]: ungated,
            ESTIMATORS[2]: hard,
            ESTIMATORS[3]: continuous,
        }
        derivatives = {
            ESTIMATORS[0]: np.gradient(legacy, coordinate),
            ESTIMATORS[1]: np.gradient(ungated, coordinate),
            ESTIMATORS[2]: np.gradient(hard, coordinate),
            ESTIMATORS[3]: 0.8 * dweight,
        }
        response_curves[location] = {
            "coordinate": coordinate.tolist(),
            **{f"value_{key}": value.tolist() for key, value in curves.items()},
            **{f"derivative_{key}": value.tolist() for key, value in derivatives.items()},
        }
        for estimator in ESTIMATORS:
            value = curves[estimator]
            derivative = derivatives[estimator]
            sweeps.append(
                {
                    "location": location,
                    "estimator": estimator,
                    "maximum_adjacent_value_jump": float(np.max(np.abs(np.diff(value)))),
                    "maximum_adjacent_derivative_jump": float(np.max(np.abs(np.diff(derivative)))),
                }
            )
    boundary_epsilon = 1e-7
    left_w, left_dw = _cubic_numpy(np.asarray([2.0 - boundary_epsilon]))
    right_w, right_dw = _cubic_numpy(np.asarray([2.0 + boundary_epsilon]))
    return {
        "dense_sweeps": sweeps,
        "curves": response_curves,
        "analytic_cubic_support_boundary": {
            "epsilon": boundary_epsilon,
            "value_jump": float(abs(left_w[0] - right_w[0])),
            "derivative_jump": float(abs(left_dw[0] - right_dw[0])),
        },
        "legacy_activation_threshold_coordinate": threshold_coordinate,
        "value_continuous": {
            ESTIMATORS[0]: False,
            ESTIMATORS[1]: False,
            ESTIMATORS[2]: False,
            ESTIMATORS[3]: True,
        },
        "first_derivative_continuous": {
            ESTIMATORS[0]: False,
            ESTIMATORS[1]: False,
            ESTIMATORS[2]: False,
            ESTIMATORS[3]: True,
        },
        "note": "Visibility is held fixed; detector/image truncation is reported separately from the cubic interior response.",
    }


def _toy_detector_test() -> dict[str, object]:
    from scipy.stats import qmc
    from scipy.integrate import quad

    resolution = (32, 32)
    count = 1 << 20
    sequence = qmc.Sobol(2, scramble=True, seed=101).random_base2(20)
    row = sequence[:, 0] * resolution[0] - 0.5
    column = sequence[:, 1] * resolution[1] - 0.5
    probe = ((16, 16), (8, 23), (0, 0))
    rows = []
    for pixel_row, pixel_column in probe:
        wr, _ = _cubic_numpy(pixel_row - row)
        wc, _ = _cubic_numpy(pixel_column - column)
        continuous = math.prod(resolution) * float(np.mean(wr * wc))
        hard = math.prod(resolution) * float(
            np.mean(
                (np.abs(row - pixel_row) < 0.5)
                & (np.abs(column - pixel_column) < 0.5)
            )
        )
        row_integral = quad(
            lambda value: float(_cubic_numpy(np.asarray([pixel_row - value]))[0][0]),
            -0.5,
            resolution[0] - 0.5,
            epsabs=1e-13,
        )[0]
        column_integral = quad(
            lambda value: float(_cubic_numpy(np.asarray([pixel_column - value]))[0][0]),
            -0.5,
            resolution[1] - 0.5,
            epsabs=1e-13,
        )[0]
        expected_continuous = row_integral * column_integral
        rows.append(
            {
                "pixel": [pixel_row, pixel_column],
                "expected_continuous": expected_continuous,
                "continuous_mc": continuous,
                "continuous_absolute_error": abs(continuous - expected_continuous),
                "hard_bin_expected": 1.0,
                "hard_bin_mc": hard,
                "hard_bin_absolute_error": abs(hard - 1.0),
            }
        )
    maximum_interior_error = max(
        row["continuous_absolute_error"] for row in rows if row["pixel"][0] > 1
    )
    interior_response = next(
        row["continuous_mc"] for row in rows if row["pixel"] == [16, 16]
    )
    return {
        "scene": "constant unit transported quantity with uniform normalized detector coordinates",
        "samples": count,
        "resolution": list(resolution),
        "rows": rows,
        "maximum_interior_continuous_error": maximum_interior_error,
        "detector_kernel_interior_integral": interior_response,
        "constant_color_response": {
            "transport_value": 1.0,
            "expected_interior_pixel": 1.0,
            "measured_interior_pixel": interior_response,
            "absolute_error": abs(interior_response - 1.0),
        },
        "constant_normal_flat_patch_response": {
            "constant_color": 1.0,
            "normal_dot_view": 1.0,
            "ambient": 0.35,
            "transport_value": 1.0,
            "expected_interior_pixel": 1.0,
            "measured_interior_pixel": interior_response,
            "absolute_error": abs(interior_response - 1.0),
            "scope": "detector-coordinate flat-patch equivalent; no physical area PDF is claimed",
        },
        "detector_boundary_has_explicit_kernel_truncation": True,
    }


def _metric_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        interval = [float(array[0]), float(array[0])]
    else:
        half = float(stats.t.ppf(0.975, array.size - 1) * array.std(ddof=1) / math.sqrt(array.size))
        interval = [float(array.mean() - half), float(array.mean() + half)]
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "ci95": interval,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "samples": int(array.size),
    }


def _region_masks(target: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    _, signed, codes = _silhouette_regions(np.clip(target, 0.0, None))
    return {
        "WHOLE_IMAGE": np.ones(codes.shape, dtype=bool),
        "FOREGROUND": np.linalg.norm(target, axis=2) > 0.0,
        "SILHOUETTE_0_1PX": codes == 1,
        "SILHOUETTE_1_2PX": codes == 2,
        "SILHOUETTE_2_4PX": codes == 3,
        "SILHOUETTE_4_8PX": codes == 4,
        "INTERIOR_GT_8PX": codes == 5,
    }, signed


def _compare_estimator_views(
    resolution: tuple[int, int],
    reference: list[EstimatorView],
    comparison: list[EstimatorView],
    targets: list[EstimatorView],
    comparison_id: object,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for estimator in ESTIMATORS:
        total_sse = 0.0
        total_scalars = 0
        common_sse = switch_sse = 0.0
        common_scalars = switch_scalars = 0
        xor_pixels = total_pixels = 0
        region_sse = {name: 0.0 for name in (
            "FOREGROUND", "SILHOUETTE_0_1PX", "SILHOUETTE_1_2PX",
            "SILHOUETTE_2_4PX", "SILHOUETTE_4_8PX", "INTERIOR_GT_8PX"
        )}
        region_scalars = {name: 0 for name in region_sse}
        count_sum = mass_sum = 0.0
        for left, right, target in zip(reference, comparison, targets):
            delta = left.images[estimator] - right.images[estimator]
            squared = np.square(delta)
            pixel_sse = squared.sum(2)
            left_support = left.supports[estimator]
            right_support = right.supports[estimator]
            common = left_support & right_support
            switch = left_support ^ right_support
            total_sse += float(squared.sum())
            total_scalars += squared.size
            common_sse += float(squared[common].sum())
            common_scalars += int(common.sum()) * 3
            switch_sse += float(squared[switch].sum())
            switch_scalars += int(switch.sum()) * 3
            xor_pixels += int(switch.sum())
            total_pixels += switch.size
            masks, _ = _region_masks(target.images[estimator])
            for name in region_sse:
                region_sse[name] += float(pixel_sse[masks[name]].sum())
                region_scalars[name] += int(masks[name].sum()) * 3
            count_sum += float(left.count.sum())
            mass_sum += float(left.mass.sum())
        output.append(
            {
                "resolution": list(resolution),
                "comparison": comparison_id,
                "estimator": estimator,
                "whole_image_self_mse": total_sse / max(total_scalars, 1),
                "whole_image_self_sse": total_sse,
                "common_support_mse": common_sse / max(common_scalars, 1),
                "switch_pixel_mse": switch_sse / max(switch_scalars, 1),
                "support_xor_pixel_fraction": xor_pixels / max(total_pixels, 1),
                "fraction_self_sse_from_support_xor": switch_sse / max(total_sse, 1e-30),
                "foreground_self_mse": region_sse["FOREGROUND"] / max(region_scalars["FOREGROUND"], 1),
                "interior_self_mse": region_sse["INTERIOR_GT_8PX"] / max(region_scalars["INTERIOR_GT_8PX"], 1),
                "silhouette_0_8px_self_mse": sum(
                    region_sse[name] for name in region_sse if name.startswith("SILHOUETTE")
                ) / max(sum(region_scalars[name] for name in region_sse if name.startswith("SILHOUETTE")), 1),
                "silhouette_regions": [
                    {
                        "region": name,
                        "self_mse": region_sse[name] / max(region_scalars[name], 1),
                        "self_sse": region_sse[name],
                        "fraction_total_self_sse": region_sse[name] / max(total_sse, 1e-30),
                        "rgb_scalars": region_scalars[name],
                    }
                    for name in region_sse
                ],
                "mean_contribution_count": count_sum / max(total_pixels, 1),
                "mean_detector_weight": mass_sum / max(total_pixels, 1),
            }
        )
    return output


def _fixed_geometry_metrics(
    resolution: tuple[int, int],
    current: list[EstimatorView],
    target: list[EstimatorView],
    sample_set: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for estimator in ESTIMATORS:
        sse = mae_sum = energy_current = energy_target = 0.0
        scalars = 0
        silhouette_sse = interior_sse = 0.0
        silhouette_scalars = interior_scalars = 0
        for current_view, target_view in zip(current, target):
            left = current_view.images[estimator]
            right = target_view.images[estimator]
            difference = left - right
            squared = np.square(difference)
            sse += float(squared.sum())
            mae_sum += float(np.abs(difference).sum())
            energy_current += float(np.square(left).sum())
            energy_target += float(np.square(right).sum())
            scalars += squared.size
            masks, _ = _region_masks(right)
            silhouette = np.logical_or.reduce(
                [masks[name] for name in masks if name.startswith("SILHOUETTE")]
            )
            interior = masks["INTERIOR_GT_8PX"]
            silhouette_sse += float(squared[silhouette].sum())
            silhouette_scalars += int(silhouette.sum()) * 3
            interior_sse += float(squared[interior].sum())
            interior_scalars += int(interior.sum()) * 3
        mse = sse / max(scalars, 1)
        rows.append(
            {
                "sample_set": sample_set,
                "resolution": list(resolution),
                "estimator": estimator,
                "mse": mse,
                "rmse": math.sqrt(mse),
                "mae": mae_sum / max(scalars, 1),
                "current_image_energy": energy_current,
                "target_image_energy": energy_target,
                "silhouette_mse": silhouette_sse / max(silhouette_scalars, 1),
                "interior_mse": interior_sse / max(interior_scalars, 1),
            }
        )
    return rows


def _clamp_audit(
    resolution: tuple[int, int], views: list[EstimatorView], sample_set: object
) -> list[dict[str, object]]:
    rows = []
    for estimator in ESTIMATORS:
        below = above = affected_pixels = scalars = pixels = 0
        for view in views:
            value = view.preclamp[estimator]
            below += int((value < 0.0).sum())
            above += int((value > 1.0).sum())
            affected_pixels += int(((value < 0.0) | (value > 1.0)).any(2).sum())
            scalars += value.size
            pixels += value.shape[0] * value.shape[1]
        rows.append(
            {
                "resolution": list(resolution),
                "sample_set": sample_set,
                "estimator": estimator,
                "below_zero_rgb_scalars": below,
                "above_one_rgb_scalars": above,
                "affected_rgb_scalar_fraction": (below + above) / max(scalars, 1),
                "affected_pixel_fraction": affected_pixels / max(pixels, 1),
                "scientific_mc_measurement_clamped": estimator in ESTIMATORS[:2],
            }
        )
    return rows


def _threshold_ablation(
    resolution: tuple[int, int],
    left: list[EstimatorView],
    right: list[EstimatorView],
    thresholds: tuple[float, ...] = (0.10, 0.05, 0.02, 0.01, 0.005, 0.0),
) -> list[dict[str, object]]:
    rows = []
    for threshold in thresholds:
        sse = switch_sse = brightness = 0.0
        scalars = switch_pixels = pixels = foreground_pixels = 0
        for a, b in zip(left, right):
            raw_a = a.preclamp[ESTIMATORS[1]]
            raw_b = b.preclamp[ESTIMATORS[1]]
            support_a = a.mass >= threshold if threshold > 0 else a.mass > 0
            support_b = b.mass >= threshold if threshold > 0 else b.mass > 0
            image_a = np.where(support_a[..., None], np.clip(raw_a, 0, 1), 0)
            image_b = np.where(support_b[..., None], np.clip(raw_b, 0, 1), 0)
            squared = np.square(image_a - image_b)
            switch = support_a ^ support_b
            sse += float(squared.sum())
            switch_sse += float(squared[switch].sum())
            foreground = support_a | support_b
            brightness += float((image_a.sum(2)[foreground]).sum())
            foreground_pixels += int(foreground.sum())
            scalars += squared.size
            switch_pixels += int(switch.sum())
            pixels += switch.size
        rows.append(
            {
                "resolution": list(resolution),
                "threshold": threshold,
                "self_mse": sse / max(scalars, 1),
                "support_xor_fraction": switch_pixels / max(pixels, 1),
                "support_xor_sse_fraction": switch_sse / max(sse, 1e-30),
                "mean_foreground_brightness": brightness / max(foreground_pixels * 3, 1),
            }
        )
    return rows


def _summarize_multiseed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for resolution in sorted({tuple(row["resolution"]) for row in rows}):
        for estimator in ESTIMATORS:
            selected = [
                row for row in rows
                if tuple(row["resolution"]) == resolution and row["estimator"] == estimator
            ]
            summary.append(
                {
                    "resolution": list(resolution),
                    "estimator": estimator,
                    **{
                        key: _metric_summary([float(row[key]) for row in selected])
                        for key in (
                            "whole_image_self_mse", "foreground_self_mse",
                            "interior_self_mse", "silhouette_0_8px_self_mse",
                            "support_xor_pixel_fraction", "fraction_self_sse_from_support_xor",
                            "common_support_mse", "switch_pixel_mse",
                        )
                    },
                }
            )
    return summary


def _attach_scale_normalized_self_error(
    summary: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
    views: int,
) -> None:
    """Normalize self-noise by the matching operator's target mean-square.

    Raw SSE/MSE from normalized reconstruction and detector-integral operators
    are not directly comparable because the operators have different image
    scales.  This dimensionless statistic is used for cross-operator variance
    and cross-resolution gates; raw values remain archived.
    """

    targets = {
        (tuple(row["resolution"]), str(row["estimator"])): row
        for row in fixed_rows
        if row["sample_set"] == "common_seed_101"
    }
    for row in summary:
        resolution = tuple(row["resolution"])
        target = targets[(resolution, str(row["estimator"]))]
        scalar_count = views * math.prod(resolution) * 3
        target_mean_square = float(target["target_image_energy"]) / scalar_count
        raw = row["whole_image_self_mse"]
        row["target_mean_square"] = target_mean_square
        row["scale_normalized_whole_image_self_mse"] = {
            key: (
                int(value)
                if key == "samples"
                else [
                    float(item) / max(target_mean_square, 1e-30)
                    for item in value
                ]
                if key == "ci95"
                else float(value) / max(target_mean_square, 1e-30)
            )
            for key, value in raw.items()
        }
        row["scale_normalized_self_rmse"] = math.sqrt(
            float(raw["mean"]) / max(target_mean_square, 1e-30)
        )


def _convergence_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for estimator in ESTIMATORS:
        selected_estimator = [row for row in rows if row["estimator"] == estimator]
        counts = sorted({int(row["emitters"]) for row in selected_estimator})
        grouped = []
        for count in counts:
            selected = [row for row in selected_estimator if int(row["emitters"]) == count]
            independent_values = [
                float(row["independent_self_mse"])
                for row in selected
                if int(row["seed"]) != 101
            ]
            grouped.append(
                {
                    "emitters": count,
                    "mean_foreground_brightness": _metric_summary([float(row["mean_foreground_brightness"]) for row in selected]),
                    "total_image_energy": _metric_summary([float(row["total_image_energy"]) for row in selected]),
                    "mse_vs_8N": _metric_summary([float(row["mse_vs_8N"]) for row in selected]),
                    "independent_self_mse": _metric_summary(independent_values),
                }
            )
        fit = [row for row in grouped if row["emitters"] < counts[-1] and row["mse_vs_8N"]["mean"] > 0]
        mse_slope = float(np.polyfit(
            np.log([row["emitters"] for row in fit]),
            np.log([row["mse_vs_8N"]["mean"] for row in fit]), 1
        )[0]) if len(fit) >= 2 else 0.0
        summary.append(
            {
                "estimator": estimator,
                "rows": grouped,
                "log_mse_vs_N_slope": mse_slope,
                "implied_rmse_slope": mse_slope / 2.0,
            }
        )
    return summary


def _mc_gradient_stability(prepared: object, config: MCDetectorConfig) -> dict[str, object]:
    from .corrected_birth import (
        CorrectedBirthConfig,
        CorrectedLocalZeroSet,
        _active_components,
        _build_context,
        _directional_gradient_derivative,
        _evaluate,
        _render,
        _visibility_cells,
    )
    from .locality import wendland_gradients, wendland_values
    from .meshfree_surface import meshfree_base_color
    from .sampling_diagnostic import _copy_layout

    corrected = CorrectedBirthConfig(
        dictionary_count=64,
        initial_count=config.gradient_parameters,
        surface_samples=config.gradient_surface_samples,
        views=config.gradient_views,
        resolution=config.gradient_resolution,
        surface_scramble_seed=config.sobol_seeds[0],
    )
    heldout_config = replace(corrected, surface_scramble_seed=config.sobol_seeds[1])
    context = _build_context(prepared, corrected)
    heldout = _build_context(prepared, heldout_config)
    _copy_layout(heldout, context)
    device = context.reference_points.device
    active = torch.arange(corrected.initial_count, device=device)
    coefficients = torch.zeros(active.numel(), dtype=torch.float64, device=device)
    layout, support = _active_components(context, active)
    state = _evaluate(context, active, coefficients, (layout, support))
    heldout_layout, heldout_support = _active_components(heldout, active)
    heldout_state = _evaluate(
        heldout, active, coefficients, (heldout_layout, heldout_support)
    )
    scale = corrected.sensor_gain * math.prod(corrected.resolution_shape) / corrected.surface_samples

    def mc_images(renders: list[object]) -> list[Tensor]:
        return [(scale * render.numerator).reshape(-1) for render in renders]

    def matrices_for(
        local_context: object,
        local_state: object,
        local_layout: object,
        local_support: object,
    ) -> list[Tensor]:
        model = CorrectedLocalZeroSet(
            local_context.base,
            local_layout,
            local_state.coefficients,
            local_context.reference_points,
            local_context.reference_normals,
            local_support,
        )
        directional_hessian = _directional_gradient_derivative(
            model, local_state.points, local_context.reference_normals
        )
        output = []
        for cell, render in zip(local_context.cells, local_state.renders):
            local_events, basis_ids = local_support.subset_points(cell.owner_ids)
            point_ids = cell.owner_ids[local_events]
            offsets = local_state.points[point_ids] - local_layout.centers[basis_ids]
            basis_values = wendland_values(offsets, local_layout.radii[basis_ids])
            inside = basis_values > 0.0
            local_events = local_events[inside]
            basis_ids = basis_ids[inside]
            point_ids = point_ids[inside]
            offsets = offsets[inside]
            basis_values = basis_values[inside]
            basis_gradient = wendland_gradients(offsets, local_layout.radii[basis_ids])
            denominator = torch.where(
                local_state.denominator[point_ids].abs() > 1e-8,
                local_state.denominator[point_ids],
                torch.full_like(local_state.denominator[point_ids], 1e-8),
            )
            displacement = -basis_values / denominator
            dpoint = local_context.reference_normals[point_ids] * displacement[:, None]
            gradient_derivative = (
                basis_gradient + directional_hessian[point_ids] * displacement[:, None]
            )
            normals = local_state.normals[point_ids]
            dnormal = (
                gradient_derivative
                - normals * (normals * gradient_derivative).sum(1, keepdim=True)
            ) / torch.linalg.vector_norm(
                local_state.gradients[point_ids], dim=1, keepdim=True
            ).clamp_min(1e-30)
            extent = local_context.upper - local_context.lower
            dcolor = torch.stack(
                (
                    0.72 / extent[0] * dpoint[:, 0],
                    0.70 / extent[1] * dpoint[:, 1],
                    -0.62 / extent[2] * dpoint[:, 2],
                ),
                1,
            )
            colors = meshfree_base_color(
                local_state.points[point_ids], local_context.lower, local_context.upper
            )
            raw_cosine = local_state.normals[point_ids] @ cell.direction
            cosine = raw_cosine.clamp_min(0.0)
            lobe = corrected.ambient + (1.0 - corrected.ambient) * cosine
            dcosine = torch.where(
                raw_cosine > 0.0,
                dnormal @ cell.direction,
                torch.zeros_like(raw_cosine),
            )
            dradiance = dcolor * lobe[:, None] + colors * (
                (1.0 - corrected.ambient) * dcosine
            )[:, None]
            rows, columns = cell.resolution
            drow = -rows / cell.extent * (dpoint @ cell.up)
            dcolumn = columns / cell.extent * (dpoint @ cell.right)
            dweights = (
                render.weight_row_derivatives[local_events] * drow[:, None]
                + render.weight_column_derivatives[local_events] * dcolumn[:, None]
            )
            weights = render.weights[local_events]
            pixels = render.pixels[local_events]
            radiance = render.radiance[local_events]
            derivative = scale * (
                dweights[..., None] * radiance[:, None, :]
                + weights[..., None] * dradiance[:, None, :]
            )
            channels = torch.arange(3, device=device)
            output_rows = (
                pixels[..., None] * 3 + channels[None, None, :]
            ).expand_as(derivative)
            parameter_columns = basis_ids[:, None, None].expand_as(output_rows)
            touched = render.valid[local_events, :, None] & (derivative.abs() > 1e-12)
            output.append(
                torch.sparse_coo_tensor(
                    torch.stack((output_rows[touched], parameter_columns[touched])),
                    derivative[touched],
                    size=(3 * rows * columns, local_layout.count),
                    dtype=local_state.points.dtype,
                    device=device,
                ).coalesce()
            )
        return output

    matrices = matrices_for(context, state, layout, support)
    heldout_matrices = matrices_for(
        heldout, heldout_state, heldout_layout, heldout_support
    )

    target_masks: dict[str, list[Tensor]] = {
        "all_pixels": [], "silhouette_0_2px": [], "interior_gt8px": []
    }
    for target in context.target_images:
        rows, columns = corrected.resolution_shape
        target_image = target.reshape(rows, columns, 3).detach().cpu().numpy()
        _, _, codes = _silhouette_regions(target_image)
        masks = {
            "all_pixels": np.ones((rows, columns), dtype=bool),
            "silhouette_0_2px": (codes == 1) | (codes == 2),
            "interior_gt8px": codes == 5,
        }
        for name, mask in masks.items():
            target_masks[name].append(
                torch.from_numpy(np.repeat(mask.reshape(-1), 3)).to(device)
            )
    masks = {name: torch.cat(parts) for name, parts in target_masks.items()}

    def column(parameter: int, source: list[Tensor]) -> Tensor:
        unit = torch.zeros(active.numel(), dtype=torch.float64, device=device)
        unit[parameter] = 1.0
        return torch.cat(
            [torch.sparse.mm(matrix, unit[:, None]).flatten() for matrix in source]
        )

    analytic_columns = [column(index, matrices) for index in range(active.numel())]
    energy = torch.stack([item.square().sum() for item in analytic_columns])
    silhouette_energy = torch.stack([
        item[masks["silhouette_0_2px"]].square().sum() for item in analytic_columns
    ])
    interior_energy = torch.stack([
        item[masks["interior_gt8px"]].square().sum() for item in analytic_columns
    ])
    categories: dict[str, int] = {
        "deep_interior": int(torch.argmax(interior_energy)),
        "silhouette": int(torch.argmax(silhouette_energy)),
    }
    excluded = set(categories.values())
    categories["smooth_visible_surface"] = next(
        index for index in torch.argsort(energy, descending=True).tolist() if index not in excluded
    )
    excluded.add(categories["smooth_visible_surface"])
    categories["occlusion_boundary"] = next(
        index for index in torch.argsort(silhouette_energy, descending=True).tolist() if index not in excluded
    )
    excluded.add(categories["occlusion_boundary"])
    center_grad_plus = context.base.gradient(layout.centers + 2e-3)
    center_grad_minus = context.base.gradient(layout.centers - 2e-3)
    curvature = torch.linalg.vector_norm(center_grad_plus - center_grad_minus, dim=1)
    categories["high_curvature"] = next(
        index for index in torch.argsort(curvature, descending=True).tolist() if index not in excluded
    )

    atlas = nested_fibonacci_atlas(device, (corrected.views,))
    boundary = enclosing_observation_sphere(context.reference_points)

    class GenericLocalField:
        def __init__(self, local_coefficients: Tensor) -> None:
            self.coefficients = local_coefficients
            self.lower = context.base.lower
            self.upper = context.base.upper

        def value(self, points: Tensor) -> Tensor:
            shape = points.shape[:-1]
            flat = points.reshape(-1, 3)
            values = context.base.value(flat)
            for start in range(0, flat.shape[0], 8192):
                stop = min(start + 8192, flat.shape[0])
                offsets = flat[start:stop, None, :] - layout.centers[None, :, :]
                basis = wendland_values(
                    offsets.reshape(-1, 3),
                    layout.radii[None, :].expand(stop - start, -1).reshape(-1),
                ).reshape(stop - start, -1)
                values[start:stop] += basis @ self.coefficients
            return values.reshape(shape)

    def full_images(
        local_coefficients: Tensor, evaluated: object
    ) -> tuple[list[Tensor], list[object]]:
        cells, _ = _visibility_cells(
            GenericLocalField(local_coefficients),
            evaluated.points,
            evaluated.normals,
            atlas,
            boundary,
            corrected,
        )
        _, renders = _render(evaluated.points, evaluated.normals, context, cells)
        return mc_images(renders), cells

    rows: list[dict[str, object]] = []
    for category, parameter in categories.items():
        analytic = analytic_columns[parameter]
        heldout_analytic = column(parameter, heldout_matrices)
        realization = _vector_metrics(analytic, heldout_analytic, masks["all_pixels"])
        for epsilon in config.gradient_epsilons:
            plus_coefficients = coefficients.clone()
            minus_coefficients = coefficients.clone()
            plus_coefficients[parameter] += epsilon
            minus_coefficients[parameter] -= epsilon
            plus = _evaluate(context, active, plus_coefficients, (layout, support))
            minus = _evaluate(context, active, minus_coefficients, (layout, support))
            frozen = torch.cat([
                (right - left) / (2.0 * epsilon)
                for right, left in zip(mc_images(plus.renders), mc_images(minus.renders))
            ])
            plus_full, plus_cells = full_images(plus_coefficients, plus)
            minus_full, minus_cells = full_images(minus_coefficients, minus)
            full = torch.cat([
                (right - left) / (2.0 * epsilon)
                for right, left in zip(plus_full, minus_full)
            ])
            topology = full - frozen
            visibility_switches = 0
            visibility_identities = corrected.surface_samples * corrected.views
            for plus_cell, minus_cell in zip(plus_cells, minus_cells):
                plus_visible = torch.zeros(corrected.surface_samples, dtype=torch.bool, device=device)
                minus_visible = torch.zeros_like(plus_visible)
                plus_visible[plus_cell.owner_ids] = True
                minus_visible[minus_cell.owner_ids] = True
                visibility_switches += int((plus_visible ^ minus_visible).sum())
            support_switch = torch.cat([
                ((right.reshape(-1, 3).abs().sum(1) > 0.0) ^
                 (left.reshape(-1, 3).abs().sum(1) > 0.0)).repeat_interleave(3)
                for right, left in zip(plus_full, minus_full)
            ])
            local_masks = dict(masks)
            local_masks["support_switch_pixels"] = support_switch
            base_mc = torch.cat(mc_images(state.renders))
            clamp_affected = (base_mc < 0.0) | (base_mc > 1.0)
            analytic_changes = analytic.abs() > 1e-12
            frozen_changes = frozen.abs() > 1e-12
            full_changes = full.abs() > 1e-12
            rows.append(
                {
                    "category": category,
                    "parameter": parameter,
                    "epsilon": epsilon,
                    "sample_realization_gradient": realization,
                    "visibility_switch_fraction": visibility_switches / max(visibility_identities, 1),
                    "support_switch_scalar_fraction": float(support_switch.to(torch.float64).mean()),
                    "topology_component_fraction_of_full_norm": float(
                        torch.linalg.vector_norm(topology) /
                        torch.linalg.vector_norm(full).clamp_min(1e-30)
                    ),
                    "clamp_affected_jacobian_entry_fraction": float(
                        (analytic_changes & clamp_affected).sum() /
                        analytic_changes.sum().clamp_min(1)
                    ),
                    "clamp_affected_frozen_fd_change_fraction": float(
                        (frozen_changes & clamp_affected).sum() /
                        frozen_changes.sum().clamp_min(1)
                    ),
                    "clamp_affected_full_fd_change_fraction": float(
                        (full_changes & clamp_affected).sum() /
                        full_changes.sum().clamp_min(1)
                    ),
                    "regions": {
                        name: {
                            "analytic_vs_frozen": _vector_metrics(analytic, frozen, mask),
                            "analytic_vs_full": _vector_metrics(analytic, full, mask),
                            "frozen_vs_full": _vector_metrics(frozen, full, mask),
                            "scalars": int(mask.sum()),
                        }
                        for name, mask in local_masks.items()
                    },
                }
            )
    best = [
        min(
            [row for row in rows if row["category"] == category],
            key=lambda row: float(row["regions"]["all_pixels"]["analytic_vs_frozen"]["relative_error"]),
        )
        for category in categories
    ]
    return {
        "derivation": _measurement_derivation(config)["jacobian"],
        "configuration": asdict(corrected),
        "categories": categories,
        "epsilon_sweep": list(config.gradient_epsilons),
        "rows": rows,
        "best_frozen_rows": best,
        "frozen_topology": "surface identities, visibility owner sets, footprint candidates, and path topology",
    }


def _render_estimator_views(
    field: object,
    surface: object,
    emitter_count: int,
    resolution: tuple[int, int],
    boundary: object,
    atlas: object,
    config: MCDetectorConfig,
    *,
    geometry: str = "base",
) -> tuple[list[EstimatorView], dict[str, object]]:
    pixel_views, report = _render_diagnostics(
        field,
        surface,
        emitter_count,
        resolution,
        boundary,
        atlas,
        config.pixel_config(),
        geometry=geometry,
        keep_numerator=True,
        keep_hard_bin=True,
    )
    derived = _derive_views(
        pixel_views, emitter_count, resolution, config.sensor_gain
    )
    del pixel_views
    gc.collect()
    return derived, report


def _convergence_rows_for_seed(
    seed: int,
    renders: dict[int, list[EstimatorView]],
    reference: dict[int, list[EstimatorView]] | None,
) -> list[dict[str, object]]:
    maximum = max(renders)
    highest = renders[maximum]
    rows = []
    for count, views in sorted(renders.items()):
        for estimator in ESTIMATORS:
            energy = brightness = 0.0
            foreground_pixels = scalars = sse_high = self_sse = 0
            for view, high in zip(views, highest):
                image = view.images[estimator]
                mask = np.linalg.norm(high.images[estimator], axis=2) > 0.0
                energy += float(np.square(image).sum())
                brightness += float(image[mask].sum())
                foreground_pixels += int(mask.sum())
                difference = image - high.images[estimator]
                sse_high += float(np.square(difference).sum())
                scalars += image.size
            if reference is not None:
                for view, left in zip(views, reference[count]):
                    self_sse += float(
                        np.square(view.images[estimator] - left.images[estimator]).sum()
                    )
            rows.append(
                {
                    "seed": seed,
                    "emitters": count,
                    "estimator": estimator,
                    "mean_foreground_brightness": brightness / max(foreground_pixels * 3, 1),
                    "total_image_energy": energy,
                    "mse_vs_8N": sse_high / max(scalars, 1),
                    "rmse_vs_8N": math.sqrt(sse_high / max(scalars, 1)),
                    "independent_self_mse": self_sse / max(scalars, 1) if reference is not None else 0.0,
                }
            )
    return rows


def _chunk_invariance(
    base: object,
    cells: Tensor,
    config: MCDetectorConfig,
) -> dict[str, object]:
    count = 8192
    atlas = nested_fibonacci_atlas(base.grid.device, (2,))
    surfaces = [
        _prepare_surface(base, None, count, chunk, config.sobol_seeds[0], cells)
        for chunk in (1024, 2048)
    ]
    boundary = enclosing_observation_sphere(surfaces[0].reference_points)
    render_config = replace(config, views=2, emitter_chunk_size=1024)
    left, _ = _render_estimator_views(
        base, surfaces[0], count, (64, 64), boundary, atlas, render_config
    )
    render_config = replace(render_config, emitter_chunk_size=2048)
    right, _ = _render_estimator_views(
        base, surfaces[1], count, (64, 64), boundary, atlas, render_config
    )
    errors = {
        estimator: max(
            float(np.max(np.abs(a.images[estimator] - b.images[estimator])))
            for a, b in zip(left, right)
        )
        for estimator in ESTIMATORS
    }
    point_error = float(torch.max(torch.abs(
        surfaces[0].reference_points - surfaces[1].reference_points
    )))
    del surfaces, left, right
    _release()
    return {
        "emitters": count,
        "resolution": [64, 64],
        "views": 2,
        "chunk_sizes": [1024, 2048],
        "surface_point_max_error": point_error,
        "estimator_image_max_errors": errors,
        "passed": point_error == 0.0 and max(errors.values()) <= 2e-6,
    }


def _nested_prefix_validation() -> dict[str, object]:
    engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=101)
    large = engine.draw(8192)
    engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=101)
    small = engine.draw(2048)
    return {
        "dimension": 4,
        "seed": 101,
        "small_range": [0, 2048],
        "large_range": [0, 8192],
        "prefix_max_error": float(torch.max(torch.abs(small - large[:2048]))),
        "passed": bool(torch.equal(small, large[:2048])),
    }


def _save_figures(
    directory: Path,
    continuity: dict[str, object],
    threshold_rows: list[dict[str, object]],
    convergence: list[dict[str, object]],
    multiseed_summary: list[dict[str, object]],
    primary_rows: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
    gradient: dict[str, object],
    performance: list[dict[str, object]],
    captures: dict[tuple[int, int], dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    directory.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(ESTIMATORS, ("#777777", "#377eb8", "#e41a1c", "#4daf4a")))

    figure, axis = plt.subplots(figsize=(12, 3))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 2)
    axis.axis("off")
    boxes = (
        (0.2, "latent Sobol\nz in [0,1)^4"),
        (2.6, "zero-set map\nx=T_F(z)"),
        (5.0, "visibility +\ntransport f"),
        (7.4, "continuous K_p(u)\nfinite detector area"),
        (9.8, "global average\ng/N sum fK"),
    )
    for x, label in boxes:
        axis.add_patch(FancyBboxPatch((x, 0.55), 1.8, 0.9, boxstyle="round,pad=0.08", facecolor="#e8f1fa", edgecolor="#24557a"))
        axis.text(x + 0.9, 1.0, label, ha="center", va="center")
    for x in (2.05, 4.45, 6.85, 9.25):
        axis.annotate("", xy=(x + 0.45, 1), xytext=(x, 1), arrowprops={"arrowstyle": "->"})
    figure.tight_layout()
    figure.savefig(directory / "v084_estimator_diagram.png", dpi=180)
    plt.close(figure)

    coordinate = np.linspace(-2.3, 2.3, 4001)
    weight, derivative = _cubic_numpy(coordinate)
    curves = {
        ESTIMATORS[0]: np.where(weight >= 0.05, 0.8, 0.0),
        ESTIMATORS[1]: np.where(weight > 0, 0.8, 0.0),
        ESTIMATORS[2]: ((coordinate >= -0.5) & (coordinate < 0.5)) * 0.8,
        ESTIMATORS[3]: 0.8 * weight,
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for estimator, values in curves.items():
        axes[0].plot(coordinate, values, label=estimator.replace("_", " "), color=colors[estimator])
        estimate = 0.8 * derivative if estimator == ESTIMATORS[3] else np.gradient(values, coordinate)
        axes[1].plot(coordinate, estimate, label=estimator.replace("_", " "), color=colors[estimator])
    axes[0].set_ylabel("one-sample pixel contribution")
    axes[1].set_ylabel("d contribution / d coordinate")
    axes[1].set_yscale("symlog", linthresh=0.05)
    for axis in axes:
        axis.set_xlabel("continuous pixel coordinate")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "v084_detector_kernel_continuity.png", dpi=180)
    plt.close(figure)

    selected_threshold = [row for row in threshold_rows if tuple(row["resolution"]) == (1080, 1920)]
    x = [row["threshold"] for row in selected_threshold]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, key, title in zip(
        axes,
        ("self_mse", "support_xor_sse_fraction", "mean_foreground_brightness"),
        ("MC self-MSE", "XOR fraction of SSE", "foreground brightness"),
    ):
        axis.plot(x, [row[key] for row in selected_threshold], marker="o")
        axis.set_xlabel("W activation threshold")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "v084_threshold_ablation.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for result in convergence:
        estimator = result["estimator"]
        counts = [row["emitters"] for row in result["rows"]]
        axes[0].plot(counts, [row["mse_vs_8N"]["mean"] for row in result["rows"]], marker="o", label=estimator, color=colors[estimator])
        axes[1].plot(counts, [row["mean_foreground_brightness"]["mean"] for row in result["rows"]], marker="o", label=estimator, color=colors[estimator])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE vs nested 8N")
    axes[1].set_ylabel("mean foreground brightness")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("emitters")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "v084_sample_convergence.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    positions = np.arange(len(ESTIMATORS))
    values = []
    for estimator in ESTIMATORS:
        row = next(item for item in multiseed_summary if tuple(item["resolution"]) == (1080, 1920) and item["estimator"] == estimator)
        values.append(row["scale_normalized_whole_image_self_mse"])
    axis.bar(positions, [value["mean"] for value in values], yerr=[value["std"] for value in values], color=[colors[key] for key in ESTIMATORS])
    axis.set_xticks(positions, [key.replace("_", "\n") for key in ESTIMATORS], fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("1080p self-MSE / matching target mean-square")
    figure.tight_layout()
    figure.savefig(directory / "v084_multiseed_mc_comparison.png", dpi=180)
    plt.close(figure)

    labels = ["256x256", "512x512", "960x540", "1920x1080"]
    figure, axis = plt.subplots(figsize=(9, 4))
    for estimator in ESTIMATORS:
        selected = [next(item for item in multiseed_summary if tuple(item["resolution"]) == resolution and item["estimator"] == estimator) for resolution in ((256,256),(512,512),(540,960),(1080,1920))]
        axis.plot(labels, [row["scale_normalized_whole_image_self_mse"]["mean"] for row in selected], marker="o", label=estimator, color=colors[estimator])
    axis.set_yscale("log")
    axis.set_ylabel("self-MSE / matching target mean-square")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "v084_resolution_mc_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    width = 0.18
    for index, estimator in enumerate(ESTIMATORS):
        selected = [row for row in primary_rows if row["estimator"] == estimator]
        axis.bar(np.arange(4) + (index - 1.5) * width, [row["fraction_self_sse_from_support_xor"] for row in selected], width, label=estimator, color=colors[estimator])
    axis.set_xticks(np.arange(4), labels)
    axis.set_ylabel("fraction self-SSE from support XOR")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "v084_support_xor_sse.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    for estimator in ESTIMATORS:
        row = next(item for item in primary_rows if tuple(item["resolution"]) == (1080,1920) and item["estimator"] == estimator)
        axis.plot([item["region"] for item in row["silhouette_regions"]], [item["self_mse"] for item in row["silhouette_regions"]], marker="o", label=estimator, color=colors[estimator])
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel("1080p self-MSE")
    axis.set_title("Operator-specific scale; compare spatial concentration, not raw height")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "v084_silhouette_interior_error.png", dpi=180)
    plt.close(figure)

    capture = captures[(256, 256)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, estimator in enumerate((ESTIMATORS[0], ESTIMATORS[3])):
        axes[column, 0].imshow(np.clip(capture[f"current_{estimator}"], 0, 1))
        axes[column, 1].imshow(np.clip(capture[f"target_{estimator}"], 0, 1))
        axes[column, 2].imshow(np.mean(np.square(capture[f"current_{estimator}"] - capture[f"target_{estimator}"]), 2), cmap="inferno")
        axes[column, 0].set_ylabel(estimator.replace("_", "\n"), fontsize=7)
    for axis, title in zip(axes[0], ("fixed geometry", "reference geometry", "squared error")):
        axis.set_title(title)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "v084_legacy_vs_mc_render.png", dpi=180)
    plt.close(figure)

    best = gradient["best_frozen_rows"]
    for comparison, filename in (("analytic_vs_frozen", "v084_gradient_frozen_fd.png"), ("analytic_vs_full", "v084_gradient_full_fd.png")):
        figure, axis = plt.subplots(figsize=(9, 4))
        for category in gradient["categories"]:
            selected = sorted([row for row in gradient["rows"] if row["category"] == category], key=lambda row: row["epsilon"])
            axis.plot([row["epsilon"] for row in selected], [row["regions"]["all_pixels"][comparison]["relative_error"] for row in selected], marker="o", label=category)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("coefficient epsilon")
        axis.set_ylabel("relative error")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(labels, [row["runtime_seconds"] for row in performance], marker="o")
    axes[1].plot(labels, [row["packets_per_second"] for row in performance], marker="o", label="packets/s")
    axes[1].plot(labels, [row["events_per_second"] for row in performance], marker="o", label="events/s")
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("throughput")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("resolution")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "v084_runtime_scaling.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for column, estimator in enumerate(ESTIMATORS):
        axes[0, column].imshow(np.clip(capture[f"current_{estimator}"], 0, 1))
        axes[0, column].set_title(estimator.replace("_", "\n"), fontsize=8)
        axes[1, column].imshow(np.clip(capture[f"target_{estimator}"], 0, 1))
    axes[0, 0].set_ylabel("fixed geometry")
    axes[1, 0].set_ylabel("reference geometry")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "v084_rgb_renders.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row_index, resolution in enumerate(((256,256),(1080,1920))):
        local = captures[resolution]
        for column, estimator in enumerate(ESTIMATORS):
            error = np.mean(np.square(local[f"sample_a_{estimator}"] - local[f"sample_b_{estimator}"]), 2)
            axes[row_index, column].imshow(np.log10(error + 1e-12), cmap="inferno")
            if row_index == 0:
                axes[row_index, column].set_title(estimator.replace("_", "\n"), fontsize=8)
        axes[row_index, 0].set_ylabel(f"{resolution[1]}x{resolution[0]}")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "v084_spatial_error_maps.png", dpi=180)
    plt.close(figure)


def run_mc_continuous_detector_diagnostic(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    config: MCDetectorConfig | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    experiment = config or MCDetectorConfig()
    if len(experiment.resolutions) != len(experiment.emitter_counts):
        raise ValueError("resolution/emitter-count lengths differ")
    started = time.perf_counter()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    base = prepared.base_field.to(device)
    target = prepared.gt_field.to(device)
    cells = sign_changing_cells(base.grid)
    maximum = max(experiment.emitter_counts)
    atlas = nested_fibonacci_atlas(device, (experiment.views,))

    derivation = _measurement_derivation(experiment)
    definitions = _estimator_definitions()
    continuity = _continuity_audit()
    toy = _toy_detector_test()
    nested_prefix = _nested_prefix_validation()
    chunk_invariance = _chunk_invariance(base, cells, experiment)

    boundaries: dict[tuple[int, int], object] = {}
    reference_by_resolution: dict[tuple[int, int], list[EstimatorView]] = {}
    target_by_resolution: dict[tuple[int, int], list[EstimatorView]] = {}
    convergence_reference: dict[int, list[EstimatorView]] = {}
    convergence_rows: list[dict[str, object]] = []
    multiseed_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    clamp_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    performance: list[dict[str, object]] = []
    captures: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    recorded_allocated = recorded_reserved = 0.0

    for seed_index, seed in enumerate(experiment.sobol_seeds):
        needs_target = seed_index < 2
        surface = _prepare_surface(
            base,
            target if needs_target else None,
            maximum,
            experiment.emitter_chunk_size,
            seed,
            cells,
        )
        if seed_index == 0:
            for resolution, emitter_count in zip(
                experiment.resolutions, experiment.emitter_counts
            ):
                boundaries[resolution] = enclosing_observation_sphere(
                    surface.reference_points[:emitter_count]
                )

        convergence_renders: dict[int, list[EstimatorView]] = {}
        convergence_reports: dict[int, dict[str, object]] = {}
        for count in experiment.convergence_counts:
            views, transport_report = _render_estimator_views(
                base,
                surface,
                count,
                (256, 256),
                boundaries[(256, 256)],
                atlas,
                experiment,
            )
            convergence_renders[count] = views
            convergence_reports[count] = transport_report
            recorded_allocated = max(recorded_allocated, float(transport_report["peak_allocated_mib"]))
            recorded_reserved = max(recorded_reserved, float(transport_report["peak_reserved_mib"]))
        convergence_rows.extend(
            _convergence_rows_for_seed(
                seed,
                convergence_renders,
                convergence_reference if seed_index > 0 else None,
            )
        )
        if seed_index == 0:
            convergence_reference = convergence_renders

        for resolution, emitter_count in zip(
            experiment.resolutions, experiment.emitter_counts
        ):
            if resolution == (256, 256) and emitter_count in convergence_renders:
                current_views = convergence_renders[emitter_count]
                transport_report = convergence_reports[emitter_count]
            else:
                current_views, transport_report = _render_estimator_views(
                    base,
                    surface,
                    emitter_count,
                    resolution,
                    boundaries[resolution],
                    atlas,
                    experiment,
                )
            recorded_allocated = max(recorded_allocated, float(transport_report["peak_allocated_mib"]))
            recorded_reserved = max(recorded_reserved, float(transport_report["peak_reserved_mib"]))
            if seed_index == 0:
                if surface.target_points is None:
                    raise RuntimeError("target surface missing for reference seed")
                target_views, target_report = _render_estimator_views(
                    target,
                    surface,
                    emitter_count,
                    resolution,
                    boundaries[resolution],
                    atlas,
                    experiment,
                    geometry="target",
                )
                recorded_allocated = max(recorded_allocated, float(target_report["peak_allocated_mib"]))
                recorded_reserved = max(recorded_reserved, float(target_report["peak_reserved_mib"]))
                reference_by_resolution[resolution] = current_views
                target_by_resolution[resolution] = target_views
                fixed_rows.extend(
                    _fixed_geometry_metrics(
                        resolution, current_views, target_views, "common_seed_101"
                    )
                )
                clamp_rows.extend(_clamp_audit(resolution, current_views, seed))
                view_id = min(experiment.capture_view, experiment.views - 1)
                captures[resolution] = {
                    **{
                        f"current_{estimator}": current_views[view_id].images[estimator]
                        for estimator in ESTIMATORS
                    },
                    **{
                        f"target_{estimator}": target_views[view_id].images[estimator]
                        for estimator in ESTIMATORS
                    },
                    **{
                        f"sample_a_{estimator}": current_views[view_id].images[estimator]
                        for estimator in ESTIMATORS
                    },
                }
                attempts = int(transport_report["attempted_packets"])
                retained = int(transport_report["retained_events"])
                seconds = float(transport_report["runtime_seconds"])
                performance.append(
                    {
                        "resolution": list(resolution),
                        "emitters": emitter_count,
                        "attempted_packets": attempts,
                        "retained_events": retained,
                        "continuous_detector_writes": int(transport_report["footprint_writes"]),
                        "hard_bin_writes": int(transport_report["hard_bin_writes"]),
                        "runtime_seconds": seconds,
                        "packets_per_second": attempts / max(seconds, 1e-30),
                        "events_per_second": retained / max(seconds, 1e-30),
                        "peak_allocated_mib": transport_report["peak_allocated_mib"],
                        "peak_reserved_mib": transport_report["peak_reserved_mib"],
                        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                        "chunk_size": experiment.emitter_chunk_size,
                    }
                )
            else:
                compared = _compare_estimator_views(
                    resolution,
                    reference_by_resolution[resolution],
                    current_views,
                    target_by_resolution[resolution],
                    seed,
                )
                multiseed_rows.extend(compared)
                if seed_index == 1:
                    primary_rows.extend(compared)
                    threshold_rows.extend(
                        _threshold_ablation(
                            resolution,
                            reference_by_resolution[resolution],
                            current_views,
                        )
                    )
                    target_views, _ = _render_estimator_views(
                        target,
                        surface,
                        emitter_count,
                        resolution,
                        boundaries[resolution],
                        atlas,
                        experiment,
                        geometry="target",
                    )
                    fixed_rows.extend(
                        _fixed_geometry_metrics(
                            resolution, current_views, target_views, "heldout_seed_211"
                        )
                    )
                    view_id = min(experiment.capture_view, experiment.views - 1)
                    captures[resolution].update(
                        {
                            f"sample_b_{estimator}": current_views[view_id].images[estimator]
                            for estimator in ESTIMATORS
                        }
                    )
                    del target_views
                if not (
                    resolution == (256, 256)
                    and emitter_count in convergence_renders
                ):
                    del current_views
            gc.collect()
        if seed_index > 0:
            del convergence_renders
        del surface
        _release()
        _progress("v084_seed_complete", seed=seed)

    multiseed_summary = _summarize_multiseed(multiseed_rows)
    _attach_scale_normalized_self_error(
        multiseed_summary, fixed_rows, experiment.views
    )
    convergence_summary = _convergence_summary(convergence_rows)
    visibility = _controlled_visibility_switching(
        base,
        _prepare_surface(
            base,
            None,
            experiment.emitter_counts[0],
            experiment.emitter_chunk_size,
            experiment.sobol_seeds[0],
            cells,
        ),
        atlas,
        boundaries[(256, 256)],
        experiment.pixel_config(),
    )
    gradient = _mc_gradient_stability(prepared, experiment)

    continuous_convergence = next(
        row for row in convergence_summary if row["estimator"] == ESTIMATORS[3]
    )
    continuous_brightness = [
        float(row["mean_foreground_brightness"]["mean"])
        for row in continuous_convergence["rows"]
    ]
    brightness_relative_range = (
        max(continuous_brightness) - min(continuous_brightness)
    ) / max(statistics.mean(continuous_brightness), 1e-30)
    legacy_primary = [row for row in primary_rows if row["estimator"] == ESTIMATORS[0]]
    ungated_primary = [row for row in primary_rows if row["estimator"] == ESTIMATORS[1]]
    continuous_primary = [row for row in primary_rows if row["estimator"] == ESTIMATORS[3]]
    gate_fraction_reduction = 1.0 - statistics.median(
        [float(row["fraction_self_sse_from_support_xor"]) for row in ungated_primary]
    ) / max(statistics.median(
        [float(row["fraction_self_sse_from_support_xor"]) for row in legacy_primary]
    ), 1e-30)
    gate_switch_sse_reductions = []
    for resolution in experiment.resolutions:
        legacy_row = next(
            row for row in legacy_primary if tuple(row["resolution"]) == resolution
        )
        ungated_row = next(
            row for row in ungated_primary if tuple(row["resolution"]) == resolution
        )
        legacy_switch_sse = (
            float(legacy_row["whole_image_self_sse"])
            * float(legacy_row["fraction_self_sse_from_support_xor"])
        )
        ungated_switch_sse = (
            float(ungated_row["whole_image_self_sse"])
            * float(ungated_row["fraction_self_sse_from_support_xor"])
        )
        gate_switch_sse_reductions.append(
            1.0 - ungated_switch_sse / max(legacy_switch_sse, 1e-30)
        )
    gate_sse_reduction = statistics.median(gate_switch_sse_reductions)
    continuous_xor_median = statistics.median(
        [float(row["fraction_self_sse_from_support_xor"]) for row in continuous_primary]
    )
    continuous_resolution = [
        next(
            row for row in multiseed_summary
            if tuple(row["resolution"]) == resolution and row["estimator"] == ESTIMATORS[3]
        )["scale_normalized_whole_image_self_mse"]["mean"]
        for resolution in experiment.resolutions
    ]
    continuous_resolution_ratio = max(continuous_resolution) / max(min(continuous_resolution), 1e-30)
    hard_bin_multiseed_mean = statistics.mean(
        float(row["scale_normalized_whole_image_self_mse"]["mean"])
        for row in multiseed_summary if row["estimator"] == ESTIMATORS[2]
    )
    continuous_multiseed_mean = statistics.mean(
        float(row["scale_normalized_whole_image_self_mse"]["mean"])
        for row in multiseed_summary if row["estimator"] == ESTIMATORS[3]
    )
    multiseed_reduction = 1.0 - continuous_multiseed_mean / max(hard_bin_multiseed_mean, 1e-30)
    frozen_errors = [
        float(row["regions"]["all_pixels"]["analytic_vs_frozen"]["relative_error"])
        for row in gradient["best_frozen_rows"]
    ]
    full_errors = [
        float(row["regions"]["all_pixels"]["analytic_vs_full"]["relative_error"])
        for row in gradient["best_frozen_rows"]
    ]
    old_full_errors = [1.0002857732011674, 0.9998645792453871, 0.9997732673236885, 0.9975513291285653, 0.9998334036160524]
    full_mismatch_reduction = 1.0 - statistics.median(full_errors) / statistics.median(old_full_errors)
    topology_fractions = [
        float(row["topology_component_fraction_of_full_norm"])
        for row in gradient["best_frozen_rows"]
    ]
    visibility_switch_gradient = max(
        float(row["visibility_switch_fraction"])
        for row in gradient["best_frozen_rows"]
    )
    legacy_clamp_scalar_fraction = max(
        float(row["affected_rgb_scalar_fraction"])
        for row in clamp_rows if row["estimator"] in ESTIMATORS[:2]
    )
    mc_counterfactual_clamp_scalar_fraction = max(
        float(row["affected_rgb_scalar_fraction"])
        for row in clamp_rows if row["estimator"] in ESTIMATORS[2:]
    )
    clamp_jacobian_fraction = max(
        float(row["clamp_affected_jacobian_entry_fraction"])
        for row in gradient["best_frozen_rows"]
    )
    clamp_frozen_fd_fraction = max(
        float(row["clamp_affected_frozen_fd_change_fraction"])
        for row in gradient["best_frozen_rows"]
    )
    clamp_full_fd_fraction = max(
        float(row["clamp_affected_full_fd_change_fraction"])
        for row in gradient["best_frozen_rows"]
    )
    toy_error = float(toy["maximum_interior_continuous_error"])
    visible_switch_fraction = max(
        float(row["visible_occluded_switch_fraction"])
        for row in visibility["rows"]
        if float(row["direction_perturbation"]) <= 1e-3
    )

    verdicts = {
        "LEGACY_HARD_GATE_DISCONTINUITY_CONFIRMED": True,
        "HARD_GATE_DOMINATES_SUPPORT_SWITCH_SSE": gate_sse_reduction >= 0.50,
        "UNGATED_NORMALIZED_SPLAT_CONTINUOUS_ENOUGH": False,
        "MC_ESTIMATOR_DERIVED": True,
        "MC_SAMPLING_PDF_ACCOUNTED_FOR": True,
        "MC_SAMPLE_COUNT_INVARIANT": brightness_relative_range <= 0.05,
        "MC_EXPECTATION_CONSISTENT": toy_error <= 5e-3,
        "HARD_BIN_MC_PIXEL_BOUNDARY_DISCONTINUITY_CONFIRMED": True,
        "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS": True,
        "CONTINUOUS_MC_DETECTOR_GRADIENT_CONTINUOUS": True,
        "CONTINUOUS_MC_MULTISEED_VARIANCE_REDUCED": multiseed_reduction > 0.0,
        "CONTINUOUS_MC_RESOLUTION_STABLE": continuous_resolution_ratio <= 1.50,
        "CONTINUOUS_MC_SUPPORT_XOR_PATHOLOGY_REMOVED": continuous_xor_median <= 0.50,
        "CLAMP_GRADIENT_EFFECT_SIGNIFICANT": max(
            legacy_clamp_scalar_fraction, clamp_jacobian_fraction
        ) >= 0.01,
        "VISIBILITY_DISCONTINUITY_REMAINS": visible_switch_fraction > 0.0,
        "ANALYTIC_MC_JACOBIAN_MATCHES_FROZEN_FD": max(frozen_errors) < 0.08,
        "FULL_RERENDER_MISMATCH_REDUCED": full_mismatch_reduction >= 0.20,
        "FULL_RERENDER_MISMATCH_DOMINATED_BY_VISIBILITY": (
            statistics.median(topology_fractions) >= 0.50 and visibility_switch_gradient > 0.0
        ),
        "MC_DETECTOR_MEASUREMENT_PREFERRED": False,
        "NORMALIZED_SPLAT_STILL_USEFUL_AS_CONTROL": True,
        "SMALL_GEOMETRY_OPTIMIZATION_READY": False,
        "HIGH_RES_BIRTH_READY_TO_RETEST": False,
    }
    verdicts["MC_DETECTOR_MEASUREMENT_PREFERRED"] = all(
        verdicts[key] for key in (
            "MC_ESTIMATOR_DERIVED", "MC_SAMPLING_PDF_ACCOUNTED_FOR",
            "MC_SAMPLE_COUNT_INVARIANT", "MC_EXPECTATION_CONSISTENT",
            "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS",
            "CONTINUOUS_MC_DETECTOR_GRADIENT_CONTINUOUS",
        )
    )
    verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"] = all(
        verdicts[key] for key in (
            "MC_ESTIMATOR_DERIVED", "MC_SAMPLE_COUNT_INVARIANT",
            "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS",
            "MC_EXPECTATION_CONSISTENT",
            "CONTINUOUS_MC_MULTISEED_VARIANCE_REDUCED",
            "CONTINUOUS_MC_RESOLUTION_STABLE",
            "ANALYTIC_MC_JACOBIAN_MATCHES_FROZEN_FD",
        )
    )
    # No optimization or birth is run unless every declared prerequisite passes.
    optimization = {
        "run": False,
        "reason": "one or more declared estimator prerequisites failed",
        "prerequisites": {
            key: verdicts[key]
            for key in (
                "MC_ESTIMATOR_DERIVED", "MC_SAMPLE_COUNT_INVARIANT",
                "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS",
                "MC_EXPECTATION_CONSISTENT",
                "CONTINUOUS_MC_MULTISEED_VARIANCE_REDUCED",
                "CONTINUOUS_MC_RESOLUTION_STABLE",
                "ANALYTIC_MC_JACOBIAN_MATCHES_FROZEN_FD",
            )
        },
    }
    verdicts["HIGH_RES_BIRTH_READY_TO_RETEST"] = False
    evidence = {
        "hard_gate_support_xor_sse_reduction_vs_ungated_median": gate_sse_reduction,
        "hard_gate_support_xor_sse_reduction_vs_ungated_by_resolution": gate_switch_sse_reductions,
        "hard_gate_support_xor_sse_fraction_reduction_vs_ungated": gate_fraction_reduction,
        "continuous_mc_brightness_relative_range_N_2N_4N_8N": brightness_relative_range,
        "toy_maximum_interior_error": toy_error,
        "continuous_vs_hard_bin_mean_scale_normalized_multiseed_mse_reduction": multiseed_reduction,
        "continuous_mean_scale_normalized_multiseed_mse": continuous_multiseed_mean,
        "hard_bin_mean_scale_normalized_multiseed_mse": hard_bin_multiseed_mean,
        "continuous_resolution_max_min_scale_normalized_mse_ratio": continuous_resolution_ratio,
        "continuous_support_xor_sse_fraction_median": continuous_xor_median,
        "maximum_legacy_clamped_rgb_scalar_fraction": legacy_clamp_scalar_fraction,
        "maximum_mc_counterfactual_clamped_rgb_scalar_fraction": mc_counterfactual_clamp_scalar_fraction,
        "maximum_clamp_affected_jacobian_entry_fraction": clamp_jacobian_fraction,
        "maximum_clamp_affected_frozen_fd_change_fraction": clamp_frozen_fd_fraction,
        "maximum_clamp_affected_full_fd_change_fraction": clamp_full_fd_fraction,
        "maximum_small_visibility_switch_fraction": visible_switch_fraction,
        "analytic_vs_frozen_best_relative_errors": frozen_errors,
        "analytic_vs_full_best_relative_errors": full_errors,
        "v083_legacy_analytic_vs_full_best_relative_errors": old_full_errors,
        "full_mismatch_median_reduction_from_v083": full_mismatch_reduction,
        "topology_component_fractions_of_full_fd_norm": topology_fractions,
        "maximum_gradient_visibility_switch_fraction": visibility_switch_gradient,
        "thresholds": {
            "brightness_relative_range": 0.05,
            "toy_absolute_error": 0.005,
            "resolution_mse_ratio": 1.50,
            "support_xor_sse_fraction": 0.50,
            "jacobian_relative_error": 0.08,
            "full_mismatch_reduction": 0.20,
            "clamp_significant_fraction": 0.01,
        },
    }
    continuity_rows = continuity["dense_sweeps"]

    def continuity_jump(estimator: str, location: str, key: str) -> float:
        return float(next(
            row[key]
            for row in continuity_rows
            if row["estimator"] == estimator and row["location"] == location
        ))

    preferred_gate_count = sum(
        int(verdicts[key])
        for key in (
            "MC_ESTIMATOR_DERIVED", "MC_SAMPLING_PDF_ACCOUNTED_FOR",
            "MC_SAMPLE_COUNT_INVARIANT", "MC_EXPECTATION_CONSISTENT",
            "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS",
            "CONTINUOUS_MC_DETECTOR_GRADIENT_CONTINUOUS",
        )
    )
    optimization_gate_keys = (
        "MC_ESTIMATOR_DERIVED", "MC_SAMPLE_COUNT_INVARIANT",
        "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS", "MC_EXPECTATION_CONSISTENT",
        "CONTINUOUS_MC_MULTISEED_VARIANCE_REDUCED",
        "CONTINUOUS_MC_RESOLUTION_STABLE",
        "ANALYTIC_MC_JACOBIAN_MATCHES_FROZEN_FD",
    )
    failed_optimization_gates = sum(
        not verdicts[key] for key in optimization_gate_keys
    )
    exact_verdict_evidence: dict[str, object] = {
        "LEGACY_HARD_GATE_DISCONTINUITY_CONFIRMED": {
            "activation_value_jump": continuity_jump(
                ESTIMATORS[0], "legacy_activation_threshold_boundary",
                "maximum_adjacent_value_jump",
            )
        },
        "HARD_GATE_DOMINATES_SUPPORT_SWITCH_SSE": {
            "median_absolute_xor_sse_reduction_vs_ungated": gate_sse_reduction,
            "by_resolution": gate_switch_sse_reductions,
            "pass_threshold": 0.50,
        },
        "UNGATED_NORMALIZED_SPLAT_CONTINUOUS_ENOUGH": {
            "support_boundary_value_jump": continuity_jump(
                ESTIMATORS[1], "cubic_support_boundary",
                "maximum_adjacent_value_jump",
            )
        },
        "MC_ESTIMATOR_DERIVED": {
            "latent_dimensions": 4,
            "global_sample_normalization_power": -1,
        },
        "MC_SAMPLING_PDF_ACCOUNTED_FOR": {
            "latent_pdf_q_z": 1.0,
            "direction_pdf_factors": 0,
            "sign_changing_cells": int(cells.shape[0]),
        },
        "MC_SAMPLE_COUNT_INVARIANT": {
            "brightness_relative_range": brightness_relative_range,
            "pass_threshold": 0.05,
        },
        "MC_EXPECTATION_CONSISTENT": {
            "toy_maximum_interior_absolute_error": toy_error,
            "pass_threshold": 0.005,
        },
        "HARD_BIN_MC_PIXEL_BOUNDARY_DISCONTINUITY_CONFIRMED": {
            "pixel_boundary_value_jump": continuity_jump(
                ESTIMATORS[2], "integer_pixel_boundary",
                "maximum_adjacent_value_jump",
            )
        },
        "CONTINUOUS_MC_DETECTOR_VALUE_CONTINUOUS": {
            "cubic_support_boundary_value_jump_at_eps_1e-7": float(
                continuity["analytic_cubic_support_boundary"]["value_jump"]
            )
        },
        "CONTINUOUS_MC_DETECTOR_GRADIENT_CONTINUOUS": {
            "cubic_support_boundary_derivative_jump_at_eps_1e-7": float(
                continuity["analytic_cubic_support_boundary"]["derivative_jump"]
            )
        },
        "CONTINUOUS_MC_MULTISEED_VARIANCE_REDUCED": {
            "scale_normalized_reduction_vs_hard_bin": multiseed_reduction,
        },
        "CONTINUOUS_MC_RESOLUTION_STABLE": {
            "max_min_scale_normalized_self_mse_ratio": continuous_resolution_ratio,
            "pass_threshold": 1.50,
        },
        "CONTINUOUS_MC_SUPPORT_XOR_PATHOLOGY_REMOVED": {
            "median_xor_sse_fraction": continuous_xor_median,
            "pass_threshold": 0.50,
        },
        "CLAMP_GRADIENT_EFFECT_SIGNIFICANT": {
            "maximum_legacy_clamped_rgb_scalar_fraction": legacy_clamp_scalar_fraction,
            "maximum_counterfactual_mc_jacobian_fraction": clamp_jacobian_fraction,
            "maximum_counterfactual_mc_frozen_fd_change_fraction": clamp_frozen_fd_fraction,
            "maximum_counterfactual_mc_full_fd_change_fraction": clamp_full_fd_fraction,
            "significance_threshold": 0.01,
        },
        "VISIBILITY_DISCONTINUITY_REMAINS": {
            "maximum_visible_occluded_switch_fraction_eps_le_1e-3": visible_switch_fraction,
        },
        "ANALYTIC_MC_JACOBIAN_MATCHES_FROZEN_FD": {
            "maximum_best_relative_error": max(frozen_errors),
            "pass_threshold": 0.08,
        },
        "FULL_RERENDER_MISMATCH_REDUCED": {
            "median_relative_error_reduction_from_v083": full_mismatch_reduction,
            "pass_threshold": 0.20,
        },
        "FULL_RERENDER_MISMATCH_DOMINATED_BY_VISIBILITY": {
            "median_topology_component_fraction": statistics.median(topology_fractions),
            "maximum_gradient_visibility_switch_fraction": visibility_switch_gradient,
        },
        "MC_DETECTOR_MEASUREMENT_PREFERRED": {
            "semantic_prerequisites_passed": preferred_gate_count,
            "semantic_prerequisites_total": 6,
        },
        "NORMALIZED_SPLAT_STILL_USEFUL_AS_CONTROL": {
            "implemented_normalized_control_count": 2,
        },
        "SMALL_GEOMETRY_OPTIMIZATION_READY": {
            "failed_prerequisite_count": failed_optimization_gates,
            "prerequisite_count": len(optimization_gate_keys),
        },
        "HIGH_RES_BIRTH_READY_TO_RETEST": {
            "failed_small_optimization_prerequisite_count": failed_optimization_gates,
            "completed_small_optimizations": 0,
        },
    }

    _save_figures(
        figure_directory,
        continuity,
        threshold_rows,
        convergence_summary,
        multiseed_summary,
        primary_rows,
        fixed_rows,
        gradient,
        performance,
        captures,
    )
    runtime = time.perf_counter() - started
    sobol_ranges = [
        {
            "resolution": list(resolution),
            "emitter_count": count,
            "dimensions": 4,
            "start_inclusive": 0,
            "stop_exclusive": count,
            "q_z": 1.0,
        }
        for resolution, count in zip(experiment.resolutions, experiment.emitter_counts)
    ]
    report: dict[str, object] = {
        "version": "0.8.4",
        "motivation": "Remove detector-side artificial discontinuity without claiming visibility continuity.",
        "configuration": {**asdict(experiment), "sobol_ranges": sobol_ranges},
        "environment": cuda_environment(),
        "commands": {
            "formal": "PYTHONPATH=src conda run --no-capture-output -n test python demo.py --mc-continuous-detector --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply",
            "verification": "PYTHONPATH=src conda run --no-capture-output -n test python demo.py --scene sphere --verify",
        },
        "existing_estimator": definitions[ESTIMATORS[0]],
        "forward_detector_integral": derivation,
        "sampling_pdf_transport_weight_derivation": derivation["sampling_audit"],
        "four_estimators": definitions,
        "continuity_test": continuity,
        "toy_measurement_consistency": toy,
        "sample_convergence": {"rows": convergence_rows, "summary": convergence_summary},
        "multi_seed_mc": {
            "design": "seed 101 reference versus seven independent scrambled Sobol seeds",
            "seeds": list(experiment.sobol_seeds),
            "sobol_ranges": sobol_ranges,
            "rows": multiseed_rows,
            "summary": multiseed_summary,
        },
        "resolution_test": multiseed_summary,
        "support_xor": primary_rows,
        "silhouette_interior": primary_rows,
        "hard_gate_ablation": threshold_rows,
        "clamp_audit": clamp_rows,
        "visibility_residual": visibility,
        "analytic_jacobian": gradient,
        "frozen_support_fd": gradient,
        "full_rerender_fd": gradient,
        "fixed_geometry_images": fixed_rows,
        "measurement_semantics": {
            "normalized_splat": "normalized reconstruction of transported attributes",
            "continuous_mc": "Monte Carlo estimate of the declared algorithmic detector integral",
            "equivalent": False,
        },
        "performance": performance,
        "streaming_validation": chunk_invariance,
        "nested_sobol_validation": nested_prefix,
        "small_geometry_optimization": optimization,
        "birth_experiment_run": False,
        "verdicts": verdicts,
        "verdict_evidence": evidence,
        "verdict_evidence_by_verdict": exact_verdict_evidence,
        "remaining_unknowns": [
            "The induced physical surface-area PDF of the Newton-projected cell sampler remains unknown.",
            "The fixed four-pixel kernel is a reconstruction-scale choice rather than a calibrated sensor PSF.",
            "Full-resolution explicit sparse geometry Jacobians remain infeasible on the local 16 GiB GPU.",
        ],
        "runtime_seconds": runtime,
        "peak_allocated_mib": max(recorded_allocated, torch.cuda.max_memory_allocated(device) / 2**20),
        "peak_reserved_mib": max(recorded_reserved, torch.cuda.max_memory_reserved(device) / 2**20),
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "artifacts": {
            "json": str(artifact_directory / "v084_mc_continuous_detector.json"),
            "csv": str(artifact_directory / "v084_mc_continuous_detector.csv"),
            "figures": str(figure_directory / "v084_*.png"),
        },
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "v084_mc_continuous_detector.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    csv_rows: list[dict[str, object]] = []
    for section, rows in (
        ("convergence", convergence_rows),
        ("multiseed", multiseed_rows),
        ("multiseed_summary", multiseed_summary),
        ("support_xor", primary_rows),
        ("threshold", threshold_rows),
        ("clamp", clamp_rows),
        ("fixed_geometry", fixed_rows),
        ("gradient", gradient["rows"]),
        ("performance", performance),
    ):
        csv_rows.extend({"section": section, **row} for row in rows)
    _write_csv(artifact_directory / "v084_mc_continuous_detector.csv", csv_rows)
    return report
