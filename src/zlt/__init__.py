"""Zero-Set Forward Light Tracing research prototype."""

from .camera import PlanarCamera
from .fields import SphereField, TorusField, ZeroSetField, unit_normals
from .tracer import (
    PhotonBatch,
    RenderResult,
    TraceResult,
    emit_photons,
    first_zero_set_intersections,
    make_scene,
    render_first_arrival,
    trace_photons,
)

__all__ = [
    "PhotonBatch",
    "PlanarCamera",
    "RenderResult",
    "SphereField",
    "TorusField",
    "TraceResult",
    "ZeroSetField",
    "emit_photons",
    "first_zero_set_intersections",
    "make_scene",
    "render_first_arrival",
    "trace_photons",
    "unit_normals",
]
