# v0.9.6 — frozen-chart topology capacity and RGB penetration

## 1. CUDA environment

The hard CUDA gate passed in the existing test Conda environment: PyTorch
`2.5.1+cu124` (CUDA runtime `12.4`) used
`Quadro RTX 5000`. The managed command sandbox initially hid the device;
running the known environment in host context exposed it. No package was
installed or changed, and no CPU density substitute or HPC resource was used.

## 2. Frozen-chart topology analysis

The tested field remains exactly `F_theta=||x||^2-1-||2x|| h_theta(x)` with
fixed sphere charts and compact Wendland support. It has the structural
invariant `F_theta(0)=-1`, because the reference gradient norm vanishes at the
origin. Therefore a centered conventional torus whose origin is outside the
solid is excluded. This does not imply that every zero set is a normal graph or
that all genus-1 members are excluded; the capacity construction below is a
counterexample to that stronger claim. Along `x=(1+s)p`, for `s>-1`,
`G=(1+s)^2-1-2(1+s)h((1+s)p)` and
`dG/ds=2(1+s)-2h-2(1+s) grad(h).p`. Here each jet's `h` is its tangent
polynomial times the planar Wendland window and the flat-centered quintic C2
normal collar. Hence coefficient-dependent bounds require simultaneous bounds
on both `h` and its collar/window/polynomial derivative. No useful global bound
holds for the fitted coefficient ranges; the grid bounds below are kept
explicitly empirical.

## 3. Representation capacity

Phase A alone used scalar/geometry oracle data from a shifted torus with
`(major, minor, center_x)=(0.5,0.3,0.5)`, chosen so the invariant origin lies in
the solid tube. The first certified member is p=1, K=256, radius=1.1
(768 coefficients), with weighted fit RMSE
`0.216877570599`. Its independently extracted 96^3 and 144^3
meshes are each one component, watertight, Euler 0, genus 1, do not touch the
domain boundary, and have minimum boundary clearance
`0.144067`.
The successful state is saved as `artifacts/v096_theta_T.pt` with SHA-256
`e137b58e55d8184b67feecf537575435a6a5ce58b0541d667dc08311592e9d4c`. A p=2 K=128 member also passed the 72^3 screen. None of the
tested p=0 K=128--1024 configurations passed; that is an empirical capacity
failure in this sweep, not a theorem about p=0.

## 4. Normal-fiber root behavior

All 2048 sphere fibers have exactly one simple root. The certified genus-1
member has 1959 one-root and 89 three-root fibers (multi-root fraction
`0.043457`), plus 5 sampled near-even-root candidates;
its minimum refined root slope magnitude is `0.0297353`.
The sampled `min dG/ds` changes from
`0.04` on the sphere to
`-5.75293` for the genus-1 state.
These dense directional scans demonstrate loss of the one-root normal-graph
regime for this member, but are explicitly empirical and not a global
monotonicity theorem.

## 5. Same-family genus-1 construction

The immutable Phase-B target image was freshly rendered from the saved p=1
coefficient vector with the existing real CURRENT/C3 finite-packet operator.
After target rendering, optimization received only its RGB tensor. The target
field, coefficients, mesh, genus, hole location, and topology labels are absent
from `_image_step`; every line-search candidate is a newly extracted real zero
set with newly traced transport. The target coefficient direction is used only
after step construction for the explicitly labeled oracle observability audit.

## 6. Image-driven topology penetration

From the exact zero-coefficient sphere, five accepted fresh-retrace steps reduce
two-view MSE from `0.172720837298` to
`0.138319926729` (19.92%). The sixth
proposal exhausts 24 fresh-retrace trials spanning three damped Gauss--Newton
models, a steepest-descent fallback, and six step lengths. The run therefore
stops by a declared line-search condition despite a 100-step maximum; it does
not silently use a five-step budget. Final 72^3, 96^3, and 144^3 audits remain
one-component watertight Euler-2 genus-0 surfaces. All 2048 final fibers still
have exactly one root. The required genus-1 penetration is not observed.

## 7. Emergent critical mode

No genuine zero-set critical event emerged. The minimum sampled true spatial
gradient ends at `1.12164` and never approaches zero. The distinct
frozen-chart denominator drops to `0.136713`; this is chart
conditioning, not a zero spatial gradient, and the two quantities are not
conflated. Root scans find neither a multi-root fiber nor a near-even-root
candidate at the endpoint. Consequently no data-driven one-dimensional
discriminant or crossing direction is asserted.

## 8. Image observability

The 6144x768 image Jacobian has effective ranks 243--298 at the recorded
states. The normalized oracle direction has 0.445--0.611 projection into the
measured row space, so it is not wholly invisible; however, its cosine with the
local negative image gradient is only -0.00868 to 0.00994. Thus the observed
failure is not pure representation failure or a total Jacobian nullspace. In
this local optimizer the RGB gradient does not point toward the available
genus-1 state, while chart conditioning deteriorates. The backward-view control
reduces MSE only `2.12%`, versus
`19.92%` for informative views. Conversely, the
same-topology ellipsoid control reduces MSE `98.37%`,
showing that the matched optimizer can perform ordinary deformation.

## 9. Need for complex continuation

Complex continuation is not activated. The real trajectory never reaches a
genuine critical zero, a multi-root transition, or a demonstrated real
continuation bottleneck. Adding a complex bridge here would force a topology
story rather than diagnose one.

## 10. Analytic-torus generalization

Not run. The prescribed experiment order made this generalization conditional
on successful same-family RGB penetration, which did not occur. The Phase-A
shifted analytic torus is only a geometry-oracle capacity construction and is
not presented as image-driven generalization.

## 11. Limitations

This is one shifted-torus capacity oracle, one seed, three 32x32 views, a
1024-source CURRENT/C3 diagnostic, frozen p=1 sphere charts, and a local
Gauss--Newton/steepest line search. Capacity is certified by 96^3/144^3 mesh
agreement and fiber scans, not by a formal characterization of the entire
function family. The image result is a reachability failure for this optimizer
and observation setup, not an impossibility theorem. An exploratory unstreamed
target render OOMed; exact emitter-axis transport chunking at 128 emitters kept
the measured run to `1221.998` MiB peak CUDA
allocation without changing quadrature or finite-packet equations. The original
allocator message was not persisted, so an exact pre-fix byte count is not
invented.

## 12. Final verdict

`CAPACITY_YES_REACHABILITY_NO`

The frozen-chart family does contain real genus-1 zero sets, disproving the
strong topology-preserving interpretation. The tested RGB-only trajectory does
not penetrate from the exact sphere into that basin: it remains in the
single-root, genus-0 regime and stops when all fresh-retrace proposals fail.
