# v0.9.2 — local manifold normal-deformation jets

## 1. Exact code changes

New `src/zlt/manifold_jets.py`: normalized tangent frames, p=0/1/2 sparse
chart/mode evaluation, C2 compact extension, implicit adapter, explicit geometry
coefficient supports and exact spatial-gradient/Hessian interface.

New `src/zlt/jet_experiment.py`: existing CURRENT derivative accumulation,
six-mode FD, fixed-budget GN, six-coefficient block birth, persistence, finite
topology checks and replay audit. New `src/zlt/jet_report.py` generates tables and
figures. New `tests/test_manifold_jets.py` and `scripts/validate_v092.py` provide
tests and isolated validation. README and the v092 figure ignore exception are
the only edits to existing files for this task.

`basis_renderer.py`, `matched_jacobian.py`, `transverse_packet.py`, source
attachment, CURRENT transmission, visibility, projection, color, detector, loss
and four-view stacking are reused without edits. Preexisting dirty changes from
earlier tasks were preserved, not attributed to this branch.

## 2. Mathematical formulation

For a frozen local phase, use the actual current reference field `F_ref` and
frame `(e1,e2,n)` at each chart. Coordinates are
`u=(x-c)·e1/r`, `v=(x-c)·e2/r`, `z=(x-c)·n/r`.

`h = Σ W(sqrt(u²+v²)) Q(z) P_p(u,v)`,
`F_theta = F_ref - |grad F_ref| h`.

`P0=a`; `P1=a+b_u u+b_v v`; `P2=P1+q_uu u²+q_uv uv+q_vv v²`.
No polynomial depends on the normal coordinate. `Q` only extends the chart
compactly off-surface: it is one for `|z|<=1/2`; for
`t=clamp(2|z|-1,0,1)`, `Q=1-10t³+15t⁴-6t⁵`. It vanishes for `|z|>=1`.
This prevents an infinite tangent cylinder reaching the opposite sheet.

The sign follows `grad F·(h n)+delta F=0`. The scale is not assumed to be one:
`grad(delta F)=-|grad F_ref| grad h-h grad|grad F_ref|`.
Spatial Hessians differentiate this complete expression, including the reference
gradient scale. This is a first-order normal-deformation model; after finite
updates the exact incremental displacement is scaled by the frozen/current
gradient-norm ratio. Frames and reference are frozen for this small phase; no
large-deformation rebase experiment is claimed.

Reference initialization is the existing basis-only field
`F_ref=1-(16/3)W(|x|/2)`. There is no analytic sphere evaluation in reconstruction.
The target is an independent ellipsoidal coordinate warp of this same radial
profile, axes `(1.025,.985,1.015)`, not a target generated from the jet dictionary.
All arms share one frozen reference coefficient; reported budgets are additional
optimized DoFs, not a replacement claim for the entire v091 field.

## 3. Tests

New tests cover orthonormal/right-handed frames, invalid normals/orders,
normalized p=0/1/2 values, compact support, disabled baseline, C2 boundary behavior,
spatial gradients/Hessians versus FD, chart-origin Hessians, displacement sign and
scale, and derivative propagation through a deformed reference. The validation
script runs all project tests, compileall, demo verification, strict JSON/CSV,
figure decoding, finite raw arrays, historical hashes and `git diff --check`.
Exact machine-readable status is in `v092_validation.json`.

## 4. Fixed-jet results

All arms use four 64² views, 256 identical source IDs/weights, the same CURRENT
operator, GN damping rule and line search, radius 1.1 and seed 391. Four GN
iterations are allowed; all failed line-search trials are retained. Initial
image MSE is 0.0260961540 and independent radial MAE is 0.0117397586.

| DoFs | Representation | Centers | Final MSE | Radial MAE |
|---:|---|---:|---:|---:|
| 12 | scalar Wendland | 12 | 0.01303256 | 0.00716963 |
| 12 | p0 | 12 | 0.01168193 | 0.00678668 |
| 12 | p1 | 4 | 0.01915280 | 0.00964737 |
| 12 | p2 | 2 | 0.02098373 | 0.01064025 |
| 24 | scalar Wendland | 24 | 0.00256863 | 0.00324209 |
| 24 | p0 | 24 | 0.00206417 | 0.00292957 |
| 24 | p1 | 8 | 0.00649276 | 0.00526070 |
| 24 | p2 | 4 | 0.01650924 | 0.00914172 |

Per-step gradient quantiles, image-Jacobian nonzeros, condition numbers, damping,
trial outcomes, geometric errors and timings are recorded in the JSON. Field
support fanout is separately measured by the replay audit. Querying sparse
point/chart pairs precedes mode evaluation; there is no dense point-by-all-bases
field array. The small diagnostic image Jacobian is deliberately dense.

## 5. Order / representation efficiency

At MSE <=0.01304808 (half the initial error), scalar and p0 reach the threshold
with 12 centers/12 DoFs; p1 reaches it with 8 centers/24 DoFs; p2 does not reach
it within the tested budgets. Thus fewer centers can trade for more scalar
parameters, but higher order does not improve equal-budget quality here.

At the looser threshold 0.01957212, p1 uses 4 centers/12 DoFs and p2 uses
4 centers/24 DoFs; scalar/p0 use 12 centers/12 DoFs in the tested grid. These are
minimum tested budgets, not proof of globally minimal representation sizes.
The target has no flat patch. Pole/equator errors are included but cannot answer
whether flat regions prefer lower order or curved regions prefer p2.

An additional actual target-mean-curvature split uses the bottom/top quartile
of 1024 directions (curvature range 0.9473–1.0255). At 24 DoFs, low/high-curvature
radial MAE is scalar 0.002780/0.004651, p0 0.002631/0.004171,
p1 0.002120/0.009059, p2 0.005137/0.017022. P1 wins the lower-curvature subset,
but p0 wins the higher-curvature subset. This does not support an automatic
"higher curvature needs p2" rule; chart coverage and location remain confounded.
See `v092_curvature_regions.png` and `curvature_regions` in the JSON.

## 6. Birth experiment

Start from one p2 jet, optimize, then score the unused locations in an eight-site
candidate pool. Candidate centers/normals are reattached to the current surface.
For each six-column image block use `G=JᵀJ/M`, `b=Jᵀr/M`,
`delta=-(G+mu I)^-1 b` and predicted half-MSE reduction
`-bᵀdelta-0.5 deltaᵀG delta`. Scoring and insertion use the SAME frozen reference
gradient scale; old active coefficients remain jointly optimizable.

Add two blocks, with three optimization steps per stage. At equal final budget
3 jets/18 coefficients:

| Policy | Final MSE |
|---|---:|
| Loss/Jacobian-driven | 0.0110320575 |
| Random, seed 391 | 0.0200194762 |
| Farthest spatial coverage | 0.0183945968 |

The loss-driven policy wins this pilot. Full block scores, selected IDs, trial
losses and all intermediate states are retained. It is not a single-scalar
k+1 score. The joint post-birth gain is not presented as a pure isolated-block
prediction-validation metric.

## 7. Image derivative validation

Unchanged CURRENT analytic transport and detector tangents consume the new
explicit `dF/dtheta`/spatial derivative supports. All six coefficients were
checked at nonzero jet coefficients using 1024 sources, four 64² views and
epsilon `1e-3,1e-4,1e-5,1e-6`. Every mode passes at least two adjacent steps:
relative L2 <1e-4, cosine >.9999, norm ratio within .1%, sign agreement >.99.
Image derivative norms are respectively 117.512, 25.155, 23.811, 8.986, 5.110,
8.272. No selected mode is unobservable. Every epsilon row is preserved in
`v092_manifold_jet_fd.json`, including coarser steps outside the strict gate.

## 8. Failures and non-successes

Higher-order jets lose at equal scalar budgets. Scalar/24 exhausted its final
line search at a plateau; the trials are retained. No accidental component was
found in the sampled topology checks, which does not prove none exists globally.

The original *bitwise* disabled-image replay gate failed with maximum difference
8.88e-16 between separate CUDA scatter renders. That original audit is preserved
under `runs/v092_manifold_jets/audit_original_bitwise_gate.json`. Field equality
remains bitwise; final image equivalence uses explicit 1e-12 tolerance. The saved
image replay difference was 2.66e-15 in the original audit. This is a numerical
summation issue, not a changed measurement operator.

## 9. Limitations / runtime

One target, one seed and two budgets; 256-source images are visibly sparse and
are not converged images. There is no Full-HD, Bunny, HPC, global atlas, Eikonal,
radius/order learning, topology birth or optimized source measure. The tangent
window/collar differs geometrically from ambient scalar-ball supports, so p0 vs
scalar is not an identical-support ablation. Normal extension can affect
off-surface transmission. No claim of universal representation superiority.

The maximal sampled radial departure from the phase reference is 0.041394. For
jet reconstructions the frozen/current gradient-norm ratio is about 0.906–1.096;
finite steps therefore must not be described as exact `delta X=h n`. All 17
saved states have one watertight component at 64³; 512 independent radial scan
directions are also recorded. These are finite checks, not topology proofs.

Fixed/birth experiment wall time: 181.84 s, excluding FD, replay audit and plots.
The normal interpretation requires a regular surface (`|grad F_ref|>0`). The
exact gradient norm can be nonsmooth at off-surface critical points; the C2
claim applies to the chart windows, not to a globally smooth norm at such points.
Surface gradient minima are recorded for every optimization state.

Peak allocated CUDA memory: 77.15 MiB. Core field evaluation remains sparse, but
this experiment does not establish million-sample scaling.

## 10. Hypothesis / Q1–Q5

- Q1: Yes, the tangent normal-jet → implicit-field → existing renderer chain
  works without downstream renderer edits.
- Q2: Yes, constant, both linear and all three quadratic image derivatives pass
  the adjacent-epsilon test on this case.
- Q3: Partially: fewer centers are possible at tested thresholds, sometimes at
  higher scalar DoF cost. Equal-budget superiority is not supported; p0 wins.
- Q4: Yes on this pilot: full-block observation scoring beats random and spatial
  coverage at the same 18-DoF budget. More targets/seeds are needed.
- Q5: Yes in all saved finite-grid/ray checks: one closed component; no claim of
  guaranteed global topology preservation or topology-changing capacity.

Overall: the formulation and observation-driven block birth are supported as
a small local-deformation prototype. The stronger claim that quadratic jets
are generally a more efficient representation is not established.
