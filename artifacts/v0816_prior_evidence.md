# Phase 0: corrected interpretation of preceding ablations

Geometry remains the zero set of a scalar field with compact Wendland coefficient DoFs. The finite-packet/ball radius controls scene-side attenuation, not detector footprint. The four directional/orthographic views and fixed cubic response are not a perspective-camera model. Global and geodesic sources share downstream color, gate, lobe, transport and detector definitions.

## Retained measurements

- v0.8.13: 1,024 charts, PER_CHART_PLUS_RING, 1920×1080, four views; 1,048,576 / 2,359,296 / 4,194,304 emitters; C1/C2/C3 integrate the same detector with 1×1 / 2×2 / 4×4 midpoint quadrature. Source coverage and visible sampling texture improve with density. The reported 26.09% → 7.83% → 2.51% coverage uses the *old reference's foreground mask*. The reported 89.3% unmatched-gradient-energy reduction is a numerical change in that historical reference-dependent proxy, not independent evidence of physical fidelity. Capture integration adds no new scene observations.
- v0.8.14: weighted surface-density CV .29503 (geodesic) versus .11124 (global), a 62.3% reduction. The reported .5-pixel gaps 2.51% versus 1.94% again use the old foreground support. Extra geodesic tangential redistribution under lambda exists, but was not established as the dominant failure. Global source removes large petals/rings while retaining fine sampling texture. Its bounded nonzero lambda-17 own-forward JVP passes FD closure at epsilon 1e-5: relative L2 4.78e-8, cosine approximately one, active-component sign agreement 100%. This validates the tested operator/column, not every candidate or visibility event.
- v0.8.15: corrected dense reference uses 134,217,728 surface samples, independent BVH hard visibility on the same grid's triangulated isosurface, matched source-energy/color/gate/camera/C3 conventions. Adjacent quadrature levels differ by <.73% RGB and <6% spatial gradient. These empirical tolerances are not zero-error or physical-camera guarantees.

## Superseded interpretations

The original 4,096-sample hard-splat target remains a historical regression fixture, not a converged Full-HD reference. Old MSE and near-zero spatial-gradient cosine cannot support a claim that source density or global sampling failed to improve Full-HD fidelity, or that the geometry Jacobian is broken.

The unchanged images evaluated against the corrected reference give:

| Source | MSE | Spatial-gradient cosine |
|---|---:|---:|
| Geodesic | .03210933 | .47573680 |
| Global fixed measure | .01847674 | .47258620 |

Global sampling has about 42.5% lower MSE and is the preferred representation for the new quadrature test. The small cosine difference is not compelling directional evidence against a reference with several-percent gradient convergence residual. Image spatial gradients are not dI/dlambda. None of these comparisons uniquely attributes residual error to source redistribution, soft attenuation or geometry bandwidth.

## Phase 1 protocol and subsequent gate

Only global emitter count changes: 2^19, 2^20, 2^21, 2^22, 2^23, 2^24. Use seed-101 nested Sobol surface-area samples, persistent IDs and the same fixed-normal attachment. All coefficients remain zero, and all 64 basis coordinates/supports remain unchanged. Fixed scalar source mass S uses weights S/N; unchanged color creates small quadrature fluctuations in integrated RGB, which must stay within 1e-4 relative instead of being hidden by per-level renormalization.

Predeclared practical gate: every view's adjacent RGB relative L2 <1%, absolute relative whole-image MSE change <5%, every view's .5-pixel missing-reference-energy fraction <1% and 1-pixel missing-reference-energy fraction <.1%, with no >10% radial-banding-proxy increase and visual inspection for new structure. The corrected dense-reference foreground is used throughout. Projected counts include hidden emitters; coverage is not observability rank. During the initial low-N audit, before the high-N results, the proposed unweighted coverage gate was explicitly replaced by energy-weighted coverage: cubic detector boundary halos create irreducible center-distance gaps even at infinite surface density. Both raw foreground fractions and energy-weighted fractions are retained; no image or sampling parameter was changed.

MSE plateau alone does not permit birth. A failed gate stops this task after Phase 1. A passed gate requires candidate validation before sequential birth, and positive sequential evidence before multiview/batch extensions. Candidates must never add emitter charts, weights or sample topology. No Phase 2 result may be inferred from a Phase 1 spatial-gradient statistic.
