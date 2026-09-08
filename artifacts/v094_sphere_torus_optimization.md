# v0.9.4 — rendered sphere-to-torus optimization

## Setup

This controlled experiment starts at the real genus-0 field `c=-0.005` and
targets the real genus-1 field `c=+0.005` in the already validated family
`F_c = F_(a=1) + c W(||x||/0.8)`. The critical mode is the single existing
Wendland coefficient, `v=[1]`; `F_c(0)=c`, so its known discriminant is `c=0`.
Topology labels are used only after optimization.

All images are produced by the repository's existing real reference-direction,
first-arrival, bilinear renderer. Every current state, line-search candidate and
target is freshly traced. Complex geometry is never rendered. The observation
uses the same 64 persistent local neck source IDs/colors and two informative
32x32 views as v0.9.3; the backward view is the observability control. This is a
small operator experiment, not CURRENT/C3 production, complete-surface sampling,
Full-HD evidence, or a performance benchmark.

## Equations and trigger

At each iteration the direct sparse local image Jacobian produces `j=Jv`. We use
`g=j^T(I-I_target)`, `h=j^Tj`, `delta=-g/h` and trigger only when
`c0*(c0+delta)<0`. No topology label, gradient penalty or hand-coded hole target
enters this decision. The image block is constructed from 64 compact support
pairs and one affected parameter. At the first iteration it has sparse nnz
[846, 813, 0] of 3072 entries/view; the weak view has zero entries.

The first torus-target prediction is `c=-.005 -> c_pred=0.00562900435199`:
`||Jv||=15.7162301068`, `g=-2.62536289267` and
`h=246.999888769`. The full step increases re-traced loss, so the common
line search accepts alpha=.5 at `c=0.000314502175994`. This partial step
still crosses zero. Predicted and actual re-traced loss are deliberately kept
separate in the JSON.

## Run A — real-only baselines

The primary ordinary discrete real optimizer is allowed to evaluate real
parameters on either side of zero. It reaches `c=0.00500000000022` in
five iterations, changes genus 0->1 and reduces MSE from
3.48655898943e-05 to 1.93676492489e-27. It does not evaluate
the singular intermediate parameter; this is a parameter jump, not continuous
real zero-set continuation.

The additional continuity-constrained real tracker rejects steps whose parameter
segment crosses the known discriminant. With the same iteration budget it
approaches `c=-1.00820762868e-07`, remains genus 0 and finishes at MSE
1.07105517157e-05. Its pole gradient and implicit derivative trend
toward singular conditioning. This baseline answers a different question from
ordinary discrete optimization and must not be substituted for it silently.

## Run B — complex-capable

The complex-capable run has the identical real optimizer and accepted endpoints.
At the triggered crossing it audits/realizes the accepted jump via
`c(t)=(-c_start)^(1-t)c_end^t exp(i*pi*(1-t))`, integrates the full implicit
normal-line equation, and corrects it after every predictor step. The bridge has
max residual 8.75e-16, minimum sampled spatial-gradient
norm 0.426846664, minimum normal-line
denominator 0.423377382, and stays at least
0.000314502176 from `c=0`. Endpoint imaginary
residual is 9.47e-95; it agrees with an
independent real zero-set solve to 3.89e-16.
No complex state is rendered.

It reaches genus 1 and the same final loss 1.93676492489e-27 as the
ordinary discrete real run. Thus it beats the continuity-constrained real tracker,
but **does not beat or improve reachability over ordinary real parameter jumps**.
The bridge is a valid continuation mechanism, not an observed optimizer benefit
in this case.

## Run C — same-side control

For target `c=-.007`, every prediction remains on the negative side. Complex
capability is available but never activated. Final c is -0.007,
MSE 9.12e-33, and topology remains genus 0.

## Run D — weak observability

The backward view has `||Jv||=0`, `h=0`, no sign decision, no update and no
complex bridge. This is reported as underdetermined; it is not confused with
zero-set singularity or chart failure.

## Conditioning and topology

Each iteration separately records (1) the true critical branch spatial gradient,
(2) `grad F dot n_ref` for source-chart conditioning, and (3) `||Jv||` for image
observability. At initialization the critical pole gradient is
0.203345697, while the sampled
source gradient and chart denominator are 0.437175054
and 0.437175054. They are not conflated.

Topology is checked with the 144^3 extraction and an analytic meridional count.
Both agree: discrete real and complex runs finish genus 1; continuous real and
same-side runs finish genus 0. The diagnostic is not an optimization signal.

## Acceptance/falsification summary

- A real-only difficulty: **false for ordinary discrete optimization**, true only
  for continuous real zero-set tracking.
- Image observability and sign-crossing evidence: true.
- Regular complex bridge and valid real re-entry: true on the 64 tracked branches.
- Actual loss beyond ordinary real-only: **false**; results match exactly.
- Genus 0->1 endpoint: true for both discrete-real and complex-capable runs.
- Non-forcing and weak-observation controls: pass.

## 1. Mathematical feasibility

Supported locally: the triggered complex path connects the selected real sample
roots without hitting its discriminant and returns to an independently solved
real endpoint.

## 2. Numerical stability

Supported for this sampled bridge by residual, gradient, denominator, endpoint
and tangent-FD checks. It is not a global regularity proof for the complete
complex surface.

## 3. Image observability

Supported in two views; exactly absent in the backward control view.

## 4. Image-preferred crossing

Supported: rendering residual and sparse J predict `c0*c_pred<0`; an actual
re-traced half step crosses and improves loss.

## 5. Actual rendered-loss benefit

Not supported relative to ordinary discrete real-only optimization. Complex and
discrete-real runs have identical real iterates and losses. A benefit exists only
relative to the explicitly continuity-constrained real tracker.

## 6. Topology outcome

Genus changes 0->1 in both the discrete-real and complex-capable runs. Therefore
the topology result cannot be causally attributed to complex continuation here.

## 7. Limitations

Known one-dimensional critical coordinate, one analytic family, local neck
observations, one seed, small historical diagnostic renderer, and a complex path
for sampled roots only. Source birth/death and the vanishing cycle are not
globally continued; multiple critical modes are absent. Direct real jumps skip
the singular parameter and are valid for this optimizer. No claim of necessity,
novelty, production readiness or whole-surface complex homotopy is made.

**Final answer:** the experiment supports that real rendering loss can drive the
sphere-like field to a torus and that the trigger identifies a critical crossing.
It supports local complex bypass as a numerically valid optional bridge. It does
**not** support the stronger statement that the bypass provides useful additional
reachability or rendered-loss improvement over ordinary discrete real optimization.

Runtime: 10.206 seconds, excluding plots/tests/validation.
