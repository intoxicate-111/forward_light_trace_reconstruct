# v0.9.5 — dense manifold-jet sphere-to-torus optimization

The reconstruction field is exactly `F=||x||^2-1-||2x||h_theta(x)` with
all 128 p=0 coefficients initialized to zero. Fibonacci
centers are fixed independently of the target; radius is 1.1.
At the initial surface, median overlap is 70
jets and uncovered fraction is 0.

The immutable target is the real torus `(R,r)=(0.64,0.36)`.
Only its three rendered RGB images enter optimization. The target equation,
mesh, genus, hole location, and v0.9.4 coordinate never enter the loss or step.
The two informative directions are `[[0.0, 0.0, 1.0], [0.5500000000111592, 0.25000000000507233, 0.796868872516168]]`; the
side direction is the weak-observation control.

The measurement operator retains CURRENT finite-packet attenuation, current
gate/lobe and C3 detector. Every trial reconstructs a real zero set, forms a
deterministic triangle-area quadrature, and retraces transmission. The existing
v0.9.2 sparse-support CURRENT tangent proposes damped Gauss--Newton steps from
256 area-weighted samples. Predicted and actual trial losses are separately
stored in JSON. No complex geometry is rendered.

Run A accepts 4 real steps. Freshly traced MSE falls from
0.0193814081226 to 0.012683910516 (34.56%); all
128 coefficients activate. The minimum sampled spatial
gradient is 1.69109349219,
so it does not approach a true zero-set critical event. The independent 72^3
mesh audit is watertight, has Euler number 2, one
component, and genus 0.

Run B (sphere-like target) reduces MSE from 0.00159512939302
to 0.000229493874317 without changing genus. Run C's weak
view reduces 0.0934993046773 to
0.0819032806376, also without a critical event or topology
change. Density runs K=32/64 finish at 0.014537639498 and
0.0141028908452; neither changes topology. CUDA was
unavailable, so K=256/512/1024 and p=1/p=2 are explicitly not claimed.

## 1. Dense representation validity

Supported at the tested scale: zero coefficients reproduce the analytic unit
sphere exactly; the extracted mesh is one watertight genus-0 component and
coverage is complete. K=128 is a CPU-limited dense pilot, not the requested
upper density regime.

## 2. Image-driven deformation

Supported. Target information entered only through images, all accepted states
were freshly real-rendered, all coefficients activated, and Run A reduced MSE
by 34.56%. This is ordinary deformation, not evidence of a hole.

## 3. Emergent critical mode

Not supported. The lowest-gradient region and its local support block are
logged at every iteration, but `min ||grad F||=1.69109`
is far from zero. No one-dimensional discriminant can honestly be inferred,
so no `c0*c_pred<0` analysis is asserted.

## 4. Topology outcome

Negative: both 32^3 trajectory checks and the independent 72^3 final audit
remain one-component, watertight, Euler=2, genus 0. Appearance improvement is
not mislabeled as a torus.

## 5. Rendered-loss outcome

Run A improves by 34.56%, while higher density improves the equal
two-step comparison only modestly. The sphere-like control improves much more
readily and does not create unnecessary topology. The strongest boxed claim is
therefore **not supported** by this experiment.

## 6. Need for complex continuation

No. The required precursor—a real, image-observable critical event limiting
ordinary optimization—did not arise. Activating a complex bypass would force
the intended story rather than test it.

## 7. Limitations

CPU-only K=128 maximum, p=0 only, 32x32 three-view diagnostic images, 32^3
optimization surface extraction, four main steps, local tangent subsampling,
one target/seed and a wide support radius needed to influence the sphere
interior. The deterministic marching-cubes area quadrature is a controlled
readout approximation; it is not Full-HD evidence or a global impossibility
result. Runtime was 1181.063 seconds excluding this
report and validation.
