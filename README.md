# Zero-Set Forward Light Tracing

This experimental research prototype asks two deliberately narrow questions: can samples of a zero set act as local directional emitters, and does a compact local parameterization induce a sparse, stable geometry-to-image Jacobian? Version 0.1 isolates forward transport; version 0.2 adds local geometry parameters and derivative analysis without inverse reconstruction or a conventional lighting model. This project makes no claim of novelty.

## Model

Geometry is defined only by the zero level set

\[
S=\{x\in\mathbb R^3:F(x)=0\}.
\]

Crucially, **\(F\) is not assumed to be a signed distance function**: neither \(|F(x)|=d(x,S)\) nor \(\|\nabla F\|=1\) is used. Temporary points \(p_i\in S\) receive analytic coordinate colors and emit unit-speed photon packets around the normalized gradient \(n_i=\nabla F(p_i)/\|\nabla F(p_i)\|\). Directions follow \(p(\omega\mid n_i)\propto\max(0,n_i^T\omega)^k\), and each trajectory is

\[
x(t)=p_i+\omega t.
\]

The default is a synchronized pulse, \(t^{emit}=0\); a uniform finite interval is also available. Packet energy is recorded as \(1/M\) for \(M\) packets per emitter, while the specified first-arrival color rule is deliberately energy-independent.

```text
zero-set point
    |
    | emit
    v
photon trajectory
    |
    +---- hit zero set ---> absorbed
    |
    +---- hit camera -----> pixel
```

For a trajectory that intersects the finite detector at \(t^{cam}>0\), the tracer samples \(F(p+\omega t)\) over \((\epsilon,t^{cam}-\epsilon)\), takes the earliest sign-change bracket, and refines it by bisection. A root \(t^{surf}<t^{cam}\) absorbs the packet; otherwise its detector arrival is \(t^{emit}+t^{cam}\). This is the only visibility rule—there is no mesh collision, sphere tracing, or z-buffer pass. The detector is a physical rectangle with a center, orthonormal frame, width, height, and pixel resolution; ray-plane intersection is analytic.

Hard rendering assigns each pixel the color of its minimum-arrival packet. Soft diagnostic mode assigns packets in one pixel normalized weights

\[
w_j=\frac{\exp[-\beta(t_j-t_{min})]}{\sum_k\exp[-\beta(t_k-t_{min})]}.
\]

Both modes explicitly build a COO transport matrix and compute

\[
I=T(F)C.
\]

In hard mode, the winning pixel/emitter entry is one. In soft mode, \(T_{ui}\) is the sum of the normalized packet weights at pixel \(u\) emitted by point \(i\); duplicate COO entries are coalesced. The demo independently accumulates packet colors and checks the result against sparse matrix multiplication.

## Install and run

Python 3.10 or newer is required.

```bash
python -m pip install -e .
python demo.py --scene sphere
python demo.py --scene torus
python demo.py --scene torus --mode soft
python demo.py --scene torus --verify
python demo.py --scene torus --output torus.png
python demo.py --jacobian --scene sphere
python demo.py --jacobian --scene torus
python demo.py --observability --scene sphere
python demo.py --jacobian --scene sphere --normal-mode current
python demo.py --observability --scene sphere --analysis-output outputs/diagnostics
```

The two implemented fields are exactly an analytic sphere and analytic torus. With the default seed, the sphere is the convex normal/camera sanity check. The side-view torus activates non-convex self-occlusion: some rays emitted from its inner wall cross the hole and re-intersect the opposite tube before reaching the detector. Every run reports zero-set and normal errors, hit and absorption fractions, sparse shape/nnz/density, operator fan-in/fan-out, direct-versus-sparse error, and measured pipeline runtime. `--verify` additionally runs deterministic hand checks for all acceptance gates through production code.

Useful controls are `--cone-power`, `--emission-interval`, `--beta`, `--emitters`, `--packets`, `--root-samples`, `--resolution`, and `--seed`. CPU execution is the reference path and uses float64 for transparent numerical checks.

## v0.2: local geometry Jacobian

The learnable geometry state is a small vector of coefficients in compact Wendland \(C^2\) bases:

\[
F(x;\lambda)=F_0(x)+\sum_k\lambda_kB_k(x),\qquad
B_k(x)=(1-q)_+^4(4q+1),\quad q=\frac{\|x-c_k\|}{r_k}.
\]

Centers and radii remain fixed. A reference emitter \(p_j^0\) keeps its identity by moving only along its reference normal, \(p_j(\lambda)=p_j^0+s_j(\lambda)n_j^0\), where a deterministic bracketed scalar solve enforces

\[
F(p_j(\lambda);\lambda)=0.
\]

The implementation does not differentiate through root-solver iterations. It explicitly evaluates the implicit derivative

\[
\frac{\partial s_j}{\partial\lambda_k}
=-\frac{B_k(p_j)}{\nabla F(p_j)^Tn_j^0},
\qquad
\frac{\partial p_j}{\partial\lambda_k}
=n_j^0\frac{\partial s_j}{\partial\lambda_k}.
\]

For derivative analysis, one reusable emitter-major low-discrepancy cone pattern fixes photon identity across every baseline, perturbation, and finite difference. Reference-direction mode fixes its orientation to \(n_j^0\) and is the primary controlled experiment; current-normal mode recomputes orientation from the perturbed field as a secondary diagnostic. Fixed first-arrival owners are bilinearly splatted to four neighboring detector pixels, producing the RGB-row convention

\[
\boxed{J=\frac{\partial\operatorname{vec}(I)}{\partial\lambda}}.
\]

Pixel-support statistics collapse the three RGB rows with their vector norm. Basis support predicts image support by selecting influenced emitters and projecting their fixed owner footprints. The analysis reports absolute thresholds \(10^{-12},10^{-9},10^{-6},10^{-4}\), a relative \(10^{-3}\) threshold, per-column support and energy, finite-difference error, and multiview singular spectra. `--analysis-output` optionally writes no more than four compact figures: basis support, example columns, matrix sparsity, and—when requested—the multiview singular spectrum.

The Jacobian is evaluated only inside a **fixed transport cell**: detector validity, zero-set absorption, first-arrival ownership, and bilinear footprint indices remain fixed. Perturbed forward traces independently detect changes to those choices and report them as transport events. Thus

\[
\boxed{\text{local differentiability}\neq\text{global differentiability}.}
\]

## Limitations

The prototype deliberately omits inverse reconstruction, differentiability across collision/visibility topology changes, dynamic geometry parameter birth, optimized basis centers or radii, neural fields, secondary reflection, indirect illumination, refraction, participating media, physical camera lenses, and physically calibrated radiometry. It also misses tangent zero-set intersections that touch without a sign change. The v0.1 renderer uses finite Monte Carlo emission; v0.2 derivative analysis instead uses fixed deterministic directions. Detector misses produce no contribution, deformations must retain the intended local normal-line root, and only sphere- and torus-based fields are supported.
