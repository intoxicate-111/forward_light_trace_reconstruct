# Zero-Set Forward Light Tracing

This experimental research prototype asks three deliberately narrow questions: can samples of a zero set act as local directional emitters, does a compact local parameterization induce a sparse geometry-to-image Jacobian, and can differential evidence predict the utility of a geometry parameter before it exists? Version 0.1 isolates transport, v0.2–v0.2.1 validate local geometry derivatives and CUDA scaling, and v0.3a falsifies one pre-birth scoring rule in a controlled inverse problem. This project makes no claim of novelty.

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
python demo.py --benchmark-cuda --scene sphere
python demo.py --benchmark-cuda --scene torus --cuda-resolution 1920 1080
python demo.py --benchmark-scaling
python demo.py --birth --birth-csv artifacts/v03a_candidates.csv --birth-figures figures
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

## v0.2.1: CUDA and Full-HD scaling

The CUDA path uses the same float64 PyTorch field, collision, camera, and footprint operations as the CPU reference. The benchmark machine has one NVIDIA Quadro RTX 5000 (16 GiB), driver 580.178.04, PyTorch 2.5.1+cu124, and CUDA 12.4. In the deterministic 16×16 equivalence case, absorption states and sparse indices match exactly, the image maximum difference is \(1.55\times10^{-15}\), and the median relative Jacobian difference is \(3.52\times10^{-16}\).

The Full-HD path is sparse by construction. It creates Jacobian candidates only for fixed first-arrival photons, their four-pixel bilinear footprints, and the 32 locally supported geometry coefficients. It never allocates a dense \((3HW)\times K\) Jacobian or \(HW\times N_{emitter}\) transport matrix. A dense \(HW\times3\) output image is the only resolution-sized dense object. The physical detector remains a fixed square throughout the sweep; 960×540 and 1920×1080 therefore use rectangular pixels so resolution is the only changed scientific variable.

Sphere scaling, with 256 emitters, eight packets per emitter, and \(K=32\):

| Resolution | Pixels | Hits | Abs. | T nnz / density | J nnz / density | Mean affected pixels / fraction per parameter | Peak MiB | Warm ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64×64 | 4,096 | 223 | 0 | 866 / 8.26e-4 | 10,002 / 2.54e-2 | 104.19 / 2.54e-2 | 29.9 | 23.34 |
| 128×128 | 16,384 | 223 | 0 | 890 / 2.12e-4 | 10,497 / 6.67e-3 | 109.34 / 6.67e-3 | 29.9 | 23.39 |
| 256×256 | 65,536 | 223 | 0 | 890 / 5.30e-5 | 10,620 / 1.69e-3 | 110.62 / 1.69e-3 | 29.9 | 23.10 |
| 512×512 | 262,144 | 223 | 0 | 890 / 1.33e-5 | 10,680 / 4.24e-4 | 111.25 / 4.24e-4 | 29.9 | 23.66 |
| 960×540 | 518,400 | 223 | 0 | 890 / 6.71e-6 | 10,680 / 2.15e-4 | 111.25 / 2.15e-4 | 50.3 | 23.69 |
| 1920×1080 | 2,073,600 | 223 | 0 | 892 / 1.68e-6 | 10,704 / 5.38e-5 | 111.50 / 5.38e-5 | 176.4 | 24.88 |

Torus scaling under the same configuration:

| Resolution | Pixels | Hits | Abs. | T nnz / density | J nnz / density | Mean affected pixels / fraction per parameter | Peak MiB | Warm ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64×64 | 4,096 | 162 | 23 | 636 / 6.07e-4 | 2,043 / 5.20e-3 | 21.28 / 5.20e-3 | 26.5 | 26.50 |
| 128×128 | 16,384 | 162 | 23 | 638 / 1.52e-4 | 2,079 / 1.32e-3 | 21.66 / 1.32e-3 | 26.5 | 26.27 |
| 256×256 | 65,536 | 162 | 23 | 642 / 3.83e-5 | 2,100 / 3.34e-4 | 21.88 / 3.34e-4 | 26.5 | 26.44 |
| 512×512 | 262,144 | 162 | 23 | 648 / 9.66e-6 | 2,124 / 8.44e-5 | 22.12 / 8.44e-5 | 29.5 | 26.62 |
| 960×540 | 518,400 | 162 | 23 | 648 / 4.88e-6 | 2,124 / 4.27e-5 | 22.12 / 4.27e-5 | 50.3 | 26.61 |
| 1920×1080 | 2,073,600 | 162 | 23 | 648 / 1.22e-6 | 2,124 / 1.07e-5 | 22.12 / 1.07e-5 | 176.4 | 28.94 |

In isolated fresh processes, Full-HD cold/warm times were 272.8/25.9 ms for the sphere and 291.1/30.1 ms for the torus. Peak allocated/reserved memory was 176.4/204 MiB and 176.4/202 MiB, respectively. The sphere warm stage breakdown in milliseconds was emitter update 11.25, photon generation 0.60, camera intersection 0.57, collision 6.94, arrival/footprint 1.22, sparse transport 3.47, and sparse Jacobian 1.07. Torus values were 13.13, 0.67, 0.57, 8.24, 1.22, 4.57, and 1.52 ms. At 256², the same sparse operator path measured 60.53 ms on CPU and 23.10 ms on CUDA, a 2.62× speedup. No custom kernel or targeted optimization was needed: resolution growth leaves raw Jacobian nnz almost constant while its density decreases sharply.

## v0.3a: utility of a nonexistent geometry DoF

This falsification experiment distinguishes a hypothetical candidate direction from an active optimization parameter. The current model contains only 32 coefficients. Each of 64 finer candidate bases is appended with coefficient zero only long enough to evaluate its eight-view Jacobian column and score it. The persistent checkpoint remains 32-dimensional. For the oracle, exactly one candidate is then instantiated in a temporary 33-dimensional model, optimized from the same checkpoint, measured, and discarded before the next candidate.

The deterministic CUDA experiment used a 64-DoF synthetic sphere target, eight 256×256 views, 256 emitters, and eight packets per emitter. The current model reduced loss from 0.0182427 to 0.0117430 over its conservative 64-step budget; its final per-step relative improvement was $9.41\times10^{-5}$. Four candidate columns matched central finite differences with maximum relative error $1.58\times10^{-9}$. Both target and candidate validation transport-event fractions were $3.05\times10^{-5}$.

| Pre-birth score | Spearman | Pearson | Top-1 actual gain | Top-5 mean gain | Top-5 regret |
|---|---:|---:|---:|---:|---:|
| Random | -0.157 | -0.110 | 6.91e-4 | 1.66e-4 | 5.26e-3 |
| Visibility | 0.330 | 0.094 | 2.41e-4 | 5.92e-5 | 5.71e-3 |
| Local residual | 0.829 | 0.722 | 8.29e-6 | 2.62e-3 | 0 |
| Jacobian norm | 0.649 | 0.358 | 1.81e-3 | 1.62e-3 | 0 |
| Raw alignment | **0.972** | 0.852 | 5.95e-3 | 2.97e-3 | 0 |
| Quadratic gain | 0.896 | 0.997 | 5.95e-3 | **3.05e-3** | 0 |
| Orthogonalized gain | 0.901 | **0.999** | **5.95e-3** | **3.05e-3** | 0 |

The orthogonalized score selected candidate 2, which was also the true best candidate, reduced loss by 0.005946 (50.6%), and had novelty ratio 0.999997. It also correctly suppressed the redundant control (novelty ratio $3.70\times10^{-8}$) and invisible control (zero Jacobian). The current multiview Jacobian had rank 32, and 55 candidates increased rank. Scoring took 0.20 s, all new-only and joint oracles took 47.81 s, and peak allocated/reserved VRAM was 68.7/92 MiB.

The result is nevertheless **not supportive under the stated falsification rule**. Orthogonalization had excellent Pearson correlation and top-k selection, but its 0.901 Spearman correlation did not outperform the much simpler raw residual alignment score at 0.972, and it tied the single-DoF quadratic baseline on top-5 mean gain. The result therefore does not justify implementing repeated dynamic birth. Candidate-level results are in [`artifacts/v03a_candidates.csv`](artifacts/v03a_candidates.csv); the three compact diagnostics are in [`figures/`](figures/).

## Limitations

The prototype deliberately omits differentiability across collision/visibility topology changes, repeated dynamic geometry birth, merge/prune policies, optimized basis centers or radii, neural fields, secondary reflection, indirect illumination, refraction, participating media, physical camera lenses, and physically calibrated radiometry. It also misses tangent zero-set intersections that touch without a sign change. The v0.3a oracle is a controlled local inverse experiment inside one fixed transport cell, not a production reconstruction system; the K=32 baseline stopped at a conservative iteration budget, so the small joint gain shared by redundant and invisible controls comes from continued optimization of existing parameters. Detector misses produce no contribution, deformations must retain the intended local normal-line root, and only sphere- and torus-based fields are supported. With the intentionally fixed 256 emitters, Full-HD images are extremely sparse; that benchmark tests operator scaling rather than image reconstruction quality.
