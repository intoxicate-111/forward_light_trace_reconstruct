# Zero-Set Forward Light Tracing

This experimental research prototype asks whether samples of a zero set can act as local directional emitters, whether a compact local parameterization induces a sparse geometry-to-image Jacobian, and whether observations can repeatedly turn evidence for nonexistent parameters into a better geometry function space. Version 0.1 isolates transport, v0.2–v0.2.2 validate local geometry derivatives and CUDA/multiview scaling, v0.3a–v0.3b test pre-birth utility prediction, v0.3c performs true sequential parameter birth, v0.3d transfers that protocol to Stanford Bunny geometry, v0.3e tests simultaneous birth and Bunny smoothing headroom, v0.3f separates detector resolution, photon density, optimization, and inverse ambiguity, v0.3g tests whether richer shared multiview evidence permits larger natural birth batches, v0.4 separates photon-arrival visualization from visibility-aware camera image formation, v0.5 factors image formation through a frozen camera-independent outgoing boundary field, v0.6 removes the renderer's mesh surface scaffold and adds depth-sensitive forward RGB transport, v0.7 removes unnecessary attenuation from the default and retests geometry birth against the corrected RGB operator, v0.8 stress-tests that birth mechanism at Full HD, 20 views, and 1.31 million attempted forward packets, and v0.8.1 audits the resulting image-sampling pathology and adds an opt-in resolution-aware footprint. This project makes no claim of novelty.

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
python demo.py --benchmark-multiview --multiview-output artifacts/v022_multiview.json --multiview-figures figures
python demo.py --birth --birth-csv artifacts/v03a_candidates.csv --birth-figures figures
python demo.py --birth-k-scaling --k-scaling-csv artifacts/v03b_k_scaling.csv --k-scaling-json artifacts/v03b_k_scaling.json --k-scaling-figures figures
python demo.py --repeated-birth --repeated-birth-csv artifacts/v03c_birth_trajectories.csv --repeated-birth-json artifacts/v03c_birth_report.json --repeated-birth-figures figures
python -m pip install -e '.[bunny]'
python demo.py --bunny --bunny-artifacts artifacts --bunny-figures figures
python demo.py --batch-birth --bunny-artifacts artifacts --bunny-figures figures
python demo.py --observation-bandwidth --bunny-artifacts artifacts --bunny-figures figures
python demo.py --natural-multiview --bunny-artifacts artifacts --bunny-figures figures
python demo.py --image-formation --bunny-artifacts artifacts --bunny-figures figures --render-output render_res
python demo.py --camera-independent --bunny-artifacts artifacts --bunny-figures figures --render-output render_res
python demo.py --meshfree-rgb --bunny-artifacts artifacts --bunny-figures figures --render-output render_res
python demo.py --appearance-ablation --bunny-artifacts artifacts --render-output render_res
python demo.py --corrected-birth --bunny-artifacts artifacts --bunny-figures figures --render-output render_res
python demo.py --high-bandwidth-birth --bunny-artifacts artifacts --bunny-figures figures --render-output render_res
```

The reference fields are an analytic sphere and analytic torus; the optional Bunny path adds a fixed mesh-derived trilinear level-set evaluator. With the default seed, the sphere is the convex normal/camera sanity check. The side-view torus activates non-convex self-occlusion: some rays emitted from its inner wall cross the hole and re-intersect the opposite tube before reaching the detector. Every run reports zero-set and normal errors, hit and absorption fractions, sparse shape/nnz/density, operator fan-in/fan-out, direct-versus-sparse error, and measured pipeline runtime. `--verify` additionally runs deterministic hand checks for all acceptance gates through production code.

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

## v0.2.2: shared multiview transport

The multiview path makes the camera-independent boundary explicit. One `SceneTransportState` contains deformed zero-set emitters, deterministic photon directions, and each packet's earliest positive surface-hit time. A camera then performs only plane intersection, the physical survival test (t^{cam}<t^{surf}), first-arrival reduction, and sparse COO image construction. The strict independent baseline invokes the identical scene-state builder separately for every camera. A fixed 4-unit root interval covers the maximum 3.3-unit torus chord; across all 20 benchmark poses this scene-space collision rule exactly matched the legacy camera-bounded tracer, including all 224 torus absorptions.

The final CUDA experiment used 20 distinct Fibonacci-sphere cameras, Full HD, 256 emitters, eight packets per emitter (2,048 photons), (K=32), five warm-ups, and ten measured repetitions per mode and count. Dense images were streamed one camera at a time; they were not retained as a 20-image batch. Every shared image, hit index, owner, absorption state, camera time, and coalesced transport operator matched its independent counterpart exactly. The Jacobian was intentionally not timed because it was not needed to answer the transport question.

Sphere results:

| Views | Independent ms | Shared ms | Speedup | Geometry ms | Detector ms | Incremental ms/view | T nnz | Peak alloc./reserved MiB | Max error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.60 | 23.22 | 1.02× | 18.18 | 5.11 | — | 722 | 205.4 / 288 | 0 |
| 2 | 43.45 | 29.26 | 1.49× | 18.64 | 10.52 | 6.04 | 1,828 | 221.5 / 288 | 0 |
| 4 | 85.06 | 38.89 | 2.19× | 18.36 | 20.30 | 4.82 | 3,368 | 221.9 / 288 | 0 |
| 8 | 171.12 | 66.35 | 2.58× | 18.17 | 47.97 | 6.86 | 6,898 | 221.9 / 288 | 0 |
| 16 | 337.23 | 108.53 | 3.11× | 18.08 | 89.88 | 5.27 | 13,372 | 221.9 / 288 | 0 |
| 20 | 419.29 | 126.21 | **3.32×** | 17.98 | 107.52 | 4.42 | 16,414 | 221.9 / 288 | 0 |

Torus results:

| Views | Independent ms | Shared ms | Speedup | Geometry ms | Detector ms | Incremental ms/view | T nnz | Peak alloc./reserved MiB | Max error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.17 | 27.16 | 1.00× | 20.62 | 6.41 | — | 428 | 205.4 / 286 | 0 |
| 2 | 51.72 | 34.81 | 1.49× | 20.25 | 14.39 | 7.65 | 828 | 221.5 / 288 | 0 |
| 4 | 96.17 | 47.87 | 2.01× | 20.31 | 27.46 | 6.53 | 2,218 | 221.9 / 288 | 0 |
| 8 | 190.35 | 73.43 | 2.59× | 20.47 | 52.56 | 6.39 | 5,722 | 221.9 / 288 | 0 |
| 16 | 377.42 | 115.51 | 3.27× | 20.53 | 94.59 | 5.26 | 12,474 | 221.9 / 288 | 0 |
| 20 | 466.86 | 141.01 | **3.31×** | 20.45 | 119.88 | 6.37 | 14,788 | 221.9 / 288 | 0 |

A linear fit (T_{shared}=a+bN) gives (a=18.61) ms and (b=5.51) ms/camera for the sphere ((R^2=0.997)), and (a=23.38) ms and (b=5.87) ms/camera for the torus ((R^2=0.998)). Shared 1→20 scaling was 5.44×/5.19×, versus 17.77×/17.18× independently; the latter is 11.2%/14.1% below ideal (20T_1), rather than perfectly linear. Reuse saved 293.09/325.86 ms at 20 views. Transport nnz grew with the cumulative camera set while density stayed around (10^{-6}); the 1→20 nnz factors were 22.73× and 34.55× because the heterogeneous poses do not receive equal numbers of packets. Peak memory was approximately constant because outputs were streamed. The result supports shared **sparse scene-centric transport**, not free dense multiview rendering. Full measurements and all camera frames are in [`artifacts/v022_multiview.json`](artifacts/v022_multiview.json), the compact rows are in [`artifacts/v022_multiview.csv`](artifacts/v022_multiview.csv), and the required runtime and speedup plots are in [`figures/`](figures/).

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

## v0.3b: aggressive geometry-capacity scaling

The v0.3b path replaces global point-by-basis evaluation with a deterministic uniform-grid lookup and sparse point/basis incidence tables. Basis centers are nested prefixes of one unscrambled Sobol surface sequence, and their radius follows \(r_K=0.60\sqrt{32/K}\). Each target has \(K_{GT}=2K\) fine bases and is normalized to a 0.003 RMS surface displacement. The candidate pool has \(M=2K\): the missing target level plus a still-finer distractor level. Candidates remain absent during scoring, which directly accumulates sparse \(j_q^Tr\) and \(j_q^Tj_q\) contributions; no dense observation-space Jacobian is allocated.

The primary run used eight deterministic 256×256 views, eight emitters per active basis, and initially eight packets per emitter. At the 262,144-photon cap, packets per emitter reduced to four at \(K=8192\) and two at \(K=16384\), preserving spatial emitter density. Oracles were exhaustive through \(K=512\); from \(K=1024\), correlations use a deterministic unbiased 128-candidate sample while top-ranked candidates are evaluated separately. The actual-gain oracle is a nonlinear 17-sample one-dimensional search with the existing \(K\) coefficients frozen.

| K | K_GT | M | Emitters | Photons | Radius | Responsive | Correlation set | Raw Spearman | Raw Pearson | Quad. Spearman | Quad. Pearson | Bases/query | Params/pixel | Score s | Peak MiB alloc./res. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 64 | 256 | 2,048 | 0.60000 | 0.984 | 64 exhaustive | 0.885 | 0.931 | 0.980 | 1.000 | 2.90 | 2.92 | 0.051 | 89.1 / 110 |
| 64 | 128 | 128 | 512 | 4,096 | 0.42426 | 0.992 | 128 exhaustive | 0.864 | 0.937 | 0.977 | 1.000 | 2.96 | 2.98 | 0.024 | 91.1 / 114 |
| 128 | 256 | 256 | 1,024 | 8,192 | 0.30000 | 0.996 | 256 exhaustive | 0.902 | 0.977 | 0.970 | 0.998 | 2.83 | 2.84 | 0.023 | 94.8 / 122 |
| 256 | 512 | 512 | 2,048 | 16,384 | 0.21213 | 0.994 | 512 exhaustive | 0.881 | 0.943 | 0.970 | 0.998 | 2.96 | 2.98 | 0.024 | 102.6 / 140 |
| 512 | 1,024 | 1,024 | 4,096 | 32,768 | 0.15000 | 0.985 | 1,024 exhaustive | 0.906 | 0.949 | 0.964 | 0.998 | 2.89 | 3.00 | 0.029 | 120.4 / 158 |
| 1,024 | 2,048 | 2,048 | 8,192 | 65,536 | 0.10607 | 0.979 | 128 unbiased | 0.887 | 0.935 | 0.962 | 1.000 | 2.92 | 3.22 | 0.033 | 175.9 / 212 |
| 2,048 | 4,096 | 4,096 | 16,384 | 131,072 | 0.07500 | 0.974 | 128 unbiased | 0.860 | 0.874 | 0.959 | 1.000 | 2.91 | 3.64 | 0.048 | 279.5 / 336 |
| 4,096 | 8,192 | 8,192 | 32,768 | 262,144 | 0.05303 | 0.970 | 128 unbiased | 0.918 | 0.962 | 0.989 | 0.998 | 2.93 | 4.60 | 0.074 | 459.2 / 544 |
| 8,192 | 16,384 | 16,384 | 65,536 | 262,144 | 0.03750 | 0.952 | 128 unbiased | 0.925 | 0.958 | 0.986 | 0.998 | 2.85 | 4.66 | 0.076 | 498.8 / 584 |
| 16,384 | 32,768 | 32,768 | 131,072 | 262,144 | 0.02652 | 0.922 | 128 unbiased | 0.941 | 0.940 | 0.991 | 0.995 | 2.90 | 4.84 | 0.072 | 558.9 / 638 |

Raw-alignment Spearman correlation stayed between 0.860 and 0.941, with median 0.894 and largest-scale value 0.941; random-score median was 0.010. Quadratic gain retained the stronger magnitude calibration, with Pearson correlation at least 0.995. Active bases per query stayed in [2.83, 2.96], while active parameters per affected pixel increased slowly from 2.92 to 4.84 rather than with \(K\). The log-log fits were \(T_{score}\propto K^{0.167}\) (\(R^2=0.500\)), allocated VRAM \(\propto K^{0.347}\) (\(R^2=0.916\)), and affected fraction \(\propto K^{-0.200}\) (\(R^2=0.618\)). The weak affected-fraction exponent is a negative diagnostic against an ideal \(1/K\) law; it is not evidence of connectivity explosion because both measured local degrees remain bounded.

Four-candidate central finite differences were run at every scale. Median relative error rose from \(1.37\times10^{-9}\) to \(5.76\times10^{-7}\). Worst relative errors reached 1.0 at the two largest scales for deliberately selected nearly-null columns, where relative normalization is ill-conditioned; all target roots nevertheless passed the \(10^{-8}\) residual test and every level recorded zero numerical failures. Target transport-event fraction rose from 0.014% to 2.62%, and responsive-candidate fraction remained 92.2% at the largest scale. The no-birth continuation gain was at most \(1.11\times10^{-3}\) and is never credited to birth.

A second deformation seed repeated \(K=32,256,1024,16384\), producing raw Spearman correlations 0.879, 0.892, 0.849, and 0.930. One \(K=32768\) attempt stopped during target construction: the fixed-RMS normalization scale was 3.9218 and 268 of the 262,144 normal-line roots failed (maximum residual 0.0427; minimum failed absolute normal derivative 0.00318). Peak memory was not limiting. Thus the retained conclusions are `PREBIRTH_SIGNAL_K_SCALING_SUPPORTED`, `MAX_VALIDATED_K=16384`, and `LOCAL_CONNECTIVITY_SCALING_SUPPORTED`. The full table, sample-relative top-k metrics, controls, continuation gains, derivatives, timings, and failure record are in [`artifacts/v03b_k_scaling.json`](artifacts/v03b_k_scaling.json) and [`artifacts/v03b_k_scaling.csv`](artifacts/v03b_k_scaling.csv); the five required plots are in [`figures/`](figures/).

## v0.3c: repeated observation-driven birth

The v0.3c experiment fixes one 1,024-element hierarchical Sobol dictionary and one target before each reconstruction. The initial 32 coarse bases are persistent; every other candidate is absent from the optimization vector until selected. A birth appends exactly one zero coefficient while preserving all old coefficients, so the measured maximum birth-only geometry jump was \(1.64\times10^{-15}\). One stateless damped Gauss–Newton/CG step then jointly optimizes the enlarged active set. Candidate Jacobians are accumulated one view at a time into sparse \(j_q^Tr\) and \(j_q^Tj_q\) statistics and discarded.

The CUDA campaign used eight 256×256 views, 8,192 emitters, 65,536 photons, fixed 0.003-RMS targets, sparse and distributed detail regimes, and target seeds 5 and 17. All deterministic selectors used the same targets and optimization schedule. Five random seeds were run for the primary sparse target; one matched random run was retained for each other target. Every trajectory grew 32→256 through 224 true births. Extending to 512 was not computationally modest: one 256-DoF trajectory already required 69–78 seconds, while the matched fixed-space solves took 0.4–3.3 seconds.

| Selector | Regime | Runs | Final geometry RMS | Geometry/DoF AUC | Geometry/time AUC | Scoring fraction | Mean total s |
|---|---|---:|---:|---:|---:|---:|---:|
| Quadratic | Sparse | 2 | **6.32e-4** | **0.369** | **0.366** | 23.5% | 70.6 |
| Raw | Sparse | 2 | 6.80e-4 | 0.420 | 0.415 | 23.1% | 71.5 |
| Local residual | Sparse | 2 | 1.06e-3 | 0.604 | 0.597 | 21.9% | 75.4 |
| Uniform | Sparse | 2 | 1.76e-3 | 0.766 | 0.761 | 21.2% | 78.0 |
| Random | Sparse | 6 | 2.45e-3 ± 1.32e-4 | 0.930 | 0.929 | 23.9% | 69.3 |
| Quadratic | Distributed | 2 | **1.77e-3** | **0.757** | **0.754** | 22.8% | 72.8 |
| Raw | Distributed | 2 | 1.86e-3 | 0.777 | 0.774 | 22.0% | 75.4 |
| Local residual | Distributed | 2 | 1.95e-3 | 0.810 | 0.806 | 21.3% | 77.9 |
| Uniform | Distributed | 2 | 2.02e-3 | 0.821 | 0.818 | 21.2% | 78.3 |
| Random | Distributed | 2 | 2.49e-3 ± 4.36e-5 | 0.938 | 0.937 | 23.7% | 70.1 |

At the common 256-DoF endpoint, quadratic birth averaged 0.00120 geometry RMS across both regimes, versus 0.00189 for the fixed uniform prefix. The fixed-space mean curve was 0.00288, 0.00262, 0.00226, 0.00189, 0.00145, and 0.000590 at \(K=32,64,128,256,512,1024\). Relative to each target's fixed 32→1024 improvement, sparse quadratic birth reached the 90%, 95%, and 99% thresholds by \(K=128/192/192\) for both target seeds; raw required \(192/192/256\). The fixed uniform curve required \(K=1024\) for all these thresholds. No 256-budget sequential method reached the same thresholds for the distributed targets, an important limit on adaptive parameter efficiency.

Sequential calibration remained strong. Per-trajectory quadratic-score versus realized-gain Spearman correlations were 0.924–0.975 for quadratic selection, with Pearson correlations 0.991–0.997. In pooled early/middle/late windows its Spearman correlation was 0.977/0.967/0.845; calibration weakened late but did not collapse. On the deliberately small 32→64, 256-master oracle experiment, true nonlinear greedy ended at 0.000802 geometry RMS, quadratic at 0.000809, and raw at 0.000926. Thus most of the small-case greedy ceiling was captured by the quadratic approximation.

Sparse quadratic and raw births landed on true nonzero-detail dictionary entries 53.6% and 51.1% of the time, compared with 11.1% for random and 14.3% for uniform selection. There were no duplicate births, zero root failures, and no invisible quadratic/raw selections. Support-scaled near-duplicate rates were 2.46%/2.68% for sparse quadratic/raw and 0.22% for each distributed variant. The best remaining quadratic score fell by 3.03e-5×/1.10e-3× for sparse quadratic/raw and 0.0247×/0.0519× for distributed variants, exposing expected late score exhaustion rather than numerical failure. No-birth continuation gains were recorded at every checkpoint and never credited to birth; the largest was 0.0413 loss units.

The primary scientific verdict is `REPEATED_OBSERVATION_DRIVEN_BIRTH_SUPPORTED`, and the separate same-budget verdict is `PARAMETER_EFFICIENCY_SUPPORTED`. Compute efficiency is not supported: sequential selection costs roughly 70–78 seconds per trajectory, dramatically more than preallocated fixed optimization, despite modest 236–278 MiB peak allocated VRAM. The full checkpoint table is in [`artifacts/v03c_birth_trajectories.csv`](artifacts/v03c_birth_trajectories.csv), the complete births, thresholds, controls, calibration, timings, and verdict basis are in [`artifacts/v03c_birth_report.json`](artifacts/v03c_birth_report.json), and five compact diagnostics are in [`figures/`](figures/).

## v0.3d: controlled Stanford Bunny transfer

The real geometry is Stanford's standard `bun_zipper.ply` (35,947 vertices, 69,451 triangles; SHA-256 `b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe`). It is centered by its bounding-box midpoint, uniformly scaled by 12.8453 to a maximum extent of 2, and otherwise keeps its canonical orientation. The original has five bottom holes. For sign determination only, a deterministic centroid fan closes each boundary loop, producing a 35,952-vertex/69,674-face watertight proxy; every geometry metric still uses the original triangles. The mesh is not redistributed under the MIT license: [`data/stanford_bunny/SOURCE.md`](data/stanford_bunny/SOURCE.md) records the official source, archive and mesh hashes, license note, and exact acquisition behavior.

Open3D closest-triangle signs are sampled once on a fixed $160^3$ grid over $[-1.2,1.2]^3$, then evaluated as a generic trilinear zero-set field. The fixed coarse base is a 1.5-voxel Gaussian smoothing of that grid, not a learned SDF. Optimization has no Eikonal term or distance constraint, and intersections still use bracketed generic $F=0$ roots. The initial base-to-original symmetric Chamfer is 0.009376, P2S mean/p95 are 0.002710/0.007064, surface RMS is 0.005892, and normal error is 0.01510.

One nested 1,024-center Sobol hierarchy is sampled only from the base surface. It uses the existing Wendland $C^2$ bases and level-scaled supports; GT error is never used for candidate placement or ranking. Eight deterministic Fibonacci cameras observe 8,192 emitters and 65,536 shared packets at 256². All six trajectories share those cameras, packets, target images, base, dictionary, optimizer, and checkpoints. The 32→256 run comprises 224 true zero-initialized births.

| Method | K | Image loss | Chamfer | P2S mean | P2S p95 | Normal error | Ear | Head | Torso | Legs | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quadratic | 32 | 36.572 | 0.009294 | 0.002578 | 0.006754 | 0.01492 | 0.003097 | 0.001693 | 0.001641 | 0.004448 | 2.34 |
| Quadratic | 64 | 32.388 | 0.009202 | 0.002434 | 0.005964 | 0.01455 | 0.002347 | 0.001678 | 0.001580 | 0.004443 | 31.61 |
| Quadratic | 128 | 30.465 | 0.009145 | 0.002330 | 0.005742 | 0.01448 | 0.002021 | 0.001643 | 0.001512 | 0.004370 | 89.47 |
| Quadratic | 192 | 29.673 | 0.009124 | 0.002289 | 0.005686 | 0.01450 | 0.001964 | 0.001635 | 0.001487 | 0.004292 | 147.81 |
| Quadratic | 256 | 29.240 | **0.009122** | 0.002279 | 0.005724 | 0.01451 | 0.001920 | 0.001619 | 0.001464 | 0.004330 | 206.34 |
| Raw | 256 | 29.504 | **0.009086** | 0.002231 | 0.005602 | 0.01445 | 0.001937 | 0.001478 | 0.001442 | 0.004237 | 204.34 |
| Uniform | 256 | 32.497 | 0.009208 | 0.002429 | 0.006201 | 0.01456 | 0.002419 | 0.001629 | 0.001552 | 0.004455 | 212.90 |
| Random mean (3) | 256 | 34.228 | 0.009245 | 0.002490 | 0.006553 | 0.01474 | 0.002755 | 0.001677 | 0.001593 | 0.004398 | 205.56 |

| Fixed K | Image loss | Chamfer | P2S mean | P2S p95 | Normal error | Solve s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 36.572 | 0.009294 | 0.002578 | 0.006754 | 0.01492 | 1.86 | 260.3 |
| 64 | 34.996 | 0.009240 | 0.002494 | 0.006420 | 0.01483 | 4.70 | 283.5 |
| 128 | 34.007 | 0.009202 | 0.002431 | 0.006092 | 0.01470 | 5.99 | 301.7 |
| 256 | 32.505 | 0.009191 | 0.002407 | 0.006105 | 0.01460 | 6.04 | 323.1 |
| 512 | 30.511 | 0.009136 | 0.002310 | 0.005912 | 0.01447 | 6.28 | 345.8 |
| 1024 | 28.452 | 0.009043 | 0.002145 | 0.005771 | 0.01442 | 6.44 | 366.1 |

Quadratic selection ends with 0.93% lower Chamfer than uniform and 1.33% lower than the random mean. Its normalized Chamfer/DoF AUC is 0.98545 versus 0.99207 and 0.99726. Raw selection is slightly better than quadratic at the endpoint and in AUC, so no claim is made that the quadratic normalization is uniformly best. No sequential 256-DoF method reaches the preregistered 90%, 95%, or 99% thresholds derived from the fixed 32→1024 improvement.

Selected quadratic score versus realized joint-optimization gain has Pearson/Spearman 0.982/0.885; early/middle/late Spearman is 0.867/0.660/0.503. The deterministic eight-candidate new-DoF-only checkpoint oracle is less convincing: quadratic Spearman is 0.357, 0.357, 0.048, and -0.310 at K=32,64,128,256, where late oracle gains are nearly all numerical zero. This calibration loss is retained as a limitation. Quadratic births fall in the independently defined top quartile of base-to-GT displacement 52.7% of the time, versus 24.1% for uniform and 23.1% for random, without using that map for selection.

Responsive candidates remain 97.4–98.1%, near-null columns remain 2.1–2.7%, active bases/query grow from 2.61 to 4.55, and active parameters per affected pixel from 2.82 to 4.90. Candidate Jacobian nnz stays near 2.72 million. There are zero root, CG, line-search, normal-line, invalid-surface, and NaN/Inf failures; the minimum root denominator is 0.385 and the largest birth-only geometry jump is $2.46\times10^{-15}$. Mean photon-state/occupied-owner transport-event fractions are 0.170%/6.05%, explicitly delimiting the fixed-cell approximation.

The Phase-1 verdict is `RABBIT_CONTROLLED_BIRTH_SUPPORTED`: all six stated scientific conditions pass, but the result is modest and mixed. The additional Phase-2 continuation gate fails because the quadratic improvement over uniform is 0.93%, not at least 10%; therefore Phase 2 was not run and its status is `PHASE2_SKIPPED`. Quadratic sequential scoring/optimization consume 11.11/96.61 s and the full trajectory takes 206.34 s at 299.8 MiB peak allocation, compared with 6.04 s for fixed K=256. This is parameter-allocation evidence, not computational efficiency or real-photograph reconstruction. Full checkpoints and diagnostics are in [`artifacts/v03d_bunny_phase1.csv`](artifacts/v03d_bunny_phase1.csv) and [`artifacts/v03d_bunny_phase1.json`](artifacts/v03d_bunny_phase1.json); the six required compact figures are in [`figures/`](figures/).

## v0.3e: simultaneous birth and Bunny headroom

Phase A replaces the strictly serial 32→256 continuation with true simultaneous zero-initialized batches. At every round it still scores every inactive candidate with the existing damped quadratic score, but greedily admits only candidates whose observation-space Jacobian cosine is at most 0.10. Dynamic birth additionally requires score at least 10% of the current maximum, batch coupling $\rho_{off}\leq0.15$, and caps the batch at 32. The selected coefficients are appended together at zero before one shared optimization; the largest measured birth-only geometry jump is below $2.4\times10^{-15}$.

| Method | Chamfer | P2S p95 | Chamfer/DoF AUC | Total s | Speedup | Rounds | Batch mean/median/max | Mean/max $\rho_{off}$ | Pearson/Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p=1 | 0.009122 | 0.005724 | 0.98511 | 129.72 | 1.00× | 224 | 1/1/1 | 0/0 | 0.982/0.885 |
| p=4 | 0.009098 | 0.005618 | 0.98386 | 35.36 | 3.67× | 56 | 4/4/4 | 0.0008/0.0131 | 0.997/0.980 |
| p=8 | 0.009090 | 0.005589 | 0.98328 | 19.02 | 6.82× | 28 | 8/8/8 | 0.0024/0.0253 | 0.996/0.986 |
| p=16 | 0.009083 | 0.005602 | 0.98295 | 10.74 | 12.08× | 14 | 16/16/16 | 0.0044/0.0183 | 0.998/0.996 |
| dynamic | **0.009070** | **0.005518** | **0.98276** | **8.38** | **15.48×** | **8** | **28/32/32** | 0.0093/0.0239 | 0.999/1.000 |

All four multi-birth methods pass the preregistered final-quality, runtime, numerical-stability, and zero-jump gates. Dynamic is therefore **BEST_MULTI_BIRTH**: it is the fastest eligible method, improves final Chamfer by 0.57% and P2S p95 by 3.60% relative to serial birth, and realizes the observation-driven schedule 29, 32, 32, 32, 32, 32, 32, 3. Its maximum pairwise cosine is 0.0896, and selected-batch joint-to-independent predicted gain remains essentially one. The Phase-A verdict is **BATCH_BIRTH_SUPPORTED**: serial rescoring is not necessary for this controlled case.

Phase B regenerates only the smoothed base and its surface-sampled dictionary at $\sigma=1.5,2.5,4.0$ voxels. GT, cameras, photon directions and colors, detector cells, emitter identities, and target images remain shared; the common target-image SHA-256 is `f6be87468380378b95cc3ba838727cb7eb4fa2ee132488ec01d3e7dffc742660`. Because each smoothed zero set is different, each shared emitter identity is transferred by deterministic closest-point projection and exact trilinear-root refinement; the projection distance and transport-state drift are reported rather than hidden.

| $\sigma$ | Base Chamfer | Fixed-256 | Fixed-1024 | Chamfer headroom | Dynamic recovered | Uniform recovered | Dynamic P95 recovered | Adaptive advantage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.009376 | 0.009191 | 0.009043 | 0.000333 | 91.96% | 62.10% | 119.54% | +29.87 pp |
| 2.5 | 0.011959 | 0.010968 | 0.011528 | 0.000431 | 98.41% | 98.41% | 73.78% | +0.002 pp |
| 4.0 | 0.020221 | 0.019618 | 0.019729 | 0.000491 | 170.03% | 241.86% | 713.72% | −71.83 pp |

Raw base error and the mandated base-minus-fixed-1024 Chamfer headroom both rise with smoothing, but fixed-space quality is non-monotone: at $\sigma=2.5$ and 4.0, fixed K=256 is better than fixed K=1024. Consequently recovered-headroom values can exceed 100%. Dynamic and matched-uniform use identical per-sigma batch schedules and take 8/11/12 rounds; their runtimes are 8.38/9.42/9.38 s and 7.19/8.32/8.17 s respectively. Dynamic births remain concentrated in independently measured high-error regions (top-quartile fractions 55.4%, 62.9%, 54.9%, versus uniform 24.1%, 26.8%, 23.2%), but that geography does not translate into a growing global advantage. At $\sigma=2.5$ the methods tie, and at 4.0 uniform is better.

The Phase-B verdict is **BUNNY_HEADROOM_ADAPTIVE_BIRTH_NOT_SUPPORTED**. The failed gates are growing adaptive advantage, absence of systematic line-search exhaustion, and monotone fixed-space reference quality. The mild $\sigma=1.5$ problem is headroom-limited, but increasing smoothing exposes a representation/optimization limitation rather than showing that the earlier small adaptive advantage was mainly a ceiling artifact. Exact per-checkpoint metrics, regional errors, projection and transport drift, calibration, timings, VRAM, and gate definitions are in [`artifacts/v03e_batch_birth.csv`](artifacts/v03e_batch_birth.csv), [`artifacts/v03e_batch_birth.json`](artifacts/v03e_batch_birth.json), [`artifacts/v03e_bunny_smoothing.csv`](artifacts/v03e_bunny_smoothing.csv), and [`artifacts/v03e_bunny_smoothing.json`](artifacts/v03e_bunny_smoothing.json). The twelve required plots are in [`figures/`](figures/).

## v0.3f: observation bandwidth and geometry scaling

The diagnostic keeps the Bunny field, 1,024-element dictionary, eight physical cameras, 8,192 emitters, optimizer, and geometry evaluator fixed. It crosses 256²/512² detector binning with 65,536/262,144 photons at $\sigma=2.5$ and 4.0. Unscrambled Sobol packet directions make every eight-packet emitter sequence an exact prefix of its 32-packet sequence; the measured prefix error is zero. Camera poses and physical detector extents are byte-identical between resolutions. Targets use their condition's own resolution and packet set. Cross-resolution image comparisons use per-scalar-observation MSE and residual/target energy, while raw half-SSE is retained.

The warm fixed-space path solves K=256 and zero-pads that solution through K=512 and 1,024. Expansion-only image differences are at most $1.39\times10^{-13}$ and geometry differences at most $6.57\times10^{-16}$. The principal $\sigma=2.5$, 256²/65k cold path is Case C: K=1,024 returns to the unoptimized base (Chamfer 0.011959) after line-search failure. Its nested-warm path is Case A and reaches 0.010720. Thus the previously reported non-monotonicity is primarily an optimization/initialization failure, not evidence that the larger function space excludes the smaller solution.

Warm-path fixed-space results are:

| $\sigma$ | Condition | Detector / photons | K=256 Chamfer | K=512 Chamfer | K=1024 Chamfer | K=1024 P95 | K=1024 normalized MSE | Cold / warm class |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 2.5 | A | 256² / 65k | 0.011024 | 0.010869 | **0.010720** | 0.014348 | 2.823e-4 | C / A |
| 2.5 | B | 256² / 262k | 0.011003 | 0.010800 | **0.010672** | 0.014231 | 1.464e-3 | C / A |
| 2.5 | C | 512² / 65k | 0.011959 | 0.011959 | 0.011959 | 0.017797 | 1.774e-4 | stalled / stalled |
| 2.5 | D | 512² / 262k | 0.011959 | 0.011959 | 0.011959 | 0.017797 | 7.394e-4 | stalled / stalled |
| 4.0 | A | 256² / 65k | 0.017377 | 0.017377 | 0.017377 | 0.034453 | 1.034e-3 | A / stalled |
| 4.0 | B | 256² / 262k | 0.017636 | 0.017479 | 0.017479 | 0.034828 | 4.313e-3 | A / A |
| 4.0 | C | 512² / 65k | 0.018075 | 0.018075 | 0.017604 | 0.036284 | 4.538e-4 | A / A |
| 4.0 | D | 512² / 262k | 0.018109 | 0.017878 | 0.017602 | 0.036376 | 1.876e-3 | A / A |

The deterministic 2×2 factorial effects below are high-minus-low differences at warm K=1,024; negative geometry and near-null effects are improvements, while positive stable-rank effects are improvements.

| $\sigma$ | Effect | Chamfer | P2S p95 | Stable rank | Near-null fraction | K1024−K256 gap |
|---:|:---|---:|---:|---:|---:|---:|
| 2.5 | Resolution | +1.263e-3 | +3.507e-3 | −2.378 | −9.77e-4 | +3.177e-4 |
| 2.5 | Photons | −2.386e-5 | −5.839e-5 | −0.609 | −3.027e-2 | −1.358e-5 |
| 2.5 | Interaction | +4.771e-5 | +1.168e-4 | +1.144 | 0 | +2.716e-5 |
| 4.0 | Resolution | +1.755e-4 | +1.690e-3 | +0.010 | −1.465e-3 | −4.104e-4 |
| 4.0 | Photons | +4.986e-5 | +2.331e-4 | +0.006 | −3.955e-2 | −9.663e-5 |
| 4.0 | Interaction | −1.042e-4 | −2.830e-4 | +0.007 | −2.930e-3 | +1.207e-4 |

At $\sigma=2.5$, A/B/C/D have responsive fractions 94.8%/97.8%/94.9%/98.0%, near-null fractions 5.66%/2.64%/5.57%/2.54%, and stable ranks 7.25/6.07/4.30/4.26. Their relative-$10^{-6}$ Gram ranks are 960/988/961/989 of 1,024. Maximum mutual cosine remains approximately one because a few columns are collinear, but the median is zero and only 0.60–0.64% of pairs exceed 0.05. Occupied detector fractions are 5.26%/19.01%/1.35%/5.25%. At fixed photon count, 512² therefore quarters occupancy without improving geometry; higher photon count restores occupied-pixel count and reduces near-null columns but does not make 512² useful. Photon-state event fractions are 0.50%/0.49%/0.88%/0.87%, and occupied-owner event fractions 17.1%/17.3%/28.2%/28.1%, exposing greater fixed-cell drift at higher resolution. No root, degenerate-gradient, or normal-line failure was hidden.

Condition B is `BEST_OBSERVATION_CONFIG`: it has the best warm K=1,024 Chamfer and P95, a monotone warm path, substantially fewer near-null columns than A, and avoids the cost and occupancy loss of 512². With unchanged dynamic rules $(\alpha,\tau,\rho_{max},p_{max})=(0.1,0.1,0.15,32)$, B produces seven full batches `[32, 32, 32, 32, 32, 32, 32]`, versus the historical A schedule `[4, 5, 7, 10, 22, 32, 32, 32, 32, 32, 16]`. Mean off-diagonal coupling falls from 0.0203 to 0.0182 and maximum selected pair cosine from 0.0953 to 0.0918. Predicted versus realized batch gain has Pearson/Spearman 0.9998/1.000, although seven rounds make the windowed calibration evidence limited.

| Method at K=256 | Chamfer | P2S mean | P2S p95 | Surface RMS | Normal error | Normalized MSE | Runtime s | Peak MiB |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dynamic | **0.010747** | **0.004347** | **0.013731** | **0.007623** | **0.031234** | **1.477e-3** | 10.68 | 819.7 |
| Uniform, matched schedule | 0.011005 | 0.004704 | 0.014630 | 0.007944 | 0.031491 | 1.516e-3 | **9.71** | 876.8 |

Dynamic beats matched uniform in all five genuine geometry metrics, so `DYNAMIC_BIRTH_HIGH_BANDWIDTH_SUPPORTED`. Its births are less concentrated in independently measured GT-error regions than the historical low-sampling run: top-25%/top-10% fractions fall from 62.9%/38.4% to 49.1%/24.1%. The 1024² escalation gate failed all five criteria and was not run.

The separate verdicts are `RESOLUTION_EFFECT_NOT_SUPPORTED`, `PHOTON_DENSITY_EFFECT_SUPPORTED`, `OBSERVATION_BANDWIDTH_LIMIT_NOT_SUPPORTED`, `OPTIMIZATION_LIMIT_SUPPORTED`, and `GEOMETRY_AMBIGUITY_NOT_SUPPORTED`. The strongest classification is **Class D, optimization bottleneck**, with **Class B, photon sampling bottleneck** as a secondary partial effect. Classes A, C, E, and F are not supported: increased detector binning does not help, warm/cold paths can recover useful K scaling, and no optimized warm path exhibits monotonically improving image fit with worsening geometry. Complete fixed rows, Gram spectra, support, transport events, timing decomposition, factorial definitions, and gate checks are in [`artifacts/v03f_observation_bandwidth.csv`](artifacts/v03f_observation_bandwidth.csv) and [`artifacts/v03f_observation_bandwidth.json`](artifacts/v03f_observation_bandwidth.json); dynamic trajectories are in [`artifacts/v03f_dynamic_high_bandwidth.csv`](artifacts/v03f_dynamic_high_bandwidth.csv) and [`artifacts/v03f_dynamic_high_bandwidth.json`](artifacts/v03f_dynamic_high_bandwidth.json). The ten compact plots are in [`figures/`](figures/).

## v0.3g: multiview evidence and natural birth capacity

This experiment removes the artificial 32-DoF dynamic-batch ceiling without retuning the validated compatibility rules. At active size (K), the raw allowance is (100K), clipped only by the remaining K=256 budget and inactive dictionary entries. A candidate must retain at least 10% of the best score, have cosine at most 0.10 with every selected candidate, and keep joint ρ_off at most 0.15. Every accepted batch is appended simultaneously at zero, so no selected coefficient exists in the persistent vector before birth. Natural and deterministic-hierarchy baselines use the exact same batch schedule.

The fixed Stanford Bunny sigma=2.5 setup uses 8,192 emitters and 256² detectors. Its nested direction ladder has 32, 320, and 3,200 packets/emitter (262,144, 2,621,440, and 26,214,400 photons); every P1/P10 prefix comparison is bit-exact. V8 is the validated camera set and an exact prefix of nested V20/V40. P100 is streamed in emitter-major chunks, retaining compact per-view first-arrival cells and capture histograms rather than full photon state or a dense observation Jacobian. Against monolithic P1, owner maps are exact, direction/target errors are zero, and the maximum base-image error is (4.44\times10^{-16}). The one-camera shared and independent implementations are exact. Camera-independent scene transport is amortized across detectors; detector work still grows with view count.

| Condition | Natural schedule | Rounds | Mean / first / max p | Unique capture | Captured multiplicity mean | Responsive / near-null | Gram rank / stable rank | Dynamic / uniform Chamfer | Fixed warm K256 / K1024 | Dynamic / uniform P95 | Dynamic s / peak MiB |
|:---|:---|---:|:---|---:|---:|:---|:---|:---|:---|:---|---:|
| V8/P1 cap32, historical | 32,32,32,32,32,32,32 | 7 | 32 / 32 / 32 | 43.51% | 1.00 | 97.75% / 2.64% | 988 / 6.067 | 0.010747 / — | 0.011003 / 0.010672 | 0.013731 / — | 10.69 / 820 |
| V8/P1 natural | 36,112,76 | 3 | 74.67 / 36 / 112 | 43.51% | 1.00 | 97.75% / 2.64% | 988 / 6.067 | 0.010796 / 0.011001 | 0.011003 / 0.010672 | 0.014016 / 0.014681 | 13.48 / 2,652 |
| V8/P10 | 47,81,58,38 | 4 | 56.00 / 47 / 81 | 43.50% | 1.00 | 98.54% / 1.46% | 1004 / 3.107 | 0.011959 / 0.011959 | 0.011959 / 0.011959 | 0.017797 / 0.017797 | 12.22 / 5,012 |
| V8/P100 | 14,29,30,34,18,99 | 6 | 37.33 / 14 / 99 | 43.50% | 1.00 | 87.99% / 12.89% | 850 / 1.522 | 0.011959 / 0.011959 | 0.011959 / 0.011959 | 0.017797 / 0.017797 | 15.46 / 3,437 |
| V20/P10 | 21,32,45,30,30,66 | 6 | 37.33 / 21 / 66 | 57.08% | 1.76 | 98.63% / 1.37% | 1005 / 2.088 | 0.011959 / 0.011959 | 0.011959 / 0.011959 | 0.017797 / 0.017797 | 28.12 / 6,242 |
| V40/P10 | 22,51,37,22,56,21,15 | 7 | 32.00 / 22 / 56 | 76.34% | 2.81 | 98.73% / 1.27% | 1009 / 2.483 | 0.011959 / 0.011959 | 0.011959 / 0.011959 | 0.017797 / 0.017797 | 58.88 / 8,339 |
| V40/P100 | 16,29,18,25,55,51,30 | 7 | 32.00 / 16 / 55 | 76.33% | 2.81 | 92.68% / 8.20% | 917 / 1.542 | 0.011959 / 0.011959 | 0.011959 / 0.011959 | 0.017797 / 0.017797 | 52.87 / 6,262 |

No natural trajectory reached its cap. All schedules sum to 224, all trajectories finish at K=256, the largest birth-only geometry jump is (2.41\times10^{-15}), maximum selected pair cosine is 0.09972, and maximum ρ_off is 0.05946. Natural V8/P1 therefore contains genuinely large batches and cuts seven rounds to three, but it is slightly worse than the historical capped trajectory in all five geometry metrics: Chamfer +0.45%, P2S mean +1.73%, P95 +2.07%, surface RMS +0.73%, and normal error +0.03%. Under the strict no-worse rule, the verdict is `NATURAL_BATCH_GROWTH_NOT_SUPPORTED`.

At fixed P10, unique capture grows from 43.50% at V8 to 57.08% at V20 and 76.34% at V40. The diagnostic V1/2/4/8/16/20/40 curve is 4.12%, 7.92%, 22.67%, 43.50%, 53.71%, 57.08%, and 76.34%; captured multiplicity reaches mean/median/p95/max 2.81/3/5/7 at V40. This supports `MULTIVIEW_CAPTURE_UTILIZATION_SUPPORTED`. V8→V40 also raises relative-1e-6 Gram rank from 1004 to 1009 and lowers p95 candidate cosine from 0.001013 to 0.000678, supporting a limited operator-level `MULTIVIEW_OBSERVABILITY_GAIN_SUPPORTED`. Near-null fraction improves by only 0.20 percentage points and geometry does not improve.

The downstream links fail. At V8, P1→P10→P100 changes mean natural batch 74.67→56.00→37.33 and rounds 3→4→6; at P10, V8→V20→V40 changes mean batch 56.00→37.33→32.00 and rounds 4→6→7. Every P10/P100 Gauss–Newton line search rejects every proposed step, leaving dynamic, matched-uniform, and warm fixed references at the base geometry; predicted-versus-realized calibration is therefore undefined for those five conditions rather than evidence of successful prediction. Only V8/P1 has positive accepted steps and defined batch-gain calibration (Pearson 0.995, Spearman 1.000; mean joint/independent predicted-gain ratio 0.983). The remaining verdicts are `PHOTON_EVIDENCE_EXPANDS_SAFE_BIRTH_NOT_SUPPORTED`, `MULTIVIEW_EVIDENCE_EXPANDS_SAFE_BIRTH_NOT_SUPPORTED`, and `SHARED_MULTIVIEW_TO_ADAPTIVE_GROWTH_CHAIN_NOT_SUPPORTED`.

The full run took 751.25 s on the local Quadro RTX 5000. Shared current-plus-target scene transport took 2.50/25.09/250.27 s for P1/P10/P100. At P10, detector time rose 0.45→1.12→2.23 s for V8/V20/V40; candidate scoring was 3.54→13.53→31.76 s and optimization was 4.02→11.49→24.16 s. These measurements do not make a many-view-is-free claim. Exact configurations, per-view coverage, candidate norms/cosines, rank spectra, every batch stop threshold and reason, fixed warm rows, CG/line-search histories, photon-efficiency metrics, correlations, and runtime decomposition are retained in [`artifacts/v03g_natural_batch.json`](artifacts/v03g_natural_batch.json), [`artifacts/v03g_multiview_capture.json`](artifacts/v03g_multiview_capture.json), and [`artifacts/v03g_multiview_birth.json`](artifacts/v03g_multiview_birth.json), with compact CSV companions and eleven [`v03g` figures](figures/).

## v0.4: visibility-aware Bunny image formation

The earlier detector output is formally a transport-event visualization. Surface point $p_e$ emits stochastic outward packet $q$; after zero-set occlusion, detector pixel $u$ selects the surviving packet with minimum flight time and displays its coordinate-coded emitter color through a bilinear footprint:

\[
q^*(u)=\arg\min_{q\mapsto u}t_q,\qquad I_{old}(u)=C_{e(q^*(u))}.
\]

This has packet-path occlusion and first-arrival ownership, but it lacks camera-ray visibility, camera-space surface depth, visible-surface compositing, and camera projection semantics. Increasing packets fills the detector plane rather than converging to the Bunny silhouette. The explicit diagnosis is `CURRENT_RENDER_IS_TRANSPORT_VISUALIZATION_NOT_IMAGE_FORMATION`, and the experimental verdict is `CURRENT_HITMAP_RENDERING_INSUFFICIENT`.

The corrected path samples one deterministic camera-independent state $X=\{p_i:F(p_i)=0\}$ from the extracted trilinear zero set. Sobol triangle samples are refined with the exact trilinear gradient and voxel-bounded Newton steps; the final maximum $|F|$ is $5.34\times10^{-11}$. Each camera orthographically projects that same state into its existing physical frame, computes camera-space depth, and resolves local projected candidates with a visibility rule. The selected hard nearest-depth plus bilinear-footprint cell is

\[
i^*(u)=\arg\min_{i:\,u\in\Pi_c(p_i)}z_c(p_i),\qquad
I_c(u)=A_c(p_{i^*(u)},n_{i^*(u)}).
\]

The fixed cell is stored as a normalized sparse COO appearance operator. A surface sample proposes at most four local pixels; nearest-depth ownership retains one contributor per occupied pixel. Across V8 the operator has 16,843–21,034 nonzeros and density $1.96\times10^{-6}$ to $2.45\times10^{-6}$. This keeps the local/sparse structure while making visibility changes an explicit non-differentiable cell boundary, as in the earlier transport Jacobian.

An orthographic ray cast against the marching-cubes surface extracted from the same zero-set grid supplies the image-formation sanity reference. Five reductions were compared on the canonical side view using the fixed 131,072-sample state. Selection uses (0.45\,IoU+0.35\,SSIM+0.20\,F1_{edge,2px}), so a visually perforated point result cannot win on silhouette IoU alone.

| Detector reduction | Footprint | IoU | SSIM | PSNR dB | Edge F1 | Depth RMSE | COO nnz | ms |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| All-depth accumulation | 3×3 Gaussian | 0.938 | 0.807 | 20.86 | 0.841 | 0.416 | 1,179,648 | 69.7 |
| Dominant weight | 3×3 Gaussian | 0.938 | 0.725 | 19.38 | 0.841 | 0.562 | 21,030 | 13.3 |
| Hard nearest depth | Point | **0.978** | 0.877 | **26.36** | 0.868 | 0.183 | 20,068 | 4.8 |
| **Hard nearest depth** | **Bilinear** | 0.959 | **0.900** | 24.93 | **1.000** | **0.038** | 20,575 | **3.8** |
| Near-depth soft hybrid | 3×3 Gaussian | 0.938 | 0.892 | 23.53 | 0.841 | 0.047 | 260,004 | 6.6 |

The point rule has the highest raw IoU but visible pinholes. Bilinear nearest-depth wins the declared balanced score, eliminates those holes, preserves sharp ownership, and most closely matches the reference boundary/depth. Constant gray, world-normal color, smooth position color, and deterministic shaded gray all recover the same 0.959 silhouette IoU. Position color has the highest same-appearance SSIM (0.939); shaded gray is retained for presentation because it exposes the ears, face, torso, legs, and surface structure without confounding geometry with packet identity.

The corrected V8 images have mean silhouette IoU/F1 of **0.954/0.977**, mean SSIM **0.891**, PSNR **24.59 dB**, two-pixel-tolerant edge F1 **0.990**, and normalized center error **0.00219**. Every view clearly retains Bunny foreground structure. The old packet maps have mean V8 IoU 0.239 and SSIM between 0.006 and 0.054. Replacing coordinate appearance with constant gray leaves the representative old-map IoU at 0.285, proving that appearance alone is not the cause.

Angular spread also fails to repair the old semantics: wide/medium/narrow cone powers 16/128/1024 give IoU 0.295/0.285/0.282 and SSIM 0.006/0.006/0.008. Raising legacy photons from 65,536 to 262,144 and 1,048,576 raises IoU only 0.183→0.285→0.304 while foreground occupancy incorrectly saturates 27.2%→68.6%→96.9%. Raising emitters from 2,048 to 32,768 behaves similarly. In contrast, corrected 32,768/131,072/524,288 surface samples give IoU 0.961/0.959/0.957 and SSIM 0.861/0.900/0.890: the semantics work before brute-force density. At fixed 131,072 samples, 512² raises IoU from 0.959 to 0.978 and PSNR from 24.93 to 26.43 dB but lowers SSIM to 0.855; resolution is a secondary, mixed refinement rather than the semantic fix.

One 131,072-sample zero-set traversal feeds all eight cameras. Shared and independently rebuilt runs agree exactly in RGB, masks, owner maps, COO indices, and COO values. Shared build/detector/total time is 0.463/0.029/0.492 s, versus 3.140/0.033/3.173 s when rebuilding the surface state for every camera, a measured 6.44× speedup at equal 48.0 MiB incremental peak allocation. Thus `SHARED_MULTIVIEW_IMAGE_FORMATION_SUPPORTED`; the precise claim is that camera-independent zero-set surface construction is amortized, not that detector projection is free.

The four final verdicts are `CURRENT_HITMAP_RENDERING_INSUFFICIENT`, `BUNNY_LIKE_IMAGE_FORMATION_SUPPORTED`, `SHARED_MULTIVIEW_IMAGE_FORMATION_SUPPORTED`, and `PHOTON_LIMIT_SECONDARY`. The old packet operator remains available for transport experiments; v0.4 adds a separate camera image-formation operator rather than silently changing historical semantics. Full candidate metrics, per-view rows, appearance/spread controls, scaling sweeps, timing, VRAM, exact camera poses, hashes, and gate definitions are in [`artifacts/v04_image_formation.csv`](artifacts/v04_image_formation.csv) and [`artifacts/v04_image_formation.json`](artifacts/v04_image_formation.json). The central [old/new/reference comparison](render_res/v04_old_vs_new.png), [eight-view Bunny montage](render_res/v04_multiview_bunny.png), reduction/appearance/spread panels, 16 exact 256×256 output/reference images, and [`v04` metric plot](figures/v04_image_formation_metrics.png) are retained.

## v0.5: camera-independent outgoing boundary field

v0.4 is retained byte-for-byte as the reference image-formation oracle, but it is not called by the v0.5 renderer. The new primary factorization is

\[
H_\Sigma=T(F),\qquad I_c=M_cH_\Sigma,
\]

instead of $I_c=R_c(F)$. `boundary_transport.py` is the only new module that queries the zero set. It receives no camera or detector: 262,144 deterministic zero-set samples emit along a global nested Fibonacci atlas containing 136 directions. An outward packet is retained only when its enclosing-sphere time precedes the next sign-changing zero-set intersection. Each of the 15,151,403 retained events stores boundary position $y_i$, outgoing direction index $\omega_i$, constant radiance $C_i$, weight $w_i$, source owner, and path length. The event digest is `596857e9…f80fe`.

`light_field.py` compresses those events into sorted occupied $(\omega,s,t)$ cells. The transverse $(s,t)$ coordinates and direction uniquely determine the outgoing ray and its intersection with the spherical boundary. The 136×256×256 conceptual field has 8,912,896 possible cells; 2,570,884 are occupied (28.84%). The compressed field is 98.08 MiB versus 204.0 MiB for dense float64 RGB, although retaining the auditable raw event table costs 1,155.97 MiB. All 136 angular bins are occupied. The union of events covers all 32,768 cells of the separate 128×256 spherical position diagnostic, so spatial occupancy alone is uninformative; joint spatial-angular occupancy is the relevant statistic.

`detector_readout.py` imports no field, mesh, tracer, or v0.4 image-formation code. A detector pixel is converted to a boundary position/direction query and locally samples the frozen field. It never projects a Bunny point, performs a Bunny depth test, or intersects the scene. The selected readout uses four angular neighbours, $h_\omega=0.12$ rad, a $h_y=0.65$ pixel local spatial kernel, and support threshold 0.5. Nearest slice, angular k-NN, spatial KDE, and combined spatial-angular KDE are all retained as controlled comparisons.

The 256² single-view result passes the required gate with IoU **0.960**, F1 **0.980**, SSIM **0.940**, PSNR **21.85 dB**, and two-pixel edge F1 **0.945**. The [reference/output/difference panel](render_res/v05_single_view.png) shows that disagreement is concentrated on the silhouette boundary. Eight detectors instantiated only after $H_\Sigma$ was frozen reach mean IoU/F1 **0.956/0.978**, SSIM **0.937**, PSNR **21.82 dB**, and edge F1 **0.950**; the complete [V8 oracle/field montage](render_res/v05_multiview.png) contains recognizable Bunny silhouettes in every view.

The post-hoc camera direction is not an atlas member: it is introduced only after field construction and lies 0.0321 rad from its nearest stored direction. Local angular interpolation gives IoU **0.960**, F1 **0.980**, SSIM **0.939**, PSNR **21.88 dB**, and edge F1 **0.995** in the [post-hoc comparison](render_res/v05_posthoc_camera.png). Discarding the first eight-camera ensemble and creating a disjoint eight-camera set from direction IDs 64–71 gives mean IoU **0.955**, F1 **0.977**, and SSIM **0.933**, with no scene traversal.

The event and field digests remain identical after the 1/2/4/8/20/40-camera tests, the off-atlas camera, and the ensemble swap. In the retained run, scene transport costs 62.18 s and sparse field construction 0.461 s once. Total readout is 0.017/0.035/0.070/0.138/0.348/0.695 s for 1/2/4/8/20/40 detectors, approximately 16.7 ms per view. Thus measured execution follows a fixed 62.64 s scene-plus-field term plus an approximately linear detector term; it does not retrace geometry as cameras are added.

Nested sample convergence exposes a real coverage threshold rather than monotone photon improvement:

| Outgoing samples | Boundary events | Joint occupancy | IoU | F1 | SSIM | Edge F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 278,650 | 236,743 | 0.090 | 0.162 | 0.278 | 0.658 | 0.356 |
| 1,114,279 | 946,919 | 0.222 | 0.831 | 0.908 | 0.729 | 0.306 |
| 4,457,212 | 3,787,923 | 0.287 | **0.964** | **0.982** | **0.946** | **0.962** |
| 17,828,865 | 15,151,403 | 0.288 | 0.960 | 0.980 | 0.940 | 0.945 |

The first level is a transport-coverage failure and the second is marginal; quality saturates by 4.46M samples, after which field discretization and bandwidth dominate. This rules out both a fundamental readout failure and the claim that arbitrary extra samples necessarily improve the image. Spatial bandwidth 0/0.65/1.25 pixels changes the dense-level post-hoc IoU by less than 0.001; angular bandwidth 0.04/0.12/0.24 rad gives IoU 0.959/0.960/0.960, so success does not depend on massive smoothing.

The CPU/CUDA gate agrees exactly on owner IDs, direction IDs, and sparse keys; maximum boundary-position and image errors are $2.78\times10^{-16}$ and $3.33\times10^{-16}$. Three repeated detector reads are bit-identical. Automated AST/interface audit confirms that scene construction has no camera parameter and detector readout has no scene/geometry parameter or forbidden scene import. The resulting verdicts are `CAMERA_INDEPENDENT_TRANSPORT_FIELD_SUPPORTED`, `CAMERA_FREE_SCENE_TRAVERSAL_SUPPORTED`, `BUNNY_FROM_TRANSPORT_FIELD_SUPPORTED`, `POSTHOC_VIEW_SYNTHESIS_SUPPORTED`, `SHARED_MULTIVIEW_TRANSPORT_IMAGE_FORMATION_SUPPORTED`, and `NO_GEOMETRY_CAMERA_PROJECTION_CONFIRMED`.

Exact timings, hashes, regional interior/boundary errors, all per-view measurements, support fractions, interpolation/bandwidth rows, convergence, memory, camera scaling, CPU/CUDA checks, and audit evidence are in [`artifacts/v05_camera_independent_transport.json`](artifacts/v05_camera_independent_transport.json) and [`artifacts/v05_camera_independent_transport.csv`](artifacts/v05_camera_independent_transport.csv). Compact [quality/convergence](figures/v05_quality_vs_samples.png), [coverage](figures/v05_light_field_coverage.png), [camera scaling](figures/v05_camera_scaling.png), and [bandwidth](figures/v05_bandwidth_sensitivity.png) plots are also retained.

## v0.6: mesh-free forward RGB zero-set rendering

v0.6 changes the production chain to

\[
F\longrightarrow\text{sign-changing cells}\longrightarrow
\{p_i,n_i,C_i\}\longrightarrow\text{forward boundary events}\longrightarrow
H_\Sigma\longrightarrow M_cH_\Sigma.
\]

There is no marching-cubes surface, triangle-area sample, mesh visibility query, or camera-to-geometry ray in that chain. `meshfree_surface.py` first finds every trilinear grid cell whose eight corner values bracket zero. An unscrambled four-dimensional Sobol sequence deterministically chooses cells and in-cell seeds. Thirty-two voxel-clamped Newton steps use the exact trilinear gradient; a sign-changing cell-edge root is the safeguarded fallback. On the retained 160³ Bunny field, all 60,882 sign-changing cells are covered by 131,072 samples. There are 107,611 Newton successes and 23,461 edge fallbacks, but zero failed roots, zero degenerate normals, and zero NaN/Inf values. Maximum/mean/median residuals are $9.98\times10^{-10}$, $6.90\times10^{-12}$, and $5.55\times10^{-17}$. The [three-axis coverage plot](figures/v06_meshfree_surface_coverage.png) visually confirms the ears, head, torso, legs, and tail. Evaluation-only comparison to repaired-mesh samples gives symmetric Chamfer 0.0101; that mesh is never read by the surface sampler or renderer.

Each sample receives a camera-independent analytic RGB albedo $C_i$ from its scene position. It emits into the same global 136-direction Fibonacci atlas as v0.5. A packet is absorbed when its next sign-changing zero-set intersection precedes its enclosing-sphere exit; the retained RGB is

\[
L_i(\omega)=C_i\exp(-\sigma d_i)
\left[a+(1-a)\max(0,n_i^T\omega)^\gamma\right].
\]

The selected interpretable setting is $\sigma=0.55$, $\gamma=1$, $a=0.35$, followed by a fixed sensor gain of 1.5. Of 8,914,043 outward packets, 1,316,657 re-intersect and are absorbed; 7,597,386 reach the boundary. Their 579.64 MiB auditable event table compresses to a 97.98 MiB sparse field with 2,568,116 of 8,912,896 bins occupied (28.81%). Each event writes at most four bilinear spatial bins, and a detector pixel reads at most 4 angular × 25 local spatial bins. Source owner IDs are retained, but v0.6 does not rebuild or claim a new end-to-end geometry-parameter Jacobian.

The [single-view output/reference/difference panel](render_res/v06_rgb_single_view.png) is visibly image-like rather than a binary occupancy mask: both ears, the face, shoulder fold, torso curvature, hind leg, and tail have continuous tonal structure. It reaches IoU **0.963**, F1 **0.981**, SSIM **0.937**, PSNR **31.47 dB**, RGB MAE **0.00679**, and edge F1 **0.951** against the evaluation-only RGB ray oracle. Foreground luminance standard deviation is **0.0953**, interior gradient energy is **0.0125**, and 5-bit quantization retains 535 foreground colors; the v0.5-style constant-field baseline has one color and effectively zero interior gradient. The [controlled operator comparison](render_res/v06_renderer_comparison.png) shows the old sparse hitmap, v0.4 projection color, v0.5 constant silhouette, v0.6 forward RGB, and oracle side by side.

The [attenuation/lobe image ablation](render_res/v06_attenuation_ablation.png) separates five cases. Distance attenuation alone changes depth-dependent brightness but retains smooth position color; its mild and strong settings produce interior gradient energies 0.00356 and 0.00343. The cosine lobe creates most local relief (0.01998 without attenuation); combining it with mild attenuation gives the selected balanced 0.01252. Thus distance attenuation contributes depth variation, but is not alone sufficient for the strongest surface relief. Numeric curves are retained in the [ablation plot](figures/v06_attenuation_ablation.png).

One scene traversal (32.62 s) and one sparse-field build (0.261 s) feed all eight 256² outputs in the [multiview montage](render_res/v06_rgb_multiview.png). Mean eight-view IoU/F1/SSIM/PSNR are **0.960/0.979/0.926/31.46 dB**. Readout costs 0.017/0.069/0.138 s for 1/4/8 cameras with zero scene retraversals, and all pre/post-camera event and field hashes are identical. In a stricter 16,384-sample 128² control, rebuilding the complete deterministic scene independently for each detector agrees with the shared path within $3.33\times10^{-16}$ and takes 4.407 s versus 0.450 s at eight views, a 9.80× speedup. CPU/CUDA maximum point and image discrepancies are $3.33\times10^{-16}$ and $1.22\times10^{-15}$.

The supported verdicts are `MESH_FREE_ZERO_SET_SURFACE_SAMPLING_SUPPORTED`, `MESH_FREE_FORWARD_RENDERING_SUPPORTED`, `DEPTH_SENSITIVE_RGB_IMAGE_FORMATION_SUPPORTED`, `FORWARD_SCENE_CENTRIC_RENDERING_PRESERVED`, `FORWARD_SCENE_CENTRIC_RENDERER_RETAINS_MULTIVIEW_CONSISTENCY`, and `MESH_FREE_NUMERICAL_ROBUSTNESS_SUPPORTED`. Full configuration, hashes, timings, quality rows, ablations, mesh-free diagnostics, sparsity scope, consistency checks, code audit, and limitations are in [`artifacts/v06_meshfree_forward_rgb.json`](artifacts/v06_meshfree_forward_rgb.json) and [`artifacts/v06_meshfree_forward_rgb.csv`](artifacts/v06_meshfree_forward_rgb.csv). Eight exact 256×256 images are retained as `render_res/v06_bunny_rgb_view*_256.png`; [view timing](figures/v06_multiview_runtime.png) is also retained.

## v0.7: attenuation decision and corrected-RGB geometry birth

### Appearance ablation

The paired experiment reuses one frozen 131,072-sample, 136-direction visibility table for exactly four image-formation variants. The position-dependent RGB field is an interpretable camera-independent diagnostic, not a material model. Normal lighting is $0.35+0.65\max(0,n^T\omega)$; attenuation, when enabled, is $\exp(-0.55d)$. All table metrics use the old v0.6 attenuated-and-lit oracle as one common reference.

| Variant | Attenuation | Lighting | IoU | F1 | SSIM | PSNR (dB) | RGB MAE | Assessment |
|---|:---:|:---:|---:|---:|---:|---:|---:|---|
| A | no | no | 0.9625 | 0.9809 | 0.7800 | 12.49 | 0.12372 | weak internal structure |
| B | yes | no | 0.9625 | 0.9809 | 0.8447 | 22.65 | 0.02723 | weak internal structure |
| C | no | yes | 0.9625 | 0.9809 | 0.8726 | 15.97 | 0.08287 | **strong internal structure** |
| D | yes | yes | 0.9625 | 0.9809 | 0.9367 | 31.47 | 0.00679 | **strong internal structure** |

C clearly retains both ears, face/head, shoulder and torso folds, leg, tail, and continuous shading. Its foreground luminance standard deviation/interior gradient energy are 0.1595/0.01998, both higher than D's 0.0953/0.01252. The raw C–D brightness differs because attenuation changes exposure, but a single least-squares exposure scale gives SSIM 0.9838 and RGB MAE 0.01672. The verdict is therefore `ATTENUATION_NOT_NEEDED`: the main/default formulation now uses C ($\sigma=0$), while attenuation remains available only as an optional diagnostic. See the [A/B/C/D panel](render_res/v07_abcd_reference.png), [focused v0.6/C/D comparison](render_res/v07_v06_C_D_reference.png), and exact [JSON](artifacts/v07_appearance_ablation.json)/[CSV](artifacts/v07_appearance_ablation.csv).

### Birth under the corrected forward RGB operator

The birth rerun starts at 32 active Wendland DoFs with a 4,096-element multiscale candidate dictionary and no prescribed active-DoF budget. It uses eight direct coarse-atlas views at 256², 32,768 mesh-free zero-set samples, forward re-intersection visibility, C appearance, and a continuous cubic boundary-light-slab footprint. For a fixed outgoing direction, projecting an event at the observation boundary or its source point onto the transverse light slab is mathematically identical; the measured maximum discrepancy is $4.44\times10^{-16}$. Visibility and pixel support are frozen only inside a local optimization cell.

The sparse analytic Jacobian differentiates implicit normal-line displacement, diagnostic color, the continuous interpolated normal field, cosine lighting, footprint motion, and normalized RGB accumulation. It intentionally does not differentiate visibility/ownership topology changes. Four finite-difference columns pass with maximum relative error **0.0166**. Batch sizes adapt as 32 below 256 DoFs, 64 below 512, and 128 thereafter. Every proposed batch receives an explicit multiview Gram coupling check; the retained maximum pairwise cosine and off-diagonal ratio are 0.841 and 0.377, below the declared 0.95/4.0 limits.

| Method at K=1,664 | RGB loss | RGB PSNR | RGB MAE | Symmetric Chamfer | P2S P95 | Runtime (s) | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| quadratic observation birth | 902.156 | 29.404 | 0.00516 | 0.005054 | 0.005275 | 23.81 | 1517 |
| matched uniform | 921.586 | 29.311 | 0.00526 | 0.005067 | 0.005603 | 15.69 | 1404 |
| matched random | 948.477 | 29.186 | 0.00556 | 0.005251 | 0.006607 | 13.52 | 1223 |
| matched raw alignment | **898.727** | **29.420** | **0.00503** | **0.004995** | **0.005261** | 19.71 | 1616 |
| cold fixed-space | 921.023 | 29.314 | 0.00527 | 0.005071 | 0.005597 | 12.64 | 1909 |

Quadratic observation-driven birth remains useful: it beats matched uniform and random in final image and geometry errors, and its image-loss/Chamfer trajectory AUCs (909.69/0.005114) beat uniform (939.05/0.005181). It is therefore parameter-efficient relative to uniform addition. It is not the overall winner: raw alignment gives lower final image loss and Chamfer, and a slightly better Chamfer AUC (0.005093). The corrected-renderer conclusion is `CORRECTED_FORWARD_RGB_BIRTH_PARTIALLY_REVALIDATED_RAW_WINS`, not a blanket confirmation of the former quadratic ranking.

The run reaches **1,664 active DoFs** in 20 rounds with mean/median/maximum batch sizes 81.6/64/128. It stops only after three consecutive zero-realized-gain rounds at 1,408/1,536/1,664 (`MARGINAL_IMAGE_GAIN_FLATTENED`), rather than at 256, 512, or 1,024. Joint predicted versus realized gains have Spearman 0.976 and Pearson 0.848. Births concentrate in independently defined high-error regions: 41.4% land in the top quartile and 19.8% in the top decile, versus 25%/10% nominal occupancy. Candidate scoring plus Gram checks costs 6.93 s, optimization 14.06 s, and the complete quadratic trajectory 23.81 s; optimization is the dominant measured component. Since uniform takes 15.69 s, observation-driven selection is not compute-efficient despite being parameter-efficient against uniform.

The [scaling curves](figures/v07_corrected_birth_scaling.png) and [target/reconstruction comparison](render_res/v07_corrected_birth_reconstruction.png) show all matched methods. Cold fixed-space references are independently optimized at K=256/512/1024/1664. Exact per-round rows, selected IDs, predicted/realized gains, coupling diagnostics, geometry metrics, timings, VRAM, visibility hashes, configuration, and limitations are retained in the [JSON report](artifacts/v07_corrected_birth.json) and [CSV summary](artifacts/v07_corrected_birth.csv).

## v0.8: Full-HD high-bandwidth birth stress test

v0.8 keeps the corrected C appearance path unchanged: no attenuation, analytic position color, and the $0.35+0.65\max(0,n^T\omega)$ normal-lighting term. The controlled Bunny run uses 65,536 mesh-free zero-set emitters, 20 deterministic Fibonacci directions/views, and 1920×1080 images. One direct packet is attempted per emitter and view, so `packets_per_emitter=20` and the aggregate is 1,310,720 attempted packets. The exact camera-direction digest is `e89f33d73a9319b92784990452552a6864798236a1574ba4bb12c68074d86acf`. Of 652,940 outward base events, 564,282 survive re-intersection visibility. This is only 0.0136 retained events per view-pixel before the 4×4 cubic footprint, an important distinction between aggregate million-packet scale and per-view Full-HD coverage. The optional 131,072-emitter/2.62M-attempt level was not run: this main optimization already peaked at 12.28 GiB allocated and about 14.0 GiB process-resident on the 16 GiB Quadro RTX 5000, so doubling event storage lacked safe local headroom.

The run starts at K=32 inside an 8,192-candidate multiscale dictionary. Its allowable capacity is $p_{max}=100K$, but each actual batch is the smallest score-ordered prefix that captures 95% of the mass above 5% of the best score, followed by spatial exclusion and explicit Gram pair-cosine/off-diagonal gates. It stops after three consecutive rounds with both relative RGB gain below $2\times10^{-4}$ and absolute Chamfer gain below $10^{-7}$, once K≥1024. The analytic corrected-RGB Jacobian passes four finite-difference columns with maximum relative error 0.0231.

| Matched method at K=4,963 | RGB loss | PSNR | Chamfer | P2S mean | P2S p95 | surface RMS | normal error | Time (s) | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| quadratic dynamic birth | 692355.04 | 19.5352 | 0.0049231 | 0.0031504 | 0.0082018 | 0.0058564 | 0.015322 | 73.36 | 12282 |
| raw-alignment matched birth | **692307.55** | **19.5355** | 0.0049182 | 0.0031439 | 0.0081557 | 0.0058526 | 0.015333 | 69.79 | 12212 |
| uniform matched | 692340.38 | 19.5353 | 0.0049749 | 0.0032110 | 0.0084523 | 0.0059140 | 0.015373 | 63.78 | 12229 |
| random matched | 692325.59 | 19.5354 | **0.0048212** | **0.0030301** | **0.0078520** | **0.0057521** | **0.015224** | **61.70** | 12220 |

Raw selection retains a small RGB advantage and beats uniform in both RGB loss and Chamfer, but random has the best geometry. Quadratic is worse than raw on both reported final errors and almost entirely fails to realize its predicted gains. Its predicted-versus-realized Pearson/Spearman correlations are 0.089/−0.200; raw gives 0.450/−0.200. There are only four batches, so these correlations are a diagnostic rather than a population estimate. Under the strict requirement that an observation policy beat both uniform and random in image and geometry quality, `BIRTH_SUPPORTED`, `OBSERVATION_DRIVEN_BIRTH_SUPPORTED`, `RAW_ALIGNMENT_BEST_SUPPORTED`, and `HIGH_BANDWIDTH_BIRTH_SUPPORTED` are false. `QUADRATIC_BEST_NOT_SUPPORTED` and `COMPUTE_EFFICIENCY_NOT_SUPPORTED` are true. Raw is DoF-efficient only in the narrower matched-uniform trajectory-AUC comparison; neither global quality nor compute superiority is established.

The dynamic quadratic batches are genuinely score-driven rather than capped at 32:

| Round | K before→after | Batch | Best/floor quadratic score | Pair max / $\rho_{off}$ | Predicted joint gain | Realized RGB gain | Chamfer gain | Birth-only jump |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32→129 | 97 | 83.992 / 4.200 | 0.915 / 0.932 | 697.72 | 0.01754 | −1.31e−4 | 1.68e−16 |
| 2 | 129→1,022 | 893 | 10.479 / 0.524 | 0.928 / 0.783 | 1152.00 | 0 | 0 | 6.21e−16 |
| 3 | 1,022→1,788 | 766 | 5.795 / 0.290 | 0.758 / 0.011 | 336.26 | 0 | 0 | 4.71e−16 |
| 4 | 1,788→4,963 | 3,175 | 0.442 / 0.022 | 0.940 / 0.083 | 408.06 | 0 | 0 | 6.21e−16 |

Every round is limited by predicted-gain saturation, never by $p_{max}$. The coupling gates reject only 0/0/2/1 candidates. Thus v0.7's repeated 32 was the legacy `_batch_size` rule below K=256, not a natural capacity of this candidate family. The v0.8 mean/median/maximum batch sizes are 1232.75/829.5/3175. The scientific stop is `MARGINAL_IMAGE_AND_GEOMETRY_GAIN_FLATTENED`; it is not a dictionary, runtime, or VRAM stop. Birth-only jumps remain at floating-point zero, showing that adding zero-coefficient parameters is continuous. The selected geography is mildly observation-related—30.85% of births lie in an independently defined top error quartile and 13.28% in the top decile—but it becomes spatially broad by round four.

Cold fixed-space optimization exposes substantial image-fitting headroom that birth does not reach:

| Fixed K | Cold RGB loss | Cold Chamfer | Warm RGB loss | Warm Chamfer |
|---:|---:|---:|---:|---:|
| 256 | 691218.10 | 0.0049343 | 691218.10 | 0.0049343 |
| 512 | 690768.53 | 0.0049585 | 691217.26 | 0.0049365 |
| 1024 | 689395.07 | 0.0049423 | 691212.86 | 0.0049397 |
| 2048 | 689932.30 | 0.0049919 | 691212.86 | 0.0049397 |
| 4096 | 689532.39 | 0.0050094 | 691212.86 | 0.0049397 |
| 4963 | **689111.18** | 0.0049801 | 691212.86 | 0.0049397 |

Even cold K=256 fits the images better than every K=4963 birth policy, while geometry is non-monotone and random matched birth gives the lowest Chamfer. Quality advantage, DoF efficiency, and compute efficiency therefore cannot be collapsed into one claim: birth loses the image-quality headroom comparison; raw has a limited AUC advantage over uniform but not random; and scoring makes raw/quadratic slower than their matched controls. Warm continuation itself flattens after K=1024, another sign that optimizer path dependence is material.

The 20-view 2×2 ablation uses a simplified one-shot K=1024 selection and separately includes fixed K=256 and K=1024. Values below are normalized RGB MSE / symmetric Chamfer:

| Resolution / photons | Raw | Uniform | Fixed K256 | Fixed K1024 |
|---|---:|---:|---:|---:|
| 256² / 327,680 | 0.00322168 / 0.0069479 | 0.00323633 / 0.0069486 | 0.00326970 / 0.0069908 | 0.00323602 / 0.0069514 |
| 256² / 1,310,720 | 0.00093181 / **0.0039896** | 0.00094149 / 0.0040550 | 0.00096365 / 0.0042399 | 0.00094162 / 0.0040482 |
| 1080p / 327,680 | 0.00343465 / 0.0077381 | 0.00345205 / 0.0074029 | 0.00345817 / 0.0073469 | 0.00344229 / 0.0077120 |
| 1080p / 1,310,720 | 0.01112844 / 0.0049572 | 0.01112894 / 0.0048932 | 0.01111140 / 0.0049343 | 0.01108210 / 0.0049423 |

Increasing photon density improves raw Chamfer at both resolutions (−0.002958 at 256² and −0.002781 at 1080p), so `PHOTON_DENSITY_GEOMETRY_EFFECT_SUPPORTED` is true. Increasing resolution worsens raw Chamfer at both photon budgets (+0.000790/+0.000968), so `RESOLUTION_GEOMETRY_EFFECT_NOT_SUPPORTED` is true. The normalized image objective is not comparable as a pure reconstruction score across photon budgets at 1080p because the target support itself becomes denser: increasing photons reveals many more non-black residual pixels. The defensible bandwidth verdict is therefore only `OBSERVATION_BANDWIDTH_LIMIT_PARTIALLY_REDUCED`: photon density reduces geometry error, but nominal resolution does not, and aggregate million-packet count still leaves sparse per-pixel support.

The dominant bottleneck has shifted toward optimization plus footprint coverage, not dictionary capacity. Quadratic candidate scoring and Gram checks take 8.65 s versus 59.47 s in optimization, yet hundreds of units of predicted joint gain produce zero accepted improvement after the first round. Frozen visibility/support makes the local Jacobian blind to topology changes, and each view receives only about 28k retained events for 2.07M pixels. The next useful experiment is therefore not a larger K or still higher resolution: it is denser per-view transport support together with a trust-region/blocked optimizer that can validate and realize large-batch steps, followed by a repeated-seed/random-policy check. Exact rows and signed factorial effects are in the [JSON](artifacts/v08_high_bandwidth.json) and [CSV](artifacts/v08_high_bandwidth.csv). See [quality-vs-DoF/time](figures/v08_high_bandwidth_quality_curves.png), [predictor calibration](figures/v08_high_bandwidth_predictor_calibration.png), [batch trajectory](figures/v08_high_bandwidth_batch_trajectory.png), [birth geography](figures/v08_high_bandwidth_birth_geography.png), [factor ablation](figures/v08_bandwidth_factor_ablation.png), and the [four-view target/reconstruction montage](render_res/v08_high_bandwidth_multiview.png).

## v0.8.1: Full-HD sampling and RGB-loss diagnostic

v0.8.1 does not rerun or rewrite the v0.8 experiment. It first audits the exact image operator used there. An outward, visible event is projected into orthographic pixel coordinates and written through a separable cubic B-spline. In the legacy path,

\[
w_{ip}=B(r_p-r_i)B(c_p-c_i),\qquad
I_p=\operatorname{clamp}\!\left(g\frac{\sum_i w_{ip}C_i\ell_i}{\sum_i w_{ip}},0,1\right),
\]

where each one-dimensional kernel has support radius two **pixels**. Thus one event has at most 16 writes, regardless of detector resolution. RGB is a normalized weighted average rather than an energy sum or pixel-area integral. Neither packet count, pixel area, resolution, view count, nor RGB-channel count normalizes the archived objective

\[
L=\tfrac12\sum_{v,p,c}(I_{vpc}-T_{vpc})^2,
\qquad \nabla_aL=\sum_vJ_v^T(I_v-T_v).
\]

Target and reconstruction use the same deterministic surface identities, direction atlas, equations, and common sample realization, while visibility and support are constructed for their respective geometry and then frozen inside a Jacobian cell.

### Observation

With 65,536 emitters and 20 views fixed, the geometry is numerically identical at every resolution (symmetric Chamfer 0.00451335), but the detector estimator is not:

| Resolution | Half raw SSE | MSE / RGB scalar | RMSE | Unique coverage | Footprint writes / pixel | Residual / target energy |
|---:|---:|---:|---:|---:|---:|---:|
| 256² | 2,043.47 | 0.001039 | 0.03224 | 0.29296 | 6.8882 | 0.00913 |
| 512² | 40,912.78 | 0.005202 | 0.07213 | 0.27986 | 1.7221 | 0.05340 |
| 960×540 | 175,280.14 | 0.011271 | 0.10616 | 0.26482 | 0.8708 | 0.17174 |
| 1920×1080 | 692,587.41 | 0.011133 | 0.10551 | 0.14577 | 0.2177 | 0.55346 |

The Full-HD half-SSE is 338.93× the 256² value. Of that change, 31.64× is the increase in pixel count, while normalized MSE still worsens by 10.71×. Therefore 692k is partly a sum-reduction scale effect and partly a genuine resolution-dependent estimator failure; it is not a comparable cross-resolution quality number by itself.

Exactly zero-support Full-HD pixels contain only 7.06% of loss and 3.91% of target energy, so `RGB_RESIDUAL_DOMINATED_BY_UNCOVERED_PIXELS=false` under the declared count≥1 definition. The pathology is nevertheless weak support: pixels with fewer than two contributions contain 69.59% of loss, and those with fewer than four contain 97.97%. The [coverage/contribution panel](figures/v081_target_render_coverage_contributions.png) and [covered/uncovered residual panel](figures/v081_covered_uncovered_residual.png) retain the exact diagnostic buffers.

Repeating the identical render with identical samples gives raw SSE exactly zero. Independent Sobol scrambles, rendered with one fixed scene-centered detector so that only samples change, give MSE 0.02667 at 256² and 0.02800 at Full HD. The Full-HD independent-sample self-SSE is 2.560× the measured geometry-reconstruction SSE. Common random numbers are therefore necessary and are already used by v0.8: independent Monte Carlo noise would dominate, but it does not directly enter the deterministic production objective. Held-out samples remain necessary for validation.

At approximately matched footprint-write density, 256²/4,096, 512²/16,384, and 1024²/65,536 emitters hold unique coverage within 0.00518 and density near 0.431 writes/pixel. RMSE still rises 0.06133→0.08265→0.11082. This rejects a packet-count-only explanation and points to the detector-space kernel scale. See the [matched-density control](figures/v081_matched_density_resolution.png).

### Inference and fix

The legacy four-pixel-wide footprint shrinks in normalized detector coordinates as resolution rises. v0.8.1 therefore adds an opt-in reference-resolution footprint. For $s_h=H/256$ and $s_w=W/256$,

\[
w_{ip}=\frac{B((r_p-r_i)/s_h)B((c_p-c_i)/s_w)}{s_hs_w},
\]

with analytic motion derivatives scaled consistently and the support threshold divided by $s_hs_w$. This is a change of reconstruction-kernel coordinates, not arbitrary post-render blur. The reciprocal area factor cancels in normalized RGB but preserves density/support semantics. Legacy behavior remains the default, so the v0.7/v0.8 operator and artifacts are unchanged.

| Resolution | Legacy coverage / RMSE | Aware coverage / RMSE | Aware footprint area |
|---:|---:|---:|---:|
| 256² | 0.28920 / 0.05736 | 0.28920 / 0.05736 | 16.0 px |
| 512² | 0.21693 / 0.08334 | 0.28914 / 0.05737 | 64.0 px |
| 960×540 | 0.14861 / 0.08033 | 0.28915 / 0.05730 | 126.60 px |

Across the three aware controls the coverage range is only $6.23\times10^{-5}$ and the RMSE range is $6.79\times10^{-5}$, which is direct numerical evidence for `FOOTPRINT_RESOLUTION_SCALING_BUG_FOUND=true`. A Full-HD aware-Jacobian run is deliberately not forced on this 16 GiB GPU because the explicit sparse stencil would grow by 31.64×; the controlled 960×540 test is the safe highest non-square level. A future Full-HD production implementation should stream or factor this expanded stencil.

### Candidate and post-fix birth result

Sparse Full-HD rendering does not make the 480 inactive quadratic scores indiscriminately tied: all are positive, but only 0.208% lie within 50%, 75%, or 90% of the best and the coefficient of variation is 2.94. The quadratic score has weak negative Spearman correlation with local zero-coverage fraction (−0.0766), moderate correlation with local RGB residual (0.2896), and positive correlation with independent geometry error (0.3708). Consequently the evidence rejects “score follows coverage holes” and supports a reduced but nonzero geometry association. The exact low/Full-HD distributions are in the [coverage](figures/v081_candidate_score_vs_coverage.png) and [geometry-error](figures/v081_candidate_score_vs_geometry_error.png) plots.

The minimal post-fix test uses 512², 20 views, 16,384 training emitters (Sobol scramble 101), an independent 16,384-emitter held-out realization (211), a fixed 256-entry dictionary, K=32→64, and only two post-birth optimization steps. The resolution-aware analytic Jacobian passes central finite differences with maximum relative error 0.05740 below the 0.08 gate.

| Method | Half raw SSE | Normalized MSE | Chamfer | P2S mean | Normal error | Same-sample gain | Held-out gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| no birth, K32 | 26,375.34 | 0.0033538 | 0.0071666 | 0.0025189 | 0.014929 | — | — |
| random, K64 | 26,345.80 | 0.0033500 | 0.0071580 | 0.0025077 | 0.014867 | 29.53 | 36.31 |
| fixed-space, K64 | 26,302.96 | 0.0033446 | 0.0071463 | 0.0024848 | 0.014867 | 72.38 | 59.31 |
| raw birth, K64 | 26,265.46 | 0.0033398 | **0.0071450** | **0.0024844** | 0.014695 | 109.88 | **106.54** |
| quadratic birth, K64 | **26,265.22** | **0.0033398** | 0.0071468 | 0.0024855 | **0.014678** | **110.12** | 105.77 |

For batch sizes 1/4/16/32, raw same-sample gains are 59.13/77.99/96.59/109.88 and held-out gains are 40.87/69.51/93.12/106.54; quadratic gives 14.20/79.80/97.26/110.12 and 15.96/67.51/93.10/105.77. The [gain plot](figures/v081_predicted_realized_same_heldout.png) shows that the recovered signal generalizes rather than merely fitting one realization.

### Updated scientific verdict

The evidence materially changes the interpretation of v0.8. Its strict `BIRTH_SUPPORTED=false` result remains an accurate record for that exact renderer/configuration, but it is **not a clean test of the underlying geometry-birth hypothesis**: the Full-HD detector footprint was resolution-inconsistent, and a small corrected control recovers observation-driven benefit over random on both training and held-out RGB as well as Chamfer. Optimizer failure is therefore not needed to explain the diagnostic result, although this test does not prove that every v0.8 large-batch optimization effect disappears.

The required verdicts are: `HIGH_RES_IMAGE_UNDERSAMPLED=true`, `RGB_LOSS_SCALE_RESOLUTION_DEPENDENT=true`, `RGB_RESIDUAL_DOMINATED_BY_UNCOVERED_PIXELS=false`, `MC_NOISE_FLOOR_DOMINATES_RGB_OBJECTIVE=false`, `FOOTPRINT_RESOLUTION_SCALING_BUG_FOUND=true`, `COMMON_RANDOM_NUMBERS_NEEDED=true` (and already used), `CANDIDATE_SCORE_CORRELATES_WITH_COVERAGE_HOLES=false`, `CANDIDATE_SCORE_CORRELATES_WITH_GEOMETRY_ERROR=true`, `BIRTH_FAILURE_EXPLAINED_BY_IMAGE_SAMPLING=true` in the material/partial sense above, `OPTIMIZER_FAILURE_STILL_NEEDED_TO_EXPLAIN_RESULTS=false`, `MATCHED_DENSITY_RESOLUTION_EFFECT_SUPPORTED=true`, and `BIRTH_SIGNAL_RECOVERS_AFTER_IMAGE_FIX=true`. Independent samples would dominate if substituted into the training objective, recorded separately as `INDEPENDENT_MC_WOULD_DOMINATE_RGB_OBJECTIVE=true`.

All thresholds, per-view buffers, raw/normalized RGB metrics, target/predicted/residual energies, score statistics, correlations, exact seeds, timings, and boolean evidence are in [`artifacts/v081_highres_sampling_diagnostic.json`](artifacts/v081_highres_sampling_diagnostic.json) and [`artifacts/v081_highres_sampling_diagnostic.csv`](artifacts/v081_highres_sampling_diagnostic.csv). The ten `v081_` figures are retained under [`figures/`](figures/). The formal local-CUDA run took 179.63 s and peaked at 5,901.64 MiB allocated / 6,310 MiB reserved.

## v0.8.2: Fixed-footprint matched-transport-density resolution control

### 1. Question

v0.8.2 asks whether higher detector resolution provides a better geometry observation signal when transport support per detector degree of freedom is held approximately fixed. The primary treatment keeps the legacy fixed 4×4 pixel-space cubic footprint and adds real emitter samples; it does not use the v0.8.1 enlarged footprint as a substitute for bandwidth.

### 2. Why v0.8 was confounded

At fixed 65,536 emitters, increasing 256² to 1920×1080 reduces footprint writes/pixel from 6.8882 to 0.21770 and unique coverage from 0.29296 to 0.14577. RMSE rises from 0.03224 to 0.10551. Resolution, samples/detector-DoF, and estimator support therefore changed together in the old Full-HD experiment; that result cannot by itself establish that resolution hurts geometry.

### 3. Low-resolution sampling reference

The exact 256²/65,536-emitter/20-view legacy reference is reproduced. Across 1,310,720 view-pixels and 1,310,720 attempted packets, it has 652,940 outward, 564,282 retained/projected events, and 9,028,512 footprint writes. The contribution counts are: coverage 0.2929565, zero 0.7070435, count<2 0.7082268, count<4 0.7102020, mean 6.8882, median 0, p90/p95/p99 23/33/57, and maximum 166; 383,984/382,433/379,844/375,146 pixels have at least 1/2/4/8 writes. The target/prediction energies are 447,613.17/439,484.76; raw SSE is 4,086.94, half-SSE 2,043.47, MSE 0.00103936, RMSE 0.032239, and MAE 0.005704. The median remains zero because the aggregate includes the black image background; density is assessed jointly with foreground coverage and the full count distribution. The [actual reference RGB](render_res/v082_low_reference_rgb.png) is archived.

### 4. Streaming renderer validation

The new path generates one global Sobol surface sequence, slices it into emitter chunks, performs outward filtering, zero-set visibility, projection, fixed-footprint splatting, and detector accumulation one view/chunk at a time, then discards chunk-local events. It never materializes the complete 41.47M-packet graph. The training set is the unscrambled prefix and independent MC sets use scrambles 101 and 211; every resolution uses `[0,N)` so larger settings are nested prefixes. Against an 8,192-emitter monolithic 64²/two-view reference, chunk sizes 1,024 and 2,048 both give exactly zero point, normal, base-image, target-image, event-count, and footprint-write differences (image tolerance 2e−6). Thus `STREAMING_TRANSPORT_VALIDATED`, `CHUNK_SIZE_INVARIANT`, and `COMMON_RANDOM_NUMBERS_VALIDATED` are true.

### 5. Matched-density construction

The pixel-ratio estimates passed all declared gates on their first transport calibration, so no adjustment was required:

| Resolution | Emitters | Attempted packets | Retained events | Writes/pixel | Coverage | Gate |
|---:|---:|---:|---:|---:|---:|---:|
| 256² | 65,536 | 1,310,720 | 564,282 | 6.88821 | 0.29296 | pass |
| 512² | 262,144 | 5,242,880 | 2,257,273 | 6.88865 | 0.28108 | pass |
| 960×540 | 518,400 | 10,368,000 | 4,463,891 | 6.88872 | 0.27806 | pass |
| 1920×1080 | 2,073,600 | 41,472,000 | 17,855,026 | 6.88851 | 0.27361 | pass |

At Full HD, the relative density error is 0.0000443, while coverage, zero-count, count<2, and median differences are 0.01935, 0.01935, 0.01832, and 0, all inside the 10%/0.03/0.03/0.05/one-count gates. Hence `LOW_RES_REFERENCE_DENSE=true` and `FULLHD_MATCHED_DENSITY_ACHIEVED=true`. See [density](figures/v082_density_vs_resolution.png), [emitter scaling](figures/v082_emitters_vs_resolution.png), and [count histograms](figures/v082_contribution_histograms.png).

### 6. Fixed-sample vs matched-density resolution test

| Resolution | Fixed-sample coverage / RMSE | Matched-density coverage / RMSE |
|---:|---:|---:|
| 256² | 0.29296 / 0.03224 | 0.29296 / 0.03224 |
| 512² | 0.27986 / 0.07213 | 0.28108 / 0.03526 |
| 960×540 | 0.26482 / 0.10616 | 0.27806 / 0.03573 |
| 1920×1080 | 0.14577 / 0.10551 | 0.27361 / 0.04077 |

Real sample scaling recovers Full-HD coverage by 0.12784 and density by 31.64× relative to the fixed-sample control, so `FIXED_SAMPLE_HIGHRES_DEGRADES=true` and `REAL_SAMPLE_INCREASE_NEEDED_FOR_HIGH_BANDWIDTH=true`. The matched images are visually continuous rather than the sparse v0.8 render, as shown by the [RGB/support montage](figures/v082_rgb_montage.png), [surface crops](figures/v082_rgb_zoom_crops.png), [count maps](figures/v082_transport_count_maps.png), and [weight maps](figures/v082_weight_maps.png). However, matched RMSE still increases 0.03224→0.04077; its max/min ratio is 1.26448, just above the declared 1.25 stability gate. Therefore `MATCHED_DENSITY_HIGHRES_IMAGE_STABLE=false`; this negative result is retained.

### 7. Monte-Carlo self-noise

Independent Sobol scrambles 101 and 211 give MC self-MSE 1.3895e−4, 2.0776e−4, 5.8519e−5, and 3.4340e−4 from low to high resolution (RMSE 0.01179, 0.01441, 0.00765, 0.01853). The max/min MSE ratio is 5.8682, failing the 1.25 gate, despite both independent sets having matched aggregate density to within 0.03%. The support-XOR fractions are only 0.000777/0.000604/0.000266/0.000690, while common-support MSE decreases 9.09e−5→9.74e−6. This implicates detector-support threshold/footprint phase changes in a very small set of pixels, rather than a failure to match total writes. `MC_SELF_NOISE_STABLE_AT_MATCHED_DENSITY=false`, so the protocol stops before geometry optimization and birth. See the [MC plot](figures/v082_mc_self_noise.png).

### 8. Geometry result

Only the required fixed imperfect geometry was evaluated: symmetric Chamfer 0.00451335, P2S mean/p95 0.00264811/0.00705907, surface RMS 0.00546712, and normal error 0.0149405 at every detector resolution. No resolution-dependent geometry optimization was run after the failed MC prerequisite; moreover, the existing explicit sparse Jacobian is not memory-safe at 2.07M emitters without a differentiable streaming/factored implementation. Accordingly `MATCHED_DENSITY_HIGHRES_GEOMETRY_IMPROVES=false` means **not established/not tested**, not measured degradation. The flat fixed-geometry line is shown in the [geometry control](figures/v082_fixed_vs_matched_density_geometry.png).

### 9. Small birth result

The K=32→64 birth comparison and five-seed follow-up were not executed because MC self-noise stability is a mandatory prerequisite. Consequently `HIGH_RES_BIRTH_SIGNAL_STRONGER=false`, `HIGH_RES_BIRTH_BEATS_RANDOM=false`, and `HIGH_RES_BIRTH_HELDOUT_SUPPORTED=false` are explicitly `NOT_TESTED_PREREQUISITE_FAILED` (MC ratio 5.8682 versus 1.25), not negative birth observations.

### 10. Scientific verdict

This run establishes that millions of real emitter samples can make fixed-footprint Full-HD transport quantitatively comparable to the 256² reference: all contribution-density gates pass, and streaming keeps peak allocation/reservation to 1,306.29/1,374 MiB. It also shows that matching aggregate writes and count distributions is not yet sufficient for a clean geometry-bandwidth claim: matched RGB narrowly misses its stability gate and independent-sample image error varies strongly because the hard support threshold amplifies rare sample-dependent support changes. The v0.8 sample-starvation confound is removed, but whether higher detector resolution improves geometry reconstruction or child selection remains unresolved. The next prerequisite is a threshold-robust detector estimator plus a streaming/factored Jacobian, followed by the same low/high K=32→64 and five-seed protocol. Exact metrics, seeds, gates, timings, and evidence are in the [JSON](artifacts/v082_high_sample_resolution_control.json) and [CSV](artifacts/v082_high_sample_resolution_control.csv); [runtime scaling](figures/v082_runtime_scaling.png) reports the practical cost.

## v0.8.3: Pixel accumulation, support switching, and gradient stability

### 1. Exact pixel estimator

v0.8.3 audits the unchanged legacy 4x4 detector operator before proposing any repair. For an in-frame event $i$, the integer anchor is `floor` of its continuous detector coordinate, the candidate offsets are −1, 0, 1, 2, and strictly positive separable cubic weights are accumulated. The exact sufficient statistics and output are

\[
N_p=\sum_i \mathbf 1[w_{ip}>0],\quad W_p=\sum_iw_{ip},\quad Q_p=\sum_iw_{ip}^2,
\]
\[
A_p=\sum_iw_{ip}C_i\left(0.35+0.65\max(0,n_i\!\cdot\!\omega_v)\right),\qquad
I_p=\mathbf 1[W_p\geq0.05]\operatorname{clamp}\!\left(1.5A_p/\max(W_p,10^{-30}),0,1\right).
\]

Thus $W_p$, but not $N_p$, is a random per-pixel denominator. Emitter count, packets/emitter, view count, retained-event count, pixel/detector area, and RGB-channel count are not separate denominators. Each view is formed independently.

### 2. Accumulation statistics

The primary pair uses scrambled Sobol seeds 101 and 211 and exactly the v0.8.2 matched-density prefixes. Writes/pixel remain 6.8876, 6.8884, 6.8881, and 6.8885 at 256², 512², 960×540, and 1080p; target-foreground writes/pixel are 23.84, 24.50, 24.66, and 24.94. Pixels with at least 64 contributions have self-MSE $5.56,4.50,3.66,14.97\times10^{-5}$, far below the corresponding 4–7 contribution bins $8.09,11.10,4.58,33.29\times10^{-3}$. Therefore many-emitter accumulation is not itself the failure: `MULTIPLE_EMITTER_ACCUMULATION_UNSTABLE=false` and `HIGH_COUNT_PIXELS_MORE_STABLE=true`.

### 3. Normalization audit

Nested 256² prefixes N/2N/4N = 65,536/131,072/262,144 produce foreground mean intensities 0.58419/0.58550/0.58634. Their relative range is 0.003669, below the declared 0.05 gate, while foreground mean $W_p$ scales 1.4970/2.9944/5.9888 as expected. The current ratio estimator therefore preserves brightness under sample scaling (`PIXEL_ESTIMATOR_NORMALIZATION_CORRECT=true`). On common support, median absolute Spearman correlations of pixel self-error with |ΔN| and |ΔW| are only 0.0585 and 0.0559, so neither ordinary count fluctuation nor random-denominator variation is the primary driver.

### 4. Support switching

The A/B support-XOR fractions are only 0.000777/0.000622/0.000263/0.000690, yet these pixels explain 81.36%/95.45%/91.22%/99.23% of total MC self-SSE. Their per-scalar MSE is 0.145/0.321/0.204/0.493, whereas common-support MSE falls from $9.14\times10^{-5}$ to $9.74\times10^{-6}$. The cubic value and first derivative remain mathematically continuous at an integer anchor: numerical total weight stays one in float32 and float64. Stencil membership nevertheless changes discretely (14 entries in the two-dimensional near-boundary test), and the final $W_p\geq0.05$ decision turns rare sampling differences into an on/off RGB jump. `COMPACT_SUPPORT_SWITCHING_DETECTED=true` and `SUPPORT_SWITCHING_DOMINATES_MC_SSE=true`.

### 5. Visibility switching

The exact hard decisions are outward $n\cdot\omega>10^{-8}$, intersection epsilon $2\times10^{-4}$, in-frame clipping, cubic `weight > 0`, and final support $W_p\geq0.05$. A same-identity, 65,536-emitter controlled direction perturbation finds visible↔occluded switches by $10^{-4}$ in some views and up to 0.000244 at $10^{-3}$; most state changes occur at grazing events. This establishes `VISIBILITY_SWITCHING_DETECTED=true`, although the detector support threshold is the more direct source of the measured image spikes.

### 6. Foreground/interior/silhouette decomposition

All coverage, zero/count<2/count<4, median, p90, p95, count, weight, intensity, and error statistics are reported separately for target, predicted, union, intersection, and >8 px interior foreground. The aggregate background is not called sparse foreground. The 0–8 px silhouette bands explain 82.65%/71.11%/53.78%/62.67% of primary self-SSE. The finest 0–1 px band alone explains 48.11% at 256². `SILHOUETTE_REGION_DOMINATES_MC_SSE=true`, while multiseed foreground and interior error both worsen substantially at Full HD.

### 7. Multi-seed MC result

Eight deterministic scrambled Sobol sets are recorded: seed 101 is the reference and seeds 211/307/401/503/601/701/809 are seven independent comparisons. Mean whole-image MSE (95% CI) is $1.286\times10^{-4}$ ([1.220,1.351]×10⁻⁴), $2.270\times10^{-4}$ ([2.165,2.374]×10⁻⁴), $6.891\times10^{-5}$ ([6.365,7.418]×10⁻⁵), and $3.288\times10^{-4}$ ([2.985,3.592]×10⁻⁴). Full HD exceeds 256² by 2.558× in every paired comparison; paired $t=12.73$, $p=1.44\times10^{-5}$. The unusual 960×540 low and 1080p high values are therefore stable effects of this raster/configuration, not one unlucky scramble. Support switches explain 80.22%/95.68%/92.07%/99.21% of mean SSE.

### 8. Sobol/grid test

Against the scrambled distribution at 1080p, the unscrambled Sobol result is 2.10 standard deviations away and pseudorandom uniform is 19.43 standard deviations away; pseudorandom self-MSE is $1.124\times10^{-3}$. The tested sequence family therefore materially affects the raster result (`SOBOL_GRID_RESONANCE_DETECTED=true`), but no checkerboard-only explanation is supported: measured checkerboard correlations remain below 0.008. Sobol structure modulates the instability; it does not replace the support-switch root cause.

### 9. Gradient stability

A controlled 256², four-view, 16,384-emitter Bunny case evaluates five basis coefficients selected as deep interior, smooth visible surface, silhouette, occlusion boundary, and high curvature over six epsilons from $10^{-2}$ to $3\times10^{-5}$. Best analytic-vs-frozen relative errors are 0.00590, 0.09719, 0.02049, 0.11651, and 0.04119; three of five pass the 0.08 all-pixel gate, so the strict all-category verdict is false. Independent-realization gradient cosines range from 0.424 to 0.796, confirming substantial sample-realization variance.

### 10. Frozen-support vs full-rerender finite differences

The same five best-epsilon analytic-vs-full relative errors are 1.00029/0.99986/0.99977/0.99755/0.99983, even though only $3.8\times10^{-6}$ to $3.4\times10^{-5}$ of RGB scalars change final support. Full rerender derivatives are dominated by discrete root/visibility/support changes that the local analytic Jacobian intentionally freezes. Hence both strict match booleans are false; the much closer frozen results isolate missing topology derivatives rather than a generic failure of the continuous spline derivative.

### 11. Root cause

Cases 1–3 are rejected: high-count pixels are more stable, and stable common-support |ΔN|/|ΔW| correlations are weak. Case 4 is the primary cause: rare compact/final-support changes carry most SSE. Case 5 is also supported by visibility switching plus silhouette concentration. Case 6 is a secondary sequence/raster interaction. The measured high-resolution effect is a combination of hard support/visibility topology and sampling phase, not overlap of many emitters.

### 12. Fix, if justified

No production estimator, footprint, visibility rule, optimizer, or birth score is changed in v0.8.3. A repair now requires a radiometrically justified treatment of the hard $W_p=0.05$ transition and detector integration, with before/after/control evidence; simply blurring, enlarging support, or dividing by counts would confound the diagnosis.

### 13. Remaining unknowns

The multiseed design is reference-versus-seven rather than all 28 pairs. Exact emitter-ID Jaccard overlap is undefined between independent point samples, so presence/count/weight/occupancy proxies are used. Full-resolution geometry Jacobians remain infeasible with the explicit sparse construction; the finite-difference study is the declared smaller controlled case.

### 14. Birth prerequisite verdict

`MC_SELF_NOISE_STATISTICALLY_CHARACTERIZED` and `NORMALIZATION_VALIDATED` pass, and the support root cause is numerically isolated. `INTERIOR_MC_STABLE` and `GRADIENT_STABILITY_ACCEPTABLE` fail: Full-HD/256² multiseed interior MSE is 6.714× and analytic Jacobians do not match full rerenders. Therefore `HIGH_RES_BIRTH_READY_TO_TEST=false`; no K=32→64 birth experiment was run, and this is a failed prerequisite rather than a negative birth result. The complete 14-part report, exact seeds and nested `[0,N)` four-dimensional Sobol ranges, thresholds, per-pixel cases, commands, 1,520.05 s runtime, 1,382.41/1,742 MiB recorded CUDA peak allocation/reservation, and all boolean evidence are in the [JSON](artifacts/v083_pixel_support_gradient_diagnostic.json) and [CSV](artifacts/v083_pixel_support_gradient_diagnostic.csv). The twelve required plots and four resolution-specific spatial maps are retained as [`figures/v083_*.png`](figures/).

## Limitations

v0.8 is one controlled Bunny run, not a general benchmark or a real-photo reconstruction. Its 20 direct Fibonacci packet directions are also its detector directions; it preserves a shared scene-centric transport state but does not validate arbitrary off-atlas cameras. The 8,192-element candidate dictionary is only a safety envelope. Visibility, ownership, and footprint topology remain frozen within each Jacobian cell. The 2×2 ablation changes the deterministic sampled target support with photon density, has no repeated stochastic trials, and uses a simplified one-shot K=1024 comparison; its signed effects are descriptive, not confidence intervals. In particular, a million attempted packets aggregated over 20 Full-HD views does not imply dense observations per pixel. The strict negative high-bandwidth verdict applies to this optimizer, dictionary, renderer approximation, and controlled target—not to every possible observation-driven birth method.

The historical statement below that adaptive stopping is omitted applies through v0.6; v0.7 adds a declared three-round marginal-gain stop after a minimum of 1,024 active DoFs. Its 4,096-element dictionary is a finite safety envelope, so the experiment is not evidence for unbounded scaling. The v0.7 birth path uses eight direct atlas directions and a differentiable cubic spatial light-slab readout rather than v0.6's auxiliary-direction angular blend. Visibility is recomputed for the base and target but frozen during each local optimization, so the Jacobian cannot predict disocclusion or owner changes. The 1,664-DoF plateau is established only for this controlled Bunny, dictionary, sampling bandwidth, and optimizer; raw-alignment superiority should not be generalized beyond them.

The prototype deliberately omits differentiability across collision/visibility topology changes, merge/prune policies, optimized basis centers or radii, adaptive stopping, neural fields, secondary reflection, indirect illumination, refraction, participating media, physical camera lenses, and physically calibrated radiometry. It also misses tangent zero-set intersections that touch without a sign change. Birth is tested with a fixed candidate bank and fixed transport cells on two synthetic sphere regimes and one controlled Bunny construction, not diverse objects, unknown geometry, or real photographs. The Bunny base is a smoothed grid rather than a deliberately decimated mesh, grid discretization contributes to measured error, and the five repair caps are only a sign proxy. Under stronger smoothing, closest-point emitter transfer changes photon states by up to 1.46% and occupied-owner cells by 41.4%; fixed-cell transport and the original Wendland dictionary therefore become increasingly strained. Cold-start fixed-K quality can be non-monotone, line searches can exhaust at $\sigma\geq2.5$, and recovered-headroom percentages become unstable or exceed 100% when the fixed-1024 denominator is small or inferior to another solution. Dynamic batch selection remains slower than its matched uniform control and does not establish computational efficiency against fixed-space optimization. v0.3f accumulates the 1,024-candidate Gram one sparse camera Jacobian at a time, but estimates it only at the cold K=256 state; v0.3g accumulates a compact 1,024×1,024 joint multiview Gram per birth round, but it likewise does not establish a global nonlinear inverse rank. The 26.2-million-photon path validates deterministic first-arrival streaming, not full differentiability through ownership changes. More detectors materially increase union capture, but the extra observations are partly redundant and do not by themselves guarantee useful geometry steps; five stronger-evidence configurations exhaust every line search. v0.4 is an orthographic point-sampled surface renderer, not a perspective, radiometric, or global-illumination model. Its Open3D triangle ray cast is an evaluation reference, not the production operator. Bilinear splats and hard depth ownership are differentiable only inside a fixed visibility cell; owner switches, silhouette crossings, and disocclusions remain non-smooth. v0.5 replaces the camera-specific scene path with a discrete orthographic light-slab atlas, but it is not a continuous plenoptic field: off-atlas quality degrades with angular distance, and arbitrary perspective/focal changes are not yet validated. Its constant radiance tests geometry occupancy rather than BRDF or interior photometric detail. v0.6 removes the mesh surface scaffold from rendering, but the committed Bunny experiment still uses the repaired Stanford mesh once to acquire the fixed scalar grid $F$ and later as an evaluation oracle. Cell-uniform sampling is not exactly surface-area uniform; the position color, exponential extinction, and cosine lobe are interpretable diagnostics rather than calibrated lighting or a BRDF. The relief depends substantially on the cosine term, so attenuation alone does not solve general inverse appearance. The scene-centric field remains a discrete orthographic light slab rather than a perspective plenoptic renderer. Raw v0.6 boundary events cost 579.64 MiB even though the field is 97.98 MiB; adaptive phase-space allocation is still needed. Forward visibility still inherits sign-change tracing and may miss tangent contacts. The repaired/grid-extracted Bunny field contains a few disconnected specks that are faithfully retained rather than post-processed away. At the smallest supports, nearly-null finite-difference columns are cancellation-limited. Detector misses contribute nothing, and deformations must retain the intended local normal-line root. With intentionally fixed emitters, the separate Full-HD benchmark images are extremely sparse; that benchmark tests operator scaling rather than image reconstruction quality.
