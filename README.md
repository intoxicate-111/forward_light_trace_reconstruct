# Zero-Set Forward Light Tracing

This experimental research prototype asks whether samples of a zero set can act as local directional emitters, whether a compact local parameterization induces a sparse geometry-to-image Jacobian, and whether observations can repeatedly turn evidence for nonexistent parameters into a better geometry function space. Version 0.1 isolates transport, v0.2–v0.2.2 validate local geometry derivatives and CUDA/multiview scaling, v0.3a–v0.3b test pre-birth utility prediction, v0.3c performs true sequential parameter birth, v0.3d transfers that protocol to Stanford Bunny geometry, v0.3e tests simultaneous birth and Bunny smoothing headroom, and v0.3f separates detector resolution, photon density, optimization, and inverse ambiguity. This project makes no claim of novelty.

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

## Limitations

The prototype deliberately omits differentiability across collision/visibility topology changes, merge/prune policies, optimized basis centers or radii, adaptive stopping, neural fields, secondary reflection, indirect illumination, refraction, participating media, physical camera lenses, and physically calibrated radiometry. It also misses tangent zero-set intersections that touch without a sign change. Birth is tested with a fixed candidate bank and fixed transport cells on two synthetic sphere regimes and one controlled Bunny construction, not diverse objects, unknown geometry, or real photographs. The Bunny base is a smoothed grid rather than a deliberately decimated mesh, grid discretization contributes to measured error, and the five repair caps are only a sign proxy. Under stronger smoothing, closest-point emitter transfer changes photon states by up to 1.46% and occupied-owner cells by 41.4%; fixed-cell transport and the original Wendland dictionary therefore become increasingly strained. Cold-start fixed-K quality can be non-monotone, line searches can exhaust at $\sigma\geq2.5$, and recovered-headroom percentages become unstable or exceed 100% when the fixed-1024 denominator is small or inferior to another solution. Dynamic batch selection remains slower than its matched uniform control and does not establish computational efficiency against fixed-space optimization. v0.3f accumulates the 1,024-candidate Gram one sparse camera Jacobian at a time, but estimates it only at the cold K=256 state; it does not establish a global nonlinear inverse rank. At the smallest supports, nearly-null finite-difference columns are cancellation-limited. Detector misses contribute nothing, and deformations must retain the intended local normal-line root. With intentionally fixed emitters, the separate Full-HD benchmark images are extremely sparse; that benchmark tests operator scaling rather than image reconstruction quality.
