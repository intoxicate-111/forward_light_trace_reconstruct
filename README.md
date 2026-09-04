# Zero-Set Forward Light Tracing

This experimental research prototype asks whether samples of a zero set can act as local directional emitters, whether a compact local parameterization induces a sparse geometry-to-image Jacobian, and whether observations can repeatedly turn evidence for nonexistent parameters into a better geometry function space. Version 0.1 isolates transport, v0.2–v0.2.2 validate local geometry derivatives and CUDA/multiview scaling, v0.3a–v0.3b test pre-birth utility prediction, and v0.3c performs true sequential parameter birth. This project makes no claim of novelty.

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

## Limitations

The prototype deliberately omits differentiability across collision/visibility topology changes, merge/prune policies, optimized basis centers or radii, adaptive stopping, neural fields, secondary reflection, indirect illumination, refraction, participating media, physical camera lenses, and physically calibrated radiometry. It also misses tangent zero-set intersections that touch without a sign change. Birth is tested only with a fixed candidate bank and fixed transport cells on two synthetic sphere deformation regimes and two target seeds, not diverse objects or unknown real geometry. The 256-DoF endpoint does not establish full-model quality on distributed detail, and repeated scoring is far slower than fixed-space optimization. Active-set conditioning was not explicitly estimated because the implementation deliberately avoids materializing a dense Gram matrix. At the smallest supports, nearly-null finite-difference columns are cancellation-limited and fixed-RMS target construction can become ill-conditioned before GPU memory is exhausted. Detector misses produce no contribution, deformations must retain the intended local normal-line root, and only sphere- and torus-based fields are supported. With the intentionally fixed 256 emitters, the separate Full-HD benchmark images are extremely sparse; that benchmark tests operator scaling rather than image reconstruction quality.
