# v0.9.7 — high-bandwidth topology reachability

## CUDA and bandwidth configuration

The hard local-CUDA gate passed on Quadro RTX 5000 with PyTorch 2.5.1+cu124 / CUDA 12.4; no HPC or CPU fallback was used. The immutable v0.9.6 target digest is `e137b58e55d8184b67feecf537575435a6a5ce58b0541d667dc08311592e9d4c` and its one-component watertight genus-1 topology was reproduced at 96^3 and 144^3. Tests cover 512^2 and 1024^2 at 16K/64K sources plus 1920x1080 at 64K. A complete 768-parameter VJP at 64K was the highest practical diagnostic on the 16-GiB device; 262K was not run. Scene/transport was frozen across detector replays. The corrected Phase-A pass took 643.0 s and peaked at 1303.2 MiB allocated CUDA memory.

The implementation reuses compact-support pairs and chunk-local coalesced COO matrices with `torch.sparse.mm`. It never forms a dense source-by-parameter or pixel-by-parameter Jacobian. The small historical-dense comparison has max error 1.78e-14, relative L2 error 2.3e-16, and 154,530 sparse entries versus 2,359,296 dense entries.

## Oracle topology-direction observability

| configuration | MSE | `||Jv_T||/sqrt(Nobs)` | randomized `range(J^T)` projection lower bound | cosine | oracle GN alpha |
|---|---:|---:|---:|---:|---:|
| 512x512_16K | 1.1526 | 3.1325 | 0.0578731 | -0.0039509 | -0.000202402 |
| 1024x1024_16K | 4.49432 | 12.9304 | 0.0517749 | -0.000979027 | -1.19341e-05 |
| 512x512_64K | 0.387069 | 1.40151 | 0.0539716 | -0.00162783 | -0.000176947 |
| 1024x1024_64K | 1.1276 | 6.26298 | 0.0522508 | -0.00134388 | -1.67628e-05 |
| 1920x1080_64K | 2.20773 | 14.2109 | 0.0535294 | -0.00903887 | -6.80908e-05 |

The topology direction produces a nonzero image response at every bandwidth, but the 32-vector output-space sketch captures only 0.0518--0.0579 of its norm in the sampled leading range of `J^T`, with no systematic bandwidth gain. This is a lower bound, not a full-rank projection claim.

## Gradient alignment

All five `cos(-J^T r,v_T)` values are negative and near zero: -0.00903887 to -0.000979027. All oracle one-dimensional GN steps are also negative. High bandwidth therefore observes `v_T` but does not locally prefer it. The Phase-A classification is **`TOPOLOGY_SIGNAL_REMAINS_MISALIGNED`**.

## Source-density effect

At 512^2, increasing sources 16K to 64K lowers MSE 1.153 to 0.3871; at 1024^2 it lowers MSE 4.494 to 1.128. This is a strong sampling/noise effect, but alignment remains near-zero/negative and randomized oracle projection remains near 0.05. Source density does not recover the topology preference.

## Detector-resolution effect

Raw and per-observation sensitivities rise with detector scaling because the unchanged renderer multiplies splats by pixel count. The scale-invariant alignment does not improve: 16K changes from -0.003951 at 512^2 to -0.000979 at 1024^2; 64K reaches -0.009039 at Full-HD, still with a negative GN step. The failure is neither detector-limited nor rescued by Full-HD.

## Jacobian spectrum

The reported singular spectra are matrix-free `JQ` spectra on `span{v_T, 12 seeded orthonormal parameter probes}`; all 13 sampled modes are above the relative 1e-6 threshold. Their useful condition ratios range from 10.7 to 11.3. Separately, 32 stateless output-space probes form `orth(J^T Omega)` for the meaningful observable-range projection. The earlier within-`Q` projection of 1.0 is explicitly retained only as a non-evidential legacy diagnostic because `Q` deliberately contains `v_T`.

## High-bandwidth RGB optimization

The selected 1024^2/16K configuration had the smallest-magnitude cosine, strong normalized sensitivity, and practical repeated-retrace cost; no tested configuration had positive oracle preference. Starting from the exact sphere and using only immutable target RGB, 20 accepted steps reduce MSE 4.49432 to 4.09937 (8.79%). It then exhausts 32 fresh-retrace candidates. Runtime was 9898.4 s. The optimizer never receives `theta_T`, `v_T`, target field/mesh, genus, or hole location.

## Topology trajectory

Every accepted primary state is genus 0. The final 72^3/96^3/144^3 extractions are one-component watertight Euler-2 surfaces; all 2,048 final normal fibers have one root. A fixed 2,048-direction offline audit at accepted iterations 0/5/10/15/20 likewise finds exactly 2,048 one-root and zero three-root fibers at every checkpoint. The final minimum sampled spatial-gradient norm is 0.988239, minimum chart denominator 0.794873, and minimum refined fiber slope 0.801846. No true zero-set critical precursor appears in the RGB trajectory.

## Oracle path tomography

Fresh renders of `theta(t)=t theta_T` show a non-monotone barrier: MSE starts at 4.49432, rises to a sampled maximum 5.17039, and only collapses to numerical zero at `t=1`. Topology changes inside `(0.725,0.740]`; the fixed 2,048-direction audit first sees 2 three-root fibers at `t=0.6`, then 27 at `t=0.725`, 34 at `t=0.74`, and reproduces the v0.9.6 target count of 89 at `t=1`. Alignment with the remaining oracle direction stays between -0.009343 and 0.01928. The spatial gradient and frozen-chart denominator are reported separately: the sampled surface gradient does not approach zero at sampled `t`, while the chart denominator becomes small. Thus tomography brackets but does not resolve an exact critical parameter, and no chart failure is mislabeled as topology criticality.

The 2-D fresh-rendered slice confirms a narrow/folded genus-1 region near oracle fractions 0.725--0.8. Moving farther along the initial negative-gradient axis lowers MSE but can return to genus 0; no broad low-loss route from the sphere is visible in this slice.

## Basin of attraction

| initialization | accepted RGB steps | initial MSE | final MSE | stop | genus 1 |
|---:|---:|---:|---:|---|---|
| 0.75 | 0 | 5.12643 | 5.12643 | TOPOLOGY_SUCCESS | yes (initial) |
| 0.50 | 10 | 5.01849 | 4.8071 | ACCEPTED_STEP_BUDGET | no |
| 0.25 | 9 | 4.71209 | 4.56322 | STABLE_NO_PROGRESS_AFTER_32_FRESH_RETRACES | no |

Only 0.75 `theta_T` is genus 1, and it already lies beyond the bracketed topology event before optimization, so its zero-step success is not an RGB-driven crossing. The 0.50 and 0.25 starts improve RGB loss but remain genus 0 at all final extraction grids with 2,048/2,048 single-root fibers. Approximate capture requires initialization already on the target-topology side; intermediate genus-0 initialization does not recover it.

## Need for complex continuation

Complex continuation was not activated. The actual RGB-only trajectory never approaches a genuine zero-set critical region or requests the topology direction; its minimum spatial gradient remains regular and the oracle GN preference at initialization is negative. The oracle path proves a real finite-parameter genus change exists, but that diagnostic cannot authorize a complex bridge in the optimizer. The present obstruction is loss alignment/basin geometry, not an observed singular real tracker.

## Final verdict

**`HIGH_BANDWIDTH_STILL_MISALIGNED`**. The evidence does **not** support the statement that v0.9.6 topology reachability failure was primarily a measurement-bandwidth limitation. Full-HD/64K measurements show nonzero topology-direction sensitivity, but gradient/oracle alignment remains near zero and negative, every local oracle GN step points away, the sphere-start high-bandwidth RGB optimizer remains genus 0, and intermediate genus-0 basin starts also fail. High bandwidth reduces source-sampling noise and changes raw sensitivity; it does not recover the missing optimization preference.
