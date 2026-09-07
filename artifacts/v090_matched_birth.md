# v0.9.0 matched CURRENT geometry-birth report

## 1. Exact local code state

Starting HEAD `bd434c1a4ff528155be35d1296d619b36ef6fb2a`, branch `main`; pre-existing dirty v0817 work preserved. No commit/push. Starting diff is embedded in the JSON.

## 2. Frozen v0.8.17 CURRENT operator

```json
{
  "operator": "v0817_CURRENT_fixed_measure_C3",
  "h": 0.02455721885871518,
  "epsilon": 0.02455721885871518,
  "path_step": 0.01227860942935759,
  "eta": 9.787458785644276e-07,
  "kappa": 4.605170185988091,
  "launch": 1.05,
  "micro": 1,
  "ambient": 0.35,
  "gate_width": 0.05,
  "gain": 1.5,
  "extent": 2.8,
  "capture": 4,
  "resolution": [
    1080,
    1920
  ],
  "views": 4,
  "count": 16777216,
  "mass": 1.0034377352493191,
  "seed": 101,
  "base_grid_digest": "c262097d30d246d7437b6ae3736508e68189236e524ef0748013219b3f7859ca",
  "source_dtype": "float64",
  "transport_dtype": "float64",
  "detector_dtype": "float32",
  "precision_note": "As v0817; float64 detector in bounded FD diagnostics only."
}
```

## 3. GT matched-soft target construction

```json
{
  "source": {
    "signature": {
      "config_digest": "62ca4976cb787bbdbc3fd85b82677cf3aaa45a383158d798e34cfdd3b30c3861",
      "field_digest": "b50e50707aba282bbe11d484df22a48f5b16e68cd8dc32404a9ab4ff0c1a4fb5",
      "count": 16777216,
      "seed": 101,
      "operator_version": "v0817_CURRENT_fixed_measure_C3",
      "shape": [
        16777216,
        3
      ],
      "dtype": "float64",
      "branch": "GT"
    },
    "definition": "Uniform area on the GT trilinear implicit surface by dominant-axis coarea rejection; fixed thereafter. No marching cubes or hard visibility.",
    "attempted_proposals": 126746624,
    "source_seconds": 11.294294404797256,
    "position_digest": "9a96d06e11b452c080c3bf546ca7b6445cf37dbcc980ace20b00aaf733a4287e",
    "normal_digest": "53b76ab2430228b4dce24ba6aa5b5ffd4c28e935eed1fe03f1af8edb87c27efd",
    "weight_digest": "e8a23943d73928dce091768b7789dedd1390b88a2ee2ef22b4e84ad52b924711",
    "source_mass": 1.0034377352493205
  },
  "render": {
    "signature": {
      "config_digest": "62ca4976cb787bbdbc3fd85b82677cf3aaa45a383158d798e34cfdd3b30c3861",
      "field_digest": "b50e50707aba282bbe11d484df22a48f5b16e68cd8dc32404a9ab4ff0c1a4fb5",
      "count": 16777216,
      "seed": 101,
      "operator_version": "v0817_CURRENT_fixed_measure_C3",
      "shape": [
        4,
        1080,
        1920,
        3
      ],
      "dtype": "float32",
      "source_digest": "9a96d06e11b452c080c3bf546ca7b6445cf37dbcc980ace20b00aaf733a4287e",
      "centers": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "radii": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "coefficients": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    "source_seconds": 0.03369776508770883,
    "transport_seconds": 84.35699388105422,
    "detector_seconds": 83.76946188602597,
    "total_seconds": 168.2056545021478,
    "peak_cuda_allocated_mib": 450.2841796875,
    "peak_cuda_reserved_mib": 518.0,
    "cpu_rss_peak_mib": 3455.7890625,
    "energy_accounting": [
      {
        "view": 0,
        "input_energy": 0.48208660721279656,
        "detected_energy": 0.48208588059424656,
        "relative_energy_difference": 1.5072365403312595e-06,
        "projected_count_sum": 16777151.126759283,
        "boundary_checked": true,
        "boundary_lost_energy": 0.0,
        "expected_in_window_energy": 0.48208660721279656,
        "numerical_energy_relative_error": 1.5072365403312595e-06
      },
      {
        "view": 1,
        "input_energy": 0.38033184206480264,
        "detected_energy": 0.38033134454034534,
        "relative_energy_difference": 1.308132536571429e-06,
        "projected_count_sum": 16777151.493571142,
        "boundary_checked": true,
        "boundary_lost_energy": 0.0,
        "expected_in_window_energy": 0.38033184206480264,
        "numerical_energy_relative_error": 1.308132536571429e-06
      },
      {
        "view": 2,
        "input_energy": 0.505118497125111,
        "detected_energy": 0.5051180408238117,
        "relative_energy_difference": 9.033549590796999e-07,
        "projected_count_sum": 16777167.391051287,
        "boundary_checked": true,
        "boundary_lost_energy": 0.0,
        "expected_in_window_energy": 0.505118497125111,
        "numerical_energy_relative_error": 9.033549590796999e-07
      },
      {
        "view": 3,
        "input_energy": 0.3495122536973156,
        "detected_energy": 0.3494421023770193,
        "relative_energy_difference": 0.00020071204815914746,
        "projected_count_sum": 16774517.776183574,
        "boundary_checked": true,
        "boundary_lost_energy": 6.942249225644554e-05,
        "expected_in_window_energy": 0.34944283120505915,
        "numerical_energy_relative_error": 2.0856860544281826e-06
      }
    ],
    "root_failures": 0,
    "attachment_residual_max": 0.0,
    "maximum_displacement": 0.0,
    "image_digest": "5701b51c2733f9b1e819667378c06f5aa6fbe697821dd4db02d5d97c0b6351bd",
    "state_persisted": true
  },
  "replay_relative_l2": 5.977006772750693e-09,
  "replay_seconds": 81.59366814699024,
  "digest": "5701b51c2733f9b1e819667378c06f5aa6fbe697821dd4db02d5d97c0b6351bd",
  "field_digest": "b50e50707aba282bbe11d484df22a48f5b16e68cd8dc32404a9ab4ff0c1a4fb5",
  "lambda_parameters": 0,
  "valid": true,
  "path": "runs/v090_matched_birth/gt_current_soft_16m.npy",
  "definition": "CURRENT finite-packet integration of the GT implicit field; independent globally uniform implicit-area source; fixed normalized source measure; no hard reference or mesh readout."
}
```

## 4. Candidate derivative validation

Both branch multi-epsilon high/medium/weak tests are in v090_birth_fd.json. They are PREFLIGHT_ONLY, not evidence for Full-HD gain.

## 5. SOFT baseline

```json
{
  "observation": {
    "whole_image_mse": 0.008013271912625788,
    "foreground_mse": 0.027952950938363703,
    "interior_gt8px_mse": 0.014197463190753474,
    "silhouette_0_8px_mse": 0.12824789980290943,
    "foreground_mean_brightness": 0.7651754702194687,
    "edge_sharpness": 0.0690252035439404,
    "reference_edge_sharpness": 0.09199675652077396,
    "edge_sharpness_ratio": 0.7503004035620939,
    "thin_feature_response_ratio": 0.468948560831648,
    "per_view": [
      {
        "gradient_magnitude_ratio": 0.7702219183689277,
        "gradient_cosine": 0.24198589548371172,
        "reference_aligned_gradient_energy": 2732.0360945138,
        "unmatched_gradient_energy": 874.7814738038414,
        "excess_hf_weak_reference": 0.0001811335037409111,
        "gradient_squared_error": 0.0017892323310189258,
        "weak_reference_pixels": 1624532
      },
      {
        "gradient_magnitude_ratio": 0.829510518146698,
        "gradient_cosine": 0.2684893078384506,
        "reference_aligned_gradient_energy": 3098.8635536881475,
        "unmatched_gradient_energy": 897.8345433509384,
        "excess_hf_weak_reference": 0.0001399857055484171,
        "gradient_squared_error": 0.001740420903701317,
        "weak_reference_pixels": 1644948
      },
      {
        "gradient_magnitude_ratio": 0.8248741841836805,
        "gradient_cosine": 0.31501964270709953,
        "reference_aligned_gradient_energy": 3278.7667100099175,
        "unmatched_gradient_energy": 915.1588501012086,
        "excess_hf_weak_reference": 0.00022824048146296617,
        "gradient_squared_error": 0.0017251054252541127,
        "weak_reference_pixels": 1561449
      },
      {
        "gradient_magnitude_ratio": 0.8269468373027188,
        "gradient_cosine": 0.3307480765973032,
        "reference_aligned_gradient_energy": 2432.50225606894,
        "unmatched_gradient_energy": 613.0620094687912,
        "excess_hf_weak_reference": 0.00011378922617483615,
        "gradient_squared_error": 0.0012208123106778703,
        "weak_reference_pixels": 1668420
      }
    ],
    "gradient_magnitude_ratio": 0.8128883645005063,
    "gradient_cosine": 0.2890607306566413,
    "reference_aligned_gradient_energy": 2885.542153570201,
    "unmatched_gradient_energy": 825.2092191811948,
    "excess_hf_weak_reference": 0.00016578722923178262,
    "gradient_squared_error": 0.0016188927426630565,
    "weak_reference_pixels": 1624837.25,
    "loss": 0.004006636830592359,
    "detected_energy": 5457068.379421935,
    "relative_rgb_l2": 0.208076231457038,
    "per_view_mse": [
      0.008768418479027075,
      0.00928122455570929,
      0.008711792554827829,
      0.005291659055174703
    ],
    "foreground_mask_iou": [
      0.9835115340931344,
      0.9809553420819396,
      0.9836267978871747,
      0.9848696882147586
    ],
    "squared_image_energy": 4830823.6875
  },
  "geometry": {
    "symmetric_chamfer": 0.006909399688703678,
    "point_to_surface_mean": 0.0020368813191692908,
    "point_to_surface_p95": 0.005573502059264804,
    "surface_rms": 0.002844314895958835,
    "normal_consistency": 0.9912441476274787,
    "normal_error": 0.0087558523725213,
    "regional_ear_p2s_mean": 0.003096727958632453,
    "regional_head_p2s_mean": 0.0018580288780309575,
    "regional_leg_p2s_mean": 0.0023943048761365616,
    "regional_torso_p2s_mean": 0.0015589800026536574,
    "bounds": [
      [
        -0.997162570593492,
        -0.9857655507213664,
        -0.7570312212098319
      ],
      [
        0.9974998618071933,
        0.9632264335200473,
        0.7709279186320754
      ]
    ],
    "surface_area": 9.22009440314828,
    "components": 1,
    "maximum_outward_from_unit_sphere": 0.3220391935736078,
    "maximum_inward_from_unit_sphere": 0.8261501198869529,
    "radial_deviation_p95": 0.6135005649973445,
    "gt_extends_outside_unit_sphere": true,
    "gt_max_radius": 1.5736361939753005,
    "source_coverage_distance_p95": 0.011538445447551766,
    "source_coverage_distance_max": 0.020772942699835113,
    "source_coverage_samples": 65536,
    "geometry_seconds": 0.30888573080301285,
    "geometry_path": "runs/v090_matched_birth/geometry/soft_initial.ply",
    "geometry_target": "MC approximation of prepared.gt_field, evaluation only; existing P2S/Chamfer definitions. Current normals updated at every evaluation."
  }
}
```

## 6. SOFT candidate predicted-vs-actual validation

```json
{
  "groups": {
    "top": [
      1,
      0
    ],
    "middle": [
      15,
      3
    ],
    "low": [
      26,
      31
    ],
    "random": [
      23,
      6
    ]
  },
  "pearson": 0.9976648839552379,
  "spearman": 0.9523809523809524,
  "top_mean_gain": 4.2473148624912194e-05,
  "random_mean_gain": 4.794427968040586e-06,
  "top_k_enrichment": 8.858856344914566,
  "top_candidates_outperform_random": true,
  "rank_directionally_positive": true,
  "responsive_fraction": 1.0,
  "null_response_candidates": [],
  "false_positive_candidates": [],
  "statement": "Primary Full-HD actual gains, same initial zero-coefficient state and fixed source for each candidate; reduced-source smoke gains excluded.",
  "pearson_pvalue": 3.1776386224205006e-08,
  "spearman_pvalue": 0.00026040002438725105,
  "best_predicted_actual_gain": 4.088072167148134e-05,
  "validation_sample_count": 8,
  "inference_caveat": "Small deterministic stratified sample, overlapping random/top strata; directional correlation is not population-level proof."
}
```

## 7. SOFT sequential birth

```json
{
  "status": "STOP_REPEATED_NO_GAIN",
  "accepted_births": 8,
  "born_dofs": 11,
  "relative_observation_improvement": 0.07922894965930707,
  "relative_chamfer_improvement": 0.006244269007700253,
  "initial_mse": 0.008013273661184718,
  "final_mse": 0.007378390405676462,
  "initial_chamfer": 0.006909399688703678,
  "final_chamfer": 0.006866255538365692
}
```

## 8. GLOBE baseline

```json
{
  "observation": {
    "whole_image_mse": 0.1266388240861304,
    "foreground_mse": 0.1490000623797363,
    "interior_gt8px_mse": 0.13060713739237342,
    "silhouette_0_8px_mse": 0.36081435512930454,
    "foreground_mean_brightness": 0.5133278276511861,
    "edge_sharpness": 0.0060593275480903975,
    "reference_edge_sharpness": 0.09199675652077396,
    "edge_sharpness_ratio": 0.06586457802696695,
    "thin_feature_response_ratio": 0.01775395482249923,
    "per_view": [
      {
        "gradient_magnitude_ratio": 0.27771923913449764,
        "gradient_cosine": 0.0032244405252516983,
        "reference_aligned_gradient_energy": 97.84338428250648,
        "unmatched_gradient_energy": 371.0826698384901,
        "excess_hf_weak_reference": 0.00020451883803243202,
        "gradient_squared_error": 0.001576456669229255,
        "weak_reference_pixels": 1624532
      },
      {
        "gradient_magnitude_ratio": 0.27610638317128255,
        "gradient_cosine": 0.0013730623894536,
        "reference_aligned_gradient_energy": 59.43076129900056,
        "unmatched_gradient_energy": 383.3721252237638,
        "excess_hf_weak_reference": 0.0002098932016987717,
        "gradient_squared_error": 0.0015062723344219753,
        "weak_reference_pixels": 1644948
      },
      {
        "gradient_magnitude_ratio": 0.27837822723153943,
        "gradient_cosine": 0.006350446525072705,
        "reference_aligned_gradient_energy": 91.4878976712146,
        "unmatched_gradient_energy": 386.16874711546234,
        "excess_hf_weak_reference": 0.00022462763694054046,
        "gradient_squared_error": 0.0015961655351954677,
        "weak_reference_pixels": 1561449
      },
      {
        "gradient_magnitude_ratio": 0.31579502231302964,
        "gradient_cosine": 0.006299615079665478,
        "reference_aligned_gradient_energy": 39.35308310239652,
        "unmatched_gradient_energy": 404.79024999279375,
        "excess_hf_weak_reference": 0.00023638244294233623,
        "gradient_squared_error": 0.001176706584376167,
        "weak_reference_pixels": 1668420
      }
    ],
    "gradient_magnitude_ratio": 0.28699971796258733,
    "gradient_cosine": 0.00431189112986087,
    "reference_aligned_gradient_energy": 72.02878158877954,
    "unmatched_gradient_energy": 386.3534480426275,
    "excess_hf_weak_reference": 0.0002188555299035201,
    "gradient_squared_error": 0.0014639002808057163,
    "weak_reference_pixels": 1624837.25,
    "loss": 0.06331939929172949,
    "detected_energy": 6325537.149859566,
    "relative_rgb_l2": 0.8271816225843002,
    "per_view_mse": [
      0.14884521571964193,
      0.1190652786725437,
      0.13339081801885394,
      0.10525388192279776
    ],
    "foreground_mask_iou": [
      0.5580524672860128,
      0.5431913956016945,
      0.6111587661103347,
      0.5997283420080444
    ],
    "squared_image_energy": 4444647.75
  },
  "geometry": {
    "symmetric_chamfer": 0.23683303443624867,
    "point_to_surface_mean": 0.2644168936879076,
    "point_to_surface_p95": 0.5912111923918419,
    "surface_rms": 0.3259051084799381,
    "normal_consistency": 0.703028366454804,
    "normal_error": 0.29697163354519596,
    "regional_ear_p2s_mean": 0.3321692531007422,
    "regional_head_p2s_mean": 0.19373760558580227,
    "regional_leg_p2s_mean": 0.0786438656954298,
    "regional_torso_p2s_mean": 0.2978633083876529,
    "bounds": [
      [
        -0.9999213884461601,
        -0.9999213884461601,
        -0.9999213884461601
      ],
      [
        0.9999214604215803,
        0.9999214604215803,
        0.9999214604215803
      ]
    ],
    "surface_area": 12.565174747220254,
    "components": 1,
    "maximum_outward_from_unit_sphere": 0.0,
    "maximum_inward_from_unit_sphere": 2.8567284971803275e-05,
    "radial_deviation_p95": 2.841673160902669e-05,
    "gt_extends_outside_unit_sphere": true,
    "gt_max_radius": 1.5736361939753005,
    "source_coverage_distance_p95": 0.01352829586010511,
    "source_coverage_distance_max": 0.02446096315026494,
    "source_coverage_samples": 65536,
    "geometry_seconds": 0.35154168703593314,
    "geometry_path": "runs/v090_matched_birth/geometry/globe_initial.ply",
    "geometry_target": "MC approximation of prepared.gt_field, evaluation only; existing P2S/Chamfer definitions. Current normals updated at every evaluation."
  }
}
```

## 9. GLOBE candidate predicted-vs-actual validation

```json
{
  "groups": {
    "top": [
      8,
      24
    ],
    "middle": [
      3,
      5
    ],
    "low": [
      15,
      13
    ],
    "random": [
      23,
      6
    ]
  },
  "pearson": 0.991779844016511,
  "spearman": 1.0,
  "top_mean_gain": 0.00010284672771652265,
  "random_mean_gain": 1.0671837753106761e-05,
  "top_k_enrichment": 9.637208707242774,
  "top_candidates_outperform_random": true,
  "rank_directionally_positive": true,
  "responsive_fraction": 1.0,
  "null_response_candidates": [],
  "false_positive_candidates": [],
  "statement": "Primary Full-HD actual gains, same initial zero-coefficient state and fixed source for each candidate; reduced-source smoke gains excluded.",
  "pearson_pvalue": 1.3800628015606155e-06,
  "spearman_pvalue": 0.0,
  "best_predicted_actual_gain": 0.00010410069733443605,
  "validation_sample_count": 8,
  "inference_caveat": "Small deterministic stratified sample, overlapping random/top strata; directional correlation is not population-level proof."
}
```

## 10. GLOBE sequential dynamic birth

```json
{
  "status": "STOP_REPEATED_NO_GAIN",
  "accepted_births": 16,
  "born_dofs": 25,
  "relative_observation_improvement": 0.022895259116711215,
  "relative_chamfer_improvement": -0.0013125557429856554,
  "initial_mse": 0.12663879858345897,
  "final_mse": 0.12373937047566168,
  "initial_chamfer": 0.23683303443624867,
  "final_chamfer": 0.2371438909957267
}
```

## 11. Geometry evaluation

Identical GT-field MC evaluation target; current-surface area samples, Chamfer, P2S, RMS, normal consistency, bounds and components recorded for each tested geometry. No MC rendering residual.

## 12. Source-measure / attachment behavior

GT uniform implicit-area, SOFT historical polygon-reference area pushed onto the base, and GLOBE analytic sphere area share normalized fixed-reference-measure semantics, but are not identical reference measures. No independent-layout control was run. Fixed weights and IDs; no source refresh. {"shell_escape_threshold": 0.1, "maximum_absolute_radial_deviation": 0.03277370145948111, "coefficient_limit_hit_rounds": 25, "note": "Saturation is reported, but causal bottleneck verdict requires a separate relaxed-bound control; not run."}

## 13. Runtime and memory

See v090_birth_performance.csv; includes source, transport, detector, candidate scoring, optimization and peak GPU/CPU memory. {"branches": {"SOFT": "STOP_REPEATED_NO_GAIN", "GLOBE": "STOP_REPEATED_NO_GAIN"}, "total_seconds": 58032.742921335855}

## 14. Failure cases

```json
{
  "SOFT": {
    "status": "STOP_REPEATED_NO_GAIN",
    "error": null,
    "false_positive_candidates": []
  },
  "GLOBE": {
    "status": "STOP_REPEATED_NO_GAIN",
    "error": null,
    "false_positive_candidates": []
  }
}
```

## 15. Exact verdicts

```json
{
  "MATCHED_CURRENT_SOFT_GT_TARGET_VALID": true,
  "TARGET_AND_RECONSTRUCTION_USE_SAME_FORWARD_OPERATOR": true,
  "BIRTH_SIGNAL_IS_NOT_DOMINATED_BY_SOURCE_LAYOUT": "UNRESOLVED",
  "SOFT_CURRENT_FORWARD_CANDIDATE_DERIVATIVE_PASSES_FD": true,
  "SOFT_NONEXISTENT_CANDIDATE_RESPONSE_IS_MEASURABLE": true,
  "SOFT_PREDICTED_GAIN_CORRELATES_WITH_ACTUAL_GAIN": true,
  "SOFT_TOP_CANDIDATES_OUTPERFORM_RANDOM": true,
  "SOFT_SEQUENTIAL_BIRTH_REDUCES_OBSERVATION_ERROR": true,
  "SOFT_SEQUENTIAL_BIRTH_REDUCES_GEOMETRY_ERROR": true,
  "SOFT_DYNAMIC_BIRTH_IS_SUCCESSFUL": true,
  "GLOBE_CURRENT_FORWARD_CANDIDATE_DERIVATIVE_PASSES_FD": true,
  "GLOBE_NONEXISTENT_CANDIDATE_RESPONSE_IS_MEASURABLE": true,
  "GLOBE_PREDICTED_GAIN_CORRELATES_WITH_ACTUAL_GAIN": true,
  "GLOBE_TOP_CANDIDATES_OUTPERFORM_RANDOM": true,
  "GLOBE_SEQUENTIAL_BIRTH_REDUCES_OBSERVATION_ERROR": true,
  "GLOBE_SEQUENTIAL_BIRTH_REDUCES_GEOMETRY_ERROR": false,
  "GLOBE_DYNAMIC_BIRTH_IS_SUCCESSFUL": false,
  "GLOBE_BIRTH_ESCAPES_INITIAL_SPHERE": false,
  "GLOBE_COEFFICIENT_LIMIT_IS_ACTIVE_BOTTLENECK": "UNRESOLVED",
  "GLOBE_FIXED_REFERENCE_SOURCE_REMAINS_USABLE": true,
  "CURRENT_SOFT_OPERATOR_SUPPORTS_GEOMETRY_BIRTH": true,
  "GLOBAL_FIXED_MEASURE_IS_COMPATIBLE_WITH_GEOMETRY_BIRTH": true,
  "OBSERVATION_DRIVEN_FUNCTION_SPACE_GROWTH_IS_SUPPORTED": true,
  "STRONGER_GLOBE_TO_BUNNY_GROWTH_IS_SUPPORTED": false
}
```

## 16. Scientific interpretation

Q1: candidate predictiveness is assessed independently by each branch Spearman/Pearson and full-budget single-birth gains. Q2/Q3: see separate observation and Chamfer trajectories; lower image loss alone is not geometry recovery. Q4: mesh-lighting/readout mismatch has been removed, but different reference measures, finite quadrature and source-layout sensitivity remain alternatives to geometry/observability. Q5: source count and weights are structurally decoupled from births; coverage/attachment usability is only established over the deformation actually reached.

## 17. Limitations

All primary images and candidate columns use 16,777,216 emitters, four 1920x1080 views. FD/smoke and timing subsets are PREFLIGHT_ONLY.

GT uniform implicit-area, SOFT historical polygon-reference area pushed onto the base, and GLOBE analytic sphere area share normalized fixed-reference-measure semantics, but are not identical reference measures. No independent-layout control was run.

v0816 established an MSE plateau, not strict quadrature convergence. Optional 32M GT convergence has not been run.

Exact accumulated image columns include cross-emitter terms. The derivative is local to fixed quadrature/topology; finite steps can cross cells and invalidate quadratic gain predictions.

Small deterministic two-level candidate bank, one damped Gauss-Newton step per birth, coefficient bound .03, and sparse single-candidate validation limit generality.

Geometry metrics use a 160^3 marching-cubes evaluation of the GT scalar field, never the observation target. Source-coverage distances use a fixed 65,536-ID diagnostic subset.

Coefficient saturation alone is not proof that relaxing the bound would improve geometry. Small nonzero sphere deformation is not recovery of Bunny-scale structure.

