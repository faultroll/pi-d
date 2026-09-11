# Ground-truth evaluation summary

Produced by `pgdehazenet/unified_ground_truth_eval.py`, evaluating each
checkpoint in `checkpoints/` against the fixed test split (35 indoor pairs,
74 outdoor pairs — `SOTS/split_txt/indoor_test.txt` / `outdoor_test.txt`).
This is the authoritative source for the numbers in Tables `ceiling`,
`psnr_main` and `ceiling_close`. Full per-image statistics (mean, std, 95% CI)
are in `ground_truth_summary.json`; this file gives the PSNR/SSIM means.

| Configuration | Indoor PSNR | Indoor SSIM | Outdoor PSNR | Outdoor SSIM |
|---|---|---|---|---|
| CAP baseline (direct CAP + ASM inversion, no network) | 18.174 ± 2.728 (n=35) | 0.804 ± 0.077 (n=35) | 18.462 ± 3.162 (n=74) | 0.780 ± 0.109 (n=74) |
| Ceiling, corrected (physical reconstruction ceiling) | 27.533 ± 4.229 (n=35) | 0.910 ± 0.039 (n=35) | 31.824 ± 3.209 (n=74) | 0.943 ± 0.021 (n=74) |
| A-Baseline (V3-t baseline) | 18.153 ± 4.361 (n=35) | 0.861 ± 0.090 (n=35) | 22.948 ± 4.476 (n=74) | 0.918 ± 0.062 (n=74) |
| B-FiLMTAff (+FiLM +T-Affine) | 18.376 ± 4.301 (n=35) | 0.863 ± 0.087 (n=35) | 22.932 ± 4.297 (n=74) | 0.918 ± 0.061 (n=74) |
| C-Depth = **Proposed** (+FiLM +T-Affine +DAF, no GAN) | 20.454 ± 3.543 (n=35) | 0.897 ± 0.054 (n=35) | 23.917 ± 2.798 (n=74) | 0.944 ± 0.033 (n=74) |
| D-Prop, seed 42 (+GAN) | 20.040 ± 3.448 (n=35) | 0.896 ± 0.056 (n=35) | 24.059 ± 2.922 (n=74) | 0.944 ± 0.038 (n=74) |
| D-Prop, seed 43 (+GAN) | 19.933 ± 3.314 (n=35) | 0.887 ± 0.058 (n=35) | 23.835 ± 2.893 (n=74) | 0.943 ± 0.037 (n=74) |
| D-Prop, seed 44 (+GAN) | 20.402 ± 3.377 (n=35) | 0.893 ± 0.055 (n=35) | 23.873 ± 3.107 (n=74) | 0.942 ± 0.038 (n=74) |

`checkpoints/model_C1-DepthOnly_V3-t.pth` ("+DAF only", no FiLM/T-Affine — the
remaining row of Table `ablation_modules`) is evaluated separately by
`must_experiments/exp5_daf_only_ablation.py`; its result is in
`../supplementary_experiments/exp5_daf_only_ablation/results.json`.

Note: "ceiling, corrected" is the physically-consistent reconstruction ceiling
(Section on ceiling analysis, C1). An uncorrected/naive CAP-blind-estimate
number is close to the CAP-baseline row above rather than to this row — this
file reports the corrected ceiling used in the paper's tables.
