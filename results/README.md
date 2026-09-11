# Results release

Companion to the `dehaze-diagnostic-framework-code` repository, which contains
the code that produced everything here and can be used to regenerate it (see
that repo's README for the exact script → table/figure mapping).

This is trimmed from a larger private results archive: it excludes an
A-branch architecture sweep, a capacity scaling-curve sweep, and an
architecture-ablation sweep (T-Gate / SFT / RepConv / dilation variants) that
were part of the research process but are **not** part of the method reported
in the paper.

## Structure

```
main_results/
├── ground_truth_summary.{json,md}   Headline PSNR/SSIM for every checkpoint below,
│                                     against the fixed 35-indoor / 74-outdoor test
│                                     split — source of Tables ceiling / psnr_main /
│                                     ceiling_close
└── checkpoints/                     7 checkpoints (.pth + .onnx each), all V3-t capacity:
    ├── model_Sprint_Finalization_A-Baseline_V3-t.*        Baseline
    ├── model_Sprint_Finalization_B-FiLMTAff_V3-t.*        +FiLM +T-Affine
    ├── model_Sprint_Finalization_C-Depth_V3-t.*           Proposed (+FiLM+T-Affine+DAF), no GAN
    ├── model_Sprint_Finalization_D-Prop-S42-R1_V3-t.*     +GAN, seed 42
    ├── model_Sprint_Finalization_D-Prop-V3t-S43_V3-t.*    +GAN, seed 43
    ├── model_Sprint_Finalization_D-Prop-V3t-S44_V3-t.*    +GAN, seed 44
    └── model_C1-DepthOnly_V3-t.*                          +DAF only (no FiLM/T-Affine)

supplementary_experiments/          Outputs of must_experiments/exp2-exp7 (see code repo)
├── exp2_lpips_eval/                 Table lpips
├── exp3_thirdparty_separation/      Tables sweepA / realscatter (AOD-Net, LightDehazeNet)
├── exp4_ceiling_error_control/      Table scale_sensitivity (raw)
├── exp5_daf_only_ablation/          "+DAF only" row + FiLM/T-Affine/DAF interaction term
├── exp6_scene_bootstrap/            Scene-level bootstrap confidence intervals
└── exp7_tables/                     scale_sensitivity_table.tex, error_decomp_table.tex,
                                      t0_truncation_stats.json — ready-to-paste LaTeX +
                                      the check that the t0=0.1 floor is never active

figures/
├── qual_indoor.pdf                  Figure qual_indoor — qualitative comparison, indoor
└── qual_outdoor.pdf                 Figure qual_outdoor — qualitative comparison, outdoor
```

## Notes

- **Seed 42 has two independent training runs** in the source archive
  (`D-Prop-S42-R1`, `D-Prop-S42-R2`) as a determinism/reproducibility check.
  Only **R1** is included here, matching the run that the code's checkpoint
  registry (`must_experiments/dehaze_common.py`) and the paper's reported
  3-seed mean/std use. R2 exists if you'd like it added for the
  reproducibility comparison itself.
- Table `cost_summary`'s per-submodule MMAC breakdown is **not** reproduced
  here — see the "Known gap" note in the code repo's README.
- Table `psnr_lit` (literature comparison) is not a local result — those
  numbers are cited from the original publications, not computed here.
- Checkpoints are plain PyTorch `state_dict` (`.pth`) and exported ONNX
  graphs (`.onnx`); see `predict.py` / `unified_ground_truth_eval.py` in the
  code repo for how to load them.
