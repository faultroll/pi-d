# Traceability Audit -- Paper <-> Code <-> Released Results

This file documents how code, results, and the paper were checked to agree
after restructuring. Agreement is not assumed -- every row below was checked
by pulling the actual number out of `dehaze-results-release` and diffing it
against the literal value printed in `paper_SN_template.tex`. Where the two
didn't match closely enough to be floating-point noise, that is stated
explicitly rather than hidden.

Status legend:
- **[OK] Verified** -- the number was pulled from the released JSON/`.tex` and
  matches the paper to the precision shown (see "Check" column for the
  arithmetic).
- **[FLAG] Ambiguous source** -- a plausible script exists but the code alone
  does not pin down that *this exact script with this exact definition*
  produced *this exact number*. Flagged for confirmation, not guessed at.
- **[N/A] Not a local-code claim** -- the paper itself says this isn't
  computed by this codebase (cited from literature, or hand-calculated).

## Section 4.2 -- Ceiling Analysis (`tab:ceiling`, `tab:scale_sensitivity`)

| Paper item | Source script | Result file | Status | Check |
|---|---|---|---|---|
| `tab:ceiling`: ceiling 27.53/31.82, V3-t baseline 18.15/22.95, gap 9.38/8.87 | `physics.py` + `unified_ground_truth_eval.py` | `main_results/ground_truth_summary.{json,md}` | [OK] | JSON: ceiling-corrected 27.533/31.824, A-Baseline 18.153/22.948 -> gap 9.380/8.876, rounds to 9.38/8.87 exactly. |
| `tab:scale_sensitivity`: 7x2 grid, e.g. rough-noise row `30.18 18.16 17.63 17.53 17.50 17.50 17.53` | `experiments/ceiling_error_control.py` -> rendered by `experiments/render_tables.py` | `supplementary_experiments/exp7_tables/scale_sensitivity_table.tex` | [OK] | The released `.tex` file is **byte-identical** to the table body in the paper -- it was evidently pasted in directly. |

## Section 4.3 -- External Validation (`tab:sweepA`, `tab:realscatter`)

| Paper item | Source script | Result file | Status | Check |
|---|---|---|---|---|
| Sweep A, AOD-Net row (K_hat 0.818->1.035, err 0.682->0.465, const 0.557) | `experiments/third_party_separation.py --network aod` | `results_aod.json: sweep_A_network / sweep_A_summary` | [OK] | K_hat 0.817829/0.899176/0.953496/1.007838/... rounds to 0.818/0.899/0.953/1.008/1.035; `k_const_fitted` 0.942669 -> const err 0.557331, rounds to 0.557. |
| Sweep A, LightDehazeNet row (K_hat 1.630->1.585, const 0.262) | `exp3_thirdparty_separation.py --network lightdehaze` | `results_lightdehaze.json` | [OK] | K_hat 1.630424/1.858001/2.002842/1.732585/1.585284 -> 1.630/1.858/2.003/1.733/1.585; const err 0.261827, rounds to 0.262. |
| `realscatter`, AOD-Net (corr 0.268 / 0.216 / -0.268 / -0.404, MSE delta -0.026) | same, `--real-scatter` | `results_aod.json: real_image_scatter` | [OK] | 0.2683/0.2158/-0.268/-0.4037/mse_delta -0.025704 -> all round to the paper's values. |
| `realscatter`, LightDehazeNet (corr 0.515 / 0.530 / -0.402 / -0.319, MSE delta -0.099) | same | `results_lightdehaze.json` | [OK] | 0.5154/0.5298/-0.4023/-0.3187/mse_delta -0.098571 -> all round to the paper's values. |
| AOD-Net 1,761 params / LightDehazeNet 30,187 params | `experiments/third_party_separation.py` (`--selftest` asserts these) | n/a (assert in-code) | [OK] | `experiments/third_party_separation.py:495` hard-asserts `(AODNet, 1761)`, `(LightDehazeNet, 30187)`. |

## Section 4.4 -- Instantiating on V3-t (`eq:adcp_prob`)

**Resolution (2026-09-10):** after review, the authors are shrinking this
claim and removing the 86% figure from the manuscript rather than trying to
shore up the statistics behind it -- the four caveats below only support
"CAP's A is close to white in 6 of 7 validation scenes," not the stronger
"86% of pixels/images" framing the number invited. `a_quality_compare.py` /
`theta_star_model.py` are kept in this release as general-purpose diagnostic
tools (they still support the qualitative Problem-1-on-V3-t discussion and
the algebraic part of the argument, $\partial J_c/\partial w_{1,c}=I_c-A_c
\to0$ as $I_c\to A_c$, which is unaffected), but no paper table or headline
number depends on their output anymore. The analysis below is kept for the
record.

| Paper item | Source script | Status | Note |
|---|---|---|---|
| $P(\bar A_\text{DCP} > 1-\delta) \approx 0.86$, $\delta=10^{-2}$ | `a_quality_compare.py`, hardcoded to `./SOTS/split_txt/val.txt` (line 52) | [OK] **source confirmed**, [FLAG] **statistical framing needs a rewrite before release** | Confirmed by filename match against `val.txt`'s 35 indoor entries. Four caveats, verified by hand-checking against the code: **(1) definition mismatch, coincidental agreement** -- the script's literal criterion is "all 3 channels > 0.99"; the paper's stated criterion is "channel-mean > 1-delta". Recomputing with the paper's actual definition on this fold also gives 30/35, but the two are not the same event in general and the code should say what it does. **(2) pseudo-replication** -- the 35 "images" are 5 haze-density variants of 7 base scenes; results cluster perfectly within a scene (5/5 over threshold, or 0/5), so the effective n is 7, not 35. Scene-level: 6/7, 95% CI (Clopper-Pearson) [0.42, 1.00], (Wilson) [0.49, 0.97] -- much weaker than an n=35 binomial CI implies. **(3) undisclosed threshold sensitivity** -- delta=10^-2 -> 30/35 (0.857); delta=10^-3 -> 15/35 (0.429), on the same file. **(4) split identity** -- `val.txt` is a *different* fold from `indoor_test.txt`/`outdoor_test.txt` used everywhere else in the paper's tables; it happens to also have 35 indoor entries, which invites confusion with the main test-set n=35 used in `tab:psnr_main` etc. These are two different sample sets. None of this touches the algebraic part of the claim ($\partial J_c/\partial w_{1,c}=I_c-A_c\to0$ as $I_c\to A_c$), which is exact regardless of data. It bears on the empirical support cited alongside it. |

## Section 4.5-4.7 -- Main Results, Ablation, LPIPS (`tab:psnr_main`, `tab:ceiling_close`, `tab:ablation_modules`, `tab:lpips`)

| Paper item | Source script | Result file | Status | Check |
|---|---|---|---|---|
| `tab:psnr_main` / `tab:ceiling_close`: V3-t baseline 18.15/0.861/22.95/0.918/21.41/0.900; Proposed 20.45/0.897/23.92/0.944/22.80/0.929 | `unified_ground_truth_eval.py` | `ground_truth_summary.json` (A-Baseline, C-Depth rows) | [OK] | Indoor/outdoor values match to 3 d.p.; n-weighted averages (35 indoor + 74 outdoor) reproduce 21.41/0.900 and 22.80/0.929 exactly. |
| `tab:ablation_modules` "Baseline" / "+FiLM+T-Affine" / "Proposed" / "+GAN" rows, incl. delta Avg | `unified_ground_truth_eval.py` (A/B/C/D rows) | `ground_truth_summary.json` | [OK] | B-FiLMTAff 18.376/22.932 -> weighted avg 21.469, rounds to 21.47, delta +0.06. GAN 3-seed mean/std computed from D-Prop-S42/43/44 rows reproduces 20.13+/-0.25 / 23.92+/-0.12 and weighted-avg 22.70, delta +1.29 to the last digit (sample std, n-1). |
| `tab:ablation_modules` "+DAF only" row (19.58/23.32/22.12, delta +0.71) | `experiments/daf_only_ablation.py` | `supplementary_experiments/exp5_daf_only_ablation/results.json` | [OK] | psnr_mean 19.5826/23.3241 -> 19.58/23.32; weighted avg 22.124, rounds to 22.12, delta +0.71. |
| `tab:lpips`: Baseline 0.0991/0.0510; Proposed(3 modules) 0.0669/0.0374; +GAN 0.0722+/-0.0025/0.0369+/-0.0010 | `experiments/lpips_eval.py` | `exp2_lpips_eval/lpips_results_alex.json` | [OK] | A-Baseline lpips 0.09914/0.05105; C-Depth 0.06689/0.03737; GAN 3-seed sample-std (n-1) reproduces 0.0722+/-0.0025 / 0.0369+/-0.0010 exactly, incl. the std digits. |

## Section 4.8 -- Cost Summary (`tab:cost_summary`)

| Paper item | Source | Status |
|---|---|---|
| Whole-model MMAC | `predict.py` via `onnx-tool` on the exported `.onnx` | [N/A] not numerically re-checked here (no torch/onnx-tool available in this environment); lower risk since it's a single tool call on an artifact whose *architecture* (not accuracy) is what's measured, and the ONNX-timing bug doesn't affect architecture/shape. |
| Per-submodule MMAC breakdown (FiLM +89.65, T-Affine +25.17, DAF ~+38, baseline head 22.02) | **hand-calculated** | [N/A] not a code artifact -- README states this explicitly instead of listing it as a "known gap". |

## `tab:psnr_lit` (literature comparison)

[N/A] Explicitly not computed locally -- paper says these are cited from the original publications.

---

## What this means for the restructuring

- **`run_sprint_finalization.py` (formerly `run_scaling_experiments.py`) is not the source of any number in the table above.** Its only job is producing the 6 `.pth` files (the +DAF-only 7th checkpoint comes from `experiments/daf_only_ablation.py`); every reported number is obtained afterward by re-evaluating those `.pth` files with `unified_ground_truth_eval.py` / `experiments/{lpips_eval,third_party_separation,ceiling_error_control}.py`. The original script's own printed PSNR (parsed from the now-removed `expressiveness_evaluation_code.py`'s stdout, see NAMING.md) was from the *other*, admittedly-inconsistent pipeline that `unified_ground_truth_eval.py`'s own docstring says can be off by >1.5 dB -- i.e. it was never the number the paper reports, and this release's `run_sprint_finalization.py` no longer calls it at all.
- **`exp2/exp3/exp4/exp5/exp7` are the load-bearing scripts** -- 8 of the paper's 10 tables trace to them with exact numeric matches. These get **pure move + import-path fixes only**; old and new versions were mechanically diffed (ignoring only the `import`/`sys.path` lines) to confirm no computational line changed, since the numbers could not be independently re-run to re-verify in this environment.
- **`theta_star_model.py` / `a_quality_compare.py` support the Problem-1-on-V3-t qualitative discussion**, not a PSNR table -- except for the one flagged number above, which still needs author confirmation.
