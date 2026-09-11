# Physics-Informed Diagnosis for Low-Cost Enhancement of Lightweight Image Dehazing Networks

This repository contains the **diagnostic framework** used in the paper (ceiling
analysis C1, structural diagnosis C2 with Problems 1/2(a)/2(b)/3), its instantiation
on the lightweight network **V3-t**, the three derived enhancement modules
(**FiLM**, **T-Affine**, **DepthAttnFuse**), and the external validation on
**AOD-Net** and **LightDehazeNet**. Trained checkpoints and the numeric results
referenced in the paper's tables are released separately (see
`dehaze-results-release`, distributed alongside this code).

**See also:** [`NAMING.md`](NAMING.md) for the paper-symbol <-> code-identifier
glossary, and [`TRACEABILITY.md`](TRACEABILITY.md) for the table-by-table audit of
which script produced which number, cross-checked against the released results.

This is a **minimal** release: earlier development versions of this codebase also
contained an architecture-search sweep (T-Gate/SFT/RepConv/dilation variants, an
A-branch M0-M8 spatial-vs-global sweep, a capacity scaling-curve sweep) and a
separate Fisher-information/parameter-utilization analysis tool. None of that is
part of the paper's reported method or reported numbers, so -- rather than keep it
around as an unused, confusing parallel path -- it has been **removed**, not just
excluded. `network.py` and `run_sprint_finalization.py` carry a short note where
that code used to live. Everything below **is** part of the reported method.

## Repository structure

```
pgdehazenet/
|-- network.py                     Model definitions: V3-t backbone, FiLM
|                                   (TConditionedResBlock), T-Affine
|                                   (TConditionedAffineHead), DepthAttnFuse (DAF),
|                                   PGDehazeNet. See the module docstring for which
|                                   constructor flags are paper-supported.
|-- physics.py                     Atmospheric Scattering Model (ASM) ops, the
|                                   Color Attenuation Prior (CAP), ASM inversion --
|                                   D0 criterion, C1 ceiling analysis
|-- train.py                       Training loop (physics losses + VGG perceptual
|                                   + optional LSGAN discriminator)
|-- predict.py                     Inference + onnx-tool MAC/param profiling.
|                                   Read the warning at the top of this file before
|                                   using it to reproduce any accuracy number.
|-- metrics.py                     PSNR / SSIM (shared by every script below)
|-- dataset_reside.py              RESIDE-SOTS dataset + split generation
|-- featuremaps.py                 Forward-hook feature map extraction (used by
|                                   train.py for periodic monitoring)
|-- unified_ground_truth_eval.py   Canonical evaluation script -- reproduces the
|                                   headline numbers in Tables ceiling, psnr_main,
|                                   ceiling_close, ablation_modules
|-- theta_star_model.py            Model-centric A-branch / theta* diagnostics --
|                                   Problem 1 / C2, structural diagnosis on V3-t
|-- a_quality_compare.py           CAP-global-A vs ASM-inversion-A comparison,
|                                   supports the Problem-1-on-V3-t discussion
|-- a_spatial_stats.py             Spatial-vs-global atmospheric-light diagnostics,
|                                   supports the C1 smoothing-rationale discussion
|-- run_sprint_finalization.py     Trains the 6 V3-t checkpoints this release's
|                                   numbers are built from. Does NOT itself compute
|                                   any reported number -- see the module docstring.
|-- prepare.md                     Original one-page setup note
|-- SOTS/split_txt/                train/val/indoor_test/outdoor_test file lists
|                                   (defines the exact reproducible split)
`-- experiments/                   Supplementary experiments (paper Sec. 4.2-4.7)
    |-- README.md                  Run order + what each script reproduces
    |-- common.py                  Shared checkpoint registry / eval utilities --
    |                               CHECKPOINT_REGISTRY is the canonical
    |                               checkpoint-name <-> paper-config map
    |-- lpips_eval.py               Table lpips
    |-- aod_net.py                  AOD-Net wrapper (loads third-party checkpoint)
    |-- lightdehazenet_net.py       LightDehazeNet wrapper (independent reimpl.
    |                                of the official architecture)
    |-- third_party_separation.py   Problem 1 & Problem 2(b) on AOD-Net /
    |                                LightDehazeNet -- Tables sweepA / realscatter
    |-- ceiling_error_control.py    Table scale_sensitivity
    |-- daf_only_ablation.py        "+DAF only" ablation row (trains a checkpoint)
    |-- scene_bootstrap.py          Scene-level bootstrap confidence intervals
    |-- render_tables.py            Generates LaTeX table snippets from
    |                                ceiling_error_control.py's output
    `-- third_party_models/         Trimmed third-party weights, see note below
```

## Setup

```bash
pip install -r requirements.txt
```

1. Download `vgg16-397923af.pth` from
   `https://download.pytorch.org/models/vgg16-397923af.pth` and place it in
   `pgdehazenet/pretrained/`.
2. Download RESIDE-SOTS from
   `https://sites.google.com/view/reside-dehaze-datasets/reside-standard` and
   uncompress it into `pgdehazenet/SOTS/` (same directory as `train.py`). The
   exact train/val/test split used for the paper is already fixed in
   `SOTS/split_txt/*.txt`; you do not need to regenerate it, but
   `dataset_reside.py` can re-derive the same split if needed.
3. Everything in this repo is run **from the `pgdehazenet/` directory** (i.e.
   `cd pgdehazenet` first) -- scripts resolve `./SOTS/...` and `outputs/...`
   relative to the current working directory, not to the script's own location.
4. `experiments/third_party_models/` already contains the two trimmed
   third-party dependencies needed by `experiments/third_party_separation.py`
   (see note below) -- no extra download needed for those.

## A bug worth knowing about: ONNX exports are last-epoch, not best-val

`train.py` exports a `.onnx` file **after the training loop finishes**, using
whatever weights `G` holds at that point -- it does **not** reload the
best-validation-loss `.pth` it saved earlier. The two can differ. Every
accuracy number in the paper (PSNR/SSIM/LPIPS, every table) is computed from
the `.pth` checkpoints, never from a `.onnx` file. The `.onnx` export exists
solely so `predict.py` can run `onnx-tool`'s MAC/param counter for Table
`cost_summary`'s whole-model row, where only the graph shape matters, not the
specific weight values. `predict.py`'s own `--model_path` defaults to a
`.onnx` path (a separate, pre-existing quirk) -- point it at a `.pth` file if
you want its printed PSNR/SSIM to match the paper.

## What reproduces what

| Paper item | Script(s) |
|---|---|
| Ceiling analysis (C1), Table `ceiling` / `ceiling_close` | `physics.py` + `unified_ground_truth_eval.py` |
| Scale-sensitivity, Table `scale_sensitivity` | `experiments/ceiling_error_control.py`, table text via `experiments/render_tables.py` |
| Problem 1 on V3-t discussion | `a_quality_compare.py`, `theta_star_model.py` |
| Problem 1 & 2(b) on AOD-Net / LightDehazeNet, Table `sweepA` / `realscatter` | `experiments/third_party_separation.py` (+ `experiments/aod_net.py`, `experiments/lightdehazenet_net.py`) |
| Literature comparison, Table `psnr_lit` | not computed locally -- values are cited from the original papers |
| Main results, Table `psnr_main` / `ceiling_close` / `ablation_modules` (Baseline, +FiLM+T-Affine, Proposed, +GAN x3 seeds) | `unified_ground_truth_eval.py` against the checkpoints in `dehaze-results-release/main_results/checkpoints/` |
| Ablation "+DAF only" row | `experiments/daf_only_ablation.py` |
| Scene-level bootstrap CIs | `experiments/scene_bootstrap.py` |
| LPIPS, Table `lpips` | `experiments/lpips_eval.py` |
| Cost, Table `cost_summary` (whole-model MMAC) | `predict.py` (onnx-tool) |
| Cost, Table `cost_summary` (**per-submodule** MMAC breakdown) | **hand-calculated** from the module definitions in `network.py`, not a script -- see note below |
| Qualitative figures (`qual_indoor.pdf`, `qual_outdoor.pdf`) | not regenerated here; released as final images in `dehaze-results-release/figures/` |
| $g_\text{ASM}$ diagram, Class A/B diagram, ceiling bar chart | pure TikZ in the paper source, no code dependency |

**Design note on the per-submodule cost breakdown:** the FiLM (+89.65 MMAC),
T-Affine (+25.17 MMAC), DepthAttnFuse (~+38 MMAC), and baseline-head (22.02
MMAC) figures in Table `cost_summary` were computed by hand from each
module's layer definitions (channel counts, kernel sizes) in `network.py`,
not generated by a script. This is a deliberate choice, not a missing tool --
a reader can recompute the same numbers directly from the class definitions.

## Third-party models (`experiments/third_party_models/`)

`experiments/third_party_separation.py` evaluates the diagnostic framework
against the **officially released weights** of two third-party networks.
Only the files actually imported at load time are kept here (not a full
clone of either upstream repository):

- **AOD-Net** -- `AODnet-by-pytorch/model.py` (architecture, needed to unpickle
  the checkpoint) + `model_pretrained/AOD_net_epoch_relu_10.pth`.
  Upstream: https://github.com/weber0522bb/AODnet-by-pytorch
- **LightDehazeNet** -- `Light-DehazeNet/trained_weights/trained_LDNet.pth`
  only; `experiments/lightdehazenet_net.py` contains an independent
  architecture reimplementation and does not import the upstream code.
  Upstream: https://github.com/hayatkhan8660-maker/Light-DehazeNet

Neither upstream repository publishes an explicit license file. The weights
are redistributed here, with attribution, solely for evaluation/reproducibility
of the numbers in Tables `sweepA` and `realscatter`, consistent with common
academic practice for benchmark comparison -- you may want to confirm this is
acceptable to you before making the repository public, or replace this with a
small download script that fetches directly from the upstream repos instead.
