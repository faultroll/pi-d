# experiments/

Supplementary experiments for the paper's Sec. 4.2-4.7, on top of the main
`pgdehazenet/` codebase (this folder's parent directory). See the top-level
`README.md`'s "What reproduces what" table for the full paper <-> script map,
and `TRACEABILITY.md` for the numeric cross-check against released results.

| File | Reproduces | Needs GPU? |
|---|---|---|
| `common.py` | Shared utilities: `CHECKPOINT_REGISTRY`, eval helpers, scene-ID parsing | No |
| `lpips_eval.py` | Table `lpips` -- LPIPS on existing checkpoints | Optional (faster with one) |
| `aod_net.py` / `lightdehazenet_net.py` | Third-party network wrappers (see `third_party_models/`) | - |
| `third_party_separation.py` | Tables `sweepA` / `realscatter` -- Problem 1 & 2(b) on AOD-Net / LightDehazeNet | Optional |
| `ceiling_error_control.py` | Table `scale_sensitivity` -- ceiling sensitivity to error type, fully synthetic | No |
| `daf_only_ablation.py` | "+DAF only" ablation row -- trains V3-t+DAF-only, evaluates, computes the interaction term | **Yes** (only script here that trains a model) |
| `scene_bootstrap.py` | Scene-level paired-bootstrap confidence intervals | No (reuses existing results) |
| `render_tables.py` | Renders `ceiling_error_control.py`'s output into the LaTeX snippets for `scale_sensitivity` | No |

## Run order

```bash
cd pgdehazenet/          # the directory containing network.py, train.py
# (experiments/ can live here, or anywhere -- pass --repo-root to point at it)

# 0) Sanity check -- no data / checkpoint / GPU needed (a few seconds)
python experiments/common.py --selftest
python experiments/third_party_separation.py --selftest
python experiments/scene_bootstrap.py --selftest

# 1) Ceiling error-type control -- fully offline, no GPU / dataset needed
python experiments/ceiling_error_control.py
python experiments/ceiling_error_control.py --real-data   # + real indoor/outdoor t_CAP error comparison

# 2) DAF-only ablation (needs pretrained/vgg16-397923af.pth, see main README)
python experiments/daf_only_ablation.py --train-mode DEBUG   # smoke test first
python experiments/daf_only_ablation.py --train-mode FULL    # full run (60 epochs, matches A/B/C/D)

# 3) LPIPS -- existing checkpoints + the DAF-only checkpoint just trained
python experiments/lpips_eval.py \
    --daf-only-checkpoint outputs/daf_only/model_C1-DepthOnly_V3-t.pth

# 4) Third-party validation (AOD-Net / LightDehazeNet)
python experiments/third_party_separation.py --network aod
python experiments/third_party_separation.py --network lightdehaze
python experiments/third_party_separation.py --network aod --real-scatter
python experiments/third_party_separation.py --network lightdehaze --real-scatter

# 5) Scene-level bootstrap CIs (after 3 above have produced records)
python experiments/scene_bootstrap.py --model-a A-Baseline --model-b D-Prop-S42

# 6) Render the scale_sensitivity LaTeX table from step 1's output
python experiments/render_tables.py
```
