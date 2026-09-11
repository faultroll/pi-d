# NAMING.md -- Paper Symbols <-> Code Identifiers

This file exists because the code predates the paper's terminology: the
architecture and diagnostic scripts were written and iterated on before the
final write-up settled on names like "D0", "FiLM", "T-Affine". This is the
single place that maps one to the other.

## Framework (paper Sec. 3)

| Paper term | Code |
|---|---|
| D0 criterion (Class A / Class B) | `physics.py` -- ASM ops, `is_class_a`-style checks; also referenced from `theta_star_model.py` |
| C1 -- ceiling analysis | `physics.py` (ASM inversion) + `unified_ground_truth_eval.py` (drives Tables `ceiling`, `ceiling_close`); scale-sensitivity supplement in `experiments/ceiling_error_control.py` |
| C2 -- structural diagnosis, theta* | `theta_star_model.py` |
| Problem 1 (on V3-t) | `a_quality_compare.py`, `theta_star_model.py` |
| Problem 1 & 2(b) (on AOD-Net / LightDehazeNet) | `experiments/third_party_separation.py` |
| Problem 2(a) | algebraic/derivation only in the paper text -- no dedicated script |
| Problem 3 | discussed in the paper text alongside Problem 1's V3-t instantiation -- no separate script |

## Enhancement modules (paper Sec. 3.4 / 4.4)

| Paper term | Code class | File |
|---|---|---|
| FiLM | `TConditionedResBlock` (per-block gamma/beta modulation conditioned on `t`); constructor flag `film_mode="PerBlock"` | `network.py` |
| T-Affine | `TConditionedAffineHead` (outputs $w_1, w_2, b$ conditioned on `t`); constructor flag `use_t_affine=True` | `network.py` |
| DepthAttnFuse / DAF | `DepthAttnFuse`; constructor flag `use_depth_attn=True` | `network.py` |
| V3-t backbone | `PGDehazeNet` with `capacity_mode="V3-t"` | `network.py` |
| $g_\text{ASM}$ affine head (no enhancement) | the `else` branch of `JBranch.forward` -- `w1, w2, b = torch.split(affine_params, [3,3,1], dim=1)` | `network.py` |

## Checkpoint names (sprint codenames <-> paper labels)

Canonical source: `experiments/common.py:CHECKPOINT_REGISTRY`.

| Sprint name | Paper label (Table `ablation_modules` / `psnr_main`) |
|---|---|
| `A-Baseline` | Baseline (V3-t) |
| `B-FiLMTAff` | +FiLM+T-Affine |
| `C-Depth` | Proposed (+FiLM+T-Affine+DAF), no GAN |
| `D-Prop-S42` / `D-Prop-S43` / `D-Prop-S44` | Proposed (+GAN), 3 seeds |
| `C1-DepthOnly` | +DAF only |

## Removed, unpublished-architecture-search identifiers

These constructor flags on `PGDehazeNet`/`JBranch`/`ABranch` still exist (so
old config dicts don't crash on construction), but the classes that
implemented them have been deleted from this release, per the note at the
top of `network.py`. Setting any of these to a non-default value raises a
`NameError` by design:

`use_repconv`, `use_t_gate`, `use_varc`, `use_a_bound`,
`dilation_mode="LightASPP"`, `channel_attn_mode` != `"None"`
(`--channel-attn-mode` in `train.py` no longer even lists `j`/`tj` as valid
choices). `dilation_mode` values `"Standard"`/`"Deep"` still run (they don't
reference a deleted class) but were never validated in the paper.
`a_net_mode` values other than the default `"M0"` also still run, for the
same reason, but are likewise unpublished/unvalidated.

The classes themselves (`TGateModulation`, `SFTLayer`, `LightASPP`,
`RepConv2d`, `RepResBlock`, `TConditionedRepResBlock`, `ChannelAttn`, plus
an unrelated internal-baseline `LightDehazeNet`/`BILDNet` pair superseded by
`experiments/lightdehazenet_net.py`'s official-weights wrapper) are gone,
not renamed -- grep the paper's PDF for any of these names and you won't find
them; they were architecture-search-only.

The standalone Fisher-information/parameter-utilization tool
(`expressiveness_evaluation_code.py` + `expressiveness/eff_param_torch.py`)
has also been removed in full. It never produced a number the paper reports
-- see `TRACEABILITY.md` for how that was confirmed.
