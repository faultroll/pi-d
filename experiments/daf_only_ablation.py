#!/usr/bin/env python3
"""
daf_only_ablation.py
============================
"Must" experiment #5 (review Section 4.5 / Section 6 item 5): add the
missing "DAF alone" ablation row.

The manuscript's Table 5 ablation ladder is:
    A: V3-t baseline
    B: A + FiLM + T-Affine                    (Delta = +0.06 dB)
    C: B + DepthAttnFuse                      (Delta = +1.34 dB, "primary marginal gain")
    D: C + training-only LSGAN                (Proposed, 3 seeds)

DepthAttnFuse's own effect (+1.34 dB) is only ever measured CONDITIONAL on
FiLM+T-Affine already being on -- there is no "DAF alone" row, so whether
that gain is DAF's own effect or an interaction with FiLM/T-Affine is
unknown (review Section 4.5 / Limitations item 3). This script trains
exactly that missing row -- V3-t + DepthAttnFuse ONLY, no FiLM, no
T-Affine, no GAN, seed 42 -- using this project's OWN train.py end to end
(same optimizer, same 60-epoch/20-warmup schedule, same loss, same data
split), so the new number is directly comparable to the existing table.

It also computes the interaction term the review explicitly asks for:
    interaction = y(FiLM+TAff+DAF) - y(FiLM+TAff) - y(DAF alone) + y(baseline)
i.e. whether DAF's marginal contribution when added on top of FiLM+T-Affine
(+1.34 dB) is bigger, smaller, or the same as DAF's effect on its own.

Training config verified against run_sprint_finalization.py:sprint_matrix's
env-var convention (this is the exact mechanism that produced
experiments/sprint/model_Sprint_Finalization_{A,B,C,D}*.pth):
    CAPACITY_MODE=V3-t  USE_DEPTH_ATTN=True  FILM_MODE=None
    USE_T_AFFINE=False  USE_GAN=False  (all other flags at train.py's
    documented defaults: CHANNEL_ATTN_MODE=None, DILATION_MODE=None,
    USE_REFINE=False, USE_REPCONV=False, USE_T_GATE=False, USE_VARC=False,
    USE_A_BOUND=False, A_NET_MODE=M0, A_GT_MODE=global)
    --seed 42 --train-mode FULL   (60 epochs, 20-epoch warmup, matching
    Section 4.2 exactly; GAN is off throughout so the warmup boundary is
    inert here, same as it is for the existing A/B/C rows)

IMPORTANT -- this is the one script in this bundle that actually needs a
GPU and real time (paper reports ~RTX 4060; expect the same order of
training time as any single one of your existing A/B/C/D sprint runs,
since the compute cost is identical -- DAF adds only ~38 MMAC).

Usage:
    cd pgdehazenet/
    # 1) quick smoke test first (3 epochs, 14 samples, no GPU required to
    #    just confirm the pipeline runs end to end without crashing):
    python experiments/daf_only_ablation.py --train-mode DEBUG

    # 2) the real run:
    python experiments/daf_only_ablation.py --train-mode FULL

    # 3) already trained it and just want the evaluation + interaction term:
    python experiments/daf_only_ablation.py --skip-train \\
        --checkpoint outputs/daf_only/model_C1-DepthOnly_V3-t.pth
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    ensure_repo_on_path, repo_path, default_split_paths,
    load_named_checkpoint, load_checkpoint_from_path,
    register_daf_only_checkpoint, evaluate_split, summarize,
)

DAF_ONLY_CONFIG = dict(
    capacity_mode="V3-t", dilation_mode="None", channel_attn_mode="None",
    use_refine=False, use_repconv=False, use_t_gate=False,
    use_varc=False, use_a_bound=False, a_net_mode="M0",
    use_depth_attn=True, film_mode="None", use_t_affine=False,
)


def check_vgg16_prereq(repo_root):
    """train.py's perceptual loss (network.py:get_vgg_features) needs
    <repo_root>/pretrained/vgg16-<hash>.pth to already exist -- this is a
    one-time environment setup step from prepare.md, not something any
    training config (including this DAF-only run) can skip, since the
    J-branch loss calls it unconditionally. Check for it BEFORE launching
    the (possibly hours-long) training subprocess, so a missing file fails
    in under a second with a clear fix instead of a multi-line traceback
    partway through epoch 1."""
    import glob
    pattern = os.path.join(str(repo_root), "pretrained", "vgg16-*.pth")
    matches = glob.glob(pattern)
    if matches:
        return
    raise FileNotFoundError(
        "\n"
        "Missing pretrained VGG16 weights -- train.py's perceptual loss needs this\n"
        "regardless of which config you train (this is not specific to exp5).\n"
        f"Expected a file matching: {pattern}\n\n"
        "Fix (one-time, on this machine):\n"
        f"  mkdir -p {os.path.join(str(repo_root), 'pretrained')}\n"
        f"  cd {os.path.join(str(repo_root), 'pretrained')}\n"
        "  wget https://download.pytorch.org/models/vgg16-397923af.pth\n"
        "  # (or curl -O https://download.pytorch.org/models/vgg16-397923af.pth)\n\n"
        "This is exactly prepare.md's step 1 -- your A-Baseline/B-FiLMTAff/C-Depth/\n"
        "D-Prop checkpoints were presumably trained on a machine (or at a time) that\n"
        "already had this file in place; it just needs to be added here too."
    )


def build_training_env(seed):
    env = dict(os.environ)
    env["CAPACITY_MODE"] = DAF_ONLY_CONFIG["capacity_mode"]
    env["USE_DEPTH_ATTN"] = str(DAF_ONLY_CONFIG["use_depth_attn"])
    env["FILM_MODE"] = DAF_ONLY_CONFIG["film_mode"]
    env["USE_T_AFFINE"] = str(DAF_ONLY_CONFIG["use_t_affine"])
    env["USE_GAN"] = "False"  # explicit: train.py's own default is 'True' if unset!
    env["CHANNEL_ATTN_MODE"] = DAF_ONLY_CONFIG["channel_attn_mode"]
    env["DILATION_MODE"] = DAF_ONLY_CONFIG["dilation_mode"]
    env["USE_REFINE"] = str(DAF_ONLY_CONFIG["use_refine"])
    env["USE_REPCONV"] = str(DAF_ONLY_CONFIG["use_repconv"])
    env["USE_T_GATE"] = str(DAF_ONLY_CONFIG["use_t_gate"])
    env["USE_VARC"] = str(DAF_ONLY_CONFIG["use_varc"])
    env["USE_A_BOUND"] = str(DAF_ONLY_CONFIG["use_a_bound"])
    env["A_NET_MODE"] = DAF_ONLY_CONFIG["a_net_mode"]
    env["A_GT_MODE"] = "global"  # matches "final proposed method uses ACAP supervision as in V3-t baseline"
    env.pop("TRANSFER_FROM", None)
    return env


def run_training(repo_root, seed, train_mode, best_model_path, onnx_path, extra_train_args=None):
    env = build_training_env(seed)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    cmd = [
        sys.executable, "train.py",
        "--seed", str(seed),
        "--train-mode", train_mode,
        "--best-model-path", best_model_path,
        "--onnx-path", onnx_path,
    ]
    if extra_train_args:
        cmd += extra_train_args
    print("Launching training subprocess:")
    print(f"  cwd = {repo_root}")
    print(f"  cmd = {' '.join(cmd)}")
    print(f"  key env overrides = CAPACITY_MODE={env['CAPACITY_MODE']} "
          f"USE_DEPTH_ATTN={env['USE_DEPTH_ATTN']} FILM_MODE={env['FILM_MODE']} "
          f"USE_T_AFFINE={env['USE_T_AFFINE']} USE_GAN={env['USE_GAN']}")
    result = subprocess.run(cmd, cwd=str(repo_root), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode}. "
                            f"Scroll up for its own error output.")
    if not os.path.exists(best_model_path):
        raise RuntimeError(f"train.py finished but {best_model_path} was not created -- "
                            f"check train.py's own logs above for what it actually saved.")
    print(f"Training finished. Checkpoint at {best_model_path}")


def evaluate_daf_only(checkpoint_path, device):
    model = load_checkpoint_from_path(checkpoint_path, DAF_ONLY_CONFIG, device)
    splits = default_split_paths()
    results = {}
    for domain, path in splits.items():
        if not os.path.exists(path):
            results[domain] = {"error": f"split not found: {path}"}
            continue
        records = evaluate_split(model, path, device)
        results[domain] = {"summary": summarize(records), "records": records}
    return results


def try_load_and_eval(name, device):
    """Best-effort: evaluate an existing registry checkpoint (A-Baseline /
    B-FiLMTAff / C-Depth) for the interaction-term computation. Returns
    None (with a printed reason) if the checkpoint isn't found rather than
    raising, so the script still reports the DAF-only row on its own even
    if the older sprint checkpoints aren't available on this machine."""
    try:
        model, config, pth_path = load_named_checkpoint(name, device)
    except FileNotFoundError as e:
        print(f"  ({name} not available for interaction-term comparison: {e})")
        return None
    splits = default_split_paths()
    results = {}
    for domain, path in splits.items():
        if not os.path.exists(path):
            return None
        records = evaluate_split(model, path, device, progress=False)
        results[domain] = summarize(records)
    del model
    return results


def weighted_avg(indoor_summary, outdoor_summary):
    n_in, n_out = indoor_summary["N"], outdoor_summary["N"]
    total = n_in + n_out
    if total == 0:
        return float("nan")
    return (indoor_summary["psnr_mean"] * n_in + outdoor_summary["psnr_mean"] * n_out) / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42, help="matches A/B/C rows' single seed (42)")
    ap.add_argument("--train-mode", type=str, default="FULL", choices=["DEBUG", "QUICK", "MEDIUM", "FULL"],
                     help="use DEBUG first as a fast smoke test (3 epochs, 14 samples)")
    ap.add_argument("--skip-train", action="store_true",
                     help="skip training, just evaluate an existing --checkpoint")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="with --skip-train: path to an already-trained DAF-only checkpoint. "
                          "without --skip-train: where to save the new checkpoint (default: "
                          "<repo>/outputs/daf_only/model_C1-DepthOnly_V3-t.pth)")
    ap.add_argument("--checkpoint-map", type=str, default=None,
                     help="JSON overriding CHECKPOINT_REGISTRY paths for A-Baseline/B-FiLMTAff/C-Depth "
                          "lookups used in the interaction-term computation")
    ap.add_argument("--skip-interaction", action="store_true",
                     help="don't try to load A-Baseline/B-FiLMTAff/C-Depth for the interaction term")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    repo_root = ensure_repo_on_path(args.repo_root)
    if args.checkpoint_map:
        from common import load_checkpoint_map_override
        load_checkpoint_map_override(args.checkpoint_map)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else
          "  (WARNING: no GPU detected -- training will be extremely slow; "
          "fine for --train-mode DEBUG, not recommended for FULL)"))

    ckpt_path = args.checkpoint or repo_path("outputs", "daf_only", "model_C1-DepthOnly_V3-t.pth")
    onnx_path = os.path.splitext(ckpt_path)[0] + ".onnx"

    print("=" * 72)
    print("Experiment 5: V3-t + DepthAttnFuse ONLY (no FiLM, no T-Affine, no GAN)")
    print(f"config = {DAF_ONLY_CONFIG}")
    print("=" * 72)

    if not args.skip_train:
        check_vgg16_prereq(repo_root)
        run_training(repo_root, args.seed, args.train_mode, ckpt_path, onnx_path)
    elif not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"--skip-train given but checkpoint not found at {ckpt_path}. "
                                 f"Pass --checkpoint /path/to/model.pth explicitly.")

    register_daf_only_checkpoint(ckpt_path)

    print("\nEvaluating DAF-only checkpoint on indoor_test.txt / outdoor_test.txt ...")
    daf_results = evaluate_daf_only(ckpt_path, device)
    out = {"config": DAF_ONLY_CONFIG, "checkpoint": ckpt_path, "seed": args.seed, "domains": {}}
    for domain, d in daf_results.items():
        if "error" in d:
            print(f"  {domain}: {d['error']}")
            out["domains"][domain] = d
        else:
            s = d["summary"]
            print(f"  {domain:8s}: N={s['N']:3d}  PSNR={s['psnr_mean']:.2f}+/-{s['psnr_std']:.2f}  "
                  f"SSIM={s['ssim_mean']:.4f}+/-{s['ssim_std']:.4f}")
            out["domains"][domain] = {"summary": s, "records": d["records"]}

    have_both = all(k in daf_results and "error" not in daf_results[k] for k in ("indoor", "outdoor"))
    if have_both:
        avg_daf = weighted_avg(daf_results["indoor"]["summary"], daf_results["outdoor"]["summary"])
        print(f"  {'Avg (35/74 weighted)':8s}: PSNR={avg_daf:.2f}")
        out["avg_psnr_35_74_weighted"] = avg_daf

    # ---- interaction term ----
    if not args.skip_interaction and have_both:
        print("\nLoading A-Baseline / B-FiLMTAff / C-Depth for the interaction-term "
              "comparison (best effort -- OK if some are missing) ...")
        others = {}
        for name in ["A-Baseline", "B-FiLMTAff", "C-Depth"]:
            r = try_load_and_eval(name, device)
            if r is not None:
                others[name] = {
                    "indoor": r["indoor"], "outdoor": r["outdoor"],
                    "avg": weighted_avg(r["indoor"], r["outdoor"]),
                }
                print(f"  {name:12s} indoor={r['indoor']['psnr_mean']:.2f}  "
                      f"outdoor={r['outdoor']['psnr_mean']:.2f}  avg={others[name]['avg']:.2f}")

        if all(k in others for k in ["A-Baseline", "B-FiLMTAff", "C-Depth"]):
            y_base = others["A-Baseline"]["avg"]
            y_film_taff = others["B-FiLMTAff"]["avg"]
            y_film_taff_daf = others["C-Depth"]["avg"]
            y_daf_alone = avg_daf

            daf_marginal_given_film_taff = y_film_taff_daf - y_film_taff
            daf_alone_effect = y_daf_alone - y_base
            film_taff_effect = y_film_taff - y_base
            interaction = (y_film_taff_daf - y_film_taff) - (y_daf_alone - y_base)
            # equivalently: y(F+D) - y(F) - y(D) + y(B)
            interaction_check = y_film_taff_daf - y_film_taff - y_daf_alone + y_base

            print("\n" + "=" * 72)
            print("Interaction term  (review Section 4.5, y(F+D) - y(F) - y(D) + y(B)):")
            print("=" * 72)
            print(f"  y(baseline)                        = {y_base:.3f} dB")
            print(f"  y(FiLM+T-Affine)                    = {y_film_taff:.3f} dB   "
                  f"[FiLM+T-Affine alone: {film_taff_effect:+.3f} dB]")
            print(f"  y(DAF alone)                        = {y_daf_alone:.3f} dB   "
                  f"[DAF alone: {daf_alone_effect:+.3f} dB]")
            print(f"  y(FiLM+T-Affine+DAF)                = {y_film_taff_daf:.3f} dB   "
                  f"[DAF given FiLM+T-Affine already on: {daf_marginal_given_film_taff:+.3f} dB]")
            print(f"\n  interaction = {interaction:+.3f} dB  (sanity check via 2nd formula: {interaction_check:+.3f} dB)")
            if abs(interaction) < 0.2:
                print("  -> small interaction: DAF's effect is roughly ADDITIVE with FiLM+T-Affine's --")
                print("     '+1.34 dB' is close to being DAF's own effect, not mostly an interaction artifact.")
            else:
                sign = "super-additive (DAF helps MORE together with FiLM+T-Affine)" if interaction > 0 \
                    else "sub-additive (DAF helps LESS together with FiLM+T-Affine)"
                print(f"  -> non-trivial interaction ({sign}): the manuscript's +1.34 dB 'DAF marginal")
                print("     contribution' number is NOT simply DAF's own, context-free effect.")

            out["interaction_term"] = {
                "y_baseline": y_base, "y_film_taff": y_film_taff,
                "y_daf_alone": y_daf_alone, "y_film_taff_daf": y_film_taff_daf,
                "daf_alone_effect": daf_alone_effect,
                "film_taff_effect": film_taff_effect,
                "daf_marginal_given_film_taff": daf_marginal_given_film_taff,
                "interaction": interaction,
            }
        else:
            print("  (need all three of A-Baseline / B-FiLMTAff / C-Depth to compute the "
                  "interaction term; skipping. Use --checkpoint-map if their paths differ "
                  "from CHECKPOINT_REGISTRY's defaults.)")

    if args.output:
        out_json = args.output
        out_dir = os.path.dirname(out_json) or "."
    else:
        out_dir = repo_path("outputs", "exp5_daf_only_ablation")
        out_json = os.path.join(out_dir, "results.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")
    print("\nNew ablation-table row to add to Table 5:")
    if have_both:
        print(f"  +DAF only     {daf_results['indoor']['summary']['psnr_mean']:.2f}          "
              f"{daf_results['outdoor']['summary']['psnr_mean']:.2f}          {avg_daf:.2f}")


if __name__ == "__main__":
    main()
