#!/usr/bin/env python3
"""
lpips_eval.py
====================
"Must" experiment #2 (review Section 4.3 / Section 6 item 2): LPIPS
measurement, inference-only, on the existing checkpoints.

The manuscript explains the GAN's -0.11 dB mean-PSNR change via the
perception-distortion trade-off (Blau & Michaeli 2018) but never measures
a perceptual metric to check that story -- a Limitations item says so
explicitly, and the advisor flags this as unresolved because the thing
being explained hasn't actually been measured yet.

This script runs LPIPS (Zhang et al. 2018) inference-only over every
existing checkpoint in common.CHECKPOINT_REGISTRY (A-Baseline,
B-FiLMTAff, C-Depth, D-Prop seeds 42/43/44, and -- if you've run
daf_only_ablation.py first -- the new DAF-only row too), on the SAME
indoor_test.txt / outdoor_test.txt split used for Table 4/5, so the numbers
slot directly into the existing tables. No training, no new data.

Needs: pip install lpips   (pulls in its own small AlexNet/VGG weights on
first run; those come from the `lpips` package's own PyPI-hosted assets,
not from any external image-hosting site).

Usage:
    cd pgdehazenet/
    python experiments/lpips_eval.py
    python experiments/lpips_eval.py --net vgg   # cross-check with a 2nd backbone
    python experiments/lpips_eval.py --models A-Baseline C-Depth D-Prop-S42 D-Prop-S43 D-Prop-S44
    python experiments/lpips_eval.py --daf-only-checkpoint outputs/daf_only/model.pth
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    ensure_repo_on_path, repo_path, default_split_paths,
    CHECKPOINT_REGISTRY, load_named_checkpoint, load_checkpoint_map_override,
    register_daf_only_checkpoint, summarize,
)


def evaluate_split_with_lpips(model, split_txt, device, lpips_fn, max_samples=None, progress=True):
    """Same contract as common.evaluate_split, plus an 'lpips' field
    per record. Kept as its own loop (rather than bolting onto
    evaluate_split) so plain PSNR/SSIM-only runs never pay for loading the
    LPIPS network."""
    import torch
    from metrics import compute_psnr, compute_ssim
    from PIL import Image
    from common import scene_id_from_hazy_path, domain_from_path, classify_haze_level

    with open(split_txt) as f:
        lines = [l.strip() for l in f.read().strip().split("\n") if l.strip()]
    pairs = [l.split() for l in lines]
    if max_samples is not None:
        pairs = pairs[:max_samples]

    iterator = pairs
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(pairs, desc=os.path.basename(split_txt), ncols=80)
        except ImportError:
            pass

    model.eval()
    records = []
    with torch.no_grad():
        for hazy_path, gt_path in iterator:
            I_pil = Image.open(hazy_path).convert("RGB")
            J_pil = Image.open(gt_path).convert("RGB")
            I_np = np.asarray(I_pil).astype(np.float32) / 255.0
            J_np = np.asarray(J_pil).astype(np.float32) / 255.0
            I_t = torch.from_numpy(np.transpose(I_np, (2, 0, 1))).unsqueeze(0).to(device)
            out = model(I_t)
            J_pred_t = out[-1] if isinstance(out, (list, tuple)) else out
            J_pred_t = torch.clamp(J_pred_t, 0, 1)
            J_pred_np = np.transpose(J_pred_t[0].cpu().numpy(), (1, 2, 0))

            # LPIPS wants [-1, 1], NCHW, float
            gt_t = torch.from_numpy(np.transpose(J_np, (2, 0, 1))).unsqueeze(0).to(device)
            lp = lpips_fn(2 * J_pred_t - 1, 2 * gt_t - 1)
            lpips_val = float(lp.mean().item())

            records.append({
                "hazy_path": hazy_path,
                "gt_path": gt_path,
                "scene_id": scene_id_from_hazy_path(hazy_path),
                "domain": domain_from_path(hazy_path),
                "haze_level": classify_haze_level(hazy_path),
                "psnr": float(compute_psnr(J_pred_np, J_np)),
                "ssim": float(compute_ssim(J_pred_np, J_np)),
                "lpips": lpips_val,
            })
    return records


def summarize_with_lpips(records):
    base = summarize(records)
    if records:
        vals = [r["lpips"] for r in records]
        base["lpips_mean"] = float(np.mean(vals))
        base["lpips_std"] = float(np.std(vals))
    else:
        base["lpips_mean"] = float("nan")
        base["lpips_std"] = float("nan")
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument("--models", type=str, nargs="+", default=None,
                     help="subset of CHECKPOINT_REGISTRY names to evaluate (default: all)")
    ap.add_argument("--net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                     help="LPIPS backbone. 'alex' is the LPIPS-paper default and is independent "
                          "of this project's own VGG perceptual training loss; pass --net vgg too "
                          "if you want a second opinion, since training already used VGG features.")
    ap.add_argument("--checkpoint-map", type=str, default=None,
                     help="JSON overriding CHECKPOINT_REGISTRY paths, see common.py docstring")
    ap.add_argument("--daf-only-checkpoint", type=str, default=None,
                     help="path to the checkpoint produced by daf_only_ablation.py; "
                          "if given, adds a 'C1-DepthOnly' row to the evaluation")
    ap.add_argument("--max-samples", type=int, default=None, help="debug: cap samples per split")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    ensure_repo_on_path(args.repo_root)
    if args.checkpoint_map:
        load_checkpoint_map_override(args.checkpoint_map)
    if args.daf_only_checkpoint:
        register_daf_only_checkpoint(args.daf_only_checkpoint)

    import torch
    import lpips as lpips_pkg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    lpips_fn = lpips_pkg.LPIPS(net=args.net).to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model_names = args.models or list(CHECKPOINT_REGISTRY.keys())
    splits = default_split_paths()
    for domain, path in splits.items():
        if not os.path.exists(path):
            print(f"WARNING: split file not found: {path} -- this must be run on a machine "
                  f"with SOTS/ populated (see prepare.md).")

    out_dir = args.output or repo_path("outputs", "exp2_lpips_eval")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    print("=" * 100)
    print(f"{'model':<20} {'domain':<9} {'N':>4} {'PSNR':>8} {'SSIM':>8} {'LPIPS(' + args.net + ')':>14}")
    print("-" * 100)
    for name in model_names:
        if name not in CHECKPOINT_REGISTRY:
            print(f"  (skipping unknown model {name!r})")
            continue
        try:
            model, config, pth_path = load_named_checkpoint(name, device)
        except FileNotFoundError as e:
            print(f"  {name:<20} SKIPPED -- {e}")
            continue

        all_results[name] = {"label": CHECKPOINT_REGISTRY[name]["label"], "checkpoint": pth_path}
        for domain, split_path in splits.items():
            if not os.path.exists(split_path):
                continue
            records = evaluate_split_with_lpips(model, split_path, device, lpips_fn,
                                                 max_samples=args.max_samples)
            summ = summarize_with_lpips(records)
            all_results[name][domain] = {"summary": summ, "records": records}
            print(f"{name:<20} {domain:<9} {summ['N']:>4} {summ['psnr_mean']:>8.2f} "
                  f"{summ['ssim_mean']:>8.4f} {summ['lpips_mean']:>10.4f} +/- {summ['lpips_std']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- headline comparison: does GAN training actually help LPIPS? ----
    print("\n" + "=" * 100)
    print("GAN ablation check (review Section 4.3): C-Depth (no GAN) vs Proposed (+GAN, 3 seeds)")
    print("=" * 100)
    seed_names = [n for n in ["D-Prop-S42", "D-Prop-S43", "D-Prop-S44"] if n in all_results]
    if "C-Depth" in all_results and seed_names:
        for domain in ["indoor", "outdoor"]:
            if domain not in all_results.get("C-Depth", {}):
                continue
            no_gan = all_results["C-Depth"][domain]["summary"]
            gan_psnr = [all_results[n][domain]["summary"]["psnr_mean"] for n in seed_names
                        if domain in all_results[n]]
            gan_lpips = [all_results[n][domain]["summary"]["lpips_mean"] for n in seed_names
                         if domain in all_results[n]]
            gan_ssim = [all_results[n][domain]["summary"]["ssim_mean"] for n in seed_names
                        if domain in all_results[n]]
            if not gan_psnr:
                continue
            print(f"\n{domain}:")
            print(f"  C-Depth (no GAN):        PSNR={no_gan['psnr_mean']:.2f}  "
                  f"SSIM={no_gan['ssim_mean']:.4f}  LPIPS={no_gan['lpips_mean']:.4f}")
            print(f"  Proposed (+GAN, {len(gan_psnr)} seeds): PSNR={np.mean(gan_psnr):.2f}+/-{np.std(gan_psnr):.2f}  "
                  f"SSIM={np.mean(gan_ssim):.4f}+/-{np.std(gan_ssim):.4f}  "
                  f"LPIPS={np.mean(gan_lpips):.4f}+/-{np.std(gan_lpips):.4f}")
            d_psnr = np.mean(gan_psnr) - no_gan["psnr_mean"]
            d_lpips = np.mean(gan_lpips) - no_gan["lpips_mean"]
            print(f"  Delta:  PSNR {d_psnr:+.2f} dB,  LPIPS {d_lpips:+.4f} "
                  f"({'lower is better -> GAN HELPS perceptually' if d_lpips < 0 else 'GAN does NOT improve LPIPS here'})")
            print("  -> if PSNR delta is negative AND LPIPS delta is negative (improves), that is real")
            print("     evidence for the perception-distortion trade-off story in Section 4.9/Limitations.")
            print("     If LPIPS delta is ~0 or positive, the PSNR drop is NOT explained by that trade-off")
            print("     and the manuscript's claim there should be softened per the review.")
    else:
        print("  (need both C-Depth and at least one D-Prop-S4x checkpoint to run this comparison)")

    out_json = os.path.join(out_dir, f"lpips_results_{args.net}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full per-sample results -> {out_json}")


if __name__ == "__main__":
    main()
