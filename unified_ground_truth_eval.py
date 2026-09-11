"""
Unified re-evaluation script -- one run produces a single self-consistent
set of ground-truth numbers for every table in the paper.

Why this script exists
-----------------------
The codebase had two independent evaluation pipelines (the
expressiveness_evaluation_code.py system and the collect_paper_data.py
system) that, for the same checkpoint and the same claimed 35-indoor +
74-outdoor test set, produced PSNR values differing by up to 1.55 dB
(C-Depth_V3-t indoor: 20.454 vs 18.901). This isn't a "which table is
right" question -- the two code paths are directly inconsistent with each
other. This script depends only on the three lowest-level, unforked
modules (network.py / physics.py / metrics.py) and re-runs
"read image -> inference -> compute metrics" from scratch, producing one
set of numbers.

It also computes the quantities Theorem 1 (ceiling) actually needs:
  - "CAP baseline": t_CAP + A_CAP (A_CAP is CAP/DCP's blind estimate from
    the hazy image, not using ground truth). This is what the paper's
    current Table 1 17.92 dB row actually measures (psnr_cap_global).
  - "Ceiling (corrected)": t_CAP + A_gt_smooth (A_gt_smooth comes from ASM
    inversion via physics.py's estimate_A_from_pair, with 1/16 smoothing).
    This is the quantity Theorem 1's text definition and Appendix B.2's
    "zero residual" proof actually describe. The function already exists
    in the codebase but was never wired into the paper's final numbers.

Neither quantity needs new training -- all checkpoints are already trained
in experiments/sprint/outputs/; this script only re-runs inference and
recomputes PSNR/SSIM. CPU is sufficient: the 35+74 image set finishes in a
few minutes; the full SOTS set (500+500) takes roughly 10-30 minutes
depending on hardware.

Usage
-----
1. Place this file in sprint_code/pgdehazenet/ (alongside network.py /
   physics.py / metrics.py).
2. Make sure SOTS/indoor/{hazy,gt} and SOTS/outdoor/{hazy,gt} contain the
   actual images, not just the split_txt filename lists.
3. Make sure experiments/sprint/outputs/model_Sprint_Finalization_*.pth
   are present.
4. With only the internal 35+74 test set:
       python unified_ground_truth_eval.py
   With the full official SOTS available locally (500 images per domain,
   needed for a fair comparison against literature values like FFA-Net /
   DehazeFormer), point INDOOR_TEST_TXT / OUTDOOR_TEST_TXT at txt files
   covering the full set (format: "hazy_path gt_path" per line, same
   layout as SOTS/split_txt/indoor_test.txt). Run both and report both;
   the paper should state clearly which split was used.
5. Output:
   - ground_truth_summary.json -- machine-readable version of every number
   - a Markdown table printed to stdout, ready to paste into the paper
"""

import os
import sys
import json
import math
import argparse

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as P          # noqa: E402  (reuse existing implementation, no reinvention)
import metrics as M          # noqa: E402
import network as N          # noqa: E402

try:
    import torch
except ImportError:
    print("Requires `pip install torch` first (CPU build is fine, no GPU needed)")
    raise


# ----------------------------------------------------------------------
#  Config: each checkpoint to evaluate, with its architecture flags
#  (flag values copied directly from experiments/sprint/
#   expressiveness_Sprint_Finalization_*.json's "config" field, verified
#   1:1 against stages A/B/C/D to avoid a transcription error causing an
#   architecture mismatch or a silent miscalculation)
# ----------------------------------------------------------------------
CHECKPOINTS = [
    dict(
        name="A-Baseline (V3-t baseline)",
        path="../experiments/sprint/model_Sprint_Finalization_A-Baseline_V3-t.pth",
        capacity_mode="V3-t", use_depth_attn=False, film_mode="None", use_t_affine=False,
    ),
    dict(
        name="B-FiLMTAff (+FiLM+T-Affine)",
        path="../experiments/sprint/model_Sprint_Finalization_B-FiLMTAff_V3-t.pth",
        capacity_mode="V3-t", use_depth_attn=False, film_mode="PerBlock", use_t_affine=True,
    ),
    dict(
        name="C-Depth (+FiLM+T-Affine+DAF, no GAN)",
        path="../experiments/sprint/model_Sprint_Finalization_C-Depth_V3-t.pth",
        capacity_mode="V3-t", use_depth_attn=True, film_mode="PerBlock", use_t_affine=True,
    ),
    dict(
        name="D-Prop-S42 (Proposed, seed 42)",
        path="../experiments/sprint/model_Sprint_Finalization_D-Prop-S42-R1_V3-t.pth",
        capacity_mode="V3-t", use_depth_attn=True, film_mode="PerBlock", use_t_affine=True,
    ),
    dict(
        name="D-Prop-S43 (Proposed, seed 43)",
        path="../experiments/sprint/model_Sprint_Finalization_D-Prop-V3t-S43_V3-t.pth",
        capacity_mode="V3-t", use_depth_attn=True, film_mode="PerBlock", use_t_affine=True,
    ),
    dict(
        name="D-Prop-S44 (Proposed, seed 44)",
        path="../experiments/sprint/model_Sprint_Finalization_D-Prop-V3t-S44_V3-t.pth",
        capacity_mode="V3-t", use_depth_attn=True, film_mode="PerBlock", use_t_affine=True,
    ),
]

# To add extra checkpoints (e.g. DP-2's stop-gradient control, a
# separately-trained DAF-only version), append an entry in the same
# format above; the script will pick it up automatically.

DEFAULT_INDOOR_TXT = "SOTS/split_txt/indoor_test.txt"
DEFAULT_OUTDOOR_TXT = "SOTS/split_txt/outdoor_test.txt"


def load_pairs(txt_path):
    pairs = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hazy_path, gt_path = line.split()
            pairs.append((hazy_path, gt_path))
    return pairs


def read_pair(hazy_path, gt_path):
    I_np = np.asarray(Image.open(hazy_path).convert("RGB")).astype(np.float32) / 255.0
    J_np = np.asarray(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0
    return I_np, J_np


def build_model(cfg, device):
    model = N.PGDehazeNet(
        capacity_mode=cfg["capacity_mode"],
        use_depth_attn=cfg["use_depth_attn"],
        film_mode=cfg["film_mode"],
        use_t_affine=cfg["use_t_affine"],
    )
    state_dict = torch.load(cfg["path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def run_network(model, I_np, device):
    x = torch.from_numpy(I_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t, A, J = model(x)
    J_np = J.squeeze(0).permute(1, 2, 0).cpu().numpy()
    J_np = np.clip(J_np, 0.0, 1.0)
    return J_np


def evaluate_group(pairs, checkpoints, device, label):
    """Run CAP-baseline / corrected-ceiling / every checkpoint for one indoor or outdoor group."""
    per_image = []
    models = {c["name"]: build_model(c, device) for c in checkpoints}

    for hazy_path, gt_path in tqdm(pairs, desc=label):
        I_np, J_np = read_pair(hazy_path, gt_path)

        # ---- Physical quantity: CAP baseline (what the paper's Table 1 17.92/18.15 row actually measures) ----
        t_cap, A_cap, _ = P.run_cap(I_np)
        J_cap = P.recover_scene(I_np, t_cap, A_cap)
        psnr_cap = M.compute_psnr(J_cap, J_np)
        ssim_cap = float(M.compute_ssim(J_cap, J_np))

        # ---- Physical quantity: corrected ceiling (the quantity Theorem 1 / Appendix B.2 actually define) ----
        A_gt_smooth = P.estimate_A_from_pair(I_np, J_np, t_cap, smooth_scale=16, eps=1e-3)
        J_ceil = P.recover_scene(I_np, t_cap, A_gt_smooth)
        psnr_ceil = M.compute_psnr(J_ceil, J_np)
        ssim_ceil = float(M.compute_ssim(J_ceil, J_np))

        row = dict(hazy=hazy_path, psnr_cap_baseline=psnr_cap, ssim_cap_baseline=ssim_cap,
                   psnr_ceiling_corrected=psnr_ceil, ssim_ceiling_corrected=ssim_ceil)

        # ---- each trained checkpoint ----
        for name, model in models.items():
            J_pred = run_network(model, I_np, device)
            row[f"psnr__{name}"] = M.compute_psnr(J_pred, J_np)
            row[f"ssim__{name}"] = float(M.compute_ssim(J_pred, J_np))

        per_image.append(row)

    # ---- aggregate mean + std + 95% CI half-width ----
    def agg(key):
        vals = [r[key] for r in per_image if math.isfinite(r[key])]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
        ci95 = 1.96 * std / math.sqrt(n) if n > 0 else float("nan")
        return dict(mean=float(mean), std=float(std), n=n, ci95_halfwidth=float(ci95))

    summary = {"n_images": len(pairs)}
    summary["cap_baseline"] = dict(psnr=agg("psnr_cap_baseline"), ssim=agg("ssim_cap_baseline"))
    summary["ceiling_corrected"] = dict(psnr=agg("psnr_ceiling_corrected"), ssim=agg("ssim_ceiling_corrected"))
    for name in models:
        summary[name] = dict(psnr=agg(f"psnr__{name}"), ssim=agg(f"ssim__{name}"))

    return summary, per_image


def fmt(agg):
    return f"{agg['mean']:.3f} +/- {agg['std']:.3f} (n={agg['n']}, 95% CI half-width={agg['ci95_halfwidth']:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indoor_txt", default=DEFAULT_INDOOR_TXT)
    ap.add_argument("--outdoor_txt", default=DEFAULT_OUTDOOR_TXT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="ground_truth_summary.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    indoor_pairs = load_pairs(args.indoor_txt)
    outdoor_pairs = load_pairs(args.outdoor_txt)
    print(f"Indoor test images: {len(indoor_pairs)} pairs (from {args.indoor_txt})")
    print(f"Outdoor test images: {len(outdoor_pairs)} pairs (from {args.outdoor_txt})")
    print("If these counts are still 35 / 74, this is the internal small split, not the official SOTS --")
    print("consider also pointing --indoor_txt / --outdoor_txt at the full official SOTS (500 each) for comparison.\n")

    results = {}
    for label, pairs in [("indoor", indoor_pairs), ("outdoor", outdoor_pairs)]:
        print(f"===== Evaluating {label} ({len(pairs)} images) =====")
        summary, per_image = evaluate_group(pairs, CHECKPOINTS, device, label)
        results[label] = summary

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {args.out}\n")

    # ---- print a Markdown table ready to paste into the paper ----
    print("### Table: CAP baseline / corrected ceiling / each checkpoint  (indoor | outdoor)\n")
    print("| Config | Indoor PSNR | Indoor SSIM | Outdoor PSNR | Outdoor SSIM |")
    print("|---|---|---|---|---|")
    rows_order = ["cap_baseline", "ceiling_corrected"] + [c["name"] for c in CHECKPOINTS]
    for key in rows_order:
        ip = results["indoor"][key]["psnr"]
        is_ = results["indoor"][key]["ssim"]
        op = results["outdoor"][key]["psnr"]
        os_ = results["outdoor"][key]["ssim"]
        print(f"| {key} | {fmt(ip)} | {fmt(is_)} | {fmt(op)} | {fmt(os_)} |")

    print("\n[Interpretation notes]")
    print("- If the 'cap_baseline' row is close to the paper's current Table 1 17.92/18.15,")
    print("  that confirms the paper's 'ceiling' currently measures CAP's own blind estimate,")
    print("  not the quantity Theorem 1 defines -- replace that Table 1 row with the")
    print("  'ceiling_corrected' numbers and update the text per the revised Theorem 1 wording.")
    print("- If a checkpoint's psnr matches the historical values in expressiveness_*.json but")
    print("  not data_summary.json (or vice versa), that confirms which of the two old pipelines")
    print("  was more reliable; use this script's results going forward and retire both old paths.")


if __name__ == "__main__":
    main()
