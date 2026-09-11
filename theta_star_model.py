#!/usr/bin/env python3
"""theta_star_model.py - Model-Centric Diagnostic for A-Branch & J-Branch Compensation State

Two modes:
  --mode model   Aggregate JSON diagnostic across the entire val set.
  --mode samples Per-sample CSV + summary JSON for correlation analysis.

Usage:
    python theta_star_model.py --checkpoint ./outputs/.../best_model.pth --mode model
    python theta_star_model.py --checkpoint ./outputs/.../best_model.pth --mode samples
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import network as N
from PIL import Image
import physics as P
from metrics import compute_psnr


# ===============================================================
# Dataset (self-contained, mirrors DehazeDataset from train.py)
# ===============================================================
from train import DehazeDataset


# ===============================================================
# Helpers
# ===============================================================
def get_sample_type(path):
    if "indoor" in path:
        return "indoor"
    if "outdoor" in path:
        return "outdoor"
    return "unknown"


def physics_recon(I, t, A):
    """Strict ASM physics: J = (I - A*(1-t)) / t, clipped to [0,1]."""
    t = np.clip(t, 0.1, 1.0)
    t3 = np.repeat(t[None, ...] if t.ndim == 2 else t, 3, axis=0)
    A3 = A if A.ndim == 3 else A[:, None, None]
    recon = (I - A3 * (1.0 - t3)) / t3
    return np.clip(recon, 0.0, 1.0)


def solve_affine_per_image(I_np, A_np, J_np):
    """Return (w1, w2, b), each a length-3 numpy vector (one value per color channel).
    Solves  J = (I-A)*w1 + A*w2 + b  via per-channel least squares over all pixels.
    """
    C = I_np.shape[0]
    w1c, w2c, bc = [], [], []
    for c in range(C):
        x = I_np[c].flatten()
        a = A_np[c].flatten()
        y = J_np[c].flatten()
        ones = np.ones_like(x)
        X = np.stack([x - a, a, ones], axis=1)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        w1c.append(float(beta[0]))
        w2c.append(float(beta[1]))
        bc.append(float(beta[2]))
    return np.array(w1c), np.array(w2c), np.array(bc)


# ===============================================================
# Config inference from checkpoint path
# ===============================================================
SUITE_EXP_LOOKUP = {
    "E0-Base":      ("M0", "global"),
    "E1-Loss":      ("M0", "spatial"),
    "E2-Pool":      ("M1", "spatial"),
    "E3-Dynamic":   ("M2", "spatial"),
    "E4-Complete":  ("M3", "spatial"),
    "E6-GlobalDyn": ("M2", "global"),
    "E7-Hybrid":    ("M4", "spatial"),
}

DEFAULT_STRUCTURE = {
    "use_depth_attn": True,
    "channel_attn_mode": "tj",
    "dilation_mode": "None",
    "film_mode": "PerBlock",
    "use_t_affine": True,
    "use_refine": False,
    "use_repconv": False,
    "use_t_gate": False,
    "use_gan": True,
}


def _infer_config(ckpt_path):
    name = Path(ckpt_path).name
    # sanitise common prefixes/suffixes
    for prefix in ["best_model_", "model_", "expressiveness_", "performance_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in [".pth", ".onnx", ".json", "_re.json"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    print(f"_infer_config, name: {name}")

    parts = name.split("_")
    print(f"_infer_config, parts: {parts}")
    if len(parts) >= 3:
        mode_candidate = parts[-1]
        print(f"_infer_config, mode_candidate: {mode_candidate}")
        if mode_candidate in {
            "V0", "V1", "V2", "V3", "V4", "V5",
            "V3-t", "V4-t", "V5-t",
            "V3-A", "V4-A", "V5-A",
        }:
            exp_candidate = "_".join(parts[1:-1])
            print(f"_infer_config, exp_candidate: {exp_candidate}")
            a_net, a_gt = SUITE_EXP_LOOKUP.get(exp_candidate, ("M0", "global"))
            cfg = {
                "capacity_mode": mode_candidate,
                "a_net_mode": a_net,
                "a_gt_mode": a_gt,
            }
            if "-ABnd" in name or "-AB-" in name:
                cfg["use_a_bound"] = True
            return cfg

    # fallback: use parent directory name (e.g. Ablation_Study_E1-Loss_V3 or E7-V3)
    parent_name = Path(ckpt_path).parent.name

    def _probe_expr_token(name):
        m = re.search(r"(E\d+-[A-Za-z]+(?:[-_][A-Za-z]+)*)", name)
        return m.group(1).replace("_", "-") if m else None

    probe_names = [Path(ckpt_path).name, parent_name]
    p = Path(ckpt_path).parent
    for _ in range(4):
        p = p.parent
        if not p or p.name in {"", ".", "outputs", "A_spatial1"}:
            break
        probe_names.append(p.name)

    expr_token = None
    for name in probe_names:
        expr_token = _probe_expr_token(name)
        if expr_token and expr_token in SUITE_EXP_LOOKUP:
            break
        expr_token = None

    if expr_token:
        a_net, a_gt = SUITE_EXP_LOOKUP[expr_token]
        cfg = {
            "capacity_mode": mode_candidate,
            "a_net_mode": a_net,
            "a_gt_mode": a_gt,
        }
        if "-ABnd" in name or "-AB-" in name or "-ABnd" in parent_name or "-AB-" in parent_name:
            cfg["use_a_bound"] = True
        return cfg

    parts = parent_name.split("_")
    if len(parts) >= 3:
        mode_candidate = parts[-1]
        exp_candidate = "_".join(parts[1:-1])
        a_net, a_gt = SUITE_EXP_LOOKUP.get(exp_candidate, ("M0", "global"))
        cfg = {
            "capacity_mode": mode_candidate,
            "a_net_mode": a_net,
            "a_gt_mode": a_gt,
        }
        if "-ABnd" in parent_name or "-AB-" in parent_name:
            cfg["use_a_bound"] = True
        return cfg

    return {
        "capacity_mode": "V3",
        "a_net_mode": "M0",
        "a_gt_mode": "global",
    }


def load_checkpoint(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def infer_a_branch_topology(state):
    keys = set(state.keys())
    a_keys = [k for k in keys if k.startswith("A_branch.")]
    has_gate = any("A_branch.gate." in k for k in a_keys)
    has_decoder = any("A_branch.decoder." in k for k in a_keys)
    has_pool_conv = any("A_branch.pool_conv" in k for k in a_keys)
    enc_candidates = [k for k in a_keys if k.endswith("encoder.0.weight")]
    encoder_ks = int(state[enc_candidates[0]].shape[2]) if enc_candidates else None

    if has_gate or has_decoder:
        return "M4"
    if has_pool_conv and encoder_ks == 3:
        return "M3"
    if has_pool_conv and encoder_ks == 1:
        return "M1_or_M2"
    return "M0"


def infer_structure_flags(state):
    """Probe checkpoint keys to determine which structural modules were present during training.

    Returns a dict that should be merged into cfg before build_model() so that
    model construction exactly matches the originally trained architecture.
    """
    keys = set(state.keys())
    flags = {}

    # Channel attention branches (check for any key under attn.fc)
    has_t_attn = any(k.startswith("t_branch.attn.") for k in keys)
    has_j_attn = any(k.startswith("j_branch.attn.") for k in keys)
    if has_t_attn and has_j_attn:
        flags["channel_attn_mode"] = "tj"
    elif has_t_attn:
        flags["channel_attn_mode"] = "t"
    elif has_j_attn:
        flags["channel_attn_mode"] = "j"
    else:
        flags["channel_attn_mode"] = "None"

    # Depth attention: presence of depth_fuse.out_proj.weight implies use_depth_attn=True
    has_depth_attn = any(k.startswith("j_branch.depth_fuse.") for k in keys)
    flags["use_depth_attn"] = has_depth_attn

    # Dilation mode
    if any(k.startswith("j_branch.lightaspp.") for k in keys):
        flags["dilation_mode"] = "LightASPP"
    elif any(k.startswith("j_branch.block_group.") for k in keys):
        flags["dilation_mode"] = "None"
    else:
        flags["dilation_mode"] = "None"

    # FiLM / SFT modes
    has_t_encoder = any(k.startswith("j_branch.t_encoder.") for k in keys)
    has_sft = any(k.startswith("j_branch.sft.") for k in keys)
    if has_t_encoder and has_sft:
        flags["film_mode"] = "Both"
    elif has_t_encoder:
        flags["film_mode"] = "PerBlock"
    elif has_sft:
        flags["film_mode"] = "SFT"
    else:
        flags["film_mode"] = "None"

    # TAffine head vs plain conv head
    if any(k.startswith("j_branch.affine_head.t_encoder.") for k in keys):
        flags["use_t_affine"] = True
    else:
        flags["use_t_affine"] = False

    # T-Gate
    has_t_gate = any(k.startswith("j_branch.t_gate.") for k in keys)
    flags["use_t_gate"] = has_t_gate

    # Refine
    has_refine = any(k.startswith("j_branch.refine.")
                     for k in keys)
    flags["use_refine"] = has_refine

    # RepConv: RepConv2d introduces `repconv1.conv3x3` naming inside JBranch's block groups
    has_repconv = any("repconv" in k for k in keys)
    flags["use_repconv"] = has_repconv
    
    # VARC
    has_varc = any("varc_alpha" in k for k in keys)
    flags["use_varc"] = has_varc
    
    # A-Bound (only inferrable via decoder shape for hybrid, otherwise fallback to filename)
    has_a_bound_hybrid = any(k == "A_branch.decoder.weight" for k in keys)
    if has_a_bound_hybrid:
        flags["use_a_bound"] = True

    return flags


def build_model(cfg, device):
    return N.PGDehazeNet(
        capacity_mode=cfg.get("capacity_mode", "V3"),
        dilation_mode=cfg.get("dilation_mode", "None"),
        use_depth_attn=cfg.get("use_depth_attn", False),
        channel_attn_mode=cfg.get("channel_attn_mode", "None"),
        film_mode=cfg.get("film_mode", "None"),
        use_refine=cfg.get("use_refine", False),
        use_repconv=cfg.get("use_repconv", False),
        use_t_affine=cfg.get("use_t_affine", False),
        use_t_gate=cfg.get("use_t_gate", False),
        use_varc=cfg.get("use_varc", False),
        use_a_bound=cfg.get("use_a_bound", False),
        a_net_mode=cfg.get("a_net_mode", "M0"),
    ).to(device)


# ===============================================================
# Core: forward pass over val set -> per-sample records
# ===============================================================
@torch.no_grad()
def run_inference_and_collect(model, dataset, device, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    model.eval()
    records = []
    paths = dataset.get_paths()

    for idx, (I, t_gt, J_gt, A_gt) in enumerate(loader):
        I = I.to(device)
        t_pred, A_pred, J_pred = model(I)

        I_np     = I[0].cpu().numpy()      # [3, H, W]
        t_np     = t_pred[0].cpu().numpy() # [1, H, W]
        A_np     = A_pred[0].cpu().numpy() # [3, H, W]
        J_np     = J_pred[0].cpu().numpy() # [3, H, W]
        J_gt_np  = J_gt[0].cpu().numpy()
        t_gt_np  = t_gt[0].cpu().numpy()
        A_gt_np  = A_gt[0].cpu().numpy()   # [3] or [3,H,W]

        psnr_val = compute_psnr(J_np, J_gt_np)

        recon_pred = physics_recon(I_np, t_np[0], A_np)
        recon_gt   = physics_recon(I_np, t_gt_np[0], A_gt_np)
        psnr_recon_pred = compute_psnr(recon_pred, J_gt_np)
        psnr_recon_gt   = compute_psnr(recon_gt, J_gt_np)

        w1_ch, w2_ch, b_ch = solve_affine_per_image(I_np, A_np, J_np)

        cache_path = paths[idx] + ".cap.npz"
        if os.path.exists(cache_path):
            cap_data = np.load(cache_path, allow_pickle=False)
            A_cap_global = cap_data.get("A_global")
            A_inv_spatial = cap_data.get("A_spatial")
        else:
            A_cap_global = None
            A_inv_spatial = None

        if A_cap_global is None:
            A_cap_global = np.full(3, np.nan, dtype=np.float32)
        if A_inv_spatial is None:
            A_inv_spatial = np.full(
                (I_np.shape[1], I_np.shape[2], 3), np.nan, dtype=np.float32
            )

        records.append({
            "path": paths[idx],
            "type": get_sample_type(paths[idx]),
            "psnr": psnr_val,
            "w1_ch": w1_ch.tolist(),
            "w2_ch": w2_ch.tolist(),
            "b_ch":  b_ch.tolist(),
            "psnr_recon_pred": psnr_recon_pred,
            "psnr_recon_gt":   psnr_recon_gt,
            "A_pred": A_np,
            "A_cap_global": A_cap_global,
            "A_inv_spatial": A_inv_spatial,
        })

    return records


# ===============================================================
# Aggregation helpers
# ===============================================================
def _arr_stats(arr):
    arr = np.asarray(arr, dtype=np.float64).flatten()
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
    }


def agg_A_stats(records):
    # per-channel aggregated
    per_ch = []
    for c in range(3):
        ch_vals = np.concatenate([r["A_pred"][c].flatten() for r in records])
        per_ch.append({
            "mean": float(np.mean(ch_vals)),
            "std":  float(np.std(ch_vals)),
            "min":  float(np.min(ch_vals)),
            "max":  float(np.max(ch_vals)),
        })

    flat_r = np.concatenate([r["A_pred"][0].flatten() for r in records])
    flat_g = np.concatenate([r["A_pred"][1].flatten() for r in records])
    corr_rg, _ = pearsonr(flat_r, flat_g)
    saturation_ratio = float(
        np.mean(
            np.concatenate([r["A_pred"].flatten() for r in records]) > 0.99
        )
    )

    return {
        "a_pred_mean_rgb":        [per_ch[c]["mean"] for c in range(3)],
        "a_pred_std_rgb":         [per_ch[c]["std"]  for c in range(3)],
        "a_pred_min_rgb":         [per_ch[c]["min"]  for c in range(3)],
        "a_pred_max_rgb":         [per_ch[c]["max"]  for c in range(3)],
        "spatial_correlation_rg": float(corr_rg),
        "saturation_ratio":       saturation_ratio,
    }


def agg_affine_stats(records):
    w1_all = np.array([r["w1_ch"] for r in records])  # [N, 3]
    w2_all = np.array([r["w2_ch"] for r in records])
    b_all  = np.array([r["b_ch"]  for r in records])

    corr, _ = pearsonr(w1_all.flatten(), w2_all.flatten())

    return {
        "w1": _arr_stats(w1_all),
        "w2": _arr_stats(w2_all),
        "b":  _arr_stats(b_all),
        "w2_ratio_to_1": float(np.mean(np.abs(w2_all.flatten() - 1.0))),
        "w1_w2_correlation": float(corr),
    }


def agg_physics_psnr(records):
    model_psnr  = float(np.mean([r["psnr"] for r in records]))
    recon_psnr  = float(np.mean([r["psnr_recon_pred"] for r in records]))
    recon_gt_psnr = float(np.mean([r["psnr_recon_gt"] for r in records]))
    return {
        "using_A_pred_and_t_pred": recon_psnr,
        "using_A_gt_and_t_gt":     recon_gt_psnr,
        "gap": model_psnr - recon_psnr,
    }


def aggregate_by_group(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["type"]].append(r)

    result = {}
    for gname, grecords in sorted(groups.items()):
        result[gname] = {
            "n_samples": len(grecords),
            "mean_psnr": float(np.mean([r["psnr"] for r in grecords])),
            "A_branch_stats": agg_A_stats(grecords),
            "J_branch_affine_stats": agg_affine_stats(grecords),
            "physics_recon_psnr": agg_physics_psnr(grecords),
        }
    return result


# ===============================================================
# Mode: model   (aggregate JSON)
# ===============================================================
def run_model_mode(records, args):
    out = {
        "checkpoint": args.checkpoint,
        "n_samples": len(records),
        "overall_psnr": float(np.mean([r["psnr"] for r in records])),
        "A_branch_stats": agg_A_stats(records),
        "J_branch_affine_stats": agg_affine_stats(records),
        "physics_recon_psnr": agg_physics_psnr(records),
        "by_group": aggregate_by_group(records),
    }

    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
        print(f"\n[INFO] Saved -> {args.output}")
    return out


# ===============================================================
# Mode: samples  (CSV + summary JSON for correlations)
# ===============================================================
def run_samples_mode(records, args):
    json_path = args.output
    csv_path = str(Path(json_path).with_suffix(".csv"))

    fieldnames = [
        "sample_id",
        "type",
        "psnr",
        "a_cap_saturated",
        "a_cap_vs_inv_error_R",
        "a_cap_vs_inv_error_G",
        "a_cap_vs_inv_error_B",
        "w2_local_corrected_std",
        "jacobian_A_sensitivity",
    ]

    rows = []
    sens_list, sat_list = [], []
    err_r, err_g, err_b = [], [], []
    w2_std_list, psnr_list = [], []

    for r in records:
        ac   = r["A_cap_global"]
        ai   = r["A_inv_spatial"]
        err_rgb = np.abs(ac - ai.mean(axis=(0, 1))) if not np.all(np.isnan(ai)) else np.array([np.nan] * 3)
        w2_std = float(np.std(r["w2_ch"]))

        cap_bc = np.broadcast_to(ac[:, None, None], r["A_pred"].shape).copy()
        sensitivity = float(np.mean(np.abs(r["A_pred"] - cap_bc)))

        a_cap_saturated = int(np.all(ac > 0.99))

        rows.append({
            "sample_id":            Path(r["path"]).name,
            "type":                 r["type"],
            "psnr":                 round(r["psnr"], 4),
            "a_cap_saturated":      a_cap_saturated,
            "a_cap_vs_inv_error_R": round(float(err_rgb[0]), 6),
            "a_cap_vs_inv_error_G": round(float(err_rgb[1]), 6),
            "a_cap_vs_inv_error_B": round(float(err_rgb[2]), 6),
            "w2_local_corrected_std": round(w2_std, 6),
            "jacobian_A_sensitivity": round(sensitivity, 6),
        })

        sens_list.append(sensitivity)
        sat_list.append(a_cap_saturated)
        err_r.append(err_rgb[0])
        err_g.append(err_rgb[1])
        err_b.append(err_rgb[2])
        w2_std_list.append(w2_std)
        psnr_list.append(r["psnr"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def corr(xs, ys):
        if len(xs) < 2:
            return None
        r, _ = pearsonr(xs, ys)
        return None if np.isnan(r) else float(r)

    cr_sens = corr(sens_list, psnr_list)
    cr_sat  = corr(sat_list, psnr_list)
    cr_w2   = corr(w2_std_list, psnr_list)

    def interp(r):
        if r is None:
            return "Not enough samples"
        if r < -0.5:
            return "Strong negative correlation (expected)"
        if r > 0.5:
            return "Strong positive correlation"
        return "Weak / negligible correlation"

    summary = {
        "checkpoint": args.checkpoint,
        "csv_path": csv_path,
        "n_samples": len(rows),
        "indoor_n": sum(1 for r in rows if r["type"] == "indoor"),
        "outdoor_n": sum(1 for r in rows if r["type"] == "outdoor"),
        "mean_a_cap_vs_inv_error_rgb": [
            float(np.nanmean(err_r)),
            float(np.nanmean(err_g)),
            float(np.nanmean(err_b)),
        ],
        "correlations": {
            "a_cap_saturated_vs_psnr": {
                "pearson_r": cr_sat,
                "interpretation": interp(cr_sat),
            },
            "w2_local_corrected_std_vs_psnr": {
                "pearson_r": cr_w2,
                "interpretation": interp(cr_w2),
            },
            "jacobian_A_sensitivity_vs_psnr": {
                "pearson_r": cr_sens,
                "interpretation": interp(cr_sens),
            },
        },
    }

    out_text = json.dumps(summary, indent=2)
    print(out_text)
    Path(json_path).write_text(out_text)
    print(f"\n[INFO] Saved samples JSON -> {json_path}")
    print(f"[INFO] Saved samples CSV  -> {csv_path}")
    return summary


# ===============================================================
# CLI
# ===============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="theta_star_model.py - Model-Centric Diagnostic"
    )
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pth, saves G.state_dict())")
    ap.add_argument("--val-txt", type=str,
                    default="./SOTS/split_txt/val.txt",
                    help="Val split text file")
    ap.add_argument("--a-gt-mode", type=str, default=None,
                    choices=["global", "spatial"],
                    help="Override dataset A-GT mode (default: inferred from checkpoint)")
    ap.add_argument("--capacity-mode", type=str, default=None,
                    help="Override capacity mode (e.g., V3, V3-t)")
    ap.add_argument("--a-net-mode", type=str, default=None,
                    choices=["M0", "M1", "M2", "M3", "M4"],
                    help="Override A-net mode")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Val batch size (default 1)")
    ap.add_argument("--use-a-bound", action='store_true',
                    help="Override: Use A Bound constraint")
    ap.add_argument("--mode", type=str, default="model",
                    choices=["model", "samples"],
                    help="Diagnostic mode")
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON path (auto-derived from checkpoint dir if omitted)")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda or cpu (default: auto)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if str(device) == "cpu":
        print("[WARN] Running on CPU; inference may be slow.")

    state = load_checkpoint(args.checkpoint, device)
    flags_from_ckpt = infer_structure_flags(state)
    print(f"[INFO] Inferred structure flags from checkpoint: {flags_from_ckpt}")

    cfg = _infer_config(args.checkpoint)

    if args.capacity_mode:
        cfg["capacity_mode"] = args.capacity_mode
    if args.a_net_mode:
        cfg["a_net_mode"] = args.a_net_mode
    if args.a_gt_mode:
        cfg["a_gt_mode"] = args.a_gt_mode
    if args.use_a_bound:
        cfg["use_a_bound"] = True
        
    topology_keys = {"capacity_mode", "a_net_mode", "a_gt_mode"}
    for k, v in flags_from_ckpt.items():
        if k not in topology_keys:
            cfg[k] = v

    a_top = infer_a_branch_topology(state)
    print(f"[INFO] Inferred A-branch topology from checkpoint: {a_top}")
    if a_top == "M4":
        cfg["a_net_mode"] = "M4"
        cfg.setdefault("a_gt_mode", "spatial")
    elif a_top == "M3":
        cfg["a_net_mode"] = "M3"
    elif a_top == "M1_or_M2":
        if cfg.get("a_net_mode") not in {"M1", "M2"}:
            cfg["a_net_mode"] = "M1"
    elif a_top == "M0":
        cfg["a_net_mode"] = "M0"

    cfg.setdefault("use_gan", True)
    for k, v in DEFAULT_STRUCTURE.items():
        if k not in cfg:
            cfg[k] = v

    a_gt_mode = cfg.get("a_gt_mode", "global")
    print(f"[INFO] Final cfg: {cfg}")
    model = build_model(cfg, device)
    model.load_state_dict(state, strict=True)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    dataset = DehazeDataset(args.val_txt, a_gt_mode=a_gt_mode)
    print(f"[INFO] Val set size: {len(dataset)} samples (a_gt_mode={a_gt_mode})")

    if args.output is None:
        ckpt_dir = str(Path(args.checkpoint).parent)
        if args.mode == "model":
            args.output = os.path.join(ckpt_dir, "theta_star_model.json")
        else:
            args.output = os.path.join(ckpt_dir, "theta_star_model_samples.json")

    records = run_inference_and_collect(model, dataset, device, batch_size=args.batch_size)

    if args.mode == "model":
        run_model_mode(records, args)
    else:
        run_samples_mode(records, args)


if __name__ == "__main__":
    main()
