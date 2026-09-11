#!/usr/bin/env python3
"""
render_tables.py
==========================
Generates LaTeX tables for unreported exp4 results and computes t0 truncation stats.
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_repo_on_path, repo_path, default_split_paths

def generate_scale_sensitivity_table(json_data, out_path):
    if "smoothing_scale_sensitivity" not in json_data:
        print("JSON missing 'smoothing_scale_sensitivity', generating...")
        from ceiling_error_control import run_smoothing_scale_sensitivity
        scale_rows = run_smoothing_scale_sensitivity(n_scenes=3)
        json_data["smoothing_scale_sensitivity"] = scale_rows

    data = json_data["smoothing_scale_sensitivity"]
    
    by_s = {}
    for r in data:
        s = r["smooth_scale_s"]
        kind = r["error_kind"]
        if s not in by_s:
            by_s[s] = {"rough_noise": [], "constant_bias": []}
        by_s[s][kind].append(r["ceiling_psnr_db"])
        
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{7}{c}{Smoothing window size $s$} \\")
    lines.append(r"\cmidrule{2-8}")
    
    s_vals = sorted(by_s.keys())
    header = "Error kind & " + " & ".join([str(s) for s in s_vals]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    for kind, display_name in [("rough_noise", "Rough noise"), ("constant_bias", "Constant bias")]:
        row = [display_name]
        for s in s_vals:
            val = np.mean(by_s[s][kind])
            row.append(f"{val:.2f}")
        lines.append(" & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")

def generate_error_decomp_table(json_data, out_path):
    if "real_data" not in json_data:
        print("JSON missing 'real_data', generating...")
        from ceiling_error_control import run_real_data_check
        real_results = run_real_data_check(indoor_n=35, outdoor_n=74, smooth_scale=16)
        json_data["real_data"] = real_results

    data = json_data["real_data"]
    
    lines = []
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Domain & High-frequency RMS & Low-freq spatial dev \\")
    lines.append(r"\midrule")
    
    for domain in ["indoor", "outdoor"]:
        if domain in data and "error" not in data[domain]:
            hf = data[domain]["mean_high_freq_rms"]
            lf = data[domain]["mean_low_freq_spatial_dev"]
            lines.append(f"{domain.capitalize()} & {hf:.4f} & {lf:.4f} \\\\")
            
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")

def compute_t0_truncation_stats(repo_root):
    import physics as P
    from PIL import Image
    splits = default_split_paths()
    stats = {}
    for domain in ["indoor", "outdoor"]:
        split_path = splits[domain]
        if not os.path.exists(split_path):
            print(f"Warning: split file not found: {split_path}")
            continue
        with open(split_path) as f:
            pairs = [l.split() for l in f.read().strip().split("\n") if l.strip()]
        
        fracs = []
        for hazy_path, _ in pairs:
            I = np.asarray(Image.open(hazy_path).convert("RGB")).astype(np.float32) / 255.0
            t_cap, _, _ = P.run_cap(I)
            trunc_frac = float(np.mean(t_cap < 0.1))
            fracs.append(trunc_frac)
        
        stats[f"{domain}_frac_pct"] = round(float(np.mean(fracs)) * 100, 4) if fracs else 0.0
        
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-exp4-json", required=False, default=None)
    ap.add_argument("--real-data", action="store_true")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    
    ensure_repo_on_path(args.repo_root)
    # Match the output dir structure of other experiments
    out_dir = repo_path("outputs", "exp7")
    os.makedirs(out_dir, exist_ok=True)
    
    data = {}
    if args.from_exp4_json and os.path.exists(args.from_exp4_json):
        with open(args.from_exp4_json, "r") as f:
            data = json.load(f)
            
    generate_scale_sensitivity_table(data, os.path.join(out_dir, "scale_sensitivity_table.tex"))
    generate_error_decomp_table(data, os.path.join(out_dir, "error_decomp_table.tex"))
    
    if args.real_data:
        stats = compute_t0_truncation_stats(args.repo_root)
        stats_path = os.path.join(out_dir, "t0_truncation_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved {stats_path}")

if __name__ == "__main__":
    main()
