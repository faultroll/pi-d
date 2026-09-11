#!/usr/bin/env python3
"""
ceiling_error_control.py
==============================
"Must" experiment #4 (review Section 3.2 / Section 6 item 4): a controlled
synthetic experiment showing
    that the ASM-inversion ceiling f_ceil (Definition 3) is sensitive to the
    *spatial-frequency composition* of the prior's error, not just its
    magnitude -- so the ceiling cannot be read as a scalar "prior quality"
    score.

What this reproduces (and generalises) from the advisor's manual check:
    Same RMSE, three error *types* -> very different ceiling PSNR:
        no error                        -> ~54 dB
        rough iid noise,  RMSE=0.05      -> ~27 dB
        smooth (constant) bias, RMSE=0.05 -> ~40 dB
    plus the fully degenerate constant-image case from Appendix B.2
    (three different t_hat all reconstruct J_gt exactly once A_inv^S
    collapses to a fixed point of the smoothing operator).

Uses physics.py's OWN Definition-2 / Definition-3 implementation
(estimate_A_from_pair, recover_scene) so results are directly comparable to
whatever produced Table 1 / Table 3 / Figure 3 in the manuscript -- nothing
here is reimplemented from scratch.

No GPU, no dataset, no checkpoint needed. Runs in well under a minute.

Usage:
    cd pgdehazenet/
    python experiments/ceiling_error_control.py
    python experiments/ceiling_error_control.py --output outputs/exp4_ceiling
    python experiments/ceiling_error_control.py --real-data \\
        --indoor-n 8 --outdoor-n 8   # optional: also run on real SOTS test images
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_repo_on_path, repo_path, default_split_paths


# =======================================================================
# 1. Synthetic scene generator
# =======================================================================
def make_synthetic_scene(seed, size=256, depth_kind="smooth"):
    """Build one synthetic (J_gt, t_true, A_true, I) scene satisfying the
    ASM exactly, so any ceiling gap we later measure is attributable purely
    to the *injected* transmission error, not to some other unmodelled
    source of mismatch.

    depth_kind:
      "smooth"  -- t varies as a smooth low-frequency depth field
                   (the normal, realistic case)
      "flat"    -- t is spatially constant (reproduces the Appendix B.2
                   degenerate-identity edge case exactly)
    """
    rng = np.random.RandomState(seed)
    H = W = size

    # J_gt: a smoothish "clean scene" with some texture, values in [0.05,0.95]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / size
    base = 0.5 + 0.25 * np.sin(3 * np.pi * xx + seed) * np.cos(2 * np.pi * yy + seed)
    texture = rng.rand(H, W) * 0.15 - 0.075
    J = np.stack([
        np.clip(base + texture + 0.05 * rng.randn(H, W), 0.05, 0.95),
        np.clip(base + texture * 0.8 + 0.05 * rng.randn(H, W), 0.05, 0.95),
        np.clip(base + texture * 1.2 + 0.05 * rng.randn(H, W), 0.05, 0.95),
    ], axis=-1).astype(np.float32)

    # t_true: smooth low-frequency "depth" field in [0.25, 0.9], or flat
    if depth_kind == "flat":
        t_true = np.full((H, W), 0.55, dtype=np.float32)
    else:
        depth = 0.5 + 0.3 * np.sin(2 * np.pi * (xx * 0.7 + yy * 0.4) + seed * 0.7)
        depth = depth + 0.1 * np.sin(5 * np.pi * xx + seed)
        t_true = np.clip(depth, 0.25, 0.9).astype(np.float32)

    # A_true: near-white atmospheric light, mildly scene-dependent
    A_true = np.array([0.90, 0.92, 0.88], dtype=np.float32) + 0.02 * (seed % 3 - 1)
    A_true = np.clip(A_true, 0.7, 1.0)
    A_field = np.broadcast_to(A_true, (H, W, 3)).astype(np.float32)

    # forward ASM synthesis (paper's Remark 5 methodology)
    t3 = t_true[..., None]
    I = t3 * J + (1 - t3) * A_field
    I = np.clip(I, 0, 1).astype(np.float32)
    return J, t_true, A_field, I


# =======================================================================
# 2. Controlled transmission-error injection, matched RMSE across types
# =======================================================================
def inject_error(t_true, kind, rmse, seed):
    """Return t_hat = t_true + error, where `error`'s RMS magnitude is
    forced to exactly `rmse` regardless of `kind`, so ceiling differences
    across kinds at fixed rmse isolate the effect of error *shape* alone."""
    rng = np.random.RandomState(seed + hash(kind) % 10000)
    H, W = t_true.shape

    if kind == "none":
        raw = np.zeros((H, W), dtype=np.float32)
    elif kind == "rough_noise":
        raw = rng.randn(H, W).astype(np.float32)  # i.i.d. -> pure high frequency
    elif kind == "constant_bias":
        raw = np.ones((H, W), dtype=np.float32)  # DC-only -> pure low frequency
    elif kind == "low_freq_drift":
        # smooth but non-constant -- half-cycle sinusoidal drift across the
        # frame; still almost entirely below the s=16 smoothing cutoff, but
        # NOT a fixed point of it, distinguishing this from "constant_bias"
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / max(H, W)
        raw = np.sin(np.pi * xx) * np.cos(np.pi * yy)
    else:
        raise ValueError(kind)

    if kind != "none":
        cur_rms = np.sqrt(np.mean(raw ** 2)) + 1e-12
        raw = raw / cur_rms * rmse

    t_hat = np.clip(t_true + raw, 0.05, 1.0).astype(np.float32)
    achieved_rmse = float(np.sqrt(np.mean((t_hat - t_true) ** 2)))
    return t_hat, achieved_rmse


# =======================================================================
# 3. Run the controlled sweep
# =======================================================================
def run_synthetic_sweep(n_scenes=6, magnitudes=(0.0, 0.05, 0.10, 0.20), smooth_scale=16, size=256):
    import physics as P
    from metrics import compute_psnr

    error_kinds = ["rough_noise", "constant_bias", "low_freq_drift"]
    rows = []
    for scene_idx in range(n_scenes):
        J, t_true, A_true, I = make_synthetic_scene(seed=scene_idx, size=size)
        for mag in magnitudes:
            kinds = ["none"] if mag == 0.0 else error_kinds
            for kind in kinds:
                t_hat, achieved_rmse = inject_error(t_true, kind, mag, seed=scene_idx)
                A_inv_S = P.estimate_A_from_pair(I, J, t_hat, smooth_scale=smooth_scale)
                fceil = P.recover_scene(I, t_hat, A_inv_S, t0=0.1)
                ceiling_psnr = float(compute_psnr(fceil, J))
                rows.append({
                    "scene": scene_idx, "error_kind": kind, "target_rmse": mag,
                    "achieved_rmse": round(achieved_rmse, 5),
                    "ceiling_psnr_db": round(ceiling_psnr, 2),
                })
    return rows


def run_degenerate_case_check():
    """Reproduces Appendix B.2's constructive proof numerically: a constant
    image where three different (t_hat, implied A) pairs all satisfy the ASM
    exactly and all reconstruct J_gt via f_ceil regardless of prior "quality"."""
    import physics as P
    from metrics import compute_psnr

    H = W = 64
    J_gt_val, I_val = 0.2, 0.5
    J = np.full((H, W, 3), J_gt_val, dtype=np.float32)
    I = np.full((H, W, 3), I_val, dtype=np.float32)

    rows = []
    for t_val, A_val in [(0.625, 1.0), (0.5, 0.8), (0.4, 0.7)]:
        # verify the (t, A) pair actually satisfies the ASM for this I, J
        asm_residual = abs((t_val * J_gt_val + (1 - t_val) * A_val) - I_val)
        t_hat = np.full((H, W), t_val, dtype=np.float32)
        A_inv_S = P.estimate_A_from_pair(I, J, t_hat, smooth_scale=16)
        fceil = P.recover_scene(I, t_hat, A_inv_S, t0=0.1)
        rows.append({
            "t_hat": t_val, "implied_A": A_val,
            "asm_residual": round(float(asm_residual), 6),
            "A_inv_S_value": round(float(A_inv_S.mean()), 4),
            "fceil_value": round(float(fceil.mean()), 4),
            "ceiling_psnr_db": round(float(compute_psnr(fceil, J)), 2),
        })
    return rows


def run_smoothing_scale_sensitivity(scales=(1, 2, 4, 8, 16, 32, 64), n_scenes=3, rmse=0.10):
    """Bonus (review Section 3.2, remediation item #4): sensitivity of the ceiling to the s
    (smoothing window) hyper-parameter, at fixed injected error."""
    import physics as P
    from metrics import compute_psnr

    rows = []
    for scene_idx in range(n_scenes):
        J, t_true, A_true, I = make_synthetic_scene(seed=scene_idx)
        t_hat_noise, _ = inject_error(t_true, "rough_noise", rmse, seed=scene_idx)
        t_hat_bias, _ = inject_error(t_true, "constant_bias", rmse, seed=scene_idx)
        for s in scales:
            for kind, t_hat in [("rough_noise", t_hat_noise), ("constant_bias", t_hat_bias)]:
                A_inv_S = P.estimate_A_from_pair(I, J, t_hat, smooth_scale=s)
                fceil = P.recover_scene(I, t_hat, A_inv_S, t0=0.1)
                rows.append({
                    "scene": scene_idx, "smooth_scale_s": s, "error_kind": kind,
                    "ceiling_psnr_db": round(float(compute_psnr(fceil, J)), 2),
                })
    return rows


# =======================================================================
# 4. Optional: same idea, but on real (I, J_gt) pairs + real tCAP error,
#    split by domain -- directly addresses the review's closing question:
#    does the 4.29 dB indoor/outdoor ceiling gap reflect a genuinely worse
#    indoor prior, or just a rougher (higher-frequency) indoor error?
#    (item 5 in the review's remediation list for Section 3.2).
# =======================================================================
def run_real_data_check(indoor_n=8, outdoor_n=8, smooth_scale=16, seed=0):
    import physics as P
    from metrics import compute_psnr
    from PIL import Image

    splits = default_split_paths()
    out = {}
    for domain, n in [("indoor", indoor_n), ("outdoor", outdoor_n)]:
        split_path = splits[domain]
        if not os.path.exists(split_path):
            out[domain] = {"error": f"split file not found: {split_path} "
                                     f"(run this on a machine with SOTS/ populated)"}
            continue
        with open(split_path) as f:
            pairs = [l.split() for l in f.read().strip().split("\n") if l.strip()]
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)

        rows = []
        for i in idx:
            hazy_path, gt_path = pairs[i]
            I = np.asarray(Image.open(hazy_path).convert("RGB")).astype(np.float32) / 255.0
            J = np.asarray(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0
            t_cap, A_cap, _ = P.run_cap(I)

            # Decompose the REAL tCAP error's spatial-frequency content by
            # comparing full-res tCAP against its own s=16 block-smoothed
            # version: the low-frequency part is what the ceiling can't see
            # regardless of how wrong it is (per Remark 3 / Appendix B.2).
            import cv2
            h, w = t_cap.shape
            small = cv2.resize(t_cap, (max(1, w // smooth_scale), max(1, h // smooth_scale)),
                                interpolation=cv2.INTER_AREA)
            t_cap_lowfreq = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            high_freq_energy = float(np.sqrt(np.mean((t_cap - t_cap_lowfreq) ** 2)))
            low_freq_level = float(np.mean(np.abs(t_cap_lowfreq - np.mean(t_cap_lowfreq))))

            A_inv_S = P.estimate_A_from_pair(I, J, t_cap, smooth_scale=smooth_scale)
            fceil = P.recover_scene(I, t_cap, A_inv_S, t0=0.1)
            rows.append({
                "hazy_path": hazy_path,
                "ceiling_psnr_db": round(float(compute_psnr(fceil, J)), 2),
                "tcap_high_freq_rms": round(high_freq_energy, 4),
                "tcap_low_freq_spatial_dev": round(low_freq_level, 4),
            })
        out[domain] = {
            "n": len(rows),
            "mean_ceiling_psnr": round(float(np.mean([r["ceiling_psnr_db"] for r in rows])), 2),
            "mean_high_freq_rms": round(float(np.mean([r["tcap_high_freq_rms"] for r in rows])), 4),
            "mean_low_freq_spatial_dev": round(float(np.mean([r["tcap_low_freq_spatial_dev"] for r in rows])), 4),
            "samples": rows,
        }
    return out


# =======================================================================
# 5. Report
# =======================================================================
def print_sweep_table(rows):
    print(f"\n{'scene':>5} {'error_kind':<16} {'target_rmse':>11} {'achieved_rmse':>13} {'ceiling_psnr':>13}")
    print("-" * 62)
    for r in rows:
        print(f"{r['scene']:>5} {r['error_kind']:<16} {r['target_rmse']:>11.3f} "
              f"{r['achieved_rmse']:>13.4f} {r['ceiling_psnr_db']:>11.2f} dB")

    print("\nMean ceiling PSNR by (error_kind, target_rmse), averaged over scenes:")
    by_key = {}
    for r in rows:
        key = (r["error_kind"], r["target_rmse"])
        by_key.setdefault(key, []).append(r["ceiling_psnr_db"])
    print(f"  {'error_kind':<16} {'rmse':>6}   {'mean_ceiling_psnr':>18}   n")
    for (kind, rmse), vals in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  {kind:<16} {rmse:>6.2f}   {np.mean(vals):>15.2f} dB   {len(vals)}")

    # headline spread at each matched rmse (excludes rmse=0.0 "none" row)
    print("\nSpread across error TYPES at matched RMSE (this is the number that")
    print("goes in the paper -- same accuracy, different ceiling):")
    for mag in sorted(set(r["target_rmse"] for r in rows if r["target_rmse"] > 0)):
        vals = {r["error_kind"]: r["ceiling_psnr_db"] for r in rows if r["target_rmse"] == mag}
        means = {}
        for k in set(r["error_kind"] for r in rows if r["target_rmse"] == mag):
            vv = [r["ceiling_psnr_db"] for r in rows if r["target_rmse"] == mag and r["error_kind"] == k]
            means[k] = float(np.mean(vv))
        spread = max(means.values()) - min(means.values())
        detail = ", ".join(f"{k}={v:.2f}dB" for k, v in sorted(means.items(), key=lambda kv: -kv[1]))
        print(f"  rmse={mag:.2f}: spread={spread:.2f} dB   ({detail})")


def print_degenerate_table(rows):
    print("\nConstant-image degenerate case (Appendix B.2, J_gt=0.2, I=0.5):")
    print(f"  {'t_hat':>7} {'implied_A':>10} {'ASM_residual':>13} {'A_inv_S':>9} {'fceil':>7} {'ceiling_psnr':>13}")
    for r in rows:
        print(f"  {r['t_hat']:>7.3f} {r['implied_A']:>10.3f} {r['asm_residual']:>13.6f} "
              f"{r['A_inv_S_value']:>9.4f} {r['fceil_value']:>7.4f} {r['ceiling_psnr_db']:>11.2f} dB")
    print("  -> if all three rows show ~0 residual and near-identical high ceiling_psnr,")
    print("     this numerically confirms Appendix B.2: the ceiling is independent of")
    print("     t_hat's accuracy whenever the induced A error is spatially smooth.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--smooth-scale", type=int, default=16, help="s in Definition 2 (paper uses 16)")
    ap.add_argument("--size", type=int, default=256, help="synthetic image side length")
    ap.add_argument("--skip-scale-sensitivity", action="store_true")
    ap.add_argument("--real-data", action="store_true",
                     help="also run the same diagnostic on real SOTS test images (needs SOTS/ populated)")
    ap.add_argument("--indoor-n", type=int, default=8)
    ap.add_argument("--outdoor-n", type=int, default=8)
    ap.add_argument("--output", type=str, default=None,
                     help="output dir for JSON results (default: <repo>/outputs/exp4_ceiling_error_control)")
    args = ap.parse_args()

    ensure_repo_on_path(args.repo_root)
    out_dir = args.output or repo_path("outputs", "exp4_ceiling_error_control")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("Experiment 4: ceiling analysis -- controlled error-type sweep")
    print("(review Section 3.2 / Appendix B.2; synthetic ground truth, no GPU)")
    print("=" * 72)

    sweep_rows = run_synthetic_sweep(n_scenes=args.n_scenes, magnitudes=tuple(args.magnitudes),
                                      smooth_scale=args.smooth_scale, size=args.size)
    print_sweep_table(sweep_rows)

    degen_rows = run_degenerate_case_check()
    print_degenerate_table(degen_rows)

    results = {
        "smooth_scale_s": args.smooth_scale,
        "sweep": sweep_rows,
        "degenerate_case": degen_rows,
    }

    if not args.skip_scale_sensitivity:
        print("\nComputing s (smoothing window) sensitivity curve ...")
        scale_rows = run_smoothing_scale_sensitivity(n_scenes=3)
        results["smoothing_scale_sensitivity"] = scale_rows
        print(f"  {'s':>4} {'error_kind':<16} {'mean_ceiling_psnr':>18}")
        by_s = {}
        for r in scale_rows:
            by_s.setdefault((r["smooth_scale_s"], r["error_kind"]), []).append(r["ceiling_psnr_db"])
        for (s, kind), vals in sorted(by_s.items()):
            print(f"  {s:>4} {kind:<16} {np.mean(vals):>15.2f} dB")

    if args.real_data:
        print("\nRunning real-data indoor vs outdoor tCAP error decomposition ...")
        real_results = run_real_data_check(indoor_n=args.indoor_n, outdoor_n=args.outdoor_n,
                                            smooth_scale=args.smooth_scale)
        results["real_data"] = real_results
        for domain, d in real_results.items():
            if "error" in d:
                print(f"  {domain}: {d['error']}")
            else:
                print(f"  {domain}: N={d['n']}  mean_ceiling_psnr={d['mean_ceiling_psnr']:.2f} dB  "
                      f"mean_high_freq_rms={d['mean_high_freq_rms']:.4f}  "
                      f"mean_low_freq_spatial_dev={d['mean_low_freq_spatial_dev']:.4f}")
        if all("error" not in d for d in real_results.values()):
            print("  -> compare mean_high_freq_rms and mean_low_freq_spatial_dev between domains:")
            print("     if indoor's tCAP error is more HIGH-frequency (not just larger) than")
            print("     outdoor's, that is direct evidence the 4.29 dB indoor/outdoor ceiling")
            print("     gap reflects error SHAPE, not only error magnitude.")

    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results -> {out_json}")


if __name__ == "__main__":
    main()
