# Quick verification: compare CAP global A vs ASM-inversion spatial A
# This script visualizes the difference to validate the approach
# Also merged with deeper analysis of physics-based J reconstruction quality

import numpy as np
from PIL import Image
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import physics as P


def estimate_A_spatial_from_pair(I_np, J_np, t_np, smooth_scale=16, eps=1e-3):
    """
    ASM inversion: A(x,y) = (I - J_gt * t) / (1 - t + eps)
    Uses J_gt (real GT) + t_cap (pseudo-label) to get per-pixel A estimate.
    Then smooths to smooth_scale x smooth_scale to reduce noise.
    """
    H, W = I_np.shape[:2]
    t3 = np.repeat(t_np[..., None] if t_np.ndim == 2 else t_np, 3, axis=2)

    # ASM inversion
    A_raw = (I_np - J_np * t3) / (1.0 - t3 + eps)
    A_raw = np.clip(A_raw, 0.0, 1.0)

    # Smooth: downsample to grid then upsample
    import cv2
    small_h = max(1, H // smooth_scale)
    small_w = max(1, W // smooth_scale)
    A_small = cv2.resize(A_raw, (small_w, small_h), interpolation=cv2.INTER_AREA)
    A_smooth = cv2.resize(A_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return A_raw, A_smooth, A_small


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def reconstruct_J(I, t, A):
    """J = (I - A*(1-t)) / max(t, 0.1)"""
    t3 = np.repeat(t[..., None] if t.ndim == 2 else t, 3, axis=2)
    A3 = A if A.ndim == 3 else A[None, None, :]
    J = (I - A3 * (1.0 - t3)) / np.maximum(t3, 0.1)
    return np.clip(J, 0.0, 1.0)


def main():
    val_txt = './SOTS/split_txt/val.txt'
    lines = open(val_txt).read().strip().split('\n')

    indoor_lines = [l for l in lines if 'indoor' in l]
    outdoor_lines = [l for l in lines if 'outdoor' in l]

    print(f"Indoor samples: {len(indoor_lines)}, Outdoor samples: {len(outdoor_lines)}")

    results = []
    summary_stats = {}

    for label, sample_lines in [("indoor", indoor_lines), ("outdoor", outdoor_lines)]:
        psnr_cap_global = []
        psnr_inv_spatial = []
        psnr_inv_global = []
        a_cap_saturated = 0
        t_mean_list = []

        for line in tqdm(sample_lines, desc=f"Processing {label.upper()}"):
            hazy_path, gt_path = line.split()
            I_np = np.array(Image.open(hazy_path).convert('RGB')).astype(np.float32) / 255.0
            J_np = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0

            # Current method: CAP
            t_cap, A_cap, _ = P.run_cap(I_np)

            # Proposed: ASM inversion
            A_raw, A_smooth_16, A_small = estimate_A_spatial_from_pair(
                I_np, J_np, t_cap, smooth_scale=16
            )

            # Compare: global CAP A vs mean of spatial A
            A_spatial_mean = A_smooth_16.mean(axis=(0, 1))

            # Compute per-pixel deviation
            A_cap_expanded = A_cap[None, None, :]  # [1,1,3]
            deviation = np.abs(A_smooth_16 - A_cap_expanded)
            max_dev = deviation.max()
            mean_dev = deviation.mean()

            # Check A_raw quality (before smoothing)
            A_raw_std = A_raw.std(axis=(0, 1))

            # Reconstruction J for PSNR evaluation
            J_with_cap_global = reconstruct_J(I_np, t_cap, A_cap)
            J_with_inv_spatial = reconstruct_J(I_np, t_cap, A_smooth_16)
            J_with_inv_global = reconstruct_J(I_np, t_cap, A_spatial_mean)

            psnr_cap_global.append(psnr(J_with_cap_global, J_np))
            psnr_inv_spatial.append(psnr(J_with_inv_spatial, J_np))
            psnr_inv_global.append(psnr(J_with_inv_global, J_np))

            if np.all(A_cap > 0.99):
                a_cap_saturated += 1
            t_mean_list.append(t_cap.mean())

            results.append({
                'label': label,
                'file': os.path.basename(hazy_path),
                'A_cap': A_cap,
                'A_spatial_mean': A_spatial_mean,
                'A_diff': np.abs(A_cap - A_spatial_mean),
                'spatial_max_dev': max_dev,
                'spatial_mean_dev': mean_dev,
                'raw_std': A_raw_std,
            })

        summary_stats[label] = {
            'n': len(sample_lines),
            'a_cap_saturated': a_cap_saturated,
            't_mean': np.mean(t_mean_list),
            'psnr_cap_global': np.mean(psnr_cap_global),
            'psnr_inv_global': np.mean(psnr_inv_global),
            'psnr_inv_spatial': np.mean(psnr_inv_spatial)
        }

    # Print results
    print("\n" + "="*80)
    print("CAP Global A vs ASM-Inversion Spatial A Comparison")
    print("="*80)
    print(f"{'Type':<8} {'File':<25} {'A_cap(RGB)':<22} {'A_inv_mean(RGB)':<22} {'Diff(RGB)':<22} {'Spatial_dev':<12}")
    print("-"*110)

    for r in results:
        ac = f"({r['A_cap'][0]:.3f},{r['A_cap'][1]:.3f},{r['A_cap'][2]:.3f})"
        am = f"({r['A_spatial_mean'][0]:.3f},{r['A_spatial_mean'][1]:.3f},{r['A_spatial_mean'][2]:.3f})"
        ad = f"({r['A_diff'][0]:.3f},{r['A_diff'][1]:.3f},{r['A_diff'][2]:.3f})"
        print(f"{r['label']:<8} {r['file']:<25} {ac:<22} {am:<22} {ad:<22} {r['spatial_mean_dev']:.4f}")

    # Summary stats
    indoor_devs = [r['spatial_mean_dev'] for r in results if r['label'] == 'indoor']
    outdoor_devs = [r['spatial_mean_dev'] for r in results if r['label'] == 'outdoor']
    indoor_diffs = [r['A_diff'].mean() for r in results if r['label'] == 'indoor']
    outdoor_diffs = [r['A_diff'].mean() for r in results if r['label'] == 'outdoor']

    print("\n" + "="*80)
    print("Summary (Differences):")
    print(f"  Indoor:  avg A_cap vs A_inv diff = {np.mean(indoor_diffs):.4f},  avg spatial variation = {np.mean(indoor_devs):.4f}")
    print(f"  Outdoor: avg A_cap vs A_inv diff = {np.mean(outdoor_diffs):.4f},  avg spatial variation = {np.mean(outdoor_devs):.4f}")
    print(f"\n  => If indoor spatial variation >> outdoor, it confirms indoor A is NOT spatially uniform")
    print(f"  => If A_cap vs A_inv diff is large for indoor, it confirms CAP A is systematically biased")

    print("\n" + "="*90)
    print("Physics Reconstruction PSNR Ceiling & Statistics (from V2 Analysis)")
    print("="*90)
    for label in summary_stats:
        stats = summary_stats[label]
        n_samples = stats['n']
        sat = stats['a_cap_saturated']
        print(f"\n--- {label.upper()} ({n_samples} samples) ---")
        print(f"  A_cap saturated (>0.99 all channels): {sat}/{n_samples} ({100*sat/n_samples:.0f}%)")
        print(f"  Average t_cap: {stats['t_mean']:.3f}")
        print(f"")
        print(f"  Physics reconstruction PSNR (using t_cap + different A):")
        print(f"    A = CAP global:        {stats['psnr_cap_global']:6.2f} dB  (current baseline)")
        print(f"    A = ASM-inv global:    {stats['psnr_inv_global']:6.2f} dB  (inv mean, same granularity)")
        print(f"    A = ASM-inv spatial:   {stats['psnr_inv_spatial']:6.2f} dB  (inv 16x16 smooth)")
        print(f"")
        print(f"    Delta (inv_global - cap): {stats['psnr_inv_global'] - stats['psnr_cap_global']:+.2f} dB")
        print(f"    Delta (inv_spatial - cap): {stats['psnr_inv_spatial'] - stats['psnr_cap_global']:+.2f} dB")


if __name__ == '__main__':
    main()
