import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import physics as P

def check_A_spatial_stats(cap_dir, pattern="**/*.cap.npz"):
    files = glob.glob(f"{cap_dir}/{pattern}", recursive=True)
    
    results = {'indoor': [], 'outdoor': []}
    
    for f in files:
        data = np.load(f)
        if 'A_spatial' not in data:
            continue
        
        A_spatial = data['A_spatial']   # [H,W,3] smoothed spatial atmospheric light
        A_cap     = data['A_global']           # [3] CAP global scalar
        t_cap     = data['t']           # [H,W]
        
        tag = 'indoor' if 'indoor' in f.lower() else 'outdoor'
        
        A_mean = A_spatial.mean(axis=(0,1))  # [3]
        A_std  = A_spatial.std(axis=(0,1))   # [3]
        
        # Check 1: value range should stay within [0,1]
        clipped_ratio = (
            np.sum(A_spatial < 0) + np.sum(A_spatial > 1)
        ) / A_spatial.size
        
        # Check 2: deviation from CAP scalar (should be close on average)
        mean_deviation = np.abs(A_mean - A_cap).mean()
        
        # Check 3: whether A is abnormal where t is near 1 (denominator -> 0)
        high_t_mask = t_cap > 0.95                       # [H,W]
        high_t_ratio = high_t_mask.mean()
        if high_t_ratio > 0:
            A_high_t = A_spatial[high_t_mask]            # A estimates for these pixels
            high_t_A_std = A_high_t.std()
        else:
            high_t_A_std = 0.0
        
        results[tag].append({
            'file':           f,
            'A_cap':          A_cap,
            'A_mean':         A_mean,
            'A_std':          A_std,
            'clipped_ratio':  clipped_ratio,
            'mean_deviation': mean_deviation,
            'high_t_ratio':   high_t_ratio,
            'high_t_A_std':   high_t_A_std,
        })
    
    for tag in ['indoor', 'outdoor']:
        arr = results[tag]
        if not arr:
            continue
        print(f"\n===== {tag.upper()} ({len(arr)} samples) =====")
        print(f"A_spatial mean (avg across samples): "
              f"{np.mean([r['A_mean'] for r in arr], axis=0)}")
        print(f"A_spatial std  (avg across samples): "
              f"{np.mean([r['A_std']  for r in arr], axis=0)}")
        print(f"A_cap mean     (avg across samples): "
              f"{np.mean([r['A_cap']  for r in arr], axis=0)}")
        print(f"mean_deviation (A_spatial_mean vs A_cap): "
              f"{np.mean([r['mean_deviation'] for r in arr]):.4f}")
        print(f"clipped_ratio  (fraction of pixels outside [0,1]): "
              f"{np.mean([r['clipped_ratio'] for r in arr])*100:.2f}%")
        print(f"high_t_ratio   (fraction of pixels with t>0.95): "
              f"{np.mean([r['high_t_ratio'] for r in arr])*100:.2f}%")
        print(f"high_t_A_std   (std of A in t>0.95 region): "
              f"{np.mean([r['high_t_A_std'] for r in arr]):.4f}")

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-10))

def check_and_visualize_A_spatial(hazy_path, gt_path, cache_path=None, save_plot=False, out_prefix="", smooth_scale=16):
    I_np = np.array(Image.open(hazy_path).convert('RGB')).astype(np.float32) / 255.
    J_np = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.
    H, W = I_np.shape[:2]

    # Get t_cap and A_cap
    if cache_path and os.path.exists(cache_path):
        data   = np.load(cache_path)
        t_cap   = data['t']          # [H,W]
        A_cap  = data['A_global']          # [3]
        if 'A_spatial' in data:
            A_spatial = data['A_spatial']
        else:
            A_spatial = P.estimate_A_from_pair(I_np, J_np, t_cap, smooth_scale=smooth_scale)
    else:
        t_cap, A_cap, _ = P.run_cap(I_np)
        A_spatial = P.estimate_A_from_pair(I_np, J_np, t_cap, smooth_scale=smooth_scale)

    danger_mask = t_cap > 0.95
    danger_ratio = danger_mask.mean()
    
    A_danger_mean = 0.0
    A_danger_std = 0.0
    if danger_mask.any():
        A_danger = A_spatial[danger_mask]
        A_danger_mean = A_danger.mean()
        A_danger_std = A_danger.std()

    if save_plot:
        # Expand for reconstruction
        t3 = np.repeat(t_cap[..., None] if t_cap.ndim == 2 else t_cap, 3, axis=2)
        A_cap_expanded = np.broadcast_to(A_cap[None, None, :], (H, W, 3)).copy()

        # Reconstruct I using different A sources
        I_rec_cap     = J_np * t3 + A_cap_expanded * (1.0 - t3)
        I_rec_spatial = J_np * t3 + A_spatial      * (1.0 - t3)

        # Reconstruction PSNR (measures A quality)
        psnr_cap     = psnr(I_rec_cap, I_np)
        psnr_spatial = psnr(I_rec_spatial, I_np)

        # Also show J recovery
        J_rec_cap     = np.clip((I_np - A_cap_expanded * (1 - t3)) / np.maximum(t3, 0.1), 0, 1)
        J_rec_spatial = np.clip((I_np - A_spatial * (1 - t3)) / np.maximum(t3, 0.1), 0, 1)
        psnr_J_cap     = psnr(J_rec_cap, J_np)
        psnr_J_spatial = psnr(J_rec_spatial, J_np)

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        # Row 1: Inputs
        axes[0,0].imshow(I_np);            axes[0,0].set_title('I (hazy input)')
        axes[0,1].imshow(J_np);            axes[0,1].set_title('J_gt (clean GT)')
        axes[0,2].imshow(t_cap, cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title(f't_cap (mean={t_cap.mean():.3f})')
        # t>0.95 danger zone
        danger = danger_mask.astype(float)
        axes[0,3].imshow(danger, cmap='Reds', vmin=0, vmax=1)
        axes[0,3].set_title(f't>0.95 zones ({danger_ratio*100:.1f}%)')

        # Row 2: A comparison (this IS the key visual)
        axes[1,0].imshow(np.clip(A_cap_expanded, 0, 1))
        axes[1,0].set_title(f'A_cap global ({A_cap[0]:.2f},{A_cap[1]:.2f},{A_cap[2]:.2f})')
        axes[1,1].imshow(np.clip(A_spatial, 0, 1))
        axes[1,1].set_title(f'A_spatial (smooth={smooth_scale})')
        diff = np.abs(A_spatial - A_cap_expanded).mean(axis=2)
        im = axes[1,2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[1,2].set_title('|A_spatial - A_cap| (mean RGB)')
        plt.colorbar(im, ax=axes[1,2])
        # Reconstruction error comparison
        err_cap = np.abs(I_rec_cap - I_np).mean(axis=2)
        err_spatial = np.abs(I_rec_spatial - I_np).mean(axis=2)
        im2 = axes[1,3].imshow(err_cap - err_spatial, cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[1,3].set_title('Recon error: cap - spatial\n(blue=spatial better)')
        plt.colorbar(im2, ax=axes[1,3])

        # Row 3: J reconstruction comparison
        axes[2,0].imshow(np.clip(J_rec_cap, 0, 1))
        axes[2,0].set_title(f'J recovered (A_cap)\nPSNR={psnr_J_cap:.2f} dB')
        axes[2,1].imshow(np.clip(J_rec_spatial, 0, 1))
        axes[2,1].set_title(f'J recovered (A_spatial)\nPSNR={psnr_J_spatial:.2f} dB')
        axes[2,2].imshow(J_np)
        axes[2,2].set_title('J_gt (reference)')
        # Summary text
        axes[2,3].axis('off')
        summary = (
            f"I reconstruction PSNR:\n"
            f"  A_cap:     {psnr_cap:.2f} dB\n"
            f"  A_spatial: {psnr_spatial:.2f} dB\n"
            f"  Delta:     {psnr_spatial - psnr_cap:+.2f} dB\n\n"
            f"J recovery PSNR:\n"
            f"  A_cap:     {psnr_J_cap:.2f} dB\n"
            f"  A_spatial: {psnr_J_spatial:.2f} dB\n"
            f"  Delta:     {psnr_J_spatial - psnr_J_cap:+.2f} dB\n\n"
            f"A_cap saturated: {'YES' if np.all(A_cap > 0.99) else 'NO'}"
        )
        axes[2,3].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                       verticalalignment='center', transform=axes[2,3].transAxes)

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f'{out_prefix}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

    return danger_ratio, A_danger_mean, A_danger_std


def main():
    val_txt = './SOTS/split_txt/val.txt'
    if not os.path.exists(val_txt):
        print(f"File not found: {val_txt}")
        return
        
    lines = open(val_txt).read().strip().split('\n')
    indoor_lines = [l for l in lines if 'indoor' in l]
    outdoor_lines = [l for l in lines if 'outdoor' in l]

    out_dir = './outputs/A_spatial_check'
    os.makedirs(out_dir, exist_ok=True)

    print("="*90)
    print("Analysis: Spatial A-Map Instability (t -> 1 noise amplification) & Quality Verification")
    print("="*90)

    for label, sample_lines in [("INDOOR", indoor_lines), ("OUTDOOR", outdoor_lines)]:
        danger_ratios = []
        danger_means = []
        danger_stds = []
        
        # We only save plots for the first image to avoid clutter
        for i, line in enumerate(tqdm(sample_lines, desc=f"Processing {label.upper()}")):
            try:
                # Some lines might have trailing spaces or different formats
                parts = line.strip().split()
                if len(parts) < 2: continue
                hazy_path = parts[0]
                gt_path = parts[1]
                
                cache_path = hazy_path + ".cap.npz"
                save_plot = (i == 0)
                
                fname = os.path.basename(hazy_path).split('.')[0]
                path_prefix = os.path.join(out_dir, f"{label.lower()}_{fname}") if save_plot else ""
                
                ratio, d_mean, d_std = check_and_visualize_A_spatial(hazy_path, gt_path, cache_path, save_plot, path_prefix)
                
                danger_ratios.append(ratio)
                if ratio > 0:
                    danger_means.append(d_mean)
                    danger_stds.append(d_std)
            except Exception as e:
                # gracefully ignore corrupted files 
                pass

        n = len(danger_ratios)
        if n == 0: continue
        
        print(f"\n--- {label} ({n} samples) ---")
        print(f"  Average t>0.95 pixel ratio: {np.mean(danger_ratios)*100:.1f}%")
        if len(danger_means) > 0:
            print(f"  In danger zones (t>0.95), average A_spatial:")
            print(f"    Mean: {np.mean(danger_means):.3f}")
            print(f"    Std:  {np.mean(danger_stds):.3f} (indicates high-frequency noise scale)")
        else:
            print("  No danger zones (t>0.95) detected.")

    print("\n" + "="*90)
    print("Interpretation:")
    print("  - 'Danger Zones' occur where t approaches 1 (no haze, clear objects).")
    print("  - The ASM-inversion formula A = (I - J*t)/(1 - t) explodes dynamically in these zones.")
    print("  - High 'Std' in these zones verifies the catastrophic high-frequency noise projection.")
    print("  - First sample from each category was saved to the A_spatial_check directory.")
    print("="*90)

if __name__ == '__main__':
    main()
