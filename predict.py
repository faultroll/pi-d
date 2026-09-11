"""predict.py -- inference/evaluation on a trained checkpoint, and the
onnx-tool MAC/param profiling used for Table `cost_summary`'s whole-model
row (see NAMING.md).

WARNING: this script's --model_path defaults to a .onnx file, but the ONNX
export in train.py is taken from the LAST training epoch, not the
best-validation-loss weights (see the comment above the `torch.onnx.export`
call in train.py). Every PSNR/SSIM/LPIPS number reported in the paper comes
from re-evaluating the saved .pth checkpoints with unified_ground_truth_eval.py,
never from a .onnx file. Pass --model_path pointing at a .pth checkpoint if
you want this script's own PSNR/SSIM output to match the paper; use the
.onnx file only for the MAC/param count, where the exact weight values
don't matter.
"""

import os
import re
import onnx
import onnxruntime as ort
import onnx_tool
from onnx_tool import create_ndarray_f32
import numpy as np
from PIL import Image
import time
import metrics as M


# -----------------------------------------------------------------
#  haze level from filename
# -----------------------------------------------------------------

def extract_haze_params(file_path):
    hazy_path = file_path.strip().split()[0]
    hazy_filename = os.path.basename(hazy_path)
    if "indoor" in hazy_path:
        # Indoor: xxxx_xx.png
        match = re.search(r'_(\d+)\.png$', hazy_filename)
        if match:
            haze_level = int(match.group(1))
            return ("indoor", str(haze_level), haze_level)
    elif "outdoor" in hazy_path:
        # Outdoor: xxxx_0.xx_0.xx.jpg
        match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)\.jpg$', hazy_filename)
        if match:
            A = float(match.group(1))
            beta = float(match.group(2))
            param_str = f"{A}_{beta}"
            return ("outdoor", param_str, (A, beta))
    return ("unknown", "", None)

def classify_haze_level(file_path):
    dataset_type, _, params = extract_haze_params(file_path)
    if dataset_type == "indoor":
        haze_level = params
        if 1 <= haze_level <= 3:
            return 'thin'
        elif 4 <= haze_level <= 6:
            return 'medium'
        elif 7 <= haze_level <= 10:
            return 'thick'
    elif dataset_type == "outdoor":
        A, beta = params
        haze_score = A * beta # empirical formula
        if haze_score < 0.08:
            return 'thin'
        elif 0.08 <= haze_score < 0.12:
            return 'medium'
        elif haze_score >= 0.12:
            return 'thick'
    return 'unknown'

# -----------------------------------------------------------------
#  Loader
# -----------------------------------------------------------------

class SimpleImageLoader:
    """
    Simple image loader with RAM cache and NPZ disk cache.
    Based on DehazeDataset implementation in train.py.
    """
    def __init__(self, txt_file, use_cache=True):
        with open(txt_file, 'r') as f:
            lines = f.read().strip().split('\n')
        self.image_pairs = [line.strip().split() for line in lines]
        self.use_cache = use_cache
        self.ram_cache = {}  # RAM cache: idx -> (I_np, J_gt_np, hazy_path)
    def __len__(self):
        return len(self.image_pairs)
    def _get_cache_path(self, hazy_path):
        return hazy_path + ".predict.npz"
    def __getitem__(self, idx):
        if self.use_cache and idx in self.ram_cache:
            return self.ram_cache[idx]
        hazy_path, gt_path = self.image_pairs[idx]
        if self.use_cache:
            cache_path = self._get_cache_path(hazy_path)
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    I_np = data['I']
                    J_gt_np = data['J_gt']
                    self.ram_cache[idx] = (I_np, J_gt_np, hazy_path)
                    return self.ram_cache[idx]
                except Exception:
                    pass
        I_pil = Image.open(hazy_path).convert('RGB')
        J_gt_pil = Image.open(gt_path).convert('RGB')
        I_np = np.array(I_pil).astype(np.float32) / 255.0 # [H,W,3]
        J_gt_np = np.array(J_gt_pil).astype(np.float32) / 255.0 # [H,W,3]
        if self.use_cache:
            cache_path = self._get_cache_path(hazy_path)
            try:
                np.savez(cache_path, I=I_np, J_gt=J_gt_np)
            except Exception:
                pass
        if self.use_cache:
            self.ram_cache[idx] = (I_np, J_gt_np, hazy_path)
        return I_np, J_gt_np, hazy_path

def SimpleImageSaver(tensor, path):
    if tensor.shape[0] == 1:
        img_array = tensor[0]
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    else:
        img_array = np.transpose(tensor, (1, 2, 0))
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    img.save(path)


# -----------------------------------------------------------------
#  Evaluation
# -----------------------------------------------------------------

def predict(model_path, test_txt, result_dir, eval_hazy_baseline=False):
    os.makedirs(result_dir, exist_ok=True)
    print(f"[1/4] Loading ONNX model: {model_path}")
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    # model_info = onnx_tool.model_profile(model_path, dynamic_shapes={"I": create_ndarray_f32((1, 3, 512, 512))})
    print(f"[2/4] Creating inference session...")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    # For PGDehazeNet: ['t_pred', 'A_pred', 'J_pred']  -> J is index 2
    # For LightDehazeNet / BILDNet: ['J_pred']         -> J is index 0
    # j_idx = 2 if len(output_names) == 3 else 0
    print(f"[3/4] Loading test images from: {test_txt}")
    loader = SimpleImageLoader(test_txt)
    total = len(loader)
    print(f"[3/4] Evaluating {total} test images...")
    records = []
    for idx in range(len(loader)):
        I_np, J_gt_np, hazy_path = loader[idx]
        I = np.transpose(I_np, (2, 0, 1))
        J_gt = np.transpose(J_gt_np, (2, 0, 1))
        # -- run inference -----------------------------------------
        I_input = np.transpose(I_np, (2, 0, 1))  # [H,W,3] -> [3,H,W]
        I_input = np.stack([I_input, I_input]) # batch_size
        # I_input = np.transpose(I_np, (2, 0, 1))[None]    # [1,3,H,W]
        start_time = time.perf_counter()
        outputs = session.run(output_names, {input_name: I_input})
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000.0
        # print(f"shape {I_np.shape}, time {execution_time:.2f}ms")
        #time_list.append(execution_time)
        # t_pred = outputs[0][0]
        # A_pred = outputs[1][0]
        J_pred = outputs[2][0]
        J_pred_np = np.transpose(J_pred, (1, 2, 0))
        # J_pred_np = np.transpose(outputs[j_idx][0], (1, 2, 0))   # [H,W,3]
        # -- save examples -----------------------------------------        if idx == 1:
        if idx == 1:
            print(f"Saving {hazy_path} results...")
            SimpleImageSaver(J_gt, result_dir + 'J_gt.png')
            SimpleImageSaver(I, result_dir + 'I_gt.png')
            SimpleImageSaver(J_pred, result_dir + 'J_pred.png')
        # -- metrics -----------------------------------------------
        haze_level = classify_haze_level(hazy_path)
        psnr_pred = M.compute_psnr(J_pred_np, J_gt_np)
        ssim_pred = M.compute_ssim(J_pred_np, J_gt_np)
        if eval_hazy_baseline:
            psnr_hazy = M.compute_psnr(I_np, J_gt_np)
            ssim_hazy = M.compute_ssim(I_np, J_gt_np)
        else:
            psnr_hazy = None
            ssim_hazy = None
        # if (idx + 1) % 50 == 0 or idx == total - 1:
        print(f"  [{idx+1:4d}/{total}]  PSNR={psnr_pred:.2f}  SSIM={ssim_pred:.4f}  t={execution_time:.1f}ms  {hazy_path}  {haze_level}")
        records.append({
            'idx':       idx,
            'hazy_path': hazy_path,
            'haze_level':     haze_level,
            # 'I_np':      I_np,
            # 'J_gt_np':   J_gt_np,
            # 'J_pred_np': J_pred_np,
            'psnr':      psnr_pred,
            'ssim':      ssim_pred,
            'psnr_hazy': psnr_hazy,
            'ssim_hazy': ssim_hazy,
            'time_ms':   execution_time,
        })
    # -- aggregate results -----------------------------------------
    LEVELS   = ['thin', 'medium', 'thick', 'unknown']
    by_level = {lv: [r for r in records if r['haze_level'] == lv] for lv in LEVELS}
    def stats(lst):
        a = np.array(lst)
        return a.mean(), a.std()
    def print_tier(label, recs):
        if not recs:
            return
        pm, ps = stats([r['psnr'] for r in recs])
        sm, ss = stats([r['ssim'] for r in recs])
        line   = (f"  {label:<10} N={len(recs):>4}  "
                  f"PSNR={pm:.2f}+-{ps:.2f}  SSIM={sm:.4f}+-{ss:.4f}")
        if eval_hazy_baseline:
            ph, _ = stats([r['psnr_hazy'] for r in recs])
            sh, _ = stats([r['ssim_hazy'] for r in recs])
            line += f"   DeltaPSNR={pm-ph:+.2f}  DeltaSSIM={sm-sh:+.4f}"
        print(line)
    print(f"\n" + "="*72)
    print(f"  Results by haze tier  ")
    print("="*72)
    for lv in LEVELS:
        print_tier(lv, by_level[lv])
    all_psnr = [r['psnr'] for r in records]
    all_ssim = [r['ssim'] for r in records]
    print(f"\n[4/4] ALL  PSNR={np.mean(all_psnr):.2f}  SSIM={np.mean(all_ssim):.4f}")
    print("  " + "-"*68)
    print_tier("ALL", records)
    tm, ts = stats([r['time_ms'] for r in records])
    print(f"\n  Inference time: {tm:.1f} +- {ts:.1f} ms  (CPU, {I_np.shape})")
    print("="*72)
    # -- per-image breakdown (sorted by PSNR, worst first) --------
    print("\nPer-image breakdown (worst 5 and best 5 by PSNR and SSIM):")
    ranked_records = sorted(records, key=lambda x: (-x['psnr'], -x['ssim']))
    for label, subset in [("Best", ranked_records[:5]), ("Worst", ranked_records[-5:][::-1])]:
        print(f"  {label}:")
        for r in subset:
            print(f"    [{r['idx']:4d}] PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}  {r['hazy_path']}")
    # print("\nFull results sorted by PSNR (descending):")
    # print(f"  {'idx':>4}  {'haze_level':<7}  {'PSNR':>6}  {'SSIM':>6}  filename")
    # for r in ranked_records:
    #     print(f"  {r['idx']:>4}  {r['haze_level']:<7}  "
    #           f"{r['psnr']:>6.2f}  {r['ssim']:>6.4f}  "
    #           f"{os.path.basename(r['hazy_path'])}")


import sys
import argparse

def parse_args():
    """Parse command line arguments. Environment variables are used as defaults."""
    parser = argparse.ArgumentParser(
        description='PGDehazeNet Prediction and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py --model_path outputs/best_model.pth --result_dir outputs/results
    python predict.py --model_path outputs/joint_net.onnx --test_txt ./SOTS/split_txt/indoor_test.txt
        """
    )
    
    # Model paths
    parser.add_argument('--model_path', type=str,
                        default=os.getenv('ONNX_PATH', './outputs/joint_net.onnx'),
                        help='Path to ONNX model file')
    
    # Data paths
    parser.add_argument('--test_txt', type=str,
                        default=os.getenv('TEST_TXT', './SOTS/split_txt/indoor_test.txt'),
                        help='Path to test image list txt file')
    
    # Output paths
    parser.add_argument('--result_dir', type=str,
                        default=os.getenv('RESULT_DIR', './outputs/results'),
                        help='Directory to save result images')
    
    # Evaluation options
    parser.add_argument('--eval_hazy_baseline', type=str,
                        default=os.getenv('EVAL_HAZY_BASELINE', 'False'),
                        help='Also compute PSNR/SSIM on raw hazy input (true/1/yes)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    eval_hazy = args.eval_hazy_baseline.lower() in ('true', '1', 'yes')
    predict(args.model_path, args.test_txt, args.result_dir, eval_hazy)
