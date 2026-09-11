"""run_sprint_finalization.py -- trains the 6 V3-t checkpoints that this
release's numbers are built from (see NAMING.md / TRACEABILITY.md for the
paper-table <-> checkpoint map).

This is a deliberately trimmed copy: the original development version of
this script also orchestrated a much larger architecture-search sweep
(dozens of configurations spanning T-Gate/SFT/RepConv/dilation variants and
an A-branch M0-M8 sweep) and, after each run, called a separate
Fisher-information/parameter-utilization tool. None of that is part of the
paper's reported method or reported numbers, so it has been removed rather
than carried along as dead weight -- see the repo README for where that
architecture-search code used to live.

What this script actually does, end to end:
  1. Train each of the 6 configs below by shelling out to train.py, which
     saves the best-validation checkpoint to outputs/model_<suite>_<name>_<mode>.pth.
  2. That's it. This script does NOT compute any of the paper's reported
     PSNR/SSIM/LPIPS numbers -- those come afterward, by pointing
     unified_ground_truth_eval.py (main table) and experiments/*.py
     (supplementary tables) at the resulting .pth files. See
     experiments/common.py:CHECKPOINT_REGISTRY for the exact filename
     each config below is expected to produce.

The Problem-1/C2 diagnostic scripts (theta_star_model.py, a_quality_compare.py,
a_spatial_stats.py) are standalone tools you point at a trained checkpoint
by hand; they are intentionally not wired into this training loop.

Usage:
    python run_sprint_finalization.py
"""
import os
import subprocess
import time
import sys
import datetime

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Logger(object):
    def __init__(self, filename=None):
        if filename is None:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"train_log_{now}.txt"
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout

# The 6 configs that produced the checkpoints this release's numbers are
# computed from. See experiments/common.py:CHECKPOINT_REGISTRY for exactly
# which released .pth file each one corresponds to, and TRACEABILITY.md for
# which paper table cites it. All 6 use mode="V3-t" and leave every
# unpublished-architecture-search flag (dilation_mode, channel_attn_mode,
# use_t_gate, use_varc, use_a_bound, refine, repconv, a_net_mode) at its
# default -- see the warning at the top of network.py.
sprint_matrix = [
    # A-Baseline / B-FiLMTAff / C-Depth: the sequential-ablation ladder
    # (Table `ablation_modules`), no GAN, seed 42.
    {"name": "A-Baseline", "mode": "V3-t", "seed": 42, "gan": False, "film_mode": "None",     "use_t_affine": False, "depth": False},
    {"name": "B-FiLMTAff", "mode": "V3-t", "seed": 42, "gan": False, "film_mode": "PerBlock", "use_t_affine": True,  "depth": False},
    {"name": "C-Depth",    "mode": "V3-t", "seed": 42, "gan": False, "film_mode": "PerBlock", "use_t_affine": True,  "depth": True},
    # D-Prop-*: Proposed (+GAN), 3 seeds -- Tables `psnr_main`, `lpips`,
    # `ablation_modules`'s "+GAN" row (mean +/- sample std across these 3).
    {"name": "D-Prop-S42-R1",  "mode": "V3-t", "seed": 42, "gan": True, "film_mode": "PerBlock", "use_t_affine": True, "depth": True},
    {"name": "D-Prop-V3t-S43", "mode": "V3-t", "seed": 43, "gan": True, "film_mode": "PerBlock", "use_t_affine": True, "depth": True},
    {"name": "D-Prop-V3t-S44", "mode": "V3-t", "seed": 44, "gan": True, "film_mode": "PerBlock", "use_t_affine": True, "depth": True},
]
# ("+DAF only" -- Table `ablation_modules`'s remaining row, checkpoint
#  C1-DepthOnly -- is trained separately by experiments/daf_only_ablation.py,
#  not by this script; see that file.)


def run_suite(suite_name, matrix):
    print(f"\n{'=' * 60}")
    print(f"Starting Suite: {suite_name}")
    print(f"{'=' * 60}")

    results = []

    for exp in matrix:
        v_name = exp["name"]
        mode = exp["mode"]
        dilation_mode = exp.get("dilation_mode", "None")
        depth = exp.get("depth", False)
        channel_attn_mode = exp.get("channel_attn_mode", "None")
        gan = exp.get("gan", False)
        film_mode = exp.get("film_mode", "None")
        use_t_affine = exp.get("use_t_affine", False)
        use_t_gate = exp.get("use_t_gate", False)
        use_varc = exp.get("use_varc", False)
        use_a_bound = exp.get("use_a_bound", False)
        refine = exp.get("refine", False)
        repconv = exp.get("repconv", False)
        a_net_mode = exp.get("a_net_mode", "M0")
        a_gt_mode = exp.get("a_gt_mode", "global")

        pth_path = os.path.join(OUTPUT_DIR, f"model_{suite_name}_{v_name}_{mode}.pth")
        onnx_path = os.path.join(OUTPUT_DIR, f"model_{suite_name}_{v_name}_{mode}.onnx")

        print(f"\n[Training] {v_name} ({mode} | GAN:{gan} FiLM:{film_mode} t-Aff:{use_t_affine} Depth:{depth})")

        env = os.environ.copy()
        env["CAPACITY_MODE"] = mode
        env["BEST_MODEL_PATH"] = pth_path
        env["ONNX_PATH"] = onnx_path
        env["DILATION_MODE"] = dilation_mode
        env["USE_DEPTH_ATTN"] = str(depth)
        env["CHANNEL_ATTN_MODE"] = channel_attn_mode
        env["USE_GAN"] = str(gan)
        env["FILM_MODE"] = str(film_mode)
        env["USE_T_AFFINE"] = str(use_t_affine)
        env["USE_T_GATE"] = str(use_t_gate)
        env["USE_VARC"] = str(use_varc)
        env["USE_A_BOUND"] = str(use_a_bound)
        env["USE_REFINE"] = str(refine)
        env["USE_REPCONV"] = str(repconv)
        env["A_NET_MODE"] = a_net_mode
        env["A_GT_MODE"] = a_gt_mode
        env["CUDA_VISIBLE_DEVICES"] = "0"  # CUDA starts indexing from 0 for NVIDIA GPUs ONLY

        seed = exp.get("seed", 42)
        env["PYTHONHASHSEED"] = str(seed)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Disable cuBLAS heuristics to enforce determinism

        train_mode = os.getenv("TRAIN_MODE", "FULL")  # 'DEBUG', 'QUICK', 'MEDIUM', 'FULL'

        start_time = time.time()

        # Real-time stdout capture so the Logger catches it.
        train_process = subprocess.Popen(["python", "train.py", "--seed", str(seed), "--train-mode", train_mode],
                                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in train_process.stdout:
            sys.stdout.write(line)
        train_process.wait()

        if train_process.returncode != 0:
            print(f"[X] Error: Training failed for {v_name}")
            continue

        train_duration = time.time() - start_time
        print(f"[O] Training completed in {train_duration:.1f}s -> {pth_path}")

        results.append({
            "Config": v_name,
            "Mode": mode,
            "GAN": "Y" if gan else "N",
            "FiLM": film_mode,
            "T-Aff": "Y" if use_t_affine else "N",
            "Depth": "Y" if depth else "N",
            "Checkpoint": pth_path,
            "Time": f"{train_duration:.1f}s",
        })

    return results


def print_table(title, results):
    print("\n" + "=" * 100)
    print(f"RESULTS: {title}")
    print("=" * 100)
    print("| Config           | Mode | GAN | FiLM     | T-Aff | Depth | Train Time | Checkpoint |")
    print("|------------------|------|-----|----------|-------|-------|------------|------------|")
    for r in results:
        print(f"| {r['Config']:<16} | {r['Mode']:<4} | {r['GAN']:<3} | {r['FiLM']:<8} | {r['T-Aff']:<5} | {r['Depth']:<5} | {r['Time']:<10} | {r['Checkpoint']} |")
    print("\nNext step: point unified_ground_truth_eval.py (and, for the")
    print("supplementary tables, experiments/*.py) at these .pth files -- see")
    print("experiments/common.py:CHECKPOINT_REGISTRY for the expected filenames.")


if __name__ == "__main__":
    sprint_results = run_suite("Sprint_Finalization", sprint_matrix)
    if sprint_results:
        print_table("Sprint Finalization -- the 6 checkpoints behind this release's numbers", sprint_results)
