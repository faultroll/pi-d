"""
common.py
=================
Shared utilities for the four "must-fix" supplementary experiments requested
in the 2026-08-21 advisor review, Section 6 ("items 1-5 are mandatory
before submission").

This module is a thin, well-tested wrapper around the project's EXISTING
code (network.py / physics.py / metrics.py / train.py) -- it does not
reimplement anything that already exists, so results stay consistent with
whatever produced Tables 2-6 in the manuscript.

Design note on imports: torch / network are imported LAZILY (inside
functions) rather than at module load time, so that the pure-numpy pieces
(physics.py-based ceiling computations, scene-id parsing, etc.) can be
imported and unit-tested even in environments where a GPU-enabled torch
build cannot be loaded (e.g. a CPU-only CI box). This makes no difference
on a normal training machine.

Run this file directly for a self-test that needs no GPU, no dataset and no
checkpoints:
    python common.py --selftest
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# Locate the pgdehazenet package root (the folder containing network.py,
# physics.py, train.py, metrics.py) so this script works whether it is
# invoked as `python experiments/exp4_....py` from the repo root, or
# `python exp4_....py` from inside experiments/, or copied elsewhere.
# ---------------------------------------------------------------------
def find_repo_root(start=None):
    here = Path(start or __file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "network.py").exists() and (candidate / "physics.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate pgdehazenet/ (folder containing network.py + physics.py). "
        "Place this experiments/ folder directly inside the pgdehazenet/ repo "
        "root, next to train.py, or run scripts with --repo-root /path/to/pgdehazenet."
    )


REPO_ROOT = None  # set by ensure_repo_on_path()


def ensure_repo_on_path(repo_root=None):
    """Insert the pgdehazenet repo root onto sys.path so `import network`,
    `import physics`, `from train import DehazeDataset` etc. work exactly as
    they do in the project's own scripts (theta_star_model.py does the same
    sys.path.insert(0, ...) trick)."""
    global REPO_ROOT
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    if not (root / "network.py").exists():
        raise RuntimeError(f"--repo-root {root} does not contain network.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    REPO_ROOT = root
    return root


def repo_path(*parts):
    if REPO_ROOT is None:
        ensure_repo_on_path()
    return str(Path(REPO_ROOT, *parts))


# ---------------------------------------------------------------------
# Known checkpoints, transcribed verbatim from
# pgdehazenet/run_sprint_finalization.py:sprint_matrix (the script that
# actually produced experiments/sprint/model_Sprint_Finalization_*.pth,
# i.e. the checkpoints behind paper Table 4/5). Every field below is a
# 1:1 copy of that matrix combined with network.PGDehazeNet's own default
# values for anything sprint_matrix did not override (a_net_mode="M0",
# a_gt_mode="global", dilation_mode="None", channel_attn_mode="None",
# use_refine/use_repconv/use_t_gate/use_varc/use_a_bound=False) -- the same
# defaults confirmed against experiments/sprint/expressiveness_*_A-Baseline_*
# .json's "config" field.
#
# ASSUMPTION flagged for the user: sprint_matrix contains TWO seed-42 runs
# for the proposed config ("D-Prop-S42-R1" and "D-Prop-S42-R2", labelled
# "Phase 0: Reproducibility double-check"). We use R1 as "the" seed-42 run
# reported in the paper's 3-seed table, since R2 is its determinism repeat.
# If your bookkeeping says otherwise, edit CHECKPOINT_REGISTRY below or
# pass --checkpoint-map to override individual paths.
# ---------------------------------------------------------------------
_BASE_CFG = dict(
    capacity_mode="V3-t", dilation_mode="None", channel_attn_mode="None",
    use_refine=False, use_repconv=False, use_t_gate=False,
    use_varc=False, use_a_bound=False, a_net_mode="M0",
)

CHECKPOINT_REGISTRY = {
    "A-Baseline": dict(
        pth="model_Sprint_Finalization_A-Baseline_V3-t.pth",
        label="V3-t baseline",
        config=dict(_BASE_CFG, use_depth_attn=False, film_mode="None", use_t_affine=False),
    ),
    "B-FiLMTAff": dict(
        pth="model_Sprint_Finalization_B-FiLMTAff_V3-t.pth",
        label="+FiLM+T-Affine",
        config=dict(_BASE_CFG, use_depth_attn=False, film_mode="PerBlock", use_t_affine=True),
    ),
    "C-Depth": dict(
        pth="model_Sprint_Finalization_C-Depth_V3-t.pth",
        label="+FiLM+T-Affine+DAF",
        config=dict(_BASE_CFG, use_depth_attn=True, film_mode="PerBlock", use_t_affine=True),
    ),
    "D-Prop-S42": dict(
        pth="model_Sprint_Finalization_D-Prop-S42-R1_V3-t.pth",
        label="Proposed (+GAN) seed42",
        config=dict(_BASE_CFG, use_depth_attn=True, film_mode="PerBlock", use_t_affine=True),
    ),
    "D-Prop-S43": dict(
        pth="model_Sprint_Finalization_D-Prop-V3t-S43_V3-t.pth",
        label="Proposed (+GAN) seed43",
        config=dict(_BASE_CFG, use_depth_attn=True, film_mode="PerBlock", use_t_affine=True),
    ),
    "D-Prop-S44": dict(
        pth="model_Sprint_Finalization_D-Prop-V3t-S44_V3-t.pth",
        label="Proposed (+GAN) seed44",
        config=dict(_BASE_CFG, use_depth_attn=True, film_mode="PerBlock", use_t_affine=True),
    ),
}
# DAF-only (item 5's new ablation row) is registered dynamically by
# daf_only_ablation.py once it has actually been trained -- see
# register_daf_only_checkpoint() below.


def register_daf_only_checkpoint(pth_path):
    CHECKPOINT_REGISTRY["C1-DepthOnly"] = dict(
        pth=pth_path,
        label="+DAF only (no FiLM, no T-Affine)",
        config=dict(_BASE_CFG, use_depth_attn=True, film_mode="None", use_t_affine=False),
    )


def find_checkpoint_file(filename, extra_roots=()):
    """Robustly locate a checkpoint by filename, since the archived
    experiments/<suite>/ folders may sit inside, next to, or several
    levels away from the pgdehazenet/ code root depending on how each
    person's working copy is laid out (the code and the archived
    checkpoints can originate from two separately-zipped folders).
    Search order:
      1. <repo_root>/<filename's own relative path as given>
      2. <repo_root>/experiments/sprint/<basename>
      3. <repo_root>/../experiments/sprint/<basename>   (sibling layout)
      4. <repo_root>/outputs/<basename>                  (freshly generated)
      5. any extra_roots passed in
      6. bounded recursive search under repo_root's parent (2 levels up,
         3 levels down) for an exact basename match, as a last resort.
    Raises FileNotFoundError listing everywhere it looked if nothing matches.
    """
    if REPO_ROOT is None:
        ensure_repo_on_path()
    basename = os.path.basename(filename)
    candidates = [
        Path(REPO_ROOT, filename),
        Path(REPO_ROOT, "experiments", "sprint", basename),
        Path(REPO_ROOT).parent / "experiments" / "sprint" / basename,
        Path(REPO_ROOT, "outputs", basename),
    ]
    for r in extra_roots:
        candidates.append(Path(r, basename))
    for c in candidates:
        if c.exists():
            return str(c)
    # last resort: bounded recursive glob search
    search_root = Path(REPO_ROOT).parent
    hits = list(search_root.glob(f"**/{basename}"))
    if hits:
        return str(hits[0])
    tried = "\n  ".join(str(c) for c in candidates) + f"\n  (recursive search under {search_root})"
    raise FileNotFoundError(
        f"Could not find checkpoint file {basename!r}. Looked in:\n  {tried}\n"
        f"Pass --checkpoint-map (JSON: {{\"A-Baseline\": {{\"pth\": \"/full/path.pth\"}}}}) "
        f"to point this script at the right location."
    )


def load_checkpoint_map_override(path):
    """Optional JSON file: {"A-Baseline": {"pth": "..."}, ...} to patch
    CHECKPOINT_REGISTRY entries (e.g. if your files live somewhere else)."""
    with open(path) as f:
        overrides = json.load(f)
    for name, patch in overrides.items():
        if name not in CHECKPOINT_REGISTRY:
            CHECKPOINT_REGISTRY[name] = {"config": dict(_BASE_CFG)}
        CHECKPOINT_REGISTRY[name].update(patch)


# ---------------------------------------------------------------------
# Model construction / loading (torch is imported lazily here only)
# mirrors experiments/collect_paper_data.py:build_and_load_model exactly.
# ---------------------------------------------------------------------
def build_model(config, device):
    ensure_repo_on_path()
    import network as N
    model = N.PGDehazeNet(
        capacity_mode=config["capacity_mode"],
        use_t_affine=config["use_t_affine"],
        use_t_gate=config.get("use_t_gate", False),
        use_depth_attn=config["use_depth_attn"],
        channel_attn_mode=config.get("channel_attn_mode", "None"),
        film_mode=config["film_mode"],
        use_refine=config.get("use_refine", False),
        use_repconv=config.get("use_repconv", False),
        use_varc=config.get("use_varc", False),
        use_a_bound=config.get("use_a_bound", False),
        a_net_mode=config.get("a_net_mode", "M0"),
        dilation_mode=config.get("dilation_mode", "None"),
    ).to(device)
    return model


def load_named_checkpoint(name, device=None):
    """Build + load one of CHECKPOINT_REGISTRY's known models.
    Returns (model, config, pth_path)."""
    import torch
    if name not in CHECKPOINT_REGISTRY:
        raise KeyError(f"Unknown checkpoint name {name!r}. Known: {list(CHECKPOINT_REGISTRY)}")
    entry = CHECKPOINT_REGISTRY[name]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_path = entry["pth"]
    if not os.path.isabs(pth_path) and not os.path.exists(pth_path):
        pth_path = find_checkpoint_file(pth_path)
    model = build_model(entry["config"], device)
    state = torch.load(pth_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, entry["config"], pth_path


def load_checkpoint_from_path(pth_path, config, device=None):
    """Same as load_named_checkpoint but for an arbitrary path + explicit config
    (used for e.g. a freshly-trained DAF-only checkpoint not yet in the registry)."""
    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    state = torch.load(pth_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Haze-tier classification & scene-id parsing
# copied verbatim from predict.py / collect_paper_data.py so numbers are
# comparable to existing tables, plus a scene_id extractor (needed for the
# scene-level bootstrap, since the 35 indoor pairs are 5 haze variants of
# only 7 distinct scenes -- Limitations item in the review).
# ---------------------------------------------------------------------
def extract_haze_params(file_path):
    hazy_path = file_path.strip().split()[0]
    hazy_filename = os.path.basename(hazy_path)
    if "indoor" in hazy_path:
        m = re.search(r'_(\d+)\.png$', hazy_filename)
        if m:
            haze_level = int(m.group(1))
            return ("indoor", str(haze_level), haze_level)
    elif "outdoor" in hazy_path:
        m = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)\.jpg$', hazy_filename)
        if m:
            A = float(m.group(1))
            beta = float(m.group(2))
            return ("outdoor", f"{A}_{beta}", (A, beta))
    return ("unknown", "", None)


def classify_haze_level(file_path):
    dataset_type, _, params = extract_haze_params(file_path)
    if dataset_type == "indoor":
        haze_level = params
        if 1 <= haze_level <= 3:
            return "thin"
        elif 4 <= haze_level <= 6:
            return "medium"
        elif 7 <= haze_level <= 10:
            return "thick"
    elif dataset_type == "outdoor":
        A, beta = params
        score = A * beta
        if score < 0.08:
            return "thin"
        elif score < 0.12:
            return "medium"
        else:
            return "thick"
    return "unknown"


def scene_id_from_hazy_path(hazy_path):
    """'./SOTS/indoor/hazy/1408_1.png' -> '1408' (matches gt/1408.png)
    './SOTS/outdoor/hazy/0364_0.85_0.12.jpg' -> '0364' (matches gt/0364.png)
    Falls back to the leading digit run of the filename stem."""
    fname = os.path.basename(hazy_path)
    m = re.match(r'^(\d+)_', fname)
    if m:
        return m.group(1)
    return os.path.splitext(fname)[0]


def domain_from_path(path):
    if "indoor" in path:
        return "indoor"
    if "outdoor" in path:
        return "outdoor"
    return "unknown"


# ---------------------------------------------------------------------
# Evaluation on a split txt file. Mirrors
# experiments/collect_paper_data.py:evaluate_pytorch, with gt_path and
# scene_id added to each record (needed by exp6's bootstrap).
# ---------------------------------------------------------------------
def evaluate_split(model, split_txt, device, max_samples=None, progress=True):
    """Run `model` over every (hazy, gt) pair listed in split_txt.
    Returns a list of per-sample dict records:
        {hazy_path, gt_path, scene_id, domain, haze_level, psnr, ssim}
    """
    import torch
    from metrics import compute_psnr, compute_ssim
    from PIL import Image

    with open(split_txt) as f:
        lines = [l.strip() for l in f.read().strip().split("\n") if l.strip()]
    pairs = [l.split() for l in lines]
    if max_samples is not None:
        pairs = pairs[:max_samples]

    model.eval()
    records = []
    iterator = pairs
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(pairs, desc=os.path.basename(split_txt), ncols=80)
        except ImportError:
            pass

    with torch.no_grad():
        for hazy_path, gt_path in iterator:
            I_pil = Image.open(hazy_path).convert("RGB")
            J_pil = Image.open(gt_path).convert("RGB")
            I_np = np.asarray(I_pil).astype(np.float32) / 255.0
            J_np = np.asarray(J_pil).astype(np.float32) / 255.0
            I_t = torch.from_numpy(np.transpose(I_np, (2, 0, 1))).unsqueeze(0).to(device)
            out = model(I_t)
            J_pred = out[-1] if isinstance(out, (list, tuple)) else out
            J_pred_np = np.transpose(J_pred[0].cpu().numpy(), (1, 2, 0))
            J_pred_np = np.clip(J_pred_np, 0, 1)

            records.append({
                "hazy_path": hazy_path,
                "gt_path": gt_path,
                "scene_id": scene_id_from_hazy_path(hazy_path),
                "domain": domain_from_path(hazy_path),
                "haze_level": classify_haze_level(hazy_path),
                "psnr": float(compute_psnr(J_pred_np, J_np)),
                "ssim": float(compute_ssim(J_pred_np, J_np)),
            })
    return records


def summarize(records):
    if not records:
        return {"N": 0, "psnr_mean": float("nan"), "ssim_mean": float("nan")}
    psnrs = [r["psnr"] for r in records]
    ssims = [r["ssim"] for r in records]
    return {
        "N": len(records),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
    }


def default_split_paths():
    return {
        "indoor": repo_path("SOTS", "split_txt", "indoor_test.txt"),
        "outdoor": repo_path("SOTS", "split_txt", "outdoor_test.txt"),
    }


# ---------------------------------------------------------------------
# Self-test: exercises everything above that does NOT need torch/a GPU/
# the real SOTS dataset, using tiny synthetic images. Run directly:
#     python common.py --selftest
# ---------------------------------------------------------------------
def _selftest():
    print("[selftest] repo root resolution ...")
    root = find_repo_root()
    print(f"  -> {root}")
    ensure_repo_on_path(root)
    import physics as P
    from metrics import compute_psnr
    print("[selftest] physics.py + metrics.py import OK")

    rng = np.random.RandomState(0)
    I = rng.rand(48, 48, 3).astype(np.float32)
    J = rng.rand(48, 48, 3).astype(np.float32)
    t = (rng.rand(48, 48).astype(np.float32) * 0.6 + 0.2)
    A_inv = P.estimate_A_from_pair(I, J, t)
    fceil = P.recover_scene(I, t, A_inv)
    assert fceil.shape == I.shape
    ceiling_psnr = compute_psnr(fceil, J)
    print(f"[selftest] Definition 2/3 pipeline OK, dummy ceiling PSNR={ceiling_psnr:.2f} dB")

    print("[selftest] scene_id / haze_level parsing ...")
    cases = [
        ("./SOTS/indoor/hazy/1408_1.png", "1408", "thin"),
        ("./SOTS/indoor/hazy/1408_7.png", "1408", "thick"),
        ("./SOTS/outdoor/hazy/0364_0.85_0.12.jpg", "0364", "medium"),
    ]
    for path, exp_scene, exp_tier in cases:
        sid = scene_id_from_hazy_path(path)
        tier = classify_haze_level(path)
        status = "OK" if (sid == exp_scene and tier == exp_tier) else "MISMATCH"
        print(f"    {path} -> scene={sid} tier={tier}  [{status}]")
        assert sid == exp_scene, f"expected scene {exp_scene}, got {sid}"
        assert tier == exp_tier, f"expected tier {exp_tier}, got {tier}"

    print("[selftest] checkpoint registry resolution (search, not loading):")
    for name, entry in CHECKPOINT_REGISTRY.items():
        try:
            p = find_checkpoint_file(entry["pth"])
            print(f"    {name:16s} FOUND     {p}")
        except FileNotFoundError:
            print(f"    {name:16s} not found (expected unless you ran this next to your own experiments/ folder)")

    print("\n[selftest] ALL CHECKS PASSED (torch-dependent loading not exercised here)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repo-root", type=str, default=None)
    args = ap.parse_args()
    if args.repo_root:
        ensure_repo_on_path(args.repo_root)
    if args.selftest:
        _selftest()
    else:
        print(__doc__)
