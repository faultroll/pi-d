#!/usr/bin/env python3
"""
third_party_separation.py
==============================
"Must" experiment #3 (review Section 3.4 / Section 6 item 3): redesigned
third-party validation, with the K*/gradient-coefficient separation and a
constant-output control the original Table 1/2 stress tests were missing.

Ships with two real, bundled, pretrained third-party networks so this runs
with zero extra downloads:
  - AOD-Net (Li et al., ICCV 2017), weights from
    https://github.com/weber0522bb/AODnet-by-pytorch
    (third_party_models/AODnet-by-pytorch/model_pretrained/AOD_net_epoch_relu_10.pth)
  - LightDehazeNet (Ullah et al., IEEE TIP 2021), official weights from
    https://github.com/hayatkhan8660-maker/Light-DehazeNet
    (third_party_models/Light-DehazeNet/trained_weights/trained_LDNet.pth)
Both architectures (aod_net.py / lightdehazenet_net.py) were
verified attribute-for-attribute against these repos' own model.py /
lightdehazeNet.py, AND against the source code embedded inside the AOD-Net
checkpoint's own legacy pickle stream (that checkpoint is a full pickled
model object, not a plain state_dict -- see aod_net.py's
load_state_dict_smart() for why that needs special handling).

Fixes the two flaws the advisor found in the original Table 1 / Table 2
stress tests (both apply to any Class-A network using the fused form
g_ASM(K;I) = K*(I-1)+b, b=1 -- i.e. AOD-Net and LightDehazeNet, per the
paper's own Example A / Section 4.3):

  Flaw 1 -- K* and the gradient coefficient g=|I-1| are NOT independent in
  the original tables: K*(x)*(1-I) = 1-J is an algebraic identity, so
  sweeping t (Table 2) or J (Table 1) moves both at once. This script
  instead solves, for any DESIRED (K*, g) pair, the (I, J, A, t) quadruple
  that realises it exactly -- letting you fix one and sweep the other.

  Flaw 2 -- the original tables never checked whether a trivial
  "constant output" model reproduces the observed error curve. This script
  always computes and reports that control alongside the real network.

Two independent sweeps, matching the advisor's own worked example exactly
(K*=1.50, g swept 0.45->0.05, A=0.98 -- see design_grid() docstring for the
closed-form derivation and a numeric self-check against those exact numbers):

  Sweep A: K* FIXED, g swept -> a genuine Problem 1 (gradient reachability)
      probe. I actually varies row to row (I=1-g), so K_hat can in
      principle respond differently at each point.

  Sweep B: g FIXED, K* swept -> NOT a capacity/Problem-2(b) test, despite
      that being the original intent. g=|I-1| is a deterministic function
      of I alone, and both networks take only I as input, so "g fixed" is
      mathematically identical to "I fixed" -- K_hat is therefore
      GUARANTEED to be identical across every row of this sweep, for ANY
      network, regardless of quality or capacity (this script checks that
      at runtime and tells you so). What Sweep B actually gives is a
      clean, constructive proof of the single-observation
      non-identifiability the review's Section 3.4 remediation item #5
      already raised: one I admits many equally "correct" target K*
      values, so no function of I alone -- however capable -- could ever
      satisfy all of them simultaneously. For an actual dynamic-range /
      capacity probe, use --real-scatter instead: real images give many
      different TRUE K* values at similar (not bit-identical) I, because
      real scenes vary in ways a synthetic constant-colour image cannot.

Usage (no checkpoint needed -- prints/saves the design tables only,
fully offline, use this to sanity-check the experimental design itself):
    python third_party_separation.py --dry-run

Usage (against the bundled real weights -- no path arguments needed):
    python third_party_separation.py --network aod
    python third_party_separation.py --network lightdehaze

Usage (strongest version -- real images, pixel-level scatter, needs SOTS/):
    python third_party_separation.py --network aod --real-scatter \\
        --split indoor_test --n-images 10

Usage (pointing at your own copies instead of the bundled ones):
    python third_party_separation.py --network aod \\
        --checkpoint /path/to/AOD_net_epoch_relu_10.pth \\
        --aod-repo-dir /path/to/AODnet-by-pytorch
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_repo_on_path, repo_path, default_split_paths


# =======================================================================
# 1. Design-grid construction (pure math, no torch) -- THE fix for Flaw 1.
# =======================================================================
def design_grid(mode, fixed_value, sweep_values, A=0.98):
    """Build a list of (K_star, g, I, J, A, t) points that all satisfy the
    ASM exactly, with K_star and g independently controlled.

    Derivation (fused form g_ASM(K;I) = K*(I-1)+b, b=1, so
    J = K*(I-1)+1  <=>  K* = (1-J)/(1-I), and writing g := |I-1| = 1-I for
    I<1):
        1 - J = K* * g                      (*)
    Given desired (K*, g):
        I = 1 - g
        J = 1 - K*g                          (from *)
    Any A can then be paired with (I, J) and the unique t solving the ASM
    I = tJ + (1-t)A recovered as:
        t = (I - A) / (J - A)

    mode="fix_kstar_sweep_g": fixed_value is K*, sweep_values are g's.
        A genuine Problem 1 (gradient reachability) probe: I=1-g varies
        row to row, so the network actually sees different inputs.

    mode="fix_g_sweep_kstar": fixed_value is g,  sweep_values are K*'s.
        CAVEAT: since g=|I-1| depends only on I, and these networks take
        only I as input, this holds I fixed across the whole sweep --
        K_hat is therefore mathematically guaranteed to be identical at
        every point, for any network. This mode does not test capacity;
        see the module docstring's "Sweep B" note for what it does show.

    Self-check: design_grid("fix_kstar_sweep_g", 1.50, [0.45,0.30,0.20,0.10,0.05])
    reproduces the advisor's own worked table (I/J/t below match to 3dp):
        g=0.45 -> I=0.550 J=0.325 t=0.656
        g=0.30 -> I=0.700 J=0.550 t=0.651
        g=0.20 -> I=0.800 J=0.700 t=0.643
        g=0.10 -> I=0.900 J=0.850 t=0.615
        g=0.05 -> I=0.950 J=0.925 t=0.545
    """
    rows = []
    for sweep_val in sweep_values:
        if mode == "fix_kstar_sweep_g":
            k_star, g = fixed_value, sweep_val
        elif mode == "fix_g_sweep_kstar":
            g, k_star = fixed_value, sweep_val
        else:
            raise ValueError(mode)

        I = 1.0 - g
        J = 1.0 - k_star * g
        if abs(J - A) < 1e-6:
            t = float("nan")  # degenerate: J==A, ASM under-determined here
        else:
            t = (I - A) / (J - A)

        # self-check the algebraic identity K* * g == 1 - J
        assert abs(k_star * g - (1 - J)) < 1e-9

        rows.append({
            "K_star": round(k_star, 6), "g": round(g, 6),
            "I": round(I, 6), "J": round(J, 6), "A": round(A, 6),
            "t": round(t, 6) if not np.isnan(t) else None,
            "t_in_valid_range": bool(0.0 <= t <= 1.0) if not np.isnan(t) else False,
        })
    return rows


def _selfcheck_against_advisor_table():
    """Numeric regression test against the advisor's own hand-derived
    table (review Section 3.4, remediation item #1). Run with --selftest."""
    rows = design_grid("fix_kstar_sweep_g", 1.50, [0.45, 0.30, 0.20, 0.10, 0.05], A=0.98)
    expected = [
        (0.550, 0.325, 0.656), (0.700, 0.550, 0.651), (0.800, 0.700, 0.643),
        (0.900, 0.850, 0.615), (0.950, 0.925, 0.545),
    ]
    print("Self-check against advisor's worked table (K*=1.50, A=0.98):")
    ok = True
    for r, (exp_I, exp_J, exp_t) in zip(rows, expected):
        match = (abs(r["I"] - exp_I) < 5e-4 and abs(r["J"] - exp_J) < 5e-4
                  and abs(r["t"] - exp_t) < 5e-4)
        ok = ok and match
        print(f"  g={r['g']:.2f}: I={r['I']:.3f} (exp {exp_I}), "
              f"J={r['J']:.3f} (exp {exp_J}), t={r['t']:.3f} (exp {exp_t})  "
              f"[{'OK' if match else 'MISMATCH'}]")
    print("PASSED" if ok else "FAILED")
    return ok


# =======================================================================
# 2. Network loading. Both architectures + get_K() live in their own
#    modules now (aod_net.py / lightdehazenet_net.py), matching
#    the two reference diagnostic scripts attribute-for-attribute -- see
#    those files' docstrings for how each was verified. This module only
#    wires up default paths to the bundled third_party_models/ checkpoints.
# =======================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATHS = {
    "aod": {
        "checkpoint": os.path.join(THIS_DIR, "third_party_models", "AODnet-by-pytorch",
                                    "model_pretrained", "AOD_net_epoch_relu_10.pth"),
        "repo_dir": os.path.join(THIS_DIR, "third_party_models", "AODnet-by-pytorch"),
    },
    "lightdehaze": {
        "checkpoint": os.path.join(THIS_DIR, "third_party_models", "Light-DehazeNet",
                                    "trained_weights", "trained_LDNet.pth"),
        "repo_dir": None,  # plain state_dict, no repo needed to unpickle
    },
}


def load_network(kind, checkpoint=None, device=None, aod_repo_dir=None):
    """Returns (model, get_K_fn) where get_K_fn(model, img_tensor) -> K map.
    checkpoint/aod_repo_dir default to the bundled third_party_models/ copies
    if not given."""
    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or DEFAULT_PATHS[kind]["checkpoint"]

    if kind == "aod":
        from aod_net import AODNet, get_K, load_state_dict_smart
        model = AODNet().to(device)
        repo_dir = aod_repo_dir or DEFAULT_PATHS["aod"]["repo_dir"]
        load_state_dict_smart(model, checkpoint, extra_syspath=repo_dir, device=device)
    elif kind == "lightdehaze":
        from lightdehazenet_net import LightDehazeNet, get_K, load_state_dict_smart
        model = LightDehazeNet().to(device)
        load_state_dict_smart(model, checkpoint, device=device)
    else:
        raise ValueError(kind)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load_network] {kind}: loaded {checkpoint} ({n_params} params)")
    return model, get_K


# =======================================================================
# 3. Run the stress test grid against a real network + the constant-output
#    control, for one sweep (list of design rows from design_grid()).
# =======================================================================
def run_grid_against_network(rows, model, get_K_fn, device, image_size=64):
    import torch
    out_rows = []
    with torch.no_grad():
        for r in rows:
            I_val = r["I"]
            I_tensor = torch.full((1, 3, image_size, image_size), I_val,
                                   dtype=torch.float32, device=device)
            k_map = get_K_fn(model, I_tensor)
            k_hat = float(k_map.mean().item())
            k_hat_per_channel = [float(k_map[0, c].mean().item()) for c in range(k_map.shape[1])]
            row = dict(r)
            row["K_hat"] = round(k_hat, 6)
            row["K_hat_per_channel"] = [round(v, 6) for v in k_hat_per_channel]
            row["abs_error"] = round(abs(k_hat - r["K_star"]), 6)
            out_rows.append(row)
    return out_rows


def add_constant_control(rows):
    """Fit the crudest possible constant-output model (mean of the REAL
    network's own K_hat over this sweep) and report its error curve
    alongside the real network's, plus how much of the real curve's
    variance the constant explains (R^2) -- the number that answers
    "is this evidence of gradient degeneracy, or just output saturation?"."""
    k_hats = [r["K_hat"] for r in rows]
    k_const = float(np.mean(k_hats))
    for r in rows:
        r["K_const_control"] = round(k_const, 6)
        r["abs_error_const_control"] = round(abs(k_const - r["K_star"]), 6)

    real_err = np.array([r["abs_error"] for r in rows])
    const_err = np.array([r["abs_error_const_control"] for r in rows])
    # R^2 of const-control's error curve explaining the real network's error curve
    if np.std(real_err) > 1e-9:
        corr = float(np.corrcoef(real_err, const_err)[0, 1])
        r2 = corr ** 2
    else:
        corr, r2 = float("nan"), float("nan")
    summary = {
        "k_const_fitted": round(k_const, 6),
        "mean_abs_error_real": round(float(np.mean(real_err)), 6),
        "mean_abs_error_const_control": round(float(np.mean(const_err)), 6),
        "corr_real_vs_const_error_curves": round(corr, 4) if not np.isnan(corr) else None,
        "r2_real_vs_const_error_curves": round(r2, 4) if not np.isnan(r2) else None,
    }
    return rows, summary


def print_grid(title, rows, with_network=False):
    print(f"\n{title}")
    if with_network:
        header = f"  {'K*':>7} {'g=|I-1|':>8} {'I':>7} {'J':>7} {'A':>6} {'t':>7} " \
                  f"{'K_hat':>8} {'|err|':>8} {'K_const':>8} {'|err|_const':>11}"
    else:
        header = f"  {'K*':>7} {'g=|I-1|':>8} {'I':>7} {'J':>7} {'A':>6} {'t':>7} {'t_valid':>7}"
    print(header)
    for r in rows:
        if with_network:
            print(f"  {r['K_star']:>7.3f} {r['g']:>8.3f} {r['I']:>7.3f} {r['J']:>7.3f} "
                  f"{r['A']:>6.3f} {(r['t'] if r['t'] is not None else float('nan')):>7.3f} "
                  f"{r['K_hat']:>8.3f} {r['abs_error']:>8.3f} "
                  f"{r['K_const_control']:>8.3f} {r['abs_error_const_control']:>11.3f}")
        else:
            t_str = f"{r['t']:.3f}" if r["t"] is not None else "  n/a"
            print(f"  {r['K_star']:>7.3f} {r['g']:>8.3f} {r['I']:>7.3f} {r['J']:>7.3f} "
                  f"{r['A']:>6.3f} {t_str:>7} {str(r['t_in_valid_range']):>7}")


# =======================================================================
# 4. Optional strongest-evidence mode: real images, per-pixel (K*, g)
#    scatter against the network's real per-pixel K_hat, entirely inside
#    the training distribution (review Section 3.4, remediation item #5).
# =======================================================================
def run_real_image_scatter(model, get_K_fn, split_name, n_images, device, seed=0):
    import torch
    from PIL import Image
    import physics as P

    splits = default_split_paths()
    split_key = "indoor" if "indoor" in split_name else "outdoor"
    split_path = splits[split_key]
    if not os.path.exists(split_path):
        return {"error": f"split file not found: {split_path}"}

    with open(split_path) as f:
        pairs = [l.split() for l in f.read().strip().split("\n") if l.strip()]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pairs), size=min(n_images, len(pairs)), replace=False)

    all_k_star, all_g, all_k_hat = [], [], []
    with torch.no_grad():
        for i in idx:
            hazy_path, gt_path = pairs[i]
            I_np = np.asarray(Image.open(hazy_path).convert("RGB")).astype(np.float32) / 255.0
            J_np = np.asarray(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0
            t_cap, A_cap, _ = P.run_cap(I_np)

            # per-pixel K* from the CAP prior's own (t, A) via the ASM-consistent
            # closed form for the fused representation: K* = (1-J)/(1-I) (mean
            # over channels, since the network emits one scalar K per pixel).
            eps = 1e-3
            K_star_map = (1.0 - J_np.mean(axis=-1)) / (1.0 - I_np.mean(axis=-1) + eps)
            g_map = np.abs(I_np.mean(axis=-1) - 1.0)

            I_t = torch.from_numpy(np.transpose(I_np, (2, 0, 1))).unsqueeze(0).to(device)
            k_map = get_K_fn(model, I_t)
            K_hat_map = k_map[0].mean(dim=0).cpu().numpy()

            # subsample pixels for a tractable scatter (every 8th pixel)
            all_k_star.append(K_star_map[::8, ::8].ravel())
            all_g.append(g_map[::8, ::8].ravel())
            all_k_hat.append(K_hat_map[::8, ::8].ravel())

    k_star = np.concatenate(all_k_star)
    g = np.concatenate(all_g)
    k_hat = np.concatenate(all_k_hat)

    # Key diagnostic numbers:
    #  - std_ratio: how much of K*'s required dynamic range K_hat actually
    #    uses. std_ratio near 0 means the network is behaving like a
    #    constant-output model regardless of what K* calls for.
    #  - corr_khat_vs_kstar: does what little K_hat DOES move, move in the
    #    right direction?
    #  - corr_abs_error_vs_g: the real-image version of the Problem-1
    #    stress test -- does |K_hat - K*| grow as the gradient coefficient
    #    g=|I-1| shrinks, on real in-distribution pixels (not just the
    #    hand-picked scalar sweep)?
    std_ratio = float(k_hat.std() / (k_star.std() + 1e-9))
    corr_khat_kstar = float(np.corrcoef(k_hat, k_star)[0, 1]) if len(k_hat) > 1 else float("nan")
    abs_error = np.abs(k_hat - k_star)
    corr_error_vs_g = float(np.corrcoef(abs_error, g)[0, 1]) if len(g) > 1 else float("nan")

    # --- Constant-output control group (review comment sec 3.2) ---
    k_const = float(k_hat.mean())              # c = E[K_hat] (network output mean as control)
    abs_error_const = np.abs(k_const - k_star)
    corr_error_vs_g_const = (
        float(np.corrcoef(abs_error_const, g)[0, 1]) if len(g) > 1 else float("nan")
    )
    # MSE comparison: MSE(K_hat, K*) vs MSE(c, K*)
    # Algebraic check: MSE_net - MSE_const = Var(K_hat) - 2*Cov(K_hat, K*)
    mse_network    = float(np.mean((k_hat   - k_star) ** 2))
    mse_const      = float(np.mean((k_const - k_star) ** 2))
    mse_delta      = mse_network - mse_const   # negative = network beats constant output
    var_khat       = float(np.var(k_hat))
    cov_khat_kstar = float(np.cov(k_hat, k_star)[0, 1]) if len(k_hat) > 1 else float("nan")

    return {
        "domain": split_key, "n_images": len(idx), "n_pixels_sampled": len(k_star),
        "corr_khat_vs_kstar": round(corr_khat_kstar, 4),
        "k_hat_std_over_kstar_std": round(std_ratio, 4),
        "corr_abs_error_vs_g": round(corr_error_vs_g, 4),
        "k_hat_mean": round(float(k_hat.mean()), 4),
        "k_hat_std": round(float(k_hat.std()), 4),
        "k_star_mean": round(float(k_star.mean()), 4),
        "k_star_std": round(float(k_star.std()), 4),
        "k_const_control":             round(k_const, 4),
        "corr_abs_error_vs_g_const":   round(corr_error_vs_g_const, 4),
        "mse_network_vs_kstar":        round(mse_network, 6),
        "mse_const_vs_kstar":          round(mse_const, 6),
        "mse_delta":                   round(mse_delta, 6),
        "var_khat":                    round(var_khat, 6),
        "cov_khat_kstar":              round(cov_khat_kstar, 6),
        "interpretation": (
            "corr near 0 (or K_hat_std << K_star_std) means the network's output "
            "barely moves with the theoretically-required target even across real, "
            "in-distribution pixels -- the constant-output failure mode, not "
            "gradient degeneracy specifically."
        ),
    }


# =======================================================================
# main
# =======================================================================
# =======================================================================
# Interpretation helpers -- adaptive, not hardcoded: each one checks what
# the numbers actually show before printing a conclusion, so the message
# stays correct even if a future network/weights combination behaves
# differently from AOD-Net/LightDehazeNet.
# =======================================================================
def _explain_sweep_a(sweep_a_net, summary_a):
    print("\nReading Sweep A (K* fixed, g swept -- genuine Problem 1 probe):")
    k_hats = [r["K_hat"] for r in sweep_a_net]
    k_star = sweep_a_net[0]["K_star"]
    all_below = all(k < k_star for k in k_hats)
    all_above = all(k > k_star for k in k_hats)
    if all_below or all_above:
        side = "below" if all_below else "above"
        print(f"  K_hat stays entirely {side} K*={k_star} across the whole sweep (never crosses it).")
        print("  That is WHY mean_abs_error_real == mean_abs_error_const_control exactly here --")
        print("  it is an algebraic identity (mean of same-signed deviations = |deviation of the")
        print("  mean|), not new evidence of gradient degeneracy by itself. What it DOES show:")
        if k_hats[-1] != k_hats[0]:
            print(f"  K_hat still moves with I ({k_hats[0]:.3f} -> {k_hats[-1]:.3f} as g -> 0), just")
            print("  never far enough to approach K* -- weak-but-real sensitivity, not a frozen output.")
        else:
            print("  K_hat does not move at all across the sweep -- consistent with a frozen output.")
    else:
        print(f"  K_hat crosses K*={k_star} somewhere in this sweep -- the mean-error identity above")
        print("  does NOT apply mechanically here; a real r2 comparison (if computed) is meaningful.")
        if summary_a.get("r2_real_vs_const_error_curves") is not None:
            print(f"  r2_real_vs_const_error_curves = {summary_a['r2_real_vs_const_error_curves']}")


def _explain_sweep_b(sweep_b_net, summary_b):
    print("\nReading Sweep B (g fixed, K* swept):")
    k_hats = [r["K_hat"] for r in sweep_b_net]
    spread = max(k_hats) - min(k_hats)
    if spread < 1e-4:
        print(f"  K_hat is identical to within {spread:.2e} across all {len(k_hats)} rows -- exactly")
        print("  as guaranteed, since g fixed means I fixed for an I-only-input network. This is")
        print("  NOT evidence about this network's capacity; it is a constructive proof of the")
        print("  single-observation non-identifiability the review already raised (Section 3.4,")
        print("  remediation item #5): one I admits many equally-valid target K* values, so no")
        print("  function of I alone could satisfy all of them. Do not cite r2/corr from this")
        print("  sweep as Problem-2(b) evidence -- use --real-scatter for an actual capacity probe.")
    else:
        print(f"  K_hat varies by {spread:.4f} across the sweep despite g (hence I) being fixed --")
        print("  unexpected for a pure I-only network; double check whether the loaded model takes")
        print("  any input besides I, or whether this run used --image-size 1 / a non-constant")
        print("  input construction. If confirmed I-only, this spread is likely numerical noise.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=str, default=None,
                     help="pgdehazenet/ location, only needed for --real-scatter (which uses "
                          "physics.py + SOTS/); --network alone works without it")
    ap.add_argument("--network", choices=["aod", "lightdehaze"], default=None,
                     help="which third-party network to stress-test (omit for --dry-run design-only mode)")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="path to trained weights for --network; defaults to the bundled "
                          "third_party_models/ copy if not given")
    ap.add_argument("--aod-repo-dir", type=str, default=None,
                     help="only for --network aod: directory containing AODnet-by-pytorch's own "
                          "model.py, needed to unpickle its full-model-object checkpoint; "
                          "defaults to the bundled third_party_models/AODnet-by-pytorch")
    ap.add_argument("--dry-run", action="store_true",
                     help="print/save the design grids only, no network needed")
    ap.add_argument("--kstar-fixed", type=float, default=1.50)
    ap.add_argument("--g-sweep", type=float, nargs="+", default=[0.45, 0.30, 0.20, 0.10, 0.05])
    ap.add_argument("--g-fixed", type=float, default=0.20)
    ap.add_argument("--kstar-sweep", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                     help="default range chosen so t stays in [0,1] at g=0.20, A=0.98 (all points physically realisable)")
    ap.add_argument("--A", type=float, default=0.98)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--real-scatter", action="store_true",
                     help="also run the per-pixel real-image scatter check (needs --repo-root "
                          "pointed at pgdehazenet/ with SOTS/ populated)")
    ap.add_argument("--split", type=str, default="indoor_test", choices=["indoor_test", "outdoor_test"])
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--selftest", action="store_true", help="verify design_grid() against the advisor's worked example and exit")
    ap.add_argument("--check-arch", action="store_true",
                     help="print parameter counts + gradient-formula self-checks for both bundled "
                          "architectures and exit (no checkpoint loading, no SOTS/ needed)")
    args = ap.parse_args()

    if args.selftest:
        ok = _selfcheck_against_advisor_table()
        sys.exit(0 if ok else 1)

    if args.check_arch:
        from aod_net import AODNet, check_gradient_formula as check_aod_grad
        from lightdehazenet_net import LightDehazeNet
        for name, cls, expected in [("AOD-Net", AODNet, 1761), ("LightDehazeNet", LightDehazeNet, 30187)]:
            m = cls()
            n = sum(p.numel() for p in m.parameters())
            print(f"{name}: {n} params "
                  f"({'matches' if n == expected else 'DOES NOT MATCH'} expected {expected})")
        print("\nGradient formula check (shared by both, same g_ASM form):")
        check_aod_grad()
        sys.exit(0)

    try:
        ensure_repo_on_path(args.repo_root)
        out_dir = args.output or repo_path("outputs", "exp3_thirdparty_separation")
    except RuntimeError:
        # pgdehazenet/ not found -- fine unless --real-scatter is requested; fall back to a
        # local output dir so --network alone still works fully standalone.
        if args.real_scatter:
            raise
        out_dir = args.output or os.path.join(THIS_DIR, "outputs", "exp3_thirdparty_separation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("Experiment 3: third-party validation -- independent K*/g separation")
    print("(review Section 3.4; requires K*(I-1)+b=1 fused-form networks)")
    print("=" * 72)

    sweep_a = design_grid("fix_kstar_sweep_g", args.kstar_fixed, args.g_sweep, A=args.A)
    sweep_b = design_grid("fix_g_sweep_kstar", args.g_fixed, args.kstar_sweep, A=args.A)

    results = {"sweep_A_fixed_kstar": sweep_a, "sweep_B_fixed_g": sweep_b}

    run_network = (not args.dry_run) and args.network
    if not run_network:
        print_grid(f"Sweep A -- K* fixed at {args.kstar_fixed}, g swept "
                    f"(genuine Problem 1 probe -- I varies row to row):", sweep_a)
        print_grid(f"Sweep B -- g fixed at {args.g_fixed}, K* swept "
                    f"(I is pinned across every row -- see caveat below, NOT a capacity test):", sweep_b)
        if not args.network:
            print("\n[dry-run] No --network given: design tables only. "
                  "Add --network {aod,lightdehaze} to test the bundled real weights.")
    else:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, get_K_fn = load_network(args.network, args.checkpoint, device, args.aod_repo_dir)

        sweep_a_net = run_grid_against_network(sweep_a, model, get_K_fn, device, args.image_size)
        sweep_a_net, summary_a = add_constant_control(sweep_a_net)
        sweep_b_net = run_grid_against_network(sweep_b, model, get_K_fn, device, args.image_size)
        sweep_b_net, summary_b = add_constant_control(sweep_b_net)

        print_grid(f"Sweep A -- K* fixed at {args.kstar_fixed}, g swept "
                    f"({args.network}, real weights + constant-output control):",
                   sweep_a_net, with_network=True)
        print(f"  summary: {summary_a}")
        print_grid(f"Sweep B -- g fixed at {args.g_fixed}, K* swept "
                    f"({args.network}, real weights + constant-output control):",
                   sweep_b_net, with_network=True)
        print(f"  summary: {summary_b}")

        _explain_sweep_a(sweep_a_net, summary_a)
        _explain_sweep_b(sweep_b_net, summary_b)

        results["sweep_A_network"] = sweep_a_net
        results["sweep_A_summary"] = summary_a
        results["sweep_B_network"] = sweep_b_net
        results["sweep_B_summary"] = summary_b

        if args.real_scatter:
            print(f"\nRunning real-image per-pixel scatter check ({args.split}, "
                  f"n_images={args.n_images}) ...")
            scatter = run_real_image_scatter(model, get_K_fn, args.split, args.n_images, device)
            results["real_image_scatter"] = scatter
            print(f"  {scatter}")

    out_json = os.path.join(out_dir, f"results_{args.network or 'design_only'}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
