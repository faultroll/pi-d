#!/usr/bin/env python3
"""
scene_bootstrap.py  (BONUS -- not one of the strict "items 1-5" must
list, but the review's own summary table lists it as item 6, "half a day,
computation only", directly reusable from whatever exp2/exp5 already
evaluated, so it is included here as a cheap add-on.)
==============================================================================
Review Section 4.1 / Limitations:
    "35 indoor pairs are five haze-density variants of only 7 distinct
    scenes -- these are repeated measures, not independent samples.
    Effective sample size is 35/(1+4*rho) for within-scene correlation
    rho, i.e. between 7 and 35."

This script:
  1. Groups per-sample PSNR records by scene_id (not by individual hazy
     image), so the 35 indoor records collapse to 7 scene-level units and
     the 74 outdoor records collapse to ~73 scene-level units (outdoor is
     ~1 variant/scene already, per Section 4.1).
  2. Computes a SCENE-CLUSTER bootstrap confidence interval for the
     baseline -> proposed PSNR delta: resample SCENES with replacement
     (not individual images), so within-scene correlation is respected
     rather than pretending 35 indoor images are 35 independent samples.
  3. Reports the per-scene paired differences table the review explicitly
     asks for ("report each of the 7 scenes' own Delta in an appendix table").

Needs: two sets of per-sample records with 'psnr' and 'scene_id' fields --
exactly what common.evaluate_split / exp2's evaluate_split_with_lpips
already produce. Two ways to get them:
  (a) pass --model-a/--model-b names from CHECKPOINT_REGISTRY and this
      script runs evaluation itself (needs GPU + checkpoints + SOTS/), or
  (b) pass --records-a/--records-b as JSON files already containing a
      'records' list (e.g. saved by lpips_eval.py or
      daf_only_ablation.py) -- no GPU needed, just aggregation.

Usage:
    # from two dumped exp2 JSON files, no GPU needed:
    python scene_bootstrap.py \\
        --records-a outputs/exp2_lpips_eval/lpips_results_alex.json --model-a A-Baseline \\
        --records-b outputs/exp2_lpips_eval/lpips_results_alex.json --model-b D-Prop-S42

    # or let it evaluate two checkpoints itself:
    python scene_bootstrap.py --model-a A-Baseline --model-b D-Prop-S42

    # self-test with synthetic data, no GPU / no dataset / no checkpoints:
    python scene_bootstrap.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_repo_on_path, repo_path, default_split_paths


def group_by_scene(records, domain=None):
    """records: list of {scene_id, domain, psnr, ...}. Returns
    {scene_id: [psnr, psnr, ...]} restricted to `domain` if given."""
    groups = {}
    for r in records:
        if domain is not None and r.get("domain") != domain:
            continue
        groups.setdefault(r["scene_id"], []).append(r["psnr"])
    return groups


def scene_level_table(records_a, records_b, domain):
    """Per-scene mean PSNR for model A and model B, paired by scene_id.
    Returns list of {scene_id, n_variants, psnr_a, psnr_b, delta}."""
    ga = group_by_scene(records_a, domain)
    gb = group_by_scene(records_b, domain)
    common_scenes = sorted(set(ga) & set(gb))
    if set(ga) != set(gb):
        missing_a = set(gb) - set(ga)
        missing_b = set(ga) - set(gb)
        if missing_a or missing_b:
            print(f"  WARNING [{domain}]: scene sets differ between model A and B "
                  f"(A missing {missing_a or '{}'}, B missing {missing_b or '{}'}) "
                  f"-- restricting to the {len(common_scenes)} common scenes.")
    rows = []
    for sid in common_scenes:
        psnr_a = float(np.mean(ga[sid]))
        psnr_b = float(np.mean(gb[sid]))
        rows.append({
            "scene_id": sid, "n_variants_a": len(ga[sid]), "n_variants_b": len(gb[sid]),
            "psnr_a": round(psnr_a, 3), "psnr_b": round(psnr_b, 3),
            "delta": round(psnr_b - psnr_a, 3),
        })
    return rows


def scene_cluster_bootstrap(scene_rows, n_boot=10000, seed=0, ci=0.95):
    """Resample SCENES (not images) with replacement, n_boot times, and
    report the bootstrap distribution of the mean per-scene delta. This is
    the review's requested "scene-cluster bootstrap" -- it respects the
    fact that the 5 haze-density variants of one scene are correlated, by
    never splitting a scene's variants across different bootstrap draws."""
    rng = np.random.RandomState(seed)
    deltas = np.array([r["delta"] for r in scene_rows])
    n = len(deltas)
    if n == 0:
        return {"error": "no common scenes to bootstrap"}
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = deltas[idx].mean()
    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_scenes": n,
        "point_estimate_mean_delta": round(float(deltas.mean()), 4),
        "bootstrap_mean_of_means": round(float(boot_means.mean()), 4),
        "bootstrap_std": round(float(boot_means.std()), 4),
        f"ci_{int(ci*100)}_low": round(float(lo), 4),
        f"ci_{int(ci*100)}_high": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def effective_sample_size_curve(n_images, variants_per_scene, rho_values=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """n_eff = n / (1 + (variants_per_scene - 1) * rho) for a range of
    plausible within-scene correlations -- reproduces the review's own
    "35/(1+4*rho), between 7 and 35" statement in general form."""
    rows = []
    for rho in rho_values:
        n_eff = n_images / (1 + (variants_per_scene - 1) * rho)
        rows.append({"rho": rho, "n_eff": round(n_eff, 2)})
    return rows


# =======================================================================
# Loading records: from a saved JSON (no GPU) or by running eval directly.
# =======================================================================
def load_records_from_json(path, model_name):
    with open(path) as f:
        data = json.load(f)
    if model_name not in data:
        raise KeyError(f"{model_name!r} not found in {path}. Available: {list(data.keys())}")
    entry = data[model_name]
    records = []
    for domain in ("indoor", "outdoor"):
        if domain in entry and "records" in entry[domain]:
            records.extend(entry[domain]["records"])
    if not records:
        raise ValueError(f"No per-sample 'records' found for {model_name!r} in {path}. "
                          f"Re-run the script that produced this JSON without any "
                          f"record-stripping option.")
    return records


def evaluate_model_live(model_name, device):
    from common import load_named_checkpoint, evaluate_split
    model, config, pth_path = load_named_checkpoint(model_name, device)
    splits = default_split_paths()
    records = []
    for domain, path in splits.items():
        if os.path.exists(path):
            records.extend(evaluate_split(model, path, device))
    return records


# =======================================================================
# Self-test with synthetic data (no GPU, no dataset)
# =======================================================================
def _selftest():
    print("[selftest] building synthetic per-sample PSNR records for 7 indoor "
          "scenes x 5 variants, model A vs model B ...")
    rng = np.random.RandomState(0)
    records_a, records_b = [], []
    true_scene_effects = rng.normal(0, 0.5, size=7)  # scene-level heterogeneity
    for scene_idx in range(7):
        scene_id = f"S{scene_idx}"
        for variant in range(1, 6):
            base_psnr = 18.0 + true_scene_effects[scene_idx] + rng.normal(0, 0.3)
            records_a.append({"scene_id": scene_id, "domain": "indoor", "psnr": base_psnr})
            # model B is uniformly ~2dB better, plus its own noise
            records_b.append({"scene_id": scene_id, "domain": "indoor",
                               "psnr": base_psnr + 2.0 + rng.normal(0, 0.3)})

    rows = scene_level_table(records_a, records_b, domain="indoor")
    assert len(rows) == 7, f"expected 7 scenes, got {len(rows)}"
    print(f"[selftest] scene-level table OK, {len(rows)} scenes:")
    for r in rows:
        print(f"    {r}")

    boot = scene_cluster_bootstrap(rows, n_boot=2000, seed=0)
    print(f"[selftest] bootstrap result: {boot}")
    assert boot["excludes_zero"], "expected a clearly non-zero synthetic +2dB effect to be detected"
    assert 1.5 < boot["point_estimate_mean_delta"] < 2.5

    eff = effective_sample_size_curve(35, 5)
    print(f"[selftest] effective sample size curve (35 images, 5 variants/scene): {eff}")
    assert eff[0]["n_eff"] == 35.0  # rho=0 -> fully independent -> n_eff = n
    assert abs(eff[-1]["n_eff"] - 7.0) < 1e-6  # rho=1 -> fully correlated -> n_eff = n_scenes

    print("\n[selftest] ALL CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model-a", type=str, default=None, help="e.g. A-Baseline")
    ap.add_argument("--model-b", type=str, default=None, help="e.g. D-Prop-S42")
    ap.add_argument("--records-a", type=str, default=None, help="JSON file (e.g. from exp2) containing model-a's records")
    ap.add_argument("--records-b", type=str, default=None, help="JSON file containing model-b's records")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    ensure_repo_on_path(args.repo_root)
    if not args.model_a or not args.model_b:
        print("Need --model-a and --model-b (names, used for labelling either way). "
              "Run with --selftest to see this script exercised end-to-end on synthetic data.")
        sys.exit(1)

    if args.records_a:
        records_a = load_records_from_json(args.records_a, args.model_a)
    else:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        records_a = evaluate_model_live(args.model_a, device)

    if args.records_b:
        records_b = load_records_from_json(args.records_b, args.model_b)
    else:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        records_b = evaluate_model_live(args.model_b, device)

    out = {"model_a": args.model_a, "model_b": args.model_b, "by_domain": {}}
    print("=" * 72)
    print(f"Experiment 6 (bonus): scene-cluster bootstrap, {args.model_a} -> {args.model_b}")
    print("=" * 72)
    for domain in ("indoor", "outdoor"):
        rows = scene_level_table(records_a, records_b, domain=domain)
        if not rows:
            continue
        boot = scene_cluster_bootstrap(rows, n_boot=args.n_boot, seed=args.seed)
        print(f"\n[{domain}] {len(rows)} scenes:")
        for r in rows:
            print(f"    scene {r['scene_id']:<8} n_a={r['n_variants_a']} n_b={r['n_variants_b']}  "
                  f"psnr_a={r['psnr_a']:.2f}  psnr_b={r['psnr_b']:.2f}  delta={r['delta']:+.2f}")
        print(f"  bootstrap ({args.n_boot} resamples of scenes, not images): {boot}")
        out["by_domain"][domain] = {"scene_table": rows, "bootstrap": boot}

    if "indoor" in out["by_domain"]:
        eff = effective_sample_size_curve(
            n_images=sum(r["n_variants_a"] for r in out["by_domain"]["indoor"]["scene_table"]),
            variants_per_scene=5)
        print(f"\nEffective sample size vs within-scene correlation rho (indoor): {eff}")
        out["indoor_effective_n_curve"] = eff

    out_dir = repo_path("outputs", "exp6_scene_bootstrap") if not args.output else os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_json = args.output or os.path.join(out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
