"""selftest_release.py -- sanity-checks the pgdehazenet/ code release.

Run from inside this directory:

    cd pgdehazenet/
    python selftest_release.py
    python selftest_release.py --checkpoints /path/to/dehaze-results-release/main_results/checkpoints

It never trains anything and never touches the SOTS dataset -- it only
imports modules, builds models, and (optionally) loads existing .pth
checkpoints. Expected running time: a few seconds, well under a minute.

What it checks, and why each check exists:

  1. Every .py file in the release imports cleanly.
  2. PGDehazeNet builds for all 6 sprint configs + the DAF-only config,
     with the expected total parameter count (7,617 for the plain V3-t
     baseline) -- this is the one paper number that could not be verified
     without a PyTorch environment (see TRACEABILITY.md).
  3. A forward pass runs end to end on dummy input for one config, to
     catch shape bugs the pruning pass could have introduced.
  4. PatchDiscriminator's parameter count is 9,521 (~9,500), matching the
     fixed docstring in train.py and Table cost_summary.
  5. Setting a removed-architecture-search flag (use_repconv=True,
     channel_attn_mode='j') raises NameError, as documented in the
     network.py module docstring and NAMING.md -- i.e. confirms the
     "unsupported paths fail loudly" design actually works, rather than
     silently doing the wrong thing.
  6. If --checkpoints points at the released .pth files, each one is
     loaded with strict=True into the exact config from
     experiments/common.py:CHECKPOINT_REGISTRY, confirming this release's
     pruned network.py is still 100% state_dict-compatible with the
     checkpoints being distributed alongside it.
  7. Runs the three offline, no-GPU/no-dataset --selftest entry points
     that already existed in experiments/common.py,
     experiments/third_party_separation.py, experiments/scene_bootstrap.py.

Exit code is 0 if everything passed, 1 otherwise -- safe to use in a CI
step if you set one up.
"""
import argparse
import os
import subprocess
import sys
import traceback

RESULTS = []  # (name, ok: bool, detail: str)


def check(name):
    """Decorator: run fn, record pass/fail, never let one check kill the rest."""
    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((name, True, detail or "ok"))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn
    return wrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default=None,
                        help="Path to dehaze-results-release/main_results/checkpoints "
                             "(optional -- skips check 6 if not given)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(here, "network.py")):
        print("ERROR: run this from inside the pgdehazenet/ directory "
              "(the one containing network.py).")
        sys.exit(1)
    sys.path.insert(0, here)

    # -- 1. import every module -------------------------------------
    modules = [
        "physics", "network", "metrics", "featuremaps", "dataset_reside",
        "train", "predict", "unified_ground_truth_eval",
        "theta_star_model", "a_quality_compare", "a_spatial_stats",
    ]

    @check("imports: top-level modules")
    def _():
        import importlib
        failed = []
        for m in modules:
            try:
                importlib.import_module(m)
            except Exception as e:
                failed.append(f"{m} ({type(e).__name__}: {e})")
        if failed:
            raise RuntimeError("failed to import: " + "; ".join(failed))
        return f"{len(modules)} modules imported cleanly"

    exp_dir = os.path.join(here, "experiments")
    sys.path.insert(0, exp_dir)
    exp_modules = [
        "common", "aod_net", "lightdehazenet_net", "third_party_separation",
        "lpips_eval", "daf_only_ablation", "scene_bootstrap",
        "ceiling_error_control", "render_tables",
    ]

    @check("imports: experiments/ modules")
    def _():
        import importlib
        failed = []
        for m in exp_modules:
            try:
                importlib.import_module(m)
            except Exception as e:
                failed.append(f"{m} ({type(e).__name__}: {e})")
        if failed:
            raise RuntimeError("failed to import: " + "; ".join(failed))
        return f"{len(exp_modules)} modules imported cleanly"

    # -- 2. build all 7 reported configs, check param counts --------
    import network as N
    sys.path.insert(0, exp_dir)
    from common import CHECKPOINT_REGISTRY, _BASE_CFG, register_daf_only_checkpoint  # type: ignore
    register_daf_only_checkpoint("model_C1-DepthOnly_V3-t.pth")

    # 6 configs come straight from the registry (authoritative -- avoids
    # this script's own copy ever drifting from common.py's). The 7th,
    # C1-DepthOnly, is only registered dynamically after
    # daf_only_ablation.py actually trains it, so it's added here using
    # the same config register_daf_only_checkpoint() would use.
    SPRINT_CONFIGS = {name: info["config"] for name, info in CHECKPOINT_REGISTRY.items()}
    SPRINT_CONFIGS["C1-DepthOnly"] = dict(
        _BASE_CFG, use_depth_attn=True, film_mode="None", use_t_affine=False
    )

    built_models = {}

    @check("network.py: all 7 reported configs build + A-Baseline param count")
    def _():
        counts = {}
        for name, kwargs in SPRINT_CONFIGS.items():
            m = N.PGDehazeNet(**kwargs)
            counts[name] = sum(p.numel() for p in m.parameters())
            built_models[name] = m
        base = counts["A-Baseline"]
        detail = ", ".join(f"{k}={v}" for k, v in counts.items())
        if base != 7617:
            raise AssertionError(
                f"A-Baseline has {base} parameters, paper says 7,617 "
                f"(all configs: {detail})"
            )
        return f"A-Baseline=7617 as expected. All counts: {detail}"

    # -- 3. forward pass on dummy input -------------------------------
    @check("network.py: forward pass on dummy 256x256 input (C-Depth config)")
    def _():
        import torch
        m = built_models.get("C-Depth") or N.PGDehazeNet(**SPRINT_CONFIGS["C-Depth"])
        m.eval()
        with torch.no_grad():
            dummy = torch.rand(1, 3, 256, 256)
            out = m(dummy)
        return f"forward pass ok, output type={type(out)}"

    # -- 4. discriminator param count ---------------------------------
    @check("train.py: PatchDiscriminator param count == 9521")
    def _():
        import train as T
        d = T.PatchDiscriminator()
        n = sum(p.numel() for p in d.parameters())
        if n != 9521:
            raise AssertionError(f"got {n}, expected 9521")
        return f"{n} parameters, matches paper's 'about 9,500'"

    # -- 5. removed-architecture-search flags fail loudly, as designed
    @check("network.py: removed flags raise NameError as documented")
    def _():
        import torch  # noqa: F401
        raised = []
        for kwargs in [
            dict(capacity_mode="V3-t", use_repconv=True),
            dict(capacity_mode="V3-t", channel_attn_mode="j"),
        ]:
            try:
                N.PGDehazeNet(**kwargs)
                raised.append(f"{kwargs}: did NOT raise (unexpected)")
            except NameError:
                pass
            except Exception as e:
                raised.append(f"{kwargs}: raised {type(e).__name__} instead of NameError")
        if raised:
            raise AssertionError("; ".join(raised))
        return "both removed-flag configs raised NameError as expected"

    # -- 6. optional: load real checkpoints ---------------------------
    if args.checkpoints:
        @check(f"checkpoints: strict load from {args.checkpoints}")
        def _():
            import torch
            results = []
            for name, info in CHECKPOINT_REGISTRY.items():
                path = os.path.join(args.checkpoints, info["pth"])
                if not os.path.exists(path):
                    results.append(f"{name}: {info['pth']} not found, skipped")
                    continue
                m = N.PGDehazeNet(**info["config"])
                sd = torch.load(path, map_location="cpu")
                m.load_state_dict(sd, strict=True)
                results.append(f"{name}: OK ({info['pth']})")
            return "; ".join(results)
    else:
        RESULTS.append(("checkpoints: strict load", True,
                        "skipped (pass --checkpoints <dir> to run this check)"))

    # -- 7. existing offline selftests --------------------------------
    for script, cwd in [
        (os.path.join(exp_dir, "common.py"), here),
        (os.path.join(exp_dir, "third_party_separation.py"), here),
        (os.path.join(exp_dir, "scene_bootstrap.py"), here),
    ]:
        name = f"subprocess selftest: {os.path.basename(script)} --selftest"

        @check(name)
        def _(script=script, cwd=cwd):
            proc = subprocess.run(
                [sys.executable, script, "--selftest"],
                cwd=cwd, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"exit code {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
                    f"stderr:\n{proc.stderr[-2000:]}"
                )
            return "exit code 0"

    # -- summary -------------------------------------------------------
    print("\n" + "=" * 70)
    print("SELFTEST SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}")
        print(f"       {detail}")
    print("-" * 70)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
