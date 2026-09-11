"""
aod_net.py
================
Official AOD-Net architecture (Li et al., "AOD-Net: All-in-One Dehazing
Network", ICCV 2017), aligned with the real, independently-run diagnostic
diagnostic script (aodnet_diagnostic_tests_v2-2.py), which already loads
real weights from https://github.com/weber0522bb/AODnet-by-pytorch and
verified the architecture layer-by-layer against the original author's
Caffe prototxt (Boyiliee/AOD-Net, test/test_template.prototxt).

The class below uses the SAME attribute names (conv1..conv5) as that
repo's own model.py, because the released checkpoint's state_dict keys are
tied to those names -- a class using different attribute names (e.g. this
file's own earlier draft, which used e_conv1..e_conv5) would fail to
load_state_dict(strict=True) against the real weights even though the
architecture is otherwise identical.

Parameter count self-check: instantiating this class and summing
parameters gives exactly 1,761 -- matching both the paper's own cited
figure and the reference diagnostic script's own printed "0. parameter count" line.

Weight loading note: the checkpoint published in weber0522bb/AODnet-by-pytorch
is a *pickled full model object* (not a plain state_dict), so unpickling it
needs that repo's own model.py importable on sys.path -- see
load_official_weights() below, which mirrors the reference script's load_real_weights()
exactly (try a safe weights_only=True load first for a plain state_dict;
fall back to weights_only=False with the repo directory added to sys.path
for the AOD-Net full-object case).
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class AODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)
        self.b = 1.0

    def forward(self, img):
        x1 = F.relu(self.conv1(img))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))
        return F.relu(k * img - k + self.b)


def get_K(model, img):
    """Forward pass, stopping right before g_ASM, returning only
    theta = K(x). Same contract as the reference diagnostic script's get_K(); does
    not reimplement the computation, just re-exposes the intermediate
    tensor network.py-style modules don't otherwise return."""
    x1 = F.relu(model.conv1(img))
    x2 = F.relu(model.conv2(x1))
    cat1 = torch.cat((x1, x2), 1)
    x3 = F.relu(model.conv3(cat1))
    cat2 = torch.cat((x2, x3), 1)
    x4 = F.relu(model.conv4(cat2))
    cat3 = torch.cat((x1, x2, x3, x4), 1)
    return F.relu(model.conv5(cat3))


def K_star(I, A, t, b=1.0, eps=1e-9):
    """Closed-form theta* for the fused form g_ASM(K;I) = K(I-1)+b.
    Algebraically identical to (1-J)/(1-I) (see design_grid() in
    third_party_separation.py) -- expressed here directly in terms of
    (I, A, t) via the ASM inverse J = (I-A)/t + A, exactly matching the reference
    diagnostic scripts' own K_star()."""
    return ((1.0 / t) * (I - A) + (A - b)) / (I - 1.0 + eps)


def synth_pair(J_gt, t, A):
    I = (t * J_gt + (1 - t) * A).clamp(0.0, 1.0)
    return I, K_star(I, A, t)


def load_state_dict_smart(model, weights_path, extra_syspath=None, device="cpu"):
    """Robust checkpoint loader covering both cases seen in the two reference
    diagnostic scripts:
      - LightDehazeNet's trained_LDNet.pth: a plain state_dict, safe to
        load with weights_only=True.
      - AOD-Net's AOD_net_epoch_relu_10.pth (weber0522bb/AODnet-by-pytorch):
        a pickled FULL model object, which needs weights_only=False and
        that repo's model.py importable on sys.path to unpickle at all.
    Tries the safe path first; only falls back to the unsafe path (and
    only inserts extra_syspath) if the safe path actually fails, so
    LightDehazeNet's checkpoint never needs extra_syspath at all.
    """
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        if extra_syspath and extra_syspath not in sys.path:
            sys.path.insert(0, extra_syspath)
        state = torch.load(weights_path, map_location=device, weights_only=False)

    if hasattr(state, "state_dict"):
        state = state.state_dict()
    elif isinstance(state, dict) and "state_dict" in state and not any(
            hasattr(v, "shape") for v in state.values()):
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def check_gradient_formula():
    """Numerically verifies d(g_ASM)/d(theta) = I - 1 via autograd -- same
    check as Section 1 of the reference diagnostic scripts. Needs torch but no
    checkpoint, no image, no GPU."""
    torch.manual_seed(0)
    I_test = torch.rand(1, 3, 8, 8) * 0.9 + 0.05
    K_test = torch.rand(1, 3, 8, 8, requires_grad=True)
    Jhat = K_test * I_test - K_test + 1.0
    dJ_dK = torch.autograd.grad(Jhat, K_test, grad_outputs=torch.ones_like(Jhat))[0]
    max_diff = (dJ_dK - (I_test - 1.0)).abs().max().item()
    print(f"  max|autograd d(Jhat)/dK - (I-1)| = {max_diff:.2e}  (should be ~0)")
    return max_diff


class ConstantOutputControl(nn.Module):
    """The 'dummy model' the advisor's review asks for: a network that
    ignores its input entirely and emits a fixed, learned-once constant
    K value everywhere. Fit it to a real checkpoint's outputs via
    fit_constant_control() below, then compare its error curve on the
    stress-test grid against the real network's error curve -- if they are
    nearly indistinguishable, the real network's apparent "error growth"
    is not evidence of gradient degeneracy (Problem 1), since a network
    that isn't tracking theta* at all would look the same."""
    def __init__(self, k_const=1.0):
        super().__init__()
        self.k_const = nn.Parameter(torch.tensor(float(k_const)), requires_grad=False)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.k_const.expand(B, 3, H, W)


def fit_constant_control(k_targets):
    """k_targets: iterable of scalar K values a real network produced across
    the stress-test grid (e.g. the mean-K per design point). Returns a
    ConstantOutputControl fit by simple averaging -- deliberately the
    crudest possible baseline, so that if even THIS beats or matches the
    real network's error curve, that is a strong (not a marginal) signal."""
    import numpy as np
    k_mean = float(np.mean(list(k_targets)))
    return ConstantOutputControl(k_const=k_mean)


if __name__ == "__main__":
    print("0. Parameter count check")
    m = AODNet()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"   AODNet parameter count: {n_params}")
    expected = 1761
    status = "MATCHES the paper's cited 1,761" if n_params == expected else \
              f"DOES NOT MATCH the paper's cited {expected} -- check architecture"
    print(f"   -> {status}")

    x = torch.rand(2, 3, 64, 64)
    k = get_K(m, x)
    y = m(x)
    assert y.shape == x.shape and k.shape == x.shape
    print(f"   forward shape check OK: input {tuple(x.shape)} -> output {tuple(y.shape)}, K {tuple(k.shape)}")

    print("\n1. Gradient-reachability formula check (autograd vs closed form I-1)")
    check_gradient_formula()

    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
        repo_dir = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"\n2. Loading real weights from {weights_path} ...")
        load_state_dict_smart(m, weights_path, extra_syspath=repo_dir)
        print("   Loaded OK, strict=True (state_dict keys matched exactly).")
    else:
        print("\n(pass a checkpoint path as argv[1], and optionally the cloned")
        print(" weber0522bb/AODnet-by-pytorch directory as argv[2], to also test loading)")
