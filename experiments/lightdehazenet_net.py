"""
lightdehazenet_net.py
============================
Official LightDehazeNet architecture (Ullah et al., IEEE TIP 2021),
copied attribute-for-attribute from the real, independently-run diagnostic
script (lightdehazenet_diagnostic_tests_v2-2.py), which loads the official
repo's own released weights
(Light-DehazeNet/trained_weights/trained_LDNet.pth).

This is intentionally a STANDALONE class. An earlier, separate
LightDehazeNet class used to live inside network.py as well (an
internal-baseline leftover, unrelated to this official-weights external
validation); it has since been removed from network.py as unused,
unpublished code, leaving this file as the only LightDehazeNet in the
release. The diagnostic script this was copied from already proved this
exact channel configuration (8, 8, 8, 16, 16, 16, 32 -> 3) loads the real
checkpoint successfully with load_state_dict(strict=True).

Parameter count self-check: instantiating this class and summing
parameters gives exactly 30,187 -- matching the paper's corrected figure
(Section 2, "layer-by-layer instantiation of the released architecture
gives 30,187 parameters", replacing the earlier-cited, incorrect 8.2K).
"""
import torch
import torch.nn as nn


class LightDehazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv_layer1 = nn.Conv2d(3, 8, 1, 1, 0, bias=True)
        self.e_conv_layer2 = nn.Conv2d(8, 8, 3, 1, 1, bias=True)
        self.e_conv_layer3 = nn.Conv2d(8, 8, 5, 1, 2, bias=True)
        self.e_conv_layer4 = nn.Conv2d(16, 16, 7, 1, 3, bias=True)
        self.e_conv_layer5 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer6 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer7 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv_layer8 = nn.Conv2d(56, 3, 3, 1, 1, bias=True)

    def forward(self, img):
        c1 = self.relu(self.e_conv_layer1(img))
        c2 = self.relu(self.e_conv_layer2(c1))
        c3 = self.relu(self.e_conv_layer3(c2))
        cc1 = torch.cat((c1, c3), 1)
        c4 = self.relu(self.e_conv_layer4(cc1))
        c5 = self.relu(self.e_conv_layer5(c4))
        c6 = self.relu(self.e_conv_layer6(c5))
        cc2 = torch.cat((c4, c6), 1)
        c7 = self.relu(self.e_conv_layer7(cc2))
        cc3 = torch.cat((c2, c5, c7), 1)
        c8 = self.relu(self.e_conv_layer8(cc3))
        return self.relu((c8 * img) - c8 + 1)


def get_K(model, img):
    """Forward pass, stopping right before g_ASM, returning only
    theta = K(x). Same contract as the reference diagnostic script's get_K()."""
    relu = model.relu
    c1 = relu(model.e_conv_layer1(img))
    c2 = relu(model.e_conv_layer2(c1))
    c3 = relu(model.e_conv_layer3(c2))
    cc1 = torch.cat((c1, c3), 1)
    c4 = relu(model.e_conv_layer4(cc1))
    c5 = relu(model.e_conv_layer5(c4))
    c6 = relu(model.e_conv_layer6(c5))
    cc2 = torch.cat((c4, c6), 1)
    c7 = relu(model.e_conv_layer7(cc2))
    cc3 = torch.cat((c2, c5, c7), 1)
    return relu(model.e_conv_layer8(cc3))


# K_star / synth_pair / load_state_dict_smart / check_gradient_formula are
# identical in form to aod_net.py's (both networks share the same
# g_ASM = K(I-1)+b), so this module simply re-exports them rather than
# duplicating the code. __all__ marks them as intentional re-exports (not
# dead imports) for static-analysis tools.
from aod_net import K_star, synth_pair, load_state_dict_smart, check_gradient_formula  # noqa: E402

__all__ = ["LightDehazeNet", "get_K", "K_star", "synth_pair",
           "load_state_dict_smart", "check_gradient_formula"]


if __name__ == "__main__":
    import sys

    print("0. Parameter count check")
    m = LightDehazeNet()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"   LightDehazeNet parameter count: {n_params}")
    expected = 30187
    status = "MATCHES the paper's corrected 30,187" if n_params == expected else \
              f"DOES NOT MATCH the paper's corrected {expected} -- check architecture"
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
        print(f"\n2. Loading real weights from {weights_path} ...")
        load_state_dict_smart(m, weights_path)
        print("   Loaded OK, strict=True (state_dict keys matched exactly).")
    else:
        print("\n(pass a checkpoint path as argv[1] to also test loading, e.g. "
              "Light-DehazeNet/trained_weights/trained_LDNet.pth)")
