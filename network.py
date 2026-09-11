
"""network.py -- V3-t architecture: t-branch / A-branch / J-branch, with the
three paper-reported enhancement modules (FiLM = TConditionedResBlock,
T-Affine = TConditionedAffineHead, DAF = DepthAttnFuse). See NAMING.md for
the full paper-symbol <-> class-name map.

This is a trimmed copy of a larger architecture-search codebase. Constructor
flags such as `use_repconv`, `use_t_gate`, `use_varc`, `use_a_bound`,
`dilation_mode` (values other than "None"), `channel_attn_mode` (values
other than "None"), and `a_net_mode` (values other than "M0") select
unpublished variants explored during development; every checkpoint released
alongside the paper uses only the defaults below, and the classes that
implemented those variants (RepConv2d/RepResBlock/TConditionedRepResBlock,
TGateModulation, ChannelAttn, SFTLayer, LightASPP) have been removed from
this release. Leaving all of the above at their defaults reproduces the
paper's V3-t exactly; setting any of them to a non-default value will raise
a NameError, by design, rather than silently doing something unreported.
"""

# model
import torch
import torch.nn as nn

# -------------------------------------------------
#  t-Coupling Methods Additions
# -------------------------------------------------

class TConditionedAffineHead(nn.Module):
    def __init__(self, feat_channels, t_channels=None):
        super().__init__()
        t_channels = t_channels or (feat_channels // 2)
        self.t_encoder = nn.Sequential(
            nn.Conv2d(1, t_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.affine_head = nn.Conv2d(feat_channels + t_channels, 7, kernel_size=1)
        nn.init.normal_(self.affine_head.weight, std=0.01)
        nn.init.zeros_(self.affine_head.bias)
    def forward(self, feat, I, t, A):
        t_feat = self.t_encoder(t)
        combined = torch.cat([feat, t_feat], dim=1)
        affine_params = self.affine_head(combined)
        w1, w2, b = torch.split(affine_params, [3, 3, 1], dim=1)
        b = b.expand(-1, 3, -1, -1)
        return (I - A) * w1 + A * w2 + b

class TConditionedResBlock(nn.Module):
    def __init__(self, channels, dilation=1, t_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.t_scale = nn.Conv2d(t_channels, channels, 1)
        self.t_shift = nn.Conv2d(t_channels, channels, 1)
        nn.init.zeros_(self.t_scale.weight)
        nn.init.zeros_(self.t_scale.bias)
        nn.init.zeros_(self.t_shift.weight)
        nn.init.zeros_(self.t_shift.bias)
    def forward(self, x, t_feat=None):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if t_feat is not None:
            gamma = self.t_scale(t_feat)
            beta = self.t_shift(t_feat)
            out = out * (1 + gamma) + beta
        return x + out

# -------------------------------------------------
#  Building blocks
# -------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class DepthAttnFuse(nn.Module):
    """
    Depth-wise feature fusion with attention.
    Learns a per-sample weighting over L intermediate feature maps.
    All computation happens on GlobalAvgPool outputs [B, L, C],
    so spatial size HxW does NOT appear in MACs - cost is negligible
    regardless of input resolution.
    Added params (channels=4, L=3 layers):
      LayerNorm(4)            :  8  params
      Linear(4->4, no bias)   : 16  params
      Linear(4->1, no bias)   :  4  params
      out_proj Conv(4,4,1)    : 20  params  (16 weight + 4 bias)
    Total                     : ~48 params, ~0 extra MACs
    """
    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.norm  = nn.LayerNorm(channels)
        self.score = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=False),
        )
        # out_proj initialised to identity-like so the module starts
        # as a near-no-op and only diverges as training proceeds.
        self.out_proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.bias)
    def forward(self, feats):
        # feats: list of [B, C, H, W]
        pooled  = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)  # [B, L, C]
        pooled  = self.norm(pooled)
        scores  = self.score(pooled).squeeze(-1)           # [B, L]
        weights = torch.softmax(scores, dim=1)             # [B, L]  sums to 1
        fused   = sum(weights[:, i].view(-1, 1, 1, 1) * f
                      for i, f in enumerate(feats))
        return self.out_proj(fused)

# -------------------------------------------------
#  Branches
# -------------------------------------------------

class MultiScaleBlockGroup(nn.Module):
    """
    Groups a sequence of blocks with exponentially increasing dilation rates
    based on the dilation_mode. Completely decoupled from the specific block implementation.
    """
    def __init__(self, block_class, channels, blocks=2, dilation_mode="None", t_channels=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        if dilation_mode == "Deep":
            actual_blocks = 4
            use_dilation = True
        elif dilation_mode in ("Standard", "LightASPP"):
            actual_blocks = blocks
            use_dilation = True
        else:
            actual_blocks = blocks
            use_dilation = False
        for i in range(actual_blocks):
            d = (2 ** i) if use_dilation else 1
            if t_channels is not None:
                self.blocks.append(block_class(channels, dilation=d, t_channels=t_channels))
            else:
                self.blocks.append(block_class(channels, dilation=d))
    def forward(self, x, return_feats=False, t_feat=None):
        if return_feats:
            feats = [x]
            for blk in self.blocks:
                x = blk(x, t_feat) if t_feat is not None and hasattr(blk, 't_scale') else blk(x)
                feats.append(x)
            return x, feats
        else:
            for blk in self.blocks:
                x = blk(x, t_feat) if t_feat is not None and hasattr(blk, 't_scale') else blk(x)
            return x

class tBranch(nn.Module):
    """
    Transmission-map branch.
    Now enhanced with Dilation and ChannelAttn.
    """
    def __init__(self, channels=4, blocks=2, dilation_mode="None", channel_attn_mode="None", use_repconv=False):
        super().__init__()
        self.dilation_mode = dilation_mode
        self.use_channel_attn = "t" in channel_attn_mode.lower()
        self.use_repconv = use_repconv
        self.init = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        block_cls = RepResBlock if self.use_repconv else ResidualBlock
        self.block_group = MultiScaleBlockGroup(block_cls, channels, blocks, dilation_mode)
        if self.use_channel_attn:
            self.attn = ChannelAttn(channels)
            self.gamma = nn.Parameter(torch.zeros(1))
        self.out = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, I):
        x = self.init(I)
        x = self.block_group(x)
        if self.use_channel_attn:
            x = x + self.gamma * self.attn(x)
        return self.out(x)
    @property
    def viz_target(self):
        """
        Dynamically returns the deepest feature hook point for visualizations.
        """
        last_block = self.block_group.blocks[-1]
        return last_block.repconv2 if self.use_repconv else last_block.conv2

#  A-Branch Configuration Map
a_branch_map = {
    "M0": {'pool': 256,      'encoder': '1x1', 'jloss': 'gt'},
    "M7": {'pool': 256,      'encoder': '3x3', 'jloss': 'gt'},
    "M1": {'pool': 32,       'encoder': '1x1', 'jloss': 'gt'},
    "M2": {'pool': 32,       'encoder': '1x1', 'jloss': 'pred'},
    "M5": {'pool': 32,       'encoder': '3x3', 'jloss': 'gt'},
    "M3": {'pool': 32,       'encoder': '3x3', 'jloss': 'pred'},
    "M8": {'pool': 'hybrid', 'encoder': '1x1', 'jloss': 'gt'},
    "M6": {'pool': 'hybrid', 'encoder': '3x3', 'jloss': 'gt'},
    "M4": {'pool': 'hybrid', 'encoder': '3x3', 'jloss': 'pred'},
}
# Global variable to store current a_net_mode (set by train.py)
current_a_net_mode = "M0"

class ABranch(nn.Module):
    """
    Atmospheric light branch.
    Enhanced with ChannelAttn to improve spectral selection (Rayleigh correction).
    Configuration (pool size, encoder type) is driven by a_net_mode via a_branch_map.
    """
    def __init__(self, channels=4, a_net_mode="M0", channel_attn_mode="None", use_a_bound=False):
        super().__init__()
        self.use_channel_attn = "a" in channel_attn_mode.lower()
        self.a_net_mode = a_net_mode
        self.use_a_bound = use_a_bound
        cfg = a_branch_map.get(a_net_mode, a_branch_map["M0"])
        encoder_mode = cfg['encoder']
        if encoder_mode == "3x3":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, channels, 1),
                nn.ReLU(inplace=True),
            )
        if self.use_channel_attn:
            self.attn = ChannelAttn(channels)
        self.pool_mode = cfg['pool']
        if self.pool_mode == 'hybrid':
            self.pool_g = nn.AdaptiveAvgPool2d(1)
            self.pool_s = nn.AvgPool2d(32, stride=32)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(1, channels // 4), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, channels // 4), 1, 1),
                nn.Sigmoid()
            )
            if self.use_a_bound:
                self.decoder = nn.Conv2d(channels, 3, 1)
            else:
                self.decoder = nn.Sequential(
                    nn.Conv2d(channels, 3, 1),
                    nn.Sigmoid()
                )
        else:
            self.pool_conv = nn.Conv2d(channels, 3, 1)
            self.a_pool = self.pool_mode
    def forward(self, I):
        _, _, H, W = I.shape
        A_feat = self.encoder(I)
        if self.use_channel_attn:
            A_feat = self.attn(A_feat)
        if self.pool_mode == 'hybrid':
            feat_g = self.pool_g(A_feat)
            feat_s = self.pool_s(A_feat)
            w = self.gate(A_feat)
            feat_g_exp = feat_g.expand_as(feat_s)
            feat_fused = w * feat_g_exp + (1.0 - w) * feat_s
            A_small = self.decoder(feat_fused)
            A_pred = nn.functional.interpolate(A_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            A_feat = self.pool_conv(A_feat)
            # fallback for older manual setting: self.a_pool could technically be set
            A_feat = nn.functional.avg_pool2d(A_feat, kernel_size=(self.a_pool, self.a_pool))
            A_pred = nn.functional.interpolate(A_feat, size=(H, W), mode='bilinear', align_corners=False)
        if self.use_a_bound:
            A_pred = 0.7 + 0.3 * torch.sigmoid(A_pred)
        else:
            A_pred = A_pred.clamp(0.0, 1.0)
        return A_pred

class JBranch(nn.Module):
    """
    Clean-image branch.
    The primary bottleneck. Enhanced with Attention and Dilation (RF).
    """
    def __init__(self, channels=4, blocks=2, dilation_mode="None", use_depth_attn=False, channel_attn_mode="None", film_mode="None", use_refine=False, use_repconv=False, use_t_affine=False, use_t_gate=False, use_varc=False):
        super().__init__()
        self.dilation_mode = dilation_mode
        self.use_depth_attn = use_depth_attn
        self.use_channel_attn = "j" in channel_attn_mode.lower()
        self.film_mode = film_mode
        self.use_refine = use_refine
        self.use_repconv = use_repconv
        self.use_t_affine = use_t_affine
        self.use_t_gate = use_t_gate
        self.use_varc = use_varc
        self.init = nn.Sequential(
            nn.Conv2d(3 + 1 + 3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        t_channels = channels // 2 if self.film_mode in ["PerBlock", "Both"] else None
        if self.film_mode in ["PerBlock", "Both"]:
            block_cls = TConditionedRepResBlock if self.use_repconv else TConditionedResBlock
            self.t_encoder = nn.Sequential(
                nn.Conv2d(1, t_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            block_cls = RepResBlock if self.use_repconv else ResidualBlock
        self.block_group = MultiScaleBlockGroup(block_cls, channels, blocks, dilation_mode, t_channels=t_channels)
        if self.use_depth_attn:
            self.depth_fuse = DepthAttnFuse(channels)
        if self.use_channel_attn:
            self.attn = ChannelAttn(channels)
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.dilation_mode == "LightASPP":
            self.lightaspp = LightASPP(channels)
        if self.film_mode in ["SFT", "Both"]:
            self.sft = SFTLayer(cond_channels=1, feat_channels=channels)
        if self.use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 3, 3, padding=1)
            )
            # Zero-init so it starts out as a pure Identity physical mapping
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
        if self.use_t_affine:
            self.affine_head = TConditionedAffineHead(channels, t_channels=channels // 2)
        else:
            self.affine_head = nn.Conv2d(channels, 7, kernel_size=1)
        if self.use_t_gate:
            self.t_gate = TGateModulation()
        
        if self.use_varc:
            self.varc_head = nn.Conv2d(channels, 3, kernel_size=1)
            nn.init.zeros_(self.varc_head.weight)
            nn.init.zeros_(self.varc_head.bias)
            self.varc_alpha = nn.Parameter(torch.tensor(-3.0))
            self.varc_k = nn.Parameter(torch.tensor(5.0))
            self.varc_theta = nn.Parameter(torch.tensor(0.85))
            
        self.act = nn.Sigmoid()
    def forward(self, I, t, A):
        x = torch.cat([I, t, A], dim=1)
        x = self.init(x)
        t_feat = self.t_encoder(t) if self.film_mode in ["PerBlock", "Both"] else None
        if self.use_depth_attn:
            feat, feats = self.block_group(x, return_feats=True, t_feat=t_feat)
            feat = self.depth_fuse(feats)
        else:
            feat = self.block_group(x, t_feat=t_feat)
        if self.dilation_mode == "LightASPP":
            feat = feat + self.lightaspp(feat)
        if self.film_mode in ["SFT", "Both"]:
            feat = self.sft(feat, t)
        if self.use_channel_attn:
            feat = feat + self.gamma * self.attn(feat)
        if self.use_t_affine:
            out = self.affine_head(feat, I, t, A)
        else:
            affine_params = self.affine_head(feat)
            w1, w2, b = torch.split(affine_params, [3, 3, 1], dim=1)
            b = b.expand(-1, 3, -1, -1)
            out = (I - A) * w1 + A * w2 + b
        if self.use_refine:
            out = out + self.refine(out)
        if self.use_t_gate:
            out = self.t_gate(out, I, t)
        if self.use_varc:
            a_mean = A.mean(dim=[2, 3], keepdim=True).mean(dim=1, keepdim=True)
            gate = torch.sigmoid(self.varc_alpha) * torch.sigmoid(
                self.varc_k * (a_mean - self.varc_theta)
            )
            out = out + gate * self.varc_head(feat)
        return self.act(out)

# -------------------------------------------------
#  Full model
# -------------------------------------------------

class PGDehazeNet(nn.Module):
    def __init__(self, capacity_mode="V1", dilation_mode="None", use_depth_attn=False, channel_attn_mode="None", film_mode="None", use_refine=False, use_repconv=False, use_t_affine=False, use_t_gate=False, a_net_mode="M0", use_varc=False, use_a_bound=False):
        super().__init__()
        scaling_map = {
            "V0": {"t": 2, "a": 2,  "j": 2},
            "V1": {"t": 4, "a": 4,  "j": 4},    # V1: 1.2K (original baseline)
            "V2": {"t": 4, "a": 4,  "j": 8},
            "V3": {"t": 4, "a": 4,  "j": 12},   # V3: 5K
            "V4": {"t": 4, "a": 4,  "j": 16},   # V4: 10K
            "V5": {"t": 4, "a": 4,  "j": 24},
            "V3-t": {"t": 6, "a": 4,  "j": 12},
            "V4-t": {"t": 8, "a": 4,  "j": 16},
            "V5-t": {"t": 8, "a": 4,  "j": 24},
            "V3-A": {"t": 4, "a": 6,  "j": 12},
            "V4-A": {"t": 4, "a": 8,  "j": 16},
            "V5-A": {"t": 4, "a": 8,  "j": 24},
        }
        conf = scaling_map.get(capacity_mode, scaling_map["V1"])
        self.t_branch = tBranch(channels=conf["t"], blocks=2, dilation_mode=dilation_mode, channel_attn_mode=channel_attn_mode, use_repconv=use_repconv)
        self.A_branch = ABranch(channels=conf["a"], a_net_mode=a_net_mode, channel_attn_mode=channel_attn_mode, use_a_bound=use_a_bound)
        self.j_branch = JBranch(channels=conf["j"], blocks=2, dilation_mode=dilation_mode, use_depth_attn=use_depth_attn, channel_attn_mode=channel_attn_mode, film_mode=film_mode, use_refine=use_refine, use_repconv=use_repconv, use_t_affine=use_t_affine, use_t_gate=use_t_gate, use_varc=use_varc)
    def forward(self, I):
        t = self.t_branch(I)
        A = self.A_branch(I)
        J = self.j_branch(I, t, A)
        return t, A, J

# -------------------------------------------------
#  Reference models
# -------------------------------------------------

# -------------------------------------------------
#  Loss functions
# -------------------------------------------------

def gradient_x(img):
    # img: (B,1,H,W)
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

# Edge-Aware Smooth
def edge_aware_smooth_loss(t, I, alpha=10.0):
    grad_t_x = gradient_x(t)
    grad_t_y = gradient_y(t)
    grad_I_x = gradient_x(I)
    grad_I_y = gradient_y(I)
    weight_x = torch.exp(-alpha * torch.abs(grad_I_x))
    weight_y = torch.exp(-alpha * torch.abs(grad_I_y))
    return (weight_x * torch.abs(grad_t_x)).mean() + (weight_y * torch.abs(grad_t_y)).mean()

# TV
def tv_loss(S):
    grad_s_x = gradient_x(S)
    grad_s_y = gradient_y(S)
    return (torch.abs(grad_s_x).mean() + torch.abs(grad_s_y).mean())

def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    mu_x = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_y = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = torch.nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu_x2
    sigma_y2 = torch.nn.functional.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu_y2
    sigma_xy = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return 1 - ssim_map.mean()

# VGG
from torchvision import models
import glob, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg = models.vgg16(pretrained=True).features.eval().to(device)
_vgg = None  # global
def get_vgg_features():
    global _vgg
    if _vgg is not None:
        return _vgg
    def load_model(model_name, model_dir):
        model = eval(f'models.{model_name}(pretrained=False)')
        path_format = os.path.join(model_dir, f'{model_name}-[a-z0-9]*.pth')
        model_paths = glob.glob(path_format)
        if not model_paths:
            raise FileNotFoundError(f"No model found matching: {path_format}")
        model_path = model_paths[0]
        print(f"Loading VGG from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    model_dir = './pretrained'
    vgg_full = load_model('vgg16', model_dir)
    _vgg = vgg_full.features.eval().to(device)
    # freeze
    for p in _vgg.parameters():
        p.requires_grad = False
    return _vgg
def perceptual_loss(pred, target, layers=[3,8,15,22]):
    vgg = get_vgg_features()  # loaded once
    loss = 0.0
    x, y = pred, target
    for i, layer in enumerate(vgg):
        x = layer(x); y = layer(y)
        if i in layers:
            # loss += torch.mean((x - y)**2)
            loss += nn.functional.mse_loss(x, y)
    return loss

def tLoss(t_pred, t_gt, I, lambda_edge):
    t_pred_expand = t_pred.expand(-1, 3, -1, -1)
    t_gt_expand = t_gt.expand(-1, 3, -1, -1)
    loss_mse = perceptual_loss(t_pred_expand, t_gt_expand)
    # loss_mse = nn.functional.mse_loss(t_pred, t_gt)
    loss_edge = edge_aware_smooth_loss(t_pred, I)
    loss_t = loss_mse + lambda_edge * loss_edge # 0.01
    return loss_t

def ALoss(A_pred, A_gt, lambda_tv):
    _, _, H, W = A_pred.shape
    if A_gt.dim() == 2:  # old fallback: [B, 3]
        A_pred_mean = A_pred.mean(dim=[2, 3])
        loss_mse = nn.functional.mse_loss(A_pred_mean, A_gt)
    else:  # new: [B, 3, H, W]
        loss_mse = nn.functional.mse_loss(A_pred, A_gt)
    # A_gt_expand = A_gt.unsqueeze(-1).unsqueeze(-1)    # [B, 3, 1, 1]
    # A_gt_expand = A_gt_expand.expand(-1, -1, H, W)    # [B, 3, H, W]
    # loss_mse = nn.functional.mse_loss(A_pred, A_gt_expand) # expand A_gt
    # t_gt_expand = t_gt.expand(-1, 3, -1, -1)
    # S_pred = A_pred
    # S_gt = A_gt_expand * (1.0 - t_gt_expand)
    # loss_mse = nn.functional.mse_loss(S_pred, S_gt) # S instead of A
    # smoothness on A_small (low-res)
    # assume A_small is accessible or recompute
    # here approximate with A_pred
    loss_tv = tv_loss(A_pred)
    loss_A = loss_mse + lambda_tv * loss_tv # 0.001
    return loss_A

def JLoss(J_pred, J_gt, t, A, I, lambda_phy):
    _, _, H, W = J_pred.shape
    loss_mse = perceptual_loss(J_pred, J_gt)
    # loss_mse = nn.functional.mse_loss(J_pred, J_gt)
    if A.dim() == 2:
        A_expand = A.unsqueeze(-1).unsqueeze(-1)    # [B, 3, 1, 1]
        A_expand = A_expand.expand(-1, -1, H, W)    # [B, 3, H, W]
    else:
        A_expand = A
    t_expand = t.expand(-1, 3, -1, -1)
    I_rec = J_pred * t_expand + A_expand * (1.0 - t_expand)
    loss_phy = nn.functional.mse_loss(I_rec, I)
    loss_J = loss_mse + lambda_phy * loss_phy
    return loss_J

# -------------------------------------------------
#  Loss weighting
# -------------------------------------------------

class PGLoss(nn.Module):
    """
    Fixed-weight baseline (1 : 1 : 1).
    Use as the ablation 'no learned weights' setting.
    Uses global current_a_net_mode to determine jloss source.
    """
    def __init__(self):
        super().__init__()
        self.lambdas = {
            't': 1.00,
            'A': 1.00,
            'J': 1.00,
        }
    def forward(self, I, preds, gts):
        t_pred  = preds['t']
        A_pred  = preds['A']
        J_pred  = preds['J']
        t_gt    = gts['t']
        A_gt    = gts['A']
        J_gt    = gts['J']
        # 1. loss_t
        loss_t = tLoss(t_pred, t_gt, I, 0) # 0.01
        # 2. loss_J
        cfg = a_branch_map.get(current_a_net_mode, a_branch_map["M0"])
        jloss_source = cfg['jloss']  # 'gt' or 'pred'
        A_for_jloss = A_pred if jloss_source == 'pred' else A_gt
        loss_J = JLoss(J_pred, J_gt, t_pred, A_for_jloss, I, 1.00)
        # 3. loss_A
        loss_A = ALoss(A_pred, A_gt, 0) # 0.001
        # 4. total_loss
        total_loss = self.lambdas['t'] * loss_t + self.lambdas['A'] * loss_A + self.lambdas['J'] * loss_J
        losses = {
            't': loss_t,
            'A': loss_A,
            'J': loss_J,
        }
        return total_loss, losses, self.lambdas

class UncertaintyWeightedLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting  (Kendall et al., CVPR 2018).
    Each task i gets a learnable log-variance  s_i = log(sigma_i^2).
    The combined loss is:
        L = cumsum_i  [ exp(-s_i) * L_i  +  s_i ]
    * exp(-s_i) automatically down-weights tasks with high uncertainty.
    * The +s_i term prevents the trivial solution s_i -> infinity.
    * All three s_i are initialised to 0  (sigma = 1, i.e. equal weights at start).
    Usage - include loss.log_vars in your optimiser param group:
        # init
        loss_fn = UncertaintyWeightedLoss(lambda_phy=1.0)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4)
        # train
        total_loss, raw_losses, effective_weights = loss_fn(I, preds, gts)
        print(f"Learned weights: t={effective_weights['t']:.3f} A={effective_weights['A']:.3f} J={effective_weights['J']:.3f}")
    After training, the effective weight of task i is  exp(-s_i.item()).
    This is the number to report in the ablation table.
    """
    def __init__(self, lambda_phy=1.0, lambda_edge=0.0, lambda_tv=0.0):
        super().__init__()
        # log(^2) for each task - learnable
        self.log_var = nn.ParameterDict({
            't': nn.Parameter(torch.zeros(1)),
            'A': nn.Parameter(torch.zeros(1)),
            'J': nn.Parameter(torch.zeros(1)),
        })
        self.lambda_phy  = lambda_phy
        self.lambda_edge = lambda_edge
        self.lambda_tv   = lambda_tv
    def forward(self, I, preds, gts):
        raw = {
            't': tLoss(preds['t'], gts['t'], I, self.lambda_edge),
            'A': ALoss(preds['A'], gts['A'], self.lambda_tv),
            'J': JLoss(preds['J'], gts['J'], preds['t'], gts['A'], I, self.lambda_phy),
        }
        total = sum(
            torch.exp(-self.log_var[k]) * raw[k] + self.log_var[k]
            for k in raw
        )
        # effective weights for logging / paper table
        effective = {k: torch.exp(-self.log_var[k]).item() for k in raw}
        return total, raw, effective
