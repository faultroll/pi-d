
# dataset

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import physics as P

# ------------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------------
class DehazeDataset(Dataset):
    def __init__(self, txt_file, a_gt_mode='global'):
        self.lines = open(txt_file).read().strip().split('\n')
        self.ram_cache = {} # In-memory caching
        self.a_gt_mode = a_gt_mode
        
    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        if idx in self.ram_cache:
            return self.ram_cache[idx]
        import os
        hazy_path, gt_path = self.lines[idx].split()
        I_pil = Image.open(hazy_path).convert('RGB')
        J_gt_pil = Image.open(gt_path).convert('RGB')
        I_np = np.array(I_pil).astype(np.float32) / 255.0       # [H,W,3]
        J_gt_np = np.array(J_gt_pil).astype(np.float32) / 255.0 # [H,W,3]
        # Use disk caching to avoid computing CAP (CPU bound) every epoch
        cache_path = hazy_path + ".cap.npz"
        if os.path.exists(cache_path):
            # print(f"Loaded {cache_path}")
            cache_update = False
            data = np.load(cache_path)
            # compatibility
            if 't' in data:
                t_gt_np = data['t']
            else:
                t_gt_np, _, _ = P.run_cap(I_np)
                cache_update = True
            if 'A_global' in data:
                A_global_np = data['A_global']
            else:
                _, A_global_np, _ = P.run_cap(I_np)
                cache_update = True
            if 'A_spatial' in data:
                A_spatial_np = data['A_spatial']
            else:
                A_spatial_np = P.estimate_A_from_pair(I_np, J_gt_np, t_gt_np)
                cache_update = True
            if cache_update:
                try:
                    np.savez(cache_path, t=t_gt_np, A_global=A_global_np, A_spatial=A_spatial_np)
                except Exception:
                    print(f"Update {cache_path} failed!")
                    pass
        else:
            t_gt_np, A_global_np, _ = P.run_cap(I_np)
            A_spatial_np = P.estimate_A_from_pair(I_np, J_gt_np, t_gt_np)
            try:
                np.savez(cache_path, t=t_gt_np, A_global=A_global_np, A_spatial=A_spatial_np)
                print(f"Saved {cache_path}")
            except Exception:
                print(f"Create {cache_path} failed!")
                pass
        if self.a_gt_mode == 'spatial':
            A_gt_np = A_spatial_np
        else:
            A_gt_np = A_global_np
        I    = torch.from_numpy(I_np).permute(2, 0, 1)          # [3,H,W]
        J_gt = torch.from_numpy(J_gt_np).permute(2, 0, 1)       # [3,H,W]
        t_gt = torch.from_numpy(t_gt_np).unsqueeze(0)           # [1,H,W]
        if self.a_gt_mode == 'spatial':
            A_gt = torch.from_numpy(A_gt_np).permute(2, 0, 1)   # [3,H,W]
        else:
            A_gt = torch.from_numpy(A_gt_np)                    # [3]
        # Save to RAM cache for lightning fast epochs 2-N
        self.ram_cache[idx] = (I, t_gt, J_gt, A_gt)
        return self.ram_cache[idx]

    def get_paths(self):
        return [line.split()[0] for line in self.lines]


# train

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import network as N
# import metrics as M
import featuremaps as F

# ------------------------------------------------------------------
#  Reproducibility
# ------------------------------------------------------------------
def set_seed(seed):
    """Fix all random sources so repeated runs are identical."""
    if seed is None:
        print("random seed")
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    print(f"seed {seed} set (with extra strict determinism)")

def worker_init_fn(worker_id):
    import random, numpy as np, torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------------------------------------------------
#  Lightweight PatchGAN Discriminator  (training-only, ~2.7 K params)
# ------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    """
    3-layer PatchGAN discriminator.
    Outputs a spatial map of real/fake scores rather than a single
    scalar - each output cell covers a ~34x34 receptive field of the
    input image, making it sensitive to local texture and detail.
    Used only during training; not exported to ONNX.
    Params: ~2.7 K  (fits the lightweight theme of the paper).
    Architecture follows Isola et al. (pix2pix, CVPR 2017) but
    reduced to 3 layers with smaller channels.
    """
    def __init__(self):
        super().__init__()
        # No BN on first layer - standard PatchGAN practice
        self.net = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  1, kernel_size=4, stride=1, padding=1),
            # no Sigmoid - LSGAN uses raw logits with MSE loss
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)   # standard GAN init
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)   # [B, 1, H', W']
def lsgan_loss_D(D, real, fake):
    """
    LSGAN discriminator loss  (Mao et al., ICCV 2017).
    More stable than vanilla BCE because gradients don't vanish
    when D is confident.
    L_D = 0.5 * E[(D(real) - 1)^2]  +  0.5 * E[(D(fake) - 0)^2]
    """
    real_loss = 0.5 * torch.mean((D(real)         - 1.0) ** 2)
    fake_loss = 0.5 * torch.mean((D(fake.detach()) - 0.0) ** 2)
    return real_loss + fake_loss
def lsgan_loss_G(D, fake):
    """
    LSGAN generator adversarial loss.
    G wants D to score its output as real (~= 1).

    L_G_adv = 0.5 * E[(D(fake) - 1)^2]
    """
    return 0.5 * torch.mean((D(fake) - 1.0) ** 2)

# ------------------------------------------------------------------
#  Weight init for G
# ------------------------------------------------------------------
def init_weights_G(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

# ------------------------------------------------------------------
#  Main training function
# ------------------------------------------------------------------

def train(cfg):
    # Print configuration
    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  Model: capacity={cfg['capacity_mode']}, a_net={cfg['a_net_mode']}, a_gt={cfg['a_gt_mode']}")
    print(f"  Structure: depth_attn={cfg['use_depth_attn']}, channel_attn={cfg['channel_attn_mode']}, "
          f"dilation={cfg['dilation_mode']}, film={cfg['film_mode']}")
    print(f"  Features: refine={cfg['use_refine']}, repconv={cfg['use_repconv']}, "
          f"t_affine={cfg['use_t_affine']}, t_gate={cfg['use_t_gate']}")
    print(f"  Training: gan={cfg['use_gan']}, epochs={cfg['total_epochs']}, "
          f"warmup={cfg['warmup_epochs']}, lr_g={cfg['lr_g']}, lr_d={cfg['lr_d']}")
    print(f"  Paths: train={cfg['train_txt']}, val={cfg['val_txt']}")
    print(f"  Output: model={cfg['best_model_path']}, onnx={cfg['onnx_path']}")
    if cfg['transfer_from']:
        print(f"  Transfer: from={cfg['transfer_from']}")
    print("=" * 60)

    set_seed(cfg['seed'])
    
    # -- CPU Limit ------------------------------------------------
    # Stop PyTorch from waking up all CPU cores for minor data operations
    torch.set_num_threads(2) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # -- data -----------------------------------------------------
    train_dataset = DehazeDataset(cfg['train_txt'], a_gt_mode=cfg['a_gt_mode'])
    if cfg['train_samples'] is not None:
        indices = np.random.choice(len(train_dataset), cfg['train_samples'], replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader  = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, 
                               num_workers=cfg['num_workers'], pin_memory=True, 
                               worker_init_fn=worker_init_fn if cfg['num_workers'] > 0 else None)
    
    val_dataset   = DehazeDataset(cfg['val_txt'], a_gt_mode=cfg['a_gt_mode'])
    if cfg['val_samples'] is not None:
        indices = np.random.choice(len(val_dataset), cfg['val_samples'], replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    val_loader    = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, 
                               num_workers=cfg['num_workers'], pin_memory=True, 
                               worker_init_fn=worker_init_fn if cfg['num_workers'] > 0 else None)
    print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
    print(f"Warm-up: {cfg['warmup_epochs']} epochs (G only)  ->  "
          f"GAN: epochs {cfg['warmup_epochs']+1}-{cfg['total_epochs']}  |  lambda_adv={cfg['lambda_adv']}")

    # -- models ---------------------------------------------------
    G = N.PGDehazeNet(
        capacity_mode=cfg['capacity_mode'],
        use_depth_attn=cfg['use_depth_attn'],
        channel_attn_mode=cfg['channel_attn_mode'],
        dilation_mode=cfg['dilation_mode'],
        film_mode=cfg['film_mode'],
        use_refine=cfg['use_refine'],
        use_repconv=cfg['use_repconv'],
        use_t_affine=cfg['use_t_affine'],
        use_t_gate=cfg['use_t_gate'],
        use_varc=cfg['use_varc'],
        use_a_bound=cfg['use_a_bound'],
        a_net_mode=cfg['a_net_mode']
    ).to(device)
    G.apply(init_weights_G)
    
    if cfg['transfer_from'] is not None:
        import os
        if os.path.exists(cfg['transfer_from']):
            print(f"Loading weights from {cfg['transfer_from']} for transfer learning...")
            baseline_state = torch.load(cfg['transfer_from'], map_location=device)
            v2_state = G.state_dict()
            transferred = 0
            initialized = 0
            for name, param in baseline_state.items():
                if name in v2_state:
                    v2_param = v2_state[name]
                    if 'j_branch' in name and 'weight' in name:
                        if param.shape[0] != v2_param.shape[0]:
                            min_ch = min(param.shape[0], v2_param.shape[0])
                            v2_param[:min_ch] = param[:min_ch]
                            initialized += v2_param.numel() - param.numel()
                            transferred += param.numel()
                        else:
                            v2_param.copy_(param)
                            transferred += param.numel()
                    else:
                        v2_param.copy_(param)
                        transferred += param.numel()
            G.load_state_dict(v2_state)
            print(f"Transfer complete: {transferred} param values transferred, {initialized} randomly initialized.")
        else:
            print(f"Warning: transfer_from path '{cfg['transfer_from']}' not found. Training from scratch.")

    D = PatchDiscriminator().to(device)   # training-only; not exported

    # -- loss (swap to N.UncertaintyWeightedLoss() for learned weights)
    criterion = N.PGLoss().to(device)
	# criterion = N.UncertaintyWeightedLoss().to(device)

    # -- optimisers -----------------------------------------------
    # If using UncertaintyWeightedLoss, add criterion.parameters() to opt_G:
    #   opt_G = optim.Adam(list(G.parameters()) + list(criterion.parameters()), lr=LR_G)
    # optimizer = optim.Adam([
    #     {'params': model.parameters(), 'lr': 1e-2},
    # ])
    opt_G = optim.Adam(G.parameters(), lr=cfg['lr_g'], betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg['lr_d'], betas=(0.5, 0.999))
    #                                                    ^^^
    # beta1=0.5 for D is standard GAN practice (faster momentum decay)

    best_val_loss = float('inf')
    for epoch in range(1, cfg['total_epochs'] + 1):
        import time
        epoch_start_time = time.time()
        G.train()
        D.train()
        use_gan = cfg['use_gan'] and (epoch > cfg['warmup_epochs'])
        # accumulators
        sum_loss_G   = 0.0
        sum_loss_D   = 0.0
        sum_loss_adv = 0.0
        n_batches    = 0
        
        opt_G.zero_grad()
        opt_D.zero_grad()
        scaler = torch.cuda.amp.GradScaler() if cfg['mixed_precision'] else None

        for I, t_gt, J_gt, A_gt in train_loader:
            I    = I.to(device)    # [B,3,H,W]
            t_gt = t_gt.to(device) # [B,1,H,W]
            J_gt = J_gt.to(device) # [B,3,H,W]
            A_gt = A_gt.to(device) # [B,3]
            
            if cfg['mixed_precision']:
                with torch.amp.autocast('cuda'):
                    t_pred, A_pred, J_pred = G(I)
                    preds = {'t': t_pred, 'A': A_pred, 'J': J_pred}
                    gts   = {'t': t_gt,   'A': A_gt,   'J': J_gt}
                    total_loss, losses, lambdas = criterion(I, preds, gts)
                    loss_adv = torch.tensor(0.0, device=device)
                    if use_gan:
                        loss_adv = lsgan_loss_G(D, J_pred)
                        total_loss = total_loss + cfg['lambda_adv'] * loss_adv
                    loss_D = torch.tensor(0.0, device=device)
                    if use_gan:
                        loss_D = lsgan_loss_D(D, J_gt, J_pred)
                        
                scaler.scale(total_loss / cfg['gradient_accumulation']).backward()
                if use_gan:
                    scaler.scale(loss_D / (cfg['gradient_accumulation'] * cfg['d_update_ratio'])).backward()
                    
                if (n_batches + 1) % cfg['gradient_accumulation'] == 0:
                    scaler.step(opt_G)
                    if use_gan and (n_batches // cfg['gradient_accumulation'] + 1) % cfg['d_update_ratio'] == 0:
                        scaler.step(opt_D)
                        opt_D.zero_grad()
                    scaler.update()
                    opt_G.zero_grad()
            else:
                t_pred, A_pred, J_pred = G(I)
                preds = {'t': t_pred, 'A': A_pred, 'J': J_pred}
                gts   = {'t': t_gt,   'A': A_gt,   'J': J_gt}
                total_loss, losses, lambdas = criterion(I, preds, gts)
                loss_adv = torch.tensor(0.0, device=device)
                if use_gan:
                    loss_adv = lsgan_loss_G(D, J_pred)
                    total_loss = total_loss + cfg['lambda_adv'] * loss_adv
                
                loss_D = torch.tensor(0.0, device=device)
                if use_gan:
                    loss_D = lsgan_loss_D(D, J_gt, J_pred)

                (total_loss / cfg['gradient_accumulation']).backward()
                if use_gan:
                    (loss_D / (cfg['gradient_accumulation'] * cfg['d_update_ratio'])).backward()

                if (n_batches + 1) % cfg['gradient_accumulation'] == 0:
                    opt_G.step()
                    if use_gan and (n_batches // cfg['gradient_accumulation'] + 1) % cfg['d_update_ratio'] == 0:
                        opt_D.step()
                        opt_D.zero_grad()
                    opt_G.zero_grad()

            sum_loss_G   += total_loss.item()
            sum_loss_D   += loss_D.item()
            if use_gan:
                sum_loss_adv += loss_adv.item()
            n_batches    += 1

            time.sleep(0.01)

        avg_G   = sum_loss_G   / n_batches
        avg_D   = sum_loss_D   / n_batches
        avg_adv = sum_loss_adv / n_batches
        # -- validation (G only, no D) -----------------------------
        G.eval()
        val_total = 0.0
        val_n     = 0
        with torch.no_grad():
            for I, t_gt, J_gt, A_gt in val_loader:
                I    = I.to(device)
                t_gt = t_gt.to(device)
                J_gt = J_gt.to(device)
                A_gt = A_gt.to(device)
                t_pred, A_pred, J_pred = G(I)
                preds = {'t': t_pred, 'A': A_pred, 'J': J_pred}
                gts   = {'t': t_gt,   'A': A_gt,   'J': J_gt}
                loss, _, _ = criterion(I, preds, gts)
                val_total += loss.item()
                val_n     += 1

                time.sleep(0.01)

        avg_val = val_total / val_n
        # -- save best G (D is not saved - training-only) ---------
        if epoch > cfg['warmup_epochs']:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(G.state_dict(), cfg['best_model_path'])
                print(f"  -> Saved best G  (val loss improved, GAN phase)")
        else:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(G.state_dict(), cfg['best_model_path'])
                print(f"  -> Saved best G  (val loss improved, warmup phase)")
            if epoch == cfg['warmup_epochs']:
                best_val_loss = float('inf')  # Reset for GAN phase
        # -- logging ----------------------------------------------
        epoch_duration = time.time() - epoch_start_time
        gan_tag = f"D={avg_D:.4f}  adv={avg_adv:.4f}" if use_gan else "warm-up (no D)"
        print(f"Epoch {epoch:02d}/{cfg['total_epochs']} [{epoch_duration:.1f}s] | "
              f"G={avg_G:.4f}  {gan_tag} | "
              f"Val={avg_val:.4f}  Best={best_val_loss:.4f}")
    # -- feature maps ---------------------------------------------
    """ import matplotlib.pyplot as plt
    # for name, module in model.named_modules():
    #     print(name, type(module))
    for idx, (I, _, _, _) in enumerate(val_loader):
        feat = F.visualize_feature_maps(G, I.to(device), G.t_branch.viz_target)
        plt.imshow(feat.cpu().numpy(), cmap='jet')
        plt.colorbar()
        plt.savefig(f"./outputs/feature_maps/feature_map_{idx:03d}.png", dpi=300)
        plt.close() """

    # -- ONNX export (G only - D is training-only) ----------------
    # KNOWN ISSUE: this exports G's weights as they are at the end of the
    # LAST epoch, not the best-validation-loss weights saved above to
    # cfg['best_model_path']. The two can differ. Every accuracy number in
    # the paper (PSNR/SSIM/LPIPS, all tables) is computed from the saved
    # .pth (best-val) checkpoints, never from the .onnx file -- the .onnx
    # export here exists only so predict.py can run onnx-tool's MAC/param
    # counter for Table `cost_summary`'s whole-model row, where the exact
    # weight values don't matter, only the graph shape. Do not use the
    # .onnx file to try to reproduce any accuracy number in the paper.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    G.eval()
    input_dict = {
        'I': torch.randn(1, 3, 256, 256).to(device),
    }
    torch.onnx.export(G, input_dict, cfg['onnx_path'],
        input_names=['I'],
        output_names=['t_pred', 'A_pred', 'J_pred'],
        dynamic_axes={
            'I':       {0: 'batch_size', 2: 'height', 3: 'width'},
            't_pred':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'A_pred':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'J_pred':  {0: 'batch_size', 2: 'height', 3: 'width'},
        },
        opset_version=13,
    )
    print(f"ONNX exported -> {cfg['onnx_path']}  (G only, D discarded after training)")


# ------------------------------------------------------------------
#  CONFIG  -  all tuneable knobs in one place
# ------------------------------------------------------------------
import argparse
import os

def parse_args():
    """Parse command line arguments. Environment variables are used as defaults."""
    parser = argparse.ArgumentParser(description='PGDehazeNet Training')
    
    # Data paths
    parser.add_argument('--train-txt', type=str, default=os.getenv('TRAIN_TXT', './SOTS/split_txt/train.txt'),
                        help='Path to training data list')
    parser.add_argument('--val-txt', type=str, default=os.getenv('VAL_TXT', './SOTS/split_txt/val.txt'),
                        help='Path to validation data list')
    parser.add_argument('--best-model-path', type=str, default=os.getenv('BEST_MODEL_PATH', './outputs/best_model.pth'),
                        help='Path to save best model')
    parser.add_argument('--onnx-path', type=str, default=os.getenv('ONNX_PATH', './outputs/joint_net.onnx'),
                        help='Path to export ONNX model')
    
    # Training config
    parser.add_argument('--seed', type=int, default=42, help='Random seed (set to 0 to disable)')
    parser.add_argument('--lr-g', type=float, default=1e-2, help='Generator learning rate')
    parser.add_argument('--lr-d', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--lambda-adv', type=float, default=0.01, help='Adversarial loss weight')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    
    # Training mode
    parser.add_argument('--train-mode', type=str, default='FULL', choices=['DEBUG', 'QUICK', 'MEDIUM', 'FULL'],
                        help='Training mode preset')
    
    # Structure params (ablation study)
    parser.add_argument('--use-depth-attn', action='store_true', default=None,
                        help='Enable depth attention')
    parser.add_argument('--no-depth-attn', action='store_true', help='Disable depth attention')
    parser.add_argument('--channel-attn-mode', type=str, default=None,
                        choices=['None'],
                        help="Channel attention mode. Only 'None' is supported in this "
                             "release -- 'j'/'tj' selected the ChannelAttn module from an "
                             "unpublished architecture-search variant, whose implementation "
                             "was removed (see the warning at the top of network.py).")
    parser.add_argument('--dilation-mode', type=str, default=None,
                        choices=['None', 'Standard', 'Deep'],
                        help="Dilation mode. The paper's V3-t uses 'None'; 'Standard'/'Deep' "
                             "still run (they only vary block count/dilation rate) but were "
                             "never validated in the paper. 'LightASPP' selected a module "
                             "that has been removed from this release and is no longer a "
                             "valid choice.")
    parser.add_argument('--use-gan', action='store_true', default=None, help='Enable GAN training')
    parser.add_argument('--no-gan', action='store_true', help='Disable GAN training')
    parser.add_argument('--film-mode', type=str, default=None,
                        choices=['None', 'PerBlock', 'SFT', 'Both'],
                        help='FiLM mode')
    parser.add_argument('--use-refine', action='store_true', default=None, help='Enable refine module')
    parser.add_argument('--no-refine', action='store_true', help='Disable refine module')
    parser.add_argument('--use-repconv', action='store_true', default=None, help='Enable RepConv')
    parser.add_argument('--no-repconv', action='store_true', help='Disable RepConv')
    parser.add_argument('--use-t-affine', action='store_true', default=None, help='Enable T affine')
    parser.add_argument('--no-t-affine', action='store_true', help='Disable T affine')
    parser.add_argument('--use-t-gate', action='store_true', default=None, help='Enable T gate')
    parser.add_argument('--no-t-gate', action='store_true', help='Disable T gate')
    parser.add_argument('--use-varc', action='store_true', default=None, help='Enable VARC')
    parser.add_argument('--no-varc', action='store_true', help='Disable VARC')
    parser.add_argument('--use-a-bound', action='store_true', default=None, help='Enable A-bound constraint (0.7-1.0)')
    parser.add_argument('--no-a-bound', action='store_true', help='Disable A-bound constraint')
    
    # Network config
    parser.add_argument('--capacity-mode', type=str, default=None,
                        choices=['V1', 'V2', 'V3', 'V4'],
                        help='Network capacity mode')
    parser.add_argument('--a-net-mode', type=str, default=None,
                        choices=['M0', 'M1', 'M2', 'M3', 'M4'],
                        help='A-branch network mode')
    parser.add_argument('--a-gt-mode', type=str, default=None,
                        choices=['global', 'spatial'],
                        help='A ground truth mode')
    parser.add_argument('--transfer-from', type=str, default=None,
                        help='Path to pretrained model for transfer learning')
    
    args = parser.parse_args()
    
    # Apply environment variable defaults for structure params if not set via CLI
    if args.use_depth_attn is None and args.no_depth_attn is False:
        args.use_depth_attn = os.getenv('USE_DEPTH_ATTN', 'False').lower() in ('true', '1', 't')
    else:
        args.use_depth_attn = args.use_depth_attn and not args.no_depth_attn
        
    if args.channel_attn_mode is None:
        args.channel_attn_mode = os.getenv('CHANNEL_ATTN_MODE', 'None')
        
    if args.dilation_mode is None:
        args.dilation_mode = os.getenv('DILATION_MODE', 'None')
        
    if args.use_gan is None and args.no_gan is False:
        args.use_gan = os.getenv('USE_GAN', 'True').lower() in ('true', '1', 't')
    else:
        args.use_gan = args.use_gan and not args.no_gan
        
    if args.film_mode is None:
        args.film_mode = os.getenv('FILM_MODE', 'None')
        
    if args.use_refine is None and args.no_refine is False:
        args.use_refine = os.getenv('USE_REFINE', 'False').lower() in ('true', '1', 't')
    else:
        args.use_refine = args.use_refine and not args.no_refine
        
    if args.use_repconv is None and args.no_repconv is False:
        args.use_repconv = os.getenv('USE_REPCONV', 'False').lower() in ('true', '1', 't')
    else:
        args.use_repconv = args.use_repconv and not args.no_repconv
        
    if args.use_t_affine is None and args.no_t_affine is False:
        args.use_t_affine = os.getenv('USE_T_AFFINE', 'False').lower() in ('true', '1', 't')
    else:
        args.use_t_affine = args.use_t_affine and not args.no_t_affine
        
    if args.use_t_gate is None and args.no_t_gate is False:
        args.use_t_gate = os.getenv('USE_T_GATE', 'False').lower() in ('true', '1', 't')
    else:
        args.use_t_gate = args.use_t_gate and not args.no_t_gate
        
    if args.use_varc is None and args.no_varc is False:
        args.use_varc = os.getenv('USE_VARC', 'False').lower() in ('true', '1', 't')
    else:
        args.use_varc = args.use_varc and not args.no_varc
        
    if args.use_a_bound is None and args.no_a_bound is False:
        args.use_a_bound = os.getenv('USE_A_BOUND', 'False').lower() in ('true', '1', 't')
    else:
        args.use_a_bound = args.use_a_bound and not args.no_a_bound
        
    if args.capacity_mode is None:
        args.capacity_mode = os.getenv('CAPACITY_MODE', 'V1')
        
    if args.a_net_mode is None:
        args.a_net_mode = os.getenv('A_NET_MODE', 'M0')
        
    if args.a_gt_mode is None:
        args.a_gt_mode = os.getenv('A_GT_MODE', 'global')
        
    if args.transfer_from is None:
        args.transfer_from = os.getenv('TRANSFER_FROM', None)
    
    # Disable seed if set to 0
    if args.seed == 0:
        args.seed = None
        
    return args


def get_train_config(args):
    """Get training config based on train mode."""
    configs = {
        'DEBUG': {
            'total_epochs': 3,
            'warmup_epochs': 1,
            'train_samples': 14,
            'val_samples': 3,
            'mixed_precision': False,
            'gradient_accumulation': 1,
            'd_update_ratio': 1,
        },
        'QUICK': {
            'total_epochs': 30,
            'warmup_epochs': 0,
            'train_samples': 98,
            'val_samples': 21,
            'mixed_precision': True,
            'gradient_accumulation': 2,
            'd_update_ratio': 1,
        },
        'MEDIUM': {
            'total_epochs': 30,
            'warmup_epochs': 10,
            'train_samples': None,
            'val_samples': None,
            'mixed_precision': True,
            'gradient_accumulation': 1,
            'd_update_ratio': 2,
        },
        'FULL': {
            'total_epochs': 60,
            'warmup_epochs': 20,
            'train_samples': None,
            'val_samples': None,
            'mixed_precision': False,
            'gradient_accumulation': 1,
            'd_update_ratio': 1,
        },
    }
    return configs[args.train_mode]


if __name__ == '__main__':
    args = parse_args()
    
    # Get train mode config
    mode_cfg = get_train_config(args)
    
    # Build config dict for train function
    cfg = {
        # Paths
        'train_txt': args.train_txt,
        'val_txt': args.val_txt,
        'best_model_path': args.best_model_path,
        'onnx_path': args.onnx_path,
        
        # Training config
        'seed': args.seed,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'lambda_adv': args.lambda_adv,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        
        # Train mode settings
        'total_epochs': mode_cfg['total_epochs'],
        'warmup_epochs': mode_cfg['warmup_epochs'],
        'train_samples': mode_cfg['train_samples'],
        'val_samples': mode_cfg['val_samples'],
        'mixed_precision': mode_cfg['mixed_precision'],
        'gradient_accumulation': mode_cfg['gradient_accumulation'],
        'd_update_ratio': mode_cfg['d_update_ratio'],
        
        # Structure params
        'use_depth_attn': args.use_depth_attn,
        'channel_attn_mode': args.channel_attn_mode,
        'dilation_mode': args.dilation_mode,
        'use_gan': args.use_gan,
        'film_mode': args.film_mode,
        'use_refine': args.use_refine,
        'use_repconv': args.use_repconv,
        'use_t_affine': args.use_t_affine,
        'use_t_gate': args.use_t_gate,
        'use_varc': args.use_varc,
        'use_a_bound': args.use_a_bound,
        
        # Network config
        'capacity_mode': args.capacity_mode,
        'a_net_mode': args.a_net_mode,
        'a_gt_mode': args.a_gt_mode,
        'transfer_from': args.transfer_from,
    }
    
    # Run training
    train(cfg)
