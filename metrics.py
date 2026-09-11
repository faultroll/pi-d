
"""metrics.py -- shared PSNR/SSIM utilities used across training,
evaluation, and all experiments/*.py scripts. Whole-model MAC/param
counting for Table `cost_summary` is done separately, in predict.py via
onnx-tool (see the warning at the top of that file about .pth vs .onnx)."""

import numpy as np

def compute_psnr(img1, img2, max_pixel=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_value

def avg_pool2d_np(img, window_size, padding=None):
    # Fastest path: OpenCV (C++-optimized)
    try:
        import cv2
        return cv2.blur(img, (window_size, window_size), borderType=cv2.BORDER_REFLECT)
    except ImportError:
        pass
    # Fast path: SciPy
    try:
        import scipy.ndimage
        return scipy.ndimage.uniform_filter(img, size=(window_size, window_size, 1), mode='reflect')
    except ImportError:
        pass
    # Fallback path: PyTorch (always available in this repo)
    import torch
    import torch.nn.functional as F
    if padding is None:
        padding = window_size // 2
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t_pad = F.pad(t, (padding, padding, padding, padding), mode='reflect')
    out = F.avg_pool2d(t_pad, kernel_size=window_size, stride=1, padding=0)
    return out.squeeze(0).permute(1, 2, 0).numpy()
def compute_ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    mu_x = avg_pool2d_np(pred, window_size)
    mu_y = avg_pool2d_np(target, window_size)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = avg_pool2d_np(pred * pred, window_size) - mu_x2
    sigma_y2 = avg_pool2d_np(target * target, window_size) - mu_y2
    sigma_xy = avg_pool2d_np(pred * target, window_size) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    ssim_value = ssim_map.mean()
    return ssim_value
