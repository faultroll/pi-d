
import numpy as np
from PIL import Image, ImageFilter

def pil_to_np(img):
    return np.asarray(img).astype(np.float32) / 255.0

def np_to_pil(arr):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def get_dark_channel(I, window_size=15):
    min_rgb = np.min(I, axis=2)  # [H,W]
    min_rgb_pil = np_to_pil(min_rgb)
    dark_channel_pil = min_rgb_pil.filter(ImageFilter.MinFilter(window_size))
    dark_channel = pil_to_np(dark_channel_pil)
    return dark_channel

def estimate_atmosphere(I, top_percent=0.001):
    dark_channel = get_dark_channel(I)
    H, W = dark_channel.shape
    num_pixels = H * W
    num_brightest = int(num_pixels * top_percent)
    indices = np.argsort(dark_channel.ravel())[-num_brightest:]
    brightest_pixels = I.reshape(-1, 3)[indices]
    A = np.max(brightest_pixels, axis=0)
    return A

def recover_scene(I, t, A, t0=0.1):
    t = np.maximum(t, t0)[..., None]
    J = (I - A) / t + A
    J = np.clip(J, 0, 1)
    return J

def run_cap(I):
    I_pil = np_to_pil(I)
    hsv_pil = I_pil.convert("HSV")
    hsv = pil_to_np(hsv_pil)
    v = hsv[..., 2]
    s = hsv[..., 1]
    theta0, theta1, theta2 = 0.1218, 0.9597, -0.7802
    d = theta0 + theta1 * v + theta2 * s
    beta = 1.0
    t = np.exp(-beta * d)
    t0 = 0.1
    t = np.clip(t, t0, 1.0)
    t_np = t[..., None].astype(np.float32)
    A = estimate_atmosphere(I)
    J = recover_scene(I, t, A)
    return t, A, J

def estimate_A_from_pair(I_np, J_np, t_np, smooth_scale=16, eps=1e-3):
    """
    ASM inversion: A(x,y) = (I - J_gt * t) / (1 - t + eps)
    """
    import cv2
    H, W = I_np.shape[:2]
    t3 = np.repeat(t_np[..., None] if t_np.ndim == 2 else t_np, 3, axis=2)
    A_raw = (I_np - J_np * t3) / (1.0 - t3 + eps)
    A_raw = np.clip(A_raw, 0.0, 1.0)
    small_h = max(1, H // smooth_scale)
    small_w = max(1, W // smooth_scale)
    A_small = cv2.resize(A_raw, (small_w, small_h), interpolation=cv2.INTER_AREA)
    A_smooth = cv2.resize(A_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return A_smooth.astype(np.float32)
