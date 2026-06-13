"""Counterfactual perturbations.

Both are deterministic so results are reproducible:

- Synthetic shadow: a fixed dark trapezoid (multiplicative darkening) crossing
  the mid-distance road surface, simulating a cast shadow.
- Motion blur: directional box-filter convolution along the ego-motion axis.
"""

import cv2
import numpy as np


def apply_shadow(img_rgb, shadow_poly, darkness=0.4):
    """Multiplicatively darken pixels inside a polygon to simulate a shadow."""
    out = img_rgb.copy()
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shadow_poly], 255)
    shadow_region = out[mask == 255].astype(np.float32) * darkness
    out[mask == 255] = shadow_region.clip(0, 255).astype(np.uint8)
    return out


def apply_motion_blur(img_rgb, kernel_size=9, angle_deg=0):
    """Directional motion blur. angle_deg=0 means horizontal (along ego-motion)."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0

    if angle_deg != 0:
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle_deg, 1.0)
        k = cv2.warpAffine(k, M, (kernel_size, kernel_size))

    k /= k.sum()
    return cv2.filter2D(img_rgb, -1, k)
