"""
Geometry helpers shared by DPVO / Metric3D paths.

Kept separate from masked_droid_slam so that importing DPVO code does not pull
in DROID-SLAM (lietorch, droid_backends, etc.).
"""
import cv2
import numpy as np


def est_calib(frame_source):
    """
    Roughly estimate intrinsics from image dimensions.

    Args:
        frame_source: object with ``get_frame(idx, rgb=False)`` returning HxWxC array.
    """
    image = frame_source.get_frame(0, rgb=False)
    h0, w0 = image.shape[:2]
    focal = float(np.max([h0, w0]))
    cx, cy = float(w0) / 2.0, float(h0) / 2.0
    return [focal, focal, cx, cy]


def get_dimention(frame_source):
    """
    (H, W) after DROID/Metric3D-style resize (matches masked_droid_slam.get_dimention).

    Args:
        frame_source: object with ``get_frame(idx, rgb=False)``.
    """
    image = frame_source.get_frame(0, rgb=False)
    h0, w0 = image.shape[:2]
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    image = cv2.resize(image, (w1, h1))
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]
    H, W = image.shape[:2]
    return H, W
