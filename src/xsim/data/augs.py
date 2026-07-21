"""Sim2real appearance augmentation: essential albumentations for the image student.

One pipeline, photometric-only: the policy's geometry (proprio <-> pixel
alignment, cube position aux labels) must stay valid, so nothing here moves
pixels except tiny cutout occlusions — spatial jitter stays DrQ rand_shift on
the GPU (nets.rand_shift), and viewpoint variation comes from the arena's
camera samplers, not from warping frames.

The picks target the sim->real gaps of the actual rig (Logitech exo cams,
RealSense wrist): white balance / exposure drift, sensor noise, focus and
motion blur, webcam JPEG artifacts, and partial occlusions. Each view of a
sample is augmented independently — the real cameras don't share a sensor or
lighting either.

Applied per sample in DataLoader workers (CPU, uint8 HWC — albumentations'
fast path) via :class:`AugmentedDataset`, so the GPU training step is
untouched. Toggle from the training script (bc.py --augs).
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch

# cv2's internal thread pool deadlocks in forked DataLoader workers; the
# workers are already process-parallel, so per-image threading buys nothing
cv2.setNumThreads(0)


def sim2real_transform(strength: float = 1.0) -> A.Compose:
    """Essential sim2real photometric stack; ``strength`` scales ranges and
    probabilities (1.0 = the tuned defaults, 0.5 = gentle)."""
    s = strength
    return A.Compose(
        [
            # exposure / white balance drift (auto-exposure and auto-WB webcams)
            A.RandomBrightnessContrast(
                brightness_limit=0.25 * s, contrast_limit=0.25 * s, p=0.8 * s
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(8 * s), sat_shift_limit=int(20 * s),
                val_shift_limit=0, p=0.5 * s,
            ),
            A.RandomGamma(gamma_limit=(100 - int(20 * s), 100 + int(20 * s)), p=0.3 * s),
            # sensor noise (low light drives up ISO on both rig cameras)
            A.OneOf(
                [
                    A.ISONoise(color_shift=(0.01, 0.05 * s), intensity=(0.1, 0.5 * s)),
                    A.GaussNoise(std_range=(0.02, 0.1 * s)),
                ],
                p=0.4 * s,
            ),
            # focus misses and 30 Hz motion blur during fast arm segments
            A.OneOf(
                [A.MotionBlur(blur_limit=(3, 5)), A.GaussianBlur(blur_limit=(3, 5))],
                p=0.3 * s,
            ),
            # webcam compression artifacts
            A.ImageCompression(quality_range=(max(1, int(100 - 60 * s)), 95), p=0.3 * s),
            # partial occlusions (cables, the arm itself, clutter)
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15),
                fill=0, p=0.3 * s,
            ),
        ]
    )


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrap a dataset whose item leads with an rgb tensor (V, 3, H, W) uint8;
    augments each view independently, leaves the other keys untouched.
    Holds only the inner dataset and the transform, so it pickles into
    spawned DataLoader workers the same way MemmapDataset does."""

    def __init__(self, ds: torch.utils.data.Dataset, transform: A.Compose):
        self.ds = ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        rgb, *rest = self.ds[i]
        views = rgb.numpy().transpose(0, 2, 3, 1)  # (V, H, W, 3) uint8
        out = np.stack([self.transform(image=v)["image"] for v in views])
        return (torch.from_numpy(out.transpose(0, 3, 1, 2)), *rest)
