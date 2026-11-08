"""Baselines: pixel-wise MSE and perceptual hashing (pHash). Both return 0-100."""

import cv2
import imagehash
import numpy as np
from PIL import Image

BASE_SIZE = (256, 256)


def mse_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Naive pixel comparison: resize both to 256x256 grayscale, 100 = identical."""
    a = cv2.resize(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), BASE_SIZE).astype(np.float64)
    b = cv2.resize(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), BASE_SIZE).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    # Normalise by a "very different" MSE level; clip to [0, 100].
    return float(np.clip(100.0 * (1.0 - mse / 5000.0), 0.0, 100.0))


def phash_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Perceptual hash similarity: 100 * (1 - hamming / 64)."""
    pa = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    pb = Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    ha, hb = imagehash.phash(pa), imagehash.phash(pb)
    return float(100.0 * (1.0 - (ha - hb) / len(ha.hash.flatten())))
