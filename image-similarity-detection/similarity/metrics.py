"""SSIM inside the overlapping region and HSV colour-histogram correlation."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def masked_ssim(gray_a: np.ndarray, gray_b_warped: np.ndarray, mask: np.ndarray) -> float:
    """SSIM computed only over pixels where both images are valid.

    Uses the bounding box of the overlap for the SSIM map, then averages the
    map over the mask itself. Result is clipped to [0, 1].
    """
    if mask.sum() < 49:
        return 0.0
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = gray_a[y0:y1, x0:x1]
    b = gray_b_warped[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    if min(a.shape) < 7:
        return 0.0
    a = cv2.GaussianBlur(a, (3, 3), 0)  # reduce sensitivity to JPEG / resampling noise
    b = cv2.GaussianBlur(b, (3, 3), 0)
    _, ssim_map = structural_similarity(a, b, data_range=255, full=True)
    return float(np.clip(ssim_map[m].mean(), 0.0, 1.0))


def hsv_histogram(img_bgr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = None if mask is None else mask.astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], m, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist


def value_histogram(img_bgr: np.ndarray) -> np.ndarray:
    v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    hist = cv2.calcHist([v], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist


def histogram_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """HSV histogram correlation of the two original colour images, mapped to [0, 1].

    Averages the H-S (colour) correlation with the V (brightness) correlation.
    Without the V term, any two grayscale images would score 1.0 because their
    hue/saturation histograms are identical.
    """
    hs = cv2.compareHist(hsv_histogram(img_a), hsv_histogram(img_b), cv2.HISTCMP_CORREL)
    v = cv2.compareHist(value_histogram(img_a), value_histogram(img_b), cv2.HISTCMP_CORREL)
    return float(np.clip(0.5 * hs + 0.5 * v, 0.0, 1.0))
