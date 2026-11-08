"""End-to-end similarity pipeline.

Preprocessing -> ORB keypoints -> ratio-test matching -> RANSAC alignment
-> SSIM (overlap only) + HSV histogram -> weighted score (0-100).
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .alignment import AlignmentResult, align, warp_to
from .metrics import histogram_correlation, masked_ssim
from .preprocessing import preprocess

# Initial weights from the proposal; tune with evaluation/evaluate.py --tune
W_INLIER, W_SSIM, W_HIST = 0.5, 0.3, 0.2
FALLBACK_CAP = 20.0  # max score when alignment fails


@dataclass
class SimilarityResult:
    score: float
    inlier_ratio: float
    ssim: float
    hist_corr: float
    aligned: bool
    n_good_matches: int
    n_inliers: int
    reason: str
    alignment: Optional[AlignmentResult] = None
    img_a: Optional[np.ndarray] = None
    img_b: Optional[np.ndarray] = None
    warped_b: Optional[np.ndarray] = None
    overlap_mask: Optional[np.ndarray] = None

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "inlier_ratio": round(self.inlier_ratio, 4),
            "ssim": round(self.ssim, 4),
            "hist_corr": round(self.hist_corr, 4),
            "aligned": self.aligned,
            "good_matches": self.n_good_matches,
            "inliers": self.n_inliers,
            "reason": self.reason,
        }


def combine(inlier_ratio: float, ssim: float, hist: float, aligned: bool,
            weights=(W_INLIER, W_SSIM, W_HIST), cap: float = FALLBACK_CAP) -> float:
    wi, ws, wh = weights
    score = 100.0 * (wi * inlier_ratio + ws * ssim + wh * hist)
    if not aligned:
        score = min(score, cap)
    return float(np.clip(score, 0.0, 100.0))


def compare_images(img_a: np.ndarray, img_b: np.ndarray, weights=None) -> SimilarityResult:
    """Compare two BGR images. img_a is treated as the reference/original."""
    weights = weights or (W_INLIER, W_SSIM, W_HIST)
    color_a, gray_a = preprocess(img_a)
    color_b, gray_b = preprocess(img_b)

    al = align(gray_a, gray_b)
    hist = histogram_correlation(color_a, color_b)

    warped = mask = None
    ssim = 0.0
    if al.success:
        warped, mask = warp_to(color_b, al.homography, color_a.shape)
        ssim = masked_ssim(gray_a, cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), mask)

    score = combine(al.inlier_ratio, ssim, hist, al.success, weights)
    return SimilarityResult(
        score=score,
        inlier_ratio=al.inlier_ratio,
        ssim=ssim,
        hist_corr=hist,
        aligned=al.success,
        n_good_matches=len(al.good_matches),
        n_inliers=al.n_inliers,
        reason=al.reason,
        alignment=al,
        img_a=color_a,
        img_b=color_b,
        warped_b=warped,
        overlap_mask=mask,
    )


# ---------- visualisation helpers ----------

def draw_matches(res: SimilarityResult, max_draw: int = 80) -> np.ndarray:
    """Side-by-side image with RANSAC inlier matches (green) drawn. Returns BGR."""
    al = res.alignment
    matches = al.good_matches
    if al.inlier_mask is not None:
        matches = [m for m, keep in zip(matches, al.inlier_mask) if keep]
    matches = sorted(matches, key=lambda m: m.distance)[:max_draw]
    # Matches are B(query) -> A(train); draw with A on the left.
    flipped = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in matches]
    return cv2.drawMatches(res.img_a, al.kp_a, res.img_b, al.kp_b, flipped, None,
                           matchColor=(0, 200, 0), singlePointColor=(0, 0, 255),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def draw_overlay(res: SimilarityResult, alpha: float = 0.5) -> Optional[np.ndarray]:
    """Blend the warped image B over A inside the overlap. Returns BGR or None."""
    if res.warped_b is None:
        return None
    out = res.img_a.copy()
    m = res.overlap_mask
    out[m] = (alpha * res.img_a[m] + (1 - alpha) * res.warped_b[m]).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out
