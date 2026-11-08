"""ORB keypoint detection, ratio-test matching and RANSAC homography alignment."""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

N_FEATURES = 2000
RATIO = 0.75
MIN_GOOD_MATCHES = 10
RANSAC_REPROJ_THRESH = 5.0


@dataclass
class AlignmentResult:
    kp_a: list
    kp_b: list
    good_matches: List[cv2.DMatch] = field(default_factory=list)
    homography: Optional[np.ndarray] = None  # maps image B -> image A
    inlier_mask: Optional[np.ndarray] = None
    inlier_ratio: float = 0.0
    success: bool = False
    reason: str = ""

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum()) if self.inlier_mask is not None else 0


def detect_and_match(gray_a: np.ndarray, gray_b: np.ndarray):
    """Detect ORB keypoints in both images and keep matches passing Lowe's ratio test."""
    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return kp_a or [], kp_b or [], []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_b, des_a, k=2)  # query = B, train = A
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    return kp_a, kp_b, good


def is_valid_homography(H: Optional[np.ndarray]) -> bool:
    """Reject degenerate homographies (flips, extreme scale, strong perspective)."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H[:2, :2])
    if det <= 0 or det < 0.01 or det > 100:
        return False
    if abs(H[2, 0]) > 0.005 or abs(H[2, 1]) > 0.005:
        return False
    return True


def align(gray_a: np.ndarray, gray_b: np.ndarray) -> AlignmentResult:
    kp_a, kp_b, good = detect_and_match(gray_a, gray_b)
    res = AlignmentResult(kp_a=kp_a, kp_b=kp_b, good_matches=good)

    if len(good) < MIN_GOOD_MATCHES:
        res.reason = f"Only {len(good)} good matches (< {MIN_GOOD_MATCHES})"
        return res

    src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)

    if not is_valid_homography(H):
        res.reason = "No valid homography found"
        return res

    mask = mask.ravel().astype(bool)
    if mask.sum() < MIN_GOOD_MATCHES:
        res.reason = f"Only {int(mask.sum())} RANSAC inliers (< {MIN_GOOD_MATCHES})"
        return res

    res.homography = H
    res.inlier_mask = mask
    # Inliers relative to the smaller keypoint set, so a cropped image is not
    # penalised for keypoints that simply are not visible in it.
    denom = max(min(len(kp_a), len(kp_b)), 1)
    res.inlier_ratio = float(min(1.0, mask.sum() / (0.5 * denom)))
    res.success = True
    res.reason = "Aligned"
    return res


def warp_to(img_b: np.ndarray, H: np.ndarray, shape_a) -> tuple:
    """Warp image B onto image A's frame. Returns (warped image, valid-pixel mask)."""
    h, w = shape_a[:2]
    warped = cv2.warpPerspective(img_b, H, (w, h))
    ones = np.full(img_b.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))  # drop interpolated borders
    return warped, valid > 0
