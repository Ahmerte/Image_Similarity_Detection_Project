"""Image similarity detection: ORB + RANSAC alignment, SSIM and colour histograms."""

from .pipeline import compare_images, SimilarityResult
from .baselines import mse_score, phash_score

__all__ = ["compare_images", "SimilarityResult", "mse_score", "phash_score"]
