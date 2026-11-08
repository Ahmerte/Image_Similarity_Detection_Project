"""Image loading and preprocessing."""

import cv2
import numpy as np

MAX_SIDE = 800


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def decode_image(data: bytes) -> np.ndarray:
    """Decode an image from raw bytes (e.g. a Streamlit upload) as BGR."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def resize_max_side(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Resize so the longest side equals max_side, keeping the aspect ratio."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess(img: np.ndarray, max_side: int = MAX_SIDE):
    """Return (resized colour image, grayscale copy)."""
    color = resize_max_side(img, max_side)
    return color, to_gray(color)
