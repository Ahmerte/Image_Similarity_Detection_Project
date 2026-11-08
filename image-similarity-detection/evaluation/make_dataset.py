"""Build the evaluation set: each original gets automatic transformations.

Usage:
    python -m evaluation.make_dataset --originals data/originals --out data/eval
    python -m evaluation.make_dataset --use-samples --out data/eval   # scikit-image sample images

Output: data/eval/<name>/original.jpg + one file per transformation,
and data/eval/pairs.csv listing (image_a, image_b, transform, label).
label = 1 for duplicate (transformed copy), 0 for unrelated pair.
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

SAMPLE_NAMES = ["astronaut", "coffee", "chelsea", "rocket", "camera", "immunohistochemistry",
                "hubble_deep_field", "retina", "colorwheel", "cat", "page", "moon",
                "text", "coins", "brick", "grass", "gravel", "clock"]


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(0, 0, 0))


def scale(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def crop(img, keep=0.7, rng=random):
    h, w = img.shape[:2]
    ch, cw = int(h * keep), int(w * keep)
    y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
    return img[y:y + ch, x:x + cw]


def jpeg(img, quality=20):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def brightness(img, delta=50):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)


def text_overlay(img, text="SAMPLE"):
    out = img.copy()
    h, w = out.shape[:2]
    fs = max(w, h) / 400
    cv2.putText(out, text, (int(w * 0.1), int(h * 0.55)), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (255, 255, 255), max(2, int(fs * 3)), cv2.LINE_AA)
    return out


TRANSFORMS = {
    "rotate_25": lambda im, r: rotate(im, 25),
    "rotate_90": lambda im, r: rotate(im, 90),
    "scale_50": lambda im, r: scale(im, 0.5),
    "scale_150": lambda im, r: scale(im, 1.5),
    "crop_70": lambda im, r: crop(im, 0.7, r),
    "jpeg_q20": lambda im, r: jpeg(im, 20),
    "brightness_p50": lambda im, r: brightness(im, 50),
    "text_overlay": lambda im, r: text_overlay(im),
    "combo_rot_scale_crop": lambda im, r: crop(scale(rotate(im, 15), 0.8), 0.8, r),
}


def load_originals(args):
    if args.use_samples:
        from skimage import data
        for name in SAMPLE_NAMES:
            try:
                img = getattr(data, name)()
            except Exception:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            yield name, img
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in sorted(Path(args.originals).iterdir()):
            if p.suffix.lower() in exts:
                img = cv2.imread(str(p))
                if img is not None:
                    yield p.stem, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", default="data/originals")
    ap.add_argument("--use-samples", action="store_true")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--unrelated-per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, names = [], []

    for name, img in load_originals(args):
        d = out / name
        d.mkdir(exist_ok=True)
        orig_path = d / "original.png"
        cv2.imwrite(str(orig_path), img)
        names.append(name)
        for tname, fn in TRANSFORMS.items():
            p = d / f"{tname}.png"
            cv2.imwrite(str(p), fn(img, rng))
            rows.append((str(orig_path), str(p), tname, 1))

    for name in names:
        others = [n for n in names if n != name]
        for other in rng.sample(others, min(args.unrelated_per_image, len(others))):
            rows.append((str(out / name / "original.png"), str(out / other / "original.png"), "unrelated", 0))

    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "transform", "label"])
        w.writerows(rows)
    print(f"{len(names)} originals, {len(rows)} pairs -> {out / 'pairs.csv'}")


if __name__ == "__main__":
    main()
