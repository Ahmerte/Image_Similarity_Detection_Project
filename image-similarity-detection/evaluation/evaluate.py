"""Evaluate MSE, pHash and our feature-based pipeline on pairs.csv.

Usage:
    python -m evaluation.evaluate --pairs data/eval/pairs.csv
    python -m evaluation.evaluate --pairs data/eval/pairs.csv --tune   # grid-search weights

Writes results/scores.csv, results/summary.csv, results/per_transform.csv.
"""

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from similarity import compare_images, mse_score, phash_score
from similarity.pipeline import combine
from similarity.preprocessing import load_image

METHODS = ["mse", "phash", "ours"]


def prf(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def best_threshold(scores, labels):
    """Threshold (0-100) that maximises F1."""
    best = (0.0, 50.0)
    for t in range(1, 100):
        f1 = prf(labels, (np.asarray(scores) >= t).astype(int))[2]
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/eval/pairs.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed threshold for all methods; default picks best F1 per method")
    ap.add_argument("--tune", action="store_true", help="Grid-search pipeline weights")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.pairs) as f:
        pairs = list(csv.DictReader(f))

    records = []
    for i, row in enumerate(pairs, 1):
        a, b = load_image(row["image_a"]), load_image(row["image_b"])
        res = compare_images(a, b)
        records.append({
            "image_a": row["image_a"], "image_b": row["image_b"],
            "transform": row["transform"], "label": int(row["label"]),
            "mse": mse_score(a, b), "phash": phash_score(a, b), "ours": res.score,
            "inlier_ratio": res.inlier_ratio, "ssim": res.ssim,
            "hist_corr": res.hist_corr, "aligned": res.aligned,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(pairs)} pairs")

    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    labels = [r["label"] for r in records]

    if args.tune:
        best = (0, None, None)
        grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
        for wi, ws in itertools.product(grid, grid):
            wh = round(1 - wi - ws, 2)
            if wh < 0:
                continue
            s = [combine(r["inlier_ratio"], r["ssim"], r["hist_corr"], r["aligned"], (wi, ws, wh))
                 for r in records]
            t = best_threshold(s, labels)
            f1 = prf(labels, (np.asarray(s) >= t).astype(int))[2]
            if f1 > best[0]:
                best = (f1, (wi, ws, wh), t)
        print(f"\nBest weights (inlier, ssim, hist) = {best[1]}, threshold = {best[2]}, F1 = {best[0]:.3f}")
        for r in records:
            r["ours"] = combine(r["inlier_ratio"], r["ssim"], r["hist_corr"], r["aligned"], best[1])

    summary, thresholds = [], {}
    for m in METHODS:
        s = [r[m] for r in records]
        t = args.threshold if args.threshold is not None else best_threshold(s, labels)
        thresholds[m] = t
        p, rc, f1 = prf(labels, (np.asarray(s) >= t).astype(int))
        summary.append({"method": m, "threshold": t, "precision": round(p, 3),
                        "recall": round(rc, 3), "f1": round(f1, 3)})

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Per transformation: mean score and detection rate (unrelated -> false positive rate)
    groups = defaultdict(list)
    for r in records:
        groups[r["transform"]].append(r)
    per_t = []
    for tname, rs in sorted(groups.items()):
        row = {"transform": tname, "n": len(rs)}
        for m in METHODS:
            s = np.array([r[m] for r in rs])
            row[f"{m}_mean"] = round(float(s.mean()), 1)
            row[f"{m}_detect"] = round(float((s >= thresholds[m]).mean()), 2)
        per_t.append(row)
    with open(out / "per_transform.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_t[0].keys()))
        w.writeheader()
        w.writerows(per_t)

    print("\n=== Overall ===")
    print(f"{'method':<8}{'thr':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for s in summary:
        print(f"{s['method']:<8}{s['threshold']:>6.0f}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}")
    print("\n=== Per transformation (mean score / detection rate) ===")
    print(f"{'transform':<22}" + "".join(f"{m:>16}" for m in METHODS))
    for row in per_t:
        print(f"{row['transform']:<22}" + "".join(
            f"{row[m + '_mean']:>9.1f} / {row[m + '_detect']:<4.2f}" for m in METHODS))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
