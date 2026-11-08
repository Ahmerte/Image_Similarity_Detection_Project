"""Streamlit interface. Run with:  streamlit run app.py"""

import cv2
import streamlit as st

from similarity import compare_images, mse_score, phash_score
from similarity.pipeline import draw_matches, draw_overlay
from similarity.preprocessing import decode_image

st.set_page_config(page_title="Image Similarity Detection", layout="wide")
st.title("Image Similarity Detection")
st.caption("ORB keypoints · RANSAC alignment · SSIM · HSV histogram → similarity score 0–100")


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("Score weights")
    w_in = st.slider("RANSAC inlier ratio", 0.0, 1.0, 0.5, 0.05)
    w_ss = st.slider("SSIM", 0.0, 1.0, 0.3, 0.05)
    w_hi = st.slider("Histogram correlation", 0.0, 1.0, 0.2, 0.05)
    total = w_in + w_ss + w_hi
    if total == 0:
        st.error("At least one weight must be > 0")
        st.stop()
    weights = (w_in / total, w_ss / total, w_hi / total)
    st.caption(f"Normalised: {weights[0]:.2f} / {weights[1]:.2f} / {weights[2]:.2f}")
    threshold = st.slider("Duplicate threshold", 0, 100, 40)

c1, c2 = st.columns(2)
file_a = c1.file_uploader("Original image (A)", type=["jpg", "jpeg", "png", "bmp", "webp"])
file_b = c2.file_uploader("Image to compare (B)", type=["jpg", "jpeg", "png", "bmp", "webp"])

if not (file_a and file_b):
    st.info("Upload two images to compare them.")
    st.stop()

img_a = decode_image(file_a.getvalue())
img_b = decode_image(file_b.getvalue())
c1.image(rgb(img_a), width="stretch")
c2.image(rgb(img_b), width="stretch")

with st.spinner("Comparing..."):
    res = compare_images(img_a, img_b, weights=weights)

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Similarity score", f"{res.score:.1f} / 100")
m2.metric("Inlier ratio", f"{res.inlier_ratio:.2f}", f"{res.n_inliers} / {res.n_good_matches} matches",
          delta_color="off")
m3.metric("SSIM (overlap)", f"{res.ssim:.2f}")
m4.metric("Histogram corr.", f"{res.hist_corr:.2f}")

if res.score >= threshold:
    st.success(f"Likely the SAME image (score ≥ {threshold})")
else:
    st.warning(f"Likely DIFFERENT images (score < {threshold})")

if not res.aligned:
    st.info(f"Alignment skipped: {res.reason}. Score capped so colour alone cannot make images look alike.")

with st.expander("Baselines"):
    b1, b2 = st.columns(2)
    b1.metric("Pixel-wise MSE", f"{mse_score(img_a, img_b):.1f}")
    b2.metric("Perceptual hash (pHash)", f"{phash_score(img_a, img_b):.1f}")

st.subheader("Why do they match?")
t1, t2 = st.tabs(["Matched keypoints", "Aligned overlay"])
with t1:
    if res.n_good_matches:
        st.image(rgb(draw_matches(res)), caption="Green lines: RANSAC inlier matches",
                 width="stretch")
    else:
        st.write("No keypoint matches found.")
with t2:
    overlay = draw_overlay(res)
    if overlay is not None:
        st.image(rgb(overlay), caption="B warped onto A (yellow outline = overlap region)",
                 width="stretch")
    else:
        st.write("No valid alignment, so no overlay.")
