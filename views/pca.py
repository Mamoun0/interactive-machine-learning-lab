import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(uploaded_file):

    image = Image.open(uploaded_file)

    image = image.convert("L")

    image = np.array(image)

    return image


# ==========================================================
# ADD GAUSSIAN NOISE
# ==========================================================

def add_noise(
    image,
    noise_level,
):

    noise = np.random.normal(
        0,
        noise_level,
        image.shape,
    )

    noisy = image + noise

    noisy = np.clip(
        noisy,
        0,
        255,
    )

    return noisy


# ==========================================================
# PCA RECONSTRUCTION
# ==========================================================

def reconstruct_image(
    noisy_image,
    n_components,
):

    pca = PCA(
        n_components=n_components,
    )

    transformed = pca.fit_transform(
        noisy_image
    )

    reconstructed = pca.inverse_transform(
        transformed
    )

    reconstructed = np.clip(
        reconstructed,
        0,
        255,
    )

    return reconstructed, pca


# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(
    original,
    reconstructed,
    pca,
):

    mse = mean_squared_error(
        original.flatten(),
        reconstructed.flatten(),
    )

    psnr = 20 * np.log10(
        255 / np.sqrt(mse)
    )

    explained = np.sum(
        pca.explained_variance_ratio_
    )

    compression = (
        original.size /
        (
            pca.n_components_
            * original.shape[1]
        )
    )

    return (
        mse,
        psnr,
        explained,
        compression,
    )


# ==========================================================
# VARIANCE PLOT
# ==========================================================

def variance_plot(pca):

    fig, ax = plt.subplots(
        figsize=(7,4)
    )

    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )

    ax.plot(
        cumulative,
        linewidth=3,
    )

    ax.set_xlabel(
        "Principal Components"
    )

    ax.set_ylabel(
        "Cumulative Explained Variance"
    )

    ax.set_title(
        "Explained Variance"
    )

    ax.grid(True)

    return fig


# ==========================================================
# PAGE
# ==========================================================

def render():

    st.title("🖼️ Principal Component Analysis")

    st.caption(
        "Image Compression & Denoising using PCA"
    )

    st.divider()

    uploaded = st.file_uploader(
        "Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
    )

    if uploaded is None:

        st.info(
            "Upload an image to begin."
        )

        return

    image = load_image(
        uploaded
    )

    st.sidebar.header(
        "⚙️ PCA Parameters"
    )

    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0,
        max_value=60,
        value=20,
    )

    max_components = min(
        image.shape
    )

    components = st.sidebar.slider(
        "Principal Components",
        min_value=5,
        max_value=max_components,
        value=min(
            50,
            max_components,
        ),
    )

    noisy = add_noise(
        image,
        noise_level,
    )

    reconstructed, pca = reconstruct_image(
        noisy,
        components,
    )

    (
        mse,
        psnr,
        explained,
        compression,
    ) = compute_metrics(
        image,
        reconstructed,
        pca,
    )

        # ==========================================================
    # DATASET SUMMARY
    # ==========================================================

    st.subheader("📊 Image Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Width",
        image.shape[1],
    )

    c2.metric(
        "Height",
        image.shape[0],
    )

    c3.metric(
        "Components",
        components,
    )

    c4.metric(
        "Noise",
        noise_level,
    )

    st.divider()

    # ==========================================================
    # IMAGE COMPARISON
    # ==========================================================

    st.subheader("🖼️ Image Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### Original")

        st.image(
            image,
            clamp=True,
            use_container_width=True,
        )

    with col2:

        st.markdown("### Noisy")

        st.image(
            noisy,
            clamp=True,
            use_container_width=True,
        )

    with col3:

        st.markdown("### Reconstructed")

        st.image(
            reconstructed,
            clamp=True,
            use_container_width=True,
        )

    st.divider()

    # ==========================================================
    # PERFORMANCE METRICS
    # ==========================================================

    st.subheader("📈 PCA Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Explained Variance",
        f"{explained*100:.2f}%",
    )

    m2.metric(
        "Compression Ratio",
        f"{compression:.2f}x",
    )

    m3.metric(
        "MSE",
        f"{mse:.2f}",
    )

    m4.metric(
        "PSNR",
        f"{psnr:.2f} dB",
    )

    st.divider()

    # ==========================================================
    # VARIANCE GRAPH
    # ==========================================================

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📊 Cumulative Explained Variance")

        fig = variance_plot(
            pca,
        )

        st.pyplot(fig)

    with right:

        st.subheader("ℹ️ PCA Summary")

        st.info(
            f"""
Principal Components: **{components}**

Noise Level: **{noise_level}**

Explained Variance:

**{explained*100:.2f}%**

Compression Ratio:

**{compression:.2f}x**
"""
        )

    st.divider()

    # ==========================================================
    # PIXEL VALUE DISTRIBUTION
    # ==========================================================

    st.subheader("📉 Pixel Intensity Distribution")

    hist1, hist2 = np.histogram(
        image.flatten(),
        bins=50,
    )

    hist3, _ = np.histogram(
        reconstructed.flatten(),
        bins=50,
    )

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    ax.plot(
        hist1,
        label="Original",
        linewidth=2,
    )

    ax.plot(
        hist3,
        label="Reconstructed",
        linewidth=2,
    )

    ax.set_xlabel("Pixel Bin")

    ax.set_ylabel("Frequency")

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    st.divider()

    # ==========================================================
    # RECONSTRUCTION QUALITY
    # ==========================================================

    difference = np.abs(
        image.astype(float)
        - reconstructed.astype(float)
    )

    st.subheader("🧩 Reconstruction Error")

    st.image(
        difference,
        clamp=True,
        use_container_width=True,
    )

    st.caption(
        "Brighter pixels indicate larger reconstruction errors."
    )

    st.divider()

    # ==========================================================
    # FOOTER
    # ==========================================================

    st.success(
        "✅ PCA successfully compressed and reconstructed the uploaded image."
    )