import streamlit as st

from utils.components import (
    hero,
    section,
    metric_row,
    footer,
)


def render():

    hero(
        "🧠 Interactive Machine Learning Laboratory",
        "Learn • Visualize • Compare Classical Machine Learning Algorithms",
    )

    metric_row(
        [
            ("Algorithms", "6"),
            ("Datasets", "2"),
            ("Evaluation Metrics", "5"),
        ]
    )

    st.divider()

    section("📚 About")

    st.write(
        """
This dashboard was developed to demonstrate the implementation,
visualization and evaluation of classical Machine Learning algorithms.

Each algorithm includes:

- Theory
- Dataset exploration
- Interactive visualization
- Training
- Evaluation
- Performance metrics
- Python code explanation
"""
    )

    st.divider()

    section("🚀 Algorithms")

    col1, col2 = st.columns(2)

    with col1:

        st.success("📈 Linear Discriminant Analysis")

        st.success("📊 Quadratic Discriminant Analysis")

        st.success("⚡ Support Vector Machine")

    with col2:

        st.success("🌳 Decision Tree")

        st.success("👥 K-Nearest Neighbors")

        st.success("🧩 Principal Component Analysis")

    footer()