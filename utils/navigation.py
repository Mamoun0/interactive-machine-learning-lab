import streamlit as st


def sidebar():

    st.sidebar.markdown("# 🧠 ML Laboratory")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Choose an Algorithm",
        (
            "🏠 Home",
            "📈 LDA",
            "📊 QDA",
            "⚡ SVM",
            "🌳 Decision Tree",
            "👥 KNN",
            "🧩 PCA",
        ),
    )

    return page.split(" ", 1)[1]