import streamlit as st

from utils.navigation import sidebar
from utils.theme import load_css

from views import (
    home,
    lda,
    qda,
    svm,
    decision_tree,
    knn,
    pca,
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Interactive Machine Learning Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# Load CSS
# ----------------------------------------------------

load_css()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

page = sidebar()

# ----------------------------------------------------
# Routing
# ----------------------------------------------------

if page == "Home":
    home.render()

elif page == "LDA":
    lda.render()

elif page == "QDA":
    qda.render()

elif page == "SVM":
    svm.render()

elif page == "Decision Tree":
    decision_tree.render()

elif page == "KNN":
    knn.render()

elif page == "PCA":
    pca.render()