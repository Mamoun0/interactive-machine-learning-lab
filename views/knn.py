import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from utils.dataset_loader import load_iris_dataset
from utils.plots import (
    iris_scatter,
    decision_boundary,
    confusion_matrix_plot,
)


# ============================================================
# DATA
# ============================================================

def load_data():

    X, y, feature_names, target_names = load_iris_dataset()

    df = pd.DataFrame(
        X,
        columns=feature_names,
    )

    df["Species"] = [target_names[i] for i in y]

    return df, X, y, feature_names, target_names


# ============================================================
# CONTROLS
# ============================================================

def controls(feature_names):

    feature_x = st.selectbox(
        "Feature X",
        feature_names,
        index=2,
    )

    feature_y = st.selectbox(
        "Feature Y",
        feature_names,
        index=3,
    )

    n_neighbors = st.slider(
        "Neighbors (K)",
        1,
        15,
        5,
    )

    weights = st.radio(
        "Weights",
        ["uniform", "distance"],
        horizontal=True,
    )

    metric = st.selectbox(
        "Distance Metric",
        [
            "euclidean",
            "manhattan",
            "minkowski",
        ],
    )

    test_size = st.slider(
        "Test Size",
        0.20,
        0.50,
        0.30,
        0.05,
    )

    return (
        feature_x,
        feature_y,
        n_neighbors,
        weights,
        metric,
        test_size,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train(
    df,
    y,
    feature_x,
    feature_y,
    n_neighbors,
    weights,
    metric,
    test_size,
):

    X = df[[feature_x, feature_y]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=42,
        test_size=test_size,
    )

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
    )


# ============================================================
# PAGE
# ============================================================

def render():

    st.title("👥 K-Nearest Neighbors")

    st.caption(
        "Interactive implementation of the KNN classifier using the Iris dataset."
    )

    df, X, y, feature_names, target_names = load_data()

    left, right = st.columns([1,2])

    # --------------------------------------------------------
    # LEFT PANEL
    # --------------------------------------------------------

    with left:

        st.subheader("⚙️ Model Controls")

        (
            feature_x,
            feature_y,
            n_neighbors,
            weights,
            metric,
            test_size,
        ) = controls(feature_names)

        st.divider()

        st.subheader("📊 Dataset Summary")

        c1, c2 = st.columns(2)

        c1.metric(
            "Samples",
            len(df),
        )

        c2.metric(
            "Classes",
            len(target_names),
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Features",
            len(feature_names),
        )

        c4.metric(
            "Test %",
            f"{int(test_size*100)}%",
        )

        st.divider()

        st.subheader("🤖 Current Model")

        st.write(f"**Neighbors:** {n_neighbors}")
        st.write(f"**Weights:** {weights.title()}")
        st.write(f"**Metric:** {metric.title()}")

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True,
            height=220,
        )

    # --------------------------------------------------------
    # RIGHT PANEL
    # --------------------------------------------------------

    with right:

        st.subheader("📊 Dataset Visualization")

        scatter = iris_scatter(
            df,
            feature_x,
            feature_y,
        )

        st.plotly_chart(
            scatter,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
    ) = train(
        df,
        y,
        feature_x,
        feature_y,
        n_neighbors,
        weights,
        metric,
        test_size,
    )

    # --------------------------------------------------------
    # INTERACTIVE PREDICTION
    # --------------------------------------------------------

    st.divider()

    st.subheader("🎯 Predict a New Flower")

    col1, col2 = st.columns(2)

    with col1:

        sample_x = st.slider(
            feature_x,
            float(df[feature_x].min()),
            float(df[feature_x].max()),
            float(df[feature_x].mean()),
            0.1,
        )

    with col2:

        sample_y = st.slider(
            feature_y,
            float(df[feature_y].min()),
            float(df[feature_y].max()),
            float(df[feature_y].mean()),
            0.1,
        )

    sample = np.array([[sample_x, sample_y]])

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

        # ============================================================
    # PREDICTION RESULT
    # ============================================================

    st.success(
        f"🌼 Predicted Species: **{target_names[prediction]}**"
    )

    prob_df = pd.DataFrame(
        {
            "Species": target_names,
            "Probability": probabilities,
        }
    )

    st.bar_chart(
        prob_df.set_index("Species")
    )

    # ============================================================
    # DECISION BOUNDARY
    # ============================================================

    st.divider()

    st.subheader("📈 Decision Boundary")

    boundary = decision_boundary(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        [feature_x, feature_y],
        title=f"KNN (K={n_neighbors}) Decision Boundary",
        query_point=[sample_x, sample_y],
    )

    st.plotly_chart(
        boundary,
        use_container_width=True,
    )

    # ============================================================
    # MODEL PERFORMANCE
    # ============================================================

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🎯 Accuracy",
        f"{accuracy:.3f}",
    )

    c2.metric(
        "📌 Precision",
        f"{precision:.3f}",
    )

    c3.metric(
        "🔁 Recall",
        f"{recall:.3f}",
    )

    c4.metric(
        "⭐ F1 Score",
        f"{f1:.3f}",
    )

    # ============================================================
    # CONFUSION MATRIX + REPORT
    # ============================================================

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("🔥 Confusion Matrix")

        heatmap = confusion_matrix_plot(
            y_test,
            predictions,
            target_names,
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True,
        )

    with right:

        st.subheader("📄 Classification Report")

        report = classification_report(
            y_test,
            predictions,
            target_names=target_names,
            output_dict=True,
        )

        report_df = (
            pd.DataFrame(report)
            .transpose()
            .round(3)
        )

        st.dataframe(
            report_df,
            use_container_width=True,
        )

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    st.divider()

    st.subheader("ℹ️ KNN Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Neighbors",
        n_neighbors,
    )

    col2.metric(
        "Weighting",
        weights.title(),
    )

    col3.metric(
        "Distance",
        metric.title(),
    )