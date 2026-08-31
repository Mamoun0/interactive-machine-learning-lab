import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
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
    feature_importance_plot,
    tree_plot,
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

    criterion = st.radio(
        "Criterion",
        ["gini", "entropy"],
        horizontal=True,
    )

    max_depth = st.slider(
        "Max Depth",
        1,
        10,
        3,
    )

    min_samples_split = st.slider(
        "Min Samples Split",
        2,
        10,
        2,
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
        criterion,
        max_depth,
        min_samples_split,
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
    criterion,
    max_depth,
    min_samples_split,
    test_size,
):

    X = df[[feature_x, feature_y]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
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

    st.title("🌳 Decision Tree")

    st.caption(
        "Interactive Decision Tree using the Iris dataset."
    )

    df, X, y, feature_names, target_names = load_data()

    left, right = st.columns([1, 2])

    with left:

        st.subheader("⚙️ Model Controls")

        (
            feature_x,
            feature_y,
            criterion,
            max_depth,
            min_samples_split,
            test_size,
        ) = controls(feature_names)

        st.divider()

        st.subheader("📊 Dataset Summary")

        c1, c2 = st.columns(2)

        c1.metric("Samples", len(df))
        c2.metric("Classes", len(target_names))

        c3, c4 = st.columns(2)

        c3.metric("Features", len(feature_names))
        c4.metric("Test %", f"{int(test_size * 100)}%")

        st.divider()

        st.subheader("🤖 Model")

        st.write(f"**Criterion:** {criterion.title()}")
        st.write(f"**Max Depth:** {max_depth}")
        st.write(f"**Min Samples Split:** {min_samples_split}")

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True,
            height=220,
        )

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
        criterion,
        max_depth,
        min_samples_split,
        test_size,
    )

    st.divider()

    st.subheader("📈 Decision Boundary")

    boundary = decision_boundary(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        [feature_x, feature_y],
        title="Decision Tree Decision Boundary",
    )

    st.plotly_chart(
        boundary,
        use_container_width=True,
    )

    st.divider()

    st.subheader("⭐ Feature Importance")

    importance = feature_importance_plot(
        model,
        [feature_x, feature_y],
    )

    st.plotly_chart(
        importance,
        use_container_width=True,
    )

    st.divider()

    st.subheader("🌳 Decision Tree Structure")

    tree = tree_plot(
        model,
        [feature_x, feature_y],
        target_names,
    )

    st.pyplot(tree)

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

    c1.metric("🎯 Accuracy", f"{accuracy:.3f}")
    c2.metric("📌 Precision", f"{precision:.3f}")
    c3.metric("🔁 Recall", f"{recall:.3f}")
    c4.metric("⭐ F1 Score", f"{f1:.3f}")

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
