import streamlit as st
import pandas as pd

from sklearn.svm import SVC
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

    kernel = st.radio(
    "Kernel",
    ["linear", "rbf", "poly"],
    horizontal=True,
)

    C = st.slider(
        "C",
        0.1,
        10.0,
        1.0,
        0.1,
    )

    gamma = "scale"
    degree = 3

    if kernel in ["rbf", "poly"]:

            gamma = st.selectbox(
                 "Gamma",
                ["scale", "auto"],
    )

    if kernel == "poly":

                degree = st.slider(
        "Polynomial Degree",
        2,
        6,
        3,
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
        kernel,
        C,
        gamma,
        degree,
        test_size,
    )


# ============================================================
# MODEL
# ============================================================

def train(
    df,
    y,
    feature_x,
    feature_y,
    kernel,
    C,
    gamma,
    degree,
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

    model = SVC(
    kernel=kernel,
    C=C,
    gamma=gamma,
    degree=degree,
    probability=True,
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

    st.title("🎯 Support Vector Machine")

    st.caption(
    "Interactive Support Vector Machine using the Iris dataset."
)

    df, X, y, feature_names, target_names = load_data()

    left, right = st.columns([1, 2])

    with left:
    
            st.subheader("⚙️ Model Controls")
    
            (
                feature_x,
                feature_y,
                kernel,
                C,
                gamma,
                degree,
                test_size,
                ) = controls(feature_names)
    
            st.divider()
    
            st.subheader("📊 Dataset Summary")

            st.divider()

            st.subheader("🤖 Model")

            st.write(f"**Kernel:** {kernel.upper()}")
            st.write(f"**C:** {C}")

    if kernel != "linear":
            st.write(f"**Gamma:** {gamma}")

    if kernel == "poly":
            st.write(f"**Degree:** {degree}")
    
            c1, c2 = st.columns(2)
    
            c1.metric("Samples", len(df))
            c2.metric("Classes", len(target_names))
    
            c3, c4 = st.columns(2)
    
            c3.metric("Features", len(feature_names))
            c4.metric("Test %", f"{int(test_size*100)}%")
    
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
    )= train(
    df,
    y,
    feature_x,
    feature_y,
    kernel,
    C,
    gamma,
    degree,
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
    title=f"SVM ({kernel.upper()}) Decision Boundary",
    support_vectors=model.support_vectors_,
)

    st.plotly_chart(
        boundary,
        use_container_width=True,
    )

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

    st.metric(
    "⭐ Support Vectors",
    model.support_vectors_.shape[0],
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

        report_df = pd.DataFrame(report).transpose().round(3)

        st.dataframe(
            report_df,
            use_container_width=True,
)