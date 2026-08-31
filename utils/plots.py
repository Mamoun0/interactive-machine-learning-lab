import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix


# ==========================================================
# IRIS SCATTER
# ==========================================================

def iris_scatter(
    df,
    x_feature,
    y_feature,
):

    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="Species",
        template="plotly_dark",
        color_discrete_sequence=[
            "#4F8BF9",
            "#22C55E",
            "#EF4444",
        ],
        height=500,
    )

    fig.update_traces(
        marker=dict(
            size=11,
            line=dict(
                color="white",
                width=1,
            ),
        )
    )

    fig.update_layout(
        title="Iris Dataset",
        legend_title="Species",
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
    )

    return fig


# ==========================================================
# DECISION BOUNDARY
# ==========================================================

def decision_boundary(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    title="Decision Boundary",
    support_vectors=None,
    query_point=None,
):

    x_min = min(
        X_train[:, 0].min(),
        X_test[:, 0].min(),
    ) - 0.5

    x_max = max(
        X_train[:, 0].max(),
        X_test[:, 0].max(),
    ) + 0.5

    y_min = min(
        X_train[:, 1].min(),
        X_test[:, 1].min(),
    ) - 0.5

    y_max = max(
        X_train[:, 1].max(),
        X_test[:, 1].max(),
    ) + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    Z = model.predict(grid)

    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # ------------------------------------------------------
    # Decision Regions
    # ------------------------------------------------------

    fig.add_trace(
        go.Contour(
            x=np.linspace(
                x_min,
                x_max,
                400,
            ),
            y=np.linspace(
                y_min,
                y_max,
                400,
            ),
            z=Z,
            opacity=0.35,
            colorscale="Viridis",
            showscale=False,
            hoverinfo="skip",
            contours=dict(
                coloring="fill",
            ),
        )
    )

    colors = [
        "#4F8BF9",
        "#22C55E",
        "#EF4444",
    ]

    labels = [
        "Setosa",
        "Versicolor",
        "Virginica",
    ]

    # ------------------------------------------------------
    # Training Samples
    # ------------------------------------------------------

    for i in range(3):

        mask = y_train == i

        fig.add_trace(
            go.Scatter(
                x=X_train[mask, 0],
                y=X_train[mask, 1],
                mode="markers",
                name=f"{labels[i]} (Train)",
                marker=dict(
                    color=colors[i],
                    size=10,
                    symbol="circle",
                    line=dict(
                        color="white",
                        width=1,
                    ),
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    + f"{feature_names[0]}: "
                    + "%{x:.2f}<br>"
                    + f"{feature_names[1]}: "
                    + "%{y:.2f}<extra></extra>"
                ),
            )
        )

    # ------------------------------------------------------
    # Testing Samples
    # ------------------------------------------------------

    for i in range(3):

        mask = y_test == i

        fig.add_trace(
            go.Scatter(
                x=X_test[mask, 0],
                y=X_test[mask, 1],
                mode="markers",
                name=f"{labels[i]} (Test)",
                marker=dict(
                    color=colors[i],
                    size=13,
                    symbol="diamond",
                    line=dict(
                        color="white",
                        width=2,
                    ),
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    + f"{feature_names[0]}: "
                    + "%{x:.2f}<br>"
                    + f"{feature_names[1]}: "
                    + "%{y:.2f}<extra></extra>"
                ),
            )
        )

    # ------------------------------------------------------
    # Support Vectors (SVM)
    # ------------------------------------------------------

    if support_vectors is not None:

        fig.add_trace(
            go.Scatter(
                x=support_vectors[:, 0],
                y=support_vectors[:, 1],
                mode="markers",
                name="Support Vectors",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="yellow",
                    line=dict(
                        color="black",
                        width=2,
                    ),
                ),
                hovertemplate="<b>Support Vector</b><extra></extra>",
            )
        )

    # ------------------------------------------------------
    # Query Point (KNN)
    # ------------------------------------------------------

    if query_point is not None:

        fig.add_trace(
            go.Scatter(
                x=[query_point[0]],
                y=[query_point[1]],
                mode="markers",
                name="Prediction Point",
                marker=dict(
                    symbol="star",
                    size=20,
                    color="gold",
                    line=dict(
                        color="white",
                        width=2,
                    ),
                ),
                hovertemplate="<b>Prediction Point</b><extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=650,
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        legend_title="Dataset",
    )

    return fig

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def confusion_matrix_plot(
    y_true,
    y_pred,
    labels,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        aspect="auto",
    )

    fig.update_layout(
        template="plotly_dark",
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=500,
        coloraxis_showscale=False,
    )

    return fig


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance_plot(
    model,
    feature_names,
):

    importance = model.feature_importances_

    fig = px.bar(
        x=feature_names,
        y=importance,
        text=np.round(
            importance,
            3,
        ),
        color=importance,
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        template="plotly_dark",
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance",
        height=500,
        coloraxis_showscale=False,
    )

    fig.update_traces(
        textposition="outside",
    )

    return fig


# ==========================================================
# DECISION TREE VISUALIZATION
# ==========================================================

def tree_plot(
    model,
    feature_names,
    target_names,
):

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )

    plt.tight_layout()

    return fig