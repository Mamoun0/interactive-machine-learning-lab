# 🧠 Interactive Machine Learning Laboratory

An interactive Streamlit application that demonstrates and visualizes six classical Machine Learning techniques through practical implementations, interactive controls, model evaluation, and data visualization.

The project was developed as a Machine Learning final project with the goal of going beyond simply training models and instead making their behavior visually understandable.

---

**🚀 Live Demo**

🔗 Streamlit App:
(https://interactive-ml-lab.streamlit.app)

---

## 📌 Project Overview

This application provides an interactive environment for exploring and comparing classical Machine Learning algorithms.

Instead of presenting only predictions and accuracy scores, the application allows users to modify model parameters and immediately observe how those changes affect the model's behavior.

The project covers:

- 📈 Linear Discriminant Analysis (LDA)
- 📊 Quadratic Discriminant Analysis (QDA)
- 🎯 Support Vector Machine (SVM)
- 🌳 Decision Tree
- 👥 K-Nearest Neighbors (KNN)
- 🧩 Principal Component Analysis (PCA)

---

## 🤖 Algorithms

### 📈 Linear Discriminant Analysis (LDA)

LDA is used for supervised classification and assumes a common covariance structure between classes.

The application provides:

- Interactive feature selection
- Train/test split
- Decision regions
- Training and testing samples
- Accuracy, Precision, Recall and F1-score
- Confusion matrix
- Classification report

---

### 📊 Quadratic Discriminant Analysis (QDA)

QDA extends the discriminant approach by allowing each class to have its own covariance matrix.

The application includes:

- Interactive feature selection
- Regularization parameter
- Quadratic decision boundaries
- Model evaluation
- Confusion matrix
- Classification report

---

### 🎯 Support Vector Machine (SVM)

SVM searches for a decision boundary that maximizes the margin between classes.

Interactive parameters include:

- Kernel
  - Linear
  - RBF
  - Polynomial
- C parameter
- Gamma
- Polynomial degree
- Train/test split

The visualization also highlights the model's support vectors.

---

### 🌳 Decision Tree

The Decision Tree classifier learns a hierarchy of decision rules from the training data.

The application includes:

- Gini / Entropy criterion
- Maximum tree depth
- Minimum samples required for splitting
- Decision boundary visualization
- Feature importance
- Actual tree structure
- Model evaluation

---

### 👥 K-Nearest Neighbors (KNN)

KNN classifies observations according to their nearest training examples.

Interactive parameters include:

- Number of neighbors (K)
- Weighting strategy
- Distance metric
- Train/test split
- Interactive prediction of a new observation

The application also visualizes the resulting decision regions.

---

### 🧩 Principal Component Analysis (PCA)

PCA is used for dimensionality reduction and image reconstruction.

The application allows users to:

- Upload an image
- Convert it to grayscale
- Add adjustable Gaussian noise
- Select the number of principal components
- Reconstruct the image
- Compare original, noisy and reconstructed images

Additional measurements include:

- Explained variance
- Compression ratio
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Reconstruction error visualization

---

## 📊 Model Evaluation

The classification algorithms are evaluated using several metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Detailed view of correct and incorrect classifications |
| Classification Report | Per-class performance summary |

PCA uses different evaluation concepts appropriate to reconstruction and dimensionality reduction, including explained variance, MSE and PSNR.

---

## 🛠️ Technologies

The project was developed using:

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Plotly
- Matplotlib
- Pillow

---

## 📁 Project Structure

```text
interactive-machine-learning-lab/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   ├── banner.png
│   ├── logo.png
│   └── style.css
│
├── utils/
│   ├── components.py
│   ├── dataset_loader.py
│   ├── helpers.py
│   ├── metrics.py
│   ├── navigation.py
│   ├── plots.py
│   └── theme.py
│
├── views/
│   ├── decision_tree.py
│   ├── home.py
│   ├── knn.py
│   ├── lda.py
│   ├── pca.py
│   ├── qda.py
│   └── svm.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
