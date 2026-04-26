import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

print("Project 03 setup completed successfully.")

# --- Task 1: Load and Explore ---

base_dir = os.path.dirname(__file__)

data_path = os.path.join(
    base_dir,
    "..",
    "resources",
    "spambase",
    "spambase.data"
)

# The Spambase data file does not include column headers.
# We load it first with numeric column names, then separate features and target.
df = pd.read_csv(data_path, header=None)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

# The last column is the target label:
# 1 = spam, 0 = non-spam.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nClass distribution:")
print(y.value_counts())

print("\nClass proportions:")
print(y.value_counts(normalize=True))

# The dataset is moderately imbalanced.
# Non-spam emails make up about 60%, while spam emails are about 40%.
# This means accuracy alone may not be a reliable metric.

# --- Baseline Model ---

# Baseline model: predict all emails as non-spam.
# This helps us understand whether future models are actually useful.
y_pred_baseline = np.zeros_like(y)

print("\nBaseline Accuracy on Full Dataset:", accuracy_score(y, y_pred_baseline))

print("\nBaseline Classification Report on Full Dataset:")
print(classification_report(y, y_pred_baseline, zero_division=0))

# The baseline model may look somewhat reasonable by accuracy,
# but it completely fails to detect spam.
# This shows why precision, recall, and F1-score matter.

# --- Task 2: Prepare Your Data ---

# Split dataset into training and testing sets.
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Scale features for models that depend on distances or coefficient magnitudes.
# The scaler is fit only on training data to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA should be applied after scaling because PCA is variance-based.
# If features are not scaled, large-range features can dominate the components.
pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("outputs/pca_cumulative_explained_variance.png")
plt.close()

# Find the first number of components that explains at least 90% of variance.
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("\nNumber of PCA components for 90% variance:", n_components_90)

# Transform scaled train and test data into PCA space.
X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)

# --- Task 3: Classifier Comparison ---

# KNN on unscaled data
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

y_pred_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN (Unscaled) Accuracy:", accuracy_score(y_test, y_pred_knn_unscaled))

print("\nKNN (Unscaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_unscaled))

# KNN on scaled data
# Scaling is important for KNN because it relies on distance calculations.
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN (Scaled) Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))

print("\nKNN (Scaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_scaled))

# KNN on PCA-reduced data
# PCA reduces dimensionality, but it may remove some useful information.
# We compare PCA-reduced data with scaled data to see whether dimensionality reduction helps.
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

y_pred_knn_pca = knn_pca.predict(X_test_pca)

print("\nKNN (PCA) Accuracy:", accuracy_score(y_test, y_pred_knn_pca))

print("\nKNN (PCA) Classification Report:")
print(classification_report(y_test, y_pred_knn_pca))

# Hyperparameter tuning for KNN
# We use scaled data because KNN is distance-based.
k_values = [1, 3, 5, 7, 9, 11, 15]

print("\nK values and CV scores:")

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    print(f"k = {k}, Mean CV Score = {scores.mean():.4f}")

# Scaling had a large impact because KNN relies on distance calculations,
# and features in this dataset have very different ranges.
# Without scaling, large-magnitude features dominate the distance.
# PCA had a smaller effect because the dataset is already fairly informative
# and does not contain a large amount of redundant or noisy features.

# Decision Tree: compare different max_depth values
# Decision Trees do not require scaling because they split data using feature thresholds,
# not distance calculations.

depth_values = [3, 5, 10, None]

print("\nDecision Tree depth comparison:")

for depth in depth_values:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)

    train_accuracy = tree_model.score(X_train, y_train)
    test_accuracy = tree_model.score(X_test, y_test)

    print(
        f"max_depth = {depth}, "
        f"Train Accuracy = {train_accuracy:.4f}, "
        f"Test Accuracy = {test_accuracy:.4f}"
    )

# Final Decision Tree model with chosen depth

best_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
best_tree.fit(X_train, y_train)

y_pred_tree = best_tree.predict(X_test)

print("\nDecision Tree (depth=10) Accuracy:", accuracy_score(y_test, y_pred_tree))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

# As max_depth increases, the model becomes more complex.
# While training accuracy continues to improve, test accuracy peaks at depth=10
# and then decreases slightly for an unlimited depth.
# This indicates overfitting at higher depths.
# Therefore, max_depth=10 provides the best balance between bias and variance.

# Decision Tree performs better than KNN on this dataset.
# It achieves higher accuracy and better recall for spam detection.
# This is likely because the dataset contains informative tabular features,
# and decision trees can effectively capture threshold-based patterns.