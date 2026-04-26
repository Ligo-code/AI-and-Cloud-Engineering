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

df = pd.read_csv(data_path, header=None)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nClass distribution:")
print(y.value_counts())

print("\nClass proportions:")
print(y.value_counts(normalize=True))

# The dataset is moderately imbalanced.
# Non-spam emails make up about 60%, while spam emails are about 40%.
# This means accuracy alone may not be a reliable metric.

# --- Task 2: Prepare Your Data ---

# Split dataset into training and testing sets.
# We use random_state=42 to make results reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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

# The baseline model achieves around 58% accuracy by predicting all emails as non-spam. 
# However, it completely fails to detect spam, which makes it ineffective. 
# This demonstrates that accuracy alone is not a reliable metric for imbalanced classification problems.

# --- Task 3: Classifier Comparison ---

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Using K=5 as a common default choice
knn.fit(X_train, y_train)

# Predict
y_pred_knn = knn.predict(X_test)

# Evaluate
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# KNN significantly improves performance compared to the baseline.
# It is able to detect spam emails with a recall of around 71%.
# However, some spam emails are still missed, indicating room for improvement.

# --- Task 4: KNN with Scaling ---

# Scaling is important for KNN because it relies on distance calculations.
# Features with larger ranges can dominate the distance if not scaled.

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN again
knn_scaled = KNeighborsClassifier(n_neighbors=5)  # Using the same K value for consistency
knn_scaled.fit(X_train_scaled, y_train)

# Predict
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

# Evaluate
print("\nKNN (Scaled) Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))

print("\nKNN (Scaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_scaled))

# Applying feature scaling significantly improved KNN performance.
# The model now detects spam emails more effectively (recall ~84%)
# while maintaining high precision (~90%).
# This confirms that scaling is important for distance-based models like KNN.

# --- Task 5: Hyperparameter tuning for KNN ---

k_values = [1, 3, 5, 7, 9, 11, 15]

print("\nK values and CV scores:")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    
    print(f"k = {k}, Mean CV Score = {scores.mean():.4f}")