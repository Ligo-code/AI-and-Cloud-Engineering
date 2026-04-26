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

# --- Task 2: Train/Test Split and Baseline ---

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Baseline model: predict all zeros (non-spam)
y_pred_baseline = np.zeros_like(y_test)

# Evaluate baseline
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("\nBaseline Accuracy:", baseline_accuracy)

print("\nBaseline Classification Report:")
print(classification_report(y_test, y_pred_baseline))

# The baseline model achieves around 58% accuracy by predicting all emails as non-spam. 
# However, it completely fails to detect spam, which makes it ineffective. 
# This demonstrates that accuracy alone is not a reliable metric for imbalanced classification problems.

