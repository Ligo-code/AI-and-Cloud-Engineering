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