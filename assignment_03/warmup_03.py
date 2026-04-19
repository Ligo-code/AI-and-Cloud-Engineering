import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the Iris dataset once at the top of the file.
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("Iris dataset loaded successfully.")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# --- Preprocessing ---
# Q1: Train/Test Split
# Split the dataset into training and testing sets.
# X = features (input data), y = target labels.
# test_size=0.2 means 20% of the data will be used for testing.
# stratify=y ensures that class distribution is preserved in both train and test sets.
# random_state=42 makes the split reproducible (same result every time you run the code).

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# Print shapes to verify the split
# X_train and X_test contain feature data
# y_train and y_test contain corresponding labels

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2: Feature Scaling

# Create a StandardScaler object.
# StandardScaler transforms each feature so it has mean ~0 and standard deviation ~1.
scaler = StandardScaler()

# Fit the scaler only on the training data.
# This prevents data leakage because the test set must remain unseen during training.
scaler.fit(X_train)
# The scaler is fit only on X_train to avoid leaking information from the test set into the training process.

# Use the fitted scaler to transform both training and test features.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each column in the scaled training data.
# These values should be very close to 0 after standardization.
print("Column means in X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# Why do we fit scaler only on training data? 
# To avoid data leakage. If we fit on the full dataset, the model indirectly learns information from the test set, 
# which leads to overly optimistic performance.

# --- KNN ---
# Q1: Train KNN on unscaled data

# Create a KNN classifier with k=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the unscaled training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("KNN (unscaled data) accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Q2: Train KNN on scaled data

# Create a new KNN model
knn_scaled = KNeighborsClassifier(n_neighbors=5)

# Fit on scaled training data
knn_scaled.fit(X_train_scaled, y_train)

# Predict on scaled test data
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Evaluate accuracy
print("\nKNN (scaled data) accuracy:", accuracy_score(y_test, y_pred_scaled))

# In this case, scaling does not significantly change performance because the Iris dataset
# is already well-behaved and features are on similar scales. After scaling, distances slightly change, 
# which can lead to minor performance drops. 
# However, in general, KNN benefits from scaling because it relies on distance calculations.

# Q3: Cross-validation for KNN (unscaled data)

# Create KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)

# Perform 5-fold cross-validation on training data
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

# Print each fold score
print("\nCross-validation scores:", cv_scores)

# Print mean and standard deviation
print("Mean CV score:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Cross-validation is more reliable than a single train/test split
# because it evaluates the model on multiple subsets of the data,
# reducing the impact of randomness in how the data is split.

# Q4: Hyperparameter tuning for KNN (choose best k)

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("\nK values and corresponding mean CV scores:")

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(knn_model, X_train, y_train, cv=5)
    
    print(f"k = {k}, Mean CV Score = {scores.mean():.4f}")

# We select the k value that gives the highest mean cross-validation score,
# as it is expected to generalize best to unseen data. I had two k values (5 and 7) with the same mean CV score, 
# so I chose k=5 because it is simpler and less likely to overfit compared to k=7 (k=5 achieves the highest mean cross-validation score.
# Although k=7 has the same score, we prefer a smaller k to capture more local patterns in the data.)

# --- Classifier Evaluation ---
# Q1: Confusion Matrix

# Create confusion matrix using predictions from unscaled KNN
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()

# Save the plot
plt.savefig("outputs/knn_confusion_matrix.png")

plt.show()

# The confusion matrix shows how often the model correctly or incorrectly classifies each class.
# In this case, there are likely very few or no misclassifications due to high accuracy.
# The confusion matrix shows perfect classification with no misclassifications.
# All predictions fall on the diagonal, indicating that the model correctly classifies all samples.
# This suggests that the dataset is well-separated and easy for KNN to model.