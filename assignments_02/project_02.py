import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The file uses semicolons as separators, not commas.
# pd.read_csv("file.csv") without sep=";" would read the entire row
# as a single column -- every value would be wrong.
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "resources", "student_performance_math.csv")

# =============================================================================
# Task 1: Load and Explore
# =============================================================================

df = pd.read_csv(DATA_PATH, sep=";")

print("=== Task 1: Load and Explore ===")
print(f"Shape: {df.shape}")
print()
print("First 5 rows:")
print(df.head())
print()
print("Data types:")
print(df.dtypes)

# Histogram of G3 with 21 bins (one per possible grade value 0-20).
# We expect to see a cluster of zeros sitting apart from the main distribution --
# those are students who did not take the final exam.
plt.figure()
plt.hist(df["G3"], bins=21, range=(-0.5, 20.5), edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 (Final Grade)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "g3_distribution.png"))
plt.close()
print(f"\nPlot saved: {os.path.join(OUTPUT_DIR, 'g3_distribution.png')}")

# =============================================================================
# Task 2: Preprocess the Data
# =============================================================================

print("\n=== Task 2: Preprocess ===")

# --- Step 1: Filter out G3=0 rows ---
# G3=0 does not mean a student scored zero -- it means they were absent
# from the final exam. Keeping these rows would teach the model a false
# pattern: it would try to predict "exam absence" as if it were a grade,
# mixing two completely different phenomena into one target variable.
print(f"Shape before filtering G3=0: {df.shape}")
df_clean = df[df["G3"] > 0].copy()
print(f"Shape after  filtering G3=0: {df_clean.shape}")
removed     = len(df) - len(df_clean)
removed_pct = removed / len(df) * 100
print(f"Rows removed: {removed} ({removed_pct:.2f}%)")

# --- Step 2: Convert yes/no columns to 1/0 ---
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

# Sanity check: if any unexpected value (e.g. "Yes", "NO") slipped through,
# map() would silently produce NaN instead of raising an error.
nan_counts = df_clean[yes_no_cols].isna().sum()
print(f"\nNaN counts after yes/no encoding:\n{nan_counts}")

# --- Step 3: Convert sex column F/M to 0/1 ---
# F=0, M=1. The dataset is from Portugal 2005; male students show a modest
# math advantage that reflects the social context, not an inherent difference.
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

# --- Step 4: Compare absences correlation before and after filtering ---
# This is a striking result worth examining.
corr_before = df["absences"].corr(df["G3"])
corr_after  = df_clean["absences"].corr(df_clean["G3"])

print(f"\nPearson corr(absences, G3) BEFORE filtering: {corr_before:.4f}")
print(f"Pearson corr(absences, G3) AFTER  filtering: {corr_after:.4f}")
# Before filtering, students with G3=0 had high absences AND G3=0 (by
# definition -- they skipped the exam). This created an artificial strong
# negative correlation between absences and G3. After removing them, the
# true (much weaker) signal remains: absences alone is a poor predictor
# of how well a student who actually showed up will perform.

print("\nAll dtypes after preprocessing:")
print(df_clean.dtypes)

print("\nG3 stats after cleaning:")
print(df_clean["G3"].describe())
