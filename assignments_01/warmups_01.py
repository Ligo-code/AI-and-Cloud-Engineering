import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# =============================================================================
# --- Pandas ---
# =============================================================================
# Note: all Pandas questions (Q1-Q7) share the same DataFrame `df`

# Pandas Q1
# Create a DataFrame and print basic information about it
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("=== Pandas Q1 ===")
print("First 3 rows:")
print(df.head(3))
print(f"Shape: {df.shape}")
print(f"Num Rows: {len(df)}")
print(f"Num Cols: {len(df.columns)}")
print("Data types:")
print(df.dtypes)

# Pandas Q2
# Filter rows: only students who passed AND have grade > 80
# The & operator combines two boolean conditions element-wise
print("\n=== Pandas Q2 ===")
passed_and_high = df[(df["passed"]) & (df["grade"] > 80)]
print("Students who passed and have grade > 80:")
print(passed_and_high)

# Pandas Q3
# Add a new column by applying a scalar operation to an existing column
# Pandas applies the +5 to every row automatically (vectorized operation)
print("\n=== Pandas Q3 ===")
df["grade_curved"] = df["grade"] + 5
print("DataFrame with curved grades:")
print(df)

# Pandas Q4
# The .str accessor lets us apply string methods to an entire column at once
# Without it, we'd have to loop over every row manually
print("\n=== Pandas Q4 ===")
df["name_upper"] = df["name"].str.upper()
print("Name and name_upper columns:")
print(df[["name", "name_upper"]])

# Pandas Q5
# groupby splits the DataFrame into groups based on a column value,
# then we apply an aggregation function (mean) to each group
print("\n=== Pandas Q5 ===")
city_mean_grade = df.groupby("city")["grade"].mean()
print("Mean grade per city:")
print(city_mean_grade)

# Pandas Q6
# .replace() substitutes one value for another throughout the column
print("\n=== Pandas Q6 ===")
df["city"] = df["city"].replace("Austin", "Houston")
print("Name and city after replacing Austin with Houston:")
print(df[["name", "city"]])

# Pandas Q7
# sort_values with ascending=False gives us descending order
# .head(3) returns just the top 3 rows
print("\n=== Pandas Q7 ===")
top3 = df.sort_values("grade", ascending=False).head(3)
print("Top 3 students by grade (descending):")
print(top3)

# =============================================================================
# --- NumPy ---
# =============================================================================

# NumPy Q1
# np.array() creates an array from a Python list
# shape: dimensions as a tuple, dtype: data type, ndim: number of dimensions
print("\n=== NumPy Q1 ===")
arr1d = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr1d}")
print(f"Shape: {arr1d.shape}")
print(f"Dtype: {arr1d.dtype}")
print(f"Ndim: {arr1d.ndim}")

# NumPy Q2
# A 2D array is like a matrix: shape gives (rows, cols)
# size is the total number of elements: rows * cols
print("\n=== NumPy Q2 ===")
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")
print(f"Size (total elements): {arr.size}")

# NumPy Q3
# Slicing syntax: arr[row_start:row_end, col_start:col_end]
# :2 means "from index 0 up to (not including) index 2"
print("\n=== NumPy Q3 ===")
top_left = arr[:2, :2]
print(f"Top-left 2x2 block:\n{top_left}")

# NumPy Q4
# np.zeros() and np.ones() fill arrays with 0s and 1s respectively
# The argument is a tuple describing the shape: (rows, cols)
print("\n=== NumPy Q4 ===")
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))
print(f"3x4 array of zeros:\n{zeros}")
print(f"2x5 array of ones:\n{ones}")

# NumPy Q5
# np.arange(start, stop, step) — like Python's range() but returns a NumPy array
# stop is exclusive, so arange(0, 50, 5) gives [0, 5, 10, ..., 45]
print("\n=== NumPy Q5 ===")
arange_arr = np.arange(0, 50, 5)
print(f"Array: {arange_arr}")
print(f"Shape: {arange_arr.shape}")
print(f"Mean: {arange_arr.mean()}")
print(f"Sum: {arange_arr.sum()}")
print(f"Std: {arange_arr.std()}")

# NumPy Q6
# np.random.normal(mean, std, size) draws random samples from a normal distribution
# With enough samples the computed mean and std should be close to the target values
print("\n=== NumPy Q6 ===")
np.random.seed(42)
random_arr = np.random.normal(0, 1, 200)
print(f"Mean of 200 random normal values: {random_arr.mean():.4f}  (target: 0)")
print(f"Std  of 200 random normal values: {random_arr.std():.4f}  (target: 1)")


# =============================================================================
# --- Matplotlib ---
# =============================================================================

# Matplotlib Q1
# plt.figure() starts a new empty figure so plots don't overlap each other
# plt.plot() draws a line through the given x, y points
print("\n=== Matplotlib Q1 ===")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure()
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
# plt.bar() takes category labels and their corresponding heights
print("\n=== Matplotlib Q2 ===")
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
# plt.scatter() plots individual points (no connecting line)
# label= assigns a name to each dataset so plt.legend() can display it
print("\n=== Matplotlib Q3 ===")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, label="Data1")
plt.scatter(x2, y2, label="Data2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q4
# plt.subplots(rows, cols) returns a figure and an array of Axes objects
# ax[0] and ax[1] are the individual subplot panels (left and right)
# On subplots we use ax.set_title() instead of plt.title()
# plt.tight_layout() automatically adjusts spacing so subplots don't overlap
print("\n=== Matplotlib Q4 ===")
fig, ax = plt.subplots(1, 2)

ax[0].plot(x, y)
ax[0].set_title("Line Plot")

ax[1].bar(subjects, scores)
ax[1].set_title("Bar Plot")

plt.tight_layout()
plt.show()

