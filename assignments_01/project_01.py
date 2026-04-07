import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from prefect import task, flow, get_run_logger

# Absolute path to the raw data files in the course repository.
# os.path.dirname(__file__) gives us the folder this script lives in,
# then we navigate up two levels to find python-200-main.
DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "python-200-main", "assignments", "resources", "happiness_project"
))

# Output directory for plots and merged CSV
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# =============================================================================
# Task 1: Load and Merge Data
# =============================================================================

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()

    # I explicitly filter for CSV files to make the pipeline more robust
    # and avoid unexpected file types (e.g. the Readme.txt in the same folder).
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    logger.info(f"Found {len(files)} CSV files")

    dfs = []
    for file in files:
        path = os.path.join(DATA_DIR, file)

        # These files use semicolon as separator and comma as decimal point.
        # Without sep=";" and decimal="," pandas reads numbers as strings.
        df = pd.read_csv(path, sep=";", decimal=",")

        # 2024 uses "Ladder score" instead of "Happiness score" — normalize it.
        if "Ladder score" in df.columns:
            df = df.rename(columns={"Ladder score": "Happiness score"})

        # Extract year from filename, e.g. "world_happiness_2019.csv" -> 2019
        year = int(file.split("_")[-1].split(".")[0])
        df["year"] = year

        dfs.append(df)
        logger.info(f"Loaded {file}: {len(df)} rows")

    merged_df = pd.concat(dfs, ignore_index=True)

    # Normalize column names to snake_case for consistent access throughout the pipeline
    merged_df.columns = (
        merged_df.columns.str.lower().str.strip().str.replace(" ", "_")
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "merged_happiness.csv")
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset to {output_path} — shape: {merged_df.shape}")

    return merged_df


# =============================================================================
# Task 2: Descriptive Statistics
# =============================================================================

# I analyze central tendency and variability, then segment the data by year
# and region to identify trends and patterns.
@task
def compute_statistics(df):
    logger = get_run_logger()

    mean_score   = df["happiness_score"].mean()
    median_score = df["happiness_score"].median()
    std_score    = df["happiness_score"].std()

    logger.info(f"Overall mean happiness:   {mean_score:.4f}")
    logger.info(f"Overall median happiness: {median_score:.4f}")
    logger.info(f"Overall std happiness:    {std_score:.4f}")

    yearly_mean = df.groupby("year")["happiness_score"].mean()
    logger.info(f"Mean happiness by year:\n{yearly_mean}")

    # "regional_indicator" is the normalized form of "Regional indicator"
    region_mean = df.groupby("regional_indicator")["happiness_score"].mean().sort_values(ascending=False)
    logger.info(f"Mean happiness by region:\n{region_mean}")

    return {
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "yearly": yearly_mean,
        "region": region_mean
    }


# =============================================================================
# Task 3: Visual Exploration
# =============================================================================

# I use a heatmap to visualize correlations between numeric variables.
@task
def create_visualizations(df):
    logger = get_run_logger()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Histogram ---
    plt.figure()
    plt.hist(df["happiness_score"].dropna(), bins=20)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_histogram.png"))
    plt.close()
    logger.info("Saved happiness_histogram.png")

    # --- Boxplot by year ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="year", y="happiness_score", data=df)
    plt.title("Happiness by Year")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_by_year.png"))
    plt.close()
    logger.info("Saved happiness_by_year.png")

    # --- Scatter GDP vs Happiness ---
    plt.figure()
    plt.scatter(df["gdp_per_capita"], df["happiness_score"], alpha=0.3, s=10)
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.title("GDP vs Happiness")
    plt.savefig(os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png"))
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    # --- Correlation heatmap ---
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include="number").drop(columns=["ranking", "year"], errors="ignore").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    logger.info("Saved correlation_heatmap.png")


# =============================================================================
# Task 4: Hypothesis Testing
# =============================================================================

# I tested whether the difference between 2019 and 2020 is statistically significant
# using an independent t-test.
@task
def hypothesis_testing(df):
    logger = get_run_logger()

    # --- Test 1: 2019 vs 2020 (pandemic effect) ---
    data_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    data_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    t_stat, p_val = stats.ttest_ind(data_2019, data_2020)
    mean_2019 = data_2019.mean()
    mean_2020 = data_2020.mean()

    logger.info(f"2019 mean: {mean_2019:.4f}, 2020 mean: {mean_2020:.4f}")
    logger.info(f"t-stat: {t_stat:.4f}, p-value: {p_val:.4f}")

    if p_val < 0.05:
        direction = "decreased" if mean_2020 < mean_2019 else "increased"
        logger.info(
            f"The difference between 2019 and 2020 is statistically significant (p < 0.05). "
            f"Global happiness {direction} during the pandemic year."
        )
    else:
        logger.info(
            "No statistically significant difference between 2019 and 2020 (p >= 0.05)."
        )

    # --- Test 2: Western Europe vs Sub-Saharan Africa ---
    europe = df[df["regional_indicator"] == "Western Europe"]["happiness_score"].dropna()
    africa = df[df["regional_indicator"] == "Sub-Saharan Africa"]["happiness_score"].dropna()

    t_stat2, p_val2 = stats.ttest_ind(europe, africa)
    logger.info(f"Western Europe mean: {europe.mean():.4f}, Sub-Saharan Africa mean: {africa.mean():.4f}")
    logger.info(f"t-stat (region): {t_stat2:.4f}, p-value (region): {p_val2:.4f}")

    if p_val2 < 0.05:
        logger.info("Statistically significant difference between Western Europe and Sub-Saharan Africa (p < 0.05).")
    else:
        logger.info("No statistically significant difference between the two regions (p >= 0.05).")

    return {"mean_2019": mean_2019, "mean_2020": mean_2020, "t_stat": t_stat, "p_val": p_val}


# =============================================================================
# Task 5: Correlation Analysis with Bonferroni Correction
# =============================================================================

@task
def correlation_analysis(df):
    logger = get_run_logger()

    target = "happiness_score"

    # Select numeric predictor columns; drop identifiers and the target itself
    exclude = {target, "ranking", "year"}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]

    results = []
    for col in numeric_cols:
        # dropna() on the pair to avoid NaN errors in pearsonr
        valid = df[[target, col]].dropna()
        r, p = pearsonr(valid[target], valid[col])
        logger.info(f"{col} -> r: {r:.4f}, p-value: {p:.4f}")
        results.append((col, r, p))

    # To control for multiple comparisons, apply a Bonferroni correction
    # to adjust the significance threshold.
    n_tests = len(results)
    adjusted_alpha = 0.05 / n_tests
    logger.info(f"Number of tests: {n_tests}, adjusted alpha (Bonferroni): {adjusted_alpha:.4f}")

    for col, r, p in results:
        if p < 0.05:
            logger.info(f"  {col} is significant at alpha=0.05")
        if p < adjusted_alpha:
            logger.info(f"  {col} remains significant after Bonferroni correction")

    return results


# =============================================================================
# Task 6: Summary Report
# =============================================================================

# I translated statistical results into insights, highlighting trends,
# regional differences, and key drivers of happiness.
@task
def summary_report(df, hypothesis_results, correlation_results):
    logger = get_run_logger()

    n_countries = df["country"].nunique()
    n_years     = df["year"].nunique()
    logger.info(f"Dataset contains {n_countries} unique countries across {n_years} years.")

    region_mean = df.groupby("regional_indicator")["happiness_score"].mean().sort_values(ascending=False)
    top3    = list(region_mean.head(3).index)
    bottom3 = list(region_mean.tail(3).index)
    logger.info(f"Top 3 regions by mean happiness:    {top3}")
    logger.info(f"Bottom 3 regions by mean happiness: {bottom3}")

    m19 = hypothesis_results["mean_2019"]
    m20 = hypothesis_results["mean_2020"]
    p   = hypothesis_results["p_val"]
    if p < 0.05:
        direction = "decreased" if m20 < m19 else "increased"
        logger.info(
            f"Pandemic effect: happiness {direction} from {m19:.3f} (2019) to {m20:.3f} (2020), "
            f"p={p:.4f} — statistically significant."
        )
    else:
        logger.info(
            f"Pandemic effect: no significant change between 2019 ({m19:.3f}) and 2020 ({m20:.3f}), p={p:.4f}."
        )

    n_tests        = len(correlation_results)
    adjusted_alpha = 0.05 / n_tests
    significant    = [(col, r, p) for col, r, p in correlation_results if p < adjusted_alpha]

    if significant:
        strongest = max(significant, key=lambda x: abs(x[1]))
        logger.info(
            f"Strongest predictor after Bonferroni correction: {strongest[0]} (r={strongest[1]:.4f})"
        )
    else:
        logger.info("No correlations remained significant after Bonferroni correction.")


# =============================================================================
# Flow: orchestrate all tasks in order
# =============================================================================

@flow
def happiness_pipeline():
    df           = load_and_merge_data()
    compute_statistics(df)
    create_visualizations(df)
    hypo_results = hypothesis_testing(df)
    corr_results = correlation_analysis(df)
    summary_report(df, hypo_results, corr_results)


if __name__ == "__main__":
    happiness_pipeline()
