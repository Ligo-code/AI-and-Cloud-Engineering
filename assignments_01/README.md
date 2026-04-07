# World Happiness Data Pipeline

This project implements an end-to-end data pipeline using Python and Prefect to analyze the World Happiness dataset across multiple years (2015–2024). The goal of the pipeline is to load, process, analyze, and interpret global happiness data in a structured and reproducible way.

## Architecture

The pipeline is organized as a sequence of tasks orchestrated within a single Prefect flow. Each task represents a specific stage of the data workflow, including data loading, transformation, analysis, visualization, statistical testing, and reporting. This modular structure improves readability, maintainability, and scalability of the pipeline.

## Data Ingestion

The pipeline dynamically loads multiple CSV files from a directory rather than hardcoding file names. Only valid CSV files are processed to ensure robustness.

During loading:
- A year column is extracted from each filename and added to the dataset
- All datasets are merged into a single DataFrame
- The combined dataset is saved for downstream processing

The raw CSV files use semicolon (`;`) as the column separator and comma (`,`) as the decimal point.
Without `sep=";"` and `decimal=","` in `pd.read_csv()`, numeric columns would be read as strings.
The 2024 file also uses `Ladder score` instead of `Happiness score` — this is normalized on load.

## Descriptive Analysis

The pipeline computes key statistical metrics to understand the distribution of happiness scores:
- Mean
- Median
- Standard deviation

It also performs grouped analysis:
- By year (to identify trends over time)
- By region (to compare geographic differences)

## Visualization

Several visualizations are generated and saved:
- Histogram of happiness scores
- Boxplot of happiness by year
- Scatter plot of GDP per capita vs happiness score
- Correlation heatmap of numeric features

These visualizations support exploratory data analysis and help reveal patterns in the data.

## Hypothesis Testing

The pipeline evaluates the impact of the COVID-19 pandemic by comparing happiness scores from 2019 and 2020 using an independent samples t-test.

It logs:
- t-statistic
- p-value
- group means
- plain-language interpretation of results

An additional hypothesis test compares regions to highlight global differences.

## Correlation Analysis

Pearson correlation coefficients are calculated between happiness score and all numeric features.

To address the multiple comparisons problem, the pipeline applies a Bonferroni correction:

adjusted_alpha = 0.05 / number_of_tests

This ensures that only statistically robust relationships are considered significant.

## Summary Report

The final stage produces a human-readable summary of key findings, including:
- Number of countries and years in the dataset
- Top 3 and bottom 3 regions by happiness score
- Interpretation of pandemic impact
- Strongest significant predictor of happiness

## Why Prefect

Prefect is used to orchestrate the workflow and provide:
- Task-level structure
- Logging
- Retry mechanisms

While the pipeline itself is relatively small, this approach reflects real-world data engineering practices where workflows must be reliable and observable.

## How to Run

Run the pipeline using:

```bash
python project_01.py

```
All outputs (datasets and visualizations) will be saved in the outputs/ directory.

## Key Takeaway

The most important part of this project is not just building the pipeline, but translating data into meaningful insights and communicating results clearly.