# Spam Email Classification

ML project to classify emails as spam (1) or non-spam (0) using the UCI Spambase dataset (https://archive.ics.uci.edu/dataset/94/spambase).

## Data
- 4,601 emails, 57 features (word freq, symbols, capital patterns)
- ~60% ham / ~40% spam

## Models
- KNN (unscaled, scaled, PCA)
- Decision Tree (depth tuning)
- Random Forest (100 trees)
- Logistic Regression (scaled, PCA)

## Results
| Model | Test Accuracy |
|---|---|
| Random Forest | 95.5% |
| Decision Tree (depth=10) | 92.5% |
| Logistic Regression | 91.9% |
| KNN (scaled) | 89.4% |
| KNN (unscaled) | 79.0% |

## Key Insights
- Scaling is critical for KNN (+10% accuracy)
- PCA had minimal impact (data already informative)
- Tree-based models perform best on tabular data
- Random Forest = best balance of accuracy and stability

## Error Analysis
Confusion Matrix (Random Forest):
[[522 9]
[ 32 358]]

- False positives (ham → spam): 9  
- False negatives (spam → ham): 32  

Model is conservative: avoids blocking real emails, lets some spam through.

## Conclusion
- Best model: Random Forest  
- Accuracy alone is not enough  
- For spam filters, minimizing false positives is critical  

## Project Structure
assignments_03/
├── project_03.py
├── resources/
│   └── spambase/
│       └── spambase.data
└── outputs/
├── boxplot_word_freq_free.png
├── boxplot_char_freq_excl.png
├── boxplot_capital_run_length_total.png
├── pca_cumulative_explained_variance.png
├── feature_importances.png
└── best_model_confusion_matrix.png

## Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Install with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python project_03.py
```
