# Credit-Card Anomaly Detection

Can unsupervised models prioritize suspicious future transactions when fraud
labels are unavailable during fitting?

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 anonymized European card transactions, with `Time`, `Amount`, 28 PCA-like features, and a hidden `Class` fraud reference.

## Approach

- Sort transactions by `Time`; fit on the earlier 80% and score the later 20%.
- Fit scaling and every detector on the earlier block only.
- Keep `Class` out of training and threshold selection.
- Rank later transactions and send the top 0.5% (285 transactions) to a fixed human-review queue.
- Reveal `Class` only to evaluate the completed ranking.

Compared models: Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.

## Results

The later block contains 56,962 transactions and 75 fraud cases. Higher is better for every metric below except no model metric is a fraud probability.

| Model | Fraud found in 285 reviews | Precision at 285 | Recall at 285 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-Class SVM | 22 | 7.72% | 29.33% | 0.0473 | 0.9319 |
| Isolation Forest | 14 | 4.91% | 18.67% | 0.0427 | 0.9500 |
| LOF | 0 | 0.00% | 0.00% | 0.0030 | 0.6341 |

**Conclusion:** One-Class SVM found the most fraud within the fixed review capacity. Isolation Forest is a faster, useful baseline. LOF focused on locally unusual but non-fraudulent transactions in this dataset. Since fraud is rare, precision at the review budget and PR-AUC are more informative than ROC-AUC alone.

An anomaly score is a ranking signal, not a fraud probability. These models should prioritize human review, not automatically decline transactions.

## Run

```bash
git clone https://github.com/DanyloKuryliak/credit-card-anomaly-detection.git
cd credit-card-anomaly-detection
uv venv
uv pip install -r requirements.txt
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
uv run jupyter lab
```

Run [`anomaly-detection-credit-card.ipynb`](anomaly-detection-credit-card.ipynb) from top to bottom.

## Limitations

- The data is anonymized, so individual feature meanings cannot be interpreted.
- A chronological split is more realistic than random validation, but it is still one historical period.
- Production deployment needs human-review feedback, drift monitoring, and cost-aware threshold selection.
