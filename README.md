# Machine Learning Assignment 2

## a. Problem statement

E-commerce sites log each visit as a **session**: which pages were opened, how long the visitor stayed, bounce and exit rates, traffic source, and whether the visit happened on a weekend. The business question is binary classification: **will this session end in a purchase?**

This project trains five classical classifiers on the same session table, reports Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC) on a held-out test split, and exposes the results through a Streamlit app where an evaluator can upload that test CSV and switch models.

Positive class = `Revenue = True` (purchase completed).

## b. Dataset description

| Item | Detail |
| --- | --- |
| Name | Online Shoppers Purchasing Intention |
| Source | UCI Machine Learning Repository ([dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)) |
| Instances | 12,330 sessions |
| Input features | 17 (10 numeric page/duration/rate features + 7 categorical / coded attributes) |
| Target | `Revenue` (boolean): purchase vs no purchase |
| Class mix | About 84.5% no purchase / 15.5% purchase (imbalanced) |
| Split | Stratified 80/20, `random_state=42` → 9,864 train / 2,466 test sessions |

Numeric inputs: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`.

Categorical / coded inputs: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

Preprocessing (shared sklearn `Pipeline` for every model): `StandardScaler` on numeric columns and `OneHotEncoder(handle_unknown="ignore")` on categorical columns. Gaussian Naive Bayes receives a dense encoded matrix. Logistic Regression and tree ensembles use `class_weight` so the minority purchase class is not ignored.

Held-out rows used in the app are stored in `test_data.csv`.

## c. Github Repository Link

**Replace this line after you create the remote repo:** `https://github.com/<your-username>/<your-repo>`

## d. Models used

All five models were trained on the same train split and scored on the same 2,466 test sessions.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8406 | 0.8934 | 0.4905 | 0.7435 | 0.5911 | 0.5138 |
| Decision Tree | 0.8362 | 0.9231 | 0.4836 | 0.8482 | 0.6160 | 0.5548 |
| kNN | 0.8731 | 0.8177 | 0.6927 | 0.3246 | 0.4421 | 0.4159 |
| Naive Bayes | 0.2729 | 0.7330 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Random Forest (Ensemble) | 0.8678 | 0.9183 | 0.5522 | 0.7749 | 0.6449 | 0.5787 |

Precision, recall, and F1 are for the purchase class. MCC is used as the primary ranking metric because accuracy is inflated by the majority “no purchase” class.

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear baseline (AUC 0.89). Balanced class weights lift recall (0.74) at the cost of precision (0.49): it flags many non-buying sessions as purchases. Useful when missing a buyer is more expensive than an extra review. |
| Decision Tree | Depth capped at 8 with `min_samples_leaf=20`. Highest AUC (0.923) and high recall (0.85), but precision stays near 0.48, so the tree still over-predicts purchases. MCC (0.55) beats logistic regression because the extra true positives outweigh the extra false positives. |
| kNN | Distance-weighted 11-NN after scaling. Highest accuracy (0.873) and precision (0.69), but recall collapses to 0.32: neighbours are dominated by the majority class, so many actual purchases are missed. Accuracy is misleading here. |
| Naive Bayes | GaussianNB on a scaled, one-hot encoded session vector. Independence is a poor fit for correlated page counts and durations. It almost always predicts purchase (recall 0.97, accuracy 0.27, MCC 0.13). Keep it as a weak baseline, not a production model for this table. |
| Random Forest (Ensemble) | 200 trees, depth 12, balanced subsample weights. Best F1 (0.645) and best MCC (0.579), with AUC close to the tree (0.918). It is the most balanced of the five: it still finds most buyers (recall 0.77) without kNN’s missed purchases or Naive Bayes’ flood of false alarms. |
| Overall Winner for your dataset? | **Random Forest (Ensemble)** — highest MCC and F1 on the held-out sessions. Decision Tree is second (slightly higher AUC, weaker precision/F1). Do not pick kNN from accuracy alone. |

## How to run locally

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

For the BITS Virtual Lab screenshot, open `notebooks/train_and_evaluate.ipynb` and Run All.

## Streamlit app

The app (`app.py`) lets you:

1. Upload a test CSV (or use bundled `test_data.csv`)
2. Choose one of the five saved classifiers
3. See Accuracy, AUC, Precision, Recall, F1, and MCC recomputed on that file
4. Inspect a classification report and a confusion-matrix heatmap
5. Compare all five models on the same uploaded sessions
