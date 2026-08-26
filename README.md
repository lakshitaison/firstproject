# 🎗️ Breast Cancer Prediction using PCA and SVD

A machine learning mini-project that predicts breast cancer patient outcomes (Alive vs. Dead) using dimensionality reduction techniques — **PCA (Principal Component Analysis)** and **SVD (Singular Value Decomposition)** — combined with an ensemble of classifiers.

## Overview

This project walks through a complete ML pipeline: loading and exploring patient data, encoding and scaling features, reducing dimensionality with PCA and SVD, training multiple classifiers, and evaluating/comparing their performance. The best-performing model is saved for reuse on new patient predictions.

## Dataset

The notebook expects a CSV file (`breastcancer_updated.csv`) containing patient records with a binary target column, `Status_Dead` (0 = Alive, 1 = Dead), along with a mix of numeric and categorical clinical/demographic features.

> Update the file path in the "Load the Dataset" step to point to your own copy of the dataset (local upload, Google Drive, or direct path).

## Pipeline

1. **Import Libraries** — pandas, numpy, matplotlib, seaborn, and scikit-learn.
2. **Load the Dataset** — read the CSV into a pandas DataFrame.
3. **Exploratory Data Analysis (EDA)** — inspect shape, dtypes, missing values, and class balance.
4. **Preprocessing** — create the binary target, label-encode categorical columns, and split features/target.
5. **Feature Scaling** — standardize features with `StandardScaler` (mean = 0, std = 1).
6. **PCA** — reduce dimensionality while retaining 95% of variance.
7. **PCA 2D Visualization** — scatter plot of the first two principal components to check class separability.
8. **SVD** — apply `TruncatedSVD` for an alternative matrix-factorization-based reduction.
9. **Combine PCA + SVD** — concatenate both feature sets and perform an 80/20 stratified train/test split.
10. **Train Multiple Models** — Logistic Regression, SVM (RBF kernel), Random Forest, and a soft-voting ensemble of all three, with 5-fold cross-validation and `class_weight='balanced'` to handle class imbalance.
11. **Identify the Best Model** — select the top performer by ROC-AUC and print a full classification report.
12. **ROC Curve Plot** — compare true/false positive rate trade-offs across all models.
13. **Confusion Matrix** — visualize and break down correct/incorrect predictions for the best model.
14. **Model Comparison Chart** — bar chart comparing accuracy and ROC-AUC across all models.
15. **Predict for a New Patient** — helper function to run the trained pipeline on a new, unseen patient record.
16. **Save the Model** — persist the scaler, PCA, SVD, encoder, and trained models to `breast_cancer_model.pkl` via `joblib`.

## Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| SVM (RBF kernel) | Non-linear decision boundary, probability estimates enabled |
| Random Forest | 200 estimators, ensemble of decision trees |
| Voting Classifier | Soft-voting ensemble of the three models above |

All models use `class_weight='balanced'` to account for class imbalance in the dataset, and are evaluated primarily by **ROC-AUC** rather than raw accuracy.

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

## Usage

1. Open `Breast_Cancer_Prediction_PCA_SVD.ipynb` in Jupyter or Google Colab.
2. Place `breastcancer_updated.csv` in the expected path (or update the load step to point to your file).
3. Run all cells in order — the notebook is organized into clearly labeled steps.
4. The best model and full preprocessing pipeline are saved to `breast_cancer_model.pkl` at the end.

### Loading the saved model later

```python
import joblib

bundle = joblib.load('breast_cancer_model.pkl')
scaler = bundle['scaler']
pca = bundle['pca']
svd = bundle['svd']
best_model = bundle['best_model']
```

## Project Structure

```
.
├── Breast_Cancer_Prediction_PCA_SVD.ipynb   # Main notebook
├── breastcancer_updated.csv                 # Dataset (not included — provide your own)
├── breast_cancer_model.pkl                  # Saved model bundle (generated after running)
└── README.md
```

## Notes

- PCA requires dense, scaled numeric input — hence the `StandardScaler` step before dimensionality reduction.
- SVD works directly on the raw (scaled) matrix and can also handle sparse data, making it a useful complement to PCA.
- The final feature set stacks PCA and SVD outputs together to give the models a richer, combined representation of the data.

## Disclaimer

This project is for educational purposes only and is **not intended for real-world clinical or diagnostic use**. Predictions are based on a limited dataset and should not be used to inform medical decisions.
