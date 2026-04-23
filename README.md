# Corporate Bankruptcy Prediction — Big Data ML Project

A machine learning pipeline built in R to predict corporate bankruptcy using financial ratio data from 6,819 Taiwanese companies. Developed as part of the Big Data module at the University of Birmingham Business School.

---

## Overview

This project applies three supervised learning models to a binary classification problem — predicting whether a company will go bankrupt based on 95 financial ratio features. The full pipeline covers data quality assessment, feature selection, data preparation, model training, evaluation, and an interactive dashboard for live predictions.

---

## Models

| Model | Accuracy | Recall | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 88.1% | 90.9% | 0.941 |
| Random Forest | 93.7% | 70.5% | 0.924 |
| Neural Network | 92.7% | 68.2% | 0.853 |

Random Forest achieved the best overall accuracy and F1 score. Logistic Regression achieved the highest recall and AUC-ROC, making it the preferred model for risk-averse credit decisions.

---

## Pipeline

```
data_quality.R          → EDA, outlier detection, distributions, correlation
feature_selection.R     → NZV removal, Random Forest Gini importance, binary flag retention
data_preparation.R      → Stratified 80/20 split, SMOTE balancing, StandardScaler
model1_logistic.R       → Logistic Regression with 10-fold CV
model2_random_forest.R  → Random Forest with mtry tuning via 5-fold CV (500 trees)
model3_neural_network.R → Feedforward NN — 2 hidden layers (16 → 8 nodes), sigmoid activation
model_comparison.R      → Combined ROC curves, metrics table, confusion matrix summary
feature_extraction.R    → Feature importance plots, LR coefficient visualisation
dashboard_app.R         → R Shiny dashboard with live prediction tool
```

---

## Key Technical Decisions

- **SMOTE** applied to training set only to address 96.8% / 3.2% class imbalance
- **22 features** selected from 95 — top 20 by RF Gini importance plus 2 binary distress flags retained on theoretical grounds (Altman, 1968; Zmijewski, 1984)
- **Binary flags protected** from near-zero variance removal — Gini importance systematically undervalues binary features (Strobl et al., 2007)
- **Scaler fit on training data only** to prevent data leakage
- **AUC-ROC** used as optimisation metric throughout — more appropriate than accuracy for imbalanced classification

---

## Dashboard

An interactive R Shiny dashboard provides:
- Dataset overview and class distribution
- Model-by-model performance metrics, confusion matrices, and ROC curves
- Feature importance and Logistic Regression coefficient visualisations
- Live prediction tool — input any company's financial ratios and receive bankruptcy probability from all three models, with stakeholder-specific recommendations for lenders, investors, and regulators

---

## Dataset

Taiwanese company financial data — 6,819 companies, 95 features, binary bankruptcy target (3.2% positive rate). Features include profitability ratios, liquidity ratios, leverage ratios, cash flow metrics, and growth rates.

---

## Dependencies

```r
packages <- c("readxl", "dplyr", "ggplot2", "tidyr", "caret", "randomForest",
              "smotefamily", "neuralnet", "pROC", "corrplot", "shiny",
              "shinydashboard", "DT")
```

---

## References

- Altman, E.I. (1968) Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *Journal of Finance*, 23(4), pp.589–609.
- Breiman, L. (2001) Random Forests. *Machine Learning*, 45(1), pp.5–32.
- Strobl, C. et al. (2007) Bias in Random Forest Variable Importance Measures. *BMC Bioinformatics*, 8(1), p.25.
- Tam, K.Y. and Kiang, M.Y. (1992) Managerial Applications of Neural Networks. *Management Science*, 38(7), pp.926–947.
- Zmijewski, M.E. (1984) Methodological Issues Related to the Estimation of Financial Distress Prediction Models. *Journal of Accounting Research*, 22, pp.59–82.
