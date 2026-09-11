# 🫀 Heart Disease Classification Model

A binary classification pipeline using Logistic Regression to predict heart disease risk based on patient clinical features. The workflow enforces strict data science standards, including data leakage prevention, feature-specific scaling, and class stratification.

---

## 📁 Dataset Information
This project utilizes the **`heart.csv`** dataset included in the root directory of this repository.

* **File:** `heart.csv`
* **Target Variable:** `HeartDisease` (`1` = Presence of heart disease, `0` = Normal)
* **Numerical Features:** `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`
* **Categorical Features:** One-hot encoded prior to modeling (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`)

---

## 📊 Key Performance Metrics
- **Accuracy:** 89%
- **ROC-AUC Score:** 0.93
- **Recall (Sensitivity):** 93% for Class 1 (Successfully detected 95 out of 102 positive cases)

![Confusion Matrix]( <img width="498" height="432" alt="image 7" src="https://github.com/user-attachments/assets/32e8d2f0-f6cd-4f45-9f0a-bf064ddaeb5e" />
 )

---

## 🛠️ Data Pipeline & Methodology
1. **Target Separation:** Separated target label `HeartDisease` from feature set $X$.
2. **Stratified Splitting:** Applied an 80/20 train-test split (`stratify=y`) to maintain class ratios across training and evaluation sets.
3. **Leakage Prevention:** Standardized continuous features using `StandardScaler` fitted **strictly on `X_train`** and transformed onto `X_test`.
4. **Model Execution:** Trained using Scikit-Learn's `LogisticRegression` solver.

---

## 📈 Classification Report

```text
              precision    recall  f1-score   support

           0       0.91      0.83      0.87        82
           1       0.87      0.93      0.90       102

    accuracy                           0.89       184
   macro avg       0.89      0.88      0.88       184
weighted avg       0.89      0.89      0.89       184


## 🧠 Key Learnings & Engineering Insights

### 1. Handling Invalid Zero Values in Clinical Metrics
* Identified that values of `0` in physiological variables like `Cholesterol` and `RestingBP` represented missing data rather than actual zero measurements.
* Replaced invalid `0` entries with `NaN` before performing missing value imputation.

### 2. Avoiding Target Leakage During Imputation
* **The Concept:** Avoided using the target label (`HeartDisease`) to calculate missing value statistics, as this introduces target leakage—leading to overly optimistic cross-validation scores that fail on unseen real-world data.
* **The Solution:** Used independent baseline features (such as demographic and physiological variables) that are naturally available when a patient first checks in, prior to diagnostic outcome determination.

### 3. Choosing Robust Central Tendency (Mean vs. Median)
* Avoided using the mean due to heavily skewed distributions and extreme clinical outliers (e.g., cholesterol readings up to 600) that artificially inflate mean calculations.
* Selected **Median** as a safer measure of central tendency robust against extreme values.

### 4. Grouped Median Imputation Over Global Averages
* Applied **Grouped Median** imputation stratified by clinical subgroups (`Sex` + `AgeGroup`) rather than a simple global median across the entire dataset.
* Preserved natural demographic variances in physiological baseline trends, creating a more realistic dataset structure and improving downstream model predictive power.

### 5. Strict Pipeline Isolation & Scaling
* Enforced strict data boundaries during feature scaling by fitting `StandardScaler` strictly on `X_train` (`fit_transform`) and applying `transform` to `X_test` to prevent data leakage across train and test splits.
