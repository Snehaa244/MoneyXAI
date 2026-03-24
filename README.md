# MoneyXAI: Explainable AI for Money Laundering Detection

## Overview

MoneyXAI is a machine learning-based model designed to detect money laundering transactions using XGBoost combined with Synthetic Minority Over-sampling Technique (SMOTE). The model addresses severe class imbalance in financial transaction datasets, improving fraud detection accuracy and adaptability over traditional rule-based Anti-Money Laundering (AML) systems.

## Features

- **Fraud Detection:** Enhances precision, recall, and F1-score for identifying suspicious transactions.
- **Class Imbalance Handling:** Uses SMOTE to generate synthetic samples for fraudulent transactions.
- **AI-driven AML System:** Shifts from rule-based detection to an adaptable machine learning approach.

## Dataset

- **Name:** SAML-D (Synthetic Financial Dataset)
- **Size:** 9,504,852 transactions
- **Fraudulent Transactions:** 0.1039% of the dataset
- **Key Features:**
  - Transaction details (amount, date, sender, receiver)
  - Payment type, currency, bank location
  - Target variable: `is_laundering`

## Methodology

### 1. Data Preprocessing

- **Feature Selection:** Removed redundant features like exact timestamps to prevent data leakage.
- **Handling Missing Values:**
  - Numerical: Imputed using median values.
  - Categorical: Imputed using the most frequent value.
- **Data Transformation:**
  - Log transformation applied to transaction amounts.
  - Ordinal encoding for categorical features.
  - RobustScaler applied to numerical values.

### 2. Handling Class Imbalance with SMOTE

- **Issue:** Fraudulent transactions are only 0.1039% of the dataset.
- **Solution:** SMOTE generates synthetic samples for fraudulent transactions to improve model balance.
- **Outcome:** Ensures better recall and precision while preventing bias towards non-fraudulent transactions.

### 3. Model Training & Evaluation

- **Models Used:**
  - Baseline: XGBoost (without SMOTE)
  - Enhanced Models: XGBoost + SMOTE, Random Forest + SMOTE, ExtraTreesClassifier + SMOTE
- **Training Process:**
  - Train-Test Split: 80% training, 20% testing
  - Hyperparameter Tuning: GridSearchCV for optimal learning rate (`eta`) and depth (`max_depth`)
- **Performance Metrics:**
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC Score

#### Model Performance Comparison:

| Model                 | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC |
| --------------------- | ------------- | ---------- | ------------ | ------- |
| XGBoost               | 99.5          | 99.4       | 99.4         | 0.8054  |
| Random Forest + SMOTE | 100.0         | 99.9       | 99.9         | 99.9    |
| ExtraTrees + SMOTE    | 100.0         | 100.0      | 100.0        | 100.0   |

✅ **Best Performing Model:** XGBoost + SMOTE

## Key Findings

### 1. Improvements Over Traditional AML Systems

- Traditional AML systems rely on rigid rule-based detection with high false positives.
- MoneyXAI provides better fraud detection with lower false positives and improved generalization.

### 2. Limitations & Challenges

- **Data Limitations:** Synthetic data was used due to real-world financial data confidentiality.
- **Computational Complexity:** SMOTE increases training time, and XGBoost requires significant hyperparameter tuning.

## Conclusion

The MoneyXAI model enhances money laundering detection by integrating SMOTE with XGBoost:

- ✅ Addresses severe class imbalance.
- ✅ Achieves high accuracy, recall, and F1-score.
- ✅ More effective than traditional AML systems.
- ✅ Scalable for real-world financial institutions.

## Installation & Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/yourrepo/moneyxai.git
   cd moneyxai
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```sh
   python train.py
   ```

## License

This project is licensed under the MIT License.
