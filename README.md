# Fraud_Detection

# 🛡️ Fraud Detection System using Machine Learning

This project involves building a machine learning model to detect fraudulent financial transactions using a large real-world dataset. The goal is to help financial institutions minimize fraud-related losses by accurately identifying suspicious activity.

---

## 📁 Project Structure

- `fraud_detection.ipynb` – Jupyter Notebook with end-to-end model pipeline.
- `fraud_data.csv` – Dataset with over 6 million transaction records.
- `README.md` – Documentation file.

---

## ⚙️ Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

---

## 🧹 1. Data Cleaning

- **Missing Values:** Checked and confirmed no missing entries in critical columns.
- **Outliers:** Handled based on domain knowledge and transaction thresholds.
- **Multi-collinearity:** Features like `oldbalanceOrg` and `newbalanceOrig` were analyzed for correlation.

Dropped columns:
- `nameOrig`, `nameDest` – anonymized IDs, not useful for modeling.
- `isFlaggedFraud` – extremely rare positive cases, not predictive.
- `step` – time-based unit not adding value to fraud prediction.

---

## 🤖 2. Fraud Detection Model

We trained and evaluated multiple classification models:

- Logistic Regression
- Decision Tree
- Random Forest (Best Performer)

After experimenting with class imbalance techniques (like SMOTE), Random Forest gave the most balanced performance.

**Final Model:**  
`RandomForestClassifier(max_depth=20, min_samples_split=10, n_estimators=200, random_state=42)`

---

## 📊 3. Variable Selection

We selected features based on:

- Domain knowledge
- Correlation analysis
- Feature importance from Random Forest

Selected key variables:
- `type`, `amount`
- `oldbalanceOrg`, `newbalanceOrig`
- `oldbalanceDest`, `newbalanceDest`

---

## 🧪 4. Model Performance

### Before Balancing:
- **Accuracy:** 99.96%
- **Recall (Fraud):** 0.71
- **Precision (Fraud):** 0.95

### After SMOTE Balancing:
- **Accuracy:** ~98%
- **Recall (Fraud):** ~0.97
- **Precision (Fraud):** Low (as expected due to over-sampling)

**Confusion Matrix Example:**
