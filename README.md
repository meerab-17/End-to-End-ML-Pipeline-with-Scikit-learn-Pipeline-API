# End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API

## Objective

The goal of this task was to build a **production-ready and reusable machine learning pipeline** using the Telco Customer Churn dataset. The pipeline is designed to predict whether a customer will churn based on their demographic and service usage data.

---

##  Methodology / Approach

1. **Data Preprocessing**  
   - Cleaned the dataset (e.g., handled missing 'TotalCharges', dropped 'customerID')
   - Converted the target 'Churn' column to binary (Yes → 1, No → 0)

2. **Pipeline Construction**  
   - Used 'ColumnTransformer' to apply:
     - 'StandardScaler' to numerical features
     - 'OneHotEncoder' to categorical features
   - Combined with:
     - 'Logistic Regression' pipeline
     - 'Random Forest' pipeline

3. **Hyperparameter Tuning**  
   - Used 'GridSearchCV' with 5-fold cross-validation to tune:
     - 'C' (for Logistic Regression)
     - 'n_estimators' and 'max_depth' (for Random Forest)

4. **Model Evaluation & Export**  
   - Evaluated models on test data using classification metrics
   - Exported the best-performing pipeline using 'joblib'

---

##  Key Results / Observations

- Both models performed well, but **Random Forest** gave better accuracy and generalization on test data.
- The pipeline correctly predicted churn for unseen customers (example output: '[0 1 0 0 0]').
- The entire flow — from preprocessing to prediction — is encapsulated in a single, reusable 'scikit-learn' pipeline.
- The final pipeline is saved as 'churn_pipeline.joblib' and can be used directly for inference in production environments.

---

##  Technologies Used

- Python, pandas, numpy  
- scikit-learn ('Pipeline', 'ColumnTransformer', 'GridSearchCV')  
- joblib (for model export)
