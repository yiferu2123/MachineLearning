# Job Salary Prediction (Supervised ML)

This project builds and evaluates multiple machine learning models to predict job salaries based on various features such as job title, location, and experience level.

## 📊 Overview
-   **File**: `Job_Salary_ML_Prediction.ipynb`
-   **Dataset**: `jobs_in_data.csv` containing 9,355 records.
-   **Target Variable**: `salary_in_usd`
-   **Goal**: Train regression models to estimate salary and identify key factors influencing pay.

## 🛠 Libraries Used
-   **Data Processing**: `pandas`, `numpy`, `scikit-learn` (StandardScaler)
-   **Visualization**: `matplotlib`, `seaborn`
-   **Modeling**: `scikit-learn` (RandomForest, SVR, KNN), `xgboost`
-   **Model Selection**: `GridSearchCV`

## 🧠 Models & Performance
The project compares several regression models using **Mean Squared Error (MSE)** and **R² Score**. Hyperparameter tuning was performed using Grid Search.

| Model | R² Score (Tuned) | MSE (Tuned) | Key Hyperparameters |
|-------|------------------|-------------|---------------------|
| **XGBoost** | **0.44** | **0.540** | `lr=0.05`, `depth=7`, `n_est=100` |
| Random Forest | 0.43 | 0.548 | `depth=10`, `n_est=300` |
| SVR | 0.39 | 0.583 | `C=1`, `epsilon=0.5`, `kernel='rbf'` |
| KNN | 0.33 | 0.644 | `n_neighbors=7` |

**Conclusion**: The **XGBoost** model performed best, explaining approximately 44% of the variance in salaries.

## 🔍 Key Findings
-   **Feature Importance**: The most influential features in predicting salary were:
    1.  **Job Title Frequency**
    2.  **Company Location**
    3.  **Job Category**
    4.  **Work Year**
-   **Model Artifact**: The best performing XGBoost model is saved as `xgboost_salary_predictor.pkl` for potential deployment.

## 🚀 Steps to Reproduce
1.  **Load Data**: Ensure `jobs_in_data.csv` is in the project root.
2.  **Run Notebook**: Execute cells to preprocess data (scaling, encoding), train models, and visualize results.
3.  **Inference**: Use the saved `.pkl` model to make new predictions.
