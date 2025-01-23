# WiDS Datathon++ University Challenge: Predicting Age from Brain Networks

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](WIDS.ipynb)
[![Python 3](https://img.shields.io/badge/Python-3-blue)](https://www.python.org/)

**This repository contains the project completed for the WiDS Datathon++ University Challenge, focusing on predicting age from functional brain networks (connectomes) derived from fMRI data.** The challenge, developed in collaboration with the Ann S. Bowers Women's Brain Health Initiative (WBHI), Cornell University, and UC Santa Barbara, aims to contribute to research on neuropsychiatric disorders and their development across genders.

**Dataset:** The datasets used for this challenge are provided by the Healthy Brain Network (HBN) initiative and can be found on Kaggle: [WiDS Datathon 2025 University Challenge](https://www.kaggle.com/competitions/widsdatathon2025-university/data).

---

## 🧠 Challenge Overview

### **Objective**
Develop machine learning models to predict the age of individuals based on their 2-dimensional functional brain networks (connectomes) derived from resting-state fMRI recordings. Subsequently, explore the models to analyze factors influencing prediction accuracy for males and females separately, offering insights into developmental differences in brain networks.

### **Key Tasks**

1. **Data Preprocessing:**
    *   Handle missing values in categorical columns by imputing with "Unknown".
    *   Impute missing values in numerical columns using the mean.
    *   Convert categorical columns to string type.
    *   One-hot encode categorical features.
    *   Scale numerical features using `StandardScaler`.

2. **Model Development:**
    *   Implement and evaluate the following regression models:
        *   **Linear Regression:** A simple baseline model.
        *   **Ridge Regression:** Linear regression with L2 regularization to prevent overfitting.
        *   **Random Forest Regressor:** An ensemble learning method for improved accuracy and robustness.

3. **Hyperparameter Tuning:**
    *   Optimize the Random Forest model using `GridSearchCV` with a reduced parameter grid for faster execution.

4. **Model Evaluation:**
    *   Assess model performance using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.
    *   Utilize K-Fold cross-validation to evaluate model stability and generalization.

5. **Feature Importance Analysis:**
    *   Visualize the top 10 most important features based on the Random Forest model.
    *   Use Principal Component Analysis (PCA) for dimensionality reduction and visualization of high-dimensional data.

### **Discussion Questions**

*   How do the prediction models perform on different age groups or developmental stages?
*   Are there specific brain regions or connections that are more predictive of age in males vs. females?
*   Can we identify any patterns or anomalies in the data that might be indicative of neuropsychiatric disorders?
*   How do the different preprocessing techniques impact model performance?

---

## 🛠️ Tools & Technologies

*   **Core:** Python, Pandas, NumPy, Jupyter Notebook
*   **Machine Learning:** Scikit-learn (Linear Regression, Ridge Regression, Random Forest, GridSearchCV, KFold)
*   **Data Visualization:** Matplotlib, Seaborn, PCA
*   **GPU Acceleration:** PyTorch, CuPy

---

## 📚 Key Learnings

1. **Data Preprocessing:** Implemented techniques for handling missing values, categorical feature encoding, and feature scaling.
2. **Regression Modeling:** Developed and evaluated different regression models for age prediction.
3. **Hyperparameter Tuning:** Utilized `GridSearchCV` to find the optimal hyperparameters for the Random Forest model.
4. **Model Evaluation:** Employed cross-validation and various metrics (RMSE, MAE, R-squared) to assess model performance.
5. **Feature Importance:** Gained insights into the most important features driving model predictions through visualization techniques.
6. **High-Dimensional Data Analysis:** Applied PCA for dimensionality reduction and visualization.
