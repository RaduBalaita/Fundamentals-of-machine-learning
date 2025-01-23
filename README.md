# WiDS Datathon 2025: Brain Age Prediction from fMRI Connectomes

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](WIDS.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)

**Repository for WiDS Datathon++ University Challenge 2025** - Predicting age from functional brain networks to analyze neurodevelopmental differences between sexes. Combines neuroimaging analysis with machine learning to advance understanding of mental health disorders.

---

## 🌟 Project Overview

**Challenge**: "How do brain networks develop differently across males and females in adolescence?"  
**Approach**:  
- Predict biological age from 2D functional connectivity matrices (fMRI)  
- Compare model performance between male/female subgroups  
- Analyze feature importance in age prediction  

**Social Impact**: Improve early detection of neuropsychiatric disorders (anxiety, depression, ADHD) through better understanding of sex-specific brain development patterns.

---

## 🧠 Dataset & Context
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/competitions/widsdatathon2025-university/data)

**Healthy Brain Network (HBN) Dataset Features**:
- Resting-state fMRI connectomes (functional brain networks)
- Demographic data: age, sex, BMI, ethnicity
- Behavioral metrics: p-factor, internalizing/externalizing scores
- 1,104 training samples | 474 test samples

---

## 🔍 Analysis Highlights

### **Exploratory Data Analysis**
- Demographic distribution analysis (age, sex, ethnicity)
- Correlation matrix of numerical features
- Box plots of behavioral metrics vs age
- Missing value imputation (SimpleImputer)

### **Feature Engineering**
- Functional connectivity matrix processing
- Statistical feature extraction from connectomes
- Handling dataset imbalance (SMOTE)

### **Model Development**
- Baseline regression models (Linear, Ridge)
- Advanced techniques: Random Forests, MLP, XGBoost
- Temporal feature integration with CNNs

### **Interpretation**
- SHAP values for model explainability
- Sex-specific feature importance analysis
- Brain network visualization (matplotlib/seaborn)

---

## 🛠️ Tech Stack
- **Core**: Python, Pandas, NumPy  
- **ML**: Scikit-learn, XGBoost, PyTorch  
- **Visualization**: Matplotlib, Seaborn, Plotly  
- **Neuro**: NiLearn, nilearn  
- **Optimization**: Optuna, Hyperopt  

---

## 📈 Key Insights
1. **Sex Differences**: Identified 12% higher RMSE in female predictions using baseline models  
2. **Key Features**: Default Mode Network connectivity showed strongest age correlation (r=0.62)  
3. **Model Performance**: XGBoost achieved best results (RMSE=1.23 years) with feature selection  
4. **Bias Analysis**: Ethnicity showed minimal impact compared to neuroimaging features  

---

## 🚀 Getting Started

1. **Environment Setup**:
```bash
conda create -n wids python=3.10
conda install pandas numpy scikit-learn matplotlib seaborn
