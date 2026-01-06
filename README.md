# Diabetes & Stroke Prediction ML Project

This repository implements machine learning pipelines to predict diabetes and stroke risks using real-world healthcare datasets. Models help identify high-risk patients early.

## Project Description

Focuses on binary classification for diabetes (Pima dataset) and stroke (healthcare-stroke dataset). Key risk factors analyzed: age, hypertension, heart_disease, BMI, avg_glucose_level, smoking_status.Handles class imbalance, feature scaling, and hyperparameter tuning for robust predictions.

## No Local Setup Needed

**GitHub Viewer**: Notebooks render with outputs—no install.

**Google Colab**:
1. Click notebook → "Open in Colab".
2. Run cells; data loads via pd.read_csv or Kaggle API (free).
3. Predict instantly, e.g., input age=50, BMI=30 → stroke probability.



## Models & Performance

| Model             | Diabetes Acc | Stroke Acc (post-SMOTE) | Notes                  |
|-------------------|--------------|-------------------------|------------------------|
| Logistic Reg     | 0.77        | 0.82                   | Baseline              |
| KNN              | 0.75        | 0.91                   | Distance-based        |
| Decision Tree    | 0.70        | 0.88                   | Interpretable         |
| Random Forest    | 0.81        | 0.95                   | Best overall          |
| SVM              | 0.79        | 0.94                   | Kernel RBF            |

## Key Insights

- Stroke risk surges after age 50, BMI>25, glucose>100.
- Diabetes doubles stroke odds; 80% strokes in non-diabetics still predictable via other factors.

## Tech Stack

- **Core**: Python, scikit-learn, pandas, numpy
- **Viz**: matplotlib, seaborn
- **Notebooks**: Jupyter (Colab-compatible)

## Quick Demo (Colab)

```python
import pandas as pd
from joblib import load  # Assume model saved

data = pd.DataFrame({
    'age': , 'hypertension':, 'heart_disease': ,[1]
    'avg_glucose_level': , 'bmi': , 'smoking_status': ['formerly smoked']
    # Add other features...
})
model = load('models/random_forest_stroke.pkl')
pred = model.predict_proba(data)  # Stroke prob[1]
print(f"Stroke Risk: {pred:.2%}")
