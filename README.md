# House Price Prediction using Machine Learning

## Project Overview

This project predicts the sales price of houses using a messy real-world dataset. The pipeline covers data cleaning, feature selection, model training, hyperparameter tuning, and evaluation. The final model is a tuned Random Forest Regressor capable of capturing the complex relationships between house features and their sale prices.

---

## Dataset

- Dataset used: `AmesHousing.csv`
- Contains both categorical and numerical features.
- Target variable: `SalePrice`

---

## Data Cleaning & Preprocessing

- Removed columns with excessive missing values (>1000 missing entries).
- Filled missing categorical values with mode.
- Filled missing numerical values with mean.
- Applied one-hot encoding for categorical features.
- Scaled numeric features using `StandardScaler` for linear models.

---

## Feature Selection

- Used `XGBRegressor` to identify the most influential features based on feature importance (gain).
- Selected top 20 features for model training.
- Used SHAP values for visualizing feature impact.

---

## Modeling

### Linear Regression

- Model trained on scaled top 20 features.
- Captured general trends but limited in handling complex interactions.
- Train R²: 0.83, Test R²: 0.85

### Decision Tree Regressor

- Model trained on unscaled top 20 features.
- High training R² (0.99) indicating overfitting.
- Test R² dropped to 0.84.

### Random Forest Regressor

- Ensemble method to reduce overfitting from Decision Tree.
- Train R²: 0.96, Test R²: 0.902
- Robust model for generalization on unseen data.

---

## Hyperparameter Tuning

- Applied `RandomizedSearchCV` to optimize Random Forest parameters:
  - `n_estimators`: 50, 100, 150, 200
  - `max_depth`: 5, 10, 15, 20, 25, 30, None
  - `min_samples_split`: 5, 10, 15, 20
- Achieved best model with tuned parameters, improving prediction performance.

---

## Evaluation

| Model      | Train R²  | Test R² | Train MSE | Test MSE |
|------------|-----------|---------|-----------|----------|
| Linear Reg | 0.830     | 0.850   | 9.9       | 1.1      |
| Decision T | 0.990     | 0.840   | 3.2       | 1.0      |
| Random F   | 0.960     | 0.902   | 2.3       | 7.8      |

- Linear Regression and Random Forest perform well in majority ranges.
- High-value houses predictions remain challenging due to data sparsity.

---

## Visualization

- Feature importance (XGBoost) visualized using barplots.
- SHAP summary plot to interpret feature impacts.
- Scatter plots comparing actual vs predicted prices for all three models.

---

## Summary & Insights

- Cleaning and preprocessing handled significant missing values.
- One-hot encoding effectively transformed categorical data.
- Feature selection via XGBoost reduced dimensionality and improved model focus.
- Ensemble methods like Random Forest improved performance and reduced overfitting.
- Models struggle with extreme high-value houses due to data sparsity.

---

## Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
```

---

## Usage

Google Colab Link: https://colab.research.google.com/drive/1U3XqgdAW5gjqbYFvlbwOSXM6NCAfxjQK?usp=sharing 

1. Clone the repository:

```bash
git clone https://github.com/chahmed312/ames-housing-price-prediction.git
cd ames-housing-price-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook
```

4. Execute cells sequentially to reproduce preprocessing, modeling, and evaluation results.

---



