# House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.9.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
[![CI](https://github.com/NasdormML/House_price_try/actions/workflows/ci.yml/badge.svg)](https://github.com/NasdormML/House_price_try/actions/workflows/ci.yml)

## Overview
This project predicts house prices based on key features such as size, location, and age. It leverages advanced machine learning techniques, including **XGBoost**, to deliver accurate price estimates. These insights can assist real estate agents, buyers, and sellers in making informed decisions.

## Key Features
- **Predictive Model**: Built using XGBoost, optimized for tabular data.
- **Feature Engineering**: Created custom features like house age and area ratios to boost model performance.
- **Performance Metrics**: Achieved competitive MAE and RMSE scores.
- **End-to-End Workflow**: Includes data preprocessing, exploratory analysis, modeling, and deployment.

---

## Results
- **Mean Absolute Error (MAE)**: 15,734  
- **Root Mean Squared Error (RMSE)**: 125  

The model provides reliable price predictions, reducing uncertainty in property valuation.

---

## Project Structure
```
House_price_prediction/
├── house/                  # Dataset folder
├── notebooks/              # Jupyter notebooks
│   ├── Visual.ipynb        # Exploratory Data Analysis
│   └── XGB_regress.ipynb   # Model training and evaluation
├── models/                 # Saved models
│   ├── xgb_model.pkl       # Main XGBoost model
│   └── trained_model.pkl   # Alternative trained model
├── scripts/                # Utility scripts
│   ├── save_model.py       # Script for saving models
│   └── deploy_model.py     # Script for deployment
├── requirements.txt        # Dependencies
└── README.md               # Project overview
```

---

## Data
The dataset contains key features influencing house prices:
- **Size**: Total area in square footage.
- **Bedrooms & Bathrooms**: Count of each.
- **Location**: Neighborhood information.
- **Year Built**: Construction year.
- **Sale Price**: Target variable for prediction.

### Data Preprocessing:
- Imputation for missing values.
- Encoding for categorical variables.
- Scaling for numerical features.

---

## Modeling
### Model: XGBoost
- Chosen for its speed and performance on tabular datasets.
- Trained using advanced hyperparameters:
  - **Learning Rate**: 0.05
  - **Max Depth**: 4
  - **Subsample**: 0.8
  - **Colsample by Tree**: 0.2
  - **n_estimators**: 200

### Validation
- Applied **cross-validation** to ensure consistent results.

---

## How to Run

### Prerequisites
Install required libraries:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NasdormML/House_price_try.git
   cd House_price_try
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- **EDA**: Run `Visual.ipynb` to explore the dataset and trends.
- **Model Training**: Use `XGB_regress.ipynb` to train and evaluate the predictive model.

---

## Business Value
- **Accurate Pricing**: Helps set realistic property prices, increasing transaction efficiency.
- **Market Insights**: Identifies key factors driving property value.
- **Risk Reduction**: Assists buyers in avoiding overpayment.

---

## Contact
Have questions or feedback? Reach out via email: **nasdorm.ml@inbox.ru**

---
