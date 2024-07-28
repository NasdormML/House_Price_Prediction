# House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)

This project aims to predict house prices using machine learning algorithms. The dataset includes features such as the size of the house, number of bedrooms, location, etc.

## Project Structure

- `house/`: Folder containing the dataset.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model evaluation.
  - `Visual.ipynb`: Exploratory Data Analysis notebook.
  - `XGB_regress.ipynb`: Notebook for model training and evaluation.
- `xgb_models/`: Folder with XGB models.
  - `xgb_model.pkl`: Saved main XGB model.
- `README.md`: Project overview and instructions.

## Data

The dataset used in this project includes the following features:
- Size of the house (square footage)
- Number of bedrooms
- Number of bathrooms
- Location (neighborhood)
- Year built
- Sale price (target variable)

The data is preprocessed to handle missing values, encode categorical variables, and normalize numerical features.

## Feature Engineering

Several new features were created to enhance the model's predictive power:
- **Age**: Difference between the year sold and the year built.
- **TotalAreaRatio**: Ratio of total basement area to ground living area.

## Modeling

The primary model used in this project is XGBoost, chosen for its high performance with tabular data. The model was trained with the following hyperparameters:
- Learning rate: 0.05
- Subsample: 0.8
- Reg lambda: 3
- Reg alpha: 1
- n_estimators: 200
- Gamma: 0.2
- Colsample by tree: 0.2
- Max depth: 4

## Evaluation

The model was evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Results:
- MAE: 15734
- RMSE: 125

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

You can install them using:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
