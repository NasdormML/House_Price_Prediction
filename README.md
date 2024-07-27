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
  - `xgb_sandbox.pkl`: Saved test XGB model.
- `README.md`: Project overview and instructions.

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
