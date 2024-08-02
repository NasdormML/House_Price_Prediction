import pytest
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Здесь должна быть функция для загрузки данных
def load_data():
    data = pd.read_csv('house/train.csv')
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Тест для обучения модели
def test_training():
    X_train, X_test, y_train, y_test = load_data()
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    numeric_columns = X_train.select_dtypes(include=['number']).columns
    
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])



    # Инициализация модели (код из вашего ноутбука)
    model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=200, max_depth=4,
                               learning_rate=0.05,
                               subsample=0.6,
                               colsample_bytree=0.4,
                               gamma=0.2,
                               reg_alpha=0.1,
                               reg_lambda=1,
                               objective='reg:squarederror'))
                               ])

    # Обучение модели
    model.fit(X_train, y_train)
    
    # Проверка, что модель обучена
    assert model is not None, "Model training failed"
    
    # Прогнозирование
    predictions = model.predict(X_test)
    
    # Проверка, что предсказания не пустые
    assert len(predictions) == len(y_test), "Prediction failed"
    
    # Оценка (например, RMSE)
    mae = mean_absolute_error(y_test, predictions,)
    assert mae < 16300, f"Model performance is not sufficient, RMSE: {mae}"

