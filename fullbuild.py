import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Загрузка данных
data = pd.read_csv('house/train.csv')

# Создание новых признаков
data['Age'] = data['YrSold'] - data['YearBuilt']
data['TotalAreaRatio'] = data['TotalBsmtSF'] / data['GrLivArea']

# Разделение данных на признаки и целевую переменную
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_columns = X_train.select_dtypes(include=['object']).columns
numeric_columns = X_train.select_dtypes(include=['number']).columns

# Создание пайплайна для обработки категориальных признаков
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Создание пайплайна для обработки числовых признаков
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

# Создание общего пайплайна
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Создание пайплайна модели
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror'))
])

# Параметры для RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': [200],
    'regressor__max_depth': [4],
    'regressor__learning_rate': [0.05],
    'regressor__subsample': [0.6],
    'regressor__colsample_bytree': [0.4],
    'regressor__gamma': [0.2],
    'regressor__reg_alpha': [0.1],
    'regressor__reg_lambda': [1]
}

# RandomizedSearchCV для подбора гиперпараметров
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,  # Уменьшил количество итераций для быстроты выполнения
    scoring='neg_mean_squared_error',
    cv=4,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Обучение модели
random_search.fit(X_train, y_train)

# Результаты RandomizedSearchCV
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

# Лучшая модель
best_model = random_search.best_estimator_

# Предсказания и метрики
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Логирование параметров, метрик и модели с помощью MLflow
with mlflow.start_run() as run:
    # Логирование параметров модели
    mlflow.log_params(random_search.best_params_)

    # Логирование метрик
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    # Логирование модели
    mlflow.sklearn.log_model(best_model, "xgb_regressor_model")

    # Получение текущего run ID
    run_id = run.info.run_id
    print("Run ID:", run_id)

# Загрузка модели из MLflow
logged_model = f'runs:/{run_id}/xgb_regressor_model'
loaded_model = mlflow.sklearn.load_model(logged_model)

# Проверка загруженной модели
loaded_predictions = loaded_model.predict(X_test)
print("Loaded model predictions:", loaded_predictions[:5])  # Вывод первых 5 предсказаний для проверки
