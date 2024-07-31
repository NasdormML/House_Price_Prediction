import pytest
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def test_model_loading():
    model = joblib.load('model/xgb_model.pkl')
    assert isinstance(model, XGBRegressor), "Модель должна быть экземпляром XGBRegressor"

def test_model_performance():
    model = joblib.load('model/xgb_model.pkl')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    assert rmse < 30000, "RMSE модели должно быть меньше 30000"

if __name__ == "__main__":
    pytest.main()
