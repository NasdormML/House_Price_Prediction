import joblib
import os


model_path = 'models/xgb_model.pkl'
saved_model_path = 'models/trained_model.pkl'

os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
joblib.dump(model_path, saved_model_path)
print(f"Model saved to {saved_model_path}")
