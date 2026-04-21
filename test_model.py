import xgboost as xgb
import json

model = xgb.XGBClassifier()
model.load_model("diabetes_model.ubj")
print("Model loaded successfully")
print(model.get_booster().feature_names)
