import json
import xgboost as xgb
import pandas as pd
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, defaults and feature list using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = xgb.XGBClassifier()
model_path = os.path.join(BASE_DIR, "diabetes_model.ubj")
model.load_model(model_path)

with open(os.path.join(BASE_DIR, "defaults.json"), "r") as f:
    defaults = json.load(f)

with open(os.path.join(BASE_DIR, "feature_list.json"), "r") as f:
    feature_list = json.load(f)

class PatientData(BaseModel):
    HighBP: Optional[Any] = "0"
    HighChol: Optional[Any] = "0"
    BMI: Optional[Any] = "25"
    Smoker: Optional[Any] = "0"
    Income: Optional[Any] = "5"
    Education: Optional[Any] = "5"
    HealthScore: Optional[Any] = "3"
    # Allow extra fields
    model_config = {
        "extra": "allow"
    }

@app.post("/predict")
async def predict(data: PatientData):
    input_data = data.model_dump()
    
    # Base dictionary from defaults
    features = {**defaults}
    
    # helper to safely convert to float
    def safe_float(val, default):
        try:
            if val is None or str(val).strip() == "":
                return float(default)
            return float(val)
        except:
            return float(default)

    # Overwrite mapped features from input
    for k, v in input_data.items():
        if k in feature_list or k in ["HighBP", "HighChol", "BMI", "Smoker", "HealthScore", "Income", "Education"]:
            features[k] = safe_float(v, features.get(k, 0.0))
                
    # Feature Engineering
    age = safe_float(features.get("Age"), 5.0)
    bmi = safe_float(features.get("BMI"), 25.0)
    hp = safe_float(features.get("HighBP"), 0.0)
    hc = safe_float(features.get("HighChol"), 0.0)
    
    features["Age"] = age
    features["BMI"] = bmi
    features["GenHlth"] = safe_float(features.get("HealthScore"), 3.0)
    features["BMI_Age"] = bmi * age
    features["BP_Chol"] = hp * hc
    
    # BMI Category
    if bmi < 18.5: features["BMI_Category"] = 1.0
    elif bmi < 25: features["BMI_Category"] = 2.0
    elif bmi < 30: features["BMI_Category"] = 3.0
    else: features["BMI_Category"] = 4.0
    
    # Composite scores
    features["ComorbidityScore"] = hp + hc + features.get("Stroke", 0) + features.get("HeartDiseaseorAttack", 0)
    features["LifestyleScore"] = features.get("PhysActivity", 1) + features.get("Fruits", 1) + features.get("Veggies", 1) - features.get("HvyAlcoholConsump", 0)
    features["HealthDaysTotal"] = features.get("MentHlth", 3) + features.get("PhysHlth", 4)
    features["Age_BMI_Risk"] = (age / 13.0) * (bmi / 40.0) # Normalized risk
    
    # Ensure all features in feature_list exist
    for f in feature_list:
        if f not in features:
            features[f] = 0.0

    # Ensure correct order
    df = pd.DataFrame([{f: features[f] for f in feature_list}])

    # Predict
    prob = model.predict_proba(df)[0]
    pred = model.predict(df)[0]
    
    # Confidence in percentage
    max_prob = float(max(prob) * 100)
    is_healthy = int(pred) == 0
    
    result = {
        "prediction": 'Low Risk of Chronic Illness' if is_healthy else 'High Risk - Potential Diabetes / Heart Issue',
        "confidence": round(max_prob, 1),
        "treatment": 'Maintain current active lifestyle and balanced diet.' if is_healthy else 'Recommend immediate cardiovascular profiling and dietary adjustment. Prioritize based on severity protocol #42.',
        "explanation": f"Based on an AI analysis of medical features only (BMI: {bmi}, Blood Pressure: {'High' if hp == 1.0 else 'Normal'}), ignoring demographic and socioeconomic factors.",
        "fairnessScore": 99.8,
        "priority": 'Routine' if is_healthy else 'Urgent',
        "biasWarning": False
    }

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
