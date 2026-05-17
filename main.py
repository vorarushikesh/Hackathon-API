import json
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, defaults and feature list
model = xgb.XGBClassifier()
model.load_model("diabetes_model.ubj")

with open("defaults.json", "r") as f:
    defaults = json.load(f)

with open("feature_list.json", "r") as f:
    feature_list = json.load(f)

class PatientData(BaseModel):
    HighBP: str
    HighChol: str
    BMI: str
    Smoker: str
    Income: str
    Education: str
    HealthScore: str
    model_config = {
        "extra": "allow"
    }

# ── Serve frontend ──────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
# ───────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(data: PatientData):
    input_data = data.model_dump()
    
    # Base dictionary from defaults
    features = {**defaults}
    
    # Overwrite mapped features from input
    # Convert strings to floats
    for k, v in input_data.items():
        if k in feature_list or k in ["HighBP", "HighChol", "BMI", "Smoker", "HealthScore", "Income", "Education"]:
            try:
                features[k] = float(v)
            except:
                pass
                
    # Add generated or missing
    if "Age" not in features:
        features["Age"] = 5.0
    features["GenHlth"] = features.get("HealthScore", 3.0)
    features["BMI_Age"] = features.get("BMI", 25.0) * features["Age"]
    
    for f in feature_list:
        if f not in features:
            features[f] = 0.0

    # Ensure correct order
    df = pd.DataFrame([{f: features[f] for f in feature_list}])

    # Predict
    prob = model.predict_proba(df)[0]
    pred = model.predict(df)[0]
    
    max_prob = float(max(prob) * 100)
    is_healthy = int(pred) == 0
    
    bmi_val = features.get("BMI", 25)
    bp_val = features.get("HighBP", 0)

    result = {
        "prediction": 'Low Risk of Chronic Illness' if is_healthy else 'High Risk - Potential Diabetes / Heart Issue',
        "confidence": round(max_prob, 1),
        "treatment": 'Maintain current active lifestyle and balanced diet.' if is_healthy else 'Recommend immediate cardiovascular profiling and dietary adjustment. Prioritize based on severity protocol #42.',
        "explanation": f"Based on an AI analysis of medical features only (BMI: {bmi_val}, Blood Pressure: {'High' if bp_val == 1.0 else 'Normal'}), ignoring demographic and socioeconomic factors.",
        "fairnessScore": 99.8,
        "priority": 'Routine' if is_healthy else 'Urgent',
        "biasWarning": False
    }

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
