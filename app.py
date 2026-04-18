import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path("artifacts/model_v1.joblib")
RANDOM_STATE = 42
APP_ENV = os.getenv("APP_ENV", "development")

class PredictionInput(BaseModel):
    feature_1: float
    feature_2: float

app = FastAPI(
    title="API do serwowania modelu ML",
    description="Proste API w FastAPI do zwracania predykcji modelu.",
    version="1.0.0",
)

def build_training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.8, 1.0],
            [1.0, 1.2],
            [1.2, 0.9],
            [3.0, 3.2],
            [3.3, 2.9],
            [2.8, 3.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y

def train_and_save_model() -> LogisticRegression:
    X, y = build_training_data()

    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model

def load_model() -> LogisticRegression:
    if not MODEL_PATH.exists():
        return train_and_save_model()

    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

@app.get("/")
def read_root() -> dict:
    return {"message": "API dziala"}

@app.get("/info")
def info() -> dict:
    return {
        "model_type": type(model).__name__,
        "number_of_features": 2,
        "classes": ["klasa_0", "klasa_1"],
        "model_path": str(MODEL_PATH),
        "app_env": APP_ENV,
    }

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app_env": APP_ENV,
    }

@app.post("/predict")
def predict(data: PredictionInput) -> dict:
    try:
        features = np.array([[data.feature_1, data.feature_2]])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return {
            "prediction": prediction,
            "predicted_class_name": f"klasa_{prediction}",
            "probability_class_0": float(probabilities[0]),
            "probability_class_1": float(probabilities[1]),
            "input_data": {
                "feature_1": data.feature_1,
                "feature_2": data.feature_2,
            },
            "app_env": APP_ENV,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Blad podczas predykcji: {str(error)}")