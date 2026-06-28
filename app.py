import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from model import (
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    build_ticket_text,
    load_artifact,
)

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    MongoClient = None
    PyMongoError = Exception

APP_ENV = os.getenv("APP_ENV", "development")
API_VERSION = "v1"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "fastapi_ml_api")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "predictions")
MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "1000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


MONGODB_ENABLED = parse_bool(os.getenv("MONGODB_ENABLED", "false"))
mongo_client: Any = None

class PredictionInput(BaseModel):
    ticket_subject: str = Field(..., min_length=3, max_length=200)
    ticket_description: str = Field(..., min_length=10, max_length=5000)
    ticket_type: str = Field(..., min_length=2, max_length=100)
    ticket_channel: str = Field(..., min_length=2, max_length=100)

app = FastAPI(
    title="Ticket Priority Classifier API",
    description="Proste API w FastAPI do przewidywania priorytetu ticketu supportowego.",
    version="1.0.0",
)

artifact = load_artifact(MODEL_PATH)
model = artifact["model"]
model_metadata = artifact["metadata"]
logger.info("Model loaded: %s", model_metadata.get("model_path", MODEL_PATH))


def get_model_metadata() -> dict:
    return model_metadata


def get_mongodb_collection():
    global mongo_client

    if not MONGODB_ENABLED:
        return None

    if MongoClient is None:
        logger.warning("MongoDB is enabled, but pymongo is not installed.")
        return None

    if mongo_client is None:
        mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
        )

    return mongo_client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]


def get_mongodb_status() -> str:
    if not MONGODB_ENABLED:
        return "disabled"

    collection = get_mongodb_collection()
    if collection is None:
        return "unavailable"

    try:
        collection.database.client.admin.command("ping")
        return "ok"
    except PyMongoError as error:
        logger.warning("MongoDB ping failed: %s", error)
        return "unavailable"


def save_prediction_history(prediction_response: dict) -> str | None:
    collection = get_mongodb_collection()
    if collection is None:
        return None

    document = {
        "created_at": datetime.now(timezone.utc),
        "app_env": APP_ENV,
        "api_version": API_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "input_data": prediction_response["input_data"],
        "predicted_priority": prediction_response["predicted_priority"],
        "confidence": prediction_response.get("confidence"),
    }

    try:
        result = collection.insert_one(document)
        return str(result.inserted_id)
    except PyMongoError as error:
        logger.warning("Could not save prediction history: %s", error)
        return None


def serialize_prediction(document: dict) -> dict:
    return {
        "id": str(document["_id"]),
        "created_at": document["created_at"].isoformat(),
        "app_env": document.get("app_env"),
        "api_version": document.get("api_version"),
        "model_name": document.get("model_name"),
        "model_version": document.get("model_version"),
        "input_data": document.get("input_data"),
        "predicted_priority": document.get("predicted_priority"),
        "confidence": document.get("confidence"),
    }


def build_prediction_response(data: PredictionInput) -> dict:
    ticket_data = data.model_dump()
    ticket_text = build_ticket_text(
        {
            "Ticket Subject": data.ticket_subject,
            "Ticket Description": data.ticket_description,
            "Ticket Type": data.ticket_type,
            "Ticket Channel": data.ticket_channel,
        }
    )
    predicted_priority = str(model.predict([ticket_text])[0])
    probabilities = model.predict_proba([ticket_text])[0]
    confidence = float(max(probabilities))

    response = {
        "predicted_priority": predicted_priority,
        "confidence": round(confidence, 4),
        "input_data": ticket_data,
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
        "app_env": APP_ENV,
    }

    prediction_id = save_prediction_history(response)
    if prediction_id is not None:
        response["prediction_id"] = prediction_id

    logger.info(
        "Prediction returned: priority=%s app_env=%s mongodb=%s",
        response["predicted_priority"],
        APP_ENV,
        "saved" if prediction_id else "not_saved",
    )
    return response


@app.get("/")
def read_root() -> dict:
    return {"message": "API dziala"}


@app.get("/info")
@app.get("/v1/info")
def info() -> dict:
    return {
        **get_model_metadata(),
        "api_version": API_VERSION,
        "app_env": APP_ENV,
        "mongodb_enabled": MONGODB_ENABLED,
    }


@app.get("/health")
@app.get("/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "app_env": APP_ENV,
        "api_version": API_VERSION,
        "mongodb_status": get_mongodb_status(),
    }


@app.post("/predict")
@app.post("/v1/predict")
def predict(data: PredictionInput) -> dict:
    try:
        return build_prediction_response(data)
    except Exception as error:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Blad podczas predykcji: {str(error)}")


@app.get("/predictions")
@app.get("/v1/predictions")
def predictions(limit: int = Query(20, ge=1, le=100)) -> dict:
    collection = get_mongodb_collection()
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB history is not available in this environment.",
        )

    try:
        cursor = collection.find().sort("created_at", -1).limit(limit)
        items = [serialize_prediction(document) for document in cursor]
        return {
            "items": items,
            "count": len(items),
            "app_env": APP_ENV,
            "api_version": API_VERSION,
        }
    except PyMongoError as error:
        logger.exception("Could not read prediction history.")
        raise HTTPException(status_code=503, detail=f"MongoDB read failed: {str(error)}")
