from datetime import datetime, timezone
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

import app as api_app


client = TestClient(api_app.app)

VALID_TICKET = {
    "ticket_subject": "Login service unavailable",
    "ticket_description": "Many users cannot log in after the latest deployment.",
    "ticket_type": "Technical issue",
    "ticket_channel": "Email",
}
VALID_PRIORITIES = ["Critical", "High", "Low", "Medium"]


class FakeInsertResult:
    inserted_id = "abc123"


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, field_name, direction):
        self.documents = sorted(
            self.documents,
            key=lambda item: item[field_name],
            reverse=direction == -1,
        )
        return self

    def limit(self, limit):
        self.documents = self.documents[:limit]
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeCollection:
    def __init__(self):
        self.inserted_documents = []
        self.documents = [
            {
                "_id": "history-1",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "app_env": "test",
                "api_version": "v1",
                "model_name": "ticket-priority-classifier",
                "model_version": "ticket-priority-v1",
                "input_data": VALID_TICKET,
                "predicted_priority": "Critical",
                "confidence": 0.87,
            }
        ]

    def insert_one(self, document):
        self.inserted_documents.append(document)
        return FakeInsertResult()

    def find(self):
        return FakeCursor(self.documents)


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_app, "MONGODB_ENABLED", False)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api_version"] == "v1"
    assert body["mongodb_status"] == "disabled"
    assert "app_env" in body


def test_v1_info_endpoint(monkeypatch):
    monkeypatch.setattr(api_app, "MONGODB_ENABLED", False)

    response = client.get("/v1/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "ticket-priority-classifier"
    assert body["model_version"] == "ticket-priority-v1"
    assert body["model_type"] == "TfidfVectorizer + LogisticRegression"
    assert "classes" in body
    assert body["input_schema"]["ticket_subject"] == "string"


def test_v1_predict_endpoint(monkeypatch):
    monkeypatch.setattr(api_app, "MONGODB_ENABLED", False)

    response = client.post("/v1/predict", json=VALID_TICKET)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_priority"] in VALID_PRIORITIES
    assert 0 <= body["confidence"] <= 1
    assert body["input_data"] == VALID_TICKET
    assert body["model_version"] == "ticket-priority-v1"
    assert body["api_version"] == "v1"


def test_predict_rejects_invalid_payload():
    response = client.post(
        "/v1/predict",
        json={
            "ticket_subject": "A",
            "ticket_description": "short",
            "ticket_type": "Technical issue",
            "ticket_channel": "Email",
        },
    )

    assert response.status_code == 422


def test_prediction_history_disabled(monkeypatch):
    monkeypatch.setattr(api_app, "MONGODB_ENABLED", False)

    response = client.get("/v1/predictions")

    assert response.status_code == 503


def test_predict_saves_history_when_mongodb_is_available(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(api_app, "MONGODB_ENABLED", True)
    monkeypatch.setattr(api_app, "get_mongodb_collection", lambda: collection)

    response = client.post("/v1/predict", json=VALID_TICKET)

    assert response.status_code == 200
    assert response.json()["prediction_id"] == "abc123"
    assert len(collection.inserted_documents) == 1
    assert collection.inserted_documents[0]["input_data"] == VALID_TICKET
    assert collection.inserted_documents[0]["predicted_priority"] in VALID_PRIORITIES
    assert "confidence" in collection.inserted_documents[0]


def test_predictions_history_endpoint(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(api_app, "MONGODB_ENABLED", True)
    monkeypatch.setattr(api_app, "get_mongodb_collection", lambda: collection)

    response = client.get("/v1/predictions")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == "history-1"
    assert body["items"][0]["predicted_priority"] == "Critical"
