import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42
MODEL_NAME = "ticket-priority-classifier"
MODEL_VERSION = "ticket-priority-v1"
MODEL_PATH = Path("artifacts/ticket_priority_model.joblib")
DEFAULT_DATA_PATH = Path("data/raw/customer_support_tickets.csv")
TEXT_COLUMNS = ["Ticket Subject", "Ticket Description", "Ticket Type", "Ticket Channel"]
TARGET_COLUMN = "Ticket Priority"


def build_sample_records() -> list[dict[str, str]]:
    return [
        {
            "Ticket Subject": "Login service unavailable",
            "Ticket Description": "Many users cannot log in after the latest deployment.",
            "Ticket Type": "Technical issue",
            "Ticket Channel": "Email",
            "Ticket Priority": "High",
        },
        {
            "Ticket Subject": "Production checkout is down",
            "Ticket Description": "Customers cannot pay for orders and revenue is affected.",
            "Ticket Type": "Technical issue",
            "Ticket Channel": "Phone",
            "Ticket Priority": "High",
        },
        {
            "Ticket Subject": "Security alert for account access",
            "Ticket Description": "Several suspicious login attempts were detected for admin users.",
            "Ticket Type": "Technical issue",
            "Ticket Channel": "Email",
            "Ticket Priority": "Critical",
        },
        {
            "Ticket Subject": "Data loss after deployment",
            "Ticket Description": "A major customer reports missing production records after release.",
            "Ticket Type": "Technical issue",
            "Ticket Channel": "Phone",
            "Ticket Priority": "Critical",
        },
        {
            "Ticket Subject": "Payment gateway outage",
            "Ticket Description": "All card payments are failing for multiple regions.",
            "Ticket Type": "Technical issue",
            "Ticket Channel": "Phone",
            "Ticket Priority": "High",
        },
        {
            "Ticket Subject": "Invoice total does not match",
            "Ticket Description": "The customer sees a different tax value on the latest invoice.",
            "Ticket Type": "Billing inquiry",
            "Ticket Channel": "Email",
            "Ticket Priority": "Medium",
        },
        {
            "Ticket Subject": "Password reset email delayed",
            "Ticket Description": "A single user reports that the password reset email arrived late.",
            "Ticket Type": "Technical issue",
            "Ticket Channel": "Chat",
            "Ticket Priority": "Medium",
        },
        {
            "Ticket Subject": "Change billing contact",
            "Ticket Description": "Please update the billing contact before the next invoice.",
            "Ticket Type": "Billing inquiry",
            "Ticket Channel": "Email",
            "Ticket Priority": "Medium",
        },
        {
            "Ticket Subject": "Question about product settings",
            "Ticket Description": "Customer asks where to change notification preferences.",
            "Ticket Type": "Product inquiry",
            "Ticket Channel": "Chat",
            "Ticket Priority": "Low",
        },
        {
            "Ticket Subject": "Feature request for dashboard export",
            "Ticket Description": "Customer would like an export button in the analytics dashboard.",
            "Ticket Type": "Feature request",
            "Ticket Channel": "Email",
            "Ticket Priority": "Low",
        },
        {
            "Ticket Subject": "Typo in help article",
            "Ticket Description": "There is a small typo in the documentation page.",
            "Ticket Type": "Product inquiry",
            "Ticket Channel": "Email",
            "Ticket Priority": "Low",
        },
    ]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip().lower(): (value or "").strip() for key, value in row.items()}


def get_value(row: dict[str, str], column_name: str) -> str:
    return row.get(column_name.strip().lower(), "")


def build_ticket_text(row: dict[str, str]) -> str:
    normalized = normalize_row(row)
    parts = [get_value(normalized, column_name) for column_name in TEXT_COLUMNS]
    return " ".join(part for part in parts if part)


def load_records_from_csv(data_path: Path) -> list[dict[str, str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with data_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def find_dataset_csv(raw_data_dir: Path = Path("data/raw")) -> Path:
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH

    csv_files = sorted(raw_data_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "Put the Kaggle CSV file in data/raw/customer_support_tickets.csv "
        "or pass --data with the exact CSV path."
    )


def prepare_training_data(records: list[dict[str, str]]) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    targets: list[str] = []

    for row in records:
        normalized = normalize_row(row)
        target = get_value(normalized, TARGET_COLUMN)
        text = build_ticket_text(row)

        if text and target:
            texts.append(text)
            targets.append(target)

    if not texts:
        raise ValueError("No valid rows found for training.")

    return texts, np.array(targets)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_pipeline(records: list[dict[str, str]]) -> Pipeline:
    texts, targets = prepare_training_data(records)
    pipeline = build_pipeline()
    pipeline.fit(texts, targets)
    return pipeline


def train_from_csv(data_path: Path) -> Pipeline:
    records = load_records_from_csv(data_path)
    return train_pipeline(records)


def train_sample_pipeline() -> Pipeline:
    return train_pipeline(build_sample_records())


def evaluate_pipeline(records: list[dict[str, str]]) -> float:
    texts, targets = prepare_training_data(records)
    class_counts = {label: int(np.sum(targets == label)) for label in set(targets)}
    can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        targets,
        test_size=0.33,
        random_state=RANDOM_STATE,
        stratify=targets if can_stratify else None,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return accuracy_score(y_test, predictions)


def build_artifact(pipeline: Pipeline, data_source: str) -> dict:
    classifier = pipeline.named_steps["classifier"]
    return {
        "model": pipeline,
        "metadata": {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_type": "TfidfVectorizer + LogisticRegression",
            "classes": [str(label) for label in classifier.classes_],
            "input_schema": {
                "ticket_subject": "string",
                "ticket_description": "string",
                "ticket_type": "string",
                "ticket_channel": "string",
            },
            "model_path": MODEL_PATH.as_posix(),
            "data_source": data_source,
        },
    }


def save_artifact(artifact: dict, output_path: Path = MODEL_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        pipeline = train_sample_pipeline()
        return build_artifact(pipeline, data_source="sample-dev-data")

    return joblib.load(model_path)


def train_and_save(data_path: Path, output_path: Path = MODEL_PATH) -> Path:
    pipeline = train_from_csv(data_path)
    artifact = build_artifact(pipeline, data_source=str(data_path))
    return save_artifact(artifact, output_path)


def train_and_predict() -> tuple[np.ndarray, np.ndarray]:
    texts, targets = prepare_training_data(build_sample_records())
    pipeline = build_pipeline()
    pipeline.fit(texts, targets)
    predictions = pipeline.predict(texts)
    return predictions, targets


def get_accuracy() -> float:
    return evaluate_pipeline(build_sample_records())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ticket priority classifier.")
    parser.add_argument("--data", type=Path, default=None, help="Path to Kaggle CSV file.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Output model path.")
    args = parser.parse_args()

    data_path = args.data if args.data else find_dataset_csv()
    saved_path = train_and_save(data_path, args.output)
    print(f"Model saved to {saved_path}")


if __name__ == "__main__":
    main()
