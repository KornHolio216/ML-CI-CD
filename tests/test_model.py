import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import (
    build_sample_records,
    build_ticket_text,
    get_accuracy,
    prepare_training_data,
    train_and_predict,
)


def test_ticket_text_is_built_from_expected_columns():
    ticket = build_sample_records()[0]
    text = build_ticket_text(ticket)

    assert "Login service unavailable" in text
    assert "Technical issue" in text


def test_prepare_training_data_returns_texts_and_priorities():
    texts, targets = prepare_training_data(build_sample_records())

    assert len(texts) == len(targets)
    assert len(texts) > 0
    assert set(targets).issubset({"Critical", "Low", "Medium", "High"})


def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be none."


def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), "Predictions length should match test labels length."


def test_predictions_value_range():
    preds, _ = train_and_predict()
    assert np.all(np.isin(preds, ["Critical", "Low", "Medium", "High"]))


def test_model_accuracy():
    accuracy = get_accuracy()
    assert accuracy >= 0.5, f"Model accuracy too low: {accuracy}"
