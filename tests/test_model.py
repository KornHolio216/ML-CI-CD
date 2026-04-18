import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be none."


def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), "Predictions length should match test labels length."


def test_predictions_value_range():
    preds, _ = train_and_predict()
    assert np.all(np.isin(preds, [0, 1])), "Predictions should only contain class 0 or 1."


def test_model_accuracy():
    accuracy = get_accuracy()
    assert accuracy >= 0.7, f"Model accuracy too low: {accuracy}"