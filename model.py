import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

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

def train_and_predict():
    X, y = build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return preds, y_test

def get_accuracy():
    preds, y_test = train_and_predict()
    return accuracy_score(y_test, preds)