"""
This file contains unit tests for
te2rules.adapter.XgboostXGBClassifierAdapter.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from te2rules.adapter import XgboostXGBClassifierAdapter

np.random.seed(42)

x_train = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 0.0],
]
y_train = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
feature_names = ["feature0", "feature1", "feature2", "feature3"]

learning_rate = 0.1
num_positive = sum(y_train)
num_negative = len(y_train) - sum(y_train)

num_trees = 3
model = XGBClassifier(n_estimators=num_trees, max_depth=2, learning_rate=learning_rate)

model.fit(np.asarray(x_train), y_train)
model_scores = model.predict_proba(np.asarray(x_train))[:, 1]

adapted_model = XgboostXGBClassifierAdapter(model, feature_names).random_forest

rules = adapted_model.get_rules(x_train)
adapted_model_scores = adapted_model.get_scores()


def test_num() -> None:
    """
    Unit test for number of trees in the adapted tree ensemble
    """
    assert adapted_model.get_num_trees() == num_trees


def test_weights_and_bias() -> None:
    """
    Unit test for weights of trees in the adapted tree ensemble and bias
    """
    weight = adapted_model.weight
    bias = adapted_model.bias
    assert weight == 1.0
    assert bias == 0.0


def test_activation() -> None:
    """
    Unit test for activation function in the adapted tree ensemble
    """
    assert adapted_model.activation == "sigmoid"


def test_scores() -> None:
    """
    Unit test for scores by the adapted tree ensemble
    """
    assert len(model_scores) == len(adapted_model_scores)
    for i in range(len(model_scores)):
        assert abs(model_scores[i] - adapted_model_scores[i]) < 10 ** (-6)


def test_named_input_columns() -> None:
    # "f0" is a column name here, even though it is not the first column.
    frame = pd.DataFrame(x_train, columns=["close_15m", "f0", "bid", "ask"])
    named_model = XGBClassifier(
        n_estimators=num_trees, max_depth=2, learning_rate=learning_rate
    )
    named_model.fit(frame, y_train)

    forest = XgboostXGBClassifierAdapter(named_model, feature_names).random_forest
    forest.get_rules(x_train)
    np.testing.assert_allclose(
        forest.get_scores(), named_model.predict_proba(frame)[:, 1], atol=1e-6
    )
