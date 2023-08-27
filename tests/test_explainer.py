"""
This file contains unit tests for te2rules.explainer.ModelExplainer.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from te2rules.explainer import ModelExplainer

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
feature_names = ["f0", "f1", "f2", "f3"]

model = GradientBoostingClassifier(n_estimators=3, max_depth=2)
model.fit(x_train, y_train)

data_for_explanation = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
]
model_pred = model.predict(data_for_explanation)

model_explainer = ModelExplainer(model=model, feature_names=feature_names)
rules = model_explainer.explain(
    X=data_for_explanation, y=model_pred, min_precision=1.00
)
data_for_rule_inference = data_for_explanation[:5]


def test_rules() -> None:
    """
    Unit test for te2rules.explainer.ModelExplainer.explain()
    """
    expected_rules = ["f3 > 0.5", "f0 <= 0.5 & f1 <= 0.5 & f2 <= 0.5 & f3 <= 0.5"]
    assert set(rules) == set(expected_rules)


def test_predict() -> None:
    """
    Unit test for te2rules.explainer.ModelExplainer.predict()
    """
    expected_y_rule_inference = [1, 1, 0, 1, 0]
    assert (
        model_explainer.predict(X=data_for_rule_inference) == expected_y_rule_inference
    )


def test_fidelity() -> None:
    """
    Unit test for te2rules.explainer.ModelExplainer.get_fidelity()
    """
    expected_fidelity = (1.0, 1.0, 1.0)
    fidelity = model_explainer.get_fidelity()
    fidelity = tuple(round(f, 2) for f in fidelity)
    assert fidelity == expected_fidelity


def test_fidelity_custom_data() -> None:
    """
    Unit test for te2rules.explainer.ModelExplainer.get_fidelity()
    """
    expected_fidelity = (0.8, 1.0, 0.67)
    fidelity = model_explainer.get_fidelity(X=x_train[10:15], y=y_train[10:15])
    fidelity = tuple(round(f, 2) for f in fidelity)
    assert fidelity == expected_fidelity

    expected_fidelity = (1.0, 1.0, 0.0)
    fidelity = model_explainer.get_fidelity(X=x_train[0:5], y=y_train[0:5])
    fidelity = tuple(round(f, 2) for f in fidelity)
    assert fidelity == expected_fidelity
