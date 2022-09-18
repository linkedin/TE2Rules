import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from te2rules.explainer import ModelExplainer

np.random.seed(42)

x_train = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
]
y_train = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
feature_names = ["f0", "f1", "f2", "f3"]

model = GradientBoostingClassifier(n_estimators=3, max_depth=2)
model.fit(x_train, y_train)

data_for_explanation = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
]
model_pred = model.predict(data_for_explanation)

model_explainer = ModelExplainer(model=model, feature_names=feature_names)
rules = model_explainer.explain(
    X=data_for_explanation, y=model_pred, min_precision=1.00
)


def test_rules():
    expected_rules = ["f3 > 0.5", "f0 <= 0.5 & f1 <= 0.5 & f2 <= 0.5 & f3 <= 0.5"]
    assert set(rules) == set(expected_rules)
