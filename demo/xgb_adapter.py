import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from te2rules.explainer import ModelExplainer
from sklearn import metrics

np.random.seed(123)
training_path = "data/adult/train.csv"
testing_path = "data/adult/test.csv"

data_train = pd.read_csv(training_path)
data_test = pd.read_csv(testing_path)

cols = list(data_train.columns)
feature_names = cols[:-1]
label_name = cols[-1]

data_train = data_train.to_numpy()
data_test = data_test.to_numpy()

x_train = data_train[:, :-1]
y_train = data_train[:, -1]

x_test = data_test[:, :-1]
y_test = data_test[:, -1]

model = XGBClassifier(booster="gbtree", seed=0, n_estimators=2)

# Comment / Uncomment following lines to train the model with / without feature names
model.fit(x_train, y_train)
# model.fit(pd.read_csv(training_path).drop(columns=['label_1']), pd.read_csv(training_path).to_numpy()[:, -1])

y_train_pred = model.predict(x_train)
y_train_pred_score = model.predict_proba(x_train)[:, 1]

y_test_pred = model.predict(x_test)
y_test_pred_score = model.predict_proba(x_test)[:, 1]

accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred_score)
auc = metrics.auc(fpr, tpr)
print(f"AUC: {auc}")

model_explainer = ModelExplainer(model=model, feature_names=feature_names)
rules = model_explainer.explain(
    X=x_train, y=y_train_pred, num_stages=2, min_precision=0.95
)
for i in range(len(rules)):
    print(f"Rule {i}: {rules[i]}")
