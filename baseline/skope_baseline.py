"""
Python script to create a tree ensemble model (with given ntrees, max_depth) and explain
it using skoperules. The script reads training and testing data and then writes the output
of the tree ensemble model (scores, class_predictions) and rules (extracted rules,
class_predictions) on both training and testing data in the results directory.

Usage: python3 skope_baseline.py training_data testing_data result_dir ntrees max_depth
"""
import numpy as np
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
import os

np.random.seed(123)

# get arguments
training_data_loc = sys.argv[1]
testing_data_loc = sys.argv[2]
result_dir = sys.argv[3]
ntrees = int(sys.argv[4])
max_depth = int(sys.argv[5])

# prepare results directory to save results
result_dir = os.path.join(result_dir, 'skoperules')
if(not os.path.exists(result_dir)):
  os.mkdir(result_dir)

# Loads data from training data location and
# testing data location and returns feature names
# label name, x_train, y_train, x_test, y_test
def load_data(training_data_loc, testing_data_loc):
  data_train = pd.read_csv(training_data_loc)
  data_test = pd.read_csv(testing_data_loc)
  cols = list(data_train.columns)

  x_train = data_train.to_numpy()[:,:-1]
  y_train = data_train.to_numpy()[:, -1]

  x_test = data_test.to_numpy()[:,:-1]
  y_test = data_test.to_numpy()[:, -1]

  feature_names = cols[:-1]
  label_name = cols[-1]
  return [feature_names, label_name], [x_train, y_train], [x_test, y_test]

# load data
[feature_names, label_name], [x_train, y_train], [x_test, y_test] = load_data(training_data_loc, testing_data_loc)

# explain tree ensemble using skope rules
skope_rules = SkopeRules(max_depth_duplication=3,
                 n_estimators=int(ntrees),
                 max_depth=int(max_depth),
                 precision_min=0.95,
                 recall_min=0.01,
                 feature_names=feature_names,
                 random_state=0)

# Predicts class predictions from tree ensemble used by skoperules.
# Skoperules uses two different underlying tree ensemble models.
# The first is a bag of N decision tree classifiers and the second is a
# bag of N decision tree regressors as the underlying tree ensemble. We consider
# only the N decision tree classifiers as the underlying Random Forest classifier.
def predict(model, x):
    y_pred = [0]*len(x)
    y_pred = np.array(y_pred)
    for m in model.estimators_[:len(model.estimators_)//2]:
        y_pred = y_pred + m.predict_proba(x)[:,1]
    y_pred = y_pred/(len(model.estimators_)//2)
    for i in range(len(y_pred)):
        if(y_pred[i] > 0.5):
            y_pred[i] = 1.0
        else:
            y_pred[i] = 0.0
    return y_pred

# Predicts class prediction scores from tree ensemble used by skoperules.
# Skoperules uses two different underlying tree ensemble models.
# The first is a bag of N decision tree classifiers and the second is a
# bag of N decision tree regressors as the underlying tree ensemble. We consider
# only the N decision tree classifiers as the underlying Random Forest classifier.
def predict_proba(model, x):
    y_pred_score = [0]*len(x)
    y_pred_score = np.array(y_pred_score)
    for m in model.estimators_[:len(model.estimators_)//2]:
        y_pred_score = y_pred_score + m.predict_proba(x)[:,1]
    y_pred_score = y_pred_score/(len(model.estimators_)//2)
    return y_pred_score

# Predicts class predictions from
# rules extracted by skoperules
def rule_apply(model, x, feature_names):
    df = pd.DataFrame(data=x, columns=feature_names)
    rules = model.rules_
    coverage = []
    for r in rules:
      support = df.query(str(r[0])).index.tolist()
      coverage = list(set(coverage).union(set(support)))

    y_pred_rules = [0.0]*len(df)
    for i in coverage:
      y_pred_rules[i] = 1.0

    return y_pred_rules

# Saves skoperule's tree ensemble model's predictions (class, scores)
# and extracted rules prediction (class)
def save_model_predictions(model, x, feature_names, description):
  y_pred_score = predict_proba(model, x)
  y_pred = predict(model, x)
  y_pred_rules = rule_apply(model, x, feature_names)

  with open(os.path.join(result_dir, 'pred_'+description+'_score.csv'), 'w') as f:
    for i in range(len(y_pred_score)):
      f.write(str(y_pred_score[i]) + '\n')

  with open(os.path.join(result_dir, 'pred_'+description+'.csv'), 'w') as f:
    for i in range(len(y_pred)):
      f.write(str(y_pred[i]) + '\n')

  with open(os.path.join(result_dir, 'pred_'+description+'_rules.csv'), 'w') as f:
    for i in range(len(y_pred_rules)):
      f.write(str(y_pred_rules[i]) + '\n')
  return

skope_rules.fit(x_train, y_train)
rules = skope_rules.rules_
with open(os.path.join(result_dir, 'rules.txt'), 'w') as f:
  for r in rules:
    f.write(str(r) + '\n')

save_model_predictions(skope_rules, x_train, feature_names, description='train')
save_model_predictions(skope_rules, x_test, feature_names, description='test')
