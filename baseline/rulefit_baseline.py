"""
Python script to create a tree ensemble model (with given ntrees, max_depth) and explain
it using rulefit. The script reads training and testing data and then writes the output
of the tree ensemble model (scores, class_predictions) and rules (extracted rules,
class_predictions) on both training and testing data in the results directory.

Usage: python3 rulefit_baseline.py training_data testing_data result_dir ntrees max_depth
"""

import numpy as np
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import os
from sklearn.ensemble import RandomForestRegressor
from rulefit import RuleFit

np.random.seed(123)

# get arguments
training_data_loc = sys.argv[1]
testing_data_loc = sys.argv[2]
result_dir = sys.argv[3]
ntrees = int(sys.argv[4])
max_depth = int(sys.argv[5])

# prepare results directory to save results
result_dir = os.path.join(result_dir, 'rulefit')
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

# explain tree ensemble using rulefit
rulefit = RuleFit(
    tree_generator=RandomForestRegressor(n_estimators=ntrees, max_depth=max_depth, random_state=0),
    exp_rand_tree_size=False,random_state=0)
rulefit.fit(x_train, y_train, feature_names=feature_names)

# Predicts class predictions from
# tree ensemble used by rule fit
def predict(model, x):
    model = model.tree_generator
    y_pred = model.predict(x)
    for i in range(len(y_pred)):
        if(y_pred[i] > 0.5):
            y_pred[i] = 1.0
        else:
            y_pred[i] = 0.0
    return y_pred

# Predicts prediction scores from
# tree ensemble used by rule fit
def predict_proba(model, x):
    model = model.tree_generator
    y_pred_score = model.predict(x)
    return y_pred_score

# Predicts class predictions from
# rules extracted by rule fit
def rule_apply(model, x):
  y_pred_rules = model.predict(x)
  for i in range(len(y_pred_rules)):
      if(y_pred_rules[i] > 0.5):
          y_pred_rules[i] = 1.0
      else:
          y_pred_rules[i] = 0.0
  return y_pred_rules

# Saves rulefit's tree ensemble model's predictions (class, scores)
# and extracted rules prediction (class)
def save_model_predictions(model, x, description):
  y_pred_score = predict_proba(model, x)
  y_pred = predict(model, x)
  y_pred_rules = rule_apply(model, x)

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

rules = rulefit.get_rules()
rules = rules[rules['coef'] != 0] # eliminate the insignificant rules
rules = rules.sort_values('support', ascending=False)
rules.to_csv(os.path.join(result_dir, 'rules.txt'))

save_model_predictions(rulefit, x_train, description='train')
save_model_predictions(rulefit, x_test, description='test')
