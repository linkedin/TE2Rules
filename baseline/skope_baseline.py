import numpy as np
import pandas as pd 
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
import os 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

np.random.seed(123)

training_data_loc = sys.argv[1]
testing_data_loc = sys.argv[2]
result_dir = sys.argv[3]
ntrees = sys.argv[4]
max_depth = sys.argv[5]

data_train = pd.read_csv(training_data_loc) 
data_test = pd.read_csv(testing_data_loc)
cols = list(data_train.columns)

x_train = data_train.to_numpy()[:,:-1]
y_train = data_train.to_numpy()[:, -1]

x_test = data_test.to_numpy()[:,:-1]
y_test = data_test.to_numpy()[:, -1]

feature_names = cols[:-1]
label_name = cols[-1]

clf = SkopeRules(max_depth_duplication=3,
                 n_estimators=int(ntrees),
                 max_depth=int(max_depth),
                 precision_min=0.95,
                 recall_min=0.01,
                 feature_names=feature_names,
                 random_state=0)

clf.fit(x_train, y_train)
rules = clf.rules_

def predict(model, x):
    y_pred_clf = [0]*len(x)
    y_pred_clf = np.array(y_pred_clf)
    for m in clf.estimators_[:len(clf.estimators_)//2]:
        y_pred_clf = y_pred_clf + m.predict_proba(x)[:,1]
    y_pred_clf = y_pred_clf/(len(clf.estimators_)//2)
    for i in range(len(y_pred_clf)):
        if(y_pred_clf[i] > 0.5):
            y_pred_clf[i] = 1.0
        else:
            y_pred_clf[i] = 0.0

    y_pred_reg = [0]*len(x)
    y_pred_reg = np.array(y_pred_reg)
    for m in clf.estimators_[len(clf.estimators_)//2:]:
        y_pred_reg = y_pred_reg + m.predict(x)
    y_pred_reg = y_pred_reg/(len(clf.estimators_)//2)
    for i in range(len(y_pred_reg)):
        if(y_pred_reg[i] > 0.5):
            y_pred_reg[i] = 1.0
        else:
            y_pred_reg[i] = 0.0
    
    assert(list(y_pred_clf) == list(y_pred_reg))
    return y_pred_clf

def predict_proba(model, x):
    y_pred_clf = [0]*len(x)
    y_pred_clf = np.array(y_pred_clf)
    for m in clf.estimators_[:len(clf.estimators_)//2]:
        y_pred_clf = y_pred_clf + m.predict_proba(x)[:,1]
    y_pred_clf = y_pred_clf/(len(clf.estimators_)//2)


    y_pred_reg = [0]*len(x)
    y_pred_reg = np.array(y_pred_reg)
    for m in clf.estimators_[len(clf.estimators_)//2:]:
        y_pred_reg = y_pred_reg + m.predict(x)
    y_pred_reg = y_pred_reg/(len(clf.estimators_)//2)


    assert(list(y_pred_clf) == list(y_pred_reg))
    return y_pred_clf

def rule_apply(rules, df):
    coverage = []
    for r in rules:
      support = df.query(str(r[0])).index.tolist()
      coverage = list(set(coverage).union(set(support)))

    y_rules = [0.0]*len(df)
    for i in coverage:
      y_rules[i] = 1.0

    return y_rules

result_dir = os.path.join(result_dir, 'skoperules')
if(not os.path.exists(result_dir)):
  os.mkdir(result_dir)

with open(os.path.join(result_dir, 'rules.txt'), 'w') as f:
  for r in rules:
    f.write(str(r) + '\n')

y_pred_score = predict_proba(clf, x_train)
y_pred = predict(clf, x_train)
y_pred_rules = rule_apply(rules, data_train)

with open(os.path.join(result_dir, 'pred_train_score.csv'), 'w') as f:
  for i in range(len(y_pred_score)):
    f.write(str(y_pred_score[i]) + '\n')

with open(os.path.join(result_dir, 'pred_train.csv'), 'w') as f:
  for i in range(len(y_pred)):
    f.write(str(y_pred[i]) + '\n')

with open(os.path.join(result_dir, 'pred_train_rules.csv'), 'w') as f:
  for i in range(len(y_pred_rules)):
    f.write(str(y_pred_rules[i]) + '\n')

y_pred_score = predict_proba(clf, x_test)
y_pred = predict(clf, x_test)
y_pred_rules = rule_apply(rules, data_test)

with open(os.path.join(result_dir, 'pred_test_score.csv'), 'w') as f:
  for i in range(len(y_pred_score)):
    f.write(str(y_pred_score[i]) + '\n')

with open(os.path.join(result_dir, 'pred_test.csv'), 'w') as f:
  for i in range(len(y_pred)):
    f.write(str(y_pred[i]) + '\n')

with open(os.path.join(result_dir, 'pred_test_rules.csv'), 'w') as f:
  for i in range(len(y_pred_rules)):
    f.write(str(y_pred_rules[i]) + '\n')

