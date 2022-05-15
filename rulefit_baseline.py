import numpy as np
import pandas as pd 
from skrules import SkopeRules
import sys 
import os 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from rulefit import RuleFit

np.random.seed(123)

training_data_loc = sys.argv[1]
testing_data_loc = sys.argv[2]
result_dir = sys.argv[3]

data_train = pd.read_csv(training_data_loc) 
data_test = pd.read_csv(testing_data_loc)
cols = list(data_train.columns)

x_train = data_train.to_numpy()[:,:-1]
y_train = data_train.to_numpy()[:, -1]

x_test = data_test.to_numpy()[:,:-1]
y_test = data_test.to_numpy()[:, -1]

feature_names = cols[:-1]
label_name = cols[-1]


# Create and Train RuleFit Model
rulefit = RuleFit(
    tree_generator=RandomForestRegressor(n_estimators=10, random_state=42), 
    exp_rand_tree_size=False)
rulefit.fit(x_train, y_train, feature_names=feature_names)

rf = rulefit.tree_generator

rules = rulefit.get_rules()
rules = rules[rules['coef'] != 0] # eliminate the insignificant rules
rules = rules.sort_values('support', ascending=False)

def predict(model, x):
    y_pred_reg = model.predict(x)
    for i in range(len(y_pred_reg)):
        if(y_pred_reg[i] > 0.5):
            y_pred_reg[i] = 1.0
        else:
            y_pred_reg[i] = 0.0
    return y_pred_reg

result_dir = os.path.join(result_dir, 'rulefit')
if(not os.path.exists(result_dir)):
  os.mkdir(result_dir)

rules.to_csv(os.path.join(result_dir, 'rules.txt'))

y_pred = predict(rf, x_train)
y_pred_rules = rulefit.predict(x_train)
for i in range(len(y_pred_rules)):
    if(y_pred_rules[i] > 0.5):
        y_pred_rules[i] = 1.0
    else:
        y_pred_rules[i] = 0.0
with open(os.path.join(result_dir, 'pred_train.csv'), 'w') as f:
  for i in range(len(y_pred)):
    f.write(str(y_pred[i]) + '\n')

with open(os.path.join(result_dir, 'pred_train_rules.csv'), 'w') as f:
  for i in range(len(y_pred_rules)):
    f.write(str(y_pred_rules[i]) + '\n')

y_pred = predict(rf, x_test)
y_pred_rules = rulefit.predict(x_test)
for i in range(len(y_pred_rules)):
    if(y_pred_rules[i] > 0.5):
        y_pred_rules[i] = 1.0
    else:
        y_pred_rules[i] = 0.0
with open(os.path.join(result_dir, 'pred_test.csv'), 'w') as f:
  for i in range(len(y_pred)):
    f.write(str(y_pred[i]) + '\n')

with open(os.path.join(result_dir, 'pred_test_rules.csv'), 'w') as f:
  for i in range(len(y_pred_rules)):
    f.write(str(y_pred_rules[i]) + '\n')








