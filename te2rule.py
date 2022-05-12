from sklearn.ensemble import GradientBoostingClassifier
from lib.trainer import Trainer
from lib.adapter import ScikitTreeAdapter, ScikitForestAdapter
from sklearn.tree import export_text, plot_tree
from lib.rule import Rule
from lib.rule_builder import RuleBuilder
import pandas as pd 
import os 

# Train model to explain
print("XGBoost Model")
trainer = Trainer(training_data_loc="data/train.csv", 
      testing_data_loc="data/test.csv",
      scikit_model=GradientBoostingClassifier(n_estimators=10))
print("Accuracy")
accuracy, auc = trainer.evaluate_model()
print(accuracy) 
print("AUC")
print(auc)

y_pred = trainer.model.predict(trainer.x_train)
if(sum(y_pred) == 0):
  print("Model doen't learn any positive")

# Explain using rules 
random_forest = ScikitForestAdapter(trainer.model, trainer.feature_names).random_forest
num_trees = random_forest.get_num_trees()
print(str(num_trees) + " trees")

rule_builder = RuleBuilder(random_forest = random_forest) 
rules = rule_builder.explain(X = trainer.x_train, y = y_pred) 
#rules = rule_builder.explain()

data_train = pd.read_csv("data/train.csv") 
y_pred_rules = rule_builder.apply(data_train)

if(not os.path.exists('result/te2rules')):
  os.mkdir('result/te2rules')

with open('result/te2rules/rules.txt', 'w') as f:
  for r in rules:
    f.write(str(r) + '\n')

with open('result/te2rules/pred_train.csv', 'w') as f:
  for i in range(len(y_pred)):
    f.write(str(y_pred[i]) + '\n')

with open('result/te2rules/pred_train_rules.csv', 'w') as f:
  for i in range(len(y_pred_rules)):
    f.write(str(y_pred_rules[i]) + '\n')



