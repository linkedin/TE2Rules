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

solutions = RuleBuilder(random_forest = random_forest, 
  X = trainer.x_train, y = y_pred).solution_rules

if(not os.path.exists('result/te2rules')):
  os.mkdir('result/te2rules')
with open('result/te2rules/te2rules.txt', 'w') as f:
  for r in solutions:
    f.write(str(r) + '\n')

# Evaluate rules
data_train = pd.read_csv("data/train.csv") 
data_train['y_pred'] = y_pred 
data_train.to_csv("data/train_pred.csv")

data_train = pd.read_csv("data/train_pred.csv") 
positive_points = data_train.query('y_pred > 0.5').index.tolist()
negative_points = data_train.query('y_pred < 0.5').index.tolist()

solutions = []
with open('result/te2rules/te2rules.txt', 'r') as f:
  solutions = f.readlines()
solutions = [r.strip() for r in solutions]

print()
print("Solutions:")
coverage = []
positive_coverage = []
negative_coverage = list(range(len(y_pred)))
for i in range(len(solutions)):
  r = solutions[i]
  print("Rule " + str(i + 1) + ": " + r)
  support = data_train.query(r).index.tolist()
  positive_support = data_train.query(r).query('y_pred > 0.5').index.tolist()
  negative_support = data_train.query(str("not (" + r + ")")).query('y_pred < 0.5').index.tolist()
  print("Precision: " + str(len(positive_support)/len(support)))
  print("Recall: " + str(len(positive_support)/len(positive_points)))
  print("Coverage: " + str(len(support)/len(y_pred)))
  coverage = list(set(coverage).union(set(support)))
  positive_coverage = list(set(positive_coverage).union(set(positive_support)))
  negative_coverage = list(set(negative_coverage).intersection(set(negative_support)))
  print("Cumulative Precision: " + str(len(positive_coverage)/len(coverage)))
  print("Cumulative Recall: " + str(len(positive_coverage)/len(positive_points)))
  print("Cumulative Coverage: " + str(len(coverage)/len(y_pred)))
  fidelity_positives = len(positive_coverage)/(len(positive_points))
  fidelity_negatives = len(negative_coverage)/(len(negative_points))
  fidelity = (len(positive_coverage) + len(negative_coverage))/(len(positive_points) + len(negative_points))
  print("Cumulative Fidelity: ")
  print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
  print()


print()
print(str(num_trees) + " trees")
print(str(len(solutions)) + " solutions")
print("Fidelity")
print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))


