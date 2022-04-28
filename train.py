from sklearn.ensemble import GradientBoostingClassifier
from lib.trainer import Trainer
from lib.adapter import ScikitTreeAdapter, ScikitForestAdapter
from sklearn.tree import export_text, plot_tree
from lib.rule import Rule
from lib.rule_builder import RuleBuilder
import numpy as np

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

# Calculating bias
positive = sum(trainer.y_train)
negative = len(trainer.y_train) - sum(trainer.y_train)
log_odds = np.log(positive/negative)
bias = log_odds

# Weights of Trees
learning_rate = trainer.model.get_params()['learning_rate']

# Get candidate rules
random_forest = ScikitForestAdapter(trainer.model, trainer.feature_names).random_forest
num_trees = random_forest.get_num_trees()
print(str(num_trees) + " trees")
candidates = random_forest.get_rules(data=trainer.x_train, feature_names=trainer.feature_names)

explainer = RuleBuilder(random_forest = random_forest, candidate_rules = candidates, labels = y_pred, 
  tree_weight = learning_rate, tree_bias = log_odds)

solutions = explainer.solution_rules
positive_points = explainer.positives

print()
print("Solutions:")
coverage = []
positive_coverage = []
for i in range(len(solutions)):
  r = solutions[i]
  print("Rule " + str(i + 1) + ": " + str(r.decision_rule))
  positive_support = list(set(positive_points).intersection(set(r.decision_support)))
  print("Precision: " + str(len(positive_support)/len(r.decision_support)))
  print("Recall: " + str(len(positive_support)/len(positive_points)))
  print("Coverage: " + str(len(r.decision_support)/len(y_pred)))
  coverage = list(set(coverage).union(r.decision_support))
  positive_coverage = list(set(positive_coverage).union(positive_support))
  print("Cumulative Precision: " + str(len(positive_coverage)/len(coverage)))
  print("Cumulative Recall: " + str(len(positive_coverage)/len(positive_points)))
  print("Cumulative Coverage: " + str(len(coverage)/len(y_pred)))
  fidelity, fidelity_positives, fidelity_negatives = explainer.get_fidelity(use_top = i + 1)
  print("Cumulative Fidelity: ")
  print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
  print()


print()
print(str(num_trees) + " trees")
print(str(len(solutions)) + " solutions")
fidelity, fidelity_positives, fidelity_negatives = explainer.get_fidelity()
print("Fidelity")
print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))


