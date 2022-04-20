from sklearn.ensemble import GradientBoostingClassifier
from lib.trainer import Trainer
from lib.adapter import ScikitTreeAdapter, ScikitForestAdapter
from sklearn.tree import export_text
from lib.rule import Rule
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

def merge(candidates):
  merged_candidates = []
  merged_candidates_rule = []
  for i in range(len(candidates)):
    if(candidates[i].decision_rule not in merged_candidates_rule):
      merged_candidates.append(candidates[i])
      merged_candidates_rule.append(candidates[i].decision_rule)
    else:
      index = merged_candidates_rule.index(candidates[i].decision_rule)
      merged_candidates[index].identity = list(set(merged_candidates[index].identity).union(set(candidates[i].identity)))

  for i in range(len(merged_candidates)):
    merged_candidates[i].create_identity_map()

  return merged_candidates

def get_fidelity(rules, y_pred):
  support = []
  for r in rules:
    support = support + r.decision_support
  support = list(set(support))

  y_pred_dup = [0]*len(y_pred)
  for s in support:
    y_pred_dup[s] = 1 

  fidelity = 0
  for i in range(len(y_pred)):
    if(y_pred_dup[i] == y_pred[i]):
      fidelity = fidelity + 1
  fidelity = fidelity / len(y_pred)
  return fidelity

def sigma(x):
  e = 2.71828
  return 1/(1+e**(-x))

print()
print("Rules from trees")
print()
print(str(len(candidates)) + " candidates")
print("Merging candidates...")
candidates = merge(candidates)
print(str(len(candidates)) + " candidates")


# Crossing 0 threshold -> crossing 0.5 probability
old_candidates = candidates
candidates = []
solutions = []
for r in old_candidates:
  min_score, max_score = random_forest.get_rule_score(r.decision_rule)
  min_score = sigma(min_score*learning_rate + bias)
  max_score = sigma(max_score*learning_rate + bias)

  decision_rule_precision = 0.00
  if(min_score >= 0.5):
    decision_rule_precision = 1.00

  solution_is_possible = True 
  if(max_score < 0.5):
    solution_is_possible = False
  
  decision_value = []
  for data_index in r.decision_support:
    decision_value.append(y_pred[data_index])
  decision_rule_precision = sum(decision_value)/len(decision_value)

  if(max(decision_value) == 0):
    solution_is_possible = False
  
  if(solution_is_possible):
    if(decision_rule_precision >= 0.95):
      solutions.append(r)
    else:
      candidates.append(r)

print()
print("Running Apriori")
print()
print("Stage 0")
print(str(len(candidates)) + " candidates")
print(str(len(solutions)) + " solutions")
print("Merging candidates and solutions...")
candidates = merge(candidates)
solutions = merge(solutions)
print(str(len(candidates)) + " candidates")
print(str(len(solutions)) + " solutions")
print()
print("Fidelity " + str(get_fidelity(solutions, y_pred)))



# Join
for stage in range(1, num_trees):
  old_candidates = candidates
  candidates = []
  for i in range(len(old_candidates)):
    for j in range(i, len(old_candidates)):
      r = old_candidates[i].join(old_candidates[j])
      if(r is not None):
        min_score, max_score = random_forest.get_rule_score(r.decision_rule)
        min_score = sigma(min_score*learning_rate + bias)
        max_score = sigma(max_score*learning_rate + bias)

        decision_rule_precision = 0.00
        if(min_score >= 0.5):
          decision_rule_precision = 1.00

        solution_is_possible = True 
        if(max_score < 0.5):
          solution_is_possible = False
                   
        decision_value = []
        for data_index in r.decision_support:
          decision_value.append(y_pred[data_index])
        decision_rule_precision = sum(decision_value)/len(decision_value)
      
        if(max(decision_value) == 0):
          solution_is_possible = False
        
        if(solution_is_possible):
          if(decision_rule_precision >= 0.95):
            solutions.append(r)
          else:
            candidates.append(r)

  print()
  print("Stage " + str(stage))
  print(str(len(candidates)) + " candidates")
  print(str(len(solutions)) + " solutions")
  print("Merging candidates and solutions...")
  candidates = merge(candidates)
  solutions = merge(solutions)
  print(str(len(candidates)) + " candidates")
  print(str(len(solutions)) + " solutions")
  print()
  print("Fidelity " + str(get_fidelity(solutions, y_pred)))


print()
print("Solutions:")
for r in solutions:
  print(r.decision_rule)

