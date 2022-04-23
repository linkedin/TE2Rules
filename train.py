from sklearn.ensemble import GradientBoostingClassifier
from lib.trainer import Trainer
from lib.adapter import ScikitTreeAdapter, ScikitForestAdapter
from sklearn.tree import export_text
from lib.rule import Rule
import numpy as np

print("XGBoost Model")
trainer = Trainer(training_data_loc="data/train.csv", 
      testing_data_loc="data/test.csv",
      scikit_model=GradientBoostingClassifier(n_estimators=15))
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
  
  positives = 0
  fidelity_positives = 0
  negatives = 0
  fidelity_negatives = 0
  for i in range(len(y_pred)):
    if(y_pred[i] == 1):
      positives = positives + 1
      if(y_pred_dup[i] == y_pred[i]):
        fidelity_positives = fidelity_positives + 1
    if(y_pred[i] == 0):
      negatives = negatives + 1
      if(y_pred_dup[i] == y_pred[i]):
        fidelity_negatives = fidelity_negatives + 1

  return (fidelity_positives + fidelity_negatives) / (positives + negatives), fidelity_positives / positives, fidelity_negatives / negatives

print()
print("Rules from trees")
print()
print(str(len(candidates)) + " candidates")
print("Merging candidates...")
candidates = merge(candidates)
print(str(len(candidates)) + " candidates")

positive_points = []
for i in range(len(y_pred)):
  if(y_pred[i] == 1):
    positive_points.append(i)
assert(len(positive_points) == sum(y_pred))

# Crossing 0 threshold -> crossing 0.5 probability
old_candidates = candidates
candidates = []
solutions = []
explained_positive_points = []
for r in old_candidates:
  min_score, max_score = random_forest.get_rule_score(r.decision_rule)

  decision_rule_precision = 0.00
  if(min_score*learning_rate + bias >= 0):
    decision_rule_precision = 1.00

  solution_is_possible = True 
  if(max_score*learning_rate + bias < 0):
    solution_is_possible = False
  
  decision_value = []
  for data_index in r.decision_support:
    decision_value.append(y_pred[data_index])
  decision_rule_precision = sum(decision_value)/len(decision_value)
  
  if(max(decision_value) == 0):
    solution_is_possible = False
  
  decision_support_positive = list(set(r.decision_support).intersection(set(positive_points)))
  
  if(len(list(set(decision_support_positive).difference(set(explained_positive_points)))) == 0):
    solution_is_possible = False
  
  if(solution_is_possible):
    if(decision_rule_precision >= 0.95):
      solutions.append(r)
      explained_positive_points = list(set(explained_positive_points).union(set(decision_support_positive)))
    else:
      candidates.append(r)

print()
print("Running Apriori")

print()
print("Stage 0")
print("Pruning Candidates")
pruned_candidates = []
for i in range(len(candidates)):
  r = candidates[i]
  decision_support_positive = list(set(r.decision_support).intersection(set(positive_points)))
  
  solution_is_possible = True
  if(len(list(set(decision_support_positive).difference(set(explained_positive_points)))) == 0):
    solution_is_possible = False
  
  if(solution_is_possible):
    pruned_candidates.append(r)
candidates = pruned_candidates

print()
print(str(len(candidates)) + " candidates")
print(str(len(solutions)) + " solutions")
print("Merging candidates and solutions...")
candidates = merge(candidates)
solutions = merge(solutions)
print(str(len(candidates)) + " candidates")
print(str(len(solutions)) + " solutions")
print()
fidelity, fidelity_positives, fidelity_negatives = get_fidelity(solutions, y_pred)
print("Fidelity")
print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
print("Unexplained Positives")
print(len(list(set(positive_points).difference(set(explained_positive_points)))))


# Join
for stage in range(1, num_trees):
  old_candidates = candidates
  candidates = []
  for i in range(len(old_candidates)):
    for j in range(i, len(old_candidates)):
      r = old_candidates[i].join(old_candidates[j])
      if(r is not None):
        min_score, max_score = random_forest.get_rule_score(r.decision_rule)
        
        decision_rule_precision = 0.00
        if(min_score*learning_rate + bias >= 0):
          decision_rule_precision = 1.00

        solution_is_possible = True 
        if(max_score*learning_rate + bias < 0):
          solution_is_possible = False
                
        decision_value = []
        for data_index in r.decision_support:
          decision_value.append(y_pred[data_index])
        decision_rule_precision = sum(decision_value)/len(decision_value)
        
        if(max(decision_value) == 0):
          solution_is_possible = False
        
        decision_support_positive = list(set(r.decision_support).intersection(set(positive_points)))
        
        if(len(list(set(decision_support_positive).difference(set(explained_positive_points)))) == 0):
          solution_is_possible = False
        
        if(solution_is_possible):
          if(decision_rule_precision >= 0.95):
            solutions.append(r)
            explained_positive_points = list(set(explained_positive_points).union(set(decision_support_positive)))
          else:
            candidates.append(r)

  print()
  print("Stage " + str(stage))
  print("Pruning Candidates")
  pruned_candidates = []
  for i in range(len(candidates)):
    r = candidates[i]
    decision_support_positive = list(set(r.decision_support).intersection(set(positive_points)))
    
    solution_is_possible = True
    if(len(list(set(decision_support_positive).difference(set(explained_positive_points)))) == 0):
      solution_is_possible = False
    
    if(solution_is_possible):
      pruned_candidates.append(r)
  candidates = pruned_candidates
  
  print()
  print(str(len(candidates)) + " candidates")
  print(str(len(solutions)) + " solutions")
  print("Merging candidates and solutions...")
  candidates = merge(candidates)
  solutions = merge(solutions)
  print(str(len(candidates)) + " candidates")
  print(str(len(solutions)) + " solutions")
  print()
  fidelity, fidelity_positives, fidelity_negatives = get_fidelity(solutions, y_pred)
  print("Fidelity")
  print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
  print("Unexplained Positives")
  print(len(list(set(positive_points).difference(set(explained_positive_points)))))


print()
print("Set Cover")

rules = [] 
support = []
for r in solutions:
  rules.append(r.decision_rule)
  support.append(r.decision_support)

covered_rules = []
while(True):
  #choose biggest one
  index = 0
  while((index < len(support)) and (len(support[index]) == 0)):
    index = index + 1
  if(index == len(support)):
    break

  for i in range(len(support)):
    if(len(support[i]) > 0):
      if(len(support[i]) > len(support[index])):
        index = i
      else:
        if(len(support[i]) == len(support[index])):
          if(len(rules[i]) < len(rules[index])):
            index = i

  covered_rules.append(index)

  #remove support
  for i in range(len(support)):
    if(index != i):
      support[i] = list(set(support[i]).difference(set(support[index])))
  support[index] = []

old_solutions = solutions
solutions = []
for i in covered_rules:
  solutions.append(old_solutions[i])

print()
print("Solutions:")
for r in solutions:
  print(r.decision_rule)

print()
print(str(num_trees) + " trees")
print(str(len(solutions)) + " solutions")
fidelity, fidelity_positives, fidelity_negatives = get_fidelity(solutions, y_pred)
print("Fidelity")
print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
