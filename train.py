from sklearn.ensemble import GradientBoostingClassifier
from lib.trainer import Trainer
from lib.adapter import ScikitTreeAdapter, ScikitForestAdapter
from sklearn.tree import export_text, plot_tree
from lib.rule import Rule
import numpy as np

def pipeline(training_data_path="data/train.csv", testing_data_path="data/test.csv", num_estimators=10, fidelity_threshold=0.95):
  print("XGBoost Model")
  trainer = Trainer(training_data_loc=training_data_path, 
        testing_data_loc=testing_data_path,
        scikit_model=GradientBoostingClassifier(n_estimators=num_estimators))
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

  # Threshold on score is 0, sigma(0) = 0.5
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
      if(decision_rule_precision >= fidelity_threshold):
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
            if(decision_rule_precision >= fidelity_threshold):
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
  positive_support = []
  for r in solutions:
    rules.append(r.decision_rule)
    positive_support.append(list(set(positive_points).intersection(set(r.decision_support))))

  covered_rules = []
  while(True):
    #choose biggest one
    index = 0
    while((index < len(positive_support)) and (len(positive_support[index]) == 0)):
      index = index + 1
    if(index == len(positive_support)):
      break

    for i in range(len(positive_support)):
      if(len(positive_support[i]) > 0):
        if(len(positive_support[i]) > len(positive_support[index])):
          index = i
        else:
          if(len(positive_support[i]) == len(positive_support[index])):
            if(len(rules[i]) < len(rules[index])):
              index = i

    covered_rules.append(index)

    #remove support
    for i in range(len(positive_support)):
      if(index != i):
        positive_support[i] = list(set(positive_support[i]).difference(set(positive_support[index])))
    positive_support[index] = []

  old_solutions = solutions
  solutions = []
  for i in covered_rules:
    solutions.append(old_solutions[i])

  rule_precision_list, rule_recall_list, rule_coverage_list = [], [], []

  print()
  print("Solutions:")
  coverage = []
  positive_coverage = []
  for i in range(len(solutions)):
    r = solutions[i]
    print("Rule " + str(i + 1) + ": " + str(r.decision_rule))
    positive_support = list(set(positive_points).intersection(set(r.decision_support)))
    # rule_precision
    rule_precision = len(positive_support)/len(r.decision_support)
    rule_precision_list.append(rule_precision)
    print("Precision: " + str(rule_precision))
    # rule_recall
    rule_recall = len(positive_support)/len(positive_points)
    rule_recall_list.append(rule_recall)
    print("Recall: " + str(rule_recall))
    # rule_coverage
    rule_coverage = len(r.decision_support)/len(y_pred)
    rule_coverage_list.append(rule_coverage)
    print("Coverage: " + str(rule_coverage))
    coverage = list(set(coverage).union(r.decision_support))
    positive_coverage = list(set(positive_coverage).union(positive_support))
    print("Cumulative Precision: " + str(len(positive_coverage)/len(coverage)))
    print("Cumulative Recall: " + str(len(positive_coverage)/len(positive_points)))
    print("Cumulative Coverage: " + str(len(coverage)/len(y_pred)))
    fidelity, fidelity_positives, fidelity_negatives = get_fidelity(solutions[0 : i+1], y_pred)
    print("Cumulative Fidelity: ")
    print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
    print()


  print()
  print(str(num_trees) + " trees")
  print(str(len(solutions)) + " solutions")
  fidelity, fidelity_positives, fidelity_negatives = get_fidelity(solutions, y_pred)
  print("Fidelity")
  print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
  
  return solutions, trainer, rule_precision_list, rule_recall_list, rule_coverage_list
