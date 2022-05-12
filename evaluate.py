import pandas as pd 

# Evaluate rules
y_pred = pd.read_csv("result/te2rules/pred_train.csv", header=None).to_numpy() 

data_train = pd.read_csv("data/train.csv") 
data_train['y_pred'] = y_pred 
data_train.to_csv("data/train_pred.csv")

data_train = pd.read_csv("data/train_pred.csv") 
positive_points = data_train.query('y_pred > 0.5').index.tolist()
negative_points = data_train.query('y_pred < 0.5').index.tolist()

rules = []
with open('result/te2rules/te2rules.txt', 'r') as f:
  rules = f.readlines()
rules = [r.strip() for r in rules]

print()
print("Rules:")
coverage = []
positive_coverage = []
negative_coverage = list(range(len(y_pred)))
for i in range(len(rules)):
  r = rules[i]
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
print(str(len(rules)) + " rules")
print("Fidelity")
print("Total: " + str(fidelity) + ", Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
