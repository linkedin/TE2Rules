import os
import pandas as pd
from sklearn import metrics

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')
result_dir = os.path.join(current_dir, 'result')
if(not os.path.exists(result_dir)):
  os.mkdir(result_dir)


def show_rule_performance(model_dir):
	print("Performance of rules on train data")
	y_pred = pd.read_csv(os.path.join(model_dir, 'pred_train.csv'), names=['model_pred']) 
	y_pred_rules = pd.read_csv(os.path.join(model_dir, 'pred_train_rules.csv'), names=['rules_pred']) 
	y = pd.concat([y_pred, y_pred_rules], axis = 1)

	fidelity = len(y.query('model_pred == rules_pred'))/len(y)
	fidelity_positives = len(y.query('model_pred == rules_pred & model_pred > 0.5'))/len(y.query('model_pred > 0.5'))
	fidelity_negatives = len(y.query('model_pred == rules_pred & model_pred < 0.5'))/len(y.query('model_pred < 0.5'))
	print("Fidelity: " + str(fidelity))
	print("Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))

def show_model_performance(model_dir, use_test=True):
	if(use_test is True):
		data_name = "test data"
		y_pred = pd.read_csv(os.path.join(model_dir, 'pred_test.csv'), names=['model_pred'])['model_pred'].to_numpy() 
		y = pd.read_csv(test_file)['label_1'].to_numpy()
	else:
		data_name = "train data"
		y_pred = pd.read_csv(os.path.join(model_dir, 'pred_train.csv'), names=['model_pred'])['model_pred'].to_numpy() 
		y = pd.read_csv(train_file)['label_1'].to_numpy()
	fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
	auc = metrics.auc(fpr, tpr)
	acc = metrics.accuracy_score(y, y_pred)
	print("Model performance on " + str(data_name))
	print("AUC: " + str(auc))
	print("Acc: " + str(acc))



print()
print("InTrees")
os.system('Rscript ./intrees_baseline.R %s %s %s' % (train_file, test_file, result_dir))
model_dir = os.path.join(result_dir, 'intrees')
show_model_performance(model_dir)
show_rule_performance(model_dir)

print()
print("TE2Rules")
os.system('python3 te2rule.py %s %s %s' % (train_file, test_file, result_dir))
model_dir = os.path.join(result_dir, 'te2rules')
show_model_performance(model_dir)
show_rule_performance(model_dir)

print()
print("Skope Rules")
os.system('python3 skope_baseline.py %s %s %s' % (train_file, test_file, result_dir))
model_dir = os.path.join(result_dir, 'skoperules')
show_model_performance(model_dir)
show_rule_performance(model_dir)

print()
print("RuleFit")
os.system('python3 rulefit_baseline.py %s %s %s' % (train_file, test_file, result_dir))
model_dir = os.path.join(result_dir, 'rulefit')
show_model_performance(model_dir)
show_rule_performance(model_dir)

