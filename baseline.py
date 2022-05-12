import os
import pandas as pd

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')
train_file = os.path.join(data_dir, 'train.csv')
result_dir = os.path.join(current_dir, 'result')
if(not os.path.exists(result_dir)):
  os.mkdir(result_dir)

def show_performance(model_dir):
	y_pred = pd.read_csv(os.path.join(model_dir, 'pred_train.csv'), names=['model_pred']) 
	y_pred_rules = pd.read_csv(os.path.join(model_dir, 'pred_train_rules.csv'), names=['rules_pred']) 
	y = pd.concat([y_pred, y_pred_rules], axis = 1)

	fidelity = len(y.query('model_pred == rules_pred'))/len(y)
	fidelity_positives = len(y.query('model_pred == rules_pred & model_pred > 0.5'))/len(y.query('model_pred > 0.5'))
	fidelity_negatives = len(y.query('model_pred == rules_pred & model_pred < 0.5'))/len(y.query('model_pred < 0.5'))
	print("Fidelity: " + str(fidelity))
	print("Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))


print()
print("InTrees")
os.system('Rscript ./intrees_baseline.R %s %s' % (train_file, result_dir))
show_performance(model_dir = os.path.join(result_dir, 'intrees'))

print()
print("TE2Rules")
os.system('python3 te2rule.py')
show_performance(model_dir = os.path.join(result_dir, 'te2rules'))

