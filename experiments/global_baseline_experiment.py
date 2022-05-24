import os
import pandas as pd
from sklearn import metrics
from time import time
import warnings
warnings.filterwarnings('ignore')

# set working directory - please set to your project directory
import sys
project_directory = '/Users/elachen/Desktop/code/TE2Rule'
sys.path.append(project_directory)

# experiment parameters
dataset_name_params = ['breast', 'compas', 'bank', 'adult']
method_name_params = ['intrees', 'skoperules', 'rulefit']
n_estimators_params = [10, 20, 50]
max_depth_params = [3, 5]
experiment_desc = "test"
need_data_prep = False

# prep datasets if needed
if need_data_prep:
    os.system('python3 data_prep/data_prep_adult.py')
    os.system('python3 data_prep/data_prep_bank.py')
    os.system('python3 data_prep/data_prep_compas.py')
    os.system('python3 data_prep/data_prep_breast.py')


def show_rule_performance(model_dir):
    print("Performance of rules on train data")
    y_pred = pd.read_csv(os.path.join(model_dir, 'pred_train.csv'), names=['model_pred']) 
    y_pred_rules = pd.read_csv(os.path.join(model_dir, 'pred_train_rules.csv'), names=['rules_pred']) 
    y = pd.concat([y_pred, y_pred_rules], axis = 1)

    fidelity = len(y.query('model_pred == rules_pred'))/len(y)
    try:
        fidelity_positives = len(y.query('model_pred == rules_pred & model_pred > 0.5'))/len(y.query('model_pred > 0.5'))
    except:
        fidelity_positives = 'NA'
    fidelity_negatives = len(y.query('model_pred == rules_pred & model_pred < 0.5'))/len(y.query('model_pred < 0.5'))
    print("Fidelity: " + str(fidelity))
    print("Positive: " + str(fidelity_positives) + ", Negative: " + str(fidelity_negatives))
    return (fidelity, fidelity_positives, fidelity_negatives)

def show_model_performance(model_dir, test_file, use_test=True):
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
    return (auc, acc)


def run_baseline_script(method_name, train_file, test_file, result_dir, n_estimators, max_depth):
    if method_name == 'intrees':
        os.system('Rscript ../intrees_baseline.R %s %s %s %d' % (train_file, test_file, result_dir, n_estimators))
    elif method_name == 'skoperules':
        os.system('python3 ../skope_baseline.py %s %s %s %d %d' % (train_file, test_file, result_dir, n_estimators, max_depth))
    elif method_name == 'rulefit':
        os.system('python3 ../rulefit_baseline.py %s %s %s %d %d' % (train_file, test_file, result_dir, n_estimators, max_depth))
    else:
        print("error!")
    
def experiment_iter(dataset_name, method_name, n_estimators, max_depth):    
    # config path
    
    data_dir = os.path.join(project_directory, "data/{}".format(dataset_name))
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    result_dir = os.path.join(project_directory, 'experiments/global_result/baselines/{}'.format(dataset_name))
    if(not os.path.exists(result_dir)):
        os.makedirs(result_dir)

    # run experiment
    print("started experiment for: {}, {}, {}, {}".format(dataset_name, method_name, n_estimators, max_depth))
    dataset_name_list.append(dataset_name)
    method_list.append(method_name)
    n_estimators_list.append(n_estimators)
    max_depth_list.append(max_depth)
    start_time = time()
    run_baseline_script(method_name, train_file, test_file, result_dir, n_estimators, max_depth)
    model_dir = os.path.join(result_dir, method_name)
    time_list.append(time() - start_time)
    
    (fidelity, fidelity_positives, fidelity_negatives) = show_rule_performance(model_dir)
    fidelity_list.append(fidelity)
    fidelity_pos_list.append(fidelity_positives)
    fidelity_neg_list.append(fidelity_negatives)
    (auc, acc) = show_model_performance(model_dir, test_file)
    model_accuracy_list.append(acc)
    model_auc_list.append(auc)
    
    rules = []
    with open(result_dir+'/{}/rules.txt'.format(method_name), 'r') as f:
        rules = f.readlines()
    num_rules_list.append(max(0, len(rules) - 1))
    
def experiment(dataset_name_params, method_name_params, n_estimators_params, max_depth_params):
    for method_name in method_name_params:
        if method_name == 'intrees':  # intrees runs RF and doesn't use max_depth
            max_depth = 'NA'
            for dataset_name in dataset_name_params:
                for n_estimators in n_estimators_params:
                    experiment_iter(dataset_name, method_name, n_estimators, max_depth)
        else:
            for dataset_name in dataset_name_params:
                for n_estimators in n_estimators_params:
                    for max_depth in max_depth_params:
                        experiment_iter(dataset_name, method_name, n_estimators, max_depth)


# experiment pipeline
dataset_name_list, method_list, n_estimators_list, max_depth_list = [], [], [], []
model_accuracy_list, model_auc_list, num_rules_list, time_list = [], [], [], []
fidelity_list, fidelity_pos_list, fidelity_neg_list = [], [], []

experiment(dataset_name_params, method_name_params, n_estimators_params, max_depth_params)

experiment_result_df = pd.DataFrame({
    'method': method_list,
    'dataset_name': dataset_name_list,
    'n_estimators': n_estimators_list,
    'max_depth': max_depth_list,
    'model_accuracy': model_accuracy_list,
    'model_auc': model_auc_list,
    'num_rules': num_rules_list,
    'time': time_list,
    'fidelity': fidelity_list,
    'fidelity_pos': fidelity_pos_list,
    'fidelity_neg': fidelity_neg_list,
})

experiment_result_df.to_csv(os.path.join(
    project_directory, 
    "experiments/global_result/baselines/experiment_summary_{}.csv".format(experiment_desc)), index = False)