# set working directory - please set to your project directory
import sys
project_directory = '/Users/elachen/Desktop/code/TE2Rule'
sys.path.append(project_directory)
import utils.experiment_utils
import sys
import os
import pandas as pd
from time import time

# grid search parameters
# Note that some of the experiments with bank and adult datasets take a long time (> 8 hours) to finish.
# Please run with smaller n_estimators, max_depth and num_stages if you would like the algorithm to converge fastly.
dataset_name_list = ["breast", "compas", "bank", "adult"]
n_estimators_list = [10, 20, 50]
max_depth_list = [3, 5]
num_stages_list = [1, 2, 3]
decision_rule_precision = 0.95
experiment_desc = "all_global_te2rules_experiments"
need_data_prep = False

# prep datasets if needed
if need_data_prep:
    os.system('python3 data_prep/data_prep_adult.py')
    os.system('python3 data_prep/data_prep_bank.py')
    os.system('python3 data_prep/data_prep_compas.py')
    os.system('python3 data_prep/data_prep_breast.py')


def experiment(dataset_name_list, n_estimators_list, max_depth_list, num_stages_list):
    for dataset_name in dataset_name_list:
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for num_stages in (num_stages_list+[n_estimators]):
                    print("experiment: {}, {} trees, {} depth, {} stages".format(dataset_name, n_estimators, max_depth, num_stages))
                    fidelities = experiment_iter(dataset_name, n_estimators, max_depth, num_stages)
                    # early stop if fidelity reaches 1
                    if fidelities == (1, 1, 1):
                        break

def experiment_iter(dataset_name, n_estimators, max_depth, num_stages):    
    # configure paths
    training_path = os.path.join(project_directory, "data/{}/train.csv".format(dataset_name))
    testing_path = os.path.join(project_directory, "data/{}/test.csv".format(dataset_name))
    output_path = os.path.join(project_directory, "experiments/global_result/te2rules/{}".format(dataset_name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Train tree ensemble model
    trainer, random_forest, y_pred, accuracy, auc = utils.experiment_utils.train_tree_ensemble(
        training_data_loc=training_path,
        testing_data_loc=testing_path,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # Explain the training data using rules
    start_time = time()
    rules, fidelities = utils.experiment_utils.explain_with_rules(
        x_train=trainer.x_train,
        y_train_pred=y_pred,
        random_forest=random_forest,
        num_stages=num_stages,
        decision_rule_precision=decision_rule_precision
    )
    total_time = time() - start_time
    print('algorithm_runtime: ', total_time)
    
    # write to summary
    with open(output_path + "/summary_t{}_d{}_s{}.txt".format(n_estimators, max_depth, num_stages), 'w') as f:
        # record results of all stages in the txt file
        f.write("dataset_name: {}\n".format(dataset_name))
        f.write("model_accuracy: {}\n".format(accuracy))
        f.write("model_auc: {}\n".format(auc))
        f.write("\nn_estimators: {}\n".format(n_estimators))
        f.write("max_depth: {}\n".format(max_depth))
        f.write("\nnum_stages: {}\n".format(num_stages))
        f.write("\n------\n")
        f.write("num_rules: {}\n".format(len(rules)))
        f.write("rule_fidelity (total, positive, negative): {}\n".format(fidelities))
        f.write("total_rule_search_time (seconds): {}\n".format(total_time))

    # write to rules
    with open(output_path + "/rules_t{}_d{}_s{}.txt".format(n_estimators, max_depth, num_stages), 'w') as f:
        for r in rules:
            f.write(' & '.join(utils.experiment_utils.dedup_rule_predicates(r.decision_rule)) + '\n')
    
    # results to be saved in the df
    _dataset_name_list.append(dataset_name)
    _n_estimators_list.append(n_estimators)
    _max_depth_list.append(max_depth),
    _num_stages_list.append(num_stages)
    _model_accuracy_list.append(accuracy)
    _model_auc_list.append(auc)
    _num_rules_list.append(len(rules))
    _fidelities_list.append(fidelities)
    _time_list.append(total_time)
    
    return fidelities



# experiment pipeline
_dataset_name_list, _n_estimators_list, _max_depth_list, _num_stages_list = [], [], [], []
_model_accuracy_list, _model_auc_list, _num_rules_list, _fidelities_list, _time_list = [], [], [], [], []

experiment(dataset_name_list, n_estimators_list, max_depth_list, num_stages_list)

experiment_result_df = pd.DataFrame({
    'dataset_name': _dataset_name_list,
    'n_estimators': _n_estimators_list,
    'max_depth': _max_depth_list,
    'num_stages': _num_stages_list,
    'model_accuracy': _model_accuracy_list,
    'model_auc': _model_auc_list,
    'num_rules': _num_rules_list,
    'fidelities': _fidelities_list,
    'time': _time_list
})

experiment_result_df.to_csv(os.path.join(project_directory, "experiments/global_result/te2rules/experiment_summary_{}.csv".format(experiment_desc), index = False))