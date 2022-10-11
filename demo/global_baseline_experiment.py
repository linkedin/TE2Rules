"""
This file has script to reproduce global explanation results in the paper
for baselines: InTrees, SkopeRules, RuleFit.
"""
import os
from time import time

import pandas as pd
from sklearn import metrics

project_directory = os.getcwd()
result_directory = os.path.join(project_directory, "results/global_result/baselines/")
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# grid search parameters
dataset_name_params = ["breast", "compas", "bank", "adult"]
method_name_params = ["intrees", "skoperules", "rulefit"]
n_estimators_params = [10, 20, 50]
max_depth_params = [3, 5]
experiment_desc = "all_global_baseline_experiments"

# prep datasets
os.system("python3 data_prep/data_prep_adult.py")
os.system("python3 data_prep/data_prep_bank.py")
os.system("python3 data_prep/data_prep_compas.py")
os.system("python3 data_prep/data_prep_breast.py")


def show_rule_performance(model_dir):
    print("Performance of rules on train data")
    y_pred = pd.read_csv(
        os.path.join(model_dir, "pred_train.csv"), names=["model_pred"]
    )
    y_pred_rules = pd.read_csv(
        os.path.join(model_dir, "pred_train_rules.csv"), names=["rules_pred"]
    )
    y = pd.concat([y_pred, y_pred_rules], axis=1)

    fidelity = len(y.query("model_pred == rules_pred")) / len(y)
    try:
        fidelity_positives = len(
            y.query("model_pred == rules_pred & model_pred > 0.5")
        ) / len(y.query("model_pred > 0.5"))
    except:
        fidelity_positives = "NA"
    fidelity_negatives = len(
        y.query("model_pred == rules_pred & model_pred < 0.5")
    ) / len(y.query("model_pred < 0.5"))
    print("Fidelity: " + str(fidelity))
    print(
        "Positive: "
        + str(fidelity_positives)
        + ", Negative: "
        + str(fidelity_negatives)
    )
    return (fidelity, fidelity_positives, fidelity_negatives)


def show_model_performance(model_dir, test_file, use_test=True):
    if use_test is True:
        data_name = "test data"
        y_pred = pd.read_csv(
            os.path.join(model_dir, "pred_test.csv"), names=["model_pred"]
        )["model_pred"].to_numpy()
        y_pred_score = pd.read_csv(
            os.path.join(model_dir, "pred_test_score.csv"), names=["model_pred"]
        )["model_pred"].to_numpy()
        y = pd.read_csv(test_file)["label_1"].to_numpy()
    else:
        data_name = "train data"
        y_pred = pd.read_csv(
            os.path.join(model_dir, "pred_train.csv"), names=["model_pred"]
        )["model_pred"].to_numpy()
        y_pred_score = pd.read_csv(
            os.path.join(model_dir, "pred_train_score.csv"), names=["model_pred"]
        )["model_pred"].to_numpy()
        y = pd.read_csv(train_file)["label_1"].to_numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_score)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(y, y_pred)
    print("Model performance on " + str(data_name))
    print("AUC: " + str(auc))
    print("Acc: " + str(acc))
    return (auc, acc)


def run_baseline_script(
    method_name, train_file, test_file, result_dir, n_estimators, max_depth
):
    if method_name == "intrees":
        os.system(
            "Rscript baseline/intrees_baseline.R %s %s %s %d %d"
            % (train_file, test_file, result_dir, n_estimators, max_depth)
        )
    elif method_name == "skoperules":
        os.system(
            "python3 baseline/skope_baseline.py %s %s %s %d %d"
            % (train_file, test_file, result_dir, n_estimators, max_depth)
        )
    elif method_name == "rulefit":
        os.system(
            "python3 baseline/rulefit_baseline.py %s %s %s %d %d"
            % (train_file, test_file, result_dir, n_estimators, max_depth)
        )
    else:
        print("error!")


def experiment_iter(dataset_name, method_name, n_estimators, max_depth):
    # configure paths
    training_path = os.path.join(
        project_directory, "data/{}/train.csv".format(dataset_name)
    )
    testing_path = os.path.join(
        project_directory, "data/{}/test.csv".format(dataset_name)
    )
    output_path = os.path.join(result_directory, dataset_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # run experiment
    start_time = time()
    run_baseline_script(
        method_name, training_path, testing_path, output_path, n_estimators, max_depth
    )
    model_dir = os.path.join(output_path, method_name)
    total_time = time() - start_time

    fidelity = show_rule_performance(model_dir)
    (auc, acc) = show_model_performance(model_dir, testing_path)

    rules = []
    with open(os.path.join(output_path, "{}/rules.txt".format(method_name)), "r") as f:
        rules = f.readlines()

    return acc, auc, len(rules), total_time, fidelity


def experiment(
    method_name_params, dataset_name_params, n_estimators_params, max_depth_params
):
    method_col = []
    dataset_name_col = []
    n_estimators_col = []
    max_depth_col = []
    model_accuracy_col = []
    model_auc_col = []
    num_rules_col = []
    time_col = []
    fidelity_col = []

    for method_name in method_name_params:
        for dataset_name in dataset_name_params:
            for n_estimators in n_estimators_params:
                for max_depth in max_depth_params:
                    print(
                        "experiment: {} method, {} data, {} trees, {} depth".format(
                            method_name, dataset_name, n_estimators, max_depth
                        )
                    )
                    accuracy, auc, num_rules, time, fidelity = experiment_iter(
                        dataset_name, method_name, n_estimators, max_depth
                    )
                    print()
                    method_col.append(method_name)
                    dataset_name_col.append(dataset_name)
                    n_estimators_col.append(n_estimators)
                    max_depth_col.append(max_depth)

                    model_accuracy_col.append(accuracy)
                    model_auc_col.append(auc)
                    num_rules_col.append(num_rules)
                    time_col.append(time)
                    fidelity_col.append(fidelity)

    experiment_result_df = pd.DataFrame(
        {
            "method": method_col,
            "dataset_name": dataset_name_col,
            "n_estimators": n_estimators_col,
            "max_depth": max_depth_col,
            "model_accuracy": model_accuracy_col,
            "model_auc": model_auc_col,
            "num_rules": num_rules_col,
            "time": time_col,
            "fidelity": fidelity_col,
        }
    )

    experiment_result_df.to_csv(
        os.path.join(
            result_directory, "experiment_summary_{}.csv".format(experiment_desc)
        ),
        index=False,
    )


experiment(
    method_name_params, dataset_name_params, n_estimators_params, max_depth_params
)
