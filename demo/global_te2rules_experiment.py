"""
This file has script to reproduce global explanation results in the paper
for TE2Rules.
"""
import os
from time import time

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from te2rules.explainer import ModelExplainer
from trainer import Trainer

project_directory = os.getcwd()
result_directory = os.path.join(project_directory, "results/global_result/te2rules/")
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# grid search parameters
dataset_name_params = ["breast", "compas", "bank", "adult"]
n_estimators_params = [10, 20, 50]
max_depth_params = [3, 5]
num_stages_params = [1, 2, 3]
min_precision = 0.95
experiment_desc = "all_global_te2rules_experiments"

# prep datasets
os.system("python3 data_prep/data_prep_adult.py")
os.system("python3 data_prep/data_prep_bank.py")
os.system("python3 data_prep/data_prep_compas.py")
os.system("python3 data_prep/data_prep_breast.py")


# Train model to explain
def train_tree_ensemble(training_data_loc, testing_data_loc, n_estimators, max_depth):
    print("XGBoost Model")
    trainer = Trainer(
        training_data_loc=training_data_loc,
        testing_data_loc=testing_data_loc,
        scikit_model=GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=1
        ),
    )
    accuracy, auc = trainer.evaluate_model()
    print("Accuracy: ", accuracy)
    print("AUC: ", auc)
    y_pred = trainer.model.predict(trainer.x_train)
    if sum(y_pred) == 0:
        print("Model doen't learn any positive")
    return trainer, y_pred, accuracy, auc


# Explain using rules
def explain_with_rules(
    model, feature_names, x_train, y_train_pred, num_stages, min_precision
):
    # build rules
    model_explainer = ModelExplainer(model=model, feature_names=feature_names)
    rules = model_explainer.explain(
        X=x_train,
        y=y_train_pred,
        num_stages=num_stages,
        min_precision=min_precision,
    )
    fidelities = model_explainer.get_fidelity()
    return rules, fidelities


def experiment(
    dataset_name_params, n_estimators_params, max_depth_params, num_stages_params
):
    dataset_col = []
    n_estimators_col = []
    max_depth_col = []
    num_stages_col = []

    model_accuracy_col = []
    model_auc_col = []

    fidelities_col = []
    time_col = []

    num_rules_col = []
    for dataset_name in dataset_name_params:
        for n_estimators in n_estimators_params:
            for max_depth in max_depth_params:
                for num_stages in num_stages_params + [n_estimators]:
                    print(
                        "experiment: {} data, {} trees, {} depth, {} stages".format(
                            dataset_name, n_estimators, max_depth, num_stages
                        )
                    )
                    accuracy, auc, num_rules, fidelities, total_time = experiment_iter(
                        dataset_name, n_estimators, max_depth, num_stages
                    )
                    print()
                    dataset_col.append(dataset_name)
                    n_estimators_col.append(n_estimators)
                    max_depth_col.append(max_depth)
                    num_stages_col.append(num_stages)

                    model_accuracy_col.append(accuracy)
                    model_auc_col.append(auc)

                    num_rules_col.append(num_rules)
                    fidelities_col.append(fidelities)
                    time_col.append(total_time)

                    # early stop if fidelity reaches 1
                    if fidelities == (1, 1, 1):
                        break

    experiment_result_df = pd.DataFrame(
        {
            "dataset_name": dataset_col,
            "n_estimators": n_estimators_col,
            "max_depth": max_depth_col,
            "num_stages": num_stages_col,
            "model_accuracy": model_accuracy_col,
            "model_auc": model_auc_col,
            "num_rules": num_rules_col,
            "fidelities": fidelities_col,
            "time": time_col,
        }
    )

    experiment_result_df.to_csv(
        os.path.join(
            result_directory, "experiment_summary_{}.csv".format(experiment_desc)
        ),
        index=False,
    )


def experiment_iter(dataset_name, n_estimators, max_depth, num_stages):
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

    # Train tree ensemble model
    trainer, y_train_pred, accuracy, auc = train_tree_ensemble(
        training_data_loc=training_path,
        testing_data_loc=testing_path,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Explain the training data using rules
    start_time = time()
    rules, fidelities = explain_with_rules(
        model=trainer.model,
        feature_names=trainer.feature_names,
        x_train=trainer.x_train,
        y_train_pred=y_train_pred,
        num_stages=num_stages,
        min_precision=min_precision,
    )
    total_time = time() - start_time
    print("algorithm_runtime: ", total_time)

    # write to summary
    with open(
        os.path.join(
            output_path,
            "summary_t{}_d{}_s{}.txt".format(n_estimators, max_depth, num_stages),
        ),
        "w",
    ) as f:
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
    with open(
        os.path.join(
            output_path,
            "rules_t{}_d{}_s{}.txt".format(n_estimators, max_depth, num_stages),
        ),
        "w",
    ) as f:
        for r in rules:
            f.write(str(r) + "\n")

    return accuracy, auc, len(rules), fidelities, total_time


experiment(
    dataset_name_params, n_estimators_params, max_depth_params, num_stages_params
)
