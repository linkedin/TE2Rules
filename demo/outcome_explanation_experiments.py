"""
This file has script to reproduce local outcome explanation results in the paper
comparing TE2Rules and Anchors.
"""
import os
from time import time

import matplotlib
import numpy as np
import pandas as pd
from anchor import anchor_tabular, utils
from sklearn.ensemble import GradientBoostingClassifier

from te2rules.explainer import ModelExplainer
from trainer import Trainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_directory = os.getcwd()

# experiment parameters
n_estimators = 20
max_depth = 3
num_stages = n_estimators

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


# set for parameters for each dataset
# the training data only contains about 30 positive instances, so we cannot set it to be too large
for (dataset_name, sample_size) in [
    ("breast", 30),
    ("compas", 100),
    ("bank", 100),
    ("adult", 100),
]:
    # configure paths
    training_path = os.path.join(
        project_directory, "data/{}/train.csv".format(dataset_name)
    )
    testing_path = os.path.join(
        project_directory, "data/{}/test.csv".format(dataset_name)
    )
    output_path = os.path.join(
        project_directory, "results/local_result/{}".format(dataset_name)
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # write to summary_stats
    f = open(os.path.join(output_path, "summary_stats.txt"), "w")
    f = open(os.path.join(output_path, "summary_stats.txt"), "a")
    f.write("dataset_name: {}\n".format(dataset_name))
    f.write("sample_size: {}\n".format(sample_size))
    f.close()

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
        min_precision=0.95,
    )
    total_time = time() - start_time
    print("algorithm_runtime: ", total_time)

    # 0. Construct TE2Rule and anchor explainers for a trained GBM model
    # Obtain TE2Rule Global rules
    te2_global_rule_list = np.array([s.decision_rule for s in rules])

    df = pd.DataFrame(trainer.x_train, columns=trainer.feature_names)
    df["y_pred"] = trainer.model.predict(trainer.x_train)

    # construct anchor explainer
    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=["negative", "positive"],
        feature_names=trainer.feature_names,
        train_data=trainer.x_train,
        categorical_names={},
    )

    # 1. sample n positive instances from the dataset
    cannot_find_anchor_count = 0
    sample_df = df[df.y_pred == True].sample(sample_size, random_state=0)
    explained_te2_instance_indices = []
    te2_rule_list, anchor_rule_list = [], []
    te2_time_list, anchor_time_list = [], []
    te2_precision_list, anchor_precision_list = [], []
    te2_recall_list, anchor_recall_list = [], []
    te2_coverage_list, anchor_coverage_list = [], []
    anchor_local_precision_list = []
    anchor_local_coverage_list = []

    for i in sample_df.index:
        # 2. for each instance, obtain anchor & TE2Rule (rule, computation time)
        # anchor
        start_time = time()
        anchor_rule = anchor_explainer.explain_instance(
            trainer.x_train[i], trainer.model.predict, threshold=0.95
        )

        try:  # sometimes (e.g. for the bank dataset) we are not able obtain valid result from anchors
            anchor_rule_support = len(
                df.query(" & ".join(anchor_rule.names()), engine="python")
            )
            anchor_rule_positive_support = len(
                df.query(" & ".join(anchor_rule.names()), engine="python").query(
                    "y_pred > 0"
                )
            )
            anchor_precision_list.append(
                anchor_rule_positive_support / anchor_rule_support
            )
            anchor_recall_list.append(anchor_rule_positive_support / sum(df.y_pred))
            anchor_coverage_list.append(anchor_rule_support / len(df))
            anchor_local_precision_list.append(anchor_rule.precision())
            anchor_local_coverage_list.append(anchor_rule.coverage())
            anchor_rule_list.append(anchor_rule.names())
            anchor_time_list.append(time() - start_time)
        except:
            cannot_find_anchor_count += 1
            continue

        # te2rule
        valid_instance = False
        start_time = time()
        for j in range(len(te2_global_rule_list)):
            te2_rule = te2_global_rule_list[j]
            if (
                len(sample_df.loc[[i]].query(" & ".join(te2_rule), engine="python")) > 0
            ):  # if rule predicates match instance
                te2_time_list.append(time() - start_time)
                te2_rule_list.append(te2_rule)
                valid_instance = True
                break
        # if TE2Rule does not find a rule for the instance, go to the next instance
        if not valid_instance:
            print("No rule found for this instance ", i)
            continue
        explained_te2_instance_indices.append(i)

        # for te2, query the dataset based on the rule
        te2_rule_support = len(df.query(" & ".join(te2_rule), engine="python"))
        te2_rule_positive_support = len(
            df.query(" & ".join(te2_rule), engine="python").query("y_pred > 0")
        )
        te2_precision_list.append(te2_rule_positive_support / te2_rule_support)
        te2_recall_list.append(te2_rule_positive_support / sum(df.y_pred))
        te2_coverage_list.append(te2_rule_support / len(df))

        print("finished rule-search for instance ", i)

    # store result in respective folder
    te2rules_result = pd.DataFrame(
        {
            "instance_id": sample_df.index,
            "rule": te2_rule_list,
            "rule_length": [len(r) for r in te2_rule_list],
            "computation_time": te2_time_list,
            "precision": te2_precision_list,
            "recall": te2_recall_list,
            "coverage": te2_coverage_list,
        }
    )

    anchors_result = pd.DataFrame(
        {
            "instance_id": sample_df.index,
            "rule": anchor_rule_list,
            "rule_length": [len(r) for r in anchor_rule_list],
            "computation_time": anchor_time_list,
            "precision": anchor_precision_list,
            "recall": anchor_recall_list,
            "coverage": anchor_coverage_list,
        }
    )

    te2rules_result.to_csv(output_path + "/te2rules_result.csv", index=False)
    anchors_result.to_csv(output_path + "/anchors_result.csv", index=False)

    # computation time, precision, recall, coverage, rule length
    # average runtime per rule
    print(
        "Average runtime per instance for te2rules: ",
        te2rules_result.computation_time.mean(),
    )
    print(
        "Average runtime per instance for anchors: ",
        anchors_result.computation_time.mean(),
    )

    # write summary result to txt
    f = open(output_path + "/summary_stats.txt", "w")
    f = open(output_path + "/summary_stats.txt", "a")
    f.write(
        "\nAverage runtime per instance for te2rules: {:.4f}\n".format(
            te2rules_result.computation_time.mean()
        )
    )
    f.write(
        "Average runtime per instance for anchors:  {:.4f}\n".format(
            anchors_result.computation_time.mean()
        )
    )
    f.close()

    # precision
    plt.style.use("seaborn-deep")
    plt.rcParams["font.size"] = "23"
    plt.rcParams.update({"figure.autolayout": True})

    plt.figure()
    plt.hist(
        [te2rules_result.precision, anchors_result.precision],
        label=["te2rules", "anchors"],
    )
    plt.legend(loc="upper left", framealpha=0.6)
    plt.xlabel("rule precision")
    plt.ylabel("number of instances")
    # plt.xlim([0, 1.1])
    plt.savefig(
        output_path + "/{}_outcome_precision.png".format(dataset_name), dpi=1000
    )
    # plt.show()
    plt.close()

    # recall
    plt.figure()
    plt.hist(
        [te2rules_result.recall, anchors_result.recall], label=["te2rules", "anchors"]
    )
    plt.legend(loc="upper left", framealpha=0.6)
    plt.xlabel("rule recall")
    plt.ylabel("number of instances")
    # plt.xlim([0, 1.1])
    plt.savefig(output_path + "/{}_outcome_recall.png".format(dataset_name), dpi=1000)
    # plt.show()
    plt.close()

    # coverage
    plt.figure()
    plt.hist(
        [te2rules_result.coverage, anchors_result.coverage],
        label=["te2rules", "anchors"],
    )
    plt.legend(loc="upper left", framealpha=0.6)
    plt.xlabel("rule coverage")
    plt.ylabel("number of instances")
    # plt.xlim([0, 1.1])
    plt.savefig(output_path + "/{}_outcome_coverage.png".format(dataset_name), dpi=1000)
    # plt.show()
    plt.close()

    # rule_length
    plt.figure()
    plt.hist(
        [te2rules_result.rule_length, anchors_result.rule_length],
        label=["te2rules", "anchors"],
    )
    plt.legend(loc="upper right", framealpha=0.6)
    plt.xlabel("rule rule_length")
    plt.ylabel("number of instances")
    plt.savefig(
        output_path + "/{}_outcome_rule_length.png".format(dataset_name), dpi=1000
    )
    # plt.show()
    plt.close()
