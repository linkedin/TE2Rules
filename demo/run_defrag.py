import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from trainer import Trainer
from time import time
import numpy as np 
from defragTrees import DefragModel

def generate_data():
    os.system("python3 data_prep/data_prep_adult.py")
    os.system("python3 data_prep/data_prep_bank.py")
    os.system("python3 data_prep/data_prep_compas.py")
    os.system("python3 data_prep/data_prep_breast.py")

def train_xgb_model(dataset_name, n_estimators, max_depth):
    training_data_loc = "./data/{}/train.csv".format(dataset_name)
    testing_data_loc = "./data/{}/test.csv".format(dataset_name)
        
    model_bundle = Trainer(
        training_data_loc=training_data_loc,
        testing_data_loc=testing_data_loc,
        scikit_model=GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=1
        ),
    )
    if sum(model_bundle.y_train_pred) == 0:
        print("Model doen't learn any positive")
    accuracy = model_bundle.get_accuracy()
    auc = model_bundle.get_auc()
    accuracy = [ float('%.3f' % x) for x in accuracy]
    auc = [ float('%.3f' % x) for x in auc]
    return model_bundle, accuracy, auc


def explain_with_rules(model_bundle, Kmax, file_name, sampling):
    # parse sklearn tree ensembles into the array of (feature index, threshold)
    model_skl = DefragModel.parseSLtrees(model_bundle.model) 
    defrag_model = DefragModel(modeltype='classification', restart=3)
    y_train_pred = model_bundle.model.predict(model_bundle.x_train)

    num_train = int(sampling*len(model_bundle.x_train))
        
    start_time = time()
    np.random.seed(123)
    defrag_model.fit(model_bundle.x_train[:num_train], y_train_pred[:num_train], model_skl, Kmax, fittype='FAB', featurename=model_bundle.feature_names)
        
    rules = defrag_model.rule_
    print("Number of (defrag) rules:", len(rules))
    labels = defrag_model.pred_
    default_label = defrag_model.pred_default_
    for i in range(len(rules)):
        r = rules[i]
        l = labels[i]
        r_str = []
        for t in r:
            if(t[1] == 0):
                r_str.append(str(model_bundle.feature_names[t[0] - 1]) + " <= " + str(t[2]))
            if(t[1] == 1):
                r_str.append(str(model_bundle.feature_names[t[0] - 1]) + " > " + str(t[2]))
        r_str = " & ".join(r_str)
        #print("rule", i + 1, ":", r_str, l)
    #print("rule", len(rules) + 1, ":", "otherwise", default_label)
    
    # Rules can be read using
    # print(str(defrag_model))
    with open('./results/global_result/defrag/' + file_name + '.txt', 'w') as f:
        f.write(str(defrag_model))

    run_time = time() - start_time

    train_fidelities = get_fidelity(
        defrag_model,
        X=model_bundle.x_train, 
        y=model_bundle.y_train_pred)
    train_fidelities = [ float('%.6f' % fid) for fid in train_fidelities]

    test_fidelities = get_fidelity(
        defrag_model,
        X=model_bundle.x_test, 
        y=model_bundle.y_test_pred)
    test_fidelities = [ float('%.6f' % fid) for fid in test_fidelities]

    return rules, run_time, train_fidelities, test_fidelities

def get_fidelity(explainer, X, y):
    y_rules = explainer.predict(X)
    fidelity_positives = 0.0
    fidelity_negatives = 0.0
    positives = 0.0 + 1e-6
    negatives = 0.0 + 1e-6
    for i in range(len(y)):
        if y[i] == 1:
            positives = positives + 1
            if y[i] == y_rules[i]:
                fidelity_positives = fidelity_positives + 1
        if y[i] == 0:
            negatives = negatives + 1
            if y[i] == y_rules[i]:
                fidelity_negatives = fidelity_negatives + 1

    fidelity = (fidelity_positives + fidelity_negatives) / (
        positives + negatives
    )
    fidelity_positives = fidelity_positives / positives
    fidelity_negatives = fidelity_negatives / negatives
    return (fidelity, fidelity_positives, fidelity_negatives)

#generate_data()

result = {}
summary = []
te2rules = pd.read_csv('./results/global_result/te2rules/experiment.csv')

for dataset in ["compas", "bank", "adult"]:
    result['dataset']=dataset
    for num_trees in [100, 200, 500]:
        result['num_trees']=num_trees
        for depth in [3, 5]:
            result['depth']=depth
            sampling = 0.1
            result['sampling'] = sampling
            model_bundle, accuracy, auc = train_xgb_model(
                dataset_name=dataset,
                n_estimators=num_trees,
                max_depth=depth)
            print("Model Trained", dataset, num_trees, depth)


            result['train_accuracy'] = accuracy[0]
            result['test_accuracy'] = accuracy[1]
            result['train_auc'] = auc[0]
            result['test_auc'] = auc[1]

            df = te2rules[te2rules.dataset == dataset]
            df = df[df.num_trees == num_trees]
            df = df[df.depth == depth]
            df = df[df.num_stages == 3]
            df = df[df.jaccard_threshold == 0.2]            
            assert(len(df) == 1)
            df = df['num_rules']
            num_te2rules = df.iloc[0]

            for Kmax in [num_te2rules, 5*num_te2rules, 10*num_te2rules]:
                print("TERules got ", num_te2rules, " rules")
                print("Running defrag with Kmax =", Kmax)
                rules, run_time, train_fidelities, test_fidelities = explain_with_rules(
                    model_bundle, Kmax=Kmax, 
                    file_name = "rules_num_trees_" + str(num_trees) + "_depth_" + str(depth) + "_Kmax_" + str(Kmax),
                    sampling=sampling)
                print("Model Explained", dataset, num_trees, depth)

                result['num_rules']= len(rules)
                result['time']=run_time
                result['train_fidelity_total']=train_fidelities[0]
                result['train_fidelity_positive']=train_fidelities[1]
                result['train_fidelity_negative']=train_fidelities[2]
                result['test_fidelity_total']=test_fidelities[0]
                result['test_fidelity_positive']=test_fidelities[1]
                result['test_fidelity_negative']=test_fidelities[2]

                df = pd.DataFrame(result, index=[0])
                summary.append(df)
                summary_df = pd.concat(summary, ignore_index=True)
                summary_df.to_csv("./results/global_result/defrag/experiment.csv",index=False)
                '''
                pd.set_option('display.max_columns', None)  
                print(summary_df)
                '''
