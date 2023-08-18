import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from trainer import Trainer
from te2rules.explainer import ModelExplainer
from time import time

import te2rules
print("Using TE2Rules version:", te2rules.__version__)

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


def explain_with_rules(model_bundle, sampling, num_stages, min_precision, jaccard_threshold):
    model_explainer = ModelExplainer(
        model=model_bundle.model, 
        feature_names=model_bundle.feature_names,
        verbose=True)
    
    num_train = int(sampling*len(model_bundle.x_train))
    
    start_time = time()
    rules = model_explainer.explain(
        X=model_bundle.x_train[:num_train],
        y=model_bundle.y_train_pred[:num_train],
        num_stages=num_stages,
        min_precision=min_precision, 
        jaccard_threshold=jaccard_threshold)
    run_time = time() - start_time

    train_fidelities = model_explainer.get_fidelity(
        X=model_bundle.x_train, 
        y=model_bundle.y_train_pred)
    train_fidelities = [ float('%.6f' % fid) for fid in train_fidelities]

    test_fidelities = model_explainer.get_fidelity(
        X=model_bundle.x_test, 
        y=model_bundle.y_test_pred)
    test_fidelities = [ float('%.6f' % fid) for fid in test_fidelities]

    return rules, run_time, train_fidelities, test_fidelities


generate_data()

result = {}
min_precision=0.95
summary = []
for dataset in ["bank", "compas", "adult"]:
    result['dataset']=dataset
    for num_trees in [100, 200, 500]:
        result['num_trees']=num_trees
        for depth in [3, 5]:
            result['depth']=depth

            model_bundle, accuracy, auc = train_xgb_model(
                dataset_name=dataset,
                n_estimators=num_trees,
                max_depth=depth)

            for num_stages in [1, 2, 3]:
                for jaccard_threshold in [0.2]:
                    for sampling_ratio in [0.1]:
                        result['num_stages']=num_stages
                        result['min_precision']=min_precision
                        
                        result['sampling']=sampling_ratio
                        result['jaccard_threshold']=jaccard_threshold

                        result['train_accuracy'] = accuracy[0]
                        result['test_accuracy'] = accuracy[1]
                        result['train_auc'] = auc[0]
                        result['test_auc'] = auc[1]

                        rules, run_time, train_fidelities, test_fidelities = explain_with_rules(
                            model_bundle, 
                            sampling=sampling_ratio, 
                            num_stages=num_stages, 
                            min_precision=min_precision,
                            jaccard_threshold=jaccard_threshold)

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
                        summary_df.to_csv("./results/global_result/te2rules/experiment.csv",index=False)
                        '''
                        pd.set_option('display.max_columns', None)  
                        print(summary_df)
                        '''
