import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from trainer import Trainer
from time import time
import numpy as np 
from converter_utils import skl2xgb

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
    output_path = "./models/" + dataset_name
    skl2xgb(model_bundle, n_estimators, max_depth, output_path)

    if sum(model_bundle.y_train_pred) == 0:
        print("Model doen't learn any positive")
    accuracy = model_bundle.get_accuracy()
    auc = model_bundle.get_auc()
    accuracy = [ float('%.3f' % x) for x in accuracy]
    auc = [ float('%.3f' % x) for x in auc]
    return model_bundle, accuracy, auc

def explain_with_rules(model_bundle, dataset_name, num_trees, depth, sampling):
    train_file = "./data/{}/train.csv".format(dataset_name)
    test_file = "./data/{}/test.csv".format(dataset_name)
    result_dir = "./models/{}".format(dataset_name)

        
    start_time = time()
    np.random.seed(123)
    os.system("Rscript ./demo/run_intrees.R %s %s %s %s %s %s"% (train_file, test_file, result_dir, num_trees, depth, sampling))
    
    intrees_model_pred =  pd.read_csv(result_dir + "/intrees/pred_train_score.csv", header=None)
    intrees_model_pred =  intrees_model_pred.to_numpy().reshape(-1)
    intrees_model_pred = intrees_model_pred[:5]
    original_model_pred = model_bundle.model.predict_proba(model_bundle.x_train[:5])[:,1]
    for j in range(5):
        assert(abs(intrees_model_pred[j] - original_model_pred[j]) < 10**-6)
    
    rules = pd.read_csv(result_dir + "/intrees/rules.txt")
    run_time = time() - start_time

    y_train_rules =  pd.read_csv(result_dir + "/intrees/pred_train_rules.csv", header=None).to_numpy()
    y_test_rules =  pd.read_csv(result_dir + "/intrees/pred_test_rules.csv", header=None).to_numpy()

    num_train = int(sampling*len(model_bundle.x_train))
    assert(num_train == len(y_train_rules))
    
    train_fidelities = get_fidelity(
        y_rules=y_train_rules,
        X=model_bundle.x_train[:num_train], 
        y=model_bundle.y_train_pred[:num_train])
    train_fidelities = [ float('%.6f' % fid) for fid in train_fidelities]
    
    test_fidelities = get_fidelity(
        y_rules=y_test_rules,
        X=model_bundle.x_test, 
        y=model_bundle.y_test_pred)
    test_fidelities = [ float('%.6f' % fid) for fid in test_fidelities]

    return rules, run_time, train_fidelities, test_fidelities

def get_fidelity(y_rules, X, y):
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
    return (fidelity, fidelity_positives, fidelity_negatives, positives, negatives)

def get_precision_recall(acc_pos, acc_neg, pos, neg):
    tp = acc_pos*pos
    fp = (1-acc_neg)*neg
    precision = tp/(tp + fp + 10**-6)    
    recall = acc_pos  
    return precision, recall

#generate_data()

result = {}
summary = []
for dataset in ["compas", "bank", "adult"]:
    result['dataset']=dataset
    for num_trees in [100, 200, 500]:
        result['num_trees']=num_trees
        for depth in [3, 5]:
            result['depth']=depth
            sampling = 0.1
            result['sampling']=sampling
            model_bundle, accuracy, auc = train_xgb_model(
                dataset_name=dataset,
                n_estimators=num_trees,
                max_depth=depth)
            print("Model Trained", dataset, num_trees, depth)


            result['train_accuracy'] = accuracy[0]
            result['test_accuracy'] = accuracy[1]
            result['train_auc'] = auc[0]
            result['test_auc'] = auc[1]

            rules, run_time, train_fidelities, test_fidelities = explain_with_rules(
                model_bundle, dataset, num_trees, depth, sampling)
            print("Model Explained", dataset, num_trees, depth)

            result['num_rules']= len(rules)
            result['time']=run_time
            result['train_fidelity_total']=train_fidelities[0]
            result['train_fidelity_positive']=train_fidelities[1]
            result['train_fidelity_negative']=train_fidelities[2]
            result['test_fidelity_total']=test_fidelities[0]
            result['test_fidelity_positive']=test_fidelities[1]
            result['test_fidelity_negative']=test_fidelities[2]

            result['train_positive']=train_fidelities[3]
            result['train_negative']=train_fidelities[4]
            result['test_positive']=test_fidelities[3]
            result['test_negative']=test_fidelities[4]
            
            precision, recall = get_precision_recall(result['train_fidelity_positive'], 
                                                    result['train_fidelity_negative'], 
                                                    result['train_positive'], 
                                                    result['train_negative'])
            result['train_precision']=precision
            result['train_recall']=recall

            precision, recall = get_precision_recall(result['test_fidelity_positive'], 
                                                    result['test_fidelity_negative'], 
                                                    result['test_positive'], 
                                                    result['test_negative'])
            result['test_precision']=precision
            result['test_recall']=recall
            
            df = pd.DataFrame(result, index=[0])
            summary.append(df)
            summary_df = pd.concat(summary, ignore_index=True)
            summary_df.to_csv("./results/global_result/intrees/experiment.csv",index=False)
            '''
            pd.set_option('display.max_columns', None)  
            print(summary_df)
            '''
