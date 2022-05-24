from sklearn.ensemble import GradientBoostingClassifier
from lib.trainer import Trainer
from lib.adapter import ScikitTreeAdapter, ScikitForestAdapter
from sklearn.tree import export_text, plot_tree
from lib.rule import Rule
from lib.rule_builder import RuleBuilder
import pandas as pd 
import os 

# Train model to explain
def train_tree_ensemble(training_data_loc, testing_data_loc, n_estimators, max_depth):
    print("XGBoost Model")
    trainer = Trainer(training_data_loc=training_data_loc, 
        testing_data_loc=testing_data_loc,
        scikit_model=GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1))
    accuracy, auc = trainer.evaluate_model()
    print("Accuracy: ", accuracy)
    print("AUC: ", auc)
    y_pred = trainer.model.predict(trainer.x_train)
    if(sum(y_pred) == 0):
        print("Model doen't learn any positive")
    random_forest = ScikitForestAdapter(trainer.model, trainer.feature_names).random_forest
    return trainer, random_forest, y_pred, accuracy, auc

# Explain using rules 
# explain_with_rules(trainer.x_train, y_pred, pd.read_csv("data/train.csv"), random_forest)
def explain_with_rules(x_train, y_train_pred, random_forest, num_stages, decision_rule_precision):
    # build rules
    rule_builder = RuleBuilder(random_forest=random_forest, num_stages=num_stages, decision_rule_precision=decision_rule_precision)
    final_rules = rule_builder.explain(X=x_train, y=y_train_pred) 
    return final_rules, rule_builder.fidelities

# rule: a list of predicate strings
def dedup_rule_predicates(rule):
    pred_dict = {}
    for pred in rule:
        f, op, val = pred.split()
        # determine direction of op
        op_type = 'equal'
        if op in ('<', '<='):
            op_type = 'less than'
        elif op in ('>', '>='):
            op_type = 'greater than'
        # store value if haven't seen (f, op_type)
        if (f, op_type) not in pred_dict:
            pred_dict[(f, op_type)] = (op, val)
        # otherwise, combine rules
        else:
            old_op, old_val = pred_dict[(f, op_type)]
            if (old_op == '<=' and op == '<' and val == old_val) or (old_op == '>=' and op == '>' and val == old_val):
                pred_dict[(f, op_type)] = (op, val)
            elif (op_type == 'less than' and val < old_val) or (op_type == 'greater than' and val > old_val):
                pred_dict[(f, op_type)] = (op, val)
    # return final predicate list
    final_rule = []
    for (f, _) in pred_dict:
        op, val = pred_dict[(f, _)]
        final_rule.append((' ').join([f, op, val]))
    return final_rule