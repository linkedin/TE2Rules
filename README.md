# TE2Rules

Tree Ensembles (eg. Gradient Boosted Trees and Random Forests) provide a higher prediction performance compared to Single Decision Trees. However, it is generally difficult for humans to make sense of Tree Ensembles (TE), resulting in a lack of model transparency and interpretability. This project presents a novel approach to convert a Tree Ensemble  trained for a binary classification task, to an equivalent, globally comprehensible rule list. This rule list (RL) captures the necessary and sufficient conditions for classification by the TE. Experiments on benchmark datasets demonstrate that (i) predictions from the RL have high fidelity w.r.t. the original TE, (ii) RL have high interpretability measured by the number of the decision rules, (iii) there is an easy trade-off between RL fidelity and its interpretability and compute time, and (iv) RL can provide a fast alternative to state-of-art rule-based instance-level outcome explanation techniques.


## Required python packages :
anchor==0.4.0 \
anchor_exp==0.0.2.0 \
beautifulsoup4==4.11.1 \
joblib==1.1.0 \
lib==4.0.0 \
lightgbm==3.3.2 \
lime==0.2.0.1 \
matplotlib==3.5.2 \
numpy==1.22.3 \
pandas==1.4.2 \
requests==2.27.1 \
scikit_learn==1.1.1 \
six==1.15.0 \
skope_rules==1.0.1 \
xgboost==1.6.1

## Required R packages :
randomForest \
inTrees 

## Experiment results :
Run the follwing python scripts to generate the results in the paper: \
```python3 global_baseline_experiment.py``` \
```python3 global_te2rules_experiment.py``` \
```python3 outcome_explanation_experiments.py``` 
