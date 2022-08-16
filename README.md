# TE2Rules

TE2Rules is a technique to explain Tree Ensemble models (TE) like XGBoost, Random Forest, trained on a binary classification task, using a rule list. The extracted rule list (RL) captures the necessary and sufficient conditions for classification by the Tree Ensemble. The algorithm used by TE2Rules is based on Apriori Rule Mining.

TE2Rules provides a ```ModelExplainer``` which takes a trained TE model and training data to extract rules. Though the algorithm can be run without any data, we would recommend against doing so. The training data is used for extracting rules with relevant combination of input features. Without data, the algorithm would try to extract rules for all possible combinations of input features, including those combinations which are extremely rare in the data. 


## Documentation


TE2Rules contains a ```ModelExplainer``` class with an ```explain()``` method which returns a rule list corresponding to the positive class prediciton of the tree ensemble. While using the rule list, any data instance that does not trigger any of the extracted rules is to be interpreted as belonging to the negative class. The ```explain()``` method has two tunable parameters to control the interpretability, faithfulness, runtime and coverage of the extracted rules. These are: 
- ```min_precision```: ```min_precision``` controls the minimum precision of extracted rules. Setting it to a smaller threhsold, allows extracting shorter (more interpretable, but less faithful) rules. By default, the algorithm uses a minimum precision threshold of 0.95.  
- ```num_stages```: The algorithm runs in stages starting from stage 1, stage 2 to all the way till stage n where n is the number of trees in the ensemble. Stopping the algorithm at an early stage  results in a few short rules (with quicker run time, but less coverage in data). By default, the algorithm explores all stages before terminating.

For evaluating the performance of the extracted rule list, the ```ModelExplainer``` provides a method ```get_fidelity()``` which returns the fractions of data for which the rule list agrees with the tree ensemble. ```get_fidelity()``` returns the fidelity on positives, negatives and overall fidelity. 


## For reproducing results in the paper :
Run the follwing python scripts to generate the results in the paper:
```
python3 global_baseline_experiment.py
python3 global_te2rules_experiment.py
python3 outcome_explanation_experiments.py
``` 
