# TE2Rules
[![License](https://img.shields.io/badge/license-BSD-green.svg)](https://github.com/groshanlal/TE2Rules/blob/master/LICENSE)
[![Paper](http://img.shields.io/badge/cs.LG-arXiv%3A2206.14359-orange.svg)](https://arxiv.org/abs/2206.14359)

Tree Ensembles (eg. Gradient Boosted Trees and Random Forests) provide a higher prediction performance compared to Single Decision Trees. However, it is generally difficult for humans to make sense of Tree Ensembles (TE), resulting in a lack of model transparency and interpretability. This project presents a novel approach to convert a Tree Ensemble  trained for a binary classification task, to an equivalent, globally comprehensible rule list. This rule list (RL) captures the necessary and sufficient conditions for classification by the TE. Experiments on benchmark datasets demonstrate that (i) predictions from the RL have high fidelity w.r.t. the original TE, (ii) RL have high interpretability measured by the number of the decision rules, (iii) there is an easy trade-off between RL fidelity and its interpretability and compute time, and (iv) RL can provide a fast alternative to state-of-art rule-based instance-level outcome explanation techniques. 

For more detailed introduction of TE2Rules, please check out our [paper](https://arxiv.org/abs/2206.14359).

## Required python packages for TE2Rules:
numpy >=1.22.3 \
pandas >=1.4.2 \
scikit-learn >=1.1.1 \
six >=1.15.0 \
testresources \
tqdm \
xgboost >=1.6.1

## Required python packages for running experiments:
anchor==0.4.0 \
anchor_exp==0.0.2.0 \
beautifulsoup4==4.11.1 \
joblib==1.1.0 \
lib==4.0.0 \
lightgbm==3.3.2 \
lime==0.2.0.1 \
matplotlib==3.5.2 \
requests==2.27.1 \
skope_rules==1.0.1

## Required R packages for running experiments:
randomForest \
inTrees 

## For reproducing results in the paper :
Run the follwing python scripts to generate the results in the paper:
```
python3 global_baseline_experiment.py
python3 global_te2rules_experiment.py
python3 outcome_explanation_experiments.py
``` 

## License
BSD 2-Clause License, see [LICENSE](https://github.com/groshanlal/TE2Rules/blob/master/LICENSE).

## Citation
Please cite [TE2Rules](https://arxiv.org/abs/2206.14359) in your publications if it helps your research:
```
@article{te2rules2022,
  title={TE2Rules: Extracting Rule Lists from Tree Ensembles},
  author={Lal, G Roshan and Chen, Xiaotong and Mithal, Varun},
  journal={arXiv preprint arXiv:2206.14359},
  year={2022}
}
```
