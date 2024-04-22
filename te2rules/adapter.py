"""
This file contains adapters to convert scikit learn and xgboost tree ensemble models
into corresponding te2rules tree ensemble models. The tree ensemble models
of te2rules have the necessary structure for explaining itself using rules.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from xgboost import XGBClassifier

from te2rules.tree import DecisionTree, LeafNode, RandomForest, TreeNode


class ScikitGradientBoostingClassifierAdapter:
    """
    Class to convert sklearn.ensemble.GradientBoostingClassifier
    into a te2rules.tree.RandomForest object.

    Usage:
    adapter = ScikitGradientBoostingClassifierAdapter(model, feature_names)
    adapted_model = adapter.random_forest
    """

    def __init__(
        self, scikit_forest: GradientBoostingClassifier, feature_names: List[str]
    ):
        self.feature_names = feature_names

        n0, n1 = scikit_forest.init_.class_prior_
        self.bias = np.log(n1 / n0)
        self.weight = scikit_forest.get_params()["learning_rate"]
        self.activation = "sigmoid"

        scikit_tree_ensemble = scikit_forest.estimators_
        for dtr in scikit_tree_ensemble:
            assert len(dtr) == 1  # binary classification
        scikit_tree_ensemble = [dtr[0] for dtr in scikit_tree_ensemble]
        self.scikit_tree_ensemble = scikit_tree_ensemble

        self.random_forest = self._convert()

    def _convert(self) -> RandomForest:
        """
        Private method to create the te2rules.tree.RandomForest
        from the sklearn.ensemble.GradientBoostingClassifier object.
        """
        decision_tree_ensemble = []
        for scikit_tree in list(self.scikit_tree_ensemble):
            decision_tree = ScikitDecisionTreeRegressorAdapter(
                scikit_tree, self.feature_names
            ).decision_tree
            decision_tree_ensemble.append(decision_tree)

        return RandomForest(
            decision_tree_ensemble,
            weight=self.weight,
            bias=self.bias,
            feature_names=self.feature_names,
            activation=self.activation,
        )


class ScikitRandomForestClassifierAdapter:
    """
    Class to convert sklearn.ensemble.RandomForestClassifier
    into a te2rules.tree.RandomForest object.

    Usage:
    adapter = ScikitRandomForestClassifierAdapter(model, feature_names)
    adapted_model = adapter.random_forest
    """

    def __init__(self, scikit_forest: RandomForestClassifier, feature_names: List[str]):
        self.feature_names = feature_names

        self.bias = 0.0
        self.weight = 1.0 / scikit_forest.get_params()["n_estimators"]
        self.activation = "linear"

        self.scikit_tree_ensemble = scikit_forest.estimators_

        self.random_forest = self._convert()

    def _convert(self) -> RandomForest:
        """
        Private method to create the te2rules.tree.RandomForest
        from the sklearn.ensemble.RandomForestClassifier object.
        """
        decision_tree_ensemble = []
        for scikit_tree in list(self.scikit_tree_ensemble):
            decision_tree = ScikitDecisionTreeClassifierAdapter(
                scikit_tree, self.feature_names
            ).decision_tree
            decision_tree_ensemble.append(decision_tree)

        return RandomForest(
            decision_tree_ensemble,
            weight=self.weight,
            bias=self.bias,
            feature_names=self.feature_names,
            activation=self.activation,
        )


class ScikitDecisionTreeRegressorAdapter:
    """
    Class to convert sklearn.tree.DecisionTreeRegressor
    into a te2rules.tree.DecisionTree object.

    Usage:
    adapter = ScikitDecisionTreeRegressorAdapter(model, feature_names)
    adapted_model = adapter.decision_tree
    """

    def __init__(self, scikit_tree: DecisionTreeRegressor, feature_names: List[str]):
        self.feature_names = feature_names

        self.feature_indices = scikit_tree.tree_.feature
        self.threshold = scikit_tree.tree_.threshold
        self.children_left = scikit_tree.tree_.children_left
        self.children_right = scikit_tree.tree_.children_right
        self.LEAF_INDEX = _tree.TREE_UNDEFINED

        self.value = scikit_tree.tree_.value
        for i in range(len(scikit_tree.tree_.value)):
            assert len(scikit_tree.tree_.value[i]) == 1  # regressor
            assert len(scikit_tree.tree_.value[i][0]) == 1  # regressor
        self.value = [val[0][0] for val in self.value]

        self.decision_tree = self._convert()

    def _convert(self) -> DecisionTree:
        """
        Private method to create the te2rules.tree.DecisionTree
        from the sklearn.tree.DecisionTreeRegressor object.
        """
        nodes: List[DecisionTree] = []

        # Create Tree Nodes
        for i in range(len(self.feature_indices)):
            node_index = self.feature_indices[i]
            if node_index != self.LEAF_INDEX:
                node_name = self.feature_names[node_index]
                nodes = nodes + [
                    DecisionTree(
                        TreeNode(node_name=node_name, threshold=self.threshold[i])
                    )
                ]
            else:
                value = self.value[i]
                nodes = nodes + [DecisionTree(LeafNode(value=value))]

        # Connect Tree Nodes with each other
        for i in range(len(self.feature_indices)):
            node_index = self.feature_indices[i]
            if node_index != self.LEAF_INDEX:
                left_node = nodes[self.children_left[i]]
                nodes[i].left = left_node
                right_node = nodes[self.children_right[i]]
                nodes[i].right = right_node

        root_node = nodes[0]
        return root_node


class ScikitDecisionTreeClassifierAdapter:
    """
    Class to convert sklearn.tree.DecisionTreeClassifier
    into a te2rules.tree.DecisionTree object.

    Usage:
    adapter = ScikitDecisionTreeClassifierAdapter(model, feature_names)
    adapted_model = adapter.decision_tree
    """

    def __init__(self, scikit_tree: DecisionTreeClassifier, feature_names: List[str]):
        self.feature_names = feature_names

        self.feature_indices = scikit_tree.tree_.feature
        self.threshold = scikit_tree.tree_.threshold
        self.children_left = scikit_tree.tree_.children_left
        self.children_right = scikit_tree.tree_.children_right
        self.LEAF_INDEX = _tree.TREE_UNDEFINED

        value = []
        for i in range(len(scikit_tree.tree_.value)):
            assert len(scikit_tree.tree_.value[i]) == 1  # binary classification
            assert len(scikit_tree.tree_.value[i][0]) == 2  # binary classification

            prob_0 = scikit_tree.tree_.value[i][0][0]
            prob_1 = scikit_tree.tree_.value[i][0][1]
            value.append(prob_1 / (prob_0 + prob_1))
        self.value = value

        self.decision_tree = self._convert()

    def _convert(self) -> DecisionTree:
        """
        Private method to create the te2rules.tree.DecisionTree
        from the sklearn.tree.DecisionTreeClassifier object.
        """
        nodes: List[DecisionTree] = []

        # Create Tree Nodes
        for i in range(len(self.feature_indices)):
            node_index = self.feature_indices[i]
            if node_index != self.LEAF_INDEX:
                node_name = self.feature_names[node_index]
                nodes = nodes + [
                    DecisionTree(
                        TreeNode(node_name=node_name, threshold=self.threshold[i])
                    )
                ]
            else:
                value = self.value[i]
                nodes = nodes + [DecisionTree(LeafNode(value=value))]

        # Connect Tree Nodes with each other
        for i in range(len(self.feature_indices)):
            node_index = self.feature_indices[i]
            if node_index != self.LEAF_INDEX:
                left_node = nodes[self.children_left[i]]
                nodes[i].left = left_node
                right_node = nodes[self.children_right[i]]
                nodes[i].right = right_node

        root_node = nodes[0]
        return root_node


class XgboostXGBClassifierAdapter:
    """
    Class to convert xgboost.sklearn.XGBClassifier
    into a te2rules.tree.RandomForest object.

    Usage:
    adapter = XgboostXGBClassifierAdapter(model, feature_names)
    adapted_model = adapter.random_forest
    """

    def __init__(self, xgb_model: XGBClassifier, feature_names: List[str]):
        self.xgb_model_json = xgb_model.get_booster().get_dump(dump_format="json")
        self.feature_names = feature_names

        self.bias = 0.0
        self.weight = 1.0
        self.activation = "sigmoid"

        self.random_forest = self._convert()

    def _build_tree(self, tree_dict: Dict[str, Any]) -> DecisionTree:
        """
        Private method to perform a DFS traversal of the model json
        and build the te2rules.tree.DecisionTree object.
        """
        if "leaf" in tree_dict:
            node = DecisionTree(LeafNode(value=float(tree_dict["leaf"])))
        else:
            # Get feature index. Ex: feature-21 would be
            # represented as f21. Get index 21 from f21.
            i = int(tree_dict["split"][1:])

            # XGBoost and Scikit Learn treat splits differently.
            # Scikit Learn uses left node for f1 <= threshold while
            # XGBoost uses left node for f1 < threshold
            node = DecisionTree(
                TreeNode(
                    node_name=self.feature_names[i],
                    threshold=float(tree_dict["split_condition"]),
                    equality_on_left=False,
                )
            )
            node.left = self._build_tree(tree_dict["children"][0])
            node.right = self._build_tree(tree_dict["children"][1])
        return node

    def _convert(self) -> RandomForest:
        """
        Private method to create the te2rules.tree.RandomForest
        from the xgboost.sklearn.XGBClassifier object.
        """
        self.tree_ensemble = []
        for i in range(len(self.xgb_model_json)):
            tree_dict = json.loads(self.xgb_model_json[i])
            node = self._build_tree(tree_dict)
            self.tree_ensemble.append(node)

        return RandomForest(
            list(self.tree_ensemble),
            weight=self.weight,
            bias=self.bias,
            feature_names=self.feature_names,
            activation=self.activation,
        )
