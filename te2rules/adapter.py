import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree

from te2rules.tree import DecisionTree, LeafNode, RandomForest, TreeNode


class ScikitGradientBoostingClassifierAdapter:
    def __init__(self, scikit_forest, feature_names):
        assert type(scikit_forest) == GradientBoostingClassifier

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

        self.random_forest = self.convert()

    def convert(self):
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
    def __init__(self, scikit_forest, feature_names):
        assert type(scikit_forest) == RandomForestClassifier

        self.feature_names = feature_names

        self.bias = 0.0
        self.weight = 1.0 / scikit_forest.get_params()["n_estimators"]
        self.activation = "linear"

        self.scikit_tree_ensemble = scikit_forest.estimators_

        self.random_forest = self.convert()

    def convert(self):
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
    def __init__(self, scikit_tree, feature_names):
        assert type(scikit_tree) == DecisionTreeRegressor

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

        self.decision_tree = self.convert()

    def convert(self):
        nodes = []

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
    def __init__(self, scikit_tree, feature_names):
        assert type(scikit_tree) == DecisionTreeClassifier

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

        self.decision_tree = self.convert()

    def convert(self):
        nodes = []

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
