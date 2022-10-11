"""
This file contains the tree and tree ensemble objects in te2rules.
The tree ensemble models of te2rules have the necessary structure for
explaining itself using rules.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from te2rules.rule import Rule


class Node:
    """
    Base class of nodes in decision trees.
    These nodes may or may not be leaf nodes.
    """

    def __init__(self, is_leaf: bool):
        self.is_leaf = is_leaf


class TreeNode(Node):
    """
    Class of internal (non-leaf) nodes in decision trees.
    Each internal node has a feature name and threshold value
    on which the split is performed.

    The left split is taken when feature value <= threshold value
    and the right split is taken when feature value > threshold value.
    """

    def __init__(self, node_name: str, threshold: float):
        super().__init__(is_leaf=False)
        self.node_name = node_name
        self.threshold = threshold

    def __str__(self) -> str:
        return self.node_name

    def get_left_clause(self) -> str:
        return self.node_name + " <= " + str(self.threshold)

    def get_right_clause(self) -> str:
        return self.node_name + " > " + str(self.threshold)


class LeafNode(Node):
    """
    Class of terminal (leaf) nodes in decision trees.
    Each leaf node has a value assigned to data reaching the leaf.
    """

    def __init__(self, value: float):
        super().__init__(is_leaf=True)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def get_leaf_clause(self) -> str:
        return "value: " + str(self.value)


class DecisionTree:
    """
    Class of decision tree.
    The decision tree object consists of:
    1) Node: TreeNode or LeafNode containing information on how the decision
        is performed at the node.
    2) Left Sub Tree: A decision tree for the left branch.
    2) Right Sub Tree: A decision tree for the right branch.

    The node should be either LeafNode or TreeNode only.
    When the node is a TreeNode, the left and right sub trees cannot be None.
    """

    def __init__(
        self,
        node: Node,
        left: Optional[DecisionTree] = None,
        right: Optional[DecisionTree] = None,
    ):
        self.node = node
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return self._str_recursive(num_indent=0)

    def _str_recursive(self, num_indent: int) -> str:
        indentation = "|   " * num_indent + "|---"
        if isinstance(self.node, LeafNode):
            string_rep = indentation + self.node.get_leaf_clause()
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                string_rep = indentation + self.node.get_left_clause() + "\n"
                string_rep += self.left._str_recursive(num_indent=num_indent + 1) + "\n"
                string_rep += indentation + self.node.get_right_clause() + "\n"
                string_rep += self.right._str_recursive(num_indent=num_indent + 1)
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")
        return string_rep

    def _propagate_decision_rule(self, decision_rule: List[str]) -> None:
        """
        Decision rule stands for the list of decisions made to reach the current node.
        Example:
        ['f1 < 0.2', 'f3 > 0.5'] can be the decisions made to reach the current node.

        This method sets the decision rule of the current node as the list of decisions
        given by the user and the decision rule of descendant nodes as the list of
        decisions made at the current node and the decisions made at each node along
        the path from the current node to the descendant node.

        This method is private.
        """
        self.decision_rule = decision_rule
        if isinstance(self.node, LeafNode):
            pass
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                left_decision_rule = decision_rule + [self.node.get_left_clause()]
                self.left._propagate_decision_rule(left_decision_rule)
                right_decision_rule = decision_rule + [self.node.get_right_clause()]
                self.right._propagate_decision_rule(right_decision_rule)
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")

    """
    def aggregate_min_decision_value(self) -> None:
        if isinstance(self.node, LeafNode):
            self.min_decision_value = self.node.value
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                self.left.aggregate_min_decision_value()
                self.right.aggregate_min_decision_value()
                self.min_decision_value = min(
                    self.left.min_decision_value, self.right.min_decision_value
                )
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")
    """
    """
    def aggregate_max_decision_value(self) -> None:
        if isinstance(self.node, LeafNode):
            self.max_decision_value = self.node.value
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                self.left.aggregate_max_decision_value()
                self.right.aggregate_max_decision_value()
                self.max_decision_value = max(
                    self.left.max_decision_value, self.right.max_decision_value
                )
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")
    """

    def _propagate_decision_support(
        self,
        data: List[List[float]],
        feature_names: List[str],
        decision_support: List[int],
    ) -> None:
        """
        Support stands for the list of indices of data records that pass through
        the current node while the decision tree makes a decision.

        This method sets the support of the current node as the list of indices
        given by the user and the support of descendant nodes as the subset (list) of
        the given indices that pass through the descendant node  while the decision
        tree makes a decision.

        This method is private.
        """
        self.decision_support = decision_support
        if isinstance(self.node, LeafNode):
            pass
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                feature_index = feature_names.index(self.node.node_name)
                left_decision_support = []
                right_decision_support = []
                for index in self.decision_support:
                    if data[index][feature_index] <= self.node.threshold:
                        left_decision_support.append(index)
                    else:
                        right_decision_support.append(index)
                self.left._propagate_decision_support(
                    data, feature_names, left_decision_support
                )
                self.right._propagate_decision_support(
                    data, feature_names, right_decision_support
                )
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")

    def get_rules(
        self, data: List[List[float]], feature_names: List[str], tree_id: int
    ) -> List[Rule]:
        """
        This method sets the decision rule and support for all nodes in the
        decision tree starting from the root node. The decision rule of the
        root node of the decision tree is set to a empty list and the support
        of the root node of the decision tree is set to the list of all indices
        in the data.

        After setting the decision rule and support, it collects the rule objects
        from all the nodes in the decision tree.
        """
        support = [i for i in range(len(data))]
        # self.aggregate_min_decision_value()
        # self.aggregate_max_decision_value()
        self._propagate_decision_rule(decision_rule=[])
        self._propagate_decision_support(data, feature_names, support)
        rules = self._collect_rules(tree_id=tree_id, node_id=0, rules=[])
        return rules

    def _collect_rules(
        self, tree_id: int, node_id: int, rules: List[Rule]
    ) -> List[Rule]:
        """
        This method collects the rule objects from all the nodes in the
        decision tree and returns the list of rules appended with list
        supplied by the user.
        1) The rule object's decision rule and support are taken from the
        corresponding node.
        2) The rule objects's identity is set to be [tree_id_node_id].
        The tree_id of the rule objects is taken from the user. The node_id
        of the current node is taken from the user.
        3) The node_ids of the left and right child are set to be 2 * node_id + 1,
        2 * node_id + 2 given the current node's node_id.

        This method is private.
        """
        rules = rules + [
            Rule(
                decision_rule=sorted(self.decision_rule),
                decision_support=self.decision_support,
                identity=[str(tree_id) + "_" + str(node_id)],
            )
        ]
        if isinstance(self.node, LeafNode):
            pass
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                rules = self.left._collect_rules(tree_id, 2 * node_id + 1, rules)
                rules = self.right._collect_rules(tree_id, 2 * node_id + 2, rules)
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")
        return rules

    def get_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        This method (when called with scores = {}) returns a dict
        that maps the indices of the data to the corresponding score
        returned by the decision tree.

        When the method is called with some scores dict, the returned
        scores has scores from the user given dict + score of the
        decision tree for the corresponding indices.

        The data is contained in the support of nodes in the decision
        tree. Hence, this method can be called only after calling get_rules().
        It will raise AttributeError otherwise.
        """
        if isinstance(self.node, LeafNode):
            if not hasattr(self, "decision_support"):
                raise AttributeError(
                    "Support of nodes in the decision trees not found, "
                    + "since no data has been supplied to the model. "
                    + "Call get_rules() with data, before calling get_scores()."
                )
            for i in self.decision_support:
                if i not in scores:
                    scores[i] = self.node.value
                else:
                    scores[i] = scores[i] + self.node.value
        elif isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                scores = self.left.get_scores(scores)
                scores = self.right.get_scores(scores)
        else:
            raise ValueError("Node has to be LeafNode or TreeNode")
        return scores

    """
    def get_rule_score(self, rule: List[str]) -> List[float]:
        if isinstance(self.node, TreeNode):
            if (self.left is None) or (self.right is None):
                raise ValueError("TreeNode cannot have None as children")
            else:
                if self.node.get_left_clause() in rule:
                    return self.left.get_rule_score(rule)

                if self.node.get_right_clause() in rule:
                    return self.right.get_rule_score(rule)
        return [self.min_decision_value, self.max_decision_value]
    """


class RandomForest:
    """
    Class of random forest.
    The random forest object consists of:
    1) list of decision tree objects,
    2) weight multiplied to each decision tree score,
        bias added to the score,
        activation function applied on the score
    3) list of feature names in the data
    """

    def __init__(
        self,
        decision_tree_ensemble: List[DecisionTree],
        weight: float,
        bias: float,
        feature_names: List[str],
        activation: str,
    ):
        self.decision_tree_ensemble = decision_tree_ensemble
        self.weight = weight
        self.bias = bias
        self.feature_names = feature_names
        self.activation = activation
        if self.activation not in ["sigmoid", "linear"]:
            raise ValueError("activation of forest can only be sigmoid or linear")

    def get_num_trees(self) -> int:
        return len(self.decision_tree_ensemble)

    def get_rules(self, data: List[List[float]]) -> List[Rule]:
        """
        This method gets rule objects from decision trees in the random forest
        which have support in the data given by the user.
        """
        rules_from_tree: List[Rule] = []
        for tree_index, decision_tree in enumerate(self.decision_tree_ensemble):
            rules = decision_tree.get_rules(
                data=data, feature_names=self.feature_names, tree_id=tree_index
            )
            rules_from_tree = rules_from_tree + rules
        return rules_from_tree

    def get_scores(self) -> List[float]:
        """
        This method returns a list of scores by the random forest
        on the data.

        The data is contained in the support of nodes in the random
        forest. Hence, this method can be called only after calling get_rules().
        It will raise AttributeError otherwise.
        """
        scores: Dict[int, float] = {}
        for tree in self.decision_tree_ensemble:
            try:
                scores = tree.get_scores(scores)
            except AttributeError:
                raise AttributeError(
                    "Support of nodes in the decision trees not found, "
                    + "since no data has been supplied to the model. "
                    + "Call get_rules() with data, before calling get_scores()."
                )

        for i in scores.keys():
            scores[i] = scores[i] * self.weight + self.bias
            scores[i] = self._activation_function(scores[i])

        assert set(range(max(scores.keys()) + 1)) == set(scores.keys())

        scores_list = []
        for i in range(max(scores.keys()) + 1):
            scores_list.append(scores[i])

        return scores_list

    """
    def get_rule_score(self, rule: List[str]) -> List[float]:
        min_score = self.bias
        max_score = self.bias
        for decision_tree in self.decision_tree_ensemble:
            score = decision_tree.get_rule_score(rule)
            min_score = min_score + score[0] * self.weight
            max_score = max_score + score[1] * self.weight
        min_score = self.activation_function(min_score)
        min_score = self.thresholding_function(min_score)
        max_score = self.activation_function(max_score)
        max_score = self.thresholding_function(max_score)
        return [min_score, max_score]
    """

    def _activation_function(self, value: float) -> float:
        """
        Method to apply activation function on the value.
        """
        if self.activation == "linear":
            transformed_value = value
        elif self.activation == "sigmoid":
            transformed_value = 1 / (1 + 2.71828 ** (-value))
        else:
            raise ValueError("activation of forest can only be sigmoid or linear")
        return transformed_value

    def _thresholding_function(self, value: float) -> float:
        """
        Method to apply thresholding function on the score of the model
        to convert the score to the binary class label.
        """
        if value >= 0.5:
            thresholded_value = 1.0
        else:
            thresholded_value = 0.0
        return thresholded_value

    def __str__(self) -> str:
        string_rep = ""
        for i in range(len(self.decision_tree_ensemble)):
            string_rep = string_rep + "Tree " + str(i) + "\n"
            string_rep = string_rep + self.decision_tree_ensemble[i].__str__() + "\n\n"
        return string_rep
