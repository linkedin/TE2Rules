"""
This file contains unit tests for te2rules.tree.RandomForest.
"""
from te2rules.tree import DecisionTree, LeafNode, RandomForest, TreeNode

tree_0 = DecisionTree(
    TreeNode("f0", 0.5),
    left=DecisionTree(
        TreeNode("f1", 0.5),
        left=DecisionTree(LeafNode(1.0)),
        right=DecisionTree(LeafNode(0.0)),
    ),
    right=DecisionTree(
        TreeNode("f3", 0.5),
        left=DecisionTree(LeafNode(-2.0)),
        right=DecisionTree(LeafNode(0.0)),
    ),
)

tree_0_str = [
    "|---f0 <= 0.5",
    "|   |---f1 <= 0.5",
    "|   |   |---value: 1.0",
    "|   |---f1 > 0.5",
    "|   |   |---value: 0.0",
    "|---f0 > 0.5",
    "|   |---f3 <= 0.5",
    "|   |   |---value: -2.0",
    "|   |---f3 > 0.5",
    "|   |   |---value: 0.0",
]

tree_1 = DecisionTree(
    TreeNode("f3", 0.5),
    left=DecisionTree(
        TreeNode("f2", 0.5),
        left=DecisionTree(LeafNode(0.017)),
        right=DecisionTree(LeafNode(-1.94)),
    ),
    right=DecisionTree(
        TreeNode("f1", 0.5),
        left=DecisionTree(LeafNode(-0.067)),
        right=DecisionTree(LeafNode(1.0)),
    ),
)

tree_1_str = [
    "|---f3 <= 0.5",
    "|   |---f2 <= 0.5",
    "|   |   |---value: 0.017",
    "|   |---f2 > 0.5",
    "|   |   |---value: -1.94",
    "|---f3 > 0.5",
    "|   |---f1 <= 0.5",
    "|   |   |---value: -0.067",
    "|   |---f1 > 0.5",
    "|   |   |---value: 1.0",
]

tree_2 = DecisionTree(
    TreeNode("f0", 0.5),
    left=DecisionTree(
        TreeNode("f1", 0.5),
        left=DecisionTree(LeafNode(0.93)),
        right=DecisionTree(LeafNode(-0.035)),
    ),
    right=DecisionTree(
        TreeNode("f2", 0.5),
        left=DecisionTree(LeafNode(-1.94)),
        right=DecisionTree(LeafNode(0.082)),
    ),
)

tree_2_str = [
    "|---f0 <= 0.5",
    "|   |---f1 <= 0.5",
    "|   |   |---value: 0.93",
    "|   |---f1 > 0.5",
    "|   |   |---value: -0.035",
    "|---f0 > 0.5",
    "|   |---f2 <= 0.5",
    "|   |   |---value: -1.94",
    "|   |---f2 > 0.5",
    "|   |   |---value: 0.082",
]


feature_names = ["f0", "f1", "f2", "f3"]
data = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
]

tree_ensemble_weights = 0.8
tree_ensemble_bias = -0.3
forest = RandomForest(
    decision_tree_ensemble=[tree_0, tree_1, tree_2],
    weight=tree_ensemble_weights,
    bias=tree_ensemble_bias,
    feature_names=feature_names,
    activation="sigmoid",
)

rules = forest.get_rules(data)
scores = forest.get_scores()


def test_str_rep() -> None:
    """
    Unit test for te2rules.tree.RandomForest.__str__()
    """
    forest_str = (
        "Tree 0\n"
        + "\n".join(tree_0_str)
        + "\n\n"
        + "Tree 1\n"
        + "\n".join(tree_1_str)
        + "\n\n"
        + "Tree 2\n"
        + "\n".join(tree_2_str)
        + "\n\n"
    )
    assert str(forest) == forest_str


def test_num() -> None:
    """
    Unit test for te2rules.tree.RandomForest.get_num_trees()
    """
    assert forest.get_num_trees() == 3


def test_rules() -> None:
    """
    Unit test for te2rules.tree.RandomForest.get_rules()
    """
    rules_str = [
        "",
        "f0 <= 0.5",
        "f0 <= 0.5 & f1 <= 0.5",
        "f0 <= 0.5 & f1 > 0.5",
        "f0 > 0.5",
        "f0 > 0.5 & f3 <= 0.5",
        "f0 > 0.5 & f3 > 0.5",
        "",
        "f3 <= 0.5",
        "f2 <= 0.5 & f3 <= 0.5",
        "f2 > 0.5 & f3 <= 0.5",
        "f3 > 0.5",
        "f1 <= 0.5 & f3 > 0.5",
        "f1 > 0.5 & f3 > 0.5",
        "",
        "f0 <= 0.5",
        "f0 <= 0.5 & f1 <= 0.5",
        "f0 <= 0.5 & f1 > 0.5",
        "f0 > 0.5",
        "f0 > 0.5 & f2 <= 0.5",
        "f0 > 0.5 & f2 > 0.5",
    ]
    assert len(rules) == len(rules_str)
    for i in range(len(rules)):
        assert str(rules[i]) == str(rules_str[i])


def test_scores() -> None:
    """
    Unit test for te2rules.tree.RandomForest.get_scores()
    """
    forest_scores_on_data = [
        1.0 + 0.017 + 0.93,
        1.0 + -0.067 + 0.93,
        1.0 + -1.94 + 0.93,
        1.0 + -0.067 + 0.93,
        0.0 + 0.017 + -0.035,
        0.0 + 1.0 + -0.035,
        -2.0 + 0.017 + -1.94,
        0.0 + 1.0 + 0.082,
    ]
    forest_scores_on_data = [
        score * tree_ensemble_weights + tree_ensemble_bias
        for score in forest_scores_on_data
    ]
    forest_scores_on_data = [
        1 / (1 + 2.71828 ** (-score)) for score in forest_scores_on_data
    ]
    assert forest_scores_on_data == scores
