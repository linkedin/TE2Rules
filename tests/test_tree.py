"""
This file contains unit tests for te2rules.tree.DecisionTree.
"""
from te2rules.tree import DecisionTree, LeafNode, TreeNode

tree = DecisionTree(
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
rules = tree.get_rules(data, feature_names, tree_id=0)
scores = tree.get_scores(scores={})


def test_str_rep() -> None:
    """
    Unit test for te2rules.tree.DecisionTree.__str__()
    """
    tree_str = [
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
    assert str(tree) == "\n".join(tree_str)


def test_rules() -> None:
    """
    Unit test for te2rules.tree.DecisionTree.get_rules()
    """
    rules_str = [
        "",
        "f3 <= 0.5",
        "f2 <= 0.5 & f3 <= 0.5",
        "f2 > 0.5 & f3 <= 0.5",
        "f3 > 0.5",
        "f1 <= 0.5 & f3 > 0.5",
        "f1 > 0.5 & f3 > 0.5",
    ]
    assert len(rules) == len(rules_str)
    for i in range(len(rules)):
        assert str(rules[i]) == str(rules_str[i])


def test_scores() -> None:
    """
    Unit test for te2rules.tree.DecisionTree.get_scores()
    """
    tree_scores_on_data = {
        0: 0.017,
        1: -0.067,
        2: -1.94,
        3: -0.067,
        4: 0.017,
        5: 1.0,
        6: 0.017,
        7: 1.0,
    }
    assert tree_scores_on_data == scores
