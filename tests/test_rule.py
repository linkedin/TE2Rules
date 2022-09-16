import pytest

from te2rules.rule import Rule


@pytest.fixture
def rule_1():
    rule = Rule(
        decision_rule=["f1 > 0.5"],
        decision_support=[0, 2, 5, 7],
        identity=["0_1,1_3", "1_1,2_3"],
    )
    rule.create_identity_map()
    return rule


@pytest.fixture
def rule_2():
    rule = Rule(
        decision_rule=["f2 > 0.5"],
        decision_support=[0, 2, 4, 7],
        identity=["1_3,2_2", "0_1,1_1"],
    )
    rule.create_identity_map()
    return rule


def test_str_rep(rule_1, rule_2):
    assert str(rule_1) == "f1 > 0.5"
    assert str(rule_2) == "f2 > 0.5"


def test_join(rule_1, rule_2):
    rule3 = rule_1.join(rule_2)
    assert str(rule3) == "f1 > 0.5 & f2 > 0.5"
    assert set(rule3.decision_rule) == set(["f1 > 0.5", "f2 > 0.5"])
    assert set(rule3.decision_support) == set([0, 2, 7])
    assert set(rule3.identity) == set(["0_1,1_3,2_2"])
