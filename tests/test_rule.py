"""
This file contains unit tests for te2rules.rule.Rule.
"""
from te2rules.rule import Rule

rule_1 = Rule(
    decision_rule=["f1 > 0.5"],
    decision_support=[0, 2, 5, 7],
    identity=["0_1,1_3", "1_1,2_3"],
)
rule_1.create_identity_map()


rule_2 = Rule(
    decision_rule=["f2 > 0.5"],
    decision_support=[0, 2, 4, 7],
    identity=["1_3,2_2", "0_1,1_1"],
)
rule_2.create_identity_map()


def test_str_rep() -> None:
    """
    Unit test for te2rules.rule.Rule.__str__()
    """
    assert str(rule_1) == "f1 > 0.5"
    assert str(rule_2) == "f2 > 0.5"


def test_join() -> None:
    """
    Unit test for te2rules.rule.Rule.join()
    """
    rule_3 = rule_1.join(rule_2)
    assert rule_3 is not None
    if rule_3 is not None:
        assert str(rule_3) == "f1 > 0.5 & f2 > 0.5"
        assert set(rule_3.decision_rule) == set(["f1 > 0.5", "f2 > 0.5"])
        assert set(rule_3.decision_support) == set([0, 2, 7])
        assert set(rule_3.identity) == set(["0_1,1_3,2_2"])
