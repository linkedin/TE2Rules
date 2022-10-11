"""
This file defines the rule object used in explaining te2rules tree ensemble models.
"""
from __future__ import annotations

from typing import Dict, List, Optional


class Rule:
    """
    This class defines the rule object used in explaining te2rules tree ensemble models.
    The rule object consists of:
    1) Terms: A list of terms whose conjunction gives us the rule.
        The terms are sorted by alphabetical order.
    2) Support: A list of indices of data records satisfying the rule.
    3) Identity: A list of k-tree node combinations that result in the rule.
        Each tree node is represented by tree_id + underscore + node_id.
        Example tree node: "2_4" (2nd tree's 4th node).

        Each tree node combination is represented as a string of k tree nodes
        sorted by tree_id, node_id.
        Example tree node combination: "0_1,2_1,3_0".

        Example identity: ["0_1,1_1,2_0", "0_1,1_3,2_1", "1_1,2_3,3_0", "1_3,2_2,3_1"]
    """

    def __init__(
        self, decision_rule: List[str], decision_support: List[int], identity: List[str]
    ):
        self.decision_rule = decision_rule
        self.decision_support = decision_support
        self.identity = identity

        if len(self.identity) == 0:
            raise ValueError(
                "Identity list contains the possible k-tree node combinations "
                + "that produce this rule. Identity list cannot be empty"
            )

    def __str__(self) -> str:
        """
        Method to return the string represenation of the rule.
        """
        string_rep = " & ".join(self.decision_rule)
        return string_rep

    def create_identity_map(self) -> None:
        """
        This method creates two new dicts (left_identity_map and right_identity_map)
        from the identity. These dicts are used by the join_identity method.

        Identity is a list of tree node combinations that result in the rule.
            a) Each tree node is represented by tree_id + underscore + node_id.
            Example tree node: "2_4" (2nd tree's 4th node).

            b) Each tree node combination is represented as a string of k tree nodes
            sorted by tree_id, node_id.
            Example tree node combination: "0_1,2_1,3_0".

            c) Example identity:
            ["0_1,1_1,2_0", "0_1,1_3,2_1", "1_1,2_3,3_0", "1_3,2_2,3_1"]

        The left_identity_map is a map of tree node combination suffix (last k-1
        tree nodes) to the first tree node.
        The right_identity_map is a map of tree node combination prefix (first k-1
        tree nodes) to the last tree node.

        Example: For a rule with
        identity = ["0_1,1_1,2_0", "0_1,1_3,2_1", "1_1,2_3,3_0", "1_3,2_2,3_1"],
        this method creates:
            left_identity_map = {
                "1_1,2_0": ["0_1"],
                "1_3,2_1": ["0_1"],
                "2_3,3_0": ["1_1"],
                "2_2,3_1": ["1_3"],
            }
            right_identity_map = {
                "0_1,1_1": ["2_0"],
                "0_1,1_3": ["2_1"],
                "1_1,2_3": ["3_0"],
                "1_3,2_2": ["3_1"],
            }
        """
        num_nodes = len(self.identity[0].split(","))
        for i in range(len(self.identity)):
            node_ids = self.identity[i].split(",")
            if num_nodes != len(node_ids):
                raise ValueError(
                    "Identity list contains the possible k-tree node combinations "
                    + "that produce this rule. Entries in the list cannot contain "
                    + "unequal number of contributing tree nodes."
                )

        left_identity_map: Dict[str, List[str]] = {}
        right_identity_map: Dict[str, List[str]] = {}
        for i in range(len(self.identity)):
            node_ids = self.identity[i].split(",")

            left_key = ",".join(node_ids[1:])
            if left_key not in left_identity_map:
                left_identity_map[left_key] = []
            left_identity_map[left_key].append(node_ids[0])

            right_key = ",".join(node_ids[:-1])
            if right_key not in right_identity_map:
                right_identity_map[right_key] = []
            right_identity_map[right_key].append(node_ids[-1])

        self.left_identity_map = left_identity_map
        self.right_identity_map = right_identity_map
        return

    def _join_identity(self, rule: Rule) -> List[str]:
        """
        This method creates the identity of a new rule resulting from
        joining (or conjunction of) the existing rule with another rule.

        The resulting rule's identity consists of a list of k+1 tree node
        combinations such that their prefix of first k tree nodes comes from
        the first (existing) rule and their suffix of last k tree nodes comes
        from the second rule.

        Example:
        identity of first rule: ["0_1,1_1,2_0", "0_1,1_3,2_1"]
        identity of second rule: ["1_1,2_0,3_0", "1_3,2_1,3_1"]

        identity of conjunction of first and second rule:
        ["0_1,1_1,2_0,3_0", 0_1,1_3,2_1,3_1"]
        """
        if not hasattr(self, "left_identity_map") or not hasattr(
            rule, "right_identity_map"
        ):
            raise AttributeError(
                "left_identity_map and  right_identity_map attributes are not set. "
                + "Call create_identity_map() on both rules before joining"
            )

        joined_identity = []
        left_keys = self.left_identity_map.keys()
        right_keys = rule.right_identity_map.keys()
        keys = list(set(left_keys).intersection(set(right_keys)))
        for key in keys:
            for i in range(len(self.left_identity_map[key])):
                for j in range(len(rule.right_identity_map[key])):
                    if key == "":
                        left_tree_id = int(self.left_identity_map[key][i].split("_")[0])
                        right_tree_id = int(
                            rule.right_identity_map[key][j].split("_")[0]
                        )
                        if left_tree_id < right_tree_id:
                            identity = (
                                self.left_identity_map[key][i]
                                + ","
                                + rule.right_identity_map[key][j]
                            )
                            joined_identity.append(identity)
                    else:
                        identity = (
                            self.left_identity_map[key][i]
                            + ","
                            + key
                            + ","
                            + rule.right_identity_map[key][j]
                        )
                        joined_identity.append(identity)

        return list(set(joined_identity))

    def _validate_identity(self, joined_identity: List[str]) -> bool:
        """
        This method validates that the given identity list is non-empty.
        """
        if len(joined_identity) > 0:
            return True
        else:
            return False

    def _join_rule(self, rule: Rule) -> List[str]:
        """
        This method creates the terms of a new rule resulting from
        joining (or conjunction of) the existing rule with another rule.

        The resulting rule's term consists of terms present in either the
        first rule or the second rule or both.
        """
        decision_rule = list(set(self.decision_rule).union(set(rule.decision_rule)))
        return sorted(decision_rule)

    def _join_support(self, rule: Rule) -> List[int]:
        """
        This method creates the support of a new rule resulting from
        joining (or conjunction of) the existing rule with another rule.

        The resulting rule's support consists of common support indices from
        both the first rule and the second rule.
        """
        decision_support = list(
            set(self.decision_support).intersection(set(rule.decision_support))
        )
        return decision_support

    def _validate_support(self, joined_support: List[int]) -> bool:
        """
        This method validates that the given support list is non-empty.
        """
        if len(joined_support) > 0:
            return True
        else:
            return False

    def join(self, rule: Rule) -> Optional[Rule]:
        """
        This method creates the new rule resulting from
        joining (or conjunction of) the existing rule with another rule.

        If such a rule is infeasible because of lack of a valid support in data
        or a lack of appropriate supporting tree nodes in the model, then the
        method returns None.
        """
        decision_rule = self._join_rule(rule)

        decision_support = self._join_support(rule)
        if self._validate_support(decision_support) is False:
            return None

        identity = self._join_identity(rule)
        if self._validate_identity(identity) is False:
            return None

        return Rule(
            decision_rule=decision_rule,
            decision_support=decision_support,
            identity=identity,
        )
