class Rule:
    def __init__(self, decision_rule, decision_support, identity):
        self.decision_rule = decision_rule
        self.decision_support = decision_support
        self.identity = identity

        if len(self.identity) == 0:
            raise ValueError(
                "Identity list contains the possible k-tree node combinations "
                + "that produce this rule. Identity list cannot be empty"
            )

    def __str__(self):
        string_rep = " & ".join(self.decision_rule)
        return string_rep

    def create_identity_map(self):
        num_nodes = len(self.identity[0].split(","))
        for i in range(len(self.identity)):
            node_ids = self.identity[i].split(",")
            if num_nodes != len(node_ids):
                raise ValueError(
                    "Identity list contains the possible k-tree node combinations "
                    + "that produce this rule. Entries in the list cannot contain "
                    + "unequal number of contributing tree nodes."
                )

        left_identity_map = {}
        right_identity_map = {}
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

    def join_identity(self, rule):
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

    def validate_identity(self, joined_identity):
        if len(joined_identity) > 0:
            return True
        else:
            return False

    def join_rule(self, rule):
        decision_rule = list(set(self.decision_rule).union(set(rule.decision_rule)))
        return sorted(decision_rule)

    def join_support(self, rule):
        decision_support = list(
            set(self.decision_support).intersection(set(rule.decision_support))
        )
        return decision_support

    def validate_support(self, joined_support):
        if len(joined_support) > 0:
            return True
        else:
            return False

    def join(self, rule, support_pruning=False):
        decision_rule = self.join_rule(rule)

        decision_support = self.join_support(rule)
        if support_pruning is True:
            if self.validate_support(decision_support) is False:
                return None

        identity = self.join_identity(rule)
        if self.validate_identity(identity) is False:
            return None

        return Rule(
            decision_rule=decision_rule,
            decision_support=decision_support,
            identity=identity,
        )
