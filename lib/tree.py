from lib.rule import Rule

class Node:
  def __init__(self, is_leaf):
    self.is_leaf = is_leaf

class TreeNode(Node):
  def __init__(self, node_name, threshold):
    super().__init__(is_leaf=False)
    self.node_name = node_name
    self.threshold = threshold

  def __str__(self):
    return self.node_name

  def get_left_clause(self):
    return self.node_name + " <= " + str(self.threshold)

  def get_right_clause(self):
    return self.node_name + " > " + str(self.threshold)


class LeafNode(Node):
  def __init__(self, value):
    super().__init__(is_leaf=True)
    self.value = value

  def __str__(self):
    return str(self.value)

  def get_leaf_clause(self):
    return "value: " + str(self.value)
      
class DecisionTree:
  def __init__(self, node, left = None, right = None):
    self.node = node
    self.left = left
    self.right = right
    
  def __str__(self, num_indent = 0):
    indentation = "|   " * num_indent + "|---"
    if(self.node.is_leaf == True):
      string_rep =  indentation + self.node.get_leaf_clause()
      if(100 in self.decision_support):
        string_rep = string_rep + "-----100 here-----"
    else:
      string_rep  =  indentation + self.node.get_left_clause() + "\n" 
      string_rep +=  self.left.__str__(num_indent=num_indent + 1) + "\n"     
      string_rep +=  indentation + self.node.get_right_clause() + "\n"       
      string_rep +=  self.right.__str__(num_indent=num_indent + 1)
    
    return string_rep

  def propagate_decision_rule(self, decision_rule = []):
    self.decision_rule = decision_rule
    if(self.node.is_leaf == False):
      left_decision_rule = decision_rule + [self.node.get_left_clause()]
      self.left.propagate_decision_rule(left_decision_rule) 
      right_decision_rule = decision_rule + [self.node.get_right_clause()]
      self.right.propagate_decision_rule(right_decision_rule) 

  def aggregate_min_decision_value(self):
    if(self.node.is_leaf == False):
      self.left.aggregate_min_decision_value()
      self.right.aggregate_min_decision_value()
      self.min_decision_value = min(self.left.min_decision_value, self.right.min_decision_value)
    else:
      self.min_decision_value = self.node.value

  def aggregate_max_decision_value(self):
    if(self.node.is_leaf == False):
      self.left.aggregate_max_decision_value()
      self.right.aggregate_max_decision_value()
      self.max_decision_value = max(self.left.max_decision_value, self.right.max_decision_value)
    else:
      self.max_decision_value = self.node.value

  def propagate_decision_support(self, data, feature_names, decision_support=None):
    if(decision_support==None):
      decision_support = [i for i in range(len(data))]
    self.decision_support = decision_support

    if(self.node.is_leaf == False):
      feature_index = feature_names.index(self.node.node_name)
      left_decision_support = []
      right_decision_support = []
      for index in self.decision_support:
        if(data[index][feature_index] <= self.node.threshold):
          left_decision_support.append(index)
        else:
          right_decision_support.append(index)
      self.left.propagate_decision_support(data, feature_names, left_decision_support)
      self.right.propagate_decision_support(data, feature_names, right_decision_support)

  def get_rules(self, data, feature_names, tree_id):
    self.aggregate_min_decision_value()
    self.aggregate_max_decision_value()
    self.propagate_decision_rule()
    self.propagate_decision_support(data, feature_names)
    rules = self.collect_rules(tree_id=tree_id, node_id=0, rules=[])
    return rules

  def collect_rules(self, tree_id, node_id, rules):
    rules = rules + [Rule(
      decision_rule=sorted(self.decision_rule), 
      decision_support=self.decision_support, 
      identity=[str(tree_id) + "_" + str(node_id)]
      )]
    if(self.node.is_leaf == False):
      rules = self.left.collect_rules(tree_id, 2*node_id + 1, rules)
      rules = self.right.collect_rules(tree_id, 2*node_id + 2, rules)
    return rules

  def get_rule_score(self, rule):
    #print(self.decision_rule)
    if(self.node.is_leaf == False):    
      if(self.node.get_left_clause() in rule):
        return self.left.get_rule_score(rule)
      
      if(self.node.get_right_clause() in rule):
        return self.right.get_rule_score(rule)
    
    return [self.min_decision_value, self.max_decision_value]


class RandomForest:
  def __init__(self, decision_tree_ensemble):
    self.decision_tree_ensemble = decision_tree_ensemble

  def get_num_trees(self):
    return len(self.decision_tree_ensemble)

  def get_rules(self, data, feature_names):
    rules_from_tree = []
    for tree_index, decision_tree in enumerate(self.decision_tree_ensemble):
      rules = decision_tree.get_rules(data=data, feature_names=feature_names, tree_id=tree_index)
      rules_from_tree = rules_from_tree + rules
    return rules_from_tree

  def get_rule_score(self, rule):
    min_score = 0
    max_score = 0
    for decision_tree in self.decision_tree_ensemble:
      score = decision_tree.get_rule_score(rule)
      min_score = min_score + score[0]
      max_score = max_score + score[1]
    return [min_score, max_score]  

  def __str__(self):
    string_rep = ""
    for i in range(len(self.decision_tree_ensemble)):
      string_rep = string_rep + "Tree " + str(i) + "\n"
      string_rep = string_rep + self.decision_tree_ensemble[i].__str__() + "\n\n"
    return string_rep









