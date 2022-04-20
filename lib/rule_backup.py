class Rule:
  def __init__(self, decision_rule, decision_support, identity):
    self.decision_rule = decision_rule
    self.decision_support = decision_support
    self.identity = identity

  def __str__(self):
    string_rep = "rule: " + str(self.decision_rule) + '\n'
    string_rep = string_rep + "support: " + str(len(self.decision_support)) + '\n'
    string_rep = string_rep + "identity: " + str(self.identity) + '\n'
    return string_rep 

  def create_identity_map(self):
    left_identity_map = {}
    right_identity_map = {}
    for i in range(len(self.identity)):
      tn = self.identity[i].split(",")
      tn = sorted(tn)

      left_key = ",".join(tn[1:])
      if(left_key not in left_identity_map):
        left_identity_map[left_key] = []
      left_identity_map[left_key].append(self.identity[i])
      
      right_key = ",".join(tn[:-1])
      if(right_key not in right_identity_map):
        right_identity_map[right_key] = []
      right_identity_map[right_key].append(self.identity[i])

    self.left_identity_map = left_identity_map
    self.right_identity_map = right_identity_map
    return

  def get_num_nodes(self):
    num_nodes = len(self.identity[0].split(","))
    '''
    for i in range(len(self.identity)):
      assert(num_nodes == len(self.identity[i].split(",")))
    '''
    return num_nodes

    
  def join_identity(self, rule):
    joined_identity = []
    num_nodes = self.get_num_nodes()
    # This can be sped up instead of brute force join
    left_keys = self.left_identity_map.keys()
    right_keys = rule.right_identity_map.keys()
    keys = list(set(left_keys).intersection(set(right_keys)))
    for key in keys:
      for i in range(len(self.left_identity_map[key])):
        for j in range(len(rule.right_identity_map[key])):
          identity_left = self.left_identity_map[key][i].split(",")
          identity_right = rule.right_identity_map[key][j].split(",")
          identity = sorted(list(set(identity_left).union(set(identity_right))))

          tree_id_left = [tn.split("_")[0] for tn in identity_left]
          tree_id_right = [tn.split("_")[0] for tn in identity_right]
          tree_id = sorted(list(set(tree_id_left).union(set(tree_id_right))))
          
          if((len(identity) == num_nodes + 1) and (len(tree_id) == num_nodes + 1) and 
            (min(tree_id_left) < min(tree_id_right)) and (max(tree_id_left) < max(tree_id_right))):
            identity = ",".join(identity)
            joined_identity.append(identity)
    
    #assert(len(list(set(joined_identity))) == len(joined_identity))
    return list(set(joined_identity))

  def validate_identity(self, joined_identity):
    if(len(joined_identity) > 0):
      return True
    else:
      return False

  def join_rule(self, rule):
    decision_rule = list(set(self.decision_rule).union(set(rule.decision_rule)))
    return sorted(decision_rule)

  def validate_rule(self, joined_rule):
    # This assumes data is one hot encoded: atmost one positive value per feature and
    # the same value cannot be both positive and negative
    is_valid = True
    feature_name = None
    feature_name_values = []
    for clause in joined_rule:
      if(clause.split("_")[0] != feature_name):
        feature_name = clause.split("_")[0]
        feature_name_values.append([])
      feature_name_values[-1].append(clause)
    
    for fnv in feature_name_values:
      positive = []
      negative = []
      for f in fnv:
        if(f[-6:] == '<= 0.5'):
          negative.append(f)
        else:
          positive.append(f)
      if(len(positive) > 1):
        is_valid = False
      if((len(positive) == 1) and (positive[0][:-5] + "<= 0.5" in negative)):
        is_valid = False
    
    return is_valid

  def join_support(self, rule):
    decision_support = list(set(self.decision_support).intersection(set(rule.decision_support)))
    return decision_support

  def validate_support(self, joined_support, support):
    extra_support = list(set(joined_support).difference(set(support)))
    if((len(joined_support) > 0) and (len(extra_support) > 0)):
      return True
    else:
      return False

  def join(self, rule, support):
    assert(self.get_num_nodes() == rule.get_num_nodes())
              
    decision_rule = self.join_rule(rule)
    if(self.validate_rule(decision_rule) == False):
      return None

    decision_support = self.join_support(rule)
    if(self.validate_support(decision_support, support) == False):
      return None

    identity = self.join_identity(rule)
    if(self.validate_identity(identity) == False):
      return None

    return Rule(decision_rule=decision_rule, decision_support=decision_support, identity=identity)

    
