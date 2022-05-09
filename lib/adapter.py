from sklearn.tree import _tree
from lib.tree import RandomForest, DecisionTree, TreeNode, LeafNode
import numpy as np

class ScikitForestAdapter:
  def __init__(self, scikit_forest, feature_names):
    n0 , n1 = scikit_forest.init_.class_prior_
    self.bias = np.log(n1/n0)    
    self.weight = scikit_forest.get_params()['learning_rate']

    scikit_tree_ensemble = scikit_forest.estimators_
    scikit_tree_ensemble = [dtr[0] for dtr in scikit_tree_ensemble]
    self.scikit_tree_ensemble = scikit_tree_ensemble
    self.feature_names = feature_names

    self.random_forest = self.convert() 

  def convert(self):
    decision_tree_ensemble = []
    for scikit_tree in list(self.scikit_tree_ensemble):
      decision_tree = ScikitTreeAdapter(scikit_tree, self.feature_names).decision_tree
      decision_tree_ensemble.append(decision_tree)
    
    return RandomForest(decision_tree_ensemble, weight=self.weight, bias=self.bias)

class ScikitTreeAdapter:
  def __init__(self, scikit_tree, feature_names):
    self.feature_names = feature_names
    
    self.feature_indices = scikit_tree.tree_.feature
    self.threshold = scikit_tree.tree_.threshold
    self.children_left = scikit_tree.tree_.children_left
    self.children_right = scikit_tree.tree_.children_right
    self.value = scikit_tree.tree_.value
    self.value = [val[0][0] for val in self.value]
    
    self.LEAF_INDEX = _tree.TREE_UNDEFINED
    
    self.decision_tree = self.convert()

  def convert(self):
    nodes = []

    # Create Tree Nodes
    for i in range(len(self.feature_indices)):
      node_index = self.feature_indices[i]
      if(node_index != self.LEAF_INDEX):
        node_name = self.feature_names[node_index]
        nodes = nodes + [DecisionTree(TreeNode(node_name = node_name, threshold = self.threshold[i]))]
      else:
        value = self.value[i]
        nodes = nodes + [DecisionTree(LeafNode(value = value))]

    # Connect Tree Nodes with each other
    for i in range(len(self.feature_indices)):
      node_index = self.feature_indices[i]
      if(node_index != self.LEAF_INDEX):
        left_node = nodes[self.children_left[i]]
        nodes[i].left = left_node
        right_node = nodes[self.children_right[i]]
        nodes[i].right = right_node

    root_node = nodes[0]
    return root_node