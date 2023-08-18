import xgboost as xgb 
import numpy as np
import json

def get_bias(model_bundle):
    model_skl = model_bundle.model
    n0, n1 = model_skl.init_.class_prior_
    bias = np.log(n1 / n0)
    return bias

def decompose_skl_model(model_skl):
    scikit_tree_ensemble = model_skl.estimators_
    for dtr in model_skl.estimators_:
        assert len(dtr) == 1  # binary classification
    scikit_tree_ensemble = [dtr[0] for dtr in scikit_tree_ensemble]

    base_weights_list = []
    default_left_list = []
    left_children_list = []
    parents_list = []
    right_children_list = []
    split_conditions_list = []
    split_indices_list = []

    for scikit_tree in list(scikit_tree_ensemble):
        feature_indices = list(scikit_tree.tree_.feature)
        threshold = list(scikit_tree.tree_.threshold)
        children_left = list(scikit_tree.tree_.children_left)
        children_left = [int(x) for x in children_left]
        children_right = list(scikit_tree.tree_.children_right)
        children_right = [int(x) for x in children_right]

        value = list(scikit_tree.tree_.value)
        for i in range(len(scikit_tree.tree_.value)):
            assert len(scikit_tree.tree_.value[i]) == 1  # regressor
            assert len(scikit_tree.tree_.value[i][0]) == 1  # regressor
        value = [val[0][0] for val in value]

        non_leaf = [1]*len(children_left)
        for i in range(len(children_left)):
            if((children_left[i] == -1) and (children_right[i] == -1)):
                non_leaf[i] = 0

        parents = [0]*len(children_left)
        parents[0] = 2147483647
        for i in range(len(children_left)):
            if(children_left[i] >= 0):
                parents[children_left[i]] = i
        for i in range(len(children_right)):
            if(children_right[i] >= 0):
                parents[children_right[i]] = i

        base_weights_list.append(value)
        default_left_list.append(non_leaf)
        left_children_list.append(children_left)
        parents_list.append(parents)
        right_children_list.append(children_right)
        split_conditions_list.append(threshold)
        split_indices_list.append(feature_indices)
    
    return base_weights_list, default_left_list, left_children_list, parents_list, right_children_list, split_conditions_list, split_indices_list

def traverse(n, left, right):
    if(n >= 0):
        nodes = [n]
        nodes = nodes + traverse(left[n], left, right)
        nodes = nodes + traverse(right[n], left, right)
        return nodes
    else:
        return []

def dfs(n, left, right):
    if(n >= 0):
        nodes = [n]
        nodes = nodes + traverse(left[n], left, right)
        nodes = nodes + traverse(right[n], left, right)
        return nodes
    else:
        return []

def bfs(n, left, right):
    nodes = []
    queue = [n]
    while(len(queue) > 0):
        n = queue[0]
        queue = queue[1:]
        if(n >= 0):
            nodes.append(n)
            queue.append(left[n])
            queue.append(right[n])
    return nodes

def skl2xgb(model_bundle, n_estimators, max_depth, output_path):
    bias = get_bias(model_bundle)

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth, 
        objective='binary:logistic')
    clf.fit(model_bundle.x_train, model_bundle.y_train)
    
    model_xgb = clf.get_booster()  
    model_xgb.save_model(output_path + "/clf.json")

    base_weights_list, default_left_list, left_children_list, parents_list, right_children_list, split_conditions_list, split_indices_list = decompose_skl_model(model_bundle.model)
    for i in range(n_estimators):
        feature_indices = split_indices_list[i]
        for j in range(len(feature_indices)):
            if(feature_indices[j] == -2):
                feature_indices[j] = 0
            else:
                feature_indices[j] = int(feature_indices[j])
        split_indices_list[i] = feature_indices

        threshold = split_conditions_list[i]
        value = base_weights_list[i]
        for j in range(len(threshold)):
            if(threshold[j] == -2):
                threshold[j] = 0.1*value[j] + bias/n_estimators
        split_conditions_list[i] = threshold

    with open(output_path + "/clf.json", "r") as f:
        json_string = f.read()
    json_dict = json.loads(json_string)
    tree_array = json_dict["learner"]["gradient_booster"]["model"]["trees"]

    for i in range(len(tree_array)):
        tree = tree_array[i]

        traversal_bfs = bfs(0, left_children_list[i], right_children_list[i])

        node_mapping = {}
        for j in range(len(traversal_bfs)):
            node_mapping[j] = traversal_bfs[j]
        
        reverse_node_mapping = {}
        for j in range(len(traversal_bfs)):
            reverse_node_mapping[traversal_bfs[j]] = j
        #print(node_mapping)

        """
        new_nodes = traverse(0, left_children_list[i], right_children_list[i])
        old_nodes = traverse(0, tree["left_children"], tree["right_children"])
        
        node_mapping = {}
        for j in range(len(new_nodes)):
            node_mapping[old_nodes[j]] = new_nodes[j]
        print(node_mapping)
        print("---")
        """
        
        new_base_weights = list(base_weights_list[i])
        for k, v in node_mapping.items():
            new_base_weights[k] = base_weights_list[i][v]
        tree["base_weights"] = new_base_weights

        new_split_indices = list(split_indices_list[i])
        for k, v in node_mapping.items():
            new_split_indices[k] = split_indices_list[i][v]
        tree["split_indices"] = new_split_indices

        new_split_conditions = list(split_conditions_list[i])
        for k, v in node_mapping.items():
            new_split_conditions[k] = split_conditions_list[i][v]
        tree["split_conditions"] = new_split_conditions

        new_default_left = list(default_left_list[i])
        for k, v in node_mapping.items():
            new_default_left[k] = default_left_list[i][v]
        tree["default_left"] = new_default_left
        
        new_left = list(left_children_list[i])
        for j in range(len(new_left)):
            if(new_left[j] > 0):
                new_left[j] = reverse_node_mapping[new_left[j]] 
        new_left_reordered = list(new_left)
        for j in range(len(new_left)):
            new_left_reordered[reverse_node_mapping[j]] = new_left[j] 
        tree["left_children"] = new_left_reordered

        new_right = list(right_children_list[i])
        for j in range(len(new_right)):
            if(new_right[j] > 0):
                new_right[j] = reverse_node_mapping[new_right[j]] 
        new_right_reordered = list(new_right)
        for j in range(len(new_right)):
            new_right_reordered[reverse_node_mapping[j]] = new_right[j]    
        tree["right_children"] = new_right_reordered

        new_parents = [0]*len(new_left_reordered)
        new_parents[0] = 2147483647
        for i in range(len(new_left_reordered)):
            if(new_left_reordered[i] >= 0):
                new_parents[new_left_reordered[i]] = i
        for i in range(len(new_right_reordered)):
            if(new_right_reordered[i] >= 0):
                new_parents[new_right_reordered[i]] = i
        tree["parents"] = new_parents

        num_nodes = len(new_parents)
        old_num_nodes = int(tree["tree_param"]["num_nodes"])
        tree["tree_param"]["num_nodes"] = str(num_nodes)
        tree["split_type"] = [0]*num_nodes
        """
        if(old_num_nodes > num_nodes):
            tree["sum_hessian"] = tree["sum_hessian"][:num_nodes]
            tree["loss_changes"] = tree["loss_changes"][:num_nodes]
        else:
            tree["sum_hessian"] = tree["sum_hessian"] + [0.0]*(num_nodes - old_num_nodes)
            tree["loss_changes"] = tree["loss_changes"] + [0.0]*(num_nodes - old_num_nodes)
        """
        tree["sum_hessian"] = [0.0]*num_nodes        
        tree["loss_changes"] = [0.0]*num_nodes        

    with open(output_path + "/clf.json", "w") as outfile:
        json.dump(json_dict, outfile)
    
    clf = xgb.XGBClassifier()
    clf.load_model(output_path + "/clf.json")
    clf.save_model(output_path + "/clf.model")
    return
