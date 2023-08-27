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

        base_weights_list.append(value)
        default_left_list.append(non_leaf)
        left_children_list.append(children_left)
        right_children_list.append(children_right)
        split_conditions_list.append(threshold)
        split_indices_list.append(feature_indices)

    return base_weights_list, default_left_list, left_children_list, right_children_list, split_conditions_list, split_indices_list

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

def reorder(values, mapping):
    reordered_values = list(values)
    for k, v in mapping.items():
        reordered_values[k] = values[v]
    return reordered_values

def reindex(indices, mapping):
    remapped_indices = list(indices)
    for j in range(len(remapped_indices)):
        if(remapped_indices[j] > 0):
            remapped_indices[j] = mapping[remapped_indices[j]]
    return remapped_indices

def get_parents(left, right):
    parents = [0]*len(left)
    parents[0] = 2147483647
    for j in range(len(left)):
        if(left[j] >= 0):
            parents[left[j]] = j
    for j in range(len(right)):
        if(right[j] >= 0):
            parents[right[j]] = j
    return parents

def update_xgb_tree(tree, base_weights, default_left,
                    left_children, right_children,
                    split_conditions, split_indices):
    # reindex the tree nodes
    # xgb models are indexed in a bfs order
    traversal_bfs = bfs(0, left_children, right_children)
    node_mapping = {}
    for j in range(len(traversal_bfs)):
        node_mapping[j] = traversal_bfs[j]

    reverse_node_mapping = {}
    for j in range(len(traversal_bfs)):
        reverse_node_mapping[traversal_bfs[j]] = j

    # reindex the tree nodes:
    # base weights, split indices, split conditions, default left
    # reorder the lists
    tree["base_weights"] = reorder(list(base_weights), node_mapping)
    tree["split_indices"] = reorder(list(split_indices), node_mapping)
    tree["split_conditions"] = reorder(list(split_conditions), node_mapping)
    tree["default_left"] = reorder(list(default_left), node_mapping)

    tree["left_children"] = reorder(list(left_children), node_mapping)
    tree["right_children"] = reorder(list(right_children), node_mapping)

    # reindex the tree nodes:
    # left child, right child
    # reorder the lists and remap the entries in the list
    tree["left_children"] = reindex(list(tree["left_children"]), reverse_node_mapping)
    tree["right_children"] = reindex(list(tree["right_children"]), reverse_node_mapping)

    # create parents list from left and right children
    tree["parents"] = get_parents(list(tree["left_children"]), list(tree["right_children"]))

    # set the correct number of nodes
    num_nodes = len(tree["parents"])
    tree["tree_param"]["num_nodes"] = str(num_nodes)
    tree["split_type"] = [0]*num_nodes
    tree["sum_hessian"] = [0.0]*num_nodes
    tree["loss_changes"] = [0.0]*num_nodes
    return

def skl2xgb(model_bundle, n_estimators, max_depth, output_path):
    # get bias term from scikit-learn model
    # xgb library doesnt use a bias term
    bias = get_bias(model_bundle)

    # get learning rate from scikit model (default: 0.1)
    # in scikit-learn, learning rate acts as a multiplier to leaf values
    learning_rate = model_bundle.model.get_params()["learning_rate"]

    # get tree parameters of xgb model from scikit-learn model
    base_weights_list, default_left_list, left_children_list, right_children_list, split_conditions_list, split_indices_list = decompose_skl_model(model_bundle.model)
    # adjust for leaf nodes:
    # feature to be split is invalid and is represented differently in the 2 libs
    # in xgb, leaf values are used as it is, while in scikit-learn, it is scaled by learning rate
    # distribute bias across leaf values, since there is no bias term in xgb
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
                threshold[j] = learning_rate*value[j] + bias/n_estimators
        split_conditions_list[i] = threshold

    # train a dummy xgb model
    # the model's learnt parameters will be over-written later
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        objective='binary:logistic')
    clf.fit(model_bundle.x_train, model_bundle.y_train)

    model_xgb = clf.get_booster()
    model_xgb.save_model(output_path + "/clf.json")

    with open(output_path + "/clf.json", "r") as f:
        json_string = f.read()
    json_dict = json.loads(json_string)
    tree_array = json_dict["learner"]["gradient_booster"]["model"]["trees"]

    # update tree parameters
    for i in range(len(tree_array)):
        update_xgb_tree(tree_array[i], base_weights_list[i], default_left_list[i],
                    left_children_list[i], right_children_list[i],
                    split_conditions_list[i], split_indices_list[i])

    with open(output_path + "/clf.json", "w") as outfile:
        json.dump(json_dict, outfile)

    # save the updated model
    clf = xgb.XGBClassifier()
    clf.load_model(output_path + "/clf.json")
    clf.save_model(output_path + "/clf.model")
    return
