import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree #, export_graphviz
# import graphviz 
from collections import defaultdict
from standard_utils import *

# Extension of scikit-learn's DecisionTreeClassifier
class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    # Learn decision tree using features 'train_data' and labels 'train_labels'
    def fit_tree(self, train_data, train_labels):
        num_true = np.sum(train_labels)
        num_false = np.sum(np.logical_not(train_labels))
        if self.class_weight == "balanced":
            self.float_class_weight = num_false/num_true
        elif isinstance(self.class_weight, dict):
            keys_list = list(self.class_weight.keys())
            assert len(keys_list)==2
            assert 0 in keys_list
            assert 1 in keys_list
            self.float_class_weight = self.class_weight[1]

        self.fit(train_data, train_labels)
        true_dict, false_dict = self.compute_TF_dict(train_data, train_labels)
        self.train_true_dict = dict(true_dict)
        self.train_false_dict = dict(false_dict)
        
        self._compute_parent()
        
        true_array = np.array(list(true_dict))
        false_array = np.array(list(false_dict))
        unique_leaf_ids = np.union1d(true_array, false_array)
        self.leaf_ids = unique_leaf_ids
        
        true_leaves = []
        
        for leaf_id in unique_leaf_ids:
            true_count = true_dict[leaf_id]
            false_count = false_dict[leaf_id]
            if true_count*self.float_class_weight > false_count:
                true_leaves.append(leaf_id)
        self.true_leaves = true_leaves
        return self
    
    # Find the parent of every leaf node
    def _compute_parent(self):
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right

        self.parent = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [0]  
        while len(stack) > 0:
            node_id = stack.pop()

            child_left = children_left[node_id]
            child_right = children_right[node_id]
            if (child_left != child_right):
                self.parent[child_left] = node_id
                self.parent[child_right] = node_id
                stack.append(child_left)
                stack.append(child_right)
    
    # Find which of the data points lands in the leaf node with identifier 'leaf_id'
    def compute_leaf_data(self, data, leaf_id):
        leaf_ids = self.apply(data)
        return np.nonzero(leaf_ids==leaf_id)[0]
    
    # Find which of the data points lands in the leaf node with identifier 'leaf_id' and for which the 
    # prediction is 'true'. 
    def compute_leaf_truedata(self, data, labels, leaf_id):
        leaf_ids = self.apply(data)
        leaf_data_indices = np.nonzero(leaf_ids==leaf_id)[0]
        leaf_failure_labels = labels[leaf_data_indices]
        leaf_failure_indices = leaf_data_indices[leaf_failure_labels]
        return leaf_failure_indices
        
    # Returns two dictionaries 'true_dict' and 'false_dict'. 
    # true_dict maps every leaf_id to the number of correctly classified 
    # data points in the leaf with that leaf_id.
    # false_dict maps every leaf_id to the number of incorrectly classified 
    # data points in the leaf with that leaf_id.
    def compute_TF_dict(self, data, labels):
        leaf_ids = self.apply(data)
        true_leaf_ids = leaf_ids[np.nonzero(labels)]
        false_leaf_ids = leaf_ids[np.nonzero(np.logical_not(labels))]
        
        true_unique, _, true_unique_counts = np.unique(true_leaf_ids, 
                                                       return_index=True, 
                                                       return_counts=True)
        true_dict = create_dict(true_unique, true_unique_counts)
        false_unique, _, false_unique_counts = np.unique(false_leaf_ids, 
                                                         return_index=True, 
                                                         return_counts=True)
        false_dict = create_dict(false_unique, false_unique_counts)
        return true_dict, false_dict
    
    # Compute precision and recall for the tree. Also compute 
    # Average Leaf Error Rate if compute_ALER is True.
    def compute_precision_recall(self, data, labels, compute_ALER=True):
        true_dict, false_dict = self.compute_TF_dict(data, labels)
        total_true = np.sum(labels)
        total_pred = 0
        total = 0
        for leaf_id in self.true_leaves:
            true_count = true_dict[leaf_id]
            false_count = false_dict[leaf_id]

            total_pred += true_count
            total += true_count + false_count
            
        precision = total_pred/total
        recall = total_pred/total_true
        
        if compute_ALER:
            average_precision = self.compute_average_leaf_error_rate(data, labels)
            return precision, recall, average_precision
        else:
            return precision, recall
    
    # Compute Average Leaf Error Rate using the trained decision tree
    def compute_average_leaf_error_rate(self, data, labels):
        num_true = np.sum(labels)
        true_dict, false_dict = self.compute_TF_dict(data, labels)
    
        avg_leaf_error_rate = 0
        for leaf_id in self.leaf_ids:
            true_count = true_dict[leaf_id]
            false_count = false_dict[leaf_id]
            if true_count + false_count > 0:
                curr_error_coverage = true_count/num_true
                curr_error_rate = true_count/(true_count + false_count)

                avg_leaf_error_rate += curr_error_coverage*curr_error_rate
        return avg_leaf_error_rate

    # Compute decision_path (the set of decisions used to arrive at
    # a certain leaf)
    def compute_decision_path(self, leaf_id, important_features_indices=None):
        assert leaf_id in self.leaf_ids

        features_arr = self.tree_.feature
        thresholds_arr = self.tree_.threshold
        
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        path = []
        curr_node = leaf_id
        while curr_node > 0:
            parent_node = self.parent[curr_node]
            
            is_left_child = (children_left[parent_node] == curr_node)
            is_right_child = (children_right[parent_node] == curr_node)
            assert (is_left_child ^ is_right_child)

            if is_left_child:
                direction = 'left'
            else:
                direction = 'right'
            curr_node = parent_node
            curr_feature = features_arr[curr_node]
            curr_threshold = np.round(thresholds_arr[curr_node], 6)
            if important_features_indices is not None:
                curr_feature_original = important_features_indices[curr_feature]
            else:
                curr_feature_original = curr_feature
            path.insert(0, (curr_node, curr_feature_original, curr_threshold, direction))
        return path
    
    # Compute error rate and error coverage for every node in the tree.
    def compute_leaf_error_rate_coverage(self, data, labels):
        total_failures = np.sum(labels)
        
        true_dict, false_dict = self.compute_TF_dict(data, labels)

        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right

        error_rate_array = np.zeros(shape=n_nodes, dtype=float)
        error_coverage_array = np.zeros(shape=n_nodes, dtype=float)
        
        stack = [(0, True)]
        while len(stack) > 0:
            node_id, traverse = stack.pop()
            child_left = children_left[node_id]
            child_right = children_right[node_id]

            if traverse:
                if (child_left != child_right):
                    stack.append((node_id, False))
                    stack.append((child_left, True))
                    stack.append((child_right, True))
                else:
                    num_true_in_node = true_dict[node_id]
                    num_false_in_node = false_dict[node_id]
                    num_total_in_node = num_true_in_node + num_false_in_node

                    if num_total_in_node > 0:
                        leaf_error_rate = (num_true_in_node/num_total_in_node)
                    else:
                        leaf_error_rate = 0.
                    leaf_error_coverage = (num_true_in_node/total_failures)
                    error_coverage_array[node_id] = leaf_error_coverage
                    error_rate_array[node_id] = leaf_error_rate
            else:
                child_left_ER = error_rate_array[child_left]
                child_right_ER = error_rate_array[child_right]
                
                child_left_EC = error_coverage_array[child_left]
                child_right_EC = error_coverage_array[child_right]
                
                child_ER = child_left_ER*child_left_EC + child_right_ER*child_right_EC
                child_EC = child_left_EC + child_right_EC

                if child_EC > 0:
                    error_rate_array[node_id] = child_ER/child_EC
                else:
                    error_rate_array[node_id] = 0.
                error_coverage_array[node_id] = child_EC

        return error_rate_array, error_coverage_array
    
# Train decision tree
def train_decision_tree(train_sparse_features, train_failure, max_depth=1, criterion="entropy"):
    num_true = np.sum(train_failure)
    num_false = np.sum(np.logical_not(train_failure))
    rel_weight = num_false/num_true
    class_weight_dict = {0: 1, 1: rel_weight}

    decision_tree = CustomDecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, class_weight=class_weight_dict)
    decision_tree.fit_tree(
        train_sparse_features, train_failure)
    return decision_tree

# Select leaf nodes with highest importance value i.e highest contribution to average leaf error rate
def important_leaf_nodes(decision_tree, precision_array, recall_array):
    leaf_ids = decision_tree.leaf_ids
    leaf_precision = precision_array[leaf_ids]
    leaf_recall = recall_array[leaf_ids]
    leaf_precision_recall = leaf_precision*leaf_recall

    important_leaves = np.argsort(-leaf_precision_recall)
    return leaf_ids[important_leaves]