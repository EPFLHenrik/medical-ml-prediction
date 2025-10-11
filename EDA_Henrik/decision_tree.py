import numpy as np
from Node import Node
class Decision_Tree():

    def __init__(self, maximum_depth=4, minimum_leaves=1, splitting_criterion="entropy", stopping_criterion=0.0):
        self.maximum_depth = maximum_depth
        self.minimum_leaves = minimum_leaves
        self.splitting_criterion = splitting_criterion
        self.stopping_criterion = stopping_criterion

    def split(self, data, split_feature, split_value):
        mask = data[:, split_feature] >= split_value

        higher_group = data[mask]
        lower_group = data[~mask]

        return higher_group, lower_group
    
    def calc_gini(self, y_data):
        n = y_data.size

        if n == 0:
            return 0
        
        p = y_data.sum() / n
        q = 1.0 - p
        return 1.0 - (p*p + q*q)
    
    def calc_entropy(self, y_data, eps=1e-12):
        n = y_data.size

        if n == 0:
            return 0
        
        p = y_data.sum() / n

        return -(p * np.log2(p + eps) + (1-p) * np.log2(1-p + eps))
    
    def information_gain_y(self, y_parent, y_left, y_right, impurity_fn):
        """IG = I(parent) - wL*I(left) - wR*I(right), using y-arrays."""
        nP = y_parent.size
        nL = y_left.size
        nR = y_right.size
        if nL == 0 or nR == 0:
            return 0.0
        imp_parent = impurity_fn(y_parent)
        imp_left   = impurity_fn(y_left)
        imp_right  = impurity_fn(y_right)
        return imp_parent - (nL/nP)*imp_left - (nR/nP)*imp_right 
    
    def split_info_y(self, y_left, y_right, eps=1e-12):
        """Intrinsic information of the split (C4.5), from y-arrays."""
        nL = y_left.size
        nR = y_right.size
        n  = nL + nR
        if n == 0:
            return 0.0
        pL = nL / n
        pR = 1.0 - pL
        si = 0.0
        if pL > 0: si -= pL * np.log2(pL + eps)
        if pR > 0: si -= pR * np.log2(pR + eps)
        return si

    def gain_ratio_y(self, y_parent, y_left, y_right):
        """Gain Ratio = InformationGain(entropy) / SplitInfo."""
        ig = self.information_gain_y(y_parent, y_left, y_right, self.entropy_y)
        si = self.split_info_y(y_left, y_right)
        if si <= 1e-12:
            return 0.0
        return ig / si


    def find_best_split(self, data):
        # choose scoring function (all operate on y arrays)
        if self.splitting_criterion == "gini":
            def score_fn(y_parent, y_left, y_right):
                return self.information_gain_y(y_parent, y_left, y_right, self.calc_gini)
        elif self.splitting_criterion == "entropy":
            def score_fn(y_parent, y_left, y_right):
                return self.information_gain_y(y_parent, y_left, y_right, self.calc_entropy)
        elif self.splitting_criterion == "gain_ratio":
            def score_fn(y_parent, y_left, y_right):
                return self.gain_ratio_y(y_parent, y_left, y_right)
        else:
            raise ValueError("criterion must be 'gini', 'entropy', or 'gain_ratio'")
        
        best_gain = -1
        best_feature = None
        best_split_val = None
        # -1 because data here includes the y-value and therefore we dont want the last value
        for feature in range(data.shape[1]-1):
            unique_vals = np.unique(data[:, feature])
            splits = [(unique_vals[i] + unique_vals[i+1])/2 for i in range(len(unique_vals)-1)]

            for val in splits:
                mask = data[:, feature] >= val
                higher_vals = data[mask]
                higher_ys = higher_vals[:, -1]
                lower_vals = data[~mask]
                lower_ys = lower_vals[:, -1]

                score = score_fn(data[:, -1], higher_ys, lower_ys)

                if score > best_gain:
                    best_gain = score
                    best_feature = feature
                    best_split_val = val

        return (best_gain, best_feature, best_split_val)

    def find_label_prob(self, data):
        labels = data[:, -1]

        return labels.sum()/labels.size


    def train(self, train_x, train_y):
        self.labels = np.unique(train_x)
        train_data = np.concatenate((train_x, np.reshape(train_y, (-1, 1))), axis=1)

        self.tree = self.build_tree(data=train_data, current_depth=0)

    def build_tree(self, data, current_depth):
        if current_depth >= self.maximum_depth:
            return None
        
        split_gain, split_feature, split_val = self.find_best_split(data)
        g1, g2 = self.split(data, split_feature=split_feature, split_value=split_val)

        label_prob = self.find_label_prob(data)

        node = Node(data, split_feature=split_feature, split_value=split_val, information_gained=split_gain, predict_prob=label_prob)

        if self.minimum_leaves > g1.shape[0] or self.minimum_leaves > g2.shape[0]:
            return node
        if split_gain <= self.stopping_criterion:
            return node
        
        current_depth += 1

        node.left_child = self.build_tree(g1, current_depth=current_depth)
        node.right_child = self.build_tree(g2, current_depth=current_depth)

        return node
        

    def predict_sample(self, sample):
        node = self.tree

        while node:
            pred_prob = node.predict_prob
            if sample[node.split_feature] >= node.split_value:
                node = node.left_child
            else:
                node = node.right_child

        return pred_prob

    def predict_proba(self, X):
        pred_probs = np.apply_along_axis(self.predict_sample, 1, X)

        return pred_probs
    
    def predict(self, X):
        pred_probs = self.predict_proba(X)

        return pred_probs.round()
    


    