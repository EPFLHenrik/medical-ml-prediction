import numpy as np
from utils.Node import Node
import matplotlib.pyplot as plt



class Decision_Tree():

    def __init__(self, maximum_depth=5, minimum_leaves=1, 
                 splitting_criterion="entropy", stopping_criterion=0.0, 
                 splits_by_feature=5, show_split_progress=False, 
                 weighting=None,
                 split_feature_ratio=None, split_row_ratio=None):
        self.maximum_depth = maximum_depth
        self.minimum_leaves = minimum_leaves
        self.splitting_criterion = splitting_criterion
        self.stopping_criterion = stopping_criterion
        self.splits_by_feature = splits_by_feature
        self.show_split_progress = show_split_progress
        self.weighting = weighting
        self.split_feature_ratio = split_feature_ratio
        self.split_row_ratio = split_row_ratio
        # progress tracker (alternative for tqdm)
        self._split_counter = 0
        self._progress_interval = 100  # print every 100 splits

    def split(self, data, split_feature, split_value):
        if self.show_split_progress:
            self._split_counter += 1
            if self._split_counter % self._progress_interval == 0:
                print(f"[Progress] Split calls: {self._split_counter}", file=sys.stderr)

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
    
    def calc_entropy(self, y_data, eps=1e-6):
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
        # ig = self.information_gain_y(y_parent, y_left, y_right, self.entropy_y)
        ig = self.information_gain_y(y_parent, y_left, y_right, self.calc_entropy)
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
        
        best_gain = -np.inf
        best_feature = None
        best_split_val = None

        n_features = data.shape[1] - 1
        n_rows = data.shape[0]

        if self.split_feature_ratio is None or n_features == 1:
            all_features = np.arange(n_features)
        else:
            ratio = float(self.split_feature_ratio)
            k = max(1, int(round(n_features * ratio)))
            k = min(k, n_features)
            all_features = np.random.choice(n_features, size=k, replace=False)

        # optionally subsample rows to speed up split search
        if self.split_row_ratio is None or self.split_row_ratio >= 1.0:
            row_idx = np.arange(n_rows)
        else:
            r = float(self.split_row_ratio)
            k_rows = max(2, int(round(n_rows * r)))
            k_rows = min(k_rows, n_rows)
            row_idx = np.random.choice(n_rows, size=k_rows, replace=False)

        for feature in all_features:
            unique_vals = np.unique(data[row_idx, feature])
            if unique_vals.size <= 1:
                unique_vals = np.unique(data[:, feature])

            if unique_vals.size <= 1:
                splits = [unique_vals[0]]
            else:
                splits = [(unique_vals[i] + unique_vals[i+1]) / 2 for i in range(len(unique_vals)-1)]
                splits = list(splits)
                if len(splits) > 0:
                    splits = np.random.choice(splits, size=min(self.splits_by_feature, len(splits)), replace=False)

            for val in splits:
                mask = data[:, feature] >= val
                higher_vals = data[mask]
                higher_ys = higher_vals[:, -1]
                lower_vals = data[~mask]
                lower_ys = lower_vals[:, -1]

                score = score_fn(data[row_idx, -1], higher_ys, lower_ys)

                if score > best_gain:
                    best_gain = score
                    best_feature = feature
                    best_split_val = val

        return (best_gain, best_feature, best_split_val)

    def find_label_prob(self, data):
        labels = data[:, -1]

        return labels.sum()/labels.size


    def train(self, train_x, train_y):
        self.labels = np.unique(train_y)  # train y not train x
        train_y[train_y == -1] = 0
        train_data = np.concatenate((train_x, np.reshape(train_y, (-1, 1))), axis=1)

        self.tree = self.build_tree(data=train_data, current_depth=0)

    def build_tree(self, data, current_depth, ):
        if current_depth >= self.maximum_depth:
            return None
        
        split_gain, split_feature, split_val = self.find_best_split(data)
        # safeguard: if no valid split was found, create a leaf node 
        if split_feature is None:
            label_prob = self.find_label_prob(data)
            return Node(
                data,
                split_feature=None,
                split_value=None,
                information_gained=0.0,
                predict_prob=label_prob
            )
        # split data into 2 groups
        g1, g2 = self.split(data, split_feature=split_feature, split_value=split_val)

        label_prob = self.find_label_prob(data)

        node = Node(data, split_feature=split_feature, split_value=split_val, information_gained=split_gain, predict_prob=label_prob)

        if self.minimum_leaves >= g1.shape[0] or self.minimum_leaves >= g2.shape[0]:
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
    
    def print_recursive(self, node, level=0, left=0, parent=-1):
        step = 20 * (1/2)**(level) # level not -level
        if node != None: 
            # self.print_recursive(node.left_child, level-1, left+step, ([level*5,left]))  # old
            self.print_recursive(node.left_child, level+1, left+step, ([level*5,left]))
            plt.scatter([left], [level*5])
            if node.left_child is not None:
                plt.text(left, 5*level + 0.1, node.node_def(), ha="center", va="bottom", fontsize=7)
            else:
                ratio = node.data[:, -1].sum()/node.data.shape[0]
                plt.text(left, 5*level + 0.1, str(np.round(ratio, 2)), ha="center", va="bottom", fontsize=7)
            if parent != -1:
                plt.plot([left, parent[1]], [level*5, parent[0]], color="black")
            # self.print_recursive(node.right_child, level-1, left-step, ([level*5,left]))  # old
            self.print_recursive(node.right_child, level+1, left-step, ([level*5,left]))

    def print_tree(self):
        fig, ax = plt.subplots(figsize=(20, 12), dpi=100)
        self.print_recursive(node=self.tree)
        fig.savefig("figure.svg")
        plt.show()
            