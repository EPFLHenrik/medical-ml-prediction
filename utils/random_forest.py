import numpy as np
from utils.Node import Node
from utils.decision_tree import Decision_Tree

class Random_Forest():


    def __init__(self, base_learners_num=10, max_tree_depth=5, min_samples_leaf=1, min_information_gain=0.0, \
                 split_feature_ratio=None, bootstrap_sample_size=None, \
                    splitting_criterion="entropy", \
                        splits_by_feature=5, show_split_progress=False, split_row_ratio=None) -> None:
        self.base_learners_num = base_learners_num
        self.max_tree_depth = max_tree_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.split_feature_ratio = split_feature_ratio
        self.bootstrap_sample_size = bootstrap_sample_size
        self.splitting_criterion = splitting_criterion
        self.splits_by_feature = splits_by_feature
        self.show_split_progress = show_split_progress
        self.split_row_ratio = split_row_ratio

    def create_bootstrap_samples(self, X, Y):
        bootstrapped_X = []
        bootstrapped_Y = []

        if self.bootstrap_sample_size is None:
            self.bootstrap_sample_size = X.shape[0]

        for _ in range(self.base_learners_num):
        
            samples = np.random.choice(X.shape[0], size = self.bootstrap_sample_size, replace=True)
            bootstrapped_X.append(X[samples])
            bootstrapped_Y.append(Y[samples])

        return bootstrapped_X, bootstrapped_Y
    
    def train(self, X, Y):

        bootstrapped_X, bootstrapped_Y = self.create_bootstrap_samples(X, Y)
        self.base_learners = []

        for index in range(self.base_learners_num):
            print("Training base learner ", index+1, "/", self.base_learners_num)
            dec_tree = Decision_Tree(self.max_tree_depth, self.min_samples_leaf, self.splitting_criterion, 
                                     self.min_information_gain, splits_by_feature=self.splits_by_feature,
                                     show_split_progress=self.show_split_progress,
                                     split_feature_ratio=self.split_feature_ratio, 
                                     split_row_ratio=self.split_row_ratio)
            
            dec_tree.train(bootstrapped_X[index], bootstrapped_Y[index])
            self.base_learners.append(dec_tree)

    def predict_proba_base(self, X):

        pred_list = []

        for learner in self.base_learners:
            pred_list.append(learner.predict_proba(X))

        return pred_list
    
    def predict_proba(self, X):

        pred_list = self.predict_proba_base(X)

        return np.mean(pred_list, axis=0)
    
    def predict(self, X):
        return np.round(self.predict_proba(X))