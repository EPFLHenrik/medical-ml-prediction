class Node:

    def __init__(self, data, split_feature, split_value, information_gained, predict_prob, root=False, left_child=None, right_child=None):
        self.root = root
        self.data = data
        self.split_feature = split_feature
        self.split_value = split_value
        self.information_gained = information_gained
        self.left_child = left_child
        self.right_child = right_child
        self.predict_prob = predict_prob

    
