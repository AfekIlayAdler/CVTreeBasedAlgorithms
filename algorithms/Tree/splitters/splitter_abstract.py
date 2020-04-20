

class Splitter:
    def __init__(self, response_variable_type, min_samples_leaf):
        self.type = response_variable_type
        self.min_samples_leaf = min_samples_leaf


class Split:
    def __init__(self, split_index: int, split_impurity: float):
        self.impurity = split_impurity
        self.split_index = split_index
