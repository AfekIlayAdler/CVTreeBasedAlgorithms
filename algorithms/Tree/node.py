class Leaf:
    def __init__(self, prediction: float, stopping_criteria: str, n_examples: str, purity: float):
        self.prediction = prediction
        self.stopping_criteria = stopping_criteria
        self.n_examples = n_examples
        self.purity = purity


class InternalNode:
    def __init__(self, n_examples, split_purity, field):
        self.n_examples = n_examples
        self.split_purity = split_purity
        self.field = field
        self.left = None
        self.right = None
        self.depth = None
        self.purity = None

    def add_depth(self, depth):
        setattr(self, 'depth', depth)

    def get_child(self, value):
        raise NotImplementedError


class NumericBinaryNode(InternalNode):
    """
    left child is smaller:
    child names are left and right
    """

    def __init__(self, n_examples, split_purity, field, splitting_point: float):
        super().__init__(n_examples, split_purity, field)
        self.thr = splitting_point

    def get_child(self, value):
        if value <= self.thr:
            return self.left
        return self.right


class CategoricalBinaryNode(InternalNode):
    """
    left child is smaller:
    child names are left and right
    """

    def __init__(self, n_examples, split_purity, field, left_values, right_values):
        super().__init__(n_examples, split_purity, field)
        self.left_values = set(left_values)
        self.right_values = set(right_values)

    def get_child(self, value):
        if value in self.left_values:
            return self.left
        return self.right
