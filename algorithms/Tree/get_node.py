from numpy import ones, square, arange, empty, array as np_array
from pandas import DataFrame

from .node import NumericBinaryNode, CategoricalBinaryNode


class GetNode:
    def __init__(self, splitter, col_name, col_type):
        self.col_name = col_name
        self.col_type = col_type
        self.splitter = splitter

    def get_numeric_col_split(self, array):
        return self.splitter.get_split(array[:, 0], array[:, 1], ones(array.shape[0])) if self.splitter.type == 'classification' \
            else self.splitter.get_split(array[:, 0], array[:, 1], array[:, 2], ones(array.shape[0]))

    @staticmethod
    def create_sorted_array_for_numeric_col_type(x, y):
        array = empty((x.shape[0], 4))
        array[:, 0], array[:, 1], array[:, 2], array[:, 3] = x, y, square(y), arange(x.shape[0])
        return array[array[:, 0].argsort()]

    def _get_numeric_node_col_type_numeric(self, array):
        """array: sorted array by the numeric column"""
        split = self.get_numeric_col_split(array)
        if split.split_index is None:
            return None, None
        thr = (array[split.split_index - 1, 0] + array[split.split_index, 0]) / 2
        left_indices = array[:split.split_index, 3]
        right_indices = array[split.split_index:, 3]
        indices = {'left': left_indices, 'right': right_indices}
        indices = {k: np_array(v).astype(int) for k, v in indices.items()}
        return NumericBinaryNode(array.shape[0], split.impurity, self.col_name, thr), indices

    def _get_categorical_node_col_type_numeric(self, x, y):
        n_examples = x.shape[0]
        array = empty((x.shape[0], 3))
        array[:, 0], array[:, 1], array[:, 2] = y, square(y), ones(x.shape[0])
        df = DataFrame(array, index=x)
        df = df.groupby(df.index).aggregate({0: 'mean', 1: 'mean', 2: 'sum'}).sort_values(0)
        if df.shape[0] == 1:  # it is a pure leaf, we can't split on this node
            return None, None
        split = self.splitter.get_split(df.index, df.loc[:, 0].values,
                                        df.loc[:, 2].values) if self.splitter.type == 'classification' \
            else self.splitter.get_split(df.index, df.loc[:, 0].values, df.loc[:, 1].values, df.loc[:, 2].values)
        left_category_values, right_category_values = df.index[:split.split_index].tolist(), df.index[
                                                                                             split.split_index:].tolist()
        left_values_set = set(left_category_values)
        left_indices, right_indices = [], []
        for i, value in enumerate(x):
            left_indices.append(i) if value in left_values_set else right_indices.append(i)
        indices = {'left': left_indices, 'right': right_indices}
        indices = {k: np_array(v).astype(int) for k, v in indices.items()}
        return CategoricalBinaryNode(n_examples, split.impurity, self.col_name, left_category_values,
                                     right_category_values), indices

    def get(self, x, y) -> tuple:
        # simple case, no cross validation score so the validation score is the purity score
        if self.col_type == 'numeric':
            array = self.create_sorted_array_for_numeric_col_type(x, y)
            node, indices = self._get_numeric_node_col_type_numeric(array)
        else:
            node, indices = self._get_categorical_node_col_type_numeric(x, y)
        if node is None:
            return None, None, None
        return node, node.split_purity, indices
