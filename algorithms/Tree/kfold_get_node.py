from sklearn.model_selection import KFold

from .get_node import GetNode
from numpy import square, sum, mean, array as np_array


class KFoldGetNode(GetNode):
    def __init__(self, splitter, col_name, col_type, k_folds=5):
        super().__init__(splitter, col_name, col_type)
        self.k_folds = k_folds

    def calculate_fold_error(self, left_val_response, right_val_response, left_train_mean, right_train_mean) -> float:
        if self.splitter.type == 'regression':
            # error = sum of squared errors from the prediction
            left_var = sum(square(left_val_response - left_train_mean))
            right_var = sum(square(right_val_response - right_train_mean))
            return left_var + right_var
        else:
            left_n, right_n = left_val_response.size, right_val_response.size
            left_p, right_p = left_val_response.mean(), right_val_response.mean()
            return left_n * left_p * (1 - left_p) + right_n * right_p(1 - right_p)

    def get(self, x, y):
        validation_error = 0
        n_examples = x.shape[0]
        kf = KFold(n_splits=self.k_folds, shuffle=True)
        cant_do_kfold = n_examples <= self.k_folds
        nans = (None, None, None)
        if self.col_type == 'numeric':
            x_col, y_col = 0, 1
            array = self.create_sorted_array_for_numeric_col_type(x, y)
            node, indices = self._get_numeric_node_col_type_numeric(array)
            if node:
                if cant_do_kfold:
                    return node, node.split_purity, indices
                for train_index, validation_index in kf.split(array):
                    train, validation = array[train_index], array[validation_index]
                    temp_node, _ = self._get_numeric_node_col_type_numeric(train)
                    if temp_node is None:
                        return node, node.split_purity, indices
                    left_train_mean = mean(train[:, y_col][train[:, x_col] <= temp_node.thr])
                    right_train_mean = mean(train[:, y_col][train[:, x_col] > temp_node.thr])
                    left_val_response = validation[:, y_col][validation[:, x_col] <= temp_node.thr]
                    right_val_response = validation[:, y_col][validation[:, x_col] > temp_node.thr]
                    validation_error += self.calculate_fold_error(left_val_response, right_val_response,
                                                                  left_train_mean,
                                                                  right_train_mean)
                return node, validation_error, indices
            else:
                return nans

        else:
            node, indices = self._get_categorical_node_col_type_numeric(x, y)
            if node:
                if cant_do_kfold:
                    return node, node.split_purity, indices
                for train_index, validation_index in kf.split(x):
                    x_train, x_val, y_train, y_val = x[train_index], x[validation_index], y[train_index], y[
                        validation_index]
                    temp_node, temp_indices = self._get_categorical_node_col_type_numeric(x_train, y_train)
                    if temp_node is None:
                        return node, node.split_purity, indices
                    left_train_mean, right_train_mean = mean(y_train[temp_indices['left']]), mean(
                        y_train[temp_indices['right']])
                    left_val_response, right_val_response = [], []
                    for i, val in enumerate(x_val):
                        left_val_response.append(
                            y_val[i]) if val in temp_node.left_values else right_val_response.append(y_val[i])
                    validation_error += self.calculate_fold_error(np_array(left_val_response),
                                                                  np_array(right_val_response),
                                                                  left_train_mean, right_train_mean)
                return node, validation_error, indices
            else:
                return nans
