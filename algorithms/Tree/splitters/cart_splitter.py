import numpy as np

from .splitter_abstract import Splitter, Split


class CartRegressionSplitter(Splitter):
    def __init__(self, min_samples_leaf):
        super().__init__('regression', min_samples_leaf)

    def get_split(self, x, mrv, mrv_square, counts):
        left_sum, left_counts, left_sum_squares = 0., 0., 0.
        split_index, best_impurity = None, np.inf
        total_sum, total_sum_squares, total_counts = np.sum(counts * mrv), np.sum(counts * mrv_square), np.sum(counts)
        previous_value = x[0]
        for i in range(1, mrv.size):
            left_sum += mrv[i - 1] * counts[i - 1]
            left_sum_squares += mrv_square[i - 1] * counts[i - 1]
            left_counts += counts[i - 1]
            if min(left_counts, (total_counts - left_counts)) < self.min_samples_leaf:
                continue
            left_mean = left_sum / left_counts
            right_mean = (total_sum - left_sum) / (total_counts - left_counts)
            left_var = left_sum_squares - left_counts * np.square(left_mean)
            right_var = (total_sum_squares - left_sum_squares) - (total_counts - left_counts) * np.square(right_mean)
            impurity = left_var + right_var
            new_value = x[i]
            if previous_value != new_value: #consider a split
                previous_value = new_value
                if impurity < best_impurity:
                    best_impurity, split_index = impurity, i
        return Split(split_index, best_impurity)


class CartTwoClassClassificationSplitter(Splitter):
    def __init__(self, min_samples_leaf):
        super().__init__('classification', min_samples_leaf)

    def get_split(self, x, mrv, counts):
        # we can look at left sum for example as number of success in the left split
        left_sum, left_counts = 0., 0.
        split_index, best_impurity = None, np.inf
        total_sum, total_counts = np.sum(counts * mrv), np.sum(counts)
        previous_value = x[0]
        for i in range(1, mrv.size):
            left_sum += mrv[i - 1] * counts[i - 1]
            left_counts += counts[i - 1]
            if min(left_counts, (total_counts - left_counts)) < self.min_samples_leaf:
                continue
            left_p = (left_sum / left_counts)
            left_var = left_counts * left_p * (1 - left_p)
            right_p = (total_sum - left_sum) / (total_counts - left_counts)
            right_var = (total_counts - left_counts) * right_p * (1 - right_p)
            impurity = left_var + right_var
            new_value = x[i]
            if previous_value != new_value: #consider a split
                previous_value = new_value
                if impurity < best_impurity:
                    best_impurity, split_index = impurity, i
        return Split(split_index, best_impurity)
