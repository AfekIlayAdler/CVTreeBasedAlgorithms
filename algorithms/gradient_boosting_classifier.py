from .Tree.node import Leaf
from .Tree import CartRegressionTree, CartRegressionTreeKFold, MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, MIN_SAMPLES_SPLIT

from numpy import mean, array, log, exp, zeros, ones
from pandas import DataFrame

from .gradient_boosting_abstract import GradientBoostingMachine
from .config import N_ESTIMATORS, LEARNING_RATE


class GradientBoostingClassifier(GradientBoostingMachine):
    """currently supports only binomial log likelihood as in the original paper of friedman"""

    def __init__(self, base_tree,
                 n_estimators,
                 learning_rate,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split):
        super().__init__(
            base_tree=base_tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)
        self.predictions_to_step_size_dicts = []

    def line_search(self, x, y):
        """
        x: tree predictions
        y : pseudo_response
        """
        n_rows = x.shape[0]
        predictions_to_step_size_dict = {}
        for index in range(n_rows):
            predictions_to_step_size_dict.setdefault(x[index], array([0., 0.]))
            predictions_to_step_size_dict[x[index]] += array([y[index], abs(y[index]) * (2 - abs(y[index]))])
        for key, value in predictions_to_step_size_dict.items():
            predictions_to_step_size_dict[key] = value[0] / value[1]
        self.predictions_to_step_size_dicts.append(predictions_to_step_size_dict)
        gamma = zeros(n_rows)
        for index in range(n_rows):
            gamma[index] = predictions_to_step_size_dict[x[index]]
        return gamma

    def fit(self, x, y):
        y = 2 * y - 1
        self.base_prediction = 0.5 * log((1 + mean(y)) / (1 - mean(y)))
        f = self.base_prediction
        for m in range(self.n_estimators):
            if m > 0 and isinstance(self.trees[-1].root, Leaf):  # if the previous tree was a bark then we stop
                return
            pseudo_response = 2 * y / (1 + exp(2 * y * f))
            h_x = self.fit_tree(x, pseudo_response)
            gamma = self.line_search(h_x, pseudo_response)
            f += self.learning_rate * gamma
            self.n_trees += 1

    def predict(self, data: DataFrame):
        prediction = ones(data.shape[0]) * self.base_prediction
        for tree_index, tree in enumerate(self.trees):
            tree_predictions = tree.predict(data.to_dict('records'))
            for i in range(tree_predictions.size):
                tree_predictions[i] = self.predictions_to_step_size_dicts[tree_index].get(tree_predictions[i])
            prediction += self.learning_rate * tree_predictions
        return 1 / (1 + exp(-2 * prediction))


class CartGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(
            base_tree=CartRegressionTree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)


class CartGradientBoostingClassifierKfold(GradientBoostingClassifier):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(
            base_tree=CartRegressionTreeKFold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split)


