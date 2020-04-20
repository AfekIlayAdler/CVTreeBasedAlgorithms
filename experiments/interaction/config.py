from numpy import arange
from numpy.random import randint, randn
from pandas import DataFrame

from experiments.default_config import CATEGORY_COLUMN_NAME, N_ROWS, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE

# exp
CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 60, 10))
A_VALUES = arange(11)
N_EXPERIMENTS = 15

# data
SIGMA = 10

# io
EXP_NAME = F"interaction_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
CATEGORICAL_COLUMNS = [CATEGORY_COLUMN_NAME]
CATEGORICAL_FEATURES = [0]
#
N_PROCESS = 1

DEBUG = False
MODELS = {
    'xgboost': ['one_hot', 'mean_imputing'],
    'catboost': ['vanilla', 'mean_imputing'],
    'sklearn': ['one_hot', 'mean_imputing'],
    'ours': ['Kfold', 'CartVanilla']}

def create_x_y(category_size, a):
    a = float(a)
    X = DataFrame()
    X[CATEGORY_COLUMN_NAME] = randint(0, category_size, N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    X['x1'] = randn(N_ROWS)
    sigma = SIGMA*randn(N_ROWS)
    left_group = [i for i in range(category_size // 2)]
    right_group = [i for i in range(category_size) if i not in left_group]
    left_indicator = (X['x1'] > 0) * 1
    y = a * left_indicator + (10-a)* X[CATEGORY_COLUMN_NAME].isin(right_group) + sigma
    return X, y


def all_experiments():
    return [(exp_number, category_size, a) for exp_number in range(N_EXPERIMENTS)
            for category_size in CATEGORIES
            for a in A_VALUES]


n_total_experiments = len(all_experiments())


