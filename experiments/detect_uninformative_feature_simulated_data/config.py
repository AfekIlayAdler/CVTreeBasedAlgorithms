from numpy import arange
from numpy.random import randint, randn
from pandas import DataFrame

from experiments.default_config import CATEGORY_COLUMN_NAME, N_ROWS, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE

CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 60, 10))
N_EXPERIMENTS = 15

# data
A1 = 3
A2 = 2
SIGMA = 10

# io
EXP_NAME = F"detect_uninformative_feature_max_depth_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
CATEGORICAL_COLUMNS = [CATEGORY_COLUMN_NAME]
CATEGORICAL_FEATURES = [2]
#
N_PROCESS = 2

DEBUG = False
MODELS = {
    'xgboost': ['one_hot', 'mean_imputing'],
    'catboost': ['vanilla', 'mean_imputing'],
    'sklearn': ['one_hot', 'mean_imputing'],
    'ours': ['Kfold', 'CartVanilla']}


def create_x_y(category_size):
    X = DataFrame()
    X['x1'] = randn(N_ROWS)
    X['x2'] = randn(N_ROWS)
    sigma = SIGMA * randn(N_ROWS)
    y = A1 * X['x1'] + A2 * X['x2'] + sigma
    X[CATEGORY_COLUMN_NAME] = randint(0, category_size, N_ROWS)
    X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
    return X, y


def all_experiments():
    return [(exp_number, category_size) for exp_number in range(N_EXPERIMENTS) for category_size in CATEGORIES]


n_total_experiments = len(all_experiments())
