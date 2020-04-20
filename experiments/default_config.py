from pathlib import Path

# gbm
MAX_DEPTH = 3
N_ESTIMATORS = 100
LEARNING_RATE = 0.1

# exp
N_PERMUTATIONS = 20

# io
MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/experiments_results/")

# data
CATEGORY_COLUMN_NAME = 'category'
VAL_RATIO = 0.15
Y_COL_NAME = 'y'
N_ROWS = 10 ** 3