from typing import Dict

from numpy import mean, square, sum
from pandas import DataFrame, Series

from .config import PANDAS_CATEGORICAL_COLS


def get_cols_dtypes(df: DataFrame) -> Dict:
    return {i: v.name for i, v in df.dtypes.to_dict().items()}


def get_col_type(col_type):
    if col_type in PANDAS_CATEGORICAL_COLS:
        return 'categorical'
    return 'numeric'


def regression_impurity(y: Series):
    return sum(square(y - mean(y)))


def classification_impurity(y: Series):
    p = np.sum(y)/y.size
    n = y.size
    return n * p * (1 - p)


impurity_dict = {'regression': regression_impurity, 'classification': classification_impurity}
