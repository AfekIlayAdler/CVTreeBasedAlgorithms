import pickle
from os import makedirs

from experiments.default_config import MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE
from experiments.one_hot_encoder import OneHotEncoder


def create_one_hot_x_x_val(x, x_val, categorical_columns):
    one_hot = OneHotEncoder()
    one_hot.fit(x.iloc[:,categorical_columns])
    x_one_hot = one_hot.transform(x.iloc[:,categorical_columns])
    x_one_hot_val = one_hot.transform(x_val.iloc[:,categorical_columns])
    for col in x.columns:
        if col not in x.columns[categorical_columns]:
            x_one_hot.loc[:, col] = x[col]
            x_one_hot_val.loc[:, col] = x_val[col]
    return x_one_hot, x_one_hot_val


def create_mean_imputing_x_x_val(x, y, x_val, categorical_columns):
    temp_x = x.copy()
    col_name = 'y'
    temp_x.loc[:, col_name] = y
    for col in x.columns[categorical_columns]:
        category_to_mean = temp_x.groupby(col)[col_name].mean().to_dict()
        temp_x[col] = temp_x[col].map(category_to_mean)
        x_val[col] = x_val[col].map(category_to_mean)
        temp_x[col] = temp_x[col].astype('float')
        x_val[col] = x_val[col].astype('float')
        x_val[col] = x_val[col].fillna(x_val[col].mean())
    temp_x = temp_x.drop(columns=[col_name])
    return temp_x, x_val


def make_dirs(dirs):
    for dir in dirs:
        if not dir.exists():
            makedirs(dir)


# def get_fitted_model(path, model, X, y_col_name):
#     if path.exists():
#         with open(path, 'rb') as input_file:
#             model = pickle.load(input_file)
#
#     else:
#         model = model(y_col_name, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,learning_rate=LEARNING_RATE)
#         model.fit(X, )
#     return model


def save_model(path, model):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

        # def compute_ntrees_nleaves(gbm):
        #     total_number_of_trees = 0
        #     total_number_of_leaves = 0
        #     if isinstance(gbm, GradientBoostingMachine):
        #         for tree in gbm.trees:
        #             if not isinstance(tree.root, Leaf):
        #                 total_number_of_trees += 1
        #                 total_number_of_leaves += tree.n_leaves
        #     elif isinstance(gbm, GradientBoostingClassifier) or isinstance(gbm, GradientBoostingRegressor):
        #         total_number_of_trees = gbm.n_estimators_
        #         for tree in gbm.estimators_:
        #             total_number_of_leaves += tree[0].get_n_leaves()
        #     elif isinstance(gbm, Booster):
        #         df = gbm.trees_to_dataframe()
        #         total_number_of_leaves = df[df['Feature'] == 'Leaf'].shape[0]
        #         total_number_of_trees = df[df['Feature'] == 'Leaf']['Tree'].max()
        #
        #     print(F'number of trees is {total_number_of_trees}')
        #     print(F'number of leaves is {total_number_of_leaves}')
