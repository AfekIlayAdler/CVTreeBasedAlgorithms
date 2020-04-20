import xgboost as xgb
from catboost import Pool, CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from .default_config import MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE
from algorithms import CartGradientBoostingRegressorKfold, \
    CartGradientBoostingRegressor


def get_fitted_model(model_name, variant, X_train, y_train, cat_features):
    if model_name == 'ours':
        model = CartGradientBoostingRegressorKfold if variant == 'Kfold' else CartGradientBoostingRegressor
        reg = model(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE, min_samples_leaf=5)
        reg.fit(X_train, y_train)
    elif model_name == 'sklearn':
        reg = GradientBoostingRegressor(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                        learning_rate=LEARNING_RATE)
        reg.fit(X_train, y_train)
    elif model_name == 'xgboost':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        param = {'max_depth': MAX_DEPTH, 'eta': LEARNING_RATE, 'objective': 'reg:squarederror'}
        reg = xgb.train(param, dtrain, N_ESTIMATORS)
    else:  # model_name == 'catboost
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        reg = CatBoostRegressor(iterations=N_ESTIMATORS,
                                depth=MAX_DEPTH,
                                learning_rate=LEARNING_RATE,
                                loss_function='RMSE', logging_level='Silent')
        reg.fit(train_pool)
    return reg



