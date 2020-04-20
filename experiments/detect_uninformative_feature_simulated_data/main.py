import multiprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.default_config import RESULTS_DIR, VAL_RATIO, MODELS_DIR
from experiments.detect_uninformative_feature_simulated_data.config import create_x_y, all_experiments, N_PROCESS, \
    n_total_experiments, CATEGORICAL_FEATURES, MODELS, DEBUG
from experiments.feature_importance_utils import get_fi_gain, get_fi_permutation, \
    get_fi_shap
from experiments.get_model import get_fitted_model
from experiments.utils import create_one_hot_x_x_val, create_mean_imputing_x_x_val, make_dirs


def worker(model_name, variant, exp_number, category_size):
    exp_name = F"{model_name}_{variant}_exp_{exp_number}_category_size_{category_size}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    if exp_results_path.exists():
        return
    np.random.seed(exp_number)
    X, y = create_x_y(category_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    results_df = pd.DataFrame(
        columns=['model', 'categories', 'exp', 'gain', 'permutation_train', 'permutation_test', 'shap_train',
                 'shap_test'])
    results_df.loc[0, ['model', 'categories', 'exp']] = [F"{model_name}_{variant}", category_size, exp_number]
    if variant == 'one_hot':
        X_train, X_test = create_one_hot_x_x_val(X_train, X_test, CATEGORICAL_FEATURES)
    if variant == 'mean_imputing':
        X_train, X_test = create_mean_imputing_x_x_val(X_train, y_train, X_test, CATEGORICAL_FEATURES)

    cat_features = None if variant == 'mean_imputing' else CATEGORICAL_FEATURES
    reg = get_fitted_model(model_name, variant, X_train, y_train, cat_features)
    fi_gain = get_fi_gain(model_name, reg, X_train)
    fi_permutation_train, fi_permutation_test = get_fi_permutation(model_name, reg, X_train, y_train, X_test, y_test,
                                                                   cat_features)
    fi_shap_train, fi_shap_test = get_fi_shap(model_name, reg, X_train, X_test, y_train, y_test, cat_features)
    if fi_shap_train is not None:
        fi_shap_train = fi_shap_train['x1'] + fi_shap_train['x2']
        fi_shap_test = fi_shap_test['x1'] + fi_shap_test['x2']
    results_df.loc[0, 'gain'] = fi_gain['x1'] + fi_gain['x2']
    results_df.loc[0, 'permutation_train'] = fi_permutation_train['x1'] + fi_permutation_train['x2']
    results_df.loc[0, 'permutation_test'] = fi_permutation_test['x1'] + fi_permutation_test['x2']
    results_df.loc[0, ['shap_train', 'shap_test']] = [fi_shap_train, fi_shap_test]
    results_df.to_csv(exp_results_path)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    print(f"n experimets for each model: {n_total_experiments}")
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        args = []
        make_dirs([RESULTS_DIR / model_name])
        for exp_number, category_size in all_experiments():
            for variant in model_variants:
                if DEBUG:
                    worker(model_name, variant, exp_number, category_size)
                else:
                    args.append((model_name, variant, exp_number, category_size))
            if not DEBUG:
                with multiprocessing.Pool(N_PROCESS) as process_pool:
                    process_pool.starmap(worker, args)
                # with concurrent.futures.ThreadPoolExecutor(4) as executor:
                #     results = list(tqdm(executor.map(lambda x: worker(*x), args), total=len(args)))
