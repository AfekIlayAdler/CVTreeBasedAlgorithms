if __name__ == '__main__':
    import numpy as np
    import pandas as pd


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def create_x_y():
        a = 0.1
        N_ROWS = 1000
        category_size = 10
        CATEGORY_COLUMN_NAME = 'category'
        X = pd.DataFrame()
        X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
        X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
        X['x1'] = np.random.randn(N_ROWS)
        sigma = 0.1 * np.random.randn(N_ROWS)
        left_group = [i for i in range(category_size // 2)]
        y = np.round(sigmoid(a * (X['x1'] > 0) * 1 + (1 - a) * X[CATEGORY_COLUMN_NAME].isin(left_group)) + sigma)
        return X, y


    X, y = create_x_y()
    reg = CartGradientBoostingClassifier(n_estimators=2, max_depth=3)
    reg.fit(X, y)
    y_hat = reg.predict(X)
