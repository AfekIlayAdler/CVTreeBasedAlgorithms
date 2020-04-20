import numpy as np
import time
from algorithms.Tree import TreeVisualizer
from algorithms import CartGradientBoostingRegressorKfold, CartGradientBoostingRegressor
from tests.test_dataset_creator import create_x_y

np.seterr(all='raise')
EXP = 'simulation'  # 'simulation'
KFOLD = True
MAX_DEPTH = 3
reg = CartGradientBoostingRegressorKfold(max_depth=3) if KFOLD else CartGradientBoostingRegressor(max_depth=3)
np.random.seed(3)
X, y = create_x_y()
start = time.time()
reg.fit(X, y)
end = time.time()
print(end - start)
tree_vis = TreeVisualizer()
tree_vis.plot(reg.trees[0].root)
# tree_vis.plot(reg.trees[1].root)
# tree_vis.plot(reg.trees[2].root)
print(reg.n_trees)
print(reg.compute_feature_importance())
