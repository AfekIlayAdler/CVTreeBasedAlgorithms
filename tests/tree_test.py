import time

import numpy as np

from algorithms import CartRegressionTreeKFold, CartRegressionTree, TreeVisualizer
from tests.test_dataset_creator import create_x_y

EXP = 'simulation'
KFOLD = True
MAX_DEPTH = 5
tree = CartRegressionTreeKFold(max_depth=MAX_DEPTH) if KFOLD else CartRegressionTree(max_depth=MAX_DEPTH)
np.random.seed(3)
X, y = create_x_y()
print(np.mean(y[X['x1'] > 0]))
print(np.mean(y[X['x1'] <= 0]))
print(y.mean())
start = time.time()
tree.fit(X, y)
end = time.time()
print(end - start)
tree_vis = TreeVisualizer()
tree_vis.plot(tree.root)
