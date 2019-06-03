import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning")) # Adding the submodule to the module search path
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning/examples")) # Adding the submodule to the module search path

import keras_dgl
from examples import utils

X, A, Y = utils.load_data(dataset='cora')

print("Just to check that this is indeed sparse, but not zero, check the column sums: ", sum(A.A))

y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = utils.get_splits(Y)

X
Y