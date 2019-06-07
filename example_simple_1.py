import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning")) # Adding the submodule to the module search path
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning/examples")) # Adding the submodule to the module search path
import numpy as np
from examples import utils
from keras.layers import Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras_dgl.layers import GraphCNN
import keras.backend as K


def make_filter(filter_type):
    """
    the wording graph_conv filters comes from signal processing & spectral filters.
    It's whatever is included in the forward propagation, and it really means, how do we
    use the graph edges.

    # graph_conv_filters input as a 2D tensor with shape: (num_filters*num_graph_nodes, num_graph_nodes)
    # num_filters is different number of graph convolution filters to be applied on graph....
    """
    if filter_type == 1:
        print("this simply ignores the connected edges, thus receives a pretty bad test_acc:")
        graph_conv_filters = np.eye(A_norm.shape[0])
        graph_conv_filters = K.constant(graph_conv_filters)
        num_filters = 1
    elif filter_type == 2:
        print("this filter includes the edges, so it should perform considerably better than before.:")
        graph_conv_filters = A_norm
        graph_conv_filters = K.constant(graph_conv_filters)
        num_filters = 1
    elif filter_type == 3:
        print("this filter includes the edges, as well as some quadratic condition on the edge weights.:")
        graph_conv_filters = np.concatenate([A_norm, np.matmul(A_norm, A_norm)], axis=0)
        graph_conv_filters = K.constant(graph_conv_filters)
        num_filters = 2
    return graph_conv_filters, num_filters

def build_simple_gcn(num_filters, graph_conv_filters):
    model = Sequential()
    model.add(GraphCNN(Y.shape[1], num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='elu',
                       kernel_regularizer=l2(5e-4)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
    model.summary()
    return model


X, A, Y = utils.load_data(dataset='cora')

print("Just to check that this is indeed sparse, but not zero, check the column sums: ", sum(A.A))

y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = utils.get_splits(Y)

A_norm = utils.preprocess_adj_numpy(A, True)

for i in [1,2,3]:
    graph_conv_filters, num_filters = make_filter(i)
    model = build_simple_gcn(num_filters,graph_conv_filters)

    nb_epochs = 100

    for epoch in range(nb_epochs):
        model.fit(X, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
        Y_pred = model.predict(X, batch_size=A.shape[0])
        _, train_acc = utils.evaluate_preds(Y_pred, [y_train], [idx_train])
        _, test_acc = utils.evaluate_preds(Y_pred, [y_test], [idx_test])
        print("Epoch: {:04d}".format(epoch), "train_acc= {:.4f}".format(train_acc[0]), "test_acc= {:.4f}".format(test_acc[0]))

# i = 1
# Epoch: 0099 train_acc= 1.0000 test_acc= 0.5130

# i = 2
# Epoch: 0017 train_acc= 1.0000 test_acc= 0.6950
# Epoch: 0099 train_acc= 1.0000 test_acc= 0.7600

# i = 3
# Epoch: 0017 train_acc= 1.0000 test_acc= 0.7930
# Epoch: 0099 train_acc= 1.0000 test_acc= 0.8030