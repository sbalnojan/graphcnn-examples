import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning")) # Adding the submodule to the module search path
sys.path.append(os.path.join(os.getcwd(), "keras-deep-graph-learning/examples")) # Adding the submodule to the module search path
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras_dgl.layers import GraphCNN
import keras.backend as K
from keras.utils import to_categorical

print("Creating our simple sample data...")
A = np.array([[1,1,1], [1,1,0], [1,0,0]])
X = np.array([[1,0,0], [1,1,0], [4,5,6]]) # features, whatever we have there...

# Notice, if we set A = identity matrix, then we'd effectively assume no edges and just do a basic
# MLP on the features.

# We could do the same by setting the graph_conv_filter below to Id.

# We could also set X to Id, and thus effectively assume no features, and in this way
# do an "edge" embedding, so effectively try to understand what's connected to what.

# We could then use that as feature in any way we like...

Y_o_dim = np.array([1,2,1])
Y =  to_categorical(Y_o_dim) # labels, whatever we wanna classify things into... in categorical form.

train_on_weight= np.array([1,1,0])
print("Now we won't do any fancy preprocessing, just basic training.")

NUM_FILTERS = 1
graph_conv_filters = A
graph_conv_filters = K.constant(graph_conv_filters)

model = Sequential()
model.add(GraphCNN(Y.shape[1], NUM_FILTERS, graph_conv_filters, input_shape=(X.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
model.summary()

model.fit(X, Y, batch_size=A.shape[0], sample_weight=train_on_weight, epochs=100, shuffle=False, verbose=0)
Y_pred = model.predict(X, batch_size=A.shape[0])
np.argmax(Y_pred, axis=1)

