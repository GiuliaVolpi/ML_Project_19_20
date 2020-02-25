from isanet.neural_network import MLPClassifier
from isanet.datasets.monk import load_monk

X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

mlp_c = MLPClassifier(X_train.shape[1],             # input dim
                      Y_train.shape[1],             # out dim
                      n_layer_units=[4],            # topology
                      activation="sigmoid",         # activation hidden layer
                      kernel_regularizer=0.001,     # l2 regularization term
                      max_epoch=600,                # Max number of Epoch
                      learning_rate=0.83,           # learning rate
                      momentum=0.9,                 # momentum term
                      nesterov=True,                # if Nesterov
                      sigma=None,                   # sigma Acc. Nesterov
                      early_stop=None,              # define the early stop
                      verbose=0)                    # verbosity
mlp_c.fit(X_train, Y_train, X_test, Y_test)
outputNet = mlp_r.predict(X_test)
