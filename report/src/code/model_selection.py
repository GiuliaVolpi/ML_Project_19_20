from isanet.model_selection import Kfold, GridSearchCV
from isanet.neural_network import MLPRegressor

dataset = np.genfromtxt('CUP/ML-CUP19-TR_tr_vl.csv',delimiter=',')
X_train, Y_train = dataset[:,:-2], dataset[:,-2:]

grid = {"n_layer_units": [[38], [20, 32]], #[20, 32] means two hidden layer
         "learning_rate": [0.014, 0.017],
         "max_epoch": [13000, 1000],
         "momentum": [0.8, 0.6],
         "nesterov": [True, False],
         "sigma": [None, 0.8, 0.6, 2, 4]
         "kernel_regularizer": [0.0001],
         "activation": ["sigmoid"],
         "early_stop": [EarlyStopping(0.00009, 20), EarlyStopping(0.09, 200)]}

mlp_r = MLPRegressor(X_train.shape[1], Y_train.shape[1])
kf = Kfold(n_splits=5, shuffle=True, random_state=1)
gs = GridSearchCV(estimator=mlp_r, param_grid = grid, cv = kf, verbose=2)
result = gs.fit(X, Y) # dict with keys as column headers and values as columns

