import sys
from os import path
import os

sys.path.insert(0, "../../")
sys.path.insert(0, "./")

from isanet.neural_network import MLPRegressor
from isanet.model import Mlp
from isanet.optimizer import SGD
from isanet.utils.model_utils import printMSE, printAcc, plotMse, save_data, load_data
from isanet.optimizer import EarlyStopping
from isanet.model_selection import Kfold, GridSearchCV

import numpy as np

dataset = np.genfromtxt('../dataset/cup10/ML-CUP19-TR_tr_vl_10.csv',delimiter=',')
split = load_data("../dataset/cup10/4folds.index")
X_train = dataset[:,:-2]
Y_train = dataset[:,-2:]

es = EarlyStopping(0.009, 200)

mlp_r = MLPRegressor(X_train.shape[1], Y_train.shape[1])

grid = {
            "n_layer_units": [[80], [100]],
            "learning_rate": [0.03, 0.06, 0.08, 0.098],
            "max_epoch": [30000],
            "momentum": [0.2, 0.4, 0.6, 0.8, 0.9],
            "nesterov": [True],
            "kernel_regularizer": [0.0001, 0.0005, 0.0009, 0.0013],
            "activation": ["sigmoid"],
            "early_stop": [es],
}
gs = GridSearchCV(estimator=mlp_r, param_grid = grid, cv = split, verbose=1)
result = gs.fit(X_train, Y_train)
save_data(result, 'large_grid_2_result.data')
