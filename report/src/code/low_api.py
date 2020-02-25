# ...
from isanet.model import Mlp
from isanet.optimizer import SGD, EarlyStopping
from isanet.datasets.monk import load_monk
import numpy as np

X_train, Y_train = load_monk("1", "train")
X_test, Y_test = load_monk("1", "test")

#create the model
model = Mlp()
# Specify the range for the weights and lambda for regularization
# Of course can be different for each layer
kernel_initializer = 0.003 
kernel_regularizer = 0.001

# Add many layers with different number of units
model.add(4, input= 17, kernel_initializer, kernel_regularizer)
model.add(1, kernel_initializer, kernel_regularizer)

es = EarlyStopping(0.00009, 20) # eps_GL and s_UP

#fix which optimizer you want to use in the learning phase
model.setOptimizer(
    SGD(lr = 0.83,          # learning rate
        momentum = 0.9,     # alpha for the momentum
        nesterov = True,    # Specify if you want to use Nesterov
        sigma = None        # sigma for the Acc. Nesterov
    ))

#start the learning phase
model.fit(X_train,
          Y_train, 
          epochs=600, 
          #batch_size=31,
          validation_data = [X_test, Y_test],
          es = es,
          verbose=0) 
            
# after trained the model the prediction operation can be
# perform with the predict method
outputNet = model.predict(X_test)