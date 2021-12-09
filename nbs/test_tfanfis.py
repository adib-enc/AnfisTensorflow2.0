# Test tfanfis model
import os
os.sys.path.append('../Models/')
import tfanfis
import numpy as np

# if __name__ == "__main__":
# set parameters
param = tfanfis.fis_parameters(
    n_input = 2,                # no. of Regressors
    n_memb = 2,                 # no. of fuzzy memberships
    batch_size = 16,            # 16 / 32 / 64 / ...
    memb_func = 'sigmoid',      # 'gaussian' / 'gbellmf' / 'sigmoid'
    optimizer = 'adam',          # sgd / adam / ...
    loss = 'huber_loss',               # mse / mae / huber_loss / mean_absolute_percentage_error / ...
    n_epochs = 50               # 10 / 25 / 50 / 100 / ...
)
# create random data
X_train = np.random.rand(param.batch_size*5, param.n_input),
X_test = np.random.rand(param.batch_size*2, param.n_input)
y_train = np.random.rand(param.batch_size*5,1),
y_test = np.random.rand(param.batch_size*2, 1)

fis = tfanfis.ANFIS(
    n_input = param.n_input,
    n_memb = param.n_memb,
    batch_size = param.batch_size,
    memb_func = param.memb_func,
    name = 'tfanfis'
)

# compile model
fis.model.compile(
    optimizer=param.optimizer,
    loss=param.loss
    #,metrics=['mse']  # ['mae', 'mse']
)

# fit model
history = fis.fit(
    X_train, y_train,
    epochs=param.n_epochs,
    batch_size=param.batch_size,
    validation_data = (X_test, y_test),
    # callbacks = [tensorboard_callback]  # for tensorboard
)

# eval model
import pandas as pd
fis.plotmfs(show_initial_weights=True)

loss_curves = pd.DataFrame(history.history)
loss_curves.plot(figsize=(8, 5))

fis.model.summary()

# get premise parameters
premise_parameters = fis.model.get_layer('fuzzyLayer').get_weights()       # alternative

# get consequence paramters
bias = fis.bias
weights = fis.weights
# conseq_parameters = fis.model.get_layer('defuzzLayer').get_weights()       # alternative
