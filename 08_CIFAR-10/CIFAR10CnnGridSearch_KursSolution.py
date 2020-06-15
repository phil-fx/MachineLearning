import os
import time
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #für die TF gpu version - sonst gibt es sehr viele GraKa infosd im Terminal

import random
random.seed(0)

import numpy as np
np.random.seed(0)

from sklearn.model_selection import ParameterGrid

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

#from plotting import *
from cifar10Data import *

cifar = CIFAR10()
cifar.data_augmentation(augment_size=5000)
cifar.data_preprocessing(preprocess_mode='MinMax')
x_train_splitted, x_val, y_train_splitted, y_val = cifar.get_splitted_train_validation_set()
x_train, y_train = cifar.get_train_set()
x_test, y_test = cifar.get_test_set()
num_classes = cifar.num_classes

modelID = str('CIFAR10_CNN_model_Augmentation_MixMax_PP_')
zeit = str(time.time())

# Save Path 
dir_path = os.path.abspath('/home/phil/MachineLearning/models/')
if not os.path.exists(dir_path):
    os.makdir(dir_path)
###model_path = os.path.join(dir_path, str(modelID) + str(zeit) + '.h5')

# Log Dir 
log_dir = os.path.abspath('/home/phil/MachineLearning/logs/')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
###model_log_dir = os.path.join(log_dir, str(modelID) + str(zeit))

def model_fn(optimizer, learning_rate, filter_block1, kernel_size_block1, filter_block2, 
             kernel_size_block2, filter_block3, kernel_size_block3, dense_layer_size):
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same')(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Conv Block 2
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Conv Block 3
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    # Dense Part
    x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"])
    return model

epochs = 10
batch_size = 256
optimizers = [Adam]
learning_rates = [1e-3]
filters_block1 = [32]
kernel_sizes_block1 = [3, 5]
filters_block2 = [32, 64]
kernel_sizes_block2 = [3, 5]
filters_block3 = [64, 128]
kernel_sizes_block3 = [7]
dense_layer_sizes = [512]

param_grid = {
    "optimizer": optimizers,
    "learning_rate": learning_rates,
    "filter_block1": filters_block1,
    "kernel_size_block1": kernel_sizes_block1,
    "filter_block2": filters_block2,
    "kernel_size_block2": kernel_sizes_block2,
    "filter_block3": filters_block3,
    "kernel_size_block3": kernel_sizes_block3,
    "dense_layer_size": dense_layer_sizes
}

results = {"best_score": -np.inf,
           "best_params": {},
           "test_scores": [],
           "params": []}
grid = ParameterGrid(param_grid)

print("Parameter combinations in total: %d" % len(grid))
for idx, comb in enumerate(grid):
    print("Running comb %d" % idx)
    curr_model = model_fn(**comb)

    model_log_dir = os.path.join(log_dir, 'modelGrid%d' % idx)
    if os.path.exists(model_log_dir):
        shutil.rmtree(model_log_dir)
        os.mkdir(model_log_dir)
    
    model_path = os.path.join(dir_path, 'modelGrid%d' % idx, '.h5')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        os.mkdir(model_path)

    tb_callback = TensorBoard(
        log_dir=model_log_dir)

    curr_model.fit(
        x=x_train_splitted, 
        y=y_train_splitted, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=[x_val, y_val],
        callbacks=[tb_callback],
        verbose=0)
    
    #curr_model.save_weights(filepath=model_path)
    #curr_model.load_weights(filepath=model_path)

    results["test_scores"].append(curr_model.evaluate(x_val, y_val, verbose=0)[1]) # Anstelle 1 sind die scores
    results["params"].append(comb)

best_run_idx = np.argmax(results["test_scores"])
results["best_score"] = results["test_scores"][best_run_idx]
results["best_params"] = results["params"][best_run_idx]

# Summary
print("Best: %f using %s\n\n" % (results["best_score"], results['best_params']))

scores = results["test_scores"]
params = results["params"]

for score, param in zip(scores, params):
    print("Acc: %f with: %r" % (score, param))
