import os
import time
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
random.seed(0)

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(0)

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from plotting import *
from TaxiRoutingData import *

excel_file_path = os.path.abspath('/home/phil/data/Taxi/dataset_taxi.xlsx') 
taxi_data = TAXIROUTING(excel_file_path=excel_file_path)
x_train, y_train = taxi_data.x_train, taxi_data.y_train
x_test, y_test = taxi_data.x_test, taxi_data.y_test
num_features = taxi_data.num_features
num_targets = taxi_data.num_targets

modelID = str('Taxi_DNN_')
zeit = str(time.time())

# Save Path 
dir_path = os.path.abspath('/home/phil/MachineLearning/models/')
if not os.path.exists(dir_path):
    os.makdir(dir_path)
model_path = os.path.join(dir_path, str(modelID) + str(zeit) + '.h5')

# Log Dir 
log_dir = os.path.abspath('/home/phil/MachineLearning/logs/')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
model_log_dir = os.path.join(log_dir, str(modelID) + str(zeit))

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred))) # Zähler: Summe der Quadr. Abweichungen
    y_true_mean = tf.math.reduce_mean(y_true) # Mittelw. von y_true
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean))) # Nenner: Summe der Quadr. Abweichungen zwischen wahren y und dem Mittelwert von y
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator)) # 1 minus den Bruch
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0) # alle neg zahlen werden zu 0.0
    return r2_clipped

# Define the DNN
def model_fn(optimizer, learning_rate, 
             dense_layer_size1, dense_layer_size2, dense_layer_size3,
             activation_str, dropout_rate, use_bn):
    # Input
    input_house = Input(shape=x_train.shape[1:])  
    
    # DenseLayer 1
    x = Dense(units=dense_layer_size1)(input_house)
    if use_bn:
        x = BatchNormalization()(x)  # Immer nach dem Dense oder Conv und VOR der Activierungsfkt
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    
    # DenseLayer 2
    x = Dense(units=dense_layer_size2)(x)
    if use_bn:
        x = BatchNormalization()(x)  # Immer nach dem Dense oder Conv und VOR der Activierungsfkt
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)

    # DenseLayer 3
    x = Dense(units=dense_layer_size3)(x)
    if use_bn:
        x = BatchNormalization()(x)  
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)

    #Output-Layer    
    x = Dense(units=num_targets)(x)
    y_pred = Activation('linear')(x)  # kann man auch weglassen, 'linear' is wie weglassen

    # Build the model
    model = Model(inputs=[input_house], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=[r_squared])
    model.summary()
    return model

# Global params
epochs = 20
batch_size = 128

params = {
    'optimizer': Adam,
    'learning_rate': 0.001,
    'dense_layer_size1': 64,
    'dense_layer_size2': 128,
    'dense_layer_size3': 32,
    # relu, elu, LeakyReLU
    'activation_str': 'relu',
    # 0.05, 0.1, 0.2
    'dropout_rate': 0.00,
    # True, False
    'use_bn': True
}

model = model_fn(**params)

def schedule_fn(epoch):
    lr = 1e-3
    if epoch < 5:
        lr = 1e-3
    elif epoch < 20:
        lr = 5e-4
    else:
        lr = 1e-4
    return 

def schedule_fn2(epoch):
    threshold = 500
    if epoch < threshold:
        return 1e-3
    else:
        return 1e-3 * np.exp(0.005 * (threshold - epoch))

lrs_callback = LearningRateScheduler(
    schedule=schedule_fn2,    # wähle oder erstelle fn's 
    verbose=1)

plateau_callback = ReduceLROnPlateau(
    monitor='val_accuracy',   # val_loss, val_accuracy
    factor=0.98, 
    patience=30,               # Wenn 3x nicht verbesert, dann LR reduce
    #min_delta=0.0001,        # abweichungsvergleich zwischen epochen
    #cooldown=0,
    verbose=1, 
    min_lr=1e-5)

earlyStopping_callback = EarlyStopping(
    monitor='val_accuracy',   # val_loss, val_accuracy
    patience=50, 
    verbose=1,
    restore_best_weights=True)

class LRTensorBoard(TensorBoard):  # diese Klasse erbt von der Tensoroard-Klasse
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs) # der konstruktor ist gleich , übergebe aber die parameter mit kwargs

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)

tb_callback = LRTensorBoard(
    log_dir=model_log_dir)

model.fit(
    x=x_train, 
    y=y_train, 
    verbose=1, 
    batch_size=batch_size, 
    epochs=epochs, 
    callbacks=[
        tb_callback, 
        #lrs_callback, 
        plateau_callback,
        earlyStopping_callback],
    validation_data=(x_test, y_test))
model.save_weights(filepath=model_path)

### wenn nur geladen wird ###
# model_path = '/home/phil/MachineLearning/models/DogsCats_CNN_Final_CustomIMG1590163165.0453773.h5'

# model.load_weights(filepath=model_path)

score = model.evaluate(
    x_test, 
    y_test, 
    verbose=0, 
    batch_size=batch_size)
print('Test performance: ', score)

y_pred = model.predict(x_test)

import seaborn as sns 

sns.residplot(y_test, y_pred, scatter_kws={'s': 2, 'alpha': 0.5})  # uhrzeit auf der x-achse und die abweichung auf der y-achse
plt.show()