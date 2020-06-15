import os
import time
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from plotting import *
from DogsCatsData import *

data = DOGSCATS()
data.data_augmentation(augment_size=5000)
data.data_preprocessing(preprocess_mode='MinMax')
x_train_splitted, x_val, y_train_splitted, y_val = data.get_splitted_train_validation_set()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()
num_classes = data.num_classes

modelID = str('DogsCats_CNN_Plot')
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

# Define the DNN
def model_fn(optimizer, learning_rate, filter_block1, kernel_size_block1, filter_block2, 
             kernel_size_block2, filter_block3, kernel_size_block3, dense_layer_size,
             kernel_initializer, bias_initializer, activation_str, dropout_rate, use_bn,
             use_global_pooling, use_additional_dense_layer):
    # Input
    input_img = Input(shape=x_train.shape[1:])
    # Conv Block 1
    x = Conv2D(filters=filter_block1, 
               kernel_size=kernel_size_block1, 
               padding='same',
               kernel_initializer=kernel_initializer, 
               bias_initializer=bias_initializer,
               name='heatmap')(input_img)
    if use_bn:
        x = BatchNormalization()(x)  # Immer nach dem Dense oder Conv und VOR der Activierungsfkt
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(filters=filter_block1, 
               kernel_size=kernel_size_block1, 
               padding='same',
               kernel_initializer=kernel_initializer, 
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 2
    x = Conv2D(filters=filter_block2, 
               kernel_size=kernel_size_block2, 
               padding='same',
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(filters=filter_block2, 
               kernel_size=kernel_size_block2, 
               padding='same',
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3
    x = Conv2D(filters=filter_block3, 
               kernel_size=kernel_size_block3, 
               padding='same',
               kernel_initializer=kernel_initializer, 
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(filters=filter_block3, 
               kernel_size=kernel_size_block3, 
               padding='same',
               kernel_initializer=kernel_initializer, 
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)

    # Conv Block 3 - ist dopplet
    x = Conv2D(filters=filter_block3, 
               kernel_size=kernel_size_block3, 
               padding='same',
               kernel_initializer=kernel_initializer, 
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = Conv2D(filters=filter_block3, 
               kernel_size=kernel_size_block3, 
               padding='same',
               kernel_initializer=kernel_initializer, 
               bias_initializer=bias_initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)
    if activation_str == 'LeakyReLU':
        x = LeakyReLU()(x)
    else:
        x = Activation(activation_str)(x)
    x = MaxPool2D()(x)  # Dimension der FeatureMap hier: 4x4 und 128 Filter bzw. Feature-Maps --> (4,4,128)

    # Dense Part
    if use_global_pooling:
        x = GlobalAveragePooling2D()(x)  #Ersetzt Flatten und Erstellt einen Wert pro Feature-Map --> 128 Werte
    else:
        x = Flatten()(x) # Entweder Flattening oder 
    if use_additional_dense_layer:        
        x = Dense(units=dense_layer_size)(x)
        if activation_str == 'LeakyReLU':
            x = LeakyReLU()(x)
        else:
            x = Activation(activation_str)(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation('softmax')(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    #model.summary()
    return model

# Global params
epochs = 1 # Sollte 100 sein
batch_size = 256

params = {
    'optimizer': Adam,
    'learning_rate': 0.001,
    'filter_block1': 32,
    'kernel_size_block1': 3,
    'filter_block2': 64,
    'kernel_size_block2': 3,
    'filter_block3': 128,
    'kernel_size_block3': 3,
    'dense_layer_size': 1024,
    # GlorotUniform, GlorotNormal, RandomNormal, RandomUniform, VarianceScaling
    'kernel_initializer': 'GlorotUniform',
    'bias_initializer': 'zeros',
    # relu, elu, LeakyReLU
    'activation_str': 'relu',
    # 0.05, 0.1, 0.2
    'dropout_rate': 0.00,
    # True, False
    'use_bn': True,
    'use_global_pooling': True,
    'use_additional_dense_layer': True
}

model = model_fn(**params)

# plateau_callback = ReduceLROnPlateau(
#     monitor='val_accuracy',   # val_loss, val_accuracy
#     factor=0.95, 
#     patience=2,               # Wenn 3x nicht verbesert, dann LR reduce
#     #min_delta=0.0001,        # abweichungsvergleich zwischen epochen
#     #cooldown=0,
#     verbose=1, 
#     min_lr=1e-5)

# earlyStopping_callback = EarlyStopping(
#     monitor='val_accuracy',   # val_loss, val_accuracy
#     patience=15, 
#     verbose=1,
#     restore_best_weights=True)

# model.fit(
#     x=x_train, 
#     y=y_train, 
#     verbose=1, 
#     batch_size=batch_size, 
#     epochs=epochs, 
#     callbacks=[
#         plateau_callback,
#         earlyStopping_callback],
#     validation_data=(x_test, y_test))
# model.save_weights(filepath=model_path)

### LOAD MODEL ###

model_path = '/home/phil/MachineLearning/models/DogsCats_CNN_Plot.h5'
model.load_weights(filepath=model_path)

# score = model.evaluate(
#     x_test, 
#     y_test, 
#     verbose=0, 
#     batch_size=batch_size)
# print('Test performance: ', score)

get_heatmap(
    img=x_test[13],
    model=model)