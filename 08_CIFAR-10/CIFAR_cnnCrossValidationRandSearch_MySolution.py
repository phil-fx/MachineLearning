import os 
import time

import random
random.seed(0)

# import numpy as np 
# np.random.seed(0)

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.model_selection import cross_val_score

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#from plotting import *
from cifar10Data import *

cifar10 = CIFAR10()
cifar10.data_augmentation(augment_size=5000)
cifar10.data_preprocessing(preprocess_mode='MinMax')
x_train, y_train = cifar10.get_train_set()
x_test, y_test = cifar10.get_test_set()
num_classes = cifar10.num_classes

modelID = str('CIFAR10_CNN_model_Augmentation_MixMax_PP_')
zeit = str(time.time())

# Save Path 
dir_path = os.path.abspath('/home/phil/MachineLearning/models/')
if not os.path.exists(dir_path):
    os.makdir(dir_path)
mnist_model_path = os.path.join(dir_path, str(modelID) + str(zeit) + '.h5')

# Log Dir 
log_dir = os.path.abspath('/home/phil/MachineLearning/logs/')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
model_log_dir = os.path.join(log_dir, str(modelID) + str(zeit))

def model_fn(optimizer, learning_rate):
    # Define the DNN 
    input_img = Input(shape=x_train.shape[1:])

    x = Conv2D(filters=15, kernel_size=7, padding='same')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(filters=10, kernel_size=5, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=15, kernel_size=5, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=7, kernel_size=5, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=20, kernel_size=10, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=7, kernel_size=5, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=15, kernel_size=7, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=7, kernel_size=5, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=20, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=256)(x)
    x = Activation('relu')(x)
    x = Dense(units=64)(x)
    x = Activation('relu')(x)
    x = Dense(units=num_classes)(x)

    y_pred = Activation('softmax')(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)

    model.summary()

    # Compile and train (fit) the model, afterwards evaluate the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    # save the model would go here
    return model

# Model params
epochs = 20
batch_size = 256
optimizer_candidates = [Adam]  #[Adam, RMSprop]
lr_candidates = [random.uniform(1e-4, 1e-3) for _ in range(10)]

param_distributions = {
    'optimizer': optimizer_candidates,
    'learning_rate': lr_candidates,
}

keras_clf = KerasClassifier(
    build_fn = model_fn,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 0)

rand_cv = RandomizedSearchCV(  # Mit der Cross-Valid
    estimator=keras_clf,
    param_distributions=param_distributions,
    n_iter=3,
    n_jobs=1,
    verbose=0,
    cv=3)

rand_result = rand_cv.fit(x_train, y_train)

# Summary
print('Best: %f using %s' % (rand_result.best_score_, rand_result.best_params_))

means = rand_result.cv_results_['mean_test_score']
stds = rand_result.cv_results_['std_test_score']
params = rand_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('Acc: %f (+/- %f) with: %r' % (mean, std, param))