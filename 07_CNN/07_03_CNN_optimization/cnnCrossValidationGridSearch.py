import os 
import time

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import cross_val_score

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from plotting import *
from mnistDataValidation import *

mnist = MNIST()
mnist.data_augmentation(augment_size=5000)
mnist.data_preprocessing(preprocess_mode='MinMax')
x_train, y_train = mnist.get_train_set()
x_test, y_test = mnist.get_test_set()
num_classes = mnist.num_classes

modelID = str('mnist_CNN_model_Augmentation_Std_PP_')
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

    x = Conv2D(filters=10, kernel_size=3, padding='same')(input_img)
    x = Activation('relu')(x)
    x = Conv2D(filters=5, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=5, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=5, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=32)(x)
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

    # tb = TensorBoard(
    #     log_dir=model_log_dir,
    #     histogram_freq=1,
    #     write_graph=True)

    return model

# Model params
epochs = 3
batch_size = 512
optimizer_candidates = [Adam, RMSprop]
lr_candidates = [1e-3, 5e-3, 1e-4]

param_grid = {
    'optimizer': optimizer_candidates,
    'learning_rate': lr_candidates,
}

grid = ParameterGrid(param_grid)  # Wenn ohne Cross-Validation
for comb in grid:
    print(comb)

# model.fit(
#     x_train_splitted, 
#     y_train_splitted, 
#     epochs=epochs,
#     batch_size=batch_size,
#     validation_data=[x_val, y_val],
#     callbacks=[tb]) <<------ wo kommt der hin ???

# score = model.evaluate(
#     x_test, 
#     y_test,
#     verbose=0)
# print('Score: ', score)

# model.save_weights(filepath=mnist_model_path)
# model.load_weights(filepath=mnist_model_path)

keras_clf = KerasClassifier(
    build_fn = model_fn,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 0)

grid_cv = GridSearchCV(  # Mit der Cross-Valid
    estimator=keras_clf,
    param_grid=param_grid,
    n_jobs=1,
    verbose=0,
    cv=3)

grid_result = grid_cv.fit(x_train, y_train)

# Summary
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('Acc: %f (+/- %f) with: %r' % (mean, std, param))