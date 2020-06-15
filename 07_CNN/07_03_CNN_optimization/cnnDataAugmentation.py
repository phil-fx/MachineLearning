import os 
import time

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *

from plotting import *
from mnistDataAugmentation import *

mnist = MNIST()
mnist.data_augmentation(augment_size=5000)
mnist.data_preprocessing()
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

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 3
batch_size = 256 # Trade-Off Geschwind. // Performance ~2er-Potenz~ [32-1024]

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

model.summary()

# Compile and train (fit) the model, afterwards evaluate the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

tb = TensorBoard(
    log_dir=model_log_dir,
    histogram_freq=1,
    write_graph=True)

model.fit(
    x_train, 
    y_train, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[x_test, y_test],
    callbacks=[tb])

score = model.evaluate(
    x_test, 
    y_test,
    verbose=0)
print('Score: ', score)

model.save_weights(filepath=mnist_model_path)
model.load_weights(filepath=mnist_model_path)

