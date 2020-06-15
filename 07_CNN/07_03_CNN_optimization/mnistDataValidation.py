import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MNIST:
    def __init__(self):
        # Load the dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train_ = None
        self.x_val = None
        self.y_train_ = None
        self.y_val = None
        # Convert to float32
        self.x_train = self.x_train.astype(np.float32)  
        self.y_train = self.y_train.astype(np.float32)  
        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # Reshape the x-Data to shape (num_examples, width, height, depth)
        self.x_train = np.expand_dims(self.x_train, axis=-1)  # 1 dim mehr für die depth info
        self.x_test = np.expand_dims(self.x_test, axis=-1)    
        # Save important data attributes as variables
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = 0
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.num_classes = 10                                 # np.max(self.y_train) +1
        # Reshape the y-Data to One-Hot encoding
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
    
    def get_train_set(self):
        return self.x_train, self.y_train
    
    def get_test_set(self):
        return self.x_test, self.y_test
    
    def get_splitted_train_validation_set(self):
        # train = 60.000
        # train_: 40.200, val: 19.800
        self.x_train_, self.x_val, self.y_train_, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.33)
        self.val_size = self.x_val.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val, self.y_train_, self.y_val
    
    def data_augmentation(self, augment_size=5000):
        # Create an instance of the image data genrator class
        image_generator = ImageDataGenerator(
            rotation_range=10,         # 15 Grad +/- drehen
            zoom_range=0.05,           # 10% zoomen
            width_shift_range=0.05,    # schieben hor. 10%
            height_shift_range=0.05,   # schieben vert. 10%
            fill_mode='constant',
            cval=0.0)
        # fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), 
                                            batch_size = augment_size, shuffle=False).next()[0] # next() gibt eine Liste zurück und die daten sind an stelle 0
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
    
    def data_preprocessing(self, preprocess_mode='standard'):
        # Preprcess the data
        if preprocess_mode == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(0 ,1))
        self.scaler.fit(self.x_train.reshape(self.train_size, 784))
        self.x_train = self.scaler.transform(self.x_train.reshape(self.train_size, 784))
        self.x_test = self.scaler.transform(self.x_test.reshape(self.test_size, 784))
        self.x_train = self.x_train.reshape(self.train_size, self.width, self.height, self.depth)
        self.x_test = self.x_test.reshape(self.test_size, self.width, self.height, self.depth)


if __name__ == '__main__':
    mnist = MNIST()
    mnist.data_augmentation()

    # img = mnist.x_train[0]
    # plt.imshow(img.reshape(28, 28), cmap='gray')
    # plt.show()

    # x_train, y_train = mnist.get_train_set()
    # print(np.max(x_train))  #255
    # print(np.min(x_train))  #0
    # print(np.mean(x_train)) #33,318
    # print(np.std(x_train))