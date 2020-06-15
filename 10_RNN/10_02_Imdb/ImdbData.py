import numpy as np 

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence

class IMDBData:
    def __init__(self, num_words, skip_top, maxlen):
        self.num_classes = 2
        self.num_words = num_words
        self.skip_top = skip_top
        self.maxlen = maxlen
        
        # word index: Word -> Index
        self.word_index = imdb.get_word_index() # wörterbuch
        self.word_index = {key: (val+3) for key, val in self.word_index.items()}
        self.word_index['<PAD>'] = 0
        self.word_index['<START>'] = 1  # start_char siehe doku des dataset
        self.word_index['<UNK>'] = 2    # oov_char
        #print(self.word_index)
        #print(len(self.word_index))

        # Index -> Word
        self.word_index = {val: key for key, val in self.word_index.items()}
        # print(self.word_index)

        # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(
            num_words = self.num_words,   # wieviele worde aus dem dictonary genommen werden
            skip_top = self.skip_top)     # die top-wörter (I, the, a, ...) werden geskipped
        # print(self.x_train[0])
        # print(self.y_train[0])

        # Save texts
        self.x_train_text = np.array(
            [[self.word_index[index] for index in review] for review in self.x_train])
        self.x_test_text = np.array(
            [[self.word_index[index] for index in review] for review in self.x_test])
        
        # Pad sequences
        print(self.x_train.shape)  # (25000,)
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen) 
        print(self.x_train.shape) # nach padding (25000,80) auch kürzere rezensionen werden auf maxlen (=80) aufgefüllt, <PAD>=0
        self.test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)

        # Save dataset size
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

        # create One-hot array for class labels
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)
    
    def get_review_text(self, review):
        review_text = [self.word_index[index] for index in review]
        return review_text

if __name__ == '__main__':
    num_words = 10000
    skip_top = 100
    maxlen = 80
    imdb_data = IMDBData(num_words, skip_top, maxlen)
    review_text = imdb_data.get_review_text(imdb_data.x_train[0])
    #print(review_text)
    print(imdb_data.x_train[1337])
    print(imdb_data.y_train[1337])

