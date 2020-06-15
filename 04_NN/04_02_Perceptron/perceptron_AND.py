import matplotlib.pyplot as plt 
import numpy as np 

def get_dataset():
    x = np.array([[0,0], [1,0], [0,1], [1,1]])
    y = np.array([0, 0, 0, 1])
    return x, y

class Perceptron():
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.w = []
        self.lr = lr 
    
    def train(self, x, y):
        N, dim = x.shape ## 4 x 2
        # Init model
        self.w = np.random.uniform(-1, 1, (dim, 1))   # (1, dim) ?? # Gleichverteiulung [-1,1]. 2 Weights
        print('Zufälliges Anfangsgewicht: ', self.w)
        #Traning
        error = 0.0
        for epoch in range(self.epochs):
            choice = np.random.choice(N) ## Zufälliges X gezogen
            x_i = x[choice]
            y_i = y[choice]
            y_hat = self.predict(x_i)
            #check if the classifiaction is false
            if y_hat != y_i:
                error += 1
                self.update_weights(x_i, y_i, y_hat)
            # print('Gewichtsveränderung innerhalb der Rechnung: ', self.w)
        print('Training Misclassification-Rate: ', error / y.shape[0])
        print('Gewicht nach Training: ', self.w)

    def test(self, x, y):
        y_pred = np.array([self.predict(x_i) for x_i in x])
        acc = sum(1 for y_p, y_i in zip(y_pred, y) if y_p == y_i) / y.shape[0]
        print('Testgenauigkeit: ', acc)
        return acc

    def update_weights(self, x, y, y_hat):
        for i in range(self.w.shape[0]):
            ## aus der fehlerfkt--> d/dx (y-y_hat)² = 2*(y-y_hat) * d/dx y_hat= x
            ## dann geichtet mit der lernrate lr --> lr * (y-y_hat) * x[i] -- die 2 ist in der lr 
            delta_w_i = self.lr * (y - y_hat) * x[i]
            self.w[i] = self.w[i] + delta_w_i

    def activation(self, signal):
        if signal > 1.5:
            return 1
        else:
            return 0

    def predict(self, x):
        input_signal = np.dot(self.w.T, x) ## dot: [1, 2]*[3, 4] = 1*2 +3*4 ## self.w
        output_signal = self.activation(input_signal)
        return output_signal

x, y = get_dataset()
lr = 0.2
epochs = 100

p = Perceptron(epochs=epochs, lr=lr)
p.train(x, y)
p.test(x, y)