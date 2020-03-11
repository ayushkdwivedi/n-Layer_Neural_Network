import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pdb

class NeuralNetwork(object):    
 
    def __init__(self, size, actn_fn, weights=None, seed=42):
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.actn_fn = actn_fn
        if weights == None:
            self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(2 / self.size[i-1]) for i in range(1, len(self.size))]
            self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
        else:
            self.weights, self.biases = self.load_weights(weights)

    def activation(self, z, derivative=False, type=None):
        # Sigmoid activation function
        type = self.actn_fn
        if type == "sigmoid":
            if derivative:
                return self.activation(z, type="sigmoid") * (1 - self.activation(z, type="sigmoid"))
            else:
                return 1 / (1 + np.exp(-z))
        
        # ReLu activation function    
        if type == "relu":
            if derivative:
                z[z <= 0.0] = 0.
                z[z > 0.0] = 1.
                return z
            else:
                return np.maximum(0, z)
        
        # Tanh activation function
        if type == "tanh":
            if derivative:
                return self.activation(z, type="tanh") * (1 - self.activation(z, type="tanh"))
            else:
                return np.tanh(z)

    def softmax(self, z):
        # Stable softmax
        expz = np.exp(z - np.max(z))
        return expz/expz.sum(axis=0, keepdims=True)

    def cost_function(self, y_true, y_pred):
        # Cross Entropy Cost function
        samples = y_true.shape[1]
        y_true = y_true.reshape(-1,)
        one_hot_labels = np.zeros((samples, 10))
        for i in range(samples):  
            one_hot_labels[i, y_true[i]] = 1

        n = y_pred.shape[1]
        y_true = one_hot_labels.T
        
        return np.mean(np.sum(-y_true * np.log(y_pred+1e-100), axis=0))

    def cost_function_prime(self, y_true, y_pred):
        """
        Dervative of Cost Function with Softmax    
        """
        samples = y_true.shape[1]
        y_true = y_true.reshape(-1,)
        one_hot_labels = np.zeros((samples, 10))
        for i in range(samples):  
            one_hot_labels[i, y_true[i]] = 1
        
        y_true = one_hot_labels.T
        cost_prime = y_pred - y_true
        return cost_prime

    def forward(self, input):
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a  = self.activation(z)
            pre_activations.append(z)
            activations.append(a)
        a = self.softmax(z)
        activations[-1] = a
        return a, pre_activations, activations

    def compute_deltas(self, pre_activations, y_true, y_pred):
        delta_L = self.cost_function_prime(y_true, y_pred)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * self.activation(pre_activations[l],
                                                                                        derivative=True) 
            deltas[l] = delta
        return deltas

    def backpropagate(self, deltas, pre_activations, activations):
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def train(self, X, y, batch_size, epochs, learning_rate, validation_split=0.2):

        history_train_losses = []
        history_test_accuracies = []

        x_train, x_test, y_train, y_test = train_test_split(X.T, y.T, test_size=validation_split, )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T 

        epoch_iterator = range(epochs)
        best_weights_acc = 0

        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size ) - 1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

            train_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            print("Epoch {}/{}".format(e+1, epochs))
            
            for idx, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                train_loss = self.cost_function(batch_y, batch_y_pred)
                train_losses.append(train_loss)

                print("{}/{}".format(idx, n_batches), end='\r')

            # weight update
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            print("Training Loss:{}".format(sum(train_losses)/len(train_losses)))
            y_test_pred = self.predict(x_test)
            test_accuracy = accuracy_score(y_test.T, y_test_pred.T)
            test_accuracies.append(test_accuracy)
            
            print("Testing Accuracy:{}".format(test_accuracy))

            if test_accuracy > best_weights_acc:
                print("Saving weights")
                best_weights_acc = test_accuracy
                self.save_weights() 

            print()
            history_train_losses.append(np.mean(train_losses))
            history_test_accuracies.append(np.mean(test_accuracies))

        history = {'train_loss': history_train_losses, 
                   'test_acc': history_test_accuracies}
        return history

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation(z)
        a = self.softmax(z)
        predictions = np.argmax(a, axis=0).reshape(1, -1).astype(int)
        return predictions

    def save_weights(self):
        model = {"size":self.size}
        for i in range(1, len(self.size)):
            model["W"+str(i)] = self.weights[i-1]
            model["b"+str(i)] = self.biases[i-1]
        np.save('best_acc_weights.npy', model)
        return None

    def load_weights(self, weights):
        model = np.load(weights).item()
        size = model["size"]

        if size == self.size:
            weights = [model["W"+str(i)] for i in range(1, len(size))]
            biases = [model["b"+str(i)] for i in range(1, len(size))]
        else:
            raise ValueError("Model dimension doesn't matches")
        return weights, biases


def load_data(path):

    data = pd.read_csv(path, delimiter=',')
    data = data.values
    X = data[:, 1:]
    y = data[:, 0]

    test_data = pd.read_csv('./Apparel/apparel-test.csv', delimiter=',')
    test_data = test_data.values
    X_test = test_data[:]

    return X, y, X_test

if __name__ == "__main__":

    path = './Apparel/apparel-trainval.csv'
    X, y, X_test = load_data(path)
    X = X/255.
    X_test = X_test/255.
    
    # with open('predictions.csv', 'w') as myfile:
    #     wr = csv.writer(myfile)
    #     wr.writerow(predictions)
    # pdb.set_trace()

    X, y = X.T, y.reshape(1, -1)
    X_test = X_test.T
  
    print("Data Loaded")
    print("Starting")

    # Hyperparameters
    num_classes = 10
    batch_size = 64
    activation_fn = "relu"
    epochs = 40
    lr = 0.0001
    hidden_layers = [256,64]
    size = [X.shape[0]] + hidden_layers
    size.append(num_classes)

    neural_net = NeuralNetwork(size=size, actn_fn=activation_fn, seed=42,weights='best_acc_weights71.npy')
    history = neural_net.train(X=X, y=y, batch_size=batch_size, epochs=epochs, 
                               learning_rate=lr, validation_split=0.2)
    history_train_losses = history["train_loss"]
    history_test_accuracies = history["test_acc"]

    plt.subplot(1,2,1)
    plt.plot(list(range(1, epochs+1)), history_train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")

    plt.subplot(1,2,2)
    plt.plot(list(range(1, epochs+1)), history_test_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.show()
