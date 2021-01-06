'''
Design of a Neural Network from scratch
'''

'''Importing required libraries'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

np.random.seed(45)

'''Neural Network Class'''
class NN:

    '''
    init function
    layers - a list of integers of length n, each integer corresponds to the number of neurons in that particular layer
    example - [8,4,5,1] has 1 input layer with 8 neurons, 2 hidden layers with 4,5 neurons respectively, 1 output layer with 1 neuron
    lr - learning rate
    epochs - number of epochs to train the nn for
    '''
    def __init__(self,layers,lr,epochs):
        self.layers=layers
        self.lr=lr
        self.epochs=epochs
    '''
    Initial weights generation with random values b/w [-1,1]
    '''
    def InitializeWeights(self):
        nodes=self.layers
        layers, weights = len(nodes), []
        for i in range(1, layers):
            w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
                  for j in range(nodes[i])]
            weights.append(np.matrix(w))
        return weights

    '''
    Function that trains the neural network by taking x_train and y_train samples as input
    '''
    def fit(self,X,Y):
        hidden_layers = len(self.layers) - 1
        weights = self.InitializeWeights()
        '''Trains the model for epochs=epochs'''
        for epoch in range(1, epochs+1):
            weights = self.Train(X_train, Y_train,weights)
            '''Prints training accuracy for every 50 epochs'''
            if(epoch % 50 == 0):
                print("Epoch {}".format(epoch))
                print("Training Accuracy:{}".format(self.Accuracy(X_train, Y_train, weights)))
        return weights
    '''
    Function that trains our model- calls ForwardPropagation and BackPropagation to periodically update weights
    '''
    def Train(self,X, Y,  weights):
        n = len(weights)
        for i in range(len(X)):
            x, y = X[i], Y[i]
            x = np.matrix(np.append(1, x))
            activations = self.ForwardPropagation(x, weights, n)
            weights = self.BackPropagation(y, activations, weights, n)
        return weights

        '''
        Function that returns overall accuracy
        '''
    def Accuracy(self,X, Y, weights):
        correct = 0

        for i in range(len(X)):
            x, y = X[i], list(Y[i])
            guess = self.Predict(x, weights)
            '''If guessed correctly'''
            if(y == guess):
                correct += 1

        return correct / len(X)
    '''
    Function that returns the predicted values(lhat) for a set of samples, uses Predict function to predict value for each sample
    input - X_train / X_test , weights
    '''
    def Predict_test(self,X,weights):
        correct = 0
        l=[]
        for i in range(len(X)):
            x= X[i]
            guess = self.Predict(x, weights)

            #l.append(guess[0])
            l.append(guess)

        return l
    '''
    Function that returns the predicted value for a given input data sample X
    input - X_train / X_test , weights
    '''
    def Predict(self,X,weights):
        n = len(weights)
        X = np.append(1, X)

        '''Forward Propagation'''
        activations = self.ForwardPropagation(X, weights, n)
        outputFinal = activations[-1].A1
        index = self.FindMaxActivation(outputFinal)

        '''Initialize prediction vector to zero'''
        yhat = [0 for i in range(len(outputFinal))]
        '''Set guessed class to 1'''
        yhat[index] = 1
        '''Returns prediction vector'''
        return yhat
        

    def BackPropagation(self,y, activations, weights, n):#his funciton is used to propogate errors backward through the layers and continuously update parameters like the weight and bias
        outputFinal = activations[-1]
        error = np.matrix(y - outputFinal)

        for j in range(n, 0, -1):
            currActivation = activations[j]

            if(j > 1):
                prevActivation = np.append(1, activations[j-1])
            else:
                prevActivation = activations[0]

            delta = np.multiply(error, self.SigmoidDerivative(currActivation))
            weights[j-1] += lr * np.multiply(delta.T, prevActivation)

            w = np.delete(weights[j-1], [0], axis=1)
            error = np.dot(delta, w)

        return weights
    def ForwardPropagation(self,x, weights, layers):#this function is used to pass on the output from the previous layers to the consecutive layers. it ultimately gives us our final prediction from the last layer
        activations, layer_input = [x], x
        for j in range(layers):
            activation = self.Sigmoid(np.dot(layer_input, weights[j].T))
            activations.append(activation)
            layer_input = np.append(1, activation)

        return activations
    def FindMaxActivation(self,output):#this function is used to find the largest output- the remaining become 0
        m, index = output[0], 0
        for i in range(1, len(output)):
            if(output[i] > m):
                m, index = output[i], i

        return index
    '''Sigmoid function'''
    def Sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

        '''Sigmoid derivative function'''
    def SigmoidDerivative(self,x):
        return np.multiply(x, 1-x)

    '''Predict function given in the template'''
    def predict(self,X):
        return yhat

        '''
    Prints confusion matrix
    y_test is list of y values in the test dataset
    y_test_obs is list of y values predicted by the model
        '''
    def CM(self,y_test,y_test_obs):

        for i in range(len(y_test_obs)):
            if(y_test_obs[i][1]>0.6):
                y_test_obs[i][1]=1
            else:
                y_test_obs[i][1]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i][1]==1 and y_test_obs[i][1]==1):
                tp=tp+1
            if(y_test[i][1]==0 and y_test_obs[i][1]==0):
                tn=tn+1
            if(y_test[i][1]==1 and y_test_obs[i][1]==0):
                fp=fp+1
            if(y_test[i][1]==0 and y_test_obs[i][1]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")

'''Loading the preprocessed csv file to the dataframe'''
df = pd.read_csv(r'../data/preprocessed.csv')

'''The feature vector X'''
X = df[['Community','Age','Weight','Delivery phase','HB','IFA','BP','Residence']]
X = np.array(X)

'''The result vector Y'''
Y = df.Result
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

'''Train test split'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

f = len(X[0]) # number of features
o = len(Y[0]) # number of output classes

layers = [f,4,5,o] # number of neurons in layers
lr, epochs = 0.20, 600 #learning rate, number of epochs

'''Creating object of class NN'''
nn=NN(layers,lr,epochs)

'''fit function being called'''
weights=nn.fit(X_train,Y_train)

'''Calculating and printing testing accuracy'''
print("Testing Accuracy: {}".format(nn.Accuracy(X_test, Y_test, weights)))

'''Calling the Predict_test function to get the predicted values of test data'''
Y_test_obs=nn.Predict_test(X_test,weights)

'''Calling the cost matrix function'''
nn.CM(Y_test,Y_test_obs)
