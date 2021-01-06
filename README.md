# Neural-Network-from-scratch-
Designing a neural network from scratch using numpy.

Implementation:

•	The class ‘NN’ is implemented where the features and functionalities of the neural network is defined.

•	Initialization : 
->The weights and biases to be used in neural network are initialized here.
-> They are initialized with random samples from uniform distribution which lies in the range (-1,1).

•	Forward propagation :
->The input given to the input layer is passed through each of the other 3 layers to obtain the predicted output.
->The hidden layers and the output layer have a summation(Z) and an activation function (A).
->The outputs of the previous layer are given as the input to the next layer.
->The inputs are multiplied with the randomly initialized weights to which the biases are added to obtain the output for the summation function Z.
->The outputs of the summation function are passed as the inputs for the activation function which is the sigmoid function.
->The outputs of the sigmoid function is the final output of the current layer.
->The output of the final layer is the prediction of the network.

•	Backward propagation:
->The error at the output layer is calculated as
 ½(true_value – predicted_value)^2. The derivative of the error with respect to true_value is = (true_value – predicted_value)
->That error is propagated backwards through the layers to update the weights and the biases.
-> delta : It is calculated by multiplying the back propagated error of the current layer and the sigmoid derivative of the output of the current layer’s activation function.
-> The weights between the current and previous layer are updated as:
W = W – lr * ( delta * activation of previous layer)
-> The error for current layer is calculated by  removing the bias from the weights of the previous layer and multiplying the result with delta .
->Finally, the newly calculated weights are returned.

•	Train: 
->For every row in the dataset which is passed as the input to the neural network, initialization of weights and biases, forward propagation and backward propagation are implemented

•	FindMaxActivation :
-> The largest of the outputs obtained are taken and the corresponding index is set to 1, while the rest are set to 0.


•	Predict:
->  This involves forward propagation of the input and the conversion of the output to one-hot encoding, where the 1 denoting the predicted class.

•	Fit :
->Used to train the neural network epoch number of times.
-> It takes X_train and Y_train samples as input

•	Accuracy :
->Used to obtain the training and testing accuracy of the neural network .
->Gives the percentage of rightly predicted outputs.


Hyperparameters: epoch = 600, Learning Rate lr = 0.2, Number of layers in the network = 4, Number of neurons in each of the 4 layers = [9,4,5,1]
