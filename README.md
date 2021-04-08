# Neural-Network-from-scratch
Designing a neural network from scratch using numpy.

Implementation:
1) Preprocessing:
	Age: replace nan with mode as a large number of ages were in and around the early 20s
	Weight: group by age and replace nan with median weight of corresponding age - this was decided as people of the same age were more likely to be closer together in weight than a wider age window
	Delivery phase: as most candidates were in phase-1, it was a safe estimate to replace nan with mode
	HB: the haemoglobin levels were found to bear a relation with age, and therefore, we replaced nan with median hb of corresponding age
	BP: nan values were replaced with median
	Education: was found to be 5 in all cases, and was inconsequential to our outcome, so we replaced nan with mode
	Residence: the best design was to group by community and replace nan with mode residence of corresponding community as people from the same community were more likely to live a similar lifestyle
2) Class NN:
	used to define the features and functionalities of the neural network
3) Initialization : 
	The weights and biases to be used in neural network are initialized with random samples from uniform distribution which lies in the range (-1,1).
4) Forward propagation :
	The input given to the input layer is passed through each of the other 3 layers to obtain the predicted output.
	The hidden layers and the output layer have a summation(Z) and an activation function (A).
	The outputs of the previous layer are given as the input to the next layer.
	The inputs are multiplied with the randomly initialized weights to which the biases are added to obtain the output for the summation function Z.
	The outputs of the summation function are passed as the inputs for the activation function which is the sigmoid function.
	The outputs of the sigmoid function is the final output of the current layer.
	The output of the final layer is the prediction of the network.
5) Backward propagation:
	The error at the output layer is calculated as
 ½(true_value – predicted_value)^2. The derivative of the error with respect to true_value is = (true_value – predicted_value)
	That error is propagated backwards through the layers to update the weights and the biases.
	delta : It is calculated by multiplying the back propagated error of the current layer and the sigmoid derivative of the output of the current layer’s activation function.
	The weights between the current and previous layer are updated as:
W = W – lr * ( delta * activation of previous layer)
	The error for current layer is calculated by  removing the bias from the weights of the previous layer and multiplying the result with delta .
	Finally, the newly calculated weights are returned.
4) Train: 
	For every row in the dataset which is passed as the input to the neural network, initialization of weights and biases, forward 		propagation and backward propagation are implemented
5) FindMaxActivation :
	The largest of the outputs obtained are taken and the corresponding index is set to 1, while the rest are set to 0.
6) Predict:
	This involves forward propagation of the input and the conversion of the output to one-hot encoding, where the 1 denoting the predicted class.
7) Fit :
	Used to train the neural network epoch number of times.
	It takes X_train and Y_train samples as input
8) Accuracy :
	Used to obtain the training and testing accuracy of the neural network .
	Gives the percentage of rightly predicted outputs.
9) Confusion Matrix:
	implemented using def CM(self,y_test,y_test_obs)
	was built using the actual values from the dataset and the predicted values of the model
	the function was also used to compute precision, recall and F1 scores as performance measures

Hyperparameters:
	epochs = 600
	Learning Rate lr = 0.2
	Number of layers in the network = 4
	Number of neurons in each of the 4 layers = [9,4,5,1]

Design features:
	Input layer: Has 8 neurons
	Hidden layers: There are two hidden layers. The first has 4 neurons while the second has 5
	Output layer: has one neuron
	Activation function used: sigmoid
	Perhaps the most profound feature of our implementation is the that all our preprocessing was based on scientifically validated facts and correlation between parameters
	
Run the following commands within src directory:
	1)python3 preprocess.py
	2)python3 main.py
NOTE: The code was written on systems supporting a Linux based OS and path specifications may be OS specific. Any errors arising in non-Linux based OSes may be due to this reason
