import numpy as np

class NeuralNet(object):

    def __init__(self): 
        # # of layers (can set this to whatever we want later, could add a parameter to the constructor for this, etc)
        self.layers = 3
        # # of neurons in each layer (can change this later) 
        # (array of size layers where layers[0] is the # of neurons in the first layer)
        self.neurons = [3,3,3]
        
        # biases and weights are stored in lists of numpy matrices (2d arrays)
        # as in weights is a list of numpy matrices, each matrix being the set of weights between two layers 
        # (each row being weights from a neuron to each of the other neurons)
        self.biases = [np.random.randn(layers, a) for a in neurons]
        self.weights = [np.random.randn(layers, a) for a in neurons] 
        

    # returns the output of a neuron given an input, input being a vector of the inputs for each neuron
    # given input, output of a layer is: sigmoid of (sum of weights*input + biases)
    def feedForward(self, input): 
        
        
        
    def gradientDescent(self):
    
    
    def backprop(self): 
       
    def sigmoid(self, x): # sigmoid function: 1 / (1 + e^ -x)
        return 1.0 /(1.0 + np.exp(-x))
    
    def sigmoidDerivative(self, x):
        return sigmoid(x) * ( 1-sigmoid(x)) # this is correct but i just looked up how to do it