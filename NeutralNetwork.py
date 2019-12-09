
from numpy import exp, array, random, dot

# source for information about neural nets & examples: http://neuralnetworksanddeeplearning.com/chap1.html 

class NeuralNode():
    n_inputs = 0 # number of inputs per neuron
    weights = [] # store the weights per input

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.setWeights(n_inputs)
    def setWeights(self, n_inputs):
        # create array of random weights btn 0 and 1
        self.weights = [random.uniform(0,1) for x in range(0,n_inputs)]

        
class NeuralLayer():
    n_nodes = 0
    nodes = [] # array of nodes in layer
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        # create n_nodes amount of NeuralNodes
        self.nodes = [NeuralNode(n_inputs) for _ in range(0, n_nodes)]

class NeuralNetwork(object):

    def __init__(self):
        # rand num gen 

        # # of layers (can set this to whatever we want later, could add a parameter to the constructor for this, etc)
        self.numLayers = 3

        # # of neurons in each layer (can change this later) 
        # array of size self.numLayers where neurons[0] is the # of neurons in the first layer
        self.neurons = [3,4,3]
        
        # biases and weights are stored in lists of numpy matrices 
        # biases matrix is an array of numpy arrays, each numpy array being the biases for that layer (biases[0] is an array of the biases for the first layer after the input)
        self.biases = [np.random.randn(a, 1) for a in self.neurons[1:]] # there are no biases for the input layer 
        
    
        # weights matrix: each row is the weights for the connections from one neuron to the neurons in the next layer
        # there are n-1 weights matrices for n layers b/c there are no weights before the first layer
        # so weights[0] is the weights between the first and second layers
        self.weights = [np.random.randn(self.numLayers, a) for a in self.neurons[1:]] 




    #  *** this might not completely work yet, just a first draft ***  
    # returns the output of the network given an input
    # output of a layer is: sigmoid of (weights*input + biases)
    # input = vector of all the activations from a given layer (or the initial input vector when the function is first called)
    # layer = index of which layer it's on starting from 0 (to use the right weight matrix for that layer) 
    def feedForward(self, input, layer): 
        if layer > (self.numLayers-1): # b/c self.numLayers is the # of layers counting from 1 not 0
            return input
        
        else: # for each layer: sigmoid of dot product of the input vector with the weight matrix (multiplying all the activations by the weights) + the biases
            rtn = sigmoid(np.dot(input, self.weights[layer]) + self.biases[layer]) 
            return self.feedForward(rtn, layer+1)

    # still need to do these
    def gradientDescent(self):
        return
    def backprop(self):  
        return

    # sigmoid function: 1 / (1 + e^ -x) 
    def sigmoid(x): 
        return 1.0 /(1.0 + exp(-x)) # this still works when the input x is a numpy array; it's applied to each element in the array individually

    # derivative of sigmoid function
    def sigmoidDerivative(x): 
        return sigmoid(x) * ( 1-sigmoid(x))   
