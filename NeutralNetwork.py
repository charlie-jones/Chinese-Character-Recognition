
from numpy import exp, array, random, dot

# source for information about neural nets & examples: http://neuralnetworksanddeeplearning.com/chap1.html 

class NeuralNode:
    n_inputs = 0 # number of inputs per neuron
    weights = [] # store the weights per input

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.setWeights(n_inputs)
        self.setBias()
    def setWeights(self, n_inputs):
        # create array of random weights btn 0 and 1
        self.weights = [random.uniform(0,1) for x in range(0,n_inputs)]
    def setBias(self):
        self.bias = random.uniform(0,1) # set bias to a random number
    def sum(self, inputs): # summation of weight * input
        return dot(inputs, self.weights)
        
class NeuralLayer:
    n_nodes = 0
    nodes = [] # array of nodes in layer
    def __init__(self, n_nodes, n_inputs):
        self.n_nodes = n_nodes
        # create n_nodes amount of NeuralNodes
        self.nodes = [NeuralNode(n_inputs) for x in range(0, n_nodes)]

class NeuralNetwork:
    n_inputs = 0
    n_outputs = 0
    n_hidden_layers = 0
    n_neurons_to_hl = 0 #num of neurons to hidden layer
    layers = []
    
    def __init__(self, n_inputs, n_outputs, n_neurons_to_hl, n_hidden_layers):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_to_hl = n_neurons_to_hl
        # biases and weights are stored in lists of numpy matrices 
        # biases matrix is an array of numpy arrays, each numpy array being the biases for that layer (biases[0] is an array of the biases for the first layer after the input)
        #self.biases = [np.random.randn(a, 1) for a in self.neurons[1:]] # there are no biases for the input layer 
        self.create()
        
    def create(self):
        # first layer
        self.layers = [NeuralLayer(self.n_neurons_to_hl,self.n_inputs)]
        # hidden layers
        self.layers += [NeuralLayer( self.n_neurons_to_hl,self.n_neurons_to_hl) for x in range(0,self.n_hidden_layers)]
        # hidden-to-output layer
        self.layers += [NeuralLayer(self.n_outputs,self.n_neurons_to_hl)]

    #  *** this might not completely work yet, just a first draft ***  
    # returns the output of the network given an input
    # output of a layer is: sigmoid of (weights*input + biases)
    # input = array of vectors of all neurons in given layer 
    # layer = index of which layer it's on starting from 0 (to use the right weight matrix for that layer) 
    def feedForward(self, inputs, layerIdx): 
        if layerIdx < len(self.layers) and len(inputs) == len(self.layers[layerIdx].neurons): # check layer is valid & have same # of inputs as neurons
            outputs = []
            for idx in range(len(inputs)):
                currNeuron = self.layers[layerIdx].neurons[idx]
                currInputs = inputs[idx]
                neuronSum = currNeuron.sum(currInputs) + currNeuron.bias # sum of that neuron + bias of that neuron
                neuronSum = sigmoid(neuronSum)
                outputs.append(neuronSum)
                inputs = outputs
            return self.feedForward(inputs, layerIdx+1)
        else:
            return input
        
       # for each layer: sigmoid of dot product of the input vector with the weight matrix (multiplying all the activations by the weights) + the biases


    # still need to do these
    def gradientDescent(self):
        return
    def backprop(self):  
        return

# sigmoid function: 1 / (1 + e^ -x) 
def sigmoid(x): 
    return 1.0 /(1.0 + exp(-x))

# derivative of sigmoid function
def sigmoidDerivative(x): 
    return sigmoid(x) * (1-sigmoid(x))   

# https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
