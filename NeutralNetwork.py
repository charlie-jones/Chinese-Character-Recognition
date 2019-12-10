
from numpy import exp, array, random, dot

# source for information about neural nets & examples: http://neuralnetworksanddeeplearning.com/chap1.html 

class NeuralNode:
    n_inputs = 0 # number of inputs per neuron
    weights = [] # store the weights per input

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.setWeights(n_inputs)
    def setWeights(self, n_inputs):
        # create array of random weights btn 0 and 1
        self.weights = [random.uniform(0,1) for x in range(0,n_inputs)]
    def sum(self, inputs): # summation of weight * input
        return dot(inputs, weights)
        
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
    n_neurons_to_hl = 0
    
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
        self.layers = [NeuronLayer(self.n_neurons_to_hl,self.n_inputs)]
        # hidden layers
        self.layers += [NeuronLayer( self.n_neurons_to_hl,self.n_neurons_to_hl) for x in range(0,self.n_hidden_layers)]
        # hidden-to-output layer
        self.layers += [NeuronLayer(self.n_outputs,self.n_neurons_to_hl)]
    
        # weights matrix: each row is the weights for the connections from one neuron to the neurons in the next layer
        # there are n-1 weights matrices for n layers b/c there are no weights before the first layer
        # so weights[0] is the weights between the first and second layers
        # weights matrix dimensions are prev layer x next layer (# of rows = # of neurons in prev layer, # of cols is # of neurons in next layer)
        
        #this would switch the dimensions (the weights matrix would be next layer x prev layer, opposite of what we currently have)
        #leaving this here b/c not sure which will work so might need to switch it to this later
        #self.weights = [np.random.randn(y, x) for x, y in zip(self.neurons[:-1], self.neurons[1:])]


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
        return sigmoid(x) * (1-sigmoid(x))   
