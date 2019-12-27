
from numpy import exp, array, random, dot
import numpy as np

# source for information about neural nets & examples: http://neuralnetworksanddeeplearning.com/chap1.html 

class NeuralNode:
    n_inputs = 0 # number of inputs per neuron
    weights = [] # store the weights per input
    bias = 0
    output = 0

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
        return dot(inputs, self.weights) + self.bias
        
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
    # input = array of =inputs
    # layers = index of which layer it's on starting from 0 (to use the right weight matrix for that layer) 
    def feedForward(self, inputs, layerIdx): 
        if (layerIdx < len(self.layers)): # check layer is valid
            outputs = []
            currLayer = self.layers[layerIdx]
            for idx in range(currLayer.n_nodes): # loop thru all nodes in current layer
                currNeuron = currLayer.nodes[idx]
                neuronSum = sigmoid(currNeuron.sum(inputs)) # sum in node w/ activation
                currNeuron.output = neuronSum # used in backpropagation
                outputs.append(neuronSum)
            inputs = outputs
            return self.feedForward(inputs, layerIdx+1)
        else:
            return inputs
        
       # for each layer: sigmoid of dot product of the input vector with the weight matrix (multiplying all the activations by the weights) + the biases

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = list()
            if i != len(self.layers)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.layers[i + 1]:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron.output)
                    neuron.delta = errors[j] * sigmoidDerivative(neuron.output)
            # for j in range(len(layer)):
            #     neuron = layer[j]
            #     neuron.delta = errors[j] * sigmoidDerivative(neuron.output)
    
    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.layers)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron.output for neuron in self.layers[i - 1]]
            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron.weights[j] += l_rate * neuron.delta * inputs[j]
                neuron.weights[-1] += l_rate * neuron.delta

    # still need to do these
    def gradientDescent(self):
        return
    def backprop(self):  
        return

    # returns a 2d array of the data from the image
    # input = input from the camera as a string
    def readImageData(self, input):
        data = input.split(",") # convert the string to a list
        #input = np.reshape(input, (128,128)) # convert list into 128 x 128 2d array
        #data = data.tolist() # convert numpy array to list
        rtn = []
        for a in data:
            b = int(a)
            rtn.append(b)
        return rtn 
    
# sigmoid function: 1 / (1 + e^ -x) 
def sigmoid(x): 
    return 1.0 /(1.0 + exp(-x))

# derivative of sigmoid function
def sigmoidDerivative(x): 
    return sigmoid(x) * (1-sigmoid(x))  

# returns an array of 2d arrays, each 2d array is the data from one image
def readTrainingData(filename):
    f = open(filename, 'r') # open the file in read mode
    contents = f.read()
    contents = contents[72:]
    input = contents.split() # convert the string to a list 
    while 'image' in input:
        input.remove('image')
    input = np.reshape(input, (-1, 16385))
    input2 = []
    for entry in input:
        entry = np.delete(entry, 0) # delete the index of the image
        #entry = np.reshape(entry, (128, 128)) # convert 1d to 2d array
        entry = entry.tolist() #convert from numpy array to list
        input2.append(entry)
    return input2


'''
def test():
    network = NeuralNetwork(16384, 6825, 5, 1 )
    data = readTrainingData("imageOutput2.txt")
    data2 = []
    for a in data:
        b = [int(i) for i in a]
        data2.append(b)
    #print(type(data2[0][0]))
    output = network.feedForward(data2[0], 0);
    maxVal = max(output)
    idx = output.index(maxVal)
    print(idx)
    return

test()'''
# https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
