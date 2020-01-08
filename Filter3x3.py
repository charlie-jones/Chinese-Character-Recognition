from numpy import exp, array, random, dot, sum, zeros, pad, amax, log, argmax, delete, reshape, newaxis, divide, subtract, multiply
import os
import pickle
#128x128
class Filter3x3:
    n_filters = 0
    filters = []
    weights = []
    biases = []
    # generate weights and biases
    # self.weights = random.randn(n_inputs, n_nodes) / n_inputs 
    # self.biases = zeros(n_nodes)

    lastInShape = []
    lastIn = []
    lastTotals = 0
    lastPoolIn = []
    lastFilterIn = []
    
    '''
    Takes 2d matrix of image and transforms it with all the 3x3 filters
    Outputs 3d array of transformed images
    '''
    def filter(self, imageMatrix): # input image 2d array/matrix
        imageMatrix = subtract(divide(imageMatrix, 255), 0.5) # make values between -0.5 and 0.5                              #
        #imageMatrix = pad(imageMatrix, (1, 1), 'constant') # pad 0s around
        self.lastFilterIn = imageMatrix
        h, w = imageMatrix.shape
        transformedImage = zeros((self.n_filters, h-2, w-2)) # same dimension as original image matrix
        for k in range(self.n_filters):
            for i in range(h-2): # iterates all possible 3x3 regions in the image
                for j in range(w-2):
                    temp3x3 = imageMatrix[i:(i+3), j:(j+3)] #selects 3x3 area using current indexes
                    transformedImage[k, i, j] = sum(temp3x3 * self.filters[k])
        return transformedImage
    
    '''
    Backward prop for filter
    '''
    def bpFilter(self, lossGradient, learn_rate):
        lossGradientFilters = zeros(self.filters.shape)
        h, w = self.lastFilterIn.shape
        for f in range(self.n_filters):
            for i in range(h-2): # iterates all possible size x size regions in the image
                for j in range(w-2):

                    tempSel = self.lastFilterIn[i:(i+3), j:(j+3)]
                    lossGradientFilters[f] += tempSel * lossGradient[f, i, j]

        # Update filters
        self.filters -= learn_rate * lossGradientFilters
        #1st layer -> return nothing
        return None

    '''
    Cuts down the size of image to get rid of redundant info
    '''
    def pool(self, imageMatrix): # pool by size of 2
        x, h, w = imageMatrix.shape
        h = h // 2
        w = w // 2
        self.lastPoolIn = imageMatrix
        transformedImage = zeros((self.n_filters, h, w)) # same dimension as original image matrix
        for k in range(self.n_filters):
            for i in range(h): # iterates all possible size x size regions in the image
                for j in range(w):
                    tempSel = imageMatrix[k, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    transformedImage[k, i, j] = amax(tempSel)
        return transformedImage

    '''
    Pooling back prop. reverse of pool()
    '''
    def bpPool(self, lossGradient):
        x, h, w = self.lastPoolIn.shape
        h = h // 2
        w = w // 2
        newGradientLoss = zeros(self.lastPoolIn.shape) # same dimension as original image matrix
        for i in range(h): # iterates all possible size x size regions in the image
            for j in range(w):
                tempPoolSel = self.lastPoolIn[0:self.n_filters, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                f, h2, w2 = tempPoolSel.shape
                maxSel = amax(tempPoolSel, axis=(1,2))
                # loop through selection to get max pixel
                for k in range(f):
                    for i2 in range(h2): 
                        for j2 in range(w2):
                            if tempPoolSel[k, i2, j2] == maxSel[k]:
                                newGradientLoss[k, i * 2 + i2, j * 2 + j2] = lossGradient[k, i, j]

        return newGradientLoss

    '''
    Calculate the probablity of result
    '''
    def softmax(self, input):

        self.lastInShape = input.shape # before flatten

        input = input.flatten()
        self.lastIn = input # after flatten

        inputLen, nodes = self.weights.shape

        totals = dot(input, self.weights) + self.biases
        self.lastTotals = totals

        ex = exp(totals)
        return ex / sum(ex, axis=0) # shape: 1D array of size = n_nodes. each node value = probablity of node is correct
    
    '''
    Derive gradient for output
    '''
    def bpSoftMax(self, lossGradient, learn_rate):
        for i, gradient in enumerate(lossGradient):
            if gradient == 0:
                continue

            # e^totals
            t_exp = exp(self.lastTotals)

            # Sum of all e^totals
            S = sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2) ## S^2
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            
            # Gradients of totals against weights/biases/input
            d_t_d_w = self.lastIn
            d_t_d_b = 1
            d_t_d_inputs = self.weights
        
            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t
        
            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[newaxis].T @ d_L_d_t[newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
        
            return d_L_d_inputs.reshape(self.lastInShape)

    def randWeightsBiases(self, n_filters, n_inputs, n_nodes):
        self.n_filters = n_filters
        self.filters = random.randn(n_filters, 3, 3) / 9 # generate 3D array (3x3xn_filters)
        self.weights = random.randn(n_inputs, n_nodes) / n_inputs
        self.biases = zeros(n_nodes)

    # save the weights of the network to a file named weights.txt (so we can save the weights after training the network)
    def saveWeights(self):
        f = open("weights.txt", "wb+") # create a new file if it doesn't already exist
        pickle.dump(self.weights, f)
        f.close()

    # read the saved weights from weights.txt and set the network's weights to those
    def readWeights(self):
        f = open("weights.txt", "rb")
        self.weights = pickle.load(f)
        f.close()
        
    # save biases to biases.txt
    def saveBiases(self):
        f = open("biases.txt", "wb+")
        pickle.dump(self.biases, f)
        f.close()
    # read biases from biases.txt and set network's biases to those
    def readBiases(self):
        f = open("biases.txt", "rb")
        self.biases = pickle.load(f)
        f.close()
####################################################
'''
Inputs 128x128 pixel array
Returns label where:
label 0 = 1
label 1 = 2
etc
'''
def getCharacter(character, filter):
    out = array(character, dtype='int')
    out = filter.filter(character)
    out = filter.pool(out)
    out = filter.softmax(out) # array of probabilities 
    return argmax(out)
'''
returns an array of  arrays, each one is the data from one image
'''
def readTrainingData(filename):
    f = open(filename, 'r') # open the file in read mode
    contents = f.read()
    input = contents.split('image ') # convert the string to a list 
    input.pop(0) # gets rid of weird character
    rtn = []
    for entry in input:
        entry = entry.split()
        entry = entry[1:]
        entry = reshape(entry, (128, 128)) # convert 1d to 2d array
        rtn.append(entry)
    return rtn

###########################################
def train():
    learning_rate = 0.005
    num_possible_inputs = 10
    num_filters = 5
    step_progress = 50

    print('started')
    loss = 0
    num_correct = 0
    filter = Filter3x3()
    filter.randWeightsBiases(num_filters, num_filters * 63 * 63, num_possible_inputs) # and random filters
    i = 1
    while i < 500: # = # of chars read
        label = 0
        for filename in os.listdir('images'):
            for character in readTrainingData('images/' + filename):
                # forward
                character = array(character, dtype='int')
                out = filter.filter(character)
                out = filter.pool(out)
                out = filter.softmax(out) # array of probabilities 

                l = -log(out[label])
                acc = 1 if argmax(out) == label else 0
                loss += l
                num_correct += acc

                # input for softmax backprop input
                gradient = zeros(num_possible_inputs)
                gradient[label] = -1 / out[label]

                #backward from here
                gradient = filter.bpSoftMax(gradient, learning_rate)
                gradient = filter.bpPool(gradient)
                gradient = filter.bpFilter(gradient, learning_rate)

                if i > 0 and i % step_progress == 0:
                    print(
                    '[Step %d] : Average Loss %.3f | Accuracy: %d%%' %
                    (i, loss / step_progress, num_correct / step_progress * 100)
                    )
                    loss = 0
                    num_correct = 0
                if i % 10 == 0:
                    print(str(i))
                i+=1
            label+=1
    filter.saveWeights()
    filter.saveBiases()
    print("done. saved")


# training Code for class (comment it before running flask app)

train()



                
            