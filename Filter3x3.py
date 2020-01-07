from numpy import exp, array, random, dot, sum, zeros, pad, amax, log, argmax, delete, reshape, newaxis, divide, subtract
import os
#128x128
class Filter3x3:
    n_filters = 0
    filters = []
    weights = []
    biases = []

    lastInShape = []
    lastIn = []
    lastTotals = 0
    lastPoolIn = []
    lastFilterIn = []

    
    def __init__(self, n_filters):
        self.n_filters = n_filters
        self.filters = random.randn(n_filters, 3, 3) / 9 # generate 3D array (3x3xn_filters)
    
    '''
    Takes 2d matrix of image and transforms it with all the 3x3 filters
    Outputs 3d array of transformed images
    '''
    def filter(self, imageMatrix): # input image 2d array/matrix
        imageMatrix = subtract(divide(imageMatrix, 255), 0.5) # make values between -0.5 and 0.5                              #
        imageMatrix = pad(imageMatrix, (1, 1), 'constant') # pad 0s around
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
        h = h // 2
        w = w // 2
        for f in range(self.n_filters):
            for i in range(h): # iterates all possible size x size regions in the image
                for j in range(w):
                    tempSel = self.lastFilterIn[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    print(lossGradient[f, i, j])
                    print(tempSel)
                    lossGradientFilters[f] += (lossGradient[f, i, j] * tempSel)

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
        
        for k in range(self.n_filters):
            for i in range(h): # iterates all possible size x size regions in the image
                for j in range(w):
                    tempSel = self.lastPoolIn[k, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    h2, w2 = tempSel.shape
                    maxSel = amax(tempSel)
                    # loop through selection to get max pixel
                    for i2 in range(h2): 
                        for j2 in range(w2):
                            if tempSel[i2, j2] == maxSel:
                                newGradientLoss[k, i * 2 + i2, j * 2 + j2] = lossGradient[k, i, j]

        return newGradientLoss

    '''
    Calculate the probablity of result
    '''
    def softmax(self, n_inputs, n_nodes, input):
        # generate weights and biases
        self.weights = random.randn(n_inputs, n_nodes) / n_inputs 
        self.biases = zeros(n_nodes)

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
        # returns an array of  arrays, each one is the data from one image
###############################################
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

print('started')
loss = 0
num_correct = 0
for filename in os.listdir('images'):
    i = 1
    label = 0
    for character in readTrainingData('images/' + filename):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0
        
        filter = Filter3x3(4)
        # forward
        character = array(character, dtype='int')
        out = filter.filter(character)
        out = filter.pool(out)
        f, h, w = out.shape
        out = filter.softmax(f * h * w, 10, out) # array of probabilities  # 6825
        
        l = -log(out[label])
        acc = 1 if argmax(out) == label else 0
        loss += l
        num_correct += acc

        # input for softmax backprop input
        gradient = zeros(10) # 6825
        gradient[label] = -1 / out[label]

        #backward from here
        gradient = filter.bpSoftMax(gradient, 1)
        gradient = filter.bpPool(gradient)
        gradient = filter.bpFilter(gradient, 1)
        print('done label: ' + str(label))
        label+=1



                
            