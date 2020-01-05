from numpy import exp, array, random, dot, sum, zeros, pad, amax, log arg
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
        imageMatrix = (imageMatrix / 255) - 0.5 # make values between -0.5 and 0.5                              #
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

        for im_region, i, j in self.iterate_regions(self.last_input):
        for f in range(self.n_filters):
            lossGradientFilters[f] += lossGradient[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * lossGradientFilters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
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
    def bpPool(self, imageMatrix, lossGradient):
        x, h, w = imageMatrix.shape
        h = h // 2
        w = w // 2
        newGradientLoss = zeros(self.lastPoolIn.shape) # same dimension as original image matrix
        
        for k in range(self.n_filters):
            for i in range(h): # iterates all possible size x size regions in the image
                for j in range(w):
                    tempSel = imageMatrix[k, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    f, h2, w2 = tempSel.shape
                    amax = amax(tempSel)
                    # loop through selection to get max pixel
                    for f2 in range(f):
                        for i2 in range(h2): 
                            for j2 in range(w2):
                                if tempSel[k2, i2, j2] = amax[j2]:
                                    newGradientLoss[f2, i * 2 + i2, j * 2 + j2] = lossGradient[f2, i, j]

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

        exp = exp(totals)
        return exp / sum(exp, axis=0) # shape: 1D array of size = n_nodes. each node value = probablity of node is correct
    
    '''
    Derive gradient for output
    '''
    def bpSoftMax(self, lossGradient, learn_rate):
        for i, gradient in enumerate(lossGradient):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2) ## S^2
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            
            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
        
            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t
        
            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
        
            return d_L_d_inputs.reshape(self.last_input_shape)
        # returns an array of  arrays, each one is the data from one image

    def readTrainingData(self, filename):
        f = open(filename, 'r') # open the file in read mode
        contents = f.read()
        contents = contents[72:]
        input = contents.split() # convert the string to a list 
        while 'image' in input:
            input.remove('image')
        input = np.reshape(input, (-1, 16385))
        rtn = []
        for entry in input:
            entry = np.delete(entry, 0) # delete the index of the image
            #entry = np.reshape(entry, (128, 128)) # convert 1d to 2d array
            entry = entry.tolist() #convert from numpy array to list
            entry = [int(x) for x in entry]
            rtn.append(entry)
        return rtn

###########################################

print('started')

loss = 0
num_correct = 0
# for loop
# forward
filter = Filter3x3(2)
out = filter.filter(array([[11, 12, 5, 2, 3 , 6], [15, 6, 10, 9, 3, 6], [10, 8, 12, 5, 3, 6], [12,2,9,6, 3, 12], [12,15,8,6,3, 6], [11,10,5,2,3, 4]]))
out = filter.pool(out, 2)
f, h, w = out.shape
out = filter.softmax(f * h * w, 6825, out) # array of probabilities



# input for softmax backprop input
gradient = np.zeros(6825)
gradient[label] = -1 / out[label]

#backward from here
filter.bpSoftMax(gradient, 1)

for i, (im, label) in enumerate(zip(test_images, test_labels)):
# Do a forward pass.
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc
    loss = -log(out[label]) # loss of accuracy for the correct answer
    acc = 1 if argmax(out) == label else 0 # 1 = correct, 0 = incorrect
    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0



                
            