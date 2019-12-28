from numpy import exp, array, random, dot, sum, zeros, pad, amax
#128x128
class Filter3x3:
    n_filters = 0
    filters = []
    weights = []
    biases = []
    def __init__(self, n_filters):
        self.n_filters = n_filters
        self.filters = random.randn(n_filters, 3, 3) / 9 # generate 3D array (3x3xn_filters)
    
    '''
    Takes 2d matrix of image and transforms it with all the 3x3 filters
    Outputs 3d array of transformed images
    '''
    def feedForward(self, imageMatrix): # input image 2d array/matrix
        imageMatrix = (imageMatrix / 255) - 0.5 # make values between -0.5 and 0.5                              #
        imageMatrix = pad(imageMatrix, (1, 1), 'constant') # pad 0s around
        h, w = imageMatrix.shape
        transformedImage = zeros((self.n_filters, h-2, w-2)) # same dimension as original image matrix
        for k in range(self.n_filters):
            for i in range(h-2): # iterates all possible 3x3 regions in the image
                for j in range(w-2):
                    temp3x3 = imageMatrix[i:(i+3), j:(j+3)] #selects 3x3 area using current indexes
                    transformedImage[k, i, j] = sum(temp3x3 * self.filters[k])
        return transformedImage

    '''
    Cuts down the size of image to get rid of redundant info
    '''
    def pool(self, imageMatrix, poolSize): # pool by size
        x, h, w = imageMatrix.shape
        h = h // poolSize
        w = w // poolSize
        transformedImage = zeros((self.n_filters, h, w)) # same dimension as original image matrix
        for k in range(self.n_filters):
            for i in range(h): # iterates all possible size x size regions in the image
                for j in range(w):
                    tempSel = imageMatrix[k, (i * poolSize):(i * poolSize + poolSize), (j * poolSize):(j * poolSize + poolSize)]
                    transformedImage[k, i, j] = amax(tempSel)
        return transformedImage

    '''
    Calculate the probablity of result
    '''
    def softmax(self, n_inputs, n_nodes, input):
        # generate weights and biases
        self.weights = random.randn(n_inputs, n_nodes) / n_inputs 
        n_inputs, n_nodes = self.weights.shape
        self.biases = zeros(n_nodes)
        input = input.flatten() # to 1d instead of 3d

        total = dot(input, self.weights) + self.biases
        ex = exp(total)
        return ex / sum(ex, axis=0) # softmax function for probablity

    print('MNIST CNN initialized!')

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass.
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

filter = Filter3x3(2)
out = filter.feedForward(array([[11, 12, 5, 2, 3 , 6], [15, 6, 10, 9, 3, 6], [10, 8, 12, 5, 3, 6], [12,2,9,6, 3, 12], [12,15,8,6,3, 6], [11,10,5,2,3, 4]]))
# print(test)
# print(test.shape)
out = filter.pool(out, 2)
# print(test)
# print(test.shape)
f, h, w = out.shape
out = filter.softmax(f * h * w, 6825, out) # array of probabilities
# print(test)
# print(test.shape)


loss = -np.log(out[label]) # loss of accuracy
acc = 1 if np.argmax(out) == label else 0 # 1 = correct, 0 = incorrect


                
            