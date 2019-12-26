from numpy import exp, array, random, dot, sum, zeros, pad

class Filter3x3:
    n_filters = 0
    filters = []
    def __init__(self, n_filters):
        self.n_filters = n_filters
        self.filters = random.randn(n_filters, 3, 3) / 9 # generate 3D array (3x3xn_filters)
    
    '''
    Takes 2d matrix of image and transforms it with all the 3x3 filters
    Outputs 3d array of transformed images
    '''
    def feedForward(self, imageMatrix): # input image 2d array/matrix
        imageMatrix = pad(imageMatrix, (1, 1), 'constant') # pad 0s around
        h, w = imageMatrix.shape
        transformedImage = zeros((self.n_filters, h-2, w-2)) # same dimension as original image matrix
        for i in range(h-2): # iterates all possible 3x3 regions in the image
            for j in range(w-2):
                temp3x3 = imageMatrix[i:(i+3), j:(j+3)] #selects 3x3 area using current indexes
                for k in range(len(self.filters)):
                    transformedImage[k, i, j] = sum(temp3x3 * self.filters[k])
        return transformedImage
test = Filter3x3(2)
print(test.feedForward(array([[11, 12, 5, 2, 3], [15, 6, 10, 9, 3], [10, 8, 12, 5, 3], [12,15,8,6, 3], [12,15,8,6, 3]])))



                
            