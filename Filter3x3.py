from numpy import exp, array, random, dot, sum, zeros, pad
from PIL import Image

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
        h, w = imageMatrix.shape
        transformedImage = np.zeros((h - 2, w - 2, self.n_filters))
        for i in range(h - 2): # iterates all possible 3x3 regions in the image
            for j in range(w - 2):
                temp3x3 = imageMatrix[i:(i+3), j:(j+3)] #selects 3x3 area using current indexes
                transformedImage[i, j] = sum(temp3x3, self.filters, axis=(1,2))
        return transformedImage




                
            