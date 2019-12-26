from numpy import exp, array, random, dot

class Filter3x3:
    n_filters = 0
    filters = []
    def __init__(self, n_filters):
        self.n_filters = n_filters
        self.filters = random.randn(n_filters, 3, 3) / 9 # generate 3D array (3x3xn_filters)
    
    def iterateImage(self, image): # input image
        h, w = image.shape
        for i in range(h - 2): # gets all possible 3x3 regions in the image
            for j in range(w - 2):
                image[i:(i+3), j:(j+3)] #selects 3x3 area using current indexes

                
            