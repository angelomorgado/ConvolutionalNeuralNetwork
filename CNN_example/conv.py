import numpy as np

#A convolution layer using 3x3 filters
class Conv3x3:
  def __init__(self, numFilters):
    self.numFilters = numFilters

    #Filters is a 3d array with dimensions (num_filters, 3,3)
    #We divide by 9 to reduce the variance of out initial values
    #(so the values aren't too different, we can consider this normalization)
    self.filters = np.random.randn(numFilters, 3, 3) / 9

  def iterateRegions(self,image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array
    '''
    h,w = image.shape

    #Iterate output image
    for i in range(h - 2):
      for j in range(w - 2):
        imgRegion = image[i:(i+3), j:(j + 3)] #it's a 2d array that contains the portion of the img
        yield imgRegion, i, j #yield is used to give values to the for function sequentially

  #Get the output image from the input image
  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input. Returns a 3d numpy array
    with dimensions(h,w,num_filters).
    Input is a 2d numpy array
    '''
    #Shape returns the height and width of the matrix in a tuple (height, width)
    h,w = input.shape

    #Create the output matrix that will have minus 2 because it's valid padding
    output = np.zeros((h - 2, w - 2, self.numFilters))

    #iterate_regions will give all regions of the image (2d array)
    for imgRegion, i, j in self.iterateRegions(input):
      output[i,j] = np.sum(imgRegion * self.filters, axis = (1, 2))

    return output
      
      
    