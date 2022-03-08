import numpy as np

# A max pooling layer using a pool size of 2
class MaxPool2:

  # Will yield all possible 2x2 segments of the input image
  # image is a 2d numpy array
  def iterateRegions(self, image):
    h, w, _ = image.shape #the _ is because it's a 3d array and we don't need it
    
    # Get the size of the output image, pool size is 2 so divide by 2
    newH = h//2 # '//' returns an int while '/' returns a float
    newW = h//2
  
    for i in range(newH):
      for j in range(newW):
        imRegion = image[(i * 2):(i * 2 + 2), (j * 2):(j*2 + 2)]
        yield imRegion, i, j

  #Performs a forward pass of the maxpool layer using the given input (image + filters).
  #Returns a 3d numpy array with dimensions (h/2, w/2, numFilters)
  def forward(self, input):
    h, w, numFilters = input.shape

    #Initialize the output 3d array with the proper size
    output = np.zeros((h // 2, w // 2, numFilters))

    for imRegion, i, j in self.iterateRegions(input):
      #np.amax returns the max number of an array given in any axis
      output[i,j] = np.amax(imRegion, axis=(0,1))

    return output