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


  #========================================== Forward ====================================
  '''
  Performs a forward pass of the maxpool layer using the given input (image + filters).
  Returns a 3d numpy array with dimensions (h/2, w/2, numFilters)
  '''
  def forward(self, input):
    h, w, numFilters = input.shape

    #Store the input values in cache {NEW}
    self.lastInput = input
    
    #Initialize the output 3d array with the proper size
    output = np.zeros((h // 2, w // 2, numFilters))

    for imRegion, i, j in self.iterateRegions(input):
      #np.amax returns the max number of an array given in any axis
      output[i,j] = np.amax(imRegion, axis=(0,1))

    return output

  #========================================== Backprop ====================================
  '''
  Performs a backward pass of the maxpool layer.
  Returns the loss gradient for this layer's inputs.
  - dL_dOut is the loss gradient for this layer's outputs, received from the softmax layer
  '''
  def backprop(self, dL_dOut):
    dL_dInput = np.zeros(self.lastInput.shape) # Create an array with the same shape as the input

    #Go through the copy of the input we saved in cache
    for imgRegion, i, j in self.iterateRegions(self.lastInput):
      h, w, f = imgRegion.shape
      amax = np.amax(imgRegion, axis=(0, 1)) # Get the max value of the region

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            #If this pixel was the max value, copy the gradient to it.
            if imgRegion[i2, j2, f2] == amax[f2]:
              dL_dInput[i * 2 + i2, j * 2 + j2, f2] = dL_dOut[i, j, f2]

    return dL_dInput