import numpy as np

# Added some lines from the conv layer project, those lines will have {NEW}

#A standart fully-connected layer with softmax activation
class Softmax:
  def __init__(self, inputLen, nodes):
    #We divide by inputLen to reduce the variance of our inicial values (normalization)
    self.weights = np.random.randn(inputLen, nodes) / inputLen
    self.biases = np.zeros(nodes)

    
  #====================================== Forward ===============================================================
  '''
  Performs a forward pass of the softmax layer using the given input. 
  Returns a 1d numpy array containing the respective probability values.
  - input can be any array with any dimensions
  '''
  def forward(self, input):
    self.lastInputShape = input.shape # Save the shape of the input in cache {NEW}
    
    input = input.flatten() # Returns the array in 1 dimension
    self.lastInput = input # Save the input in cache {NEW}

    inputLen, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases # xn*wn+bn <-- It's the value we'll use
    self.lastTotals = totals # Save the output before applying the activation function in cache {NEW}

    exp = np.exp(totals) # Returns the exponential

    return exp / np.sum(exp, axis = 0) # <-- softmax formula

  #====================================== Backprop ===============================================================
  '''
   backprop function of this layer. Returns the loss gradient for this layer's input
   note: dl_dOut stands for ∂L/∂out, and it represents the loss gradient for this layers output
   learnRate is a float that indicates the speed at which the softmax trains
  '''
  def backprop(self, dL_dOut, learnRate):
    # We know only 1 element of dL_dOut will be nonzero (the correct class)
    # The enumerate() function iterates the object and counts how many iterations have passed
    for i, gradient in enumerate(dL_dOut):
      
      # If it isn't the correct class
      if gradient == 0:
        continue

      # e^totals
      tExp = np.exp(self.lastTotals) # We use the saved totals, the argument for the softmax function

      # Sum of all e^totals
      S = np.sum(tExp)

      # Gradients of out[i] against totals
      dOut_dT = -tExp[i] * tExp / (S ** 2) # derivative for wrong classes || Applied to all elements of list
      dOut_dT[i] = tExp[i] * (S - tExp[i]) / (S ** 2) # derivatives for right class || Changes the correct

      '''
      "Weight, biases and input gradients"

      All formulas were made on paper
      '''
      # Gradients of totals aginst weights/biases/input --> ∂t/∂w || ∂t/∂b || ∂t/∂input
      dT_dW = self.lastInput # This is the input softmax receives flattened
      dT_dB = 1
      dT_dInputs = self.weights

      # Gradients of loss against totals
      dL_dT = gradient * dOut_dT

      # Gradients of loss against weights/biases/input || use np.dot (@) with arrays and * with digit (bias)
      dL_dW = dT_dW[np.newaxis].T @ dL_dT[np.newaxis] # a @ b--> np.dot(a,b) || np.newaxis increases the dimension by 1
      dL_dB = dT_dB * dL_dT
      dL_dInputs = dT_dInputs @ dL_dT

      #Update weights/biases
      self.weights -= learnRate * dL_dW
      self.biases -= learnRate * dL_dB

      #return the dL_dInputs
      return dL_dInputs.reshape(self.lastInputShape) # Return the input in the correct shape stored in cache
      
      
      
      
      