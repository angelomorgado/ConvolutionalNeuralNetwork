import numpy as np

#A standart fully-connected layer with softmax activation
class Softmax:
  def __init__(self, inputLen, nodes):
    #We divide by inputLen to reduce the variance of our inicial values (normalization)
    self.weights = np.random.randn(inputLen, nodes) / inputLen
    self.biases = np.zeros(nodes)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input. 
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions
    '''
    input = input.flatten() # Returns the array in 1 dimension

    inputLen, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases # xn * wn + bn <--It's the value we'll use

    exp = np.exp(totals) # Returns the exponential

    return exp / np.sum(exp, axis = 0) # <-- softmax formula