import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# The mnist package handles the MNIST dataset for us
# --> https://github.com/datapythonista/mnist

# get our dataset / only use 1000 data
testImages = mnist.train_images()[:1000]
testLabels = mnist.train_labels()[:1000]

# Example of convolution
conv = Conv3x3(8) # get a convolution layer with 8 filters #28x28x1 -> 26x26x8
pool = MaxPool2() # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss
  - image is a 2d numpy array
  - label is a digit, the correct one
  '''

  #We transform the image from [0,255] to [-0.5, 0.5] to make it easier
  # to work with. This is standart practice (normalization)
  out = conv.forward((image / 255) - 0.5) # <-- Convolation
  out = pool.forward(out) # <-- Pooling
  out = softmax.forward(out) # <-- Softmax

  #Calculate cross-entropy loss and accuracy. np.log() is the natural log
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0 # Verify if the cnn got it right

  return out, loss, acc


print('MNIST CNN initialized!')

loss = 0
numCorrect = 0

for i, (im, label) in enumerate(zip(testImages, testLabels)):
  # Do a forward pass
  out, l, acc = forward(im, label)
  loss += l
  numCorrect += acc

  # Print stats every 100 steps.
  if i % 100 == 99:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, numCorrect)
    )
    loss = 0
    numCorrect = 0