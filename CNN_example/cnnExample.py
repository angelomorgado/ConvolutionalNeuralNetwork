import mnist
import numpy as np
from conv import Conv3x3 as convLayer
from maxpool import MaxPool2 as maxpool
from softmax import Softmax as softmax

# The mnist package handles the MNIST dataset for us
# --> https://github.com/datapythonista/mnist

# get our dataset / only use 1000 data
trainImages = mnist.train_images()[:1000]
trainLabels = mnist.train_labels()[:1000]

# Example of convolution
conv = convLayer(8) # get a convolution layer with 8 filters
outputConv = conv.forward(trainImages[0]) # get the output of the first image

# Example of pooling
pool = maxpool()
outputPool = pool.forward(outputConv)

print('Conv layer output:\n')
print(outputConv.shape) # (26, 26, 8)
print('Pool layer output:\n')
print(outputPool.shape) # (13, 13, 8)
