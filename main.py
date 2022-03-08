import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# The mnist package handles the MNIST dataset for us
# --> https://github.com/datapythonista/mnist

# get our dataset / only use 1000 data
trainImages = mnist.train_images()[:1000]
trainLabels = mnist.train_labels()[:1000]

testImages = mnist.test_images()[:1000]
testLabels = mnist.test_labels()[:1000]

# Example of convolution
conv = Conv3x3(8) # get a convolution layer with 8 filters #28x28x1 -> 26x26x8
pool = MaxPool2() # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

#================================================ Forward ===================================================
def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss
  - image is a 2d numpy array
  - label is a digit, the correct one
  '''

  #We transform the image from [0,255] to [-0.5, 0.5] to make it easier
  # to work with. This is standart practice (normalization)
  out = conv.forward((image / 255) - 0.5) # <-- Convolution
  out = pool.forward(out) # <-- Pooling
  out = softmax.forward(out) # <-- Softmax

  #Calculate cross-entropy loss and accuracy. np.log() is the natural log
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0 # Verify if the cnn got it right

  return out, loss, acc

#=========================================== Train ===================================================
def train(img, label, learnRate = 0.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy
  - img is a 2d numpy array
  - label is a digit (the correct one)
  - learnRate is the learning rate (how fast is trains)
  '''
  # Forward
  out, loss, acc = forward(img, label)

  # Calculate initial gradient --> All the gradients are zeros except the correct one
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, learnRate)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, learnRate)

  return loss, acc


#=========================================== Main ====================================================
if __name__ == '__main__':
  print('MNIST CNN initialized!')

  # Train the CNN for 3 epoch
  for epoch in range(3):
    #Get the epoch status
    print('------------ Epoch %d ------------' % (epoch + 1))

    #Shuffle the training data
    permutation = np.random.permutation(len(trainImages)) #<-- does the shuffling
    trainImages = trainImages[permutation]
    trainLabels = trainLabels[permutation]

    # Begin training
    loss = 0
    numCorrect = 0
    
    for i, (img, label) in enumerate(zip(trainImages, trainLabels)):
      # Do a forward pass
      #out, l, acc = forward(im, label)
      #loss += l
      #numCorrect += acc
    
      # Print stats every 100 steps.
      if i > 0 and i % 100 == 99:
        print(
          '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
          (i + 1, loss / 100, numCorrect)
        )
        loss = 0
        numCorrect = 0
  
      l, acc = train(img, label)
      loss += l
      numCorrect += acc

  print('============= CNN trained!! ==============')

  # Test the CNN
  print('\n--- Testing the CNN ---')
  loss = 0
  numCorrect = 0

  for img, label in zip(testImages, testLabels):
    out, l, acc = forward(img, label)
    loss += l
    numCorrect += acc

  numTests = len(testImages)
  print('Test loss: ', loss / numTests)
  print('Test accuracy: ', numCorrect / numTests)