# Feed forward <-- Beggining to end
out = conv.forward((image / 255) - 0.5)
out = pool.forward(out)
out = softmax.forward(out)

# Calculate initial gradient
gradient = np.zeros(10)
gradient[label] = -1 / out[label] 

# Backprop <-- End to Beggining
gradient = softmax.backprop(gradient)
gradient = pool.backprop(gradient)
gradient = conv.backprop(gradient)

'''
Basically:

1ºstep:
  -> Do the feedforward from the conv layer to the softmax layer

2ºstep:
  -> Calculate the initial gradient so we can start improving it

3ºstep:
  -> Do the backward propagation from the softmax layer to the conv layer so we can improve our
    weights and therefor our prediction

4ºstep:
  -> Repeat the process n times according to the epochs
'''