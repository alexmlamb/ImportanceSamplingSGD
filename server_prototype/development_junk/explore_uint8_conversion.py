
import theano
import numpy as np

Xdata = np.random.randint(low=0, high=256, size=(256,)).astype(np.uint8)

print Xdata

X = theano.tensor.fvector()
Z = (X / 255.0).mean()

f = theano.function([X], Z)

f(Xdata)

