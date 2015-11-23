__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function

import theano
from theano import tensor as T
import numpy as np
from load import mnist
from load import mnist_with_noise
from PIL import Image
#from foxhound.utils.vis import grayscale_grid_vis, unit_scale
from scipy.misc import imsave
import scipy as sp
import numpy as np
from PIL import Image
from scipy import signal



def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h):
    h0 = T.nnet.sigmoid(T.dot(X, w_h[0]))
    pyx = T.nnet.softmax(T.dot(h0,w_h[1]))
    return pyx

def first_layer_out

def compute_grads_and_weights_mnist(A_indices, segment, L_measurements,ntrain=50000,ntest=10000,mb_size=128,nhidden=625 ):
    trX, teX, trY, teY = mnist(ntrain=ntrain,ntest=ntest,onehot=True)
    seq = mnist_with_noise([trX,trY],0)
    X = T.fmatrix()
    Y = T.fmatrix()
    w_h = [init_weights((784, nhidden)), init_weights((nhidden, 10))]
    py_x = model(X, w_h)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = w_h
    updates = sgd(cost, params)
    grads = T.grad(cost=cost,wrt=params)
    grad_for_norm = T.grad(cost=cost,wrt=params)

    train = theano.function(inputs=[X, Y], outputs=[cost,grads[0],grads[1]], updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)
    get_grad = theano.function(inputs=[X,Y],outputs=[cost,grad_for_norm[0],grad_for_norm[1]], allow_input_downcast=True)

    for i in range(1):
        for start, end in zip(range(0, len(trX), mb_size), range(mb_size, len(trX), mb_size)):
            cost,grads0,grads1 = train(trX[start:end], trY[start:end])
        y,p_y =  predict(teX)
        print np.mean(np.argmax(teY, axis=1) == y)

def main_loop( ):
    ntrain = 5000
    ntest  = 1000
    nhidden_units = 625
    A_indices = np.array([1,2,3,4])
    segment = "train"
    L_measurements = ["gradient_square_norm"]
    compute_grads_and_weights_mnist(A_indices,segment,L_measurements,
                                    ntrain=ntrain,ntest=ntest,nhidden=nhidden_units)

if __name__ == "__main__":
    main_loop()





