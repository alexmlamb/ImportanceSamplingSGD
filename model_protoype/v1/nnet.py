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

def model(X, w_h, num_layers, Layer_inputs):
    print num_layers
    for i in range(0,num_layers-1):
        print num_layers
        Layer_inputs.append(T.nnet.sigmoid(T.dot(Layer_inputs[i], w_h[i])))
    return T.nnet.softmax(T.dot(Layer_inputs[num_layers-1],w_h[num_layers-1]))

# output is a list of squared norm of gradients per example
# input is input matrix and
def compute_grad_norms(X, cost, layerLst):
  gradient_norm_f = 0.0

  for index in range(1, len(layerLst)):
      output_layer = layerLst[index]
      input_layer = layerLst[index - 1]
      gradient_norm_f += (input_layer**2).sum(axis = 1) * (T.grad(cost, output_layer)**2).sum(axis = 1)

  gradient_norm_f = T.sqrt(gradient_norm_f)
  return gradient_norm_f


def compute_grads_and_weights_mnist(A_indices, segment, L_measurements,ntrain=50000,ntest=10000,mb_size=128,nhidden=625,nhidden_layers=1 ):
    trX, teX, trY, teY = mnist(ntrain=ntrain,ntest=ntest,onehot=True)
    seq = mnist_with_noise([trX,trY],0)
    X = T.fmatrix()
    Y = T.fmatrix()
    nlayers = nhidden_layers + 1
    w_h = [init_weights((784, nhidden))]*nhidden_layers + [init_weights((nhidden, 10))]

    Layers = [X]
    py_x = model(X, w_h,nlayers,Layers)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = w_h
    updates = sgd(cost, params)
    squared_norm_var = compute_grad_norms(X,cost,Layers)

    #train = theano.function(inputs=[X, Y], outputs=[cost], updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)

    for i in range(1):
        #for start, end in zip(range(0, len(trX), mb_size), range(mb_size, len(trX), mb_size)):
        #    cost= train(trX[start:end], trY[start:end])
        y,p_y =  predict(teX)
        print p_y[0]#np.mean(np.argmax(teY, axis=1) == y)

def main_loop( ):
    ntrain = 5000
    ntest  = 1000
    nhidden_units = 625
    nhidden_layers = 2
    A_indices = np.array([1,2,3,4])
    segment = "train"
    L_measurements = ["gradient_square_norm"]
    compute_grads_and_weights_mnist(A_indices,segment,L_measurements,
                                    ntrain=ntrain,ntest=ntest,nhidden=nhidden_units,
                                    nhidden_layers=nhidden_layers)

if __name__ == "__main__":
    main_loop()





