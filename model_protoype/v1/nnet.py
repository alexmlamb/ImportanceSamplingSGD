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
from scipy.misc import imsave
import scipy as sp
import numpy as np
from PIL import Image
from scipy import signal
from load_data import load_data
from config import get_config

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

def model1(X, w_h, num_layers, Layer_inputs):
    print num_layers
    for i in range(0,num_layers-2):
        print num_layers
        Layer_inputs.append(T.nnet.sigmoid(T.dot(Layer_inputs[i], w_h[i])))
    return T.nnet.softmax(T.dot(Layer_inputs[num_layers-2],w_h[num_layers-1]))

def model(X, w_h, num_layers, Layer_inputs):
    Layer_inputs.append(T.nnet.sigmoid(T.dot(Layer_inputs[0], w_h[0])))
    Layer_inputs.append(T.nnet.softmax(T.dot(Layer_inputs[1],w_h[num_layers-1])))
    return Layer_inputs[2]

# output is a list of squared norm of gradients per example
# input is input matrix and
def compute_grad_norms(X, cost, layerLst):
  gradient_norm_f = 0.0

  for index in range(1, len(layerLst)):
      output_layer = layerLst[index]
      input_layer = layerLst[index - 1]
      gradient_norm_f += (input_layer**2).sum(axis = 1) * (T.grad(cost, output_layer)**2).sum(axis = 1)

  #gradient_norm_f = T.sqrt(gradient_norm_f)
  return gradient_norm_f

def get_data(A_indices,rval):

    res =[]
    for i in range(len(rval)):
        res.append( (rval[i])[A_indices] )
    return res

def compute_grads_and_weights(data, segment, L_measurements, nhidden=625,nhidden_layers=1 ):
    trX, trY, vX, vY, teX, teY = data

    X = T.fmatrix()
    Y = T.fmatrix()
    nlayers = nhidden_layers + 2
    #w_h = [init_weights((784, nhidden))] + [init_weights((nhidden, nhidden))]*(nhidden_layers) + [init_weights((nhidden, 10))]
    w_h = [init_weights((trX.shape[1], nhidden)), init_weights((nhidden, trY.shape[1]))]
    Layers = [X]
    py_x = model(X, w_h,nlayers,Layers)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = w_h
    updates = sgd(cost, params)
    squared_norm_var = compute_grad_norms(X,cost,Layers)

    train = theano.function(inputs=[X, Y], outputs=[cost,squared_norm_var], updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)

    cost,sq_grad_norm,individual_cost= train(trX, trY)
    #y,p_y =  predict(teX)
    #print np.mean(np.argmax(teY, axis=1) == y)
    #print trX.shape, trY.shape
    res = {}
    for key in L_measurements:
        if key == "importance_weight":
            res[key] = sq_grad_norm
        if key == "gradient_square_norm":
            res[key] = sq_grad_norm
        if key == "loss":
            res[key] = cost
    return res

def main_loop( ):
    ntrain = 50000
    ntest  = 10000
    nhidden_units = 625
    nhidden_layers = 0
    A_indices = np.array([1,2,3,4])
    segment = "train"
    L_measurements = ["gradient_square_norm"]
    rval = load_data(get_config())
    data = get_data(A_indices,rval)
    compute_grads_and_weights(data,segment,L_measurements,
                                    nhidden=nhidden_units,
                                    nhidden_layers=nhidden_layers)

if __name__ == "__main__":
    main_loop()





