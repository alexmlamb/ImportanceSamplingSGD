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

def model(X, w_h, Layer_inputs):
    for i in range(0, len(w_h) - 1):
        Layer_inputs.append(T.nnet.sigmoid(T.dot(Layer_inputs[i], w_h[i])))

    Layer_inputs.append(T.nnet.softmax(T.dot(Layer_inputs[-1],w_h[-1])))

    return Layer_inputs[-1]

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

def init_parameters(num_input, num_output, hidden_sizes):
    w_h = [init_weights((num_input, hidden_sizes[0]))]
    
    for i in range(1, len(hidden_sizes) - 1):
        w_h += [init_weights((hidden_sizes[i], hidden_sizes[i + 1]))]

    w_h += [init_weights((hidden_sizes[-1], num_output))]

    return w_h

class nnet:

    def __init__(self):

        config = get_config()

        nhidden_layers = len(config["hidden_sizes"])
        nhidden = config["hidden_sizes"][0]

        X = T.fmatrix()
        Y = T.imatrix()
        num_input = 784
        num_output = 10

        w_h = init_parameters(num_input, num_output, config["hidden_sizes"])

        print w_h

        Layers = [X]


        py_x = model(X, w_h,Layers)
        y_x = T.argmax(py_x, axis=1)

        individual_cost = -1.0 * (T.log(py_x)[T.arange(Y.shape[0]), Y])
        cost = T.mean(individual_cost)
        params = w_h
        updates = sgd(cost, params)
        squared_norm_var = compute_grad_norms(X,cost,Layers)

        self.train = theano.function(inputs=[X, Y], outputs=[cost,squared_norm_var, individual_cost], updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)


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



    def compute_grads_and_weights(self, data, L_measurements):
        X, Y = data

        Y = np.reshape(Y, (Y.shape[0], 1))
        print "X shape", X.shape
        print "Y shape", Y.shape

        cost,sq_grad_norm,individual_cost = self.train(X, Y)
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

    myNet = nnet()

    ntrain = 50000
    ntest  = 10000
    nhidden_units = 625
    nhidden_layers = 0
    A_indices = np.array([1,2,3,4])
    segment = "train"
    L_measurements = ["gradient_square_norm"]
    data = load_data(get_config())[segment]
    myNet.compute_grads_and_weights(data,L_measurements)

if __name__ == "__main__":
    main_loop()





