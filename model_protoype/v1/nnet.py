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
import cPickle as pickle

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

class nnet:

    def __init__(self):
        config = get_config()
        self.data = load_data(config)
        print "%s data loaded..." %config["dataset"]
        nhidden_layers = len(config["hidden_sizes"])
        nhidden = config["hidden_sizes"][0]
        print "num_hidden_layers      :",nhidden_layers
        print "hidden_units_per_layer :",nhidden
        X = T.fmatrix()
        Y = T.imatrix()
        num_input  = config["num_input"]
        num_output = 10

        self.W = self.init_weights(num_input,num_output,nhidden,nhidden_layers)

        # list of layers in the nnet including the input and output layer
        L_layers = []
        py_x = self.model(X, self.W, nhidden_layers, L_layers)
        y_x = T.argmax(py_x, axis=1)

        individual_cost = -1.0 * (T.log(py_x)[T.arange(Y.shape[0]), Y])
        cost = T.mean(individual_cost)
        updates = self.sgd(cost, self.W)
        squared_norm_var = self.compute_grad_norms(X,cost,L_layers)

        self.train = theano.function(inputs=[X, Y], outputs=[cost,squared_norm_var, individual_cost], updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)

    def get_weight(self,shape):
        return theano.shared(floatX(np.random.randn(*shape) * 0.01))

    def init_weights(self,num_i,num_o,num_h,num_hlayers):
        W = []
        W.append(self.get_weight((num_i,num_h)))
        for i in range(num_hlayers-1):
            W.append(self.get_weight((num_h,num_h)))
        W.append(self.get_weight((num_h,num_o)))
        return W

    def sgd(self,cost, params, lr=0.01):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates

    # input  -> layer[0],
    # hidden ->layer[1:n_hidden+1]
    # output -> layer[n_hidden+1]
    def model(self,X, W, num_hlayers, layer_inputs):
        layer_inputs.append(X)
        for i in range(num_hlayers):
            #layer_inputs.append(T.nnet.sigmoid(T.dot(layer_inputs[i], W[i])))
            layer_inputs.append(T.maximum(0.0, T.dot(layer_inputs[i], W[i])))
        layer_inputs.append(T.nnet.softmax(T.dot(layer_inputs[num_hlayers],W[num_hlayers])))
        return layer_inputs[num_hlayers+1]

    def compute_grad_norms(self, X, cost, layerLst):
        gradient_norm_f = 0.0

        for index in range(1, len(layerLst)):
            output_layer = layerLst[index]
            input_layer = layerLst[index - 1]
            gradient_norm_f += (input_layer**2).sum(axis = 1) * (T.grad(cost, output_layer)**2).sum(axis = 1)

        #gradient_norm_f = T.sqrt(gradient_norm_f)
        return gradient_norm_f

    def update_params(self, params):
        params = pickle.loads(params)
        for i in len(params):
            self.W[i].set_value(params[i])

    def compute_grads_and_weights(self, A_indices, L_measurements, segment):
        X = self.data[segment][0][A_indices]
        Y = self.data[segment][1][A_indices]

        Y = np.reshape(Y, (Y.shape[0], 1))
        print "X shape", X.shape
        print "Y shape", Y.shape

        cost,sq_grad_norm,individual_cost = self.train(X, Y)

        res = {}
        for key in L_measurements:
            if key == "importance_weight":
                res[key] = sq_grad_norm
            if key == "gradient_square_norm":
                res[key] = sq_grad_norm
            if key == "loss":
                res[key] = individual_cost
        return res

def main_loop( ):
    myNet = nnet()
    A_indices = np.array([1,2,3,4])
    segment = "train"
    L_measurements = ["gradient_square_norm","loss"]
    res = myNet.compute_grads_and_weights(A_indices,L_measurements,segment)
    print res

if __name__ == "__main__":
    main_loop()