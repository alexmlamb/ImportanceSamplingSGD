__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
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
from load_data import load_data, normalizeMatrix
import cPickle as pickle

class NeuralNetwork: 

    @staticmethod
    def floatX(X):
        return np.asarray(X, dtype=theano.config.floatX)

    @staticmethod
    def init_weights(shape, scale = 0.01):
        return theano.shared(NeuralNetwork.floatX(np.random.randn(*shape)) * scale)

    @staticmethod
    def sgd(cost, params, momemtum, lr, mr):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g, v in zip(params, grads, momemtum):
            v_prev = v
            updates.append([v, mr * v - g * lr])
            v = mr*v - g*lr
            updates.append([p, p  - mr*v_prev + (1 + mr)*v ])

        return updates

    @staticmethod
    def model(X, w_h, b_h, Layer_inputs):
        for i in range(0, len(w_h) - 1):
            Layer_inputs.append(T.maximum(0.0, b_h[i] + T.dot(Layer_inputs[i], w_h[i])))

        Layer_inputs.append(T.nnet.softmax(b_h[-1] + T.dot(Layer_inputs[-1],w_h[-1])))

        return Layer_inputs[-1]

    # output is a list of squared norm of gradients per example
    # input is input matrix and
    @staticmethod
    def compute_grad_norms(X, cost, layerLst):
        gradient_norm_f = 0.0

        for index in range(1, len(layerLst)):
            output_layer = layerLst[index]
            input_layer = layerLst[index - 1]
            gradient_norm_f += (input_layer**2).sum(axis = 1) * (T.grad(cost, output_layer)**2).sum(axis = 1)

        #gradient_norm_f = T.sqrt(gradient_norm_f)
        return gradient_norm_f

    @staticmethod
    def init_parameters(num_input, num_output, hidden_sizes, scale):
        w_h = [NeuralNetwork.init_weights((num_input, hidden_sizes[0]), scale)]
        b_h = [NeuralNetwork.init_weights((hidden_sizes[0],), scale = 0.0)]

        for i in range(1, len(hidden_sizes) - 1):
            w_h += [NeuralNetwork.init_weights((hidden_sizes[i], hidden_sizes[i + 1]), scale)]
            b_h += [NeuralNetwork.init_weights((hidden_sizes[i + 1],), scale = 0.0)]

        w_h += [NeuralNetwork.init_weights((hidden_sizes[-1], num_output), scale)]
        b_h += [NeuralNetwork.init_weights((num_output,), scale = 0.0)]

        return w_h, b_h


    def __init__(self, model_config):
        self.data = load_data(model_config)
        self.mean = self.data["mean"]
        self.std = self.data["std"]
        print "%s data loaded..." % model_config["dataset"]
        nhidden_layers = len(model_config["hidden_sizes"])
        nhidden = model_config["hidden_sizes"][0]
        print "num_hidden_layers      :",nhidden_layers
        print "hidden_units_per_layer :",nhidden
        X = T.fmatrix()
        Y = T.ivector()
        scaling_factors = T.fvector()
        num_input = model_config["num_input"]
        num_output = 10

        w_h, b_h = NeuralNetwork.init_parameters(num_input, num_output, model_config["hidden_sizes"],scale=0.01)
        w_m, b_m, = NeuralNetwork.init_parameters(num_input, num_output, model_config["hidden_sizes"],scale=0.0)
        self.parameters = w_h + b_h
        self.momentum   = w_m + b_m

        Layers = [X]


        py_x = NeuralNetwork.model(X, w_h, b_h, Layers)

        y_x = T.argmax(py_x, axis=1)

        individual_cost = -1.0 * (T.log(py_x)[T.arange(Y.shape[0]), Y])
        cost = T.mean(individual_cost)

        scaled_individual_cost = scaling_factors * individual_cost
        scaled_cost = T.mean(scaled_individual_cost)

        updates = NeuralNetwork.sgd(scaled_cost, self.parameters, self.momentum, model_config["learning_rate"], model_config["momentum_rate"])
        squared_norm_var = NeuralNetwork.compute_grad_norms(X,cost,Layers)

        accuracy = T.mean(T.eq(T.argmax(py_x, axis = 1), Y))

        self.train = theano.function(inputs=[X, Y, scaling_factors],
                                     outputs=[cost, squared_norm_var, individual_cost, accuracy],
                                     updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)

        self.get_attributes = theano.function(  inputs=[X, Y],
                                                outputs=[cost, squared_norm_var, individual_cost, accuracy],
                                                allow_input_downcast=True)




    # Note from Guillaume : This is a teaching moment
    # about style in python.
    def update_params(self, params):
        params = pickle.loads(params)
        for i in len(params):
            self.W[i].set_value(params[i])

    def compute_grads_and_weights(self, data, L_measurements):
        X, Y = data

        cost,sq_grad_norm,individual_cost, accuracy = self.get_attributes(X, Y)

        res = {}
        for key in L_measurements:
            if key == "importance_weight":
                res[key] = sq_grad_norm
            if key == "gradient_square_norm":
                res[key] = sq_grad_norm
            if key == "loss":
                res[key] = individual_cost
            if key == "accuracy":
                res[key] = accuracy
        return res




