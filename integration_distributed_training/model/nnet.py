__author__ = 'chinna'

import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function

from theano import tensor as T
from load import mnist
from load import mnist_with_noise
from scipy.misc import imsave
import scipy as sp

from scipy import signal
from load_data import load_data, normalizeMatrix
import cPickle as pickle

# Tool to get the special formulas for gradient norms and variance.
from fast_individual_gradient_norms.expression_builder import SumGradSquareNormAndVariance

class NeuralNetwork:


    def __init__(self, model_config):
        nhidden_layers = len(model_config["hidden_sizes"])
        nhidden = model_config["hidden_sizes"][0]
        print "num_hidden_layers      :",nhidden_layers
        print "hidden_units_per_layer :",nhidden
        X = T.fmatrix()
        Y = T.ivector()
        scaling_factors = T.fvector()
        num_input = model_config["num_input"]
        num_output = 10

        L_W, L_b = NeuralNetwork.build_parameters(num_input, num_output, model_config["hidden_sizes"], scale=0.01)
        L_W_momentum, L_b_momentum, = NeuralNetwork.build_parameters(num_input, num_output, model_config["hidden_sizes"], scale=0.0, name_suffix="_momentum")
        self.parameters = L_W + L_b
        self.momentum   = L_W_momentum + L_b_momentum

        print self.parameters

        (L_layer_inputs, L_layer_desc) = NeuralNetwork.build_layers(X, L_W, L_b)
        py_x = L_layer_inputs[-1]

        y_x = T.argmax(py_x, axis=1)

        individual_cost = -1.0 * (T.log(py_x)[T.arange(Y.shape[0]), Y])
        cost = T.mean(individual_cost)

        scaled_individual_cost = scaling_factors * individual_cost
        scaled_cost = T.mean(scaled_individual_cost)

        updates = NeuralNetwork.sgd(scaled_cost, self.parameters, self.momentum, model_config["learning_rate"], model_config["momentum_rate"])


        sgsnav = SumGradSquareNormAndVariance()
        for layer_desc in L_layer_desc:
            sgsnav.add_layer_for_gradient_square_norm(input=layer_desc['input'], weight=layer_desc['weight'],
                                                      bias=layer_desc['bias'], output=layer_desc['output'], cost=cost)

            sgsnav.add_layer_for_gradient_variance( input=layer_desc['input'], weight=layer_desc['weight'],
                                                    bias=layer_desc['bias'], output=layer_desc['output'], cost=cost)

        individual_gradient_squared_norm = sgsnav.accumulated_sum_gradient_square_norm
        individual_gradient_variance = sgsnav.get_sum_gradient_variance


        accuracy = T.mean(T.eq(T.argmax(py_x, axis = 1), Y))

        self.train = theano.function(inputs=[X, Y, scaling_factors],
                                     outputs=[cost, individual_gradient_squared_norm, individual_cost, accuracy],
                                     updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)

        self.get_attributes = theano.function(  inputs=[X, Y],
                                                outputs=[cost, individual_gradient_squared_norm, individual_cost, accuracy],
                                                allow_input_downcast=True)

        print "Model compilation complete"
        self.data = load_data(model_config)
        self.mean = self.data["mean"]
        self.std = self.data["std"]
        print "%s data loaded..." % model_config["dataset"]


    @staticmethod
    def build_layers(X, L_W, L_b):
        L_layer_inputs = [X]
        L_layer_desc = []
        for i in range(0, len(L_W)):

            print "accessing layer", i

            # the next inputs are always the last ones in the list
            inputs = L_layer_inputs[-1]
            weights = L_W[i]
            biases = L_b[i]
            activations = biases + T.dot(inputs, weights)

            if i < len(L_W) - 1:
                # all other layers except the last
                layer_outputs = T.maximum(0.0, activations)
            else:
                # last layer
                layer_outputs = T.nnet.softmax(activations)

            L_layer_inputs.append(layer_outputs)

            # This is formatted to be fed to the "fast_individual_gradient_norms".
            # The naming changes a bit because of that, because we're realling referring
            # to the linear component itself and not the stuff that happens after.
            layer_desc = {'weight' : weights, 'bias' : biases, 'output':activations, 'input':inputs}
            L_layer_desc.append(layer_desc)

        return L_layer_inputs, L_layer_desc


    @staticmethod
    def floatX(X):
        return np.asarray(X, dtype=theano.config.floatX)

    @staticmethod
    def init_weights(shape, name, scale = 0.01):
        return theano.shared(NeuralNetwork.floatX(np.random.randn(*shape) * scale), name=name)

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
    def build_parameters(num_input, num_output, hidden_sizes, scale, name_suffix=""):

        # The name suffix is to be used in the case of the momentum.

        L_sizes = [num_input] + hidden_sizes + [num_output]
        L_W = []
        L_b = []

        for (layer_number, (dim_in, dim_out)) in enumerate(zip(L_sizes, L_sizes[1:])):
            W = NeuralNetwork.init_weights((dim_in, dim_out), scale=scale, name=("%0.3d_weight%s"%(layer_number, name_suffix)))
            b = NeuralNetwork.init_weights((dim_out,), scale=0.0, name=("%0.3d_bias%s"%(layer_number, name_suffix)))
            L_W.append(W)
            L_b.append(b)

        return L_W, L_b


    def compute_grads_and_weights(self, data, L_measurements):
        X, Y = data

        cost, sq_grad_norm, individual_cost, accuracy = self.get_attributes(X, Y)

        # DEBUG
        number_of_invalid_values = np.logical_not(np.isfinite(sq_grad_norm)).sum()
        if 0 < number_of_invalid_values:
            print "In compute_grads_and_weights, you have %d invalid values for sq_grad_norm." % number_of_invalid_values

        number_of_invalid_values = np.logical_not(np.isfinite(individual_cost)).sum()
        if 0 < number_of_invalid_values:
            print "In compute_grads_and_weights, you have %d invalid values for individual_cost." % number_of_invalid_values

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
