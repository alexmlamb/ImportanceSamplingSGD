
__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function

from load_data import load_data, normalizeMatrix
import nnet

import cPickle

import numpy as np

import time

class ModelAPI():

    def __init__(self, model_config):
        self.model_config = model_config
        self.nnet = nnet.NeuralNetwork(model_config)

    def get_serialized_parameters(self):

        parameter_numpy_object_dict = {}

        for param in self.nnet.parameters:
            parameter_numpy_object_dict[param.name] = param.get_value(borrow=True)


        return cPickle.dumps(parameter_numpy_object_dict, cPickle.HIGHEST_PROTOCOL)

    def set_serialized_parameters(self, serialized_parameters):

        # You store a copy of the parameters that I pass you here.
        # You transfer them to the parameters.

        parameter_numpy_object_dict = cPickle.loads(serialized_parameters)

        

        for param in self.nnet.parameters:
            saved_value = parameter_numpy_object_dict[param.name]
            param.set_value(saved_value)

    def worker_process_minibatch(self, A_indices, segment, L_measurements):
        assert segment in ["train", "valid", "test"]

        #print "Call to worker_process_minibatch."

        # This assumes that the worker knows how to get the data,
        # which puts the burden on the actual implementations.

        # It also assumes that the worker is able to get the things
        # that we are asking for. It can also put dummy values in a lot
        # of things, but it absolutely has to put something in "importance_weight"
        # because that it going to be used.

        for key in L_measurements:
            assert key in ["importance_weight", "gradient_square_norm", "loss", "accuracy"]

        curr_data = (normalizeMatrix(self.nnet.data[segment][0][A_indices], self.nnet.mean, self.nnet.std), self.nnet.data[segment][1][A_indices])

        res = self.nnet.compute_grads_and_weights(curr_data,L_measurements)

        # Returns a full array for every data point in the minibatch.
        return res


    def master_process_minibatch(self, A_indices, A_scaling_factors, segment):
        assert A_indices.shape == A_scaling_factors.shape, "Failed to assertion that %s == %s." % (A_indices.shape, A_scaling_factors.shape)
        assert segment in ["train"]

        #print "Call to master_process_minibatch."

        X = normalizeMatrix(self.nnet.data[segment][0][A_indices], self.nnet.mean, self.nnet.std)
        Y = self.nnet.data[segment][1][A_indices]

        self.nnet.train(X, Y, A_scaling_factors)

        # Returns nothing. The master should have used this call to
        # update its internal parameters.
        return


if __name__ == "__main__":

    #Run "unit tests".

    from unit_tests import all_tests

    all_tests()
