
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

import hashlib


class ModelAPI():

    def __init__(self, model_config):
        self.model_config = model_config
        self.nnet = nnet.NeuralNetwork(model_config)

    def get_serialized_parameters(self):

        parameter_numpy_object_dict = {}

        for param in self.nnet.parameters:
            value = param.get_value().copy().astype(np.float32)
            parameter_numpy_object_dict[param.name] = value
            number_of_invalid_values = np.logical_not(np.isfinite(value)).sum()
            if 0 < number_of_invalid_values:
                print "In get_serialized_parameters, you have %d invalid values for parameter %s." % (number_of_invalid_values, param.name)
                print value
                print "Starting debugger."
                import pdb; pdb.set_trace()

        parameters_str = cPickle.dumps(parameter_numpy_object_dict, cPickle.HIGHEST_PROTOCOL)

        #print "DEBUG : Pickling values for parameters : "
        #print sorted(parameter_numpy_object_dict.keys())
        #print "DEBUG : Called get_serialized_parameters without a problem. Hashed sha256 : %s." % hashlib.sha256(parameters_str).hexdigest()

        return parameters_str

    def set_serialized_parameters(self, serialized_parameters):

        # You store a copy of the parameters that I pass you here.
        # You transfer them to the parameters.

        parameter_numpy_object_dict = cPickle.loads(serialized_parameters)
        #print "DEBUG : Entering set_serialized_parameters. Got argument with hash sha256 : %s." % hashlib.sha256(serialized_parameters).hexdigest()
        #print "DEBUG : Reading pickled values for parameters : "
        #print sorted(parameter_numpy_object_dict.keys())

        for param in self.nnet.parameters:

            saved_value = parameter_numpy_object_dict[param.name]

            assert saved_value.dtype == np.float32, "Failed to get a numpy array of np.float32 for parameter %s. The dtype is %s." % (param.name, saved_value.dtype)
            number_of_invalid_values = np.logical_not(np.isfinite(saved_value)).sum()
            if 0 < number_of_invalid_values:
                print "In set_serialized_parameters, you have %d invalid values for parameter %s." % (number_of_invalid_values, param.name)
                print saved_value
                print "Starting debugger."
                import pdb; pdb.set_trace()

            param.set_value(saved_value)

    def worker_process_minibatch(self, A_indices, segment, L_measurements):
        # This is one more step towards having simply that self.nnet implements the ModelAPI interface.
        return self.nnet.worker_process_minibatch(A_indices, segment, L_measurements)


    def master_process_minibatch(self, A_indices, A_scaling_factors, segment):
        # This is one more step towards having simply that self.nnet implements the ModelAPI interface.
        return self.nnet.master_process_minibatch(A_indices, A_scaling_factors, segment)


if __name__ == "__main__":

    #Run "unit tests".

    from unit_tests import all_tests

    all_tests()
