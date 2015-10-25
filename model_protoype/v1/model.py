__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
from load_data import load_data
from config import get_config
from nnet import compute_grads_and_weights,get_data

import numpy as np

import time
from nnet import main_loop

SIMULATED_WORKER_PROCESS_MINIBATCH_TIME = 0.2
SIMULATED_MASTER_PROCESS_MINIBATCH_TIME = 1.0

class ModelAPI():

    def __init__(self):
        self.serialized_parameters_shape = (100,)
        self.config = get_config()

        self.nnet = nnet()

    def get_serialized_parameters(self):
        return np.random.rand(*self.serialized_parameters_shape).astype(np.float32)

    def set_serialized_parameters(self, serialized_parameters):
        assert type(serialized_parameters) == np.ndarray
        assert serialized_parameters.dtype == np.float32

        # You store a copy of the parameters that I pass you here.
        # You transfer them to the parameters.

    def update_data(self):
        self.data = load_data(self.config)

    def worker_process_minibatch(self, A_indices, segment, L_measurements):
        assert segment in ["train", "valid", "test"]

        # This assumes that the worker knows how to get the data,
        # which puts the burden on the actual implementations.

        # It also assumes that the worker is able to get the things
        # that we are asking for. It can also put dummy values in a lot
        # of things, but it absolutely has to put something in "importance_weight"
        # because that it going to be used.

        for key in L_measurements:
            assert key in ["importance_weight", "gradient_square_norm", "loss"]

        # Sleep to simulate work time.
        #time.sleep(SIMULATED_WORKER_PROCESS_MINIBATCH_TIME)
        curr_data = (self.data[segment][0][A_indices], self.data[segment][1][A_indices])

        res = compute_grads_and_weights(curr_data,segment,L_measurements)

        # Returns a full array for every data point in the minibatch.
        return res


    def master_process_minibatch(self, A_indices, A_scaling_factors, segment):
        assert A_indices.shape == A_scaling_factors.shape, "Failed to assertion that %s == %s." % (A_indices.shape, A_scaling_factors.shape)
        assert segment in ["train"]

        X = self.data[segment][A_indices]
        Y = self.data[segment][A_indices]

        self.nnet.train(X, Y)

        # Sleep to simulate work time.
        #time.sleep(SIMULATED_MASTER_PROCESS_MINIBATCH_TIME)

        

        # Returns nothing. The master should have used this call to
        # update its internal parameters.
        return



