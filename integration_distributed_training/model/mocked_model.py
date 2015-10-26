
import numpy as np

import time

SIMULATED_WORKER_PROCESS_MINIBATCH_TIME = 0.2
SIMULATED_MASTER_PROCESS_MINIBATCH_TIME = 1.0

class ModelAPI():

    def __init__(self, model_config=None):
        self.serialized_parameters_shape = (100,)

    def get_serialized_parameters(self):
        return np.random.rand(*self.serialized_parameters_shape).astype(np.float32).tostring(order='C')

    def set_serialized_parameters(self, serialized_parameters):
        pass
        #assert type(serialized_parameters) == np.ndarray
        #assert serialized_parameters.dtype == np.float32

    def worker_process_minibatch(self, A_indices, segment, L_measurements):
        assert segment in ["train", "valid", "test"]

        # This assumes that the worker knows how to get the data,
        # which puts the burden on the actual implementations.

        # It also assumes that the worker is able to get the things
        # that we are asking for. It can also put dummy values in a lot
        # of things, but it absolutely has to put something in "importance_weight"
        # because that it going to be used.

        for key in L_measurements:
            assert key in ["importance_weight", "gradient_square_norm", "loss", "accuracy"]

        # Sleep to simulate work time.
        time.sleep(SIMULATED_WORKER_PROCESS_MINIBATCH_TIME)

        res = {}
        for key in L_measurements:
            res[key] = np.random.rand(*A_indices.shape).astype(np.float32)

        # Returns a full array for every data point in the minibatch.
        return res


    def master_process_minibatch(self, A_indices, A_scaling_factors, segment):
        assert A_indices.shape == A_scaling_factors.shape, "Failed to assertion that %s == %s." % (A_indices.shape, A_scaling_factors.shape)
        assert segment in ["train"]

        # Sleep to simulate work time.
        time.sleep(SIMULATED_MASTER_PROCESS_MINIBATCH_TIME)

        # Returns nothing. The master should have used this call to
        # update its internal parameters.
        return
